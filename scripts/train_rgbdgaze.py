"""RGBDGaze 训练入口（独立脚本，不依赖 train.py / Gaze360 管线）。

运行前先构建索引：
    python -m tools.data.rgbdgaze_preprocess --split sample

用法：
    python scripts/train_rgbdgaze.py [--epochs N] [--batch-size B]
                                     [--lr L] [--index-dir DIR] [--device cuda]
                                     [--save-dir DIR]

数据：
    读 data/RGBDGaze/index/ 下的 index.csv + train/val/test 清单，
    经 RGBDGazeDataset 加载为 (face=RGB, other_face=DEPTH, label=2D)。

模型：
    RGBDGazeModel（CLIP 语义流 + ResNet depth 流 + 2D 回归 head）。

损失：
    MSE on 2D 屏幕坐标（设备归一化后 [0,1] 量级）。
"""

import argparse
import os
import sys
from pathlib import Path

# 路径引导（同其他脚本）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader

from datetime import datetime
from configs.gaze360_config import DEVICE as DEFAULT_DEVICE, experiment_dirs
from configs.rgbdgaze_config import RGBDGaze_INDEX_DIR
from gazelab.datasets.rgbdgaze import (RGBDGazeDataset, rgb_preprocess,
                                       depth_preprocess)
from gazelab.models.rgbdgaze import RGBDGazeModel


def build_dataloader(split: str, index_dir: Path, batch_size: int,
                     shuffle: bool, num_workers: int = 0):
    index_csv = index_dir / "index.csv"
    list_txt = index_dir / f"{split}.txt"
    # 根据清单行号从 index 取样（index 行号 = split.txt 行号）
    paths = [l.strip() for l in list_txt.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    # 简化：直接读 index.csv，构建行号 -> rgb 映射
    import csv
    with open(index_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rgb_to_idx = {r["rgb_path"]: i for i, r in enumerate(rows)}
    idx = [rgb_to_idx[p] for p in paths]

    ds = RGBDGazeDataset(index_csv,
                         color_transform=rgb_preprocess,
                         depth_transform=depth_preprocess)  # 全量
    # 过滤出属于该 split 的子集（通过取样包装）
    class Subset(torch.utils.data.Dataset):
        def __init__(self, base, idces):
            self.base = base; self.idces = idces
        def __len__(self):
            return len(self.idces)
        def __getitem__(self, i):
            return self.base[self.idces[i]]
    sub = Subset(ds, idx)
    return DataLoader(sub, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--index-dir", default=str(RGBDGaze_INDEX_DIR))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--device", default="auto")
    p.add_argument("--save-dir", default=None,
                   help="权重输出目录（默认 out/rgbdgaze/train/checkpoints）")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader 工作进程数（默认 4）")
    return p.parse_args()


def main():
    args = parse()
    device = args.device if args.device != "auto" else DEFAULT_DEVICE
    index_dir = Path(args.index_dir)

    # 统一输出目录布局（experiment_dirs 在 configs 定义，各脚本复用）
    run_dirs = experiment_dirs("rgbdgaze", "train")
    save_dir = Path(args.save_dir) if args.save_dir else run_dirs["checkpoint"]
    log_dir = run_dirs["log"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{log_time}_train_rgbdgaze_log.txt"

    def write_log(msg):
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    # 模型
    model = RGBDGazeModel().to(device)
    # 损失：MSE（2D 坐标，已归一化）
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_dl = build_dataloader("train", index_dir, args.batch_size, True,
                                args.num_workers)
    test_dl = build_dataloader("test", index_dir, args.batch_size, False,
                               args.num_workers)

    save_dir.mkdir(parents=True, exist_ok=True)
    n_train, n_test = len(train_dl.dataset), len(test_dl.dataset)
    header = (f"{'='*60}\n"
              f"  RGBDGaze 训练启动\n"
              f"  训练集: {n_train} 样本 / 测试集: {n_test} 样本\n"
              f"  batch_size: {args.batch_size}  epochs: {args.epochs}\n"
              f"  学习率: {args.lr}  设备: {device}\n"
              f"  模型参数: {sum(p.numel() for p in model.parameters()):,}\n"
              f"{'='*60}")

    def emit(msg):
        print(msg)
        write_log(msg)

    emit(header)
    emit(f"[rgbdgaze] 写入: log -> {log_file} | checkpoint -> {save_dir}")

    best_test = float("inf")
    for epoch in range(args.epochs):
        epoch_t0 = time.time()
        model.train()
        run_loss = 0.0
        n_step = 0
        for i, (inp, label) in enumerate(train_dl):
            face = inp.face.to(device).float()
            depth = inp.other_face.to(device).float()
            label = label.to(device)
            optimizer.zero_grad()
            pred = model(face, depth)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            n_step += 1
            if i % 50 == 0 or i == len(train_dl) - 1:
                lr_now = optimizer.param_groups[0]["lr"]
                msg = (f"[epoch {epoch}] step {i}/{len(train_dl)}  "
                       f"loss {loss.item():.6f}  lr {lr_now:.2e}  "
                       f"{time.time()-epoch_t0:.1f}s")
                emit(msg)
                epoch_t0 = time.time()  # 计时归零便于看 step 间隔

        # 测试
        model.eval()
        test_err = 0.0
        test_err_x = 0.0
        test_err_y = 0.0
        cnt = 0
        t_test = time.time()
        with torch.no_grad():
            for inp, label in test_dl:
                face = inp.face.to(device).float()
                depth = inp.other_face.to(device).float()
                label = label.to(device)
                pred = model(face, depth)
                diff = (pred - label)
                test_err += diff.norm(dim=-1).sum().item()
                test_err_x += diff[:, 0].abs().sum().item()
                test_err_y += diff[:, 1].abs().sum().item()
                cnt += label.size(0)

        mean_train = run_loss / max(n_step, 1)
        mean_err = test_err / max(cnt, 1)
        mean_err_x = test_err_x / max(cnt, 1)
        mean_err_y = test_err_y / max(cnt, 1)
        saved = ""
        if mean_err < best_test:
            best_test = mean_err
            ckpt_name = f"best_{args.epochs}ep.pt"
            torch.save(model.state_dict(), os.path.join(save_dir, ckpt_name))
            saved = f"  [保存 best → {ckpt_name} ({mean_err:.4f})]"

        msg = (f"[epoch {epoch}]  train_loss {mean_train:.6f}  "
               f"test_L2 {mean_err:.4f} (x:{mean_err_x:.4f} y:{mean_err_y:.4f})  "
               f"测试耗时 {time.time()-t_test:.1f}s{saved}")
        emit(msg)

    emit(f"[rgbdgaze] 训练完成 {args.epochs} epochs，best test_L2 = {best_test:.4f}")


if __name__ == "__main__":
    main()