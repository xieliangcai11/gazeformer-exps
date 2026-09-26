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
from torch.utils.data import DataLoader

from configs.config import DEVICE as DEFAULT_DEVICE
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
    p.add_argument("--save-dir", default=str(Path("out/rgbdgaze")))
    return p.parse_args()


def main():
    args = parse()
    device = args.device if args.device != "auto" else DEFAULT_DEVICE
    index_dir = Path(args.index_dir)

    # 模型
    model = RGBDGazeModel().to(device)
    # 损失：MSE（2D 坐标，已归一化）
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_dl = build_dataloader("train", index_dir, args.batch_size, True)
    test_dl = build_dataloader("test", index_dir, args.batch_size, False)

    os.makedirs(args.save_dir, exist_ok=True)
    print(f"[rgbdgaze] 训练集 {len(train_dl.dataset)} 样本 / 测试集 "
          f"{len(test_dl.dataset)} 样本")
    print(f"[rgbdgaze] 模型参数: {sum(p.numel() for p in model.parameters()):,}")

    best_test = float("inf")
    for epoch in range(args.epochs):
        model.train()
        run_loss = 0.0
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
            if i % 50 == 0:
                print(f"[{epoch}] step {i} loss {loss.item():.6f}")
        # 测试
        model.eval()
        test_err = 0.0
        cnt = 0
        with torch.no_grad():
            for inp, label in test_dl:
                face = inp.face.to(device).float()
                depth = inp.other_face.to(device).float()
                label = label.to(device)
                pred = model(face, depth)
                err = (pred - label).norm(dim=-1).sum().item()
                test_err += err
                cnt += label.size(0)
        mean_err = test_err / max(cnt, 1)
        print(f"[{epoch}] train_loss {run_loss/len(train_dl):.6f} "
              f"test_mean_L2 {mean_err:.4f}")
        if mean_err < best_test:
            best_test = mean_err
            torch.save(model.state_dict(), os.path.join(args.save_dir, "best.pt"))
            print(f"  saved best ({mean_err:.4f})")


if __name__ == "__main__":
    main()