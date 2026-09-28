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


class Subset(torch.utils.data.Dataset):
    """按 idces 对基础数据集取样（模块级，可被 DataLoader 多进程 pickle）。"""

    def __init__(self, base, idces):
        self.base = base
        self.idces = idces

    def __len__(self):
        return len(self.idces)

    def __getitem__(self, i):
        return self.base[self.idces[i]]

    @property
    def cm_px(self):
        """该子集每个样本的 (cm/px_x, cm/px_y)，与数据顺序一致。"""
        return self.base.cm_px[self.idces]

    @property
    def screen_size(self):
        """该子集每个样本的 (px宽, px高)，与数据顺序一致。"""
        return self.base.screen_size[self.idces]


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
    # 过滤出属于该 split 的子集（用模块级 Subset，可多进程 pickle）
    sub = Subset(ds, idx)
    return DataLoader(sub, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers)


def evaluate_split(model, dl, device):
    """在 DataLoader(不 shuffle) 上评估，返回可解释指标。

    label 是归一化 [0,1] 屏幕坐标（0=左/上，1=右/下）。
    1. cm 误差：归一化误差 * screen(px) * cm/px = 物理厘米（每设备真实尺寸）
    2. 角度误差：屏幕坐标 -> 视线方向，夹角（度）。
       几何：用户眼睛位于屏幕中心正前方 VIEWING_DISTANCE_CM(默认30) cm，
       屏幕坐标 (归一化) -> 物理厘米 (x_cm, y_cm)，
       视线方向向量 = (x_cm - 屏中心, y_cm - 屏中心, VIEWING_DISTANCE_CM) 归一化。
       预测方向与真值方向的夹角即角度误差（度）。
       返回 mean 与 95 分位。
    """
    import numpy as np
    from configs.rgbdgaze_config import VIEWING_DISTANCE_CM

    model.eval()
    dset = dl.dataset
    cm_arr = np.asarray(dset.cm_px, dtype=np.float32)      # [N,2] cm/px
    scr = np.asarray(dset.screen_size, dtype=np.float32)    # [N,2] px
    D = VIEWING_DISTANCE_CM

    L2n = 0.0; xA = 0.0; yA = 0.0; cnt = 0
    cma = [0.0, 0.0, 0.0]
    ang_acc = []
    start = 0
    with torch.no_grad():
        for inp, label in dl:
            face = inp.face.to(device).float()
            depth = inp.other_face.to(device).float()
            label = label.to(device)
            pred = model(face, depth)
            diff = (pred - label)
            B = label.size(0)
            cm_b = torch.from_numpy(cm_arr[start:start+B]).to(device)
            scr_b = torch.from_numpy(scr[start:start+B]).to(device)
            # cm 误差（物理距离）
            cm_x = diff[:, 0] * scr_b[:, 0] * cm_b[:, 0]
            cm_y = diff[:, 1] * scr_b[:, 1] * cm_b[:, 1]
            cm_dist = torch.sqrt(cm_x**2 + cm_y**2)
            L2n += diff.norm(dim=-1).sum().item()
            xA += diff[:, 0].abs().sum().item()
            yA += diff[:, 1].abs().sum().item()
            cma[0] += cm_dist.sum().item()
            cma[1] += cm_x.abs().sum().item()
            cma[2] += cm_y.abs().sum().item()

            # 角度误差：每个样本 绝对注视点 -> 视线方向 -> 夹角
            # 归一化坐标 -> 物理厘米（相对屏幕左上角）
            # 屏中心在 (scr_w_cm/2, scr_h_cm/2)，视线方向再看向屏幕点
            scr_w_cm = scr_b[:, 0] * cm_b[:, 0]   # [B] 每设备屏宽 cm
            scr_h_cm = scr_b[:, 1] * cm_b[:, 1]   # [B] 每设备屏高 cm
            px_cm = pred[:, 0] * scr_w_cm          # 预测点 x cm
            py_cm = pred[:, 1] * scr_h_cm
            tx_cm = label[:, 0] * scr_w_cm
            ty_cm = label[:, 1] * scr_h_cm
            # 相对屏中心
            px = px_cm - scr_w_cm / 2
            py = py_cm - scr_h_cm / 2
            tx = tx_cm - scr_w_cm / 2
            ty = ty_cm - scr_h_cm / 2
            # 视线方向（眼睛在 z=D 前方）
            p_vec = torch.stack([px, py, torch.full_like(px, D)], dim=-1)
            t_vec = torch.stack([tx, ty, torch.full_like(tx, D)], dim=-1)
            p_n = p_vec / p_vec.norm(dim=-1, keepdim=True)
            t_n = t_vec / t_vec.norm(dim=-1, keepdim=True)
            cosim = (p_n * t_n).sum(dim=-1).clamp(-1.0, 1.0)
            angle_deg = torch.acos(cosim) * 180.0 / 3.141592653589793
            ang_acc.append(angle_deg.cpu().numpy())

            cnt += B
            start += B

    ang_all = np.concatenate(ang_acc) if ang_acc else np.array([0.0])
    # 统计口径（对齐任务规格）
    n = len(ang_all)
    target_angle = 1.91  # 规格：1.91°@95% 样本满足
    sorted_a = np.sort(ang_all)
    top95_mean = float(sorted_a[:int(n * 0.95)].mean()) if n > 0 else 0.0
    return dict(
        l2_norm=L2n/max(cnt,1), l2_x=xA/max(cnt,1), l2_y=yA/max(cnt,1),
        em_dist=cma[0]/max(cnt,1),
        em_x=cma[1]/max(cnt,1), em_y=cma[2]/max(cnt,1),
        ang_mean=float(ang_all.mean()),
        ang_median=float(np.median(ang_all)),
        ang_top95_mean=top95_mean,
        ang_p95=float(np.percentile(ang_all, 95)),
        ang_ratio_le_191=float((ang_all <= target_angle).mean() * 100),
        count=cnt,
    )

def parse():
    p = argparse.ArgumentParser()
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
    val_dl = build_dataloader("val", index_dir, args.batch_size, False,
                              args.num_workers)
    test_dl = build_dataloader("test", index_dir, args.batch_size, False,
                               args.num_workers)

    save_dir.mkdir(parents=True, exist_ok=True)
    n_train, n_val, n_test = (len(train_dl.dataset), len(val_dl.dataset),
                              len(test_dl.dataset))
    header = (f"{'='*60}\n"
              f"  RGBDGaze 训练启动\n"
              f"  训练集: {n_train} 样本 / 验证集: {n_val} 样本 / 测试集: {n_test} 样本\n"
              f"  batch_size: {args.batch_size}  epochs: {args.epochs}\n"
              f"  学习率: {args.lr}  设备: {device}\n"
              f"  模型参数: {sum(p.numel() for p in model.parameters()):,}\n"
              f"  选模型依据: 每 epoch 用 val 集评估，best 存 checkpoints\n"
              f"  最终评估: 训练结束后用 test 集独立评估\n"
              f"{'='*60}")

    def emit(msg):
        print(msg)
        write_log(msg)

    emit(header)
    emit(f"[rgbdgaze] 写入: log -> {log_file} | checkpoint -> {save_dir}")

    best_val = float("inf")
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
                epoch_t0 = time.time()

        # 每 epoch 用验证集评估（据此选 best，避免测试集泄漏）
        t_val = time.time()
        vres = evaluate_split(model, val_dl, device)
        mean_train = run_loss / max(n_step, 1)
        mean_val = vres["l2_norm"]
        mean_val_cm = vres["em_dist"]
        saved = ""
        if mean_val_cm < best_val:
            best_val = mean_val_cm
            ckpt_name = "best.pt"
            torch.save(model.state_dict(), os.path.join(save_dir, ckpt_name))
            saved = f"  [保存 best → {ckpt_name} (val {mean_val_cm:.2f}cm)]"

        msg = (f"[epoch {epoch}]  train_loss {mean_train:.6f}  "
               f"val_L2 {mean_val:.4f}  val_em {mean_val_cm:.2f}cm  "
               f"val_angle {vres['ang_mean']:.2f}°(p95 {vres['ang_p95']:.2f}°)  "
               f"验证耗时 {time.time()-t_val:.1f}s{saved}")
        emit(msg)

    # ---- 训练结束：用测试集做最终独立评估（加载 best 权重） ----
    emit("=" * 60)
    emit("[final] 用验证集最优权重在测试集上做最终评估...")
    best_ckpt = os.path.join(save_dir, "best.pt")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device,
                                         weights_only=False))
    fres = evaluate_split(model, test_dl, device)
    mean_test = fres["l2_norm"]
    mean_test_cm = fres["em_dist"]
    emit(f"[final] test_L2 {mean_test:.4f}  test_em {mean_test_cm:.2f}cm  "
         f"(x:{fres['em_x']:.2f} y:{fres['em_y']:.2f})")
    emit(f"[final] test_angle  平均 {fres['ang_mean']:.2f}° | "
         f"中位 {fres['ang_median']:.2f}° | "
         f"95%分位 {fres['ang_p95']:.2f}° | "
         f"前95%平均 {fres['ang_top95_mean']:.2f}°")
    emit(f"[final] ≤1.91° 样本占比 {fres['ang_ratio_le_191']:.1f}%  "
         f"样本 {fres['count']}")
    emit(f"[rgbdgaze] 训练完成 {args.epochs} epochs，best val = {best_val:.2f}cm，"
         f"最终 test = {mean_test_cm:.2f}cm")
    emit("=" * 60)


if __name__ == "__main__":
    main()