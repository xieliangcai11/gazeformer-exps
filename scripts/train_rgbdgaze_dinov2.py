"""RGBDGaze 新模型（DINOv2 + 逆深度 + BlockMoba）训练入口。

数据/目录/输出全部复用 train_rgbdgaze.py 的规范，仅换模型与损失。

用法：
    python scripts/train_rgbdgaze_dinov2.py [--epochs N] [--batch-size B]
                                           [--lr LR] [--device cuda]
                                           [--save-dir DIR] [--dino-ckpt PATH]

损失（训练 = 验收，同一把尺子）：
    L = MSE(2D坐标) + λ * 几何角度损失
      角度损失：把 2D 屏幕点 + VIEWING_DISTANCE_CM 转成视线方向，求其对与 really 的夹角。
      这样模型直接以"角度"这个验收指标为中心学习。
  选择最优模型：按验证集平均角度（对齐 1.91°@95% 规格）。
"""

import argparse
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime

from configs.gaze360_config import DEVICE as DEFAULT_DEVICE, experiment_dirs
from configs.rgbdgaze_config import RGBDGaze_INDEX_DIR, VIEWING_DISTANCE_CM
from gazelab.datasets.rgbdgaze import (RGBDGazeDataset, rgb_preprocess,
                                       inverse_depth_preprocess)
from gazelab.models.rgbdgaze_dinov2 import RGBDGazeDINOv2, DINOV2_CKPT


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
        return self.base.cm_px[self.idces]

    @property
    def screen_size(self):
        return self.base.screen_size[self.idces]


def build_dataloader(split, index_dir, batch_size, shuffle, num_workers=0):
    index_csv = index_dir / "index.csv"
    list_txt = index_dir / f"{split}.txt"
    paths = [l.strip() for l in list_txt.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    import csv
    with open(index_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rgb_to_idx = {r["rgb_path"]: i for i, r in enumerate(rows)}
    idx = [rgb_to_idx[p] for p in paths]

    ds = RGBDGazeDataset(index_csv,
                         color_transform=rgb_preprocess,
                         depth_transform=inverse_depth_preprocess)
    sub = Subset(ds, idx)
    return DataLoader(sub, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers)


def angular_loss(pred_xy, label_xy, D):
    """2D 屏幕归一化坐标 -> 视线方向 -> 夹角（度），可微、数值稳健。

    pred/label: [B,2] 归一化[0,1]屏幕坐标。
    用 atan2(|叉积|, 点积) 求角度：梯度有界（不像 acos 在 cos→±1 处无界，
    能避免训练早期由随机初始化导致的梯度爆炸/NaN）。方向用归一化坐标即可，
    角度对坐标平面的均匀缩放不变。
    """
    eps = 1e-8
    pred_v = pred_xy - 0.5
    true_v = label_xy - 0.5
    pn = pred_v / pred_v.norm(dim=-1, keepdim=True).clamp_min(eps)
    tn = true_v / true_v.norm(dim=-1, keepdim=True).clamp_min(eps)
    dot = (pn * tn).sum(-1).clamp(-1.0, 1.0)          # cos(夹角)
    cross = (pn[:, 0] * tn[:, 1] - pn[:, 1] * tn[:, 0]).abs()  # |sin(夹角)|
    angle = torch.atan2(cross, dot) * 180.0 / 3.141592653589793
    return angle


def evaluate_split(model, dl, device):
    """在 DataLoader(不 shuffle) 上评估，返回多口径指标（与原 train_rgbdgaze 一致）。

    返回值 dict：
      l2_norm/l2_x/l2_y : 归一化坐标 L2（及 x/y 分量）
      em_dist/em_x/em_y : 物理厘米误差（用每设备屏幕尺寸 cm_px 换算）
      ang_mean/ang_median/ang_p95/ang_top95_mean : 角度误差口径（度）
      ang_ratio_le_191  : 角度 ≤1.91° 的样本占比(%)（任务规格）
      count             : 样本数
    """
    from configs.rgbdgaze_config import VIEWING_DISTANCE_CM

    model.eval()
    dset = dl.dataset
    cm_arr = np.asarray(dset.cm_px, dtype=np.float32)      # [N,2] cm/px
    scr = np.asarray(dset.screen_size, dtype=np.float32)   # [N,2] px
    D = VIEWING_DISTANCE_CM

    L2n = 0.0; xA = 0.0; yA = 0.0; cnt = 0
    cma = [0.0, 0.0, 0.0]
    ang_acc = []
    start = 0
    with torch.no_grad():
        for inp, label in dl:
            face = inp.rgb.to(device).float()
            depth = inp.depth.to(device).float()
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

            # 角度误差：绝对注视点(归一化) -> 物理厘米 -> 视线方向 -> 夹角
            scr_w_cm = scr_b[:, 0] * cm_b[:, 0]            # 每设备屏宽 cm
            scr_h_cm = scr_b[:, 1] * cm_b[:, 1]            # 每设备屏高 cm
            px_cm = pred[:, 0] * scr_w_cm; py_cm = pred[:, 1] * scr_h_cm
            tx_cm = label[:, 0] * scr_w_cm; ty_cm = label[:, 1] * scr_h_cm
            px = px_cm - scr_w_cm / 2; py = py_cm - scr_h_cm / 2
            tx = tx_cm - scr_w_cm / 2; ty = ty_cm - scr_h_cm / 2
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
    n = len(ang_all)
    target_angle = 1.91  # 规格：1.91°@95% 样本满足
    sorted_a = np.sort(ang_all)
    top95_mean = float(sorted_a[:int(n * 0.95)].mean()) if n > 0 else 0.0
    return dict(
        l2_norm=L2n/max(cnt, 1), l2_x=xA/max(cnt, 1), l2_y=yA/max(cnt, 1),
        em_dist=cma[0]/max(cnt, 1),
        em_x=cma[1]/max(cnt, 1), em_y=cma[2]/max(cnt, 1),
        ang_mean=float(ang_all.mean()),
        ang_median=float(np.median(ang_all)),
        ang_top95_mean=top95_mean,
        ang_p95=float(np.percentile(ang_all, 95)),
        ang_ratio_le_191=float((ang_all <= target_angle).mean() * 100),
        count=cnt,
    )


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--index-dir", default=str(RGBDGaze_INDEX_DIR))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--device", default="auto")
    p.add_argument("--save-dir", default=None)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--dino-ckpt", default=DINOV2_CKPT)
    p.add_argument("--ang-weight", type=float, default=1.0,
                   help="角度损失在总损失中的权重")
    return p.parse_args()


def main():
    args = parse()
    device = args.device if args.device != "auto" else DEFAULT_DEVICE
    index_dir = Path(args.index_dir)

    run_dirs = experiment_dirs("rgbdgaze", "train")
    save_dir = Path(args.save_dir) if args.save_dir else run_dirs["checkpoint"]
    log_dir = run_dirs["log"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{log_time}_train_rgbdgaze_dinov2_log.txt"

    def write_log(msg):
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    model = RGBDGazeDINOv2(dino_ckpt=args.dino_ckpt).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_dl = build_dataloader("train", index_dir, args.batch_size, True,
                                args.num_workers)
    val_dl = build_dataloader("val", index_dir, args.batch_size, False,
                              args.num_workers)

    save_dir.mkdir(parents=True, exist_ok=True)
    n_train, n_val = len(train_dl.dataset), len(val_dl.dataset)
    header = (f"{'='*60}\n"
              f"  RGBDGaze DINOv2 训练启动\n"
              f"  训练集: {n_train} / 验证集: {n_val}\n"
              f"  batch_size: {args.batch_size} epochs:{args.epochs} lr:{args.lr}\n"
              f"  设备: {device}  DINOv2权重: {args.dino_ckpt}\n"
              f"  损失: MSE + {args.ang_weight}*AngularLoss\n"
              f"  模型参数: {sum(p.numel() for p in model.parameters()):,}\n"
              f"  选模型: 每 epoch 按验证集平均角度选 best\n"
              f"{'='*60}")
    print(header)
    write_log(header)

    best_val = float("inf")
    for epoch in range(args.epochs):
        epoch_t0 = time.time()
        model.train()
        run_loss = 0.0
        n_step = 0
        for i, (inp, label) in enumerate(train_dl):
            face = inp.rgb.to(device).float()
            depth = inp.depth.to(device).float()  # [B,1,H,W] 逆深度
            label = label.to(device)
            optimizer.zero_grad()
            pred = model(face, depth)  # [B,2]
            loss_coord = criterion(pred, label)
            loss_ang = angular_loss(pred, label, VIEWING_DISTANCE_CM).mean()
            loss = loss_coord + args.ang_weight * loss_ang
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            n_step += 1
            if i % 50 == 0 or i == len(train_dl) - 1:
                msg = (f"[epoch {epoch}] step {i}/{len(train_dl)}  "
                       f"loss {loss.item():.6f} (mse {loss_coord.item():.5f} / "
                       f"ang {loss_ang.item():.3f})  {time.time()-epoch_t0:.1f}s")
                print(msg); write_log(msg)
                epoch_t0 = time.time()

        # 验证：多口径评估（L2/厘米/角度/≤1.91° 占比）
        vres = evaluate_split(model, val_dl, device)
        mean_train = run_loss / max(n_step, 1)
        # 选最优模型：按平均角度（对齐 1.91°@95% 规格）
        mean_ang = vres["ang_mean"]
        saved = ""
        if mean_ang < best_val:
            best_val = mean_ang
            ckpt_name = "best_dinov2.pt"
            torch.save(model.state_dict(), os.path.join(save_dir, ckpt_name))
            saved = f"  [保存 best → {ckpt_name} (val {mean_ang:.2f}°)]"
        msg = (f"[epoch {epoch}]  train_loss {mean_train:.6f}  "
               f"val_L2 {vres['l2_norm']:.4f}  val_em {vres['em_dist']:.2f}cm  "
               f"val_ang 平均 {vres['ang_mean']:.2f}° | 中位 {vres['ang_median']:.2f}° | "
               f"95%分位 {vres['ang_p95']:.2f}° | 前95%平均 {vres['ang_top95_mean']:.2f}° | "
               f"≤1.91° {vres['ang_ratio_le_191']:.1f}%{saved}")
        print(msg); write_log(msg)


if __name__ == "__main__":
    main()