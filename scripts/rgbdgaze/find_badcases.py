"""临时脚本：用现有最优模型筛选训练集中误差最大的样本，导出图片供人工查看。

从当前最优模型对训练集做一次推理，按误差排序，导出误差最大的前 N 个样本
（RGB 人脸裁剪 + 真实/预测注视点标注图），存到 temp/badcases/ 供人工分类：
  - 数据噪声（标注错误、完全不可用）  → 剔除
  - 真实困难（模糊但有信息）          → 过采样/加强增强
"""
import sys
sys.path[0:0] = ['src']
import csv
import math
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from configs.rgbdgaze_config import RGBDGaze_INDEX_DIR
from gazelab.datasets.rgbdgaze import (
    RGBDGazeDataset, rgb_preprocess, _resolve_data_path,
)
from gazelab.models.dinov2_gaze.rgbdgaze_dinov2 import RGBDGazeDINOv2

# ---- 配置 ----
CKPT = Path("out/rgbdgaze/ablation/checkpoints/champion_final_test/best.pt")
RESOLUTION = 224  # checkpoint 是 224 版（pos_embed 257 tokens）
TOP_N = 40          # 导出误差最大的前 N 个
OUT_DIR = Path("temp/train_badcases")

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- 加载模型 ----
model = RGBDGazeDINOv2(unfreeze_last=12, img_size=RESOLUTION)
model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False))
model.to(device).eval()
print(f"[load] {CKPT}")

index_csv = RGBDGaze_INDEX_DIR / "index.csv"
rows = list(csv.DictReader(open(index_csv, encoding="utf-8")))
path_to_idx = {r["rgb_path"]: i for i, r in enumerate(rows)}
train_idx = [path_to_idx[l.strip()] for l in
             (RGBDGaze_INDEX_DIR / "train.txt").read_text(encoding="utf-8").splitlines()
             if l.strip()]
print(f"[data] 训练集 {len(train_idx)} 样本")

ds = RGBDGazeDataset(index_csv, color_transform=rgb_preprocess,
                     image_size=RESOLUTION)

# ---- 全量推理 ----
errors = []
with torch.no_grad():
    for start in range(0, len(train_idx), 64):
        batch_idx = train_idx[start:start+64]
        faces, geoms = [], []
        for idx in batch_idx:
            inp, _ = ds[idx]
            faces.append(inp.rgb)
            geoms.append(inp.geom)
        f = torch.stack(faces).to(device).float()
        g = torch.stack(geoms).to(device).float()
        pred = model(f, geom=g)
        for i, idx in enumerate(batch_idx):
            r = rows[idx]
            sw_cm = float(r["screen_w"]) * float(r["cm_px_x"])
            sh_cm = float(r["screen_h"]) * float(r["cm_px_y"])
            gx = float(r["gaze_x"]) / float(r["screen_w"])
            gy = float(r["gaze_y"]) / float(r["screen_h"])
            e = math.hypot((pred[i,0].item()-gx)*sw_cm,
                           (pred[i,1].item()-gy)*sh_cm)
            errors.append((e, idx, pred[i].cpu().numpy(), gx, gy))
        if (start//64) % 200 == 0:
            print(f"  {start+len(batch_idx)}/{len(train_idx)}")

errors.sort(key=lambda x: -x[0])
print(f"\n[train] 全量推理完成。误差分布:")
for q in (50, 90, 95, 99):
    print(f"  p{q}: {np.percentile([e[0] for e in errors], q):.2f}cm")
print(f"  >4cm: {sum(1 for e in errors if e[0]>4)} 个")
print(f"  >6cm: {sum(1 for e in errors if e[0]>6)} 个")

# ---- 导出 top N 的可视化 ----
OUT_DIR.mkdir(parents=True, exist_ok=True)
summary_lines = []
for rank, (err, idx, pred, gx, gy) in enumerate(errors[:TOP_N]):
    row = rows[idx]
    # 读原图并复现裁剪
    rgb_path = _resolve_data_path(row["rgb_path"])
    import cv2
    img = cv2.imread(str(rgb_path))[:, :, ::-1]
    img = np.rot90(img, 1)
    bx, by, bw, bh = (float(row["bbox_x"]), float(row["bbox_y"]),
                      float(row["bbox_w"]), float(row["bbox_h"]))
    H, W = img.shape[:2]
    cx, cy = bx + bw/2, by + bh/2
    side = max(bw, bh) * 1.2
    half = side / 2
    x0, x1 = max(0,int(cx-half)), min(W,int(cx+half))
    y0, y1 = max(0,int(cy-half)), min(H,int(cy+half))
    crop = img[y0:y1, x0:x1]

    # 画预测点和GT点（归一化 -> 像素比例位置）
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.imshow(np.clip(crop, 0, 255).astype(np.uint8))
    h, w = crop.shape[:2]
    ax.plot(gx*w, gy*h, "go", ms=12, label="GT")
    ax.plot(pred[0]*w, pred[1]*h, "r*", ms=14, label="Pred")
    ax.plot([gx*w, pred[0]*w], [gy*h, pred[1]*h], "b--", lw=1)
    ax.set_title(f"rank{rank} idx{idx} err={err:.2f}cm\n"
                 f"{row['subject']}/{row['activity']}", fontsize=9)
    ax.axis("off")
    ax.legend(fontsize=7)
    plt.tight_layout()
    fname = OUT_DIR / f"rank{rank:03d}_idx{idx}_{row['subject']}_{row['activity'][:4]}_{err:.1f}cm.png"
    plt.savefig(fname, dpi=80)
    plt.close()
    summary_lines.append(f"rank{rank:03d} | idx={idx} | {row['subject']}/{row['activity']} | "
                         f"err={err:.2f}cm | pred=({pred[0]:.2f},{pred[1]:.2f}) gt=({gx:.2f},{gy:.2f})")

# 写摘要文件
with open(OUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
    f.write("训练集误差最大样本（供人工分类：数据噪声→剔除 / 真实困难→过采样）\n")
    f.write("="*70 + "\n")
    for line in summary_lines:
        f.write(line + "\n")
    f.write(f"\n全量统计: >4cm={sum(1 for e in errors if e[0]>4)}个 "
            f">6cm={sum(1 for e in errors if e[0]>6)}个\n")
print(f"\n[export] {TOP_N} 张图 -> {OUT_DIR}/")
print(f"[export] 摘要 -> {OUT_DIR}/summary.txt")