"""临时脚本：用冠军权重对 20 组真实测试样本做定性+定量评估并可视化。"""
import sys
sys.path[0:0] = ['src']
import csv
import math
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from configs.rgbdgaze_config import RGBDGaze_INDEX_DIR
from gazelab.datasets.rgbdgaze import RGBDGazeDataset, rgb_preprocess
from gazelab.models.dinov2_gaze import RGBDGazeDINOv2
from pathlib import Path

CKPT = Path("out/rgbdgaze/ablation/checkpoints/champion_final_test/best.pt")
N_SAMPLES = 20

device = "cuda" if torch.cuda.is_available() else "cpu"
model = RGBDGazeDINOv2(unfreeze_last=12)
model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=False))
model.to(device).eval()
print("模型加载成功:", CKPT)

index_csv = RGBDGaze_INDEX_DIR / "index.csv"
rows = list(csv.DictReader(open(index_csv, encoding="utf-8")))
test_set = [path_to_idx for path_to_idx in []]
path_to_idx = {r["rgb_path"]: i for i, r in enumerate(rows)}
test_set = [path_to_idx[p.strip()] for p in
            open(RGBDGaze_INDEX_DIR / "test.txt", encoding="utf-8") if p.strip()]
print(f"test 集共 {len(test_set)} 样本，取 {N_SAMPLES}")

ds = RGBDGazeDataset(index_csv, color_transform=rgb_preprocess)

rng = np.random.RandomState(0)
picks = rng.choice(len(test_set), N_SAMPLES, replace=False)
picks = [test_set[i] for i in picks]

fig, axes = plt.subplots(4, 5, figsize=(20, 16))
results = []
with torch.no_grad():
    for k, idx in enumerate(picks):
        row = rows[idx]
        inp, label = ds[idx]
        face = inp.rgb.unsqueeze(0).to(device).float()
        geom = inp.geom.unsqueeze(0).to(device).float()
        pred = model(face, geom=geom)[0].cpu().numpy()
        label = label.numpy()
        sw_cm = float(row["screen_w"]) * float(row["cm_px_x"])
        sh_cm = float(row["screen_h"]) * float(row["cm_px_y"])
        err_x = (pred[0] - label[0]) * sw_cm
        err_y = (pred[1] - label[1]) * sh_cm
        err_cm = math.hypot(err_x, err_y)
        results.append(err_cm)
        ax = axes[k // 5, k % 5]
        ax.set_xlim(0, 1); ax.set_ylim(1, 0)
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="k"))
        ax.plot(label[0], label[1], "go", ms=10, label="GT")
        ax.plot(pred[0], pred[1], "r*", ms=12, label="Pred")
        ax.plot([label[0], pred[0]], [label[1], pred[1]], "b--", lw=1)
        ax.set_title(f"#{idx} {row['subject']}/{row['activity'][:3]}\n"
                     f"err={err_cm:.2f}cm", fontsize=9)
        ax.set_aspect("equal")
        if k == 0:
            ax.legend(fontsize=8)

plt.suptitle(f"champion_final_test best.pt — {N_SAMPLES} test samples\n"
             f"mean err = {np.mean(results):.2f} cm", fontsize=14)
plt.tight_layout()
plt.savefig("eval_visual_result.png", dpi=90)
print(f"20 样本误差: {['%.2f' % r for r in results]}")
print(f"平均: {np.mean(results):.2f}cm  中位: {np.median(results):.2f}cm  "
      f"最大: {np.max(results):.2f}cm")
print("图已保存 -> eval_visual_result.png")
