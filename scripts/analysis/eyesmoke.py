"""冒烟检查：eye_scale 收紧眼框后，眼部裁剪是否真的对准眼睛、虹膜占比是否提高。

输出对比图：上排 = 历史行为(eye_scale=0)，下排 = eye_scale=1.5；
每排 4 个样本 × (左眼, 右眼)。用同一帧，便于目视判断。
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/data1/xieliangcai/code/gazeformer-exps")
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from configs.rgbdgaze_config import RGBDGaze_INDEX_DIR  # noqa: E402
from gazelab.datasets.rgbdgaze import RGBDGazeDataset  # noqa: E402
from gazelab.datasets.common import rgb_preprocess  # noqa: E402

idx = RGBDGaze_INDEX_DIR / "index.csv"
rows = []
import csv  # noqa: E402
with open(idx, encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        if r["subject"] == "p11" and r["activity"] == "walking":
            rows.append(r)
        if len(rows) >= 4:
            break

tiles = []
for name, sc_ in (("eye_scale=0 (历史)", 0.0), ("eye_scale=1.5 (收紧)", 1.5)):
    ds = RGBDGazeDataset(idx, color_transform=None, image_size=224,
                         eye_crops=True, eye_scale=sc_)
    # 用行号定位：与 rows 的 rgb_path 对齐
    with open(idx, encoding="utf-8") as fh:
        all_paths = [r["rgb_path"] for r in csv.DictReader(fh)]
    line = []
    for r in rows:
        i = all_paths.index(r["rgb_path"])
        face, el, er = ds._make_face(ROOT / r["rgb_path"],
                                     (float(r["bbox_x"]), float(r["bbox_y"]),
                                      float(r["bbox_w"]), float(r["bbox_h"])), i,
                                     [np.random.RandomState(0)] * 3)
        for t in (el, er):
            a = np.asarray(t)
            if a.ndim == 3 and a.shape[0] == 3:
                a = a.transpose(1, 2, 0)
            a = np.ascontiguousarray(a)
            a = ((a - a.min()) / max(float(np.ptp(a)), 1e-6) * 255).astype(np.uint8)
            line.append(cv2.resize(a, (140, 140)))
    cap = np.full((18, 140 * len(line), 3), 255, np.uint8)
    cv2.putText(cap, name, (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 180), 1)
    tiles += [cap, np.hstack(line)]

sheet = np.vstack(tiles)
cv2.imwrite(str(ROOT / "out/rgbdgaze/analysis/eye_scale_check.png"), sheet)
print("已写出 out/rgbdgaze/analysis/eye_scale_check.png")
print("形状:", sheet.shape)
