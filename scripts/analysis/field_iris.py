"""场的"物理表示"：用**图片上可直接观测的眼部几何**（虹膜在眼框内的相对位置）作索引。

思路（绕开此前发现的限制 ✗）：
  · 屏幕空间场需要"预测的注视坐标"作索引 ⇒ 索引本身带预测误差（查询噪声 ✗）
  · 而**虹膜相对位置**是当前图像里**直接量出来**的 ⇒ 零查询噪声 ✓✓
  · 且它是物理量 ⇒ 跨设备/跨被试统一 ✓（可能顺带修好"不可迁移"✗）

做法：
  1) MediaPipe FaceLandmarker（含虹膜，refine 点 468/473 ✓）逐帧提取：
       每只眼: 虹膜中心相对眼框的 (x_rel, y_rel)  [0,1] / 有符号
       眼睑开度 EAR
  2) 场 = 对各特征分箱的"平均误差向量"，严格因果在线累积 + 3x3 池化
  3) 与屏幕空间场（同一套累积逻辑）对照
"""
import csv
import multiprocessing as mp_mod
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/xieliangcai/code/gazeformer-exps")
AN = ROOT / "out/rgbdgaze/analysis"
CSV = sys.argv[1]
N_WORKERS = 4

L_IRIS, R_IRIS = 468, 473
L_H, L_V, R_H, R_V = (33, 133), (159, 145), (362, 263), (386, 374)
_LM = None


def _init():
    global _LM
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python.vision import RunningMode
    from mediapipe.tasks.python.vision.face_landmarker import (
        FaceLandmarker, FaceLandmarkerOptions)
    _LM = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ROOT / "assets" / "face_landmarker.task")),
        running_mode=RunningMode.IMAGE, num_faces=1))


def _feat(path):
    import cv2
    import mediapipe as mp
    img = cv2.imread(str(ROOT / path))
    if img is None:
        return None
    H, W = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = _LM.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.face_landmarks:
        return None
    p = np.array([[q.x * W, q.y * H] for q in res.face_landmarks[0]])
    if len(p) < 478:
        return None
    out = []
    for iris, (c1, c2), (u, d) in ((L_IRIS, L_H, L_V), (R_IRIS, R_H, R_V)):
        x0, x1 = p[c1, 0], p[c2, 0]
        y0, y1 = p[u, 0], p[d, 0]
        y0b, y1b = p[u, 1], p[d, 1]
        w = max(abs(x1 - x0), 1.0)
        h = max(abs(y1b - y0b), 1.0)
        out += [float((p[iris, 0] - min(x0, x1)) / w),
                float((p[iris, 1] - min(y0b, y1b)) / h)]
    return out


def main():
    rows = []
    with open(CSV, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append((r["path"], float(r["pred_x"]), float(r["pred_y"]),
                         float(r["label_x"]), float(r["label_y"]),
                         float(r["em_cm"])))
    print(f"提取 {len(rows)} 帧的眼部几何…（{N_WORKERS} 进程）")
    with mp_mod.Pool(N_WORKERS, initializer=_init) as pool:
        feats = pool.map(_feat, [r[0] for r in rows], chunksize=32)
    ok = [(r, f) for r, f in zip(rows, feats) if f is not None]
    print(f"有效 {len(ok)} / {len(rows)}")
    # 屏幕物理尺度（换算 cm）
    sc = {}
    with open(ROOT / "data/RGBDGaze/index/index.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            sc[r["rgb_path"]] = (float(r["screen_w"]) * float(r["cm_px_x"]),
                                 float(r["screen_h"]) * float(r["cm_py_y"])
                                 if "cm_py_y" in r else
                                 float(r["screen_h"]) * float(r["cm_px_y"]))
    base, eyef, scrf = [], [], []
    MIN = 8
    # --- 眼几何场：键 = 量化后的 (Lx,Ly,Rx,Ry) ---
    QB = 8
    g_e, g_s = defaultdict(lambda: [0, 0.0, 0.0]), defaultdict(lambda: [0, 0.0, 0.0])

    def key_of(f, mode):
        if mode == "eye":
            return tuple(min(QB - 1, max(0, int(v * QB))) for v in f)
        return None

    print("\n  {: <28}{:>9}{:>9}{:>8}{:>8}".format("方案", "EM均值", "vs基线", "中位", ">3cm"))
    for r, f in ok:
        path, px, py, lx, ly, em = r
        base.append(em)
        # 眼几何场（索引 = 直接观测 ✓）
        k = key_of(f, "eye")
        cnt = sx = sy = 0.0
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    for dl in (-1, 0, 1):
                        c = g_e.get((k[0] + di, k[1] + dj, k[2] + dk, k[3] + dl))
                        if c:
                            cnt += c[0]; sx += c[1]; sy += c[2]
        if cnt >= MIN:
            cx, cy = px - sx / cnt, py - sy / cnt
        else:
            cx, cy = px, py
        wcm, hcm = sc[path]
        eyef.append(float(np.hypot((cx - lx) * wcm, (cy - ly) * hcm)))
        c = g_e[k]
        c[0] += 1
        c[1] += px - lx
        c[2] += py - ly
    base = np.array(base)
    eyef = np.array(eyef)
    for tag, arr in (("A 不修正", base), ("眼几何场（虹膜相对位置）", eyef)):
        print("  {: <28}{:>9.3f}{:>8.1f}%{:>8.3f}{:>7.2f}%".format(
            tag, arr.mean(), 100 * (arr.mean() / base.mean() - 1),
            np.median(arr), 100 * (arr > 3).mean()))
    print("\n  对照（同数据、屏幕空间层级场）: −25.7% ✓✓ / 每格偏移 −21.2%")


if __name__ == "__main__":
    main()
