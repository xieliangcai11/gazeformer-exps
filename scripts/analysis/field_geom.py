"""误差场的两种**表示**对照（因果在线）：屏幕空间 vs 角度空间，± 时间衰减。

动机（这是此前没测的一类改进 ✗）：
  · cm 误差 = 视距 × tan(角度偏差) ⇒ "收缩"本质是**角度增益**误差 ✓
  · 按偏航/俯仰（yaw/pitch，度）建场：格子少（本例 yaw ±6.5°、pitch ±14.5° ✓）、
    样本效率高 ✓、且**跨设备天然统一**（不同屏尺寸在角度空间重合 ✓✓）
  · 场会缓慢漂移（前半 vs 后半相关 0.86 ✓ 非 1.0）⇒ 加时间衰减（EMA）试试 ✓

对照：屏幕空间（现有）/ 角度空间 / 各自 ± 衰减
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/xieliangcai/code/gazeformer-exps")
CSV = sys.argv[1]
D_CM = 30.0          # 基准视距（与训练口径一致）
MIN = 8
BIN = 2.0            # 角度网格：2 度


def load():
    sc = {}
    for _idx in (ROOT / "data/RGBDGaze/index", ROOT / "data/GazeCapture/index"):
        if not _idx.exists():
            continue
        with open(_idx / "index.csv", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                pk = "rgb_path" if "rgb_path" in r else "frame_path"
                sc[r[pk]] = (float(r["screen_w"]) * float(r["cm_px_x"]),
                             float(r["screen_h"]) * float(r["cm_px_y"]),
                             float(r["gaze_x"]) / float(r["screen_w"]),
                             float(r["gaze_y"]) / float(r["screen_h"]))
    sess = defaultdict(list)
    for r in csv.DictReader(open(CSV, encoding="utf-8")):
        if r["path"] not in sc:
            continue
        sess[r["path"].rsplit("/", 1)[0]].append(
            dict(fi=int(r["frame_index"]),
                 P=np.array([float(r["pred_x"]), float(r["pred_y"])]),
                 L=np.array([float(r["label_x"]), float(r["label_y"])]),
                 S=sc[r["path"]][:2]))
    for s in sess:
        sess[s].sort(key=lambda r: r["fi"])
    return sess


sess = load()


def key_screen(pt, G=8):
    return (min(G - 1, max(0, int(pt[0] * G))), min(G - 1, max(0, int(pt[1] * G))))


def key_angle(pt, w_cm, h_cm):
    yaw = np.degrees(np.arctan((pt[0] - 0.5) * w_cm / D_CM))
    pit = np.degrees(np.arctan((pt[1] - 0.5) * h_cm / D_CM))
    return (int(np.floor(yaw / BIN)), int(np.floor(pit / BIN)))


def run(mode="angle", decay=0.0):
    errs = []
    for s, v in sess.items():
        grid = defaultdict(lambda: [0.0, 0.0, 0.0])
        for r in v:
            P, L = r["P"], r["L"]
            w_cm, h_cm = r["S"]
            if mode == "angle":
                k0 = key_angle(P, w_cm, h_cm)
            else:
                k0 = key_screen(P)
            cnt = sx = sy = 0.0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    c = grid.get((k0[0] + di, k0[1] + dj))
                    if c:
                        cnt += c[0]; sx += c[1]; sy += c[2]
            if cnt >= MIN:
                corr = P - np.array([sx / cnt, sy / cnt])
            else:
                corr = P
            errs.append(float(np.hypot((corr[0] - L[0]) * w_cm,
                                       (corr[1] - L[1]) * h_cm)))
            d = P - L
            for di in (0,):
                c = grid[(k0[0], k0[1])]
                c[0] += 1
                c[1] += float(d[0])
                c[2] += float(d[1])
        if decay > 0:
            pass
    return np.array(errs)


def run_decay(mode="angle", tau=3000.0):
    """带时间衰减：每个 cell 的累计量按 exp(-Δt/tau) 衰减（用逐帧乘子近似）。"""
    mult = np.exp(-1.0 / tau)
    errs = []
    for s, v in sess.items():
        grid = defaultdict(lambda: [0.0, 0.0, 0.0])
        for r in v:
            P, L = r["P"], r["L"]
            w_cm, h_cm = r["S"]
            k0 = key_angle(P, w_cm, h_cm) if mode == "angle" else key_screen(P)
            cnt = sx = sy = 0.0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    c = grid.get((k0[0] + di, k0[1] + dj))
                    if c:
                        cnt += c[0]; sx += c[1]; sy += c[2]
            corr = P - np.array([sx / cnt, sy / cnt]) if cnt >= MIN else P
            errs.append(float(np.hypot((corr[0] - L[0]) * w_cm,
                                       (corr[1] - L[1]) * h_cm)))
            for kk in list(grid.keys()):
                grid[kk][0] *= mult
                grid[kk][1] *= mult
                grid[kk][2] *= mult
            c = grid[(k0[0], k0[1])]
            c[0] += 1
            c[1] += float(P[0] - L[0])
            c[2] += float(P[1] - L[1])
    return np.array(errs)


base = np.array([float(np.hypot((r["P"][0] - r["L"][0]) * r["S"][0],
                                (r["P"][1] - r["L"][1]) * r["S"][1]))
                 for s in sess for r in sess[s]])
print("=" * 96)
print(f"【场的表示对照】{Path(CSV).stem}  基线 EM {base.mean():.3f}"
      f"  >3cm {100 * (base > 3).mean():.2f}%  视距假设 {D_CM:.0f}cm  角度格宽 {BIN}°")
print("=" * 96)
print(f"  {'表示 / 衰减':<34}{'EM均值':>9}{'vs基线':>9}{'中位':>8}{'>3cm':>8}")
for mode, tag in (("screen", "屏幕空间（现有 G=8 + 3x3）"), ("angle", "角度空间（yaw/pitch 2° + 3x3）")):
    e = run(mode)
    print(f"  {tag:<34}{e.mean():>9.3f}{100 * (e.mean() / base.mean() - 1):>8.1f}%"
          f"{np.median(e):>8.3f}{100 * (e > 3).mean():>7.2f}%")
for mode, tag in (("angle", "角度空间 + 衰减 τ=3000帧"), ("angle", "角度空间 + 衰减 τ=1000帧")):
    tau = 3000.0 if "3000" in tag else 1000.0
    e = run_decay(mode, tau)
    print(f"  {tag:<34}{e.mean():>9.3f}{100 * (e.mean() / base.mean() - 1):>8.1f}%"
          f"{np.median(e):>8.3f}{100 * (e > 3).mean():>7.2f}%")
