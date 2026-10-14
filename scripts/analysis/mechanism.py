"""误差可利用结构的机制分辨：在**固定的新鲜度预算**下，比较三种选点准则。

统一设定（严格因果）：池子 = 当前帧之前 300 帧（约 10 秒）；从中选 C=30 帧做修正。
  · time    : 最近的 30 帧（= 现有在线方案的实质）
  · space   : 注视位置（真值）离当前帧最近的 30 帧
  · vel     : 注视"速度"最接近当前帧的 30 帧
若 time ≫ space ⇒ 误差随时间漂移（同位置也会变）；
若 vel ≈ time ≫ space ⇒ 误差主要随"运动状态/速度"变化（可能与动画同步/滞后有关）。

另外单独测：误差与当前速度的相关（分桶），以及 lag-k 的误差自相关。
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

IDX = Path("/data1/xieliangcai/code/gazeformer-exps/data/RGBDGaze/index")
sc = {}
with open(IDX / "index.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        sc[r["rgb_path"]] = (float(r["screen_w"]) * float(r["cm_px_x"]),
                             float(r["screen_h"]) * float(r["cm_px_y"]))
CSV = sys.argv[1]

sess = defaultdict(list)
for r in csv.DictReader(open(CSV, encoding="utf-8")):
    sess[r["path"].rsplit("/", 1)[0]].append(
        dict(path=r["path"], fi=int(r["frame_index"]),
             P=np.array([float(r["pred_x"]), float(r["pred_y"])]),
             L=np.array([float(r["label_x"]), float(r["label_y"])]),
             S=sc[r["path"]]))
for s in sess:
    sess[s].sort(key=lambda r: r["fi"])

POOL, C = 300, 30
STD, VEL = [], []
base_all = []
acc = {k: [] for k in ("time", "space", "vel", "space_rec")}

for s, v in sess.items():
    L = np.array([r["L"] for r in v])
    P = np.array([r["P"] for r in v])
    S = np.array([r["S"] for r in v])
    V = np.zeros(len(v))                        # 速度（cm/帧）
    if len(v) > 1:
        d = np.diff(L, axis=0) * S[1:]
        V[1:] = np.hypot(d[:, 0], d[:, 1])
    D = P - L
    e = np.hypot(D[:, 0] * S[:, 0], D[:, 1] * S[:, 1])
    base_all.append(e)
    STD.append((V, e))
    for t in range(POOL, len(v)):
        lo = t - POOL
        idx = np.arange(lo, t)
        for key, sel in (("time", idx[-C:]),
                         ("space", idx[np.argsort(((L[idx] - L[t]) ** 2).sum(1))[:C]]),
                         ("vel", idx[np.argsort(np.abs(V[idx] - V[t]))[:C]]),
                         ("space_rec", None)):
            if key == "space_rec":              # 近 60 帧里按位置选 30 帧（近因+空间）
                rec = idx[-60:]
                sel = rec[np.argsort(((L[rec] - L[t]) ** 2).sum(1))[:C]]
            corr = P[t] - D[sel].mean(0)
            acc[key].append(float(np.hypot((corr[0] - L[t][0]) * S[t][0],
                                           (corr[1] - L[t][1]) * S[t][1])))

base = np.concatenate(base_all)
print("=" * 100)
print(f"【机制分辨】{Path(CSV).stem}  池子=过去 {POOL} 帧  选 {C} 帧")
print("=" * 100)
sub = np.concatenate([e[POOL:] for _v, e in STD])
print(f"  不标定基线： EM {sub.mean():.3f}  中位 {np.median(sub):.3f}  "
      f">3cm {100 * (sub > 3).mean():.2f}%")
print(f"  {'选点准则':<26}{'EM均值':>9}{'vs基线':>9}{'中位':>8}{'>3cm':>8}")
for k, tag in (("time", "最近 30 帧（时间）"), ("space", "注视位置最近 30 帧（空间）"),
               ("vel", "速度最接近 30 帧（运动）"), ("space_rec", "近 60 帧内按位置选 30")):
    a = np.array(acc[k])
    print(f"  {tag:<26}{a.mean():>9.3f}{100 * (a.mean() / sub.mean() - 1):>8.1f}%"
          f"{np.median(a):>8.3f}{100 * (a > 3).mean():>7.2f}%")

# 误差 vs 速度
V = np.concatenate([x[0] for x, _ in STD])
E = np.concatenate([x[1] for _v, x in STD])
qs = np.percentile(V, [0, 20, 40, 60, 80, 100])
print("\n  误差 vs 注视速度（cm/帧）：")
for a, b in zip(qs[:-1], qs[1:]):
    m = (V >= a) & (V <= b if b == qs[-1] else V < b)
    if m.sum():
        print(f"    速度 {a:.3f}~{b:.3f}: n={m.sum():>5}  EM {E[m].mean():.3f}  "
              f"中位 {np.median(E[m]):.3f}  >3cm {100 * (E[m] > 3).mean():.2f}%")
# 误差自相关（滞后 k 帧）
print("\n  误差的自相关（滞后 k 帧，按 session 平均）：")
for k in (1, 5, 15, 30, 60, 150, 300):
    cs = []
    for v in sess.values():
        L = np.array([r["L"] for r in v]); P = np.array([r["P"] for r in v])
        S = np.array([r["S"] for r in v])
        D = (P - L) * S
        if len(v) > k + 10:
            cs.append(np.corrcoef(D[:-k, 0], D[k:, 0])[0, 1])
    print(f"    lag={k:>3}: corr(dx) = {np.nanmean(cs):+.3f}")
