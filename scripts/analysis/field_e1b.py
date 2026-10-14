"""E1 终版：机会门控（gated）vs 全采纳（periodic），在**匹配的标注预算**下比较。

协议：每 N 帧出现一次"已知答案的机会"（burst = 连续 m 帧，模拟一次短暂注视）。
  periodic : 所有机会都采纳
  gated    : 只有该机会的中心注视位置距离池中最近点 > θ 时才采纳（改善覆盖）
估计器固定：池（最近 20 个已知点）取注视位置最近 10 个的平均偏差。
一律只统计每个 session 第 300 帧之后的稳态，基线也在同一批帧上算。
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
WARM, C, L = 300, 10, 20

sess = defaultdict(list)
for r in csv.DictReader(open(CSV, encoding="utf-8")):
    sess[r["path"].rsplit("/", 1)[0]].append(
        dict(P=np.array([float(r["pred_x"]), float(r["pred_y"])]),
             L=np.array([float(r["label_x"]), float(r["label_y"])]),
             S=sc[r["path"]]))
for s in sess:
    sess[s].sort(key=lambda r: 0)

base = np.concatenate([np.array([float(np.hypot((r["P"][0] - r["L"][0]) * r["S"][0],
                                              (r["P"][1] - r["L"][1]) * r["S"][1]))
                                  for r in sess[s][WARM:]]) for s in sess])
print("=" * 100)
print(f"【E1 终版】{Path(CSV).stem}  稳态基线（跳过前 {WARM} 帧）EM {base.mean():.3f}"
      f"  >3cm {100 * (base > 3).mean():.2f}%")
print("=" * 100)


def run(N=600, m=5, theta=None):
    errs, n, k = [], 0, 0
    for s, v in sess.items():
        pool = []
        for t, r in enumerate(v):
            if t % N == 0:
                blk = [v[t + j] for j in range(m) if t + j < len(v)]
                cen = np.mean([b["L"] for b in blk], axis=0)
                ok = True
                if theta is not None and pool:
                    ok = min(float(np.hypot((q[0][0] - cen[0]) * r["S"][0],
                                            (q[0][1] - cen[1]) * r["S"][1]))
                             for q in pool) > theta
                if ok:
                    for b in blk:
                        pool.append((b["L"], b["P"]))
                    pool = pool[-L:]
                    if t >= WARM:
                        k += len(blk)
            if t < WARM:
                continue
            n += 1
            if not pool:
                corr = r["P"]
            else:
                D = np.array([q[1] - q[0] for q in pool])
                Ll = np.array([q[0] for q in pool])
                sel = np.argsort(((Ll - r["L"]) ** 2).sum(1))[:min(C, len(pool))]
                corr = r["P"] - D[sel].mean(0)
            errs.append(float(np.hypot((corr[0] - r["L"][0]) * r["S"][0],
                                       (corr[1] - r["L"][1]) * r["S"][1])))
    a = np.array(errs)
    return a.mean(), 100 * (a > 3).mean(), 100 * k / max(n, 1)


print(f"\n  {'协议':<30}{'占空比':>9}{'EM均值':>9}{'vs基线':>9}{'>3cm':>8}")
for N in (600, 300, 150, 90, 60):
    m, t3, d = run(N=N, m=5)
    print(f"  {'periodic 每 %d 帧 5 帧' % N:<30}{d:>8.1f}%{m:>9.3f}"
          f"{100 * (m / base.mean() - 1):>8.1f}%{t3:>7.2f}%")
print()
for N in (300, 150, 90):
    for th in (0.20, 0.35, 0.50):
        m, t3, d = run(N=N, m=5, theta=th)
        print(f"  {'gated  每 %d 帧/θ=%.2f' % (N, th):<30}{d:>8.1f}%{m:>9.3f}"
              f"{100 * (m / base.mean() - 1):>8.1f}%{t3:>7.2f}%")
