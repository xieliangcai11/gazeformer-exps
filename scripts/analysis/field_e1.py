"""E1 修正版：同等标注预算下，周期性取点 vs 主动铺点（coverage-driven）。

修正上一版的错误：池子从**空**开始，只有被"取"的帧才进池（不再 warmup 全取 ✗）。
统计稳态（跳过每 session 前 300 帧）的占空比与误差，保证公平对比。
估计器固定：池（最近 20 个已知点）取注视位置最近 10 个，用其平均偏差修正。
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
WARM = 300          # 稳态起点（同时作为"初始铺点"的观察期）

sess = defaultdict(list)
for r in csv.DictReader(open(CSV, encoding="utf-8")):
    sess[r["path"].rsplit("/", 1)[0]].append(
        dict(P=np.array([float(r["pred_x"]), float(r["pred_y"])]),
             L=np.array([float(r["label_x"]), float(r["label_y"])]),
             S=sc[r["path"]]))
for s in sess:
    sess[s].sort(key=lambda r: 0)

allbase = np.concatenate([np.array([float(np.hypot((r["P"][0] - r["L"][0]) * r["S"][0],
                                                 (r["P"][1] - r["L"][1]) * r["S"][1]))
                                     for r in sess[s][WARM:]]) for s in sess])
print("=" * 104)
print(f"【E1 修正版】{Path(CSV).stem}   稳态基线（跳过前 {WARM} 帧）EM "
      f"{allbase.mean():.3f}  >3cm {100 * (allbase > 3).mean():.2f}%")
print("=" * 104)


def run(strategy, param, C=10, L=20):
    errs, n, k = [], 0, 0
    for s, v in sess.items():
        pool = []
        for t, r in enumerate(v):
            if t >= WARM:
                n += 1
            take = False
            if strategy == "periodic":
                take = (t % param == 0)
            elif strategy == "coverage":
                if not pool:
                    take = True
                else:
                    d = min(float(np.hypot((q[1][0] - r["L"][0]) * r["S"][0],
                                           (q[1][1] - r["L"][1]) * r["S"][1]))
                            for q in pool)
                    take = d > param
            if take:
                pool.append((r["L"], r["P"]))
                pool = pool[-L:]
                if t >= WARM:
                    k += 1
            if t < WARM:
                continue
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


print(f"\n  {'策略':<26}{'稳态占空比':>12}{'EM均值':>9}{'vs基线':>9}{'>3cm':>8}")
for k in (600, 300, 150, 90, 60, 30, 10):
    m, t3, d = run("periodic", k)
    print(f"  {'periodic 每 %d 帧取 1' % k:<26}{d:>11.1f}%{m:>9.3f}"
          f"{100 * (m / allbase.mean() - 1):>8.1f}%{t3:>7.2f}%")
for th in (0.05, 0.10, 0.15, 0.25, 0.40, 0.60):
    m, t3, d = run("coverage", th)
    print(f"  {'coverage 阈值 %.2f' % th:<26}{d:>11.1f}%{m:>9.3f}"
          f"{100 * (m / allbase.mean() - 1):>8.1f}%{t3:>7.2f}%")
