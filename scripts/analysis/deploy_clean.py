"""可部署协议·干净版：无 oracle、严格因果。

修正前面两个缺陷：
 (1) 查询必须用**预测注视点** P_t（真实系统只有这个 ✗），不能用真值 L_t（oracle）；
 (2) 标注按帧**逐个到达**（burst 的后续帧在它们各自经过时才可用），不用未来帧。
标注预算模型：每 N 帧出现一次机会（连续 m 帧的短暂注视），全部采纳。
估计器：池（最近 20 个已到达的已知点）中取"离 P_t 最近"的 10 个，用其平均偏差修正。
同时给出 oracle 查询版本，量化"用真值当查询"会虚高多少。
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
print("=" * 104)
print(f"【可部署协议·干净版】{Path(CSV).stem}  稳态基线 EM {base.mean():.3f}"
      f"  >3cm {100 * (base > 3).mean():.2f}%")
print("=" * 104)


def run(N=600, m=5, C=C, L=L, oracle=False, Cw=None):
    errs, n, k = [], 0, 0
    for s, v in sess.items():
        pool = []
        for t, r in enumerate(v):
            # 机会：每 N 帧开始的一次短暂注视，其帧随经过逐帧变为已知
            if (t % N) < m and t % N < len(v):
                pool.append((r["L"], r["P"]))
                pool = pool[-L:]
                if t >= WARM:
                    k += 1
            if t < WARM:
                continue
            n += 1
            q = r["L"] if oracle else r["P"]
            if not pool:
                corr = r["P"]
            else:
                D = np.array([a[1] - a[0] for a in pool])
                Ll = np.array([a[0] for a in pool])
                sel = np.argsort(((Ll - q) ** 2).sum(1))[:min(C, len(pool))]
                corr = r["P"] - D[sel].mean(0)
            errs.append(float(np.hypot((corr[0] - r["L"][0]) * r["S"][0],
                                       (corr[1] - r["L"][1]) * r["S"][1])))
    a = np.array(errs)
    return a.mean(), 100 * (a > 3).mean(), 100 * k / max(n, 1)


print(f"\n  {'协议':<34}{'占空比':>8}{'EM均值':>9}{'vs基线':>9}{'>3cm':>8}")
for N, m in ((600, 5), (300, 5), (150, 5), (60, 5), (300, 15), (150, 15), (60, 15)):
    mm, t3, d = run(N=N, m=m)
    print(f"  {'每 %d 帧机会 %d 帧（预测查询）' % (N, m):<30}{d:>8.1f}%{mm:>9.3f}"
          f"{100 * (mm / base.mean() - 1):>8.1f}%{t3:>7.2f}%")
print("  " + "-" * 92)
for N, m in ((600, 5), (300, 5), (150, 5)):
    mm, t3, d = run(N=N, m=m, oracle=True)
    print(f"  {'每 %d 帧机会 %d 帧（oracle 查询）' % (N, m):<30}{d:>8.1f}%{mm:>9.3f}"
          f"{100 * (mm / base.mean() - 1):>8.1f}%{t3:>7.2f}%")
print("  " + "-" * 92)
for Cw in (3, 5, 20):
    mm, t3, d = run(N=150, m=15, C=Cw)
    print(f"  {'每 150 帧 15 帧, 取最近 C=%d' % Cw:<30}{d:>8.1f}%{mm:>9.3f}"
          f"{100 * (mm / base.mean() - 1):>8.1f}%{t3:>7.2f}%")
print(f"\n  参考：场（离线 4x4 留一，非因果）= 0.932 vs 全量基线 1.472（−36.7%）")
