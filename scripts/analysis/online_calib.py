"""在线（递推）标定仿真：只用**当前帧之前**的若干帧估计偏置，逐帧修正。

目的：把"持续标定"的可获得收益**上界**量出来，并量化漂移造成的代价。零训练。

对照档：
  A 不标定（下界） / B 在线标定（只用过去 K 帧，可加时间衰减）
  C 事后 oracle（用整场均值 ⇒ 离线上界）

用法: python online_calib.py <预测CSV>
CSV 需含 path, frame_index, pred_x, pred_y, label_x, label_y
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

sess = defaultdict(list)
for r in csv.DictReader(open(sys.argv[1], encoding="utf-8")):
    sess[r["path"].rsplit("/", 1)[0]].append(
        dict(path=r["path"], fi=int(r["frame_index"]),
             pred=(float(r["pred_x"]), float(r["pred_y"])),
             lab=(float(r["label_x"]), float(r["label_y"]))))
for s in sess:
    sess[s].sort(key=lambda r: r["fi"])


def session_arrays(v):
    P = np.array([r["pred"] for r in v], dtype=np.float64)
    L = np.array([r["lab"] for r in v], dtype=np.float64)
    S = np.array([sc[r["path"]] for r in v], dtype=np.float64)   # (w_cm, h_cm)
    D = P - L                                                    # 归一化偏差
    return P, L, S, D


def ev(mode, K=0, decay=0.0):
    """mode: 'none' | 'online' | 'oracle'。返回 (全部帧误差, 稳态帧误差)。"""
    errs, steady = [], []
    for s, v in sess.items():
        P, L, S, D = session_arrays(v)
        md_oracle = D.mean(0)
        for t in range(len(v)):
            if mode == "none":
                corr = P[t]
            elif mode == "oracle":
                corr = P[t] - md_oracle
            else:
                if t < 2:
                    corr = P[t]
                else:
                    lo = max(0, t - K)
                    w = np.exp(-decay * np.arange(t - lo - 1, -1, -1))
                    corr = P[t] - (D[lo:t] * w[:, None]).sum(0) / w.sum()
            e = float(np.hypot((corr[0] - L[t][0]) * S[t][0],
                               (corr[1] - L[t][1]) * S[t][1]))
            errs.append(e)
            if mode == "online" and t >= 2 * K:
                steady.append(e)
    return np.array(errs), np.array(steady)


base, _ = ev("none")
orc, _ = ev("oracle")
print("=" * 104)
print(f"【在线标定仿真】{Path(sys.argv[1]).stem}   n={len(base)}  sessions={len(sess)}")
print("=" * 104)
print(f"  A 不标定          : EM 均值 {base.mean():.3f}  中位 {np.median(base):.3f}  "
      f">3cm {100 * (base > 3).mean():.2f}%  >5cm {100 * (base > 5).mean():.2f}%  "
      f"max {base.max():.2f}")
print(f"  C oracle(整场均值): EM 均值 {orc.mean():.3f} "
      f"({100 * (orc.mean() / base.mean() - 1):+.1f}%)  ← 离线上界")

print(f"\n  {'K(帧)':>7}{'≈秒':>6}{'加权':>8}{'EM均值':>9}{'相对基线':>10}"
      f"{'稳态部分':>10}{'中位':>8}{'>3cm':>8}{'max':>7}")
for K, dec, tag in ((30, 0.0, "等权"), (60, 0.0, "等权"), (150, 0.0, "等权"),
                    (300, 0.0, "等权"), (600, 0.0, "等权"), (1200, 0.0, "等权"),
                    (150, 0.02, "衰0.02"), (300, 0.02, "衰0.02"),
                    (600, 0.01, "衰0.01"), (1500, 0.005, "衰0.005")):
    e, ea = ev("online", K, dec)
    print(f"  {K:>7}{K / 30:>6.0f}{tag:>9}{e.mean():>9.3f}"
          f"{100 * (e.mean() / base.mean() - 1):>9.1f}%{ea.mean():>10.3f}"
          f"{np.median(ea):>8.3f}{100 * (ea > 3).mean():>7.2f}%{e.max():>7.2f}")
