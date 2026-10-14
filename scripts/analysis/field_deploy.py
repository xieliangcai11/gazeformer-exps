"""把"误差场"变成可部署方法：网格热力图 + 查询方式对照（严格因果）。

关键设计问题（本实验要回答的）：
  误差场的**索引键**应该用"预测注视点"还是"真值注视点"？
    · 测试时只有预测可用 ⇒ 若键用真值、查询用预测，则必然查错格（键不匹配 ✗）
    · 若键与查询都用预测（同一空间 ✓），则一致、但表本身带噪
  再加**迭代一次**（用修正后的位置重新查表）看能否进一步改善。

5 路：
  A 不修正
  B 表键=预测，查=预测            （一致、完全因果）
  C 同 B + 1 次迭代              （用修正后位置重查）
  D 表键=真值，查=真值            （oracle ⇒ 场本身的上界）
  E 表键=真值，查=预测            （我此前的做法 ⇒ 量化"键不匹配"的代价）

另给出**预热曲线**（按 session 内已积累帧数分桶）—— 回答"要多久标定才起效"。
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/xieliangcai/code/gazeformer-exps")
CSV = sys.argv[1]
G = 6                      # 6x6 网格
SMOOTH = 1                 # 1=仅本格；3=3x3 邻域加权
MINCNT = 5                 # 本格样本数下限，低于则用 3x3 邻域

sc = {}
for _idx in (ROOT / "data/RGBDGaze/index", ROOT / "data/GazeCapture/index"):
    if not _idx.exists():
        continue
    with open(_idx / "index.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            pk = "rgb_path" if "rgb_path" in r else "frame_path"
            sc[r[pk]] = (float(r["screen_w"]) * float(r["cm_px_x"]),
                         float(r["screen_h"]) * float(r["cm_px_y"]))

sess = defaultdict(list)
for r in csv.DictReader(open(CSV, encoding="utf-8")):
    sess[r["path"].rsplit("/", 1)[0]].append(
        dict(fi=int(r["frame_index"]),
             P=np.array([float(r["pred_x"]), float(r["pred_y"])]),
             L=np.array([float(r["label_x"]), float(r["label_y"])]),
             S=sc[r["path"]]))
for s in sess:
    sess[s].sort(key=lambda r: r["fi"])


def cell(v):
    return (min(G - 1, max(0, int(v[0] * G))), min(G - 1, max(0, int(v[1] * G))))


def lookup(cnt, acc, q):
    """查表：优先本格（≥MINCNT），否则 3x3 邻域加权。返回 None 表示无可用估计。"""
    c = cell(q)
    if cnt[c] >= MINCNT:
        return acc[c] / cnt[c]
    if SMOOTH == 3:
        i0, j0 = c
        s = np.zeros(2)
        n = 0.0
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                i, j = i0 + di, j0 + dj
                if 0 <= i < G and 0 <= j < G and cnt[i, j] > 0:
                    w = 1.0 / (1 + abs(di) + abs(dj))
                    s += w * acc[i, j]
                    n += w * cnt[i, j] * 0 + w
        if n > 0:
            return s / n
    return None


def run(mode):
    errs, elapsed = [], []
    for s, v in sess.items():
        cnt = np.zeros((G, G))
        acc = np.zeros((G, G, 2))
        cntL = np.zeros((G, G))
        accL = np.zeros((G, G, 2))
        for t, r in enumerate(v):
            q = r["P"]
            corr = r["P"]
            if mode in ("B", "C", "E"):
                if mode == "E":
                    est = lookup(cntL, accL, q)
                else:
                    est = lookup(cnt, acc, q)
                if est is not None:
                    corr = r["P"] - est
                if mode == "C":
                    est2 = lookup(cnt, acc, corr)
                    if est2 is not None:
                        corr = r["P"] - est2
            elif mode == "D":
                est = lookup(cntL, accL, r["L"])
                if est is not None:
                    corr = r["P"] - est
            e = float(np.hypot((corr[0] - r["L"][0]) * r["S"][0],
                               (corr[1] - r["L"][1]) * r["S"][1]))
            errs.append(e)
            elapsed.append(t)
            # 更新（在修正之后，严格因果）
            d = r["P"] - r["L"]
            c = cell(r["P"])
            cnt[c] += 1
            acc[c] += d
            c2 = cell(r["L"])
            cntL[c2] += 1
            accL[c2] += d
    return np.array(errs), np.array(elapsed)


base, el = run("A")
print("=" * 104)
print(f"【误差场可部署化】{Path(CSV).stem}  n={len(base)}  G={G}×{G}  smooth={SMOOTH}"
      f"  基线 EM {base.mean():.3f}")
print("=" * 104)
print(f"  {'方案':<44}{'EM均值':>9}{'vs基线':>9}{'中位':>8}{'>3cm':>8}")
res = {}
for m, tag in (("A", "A 不修正"), ("B", "B 表键=预测, 查=预测（完全因果）"),
               ("C", "C 同 B + 迭代 1 次"), ("E", "E 表键=真值, 查=预测（键不匹配）"),
               ("D", "D 表键=真值, 查=真值（oracle 上界）")):
    e, _ = run(m)
    res[m] = e
    print(f"  {tag:<44}{e.mean():>9.3f}{100 * (e.mean() / base.mean() - 1):>8.1f}%"
          f"{np.median(e):>8.3f}{100 * (e > 3).mean():>7.2f}%")

print("\n  【预热曲线】按 session 内已积累帧数分桶（EM 均值）")
print(f"  {'已积累帧数':<16}" + "".join(f"{k:>14}" for k in
                                    ("A 不修正", "B 因果", "C 迭代", "D oracle")))
bins = [(0, 300), (300, 1000), (1000, 3000), (3000, 6000), (6000, 10 ** 9)]
for a, b in bins:
    hi = "inf" if b >= 10 ** 9 else str(b)
    row = f"  {a}-{hi:<12}"
    for m in ("A", "B", "C", "D"):
        e = res[m][(el >= a) & (el < b)]
        row += f"{(e.mean() if len(e) else float('nan')):>14.3f}"
    print(row)
