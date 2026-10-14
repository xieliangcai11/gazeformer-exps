"""可部署标定协议：低占空比已知点 + 近因滚动池 + 空间选点。

协议模型（贴近真实 UX）：
  每 N 帧，应用层才"知道"一次答案（例如用户短暂看向一个已知 UI 元素/闪烁点），
  其余帧没有任何标注。
  维护"最近 L 次已知点"的滚动池（≈ L·N/30 秒的历史）。
  修正当前帧时：从池中选**注视位置最近**的 C 个已知点，取其 (pred-label) 均值减掉。

对比：不修正 / 时间选点 / 空间选点；并给出占空比 vs 收益曲线（= 论文主图）。
另附：误差-速度分桶、误差自相关（回答"漂移多快"）。
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


def run(N=150, L=20, C=10, sel="space", m=1, warm=True):
    """N: 每 N 帧到账一次；m: 每次到账给 m 帧（连续）；L: 池大小(靠后的 L 个已知点)；
    C: 选几个；sel: space|time。返回评估帧误差。"""
    errs = []
    for s, v in sess.items():
        pool = []                                    # (t, P, L)
        for t in range(len(v)):
            if t % N == 0:
                for k in range(m):
                    if t + k < len(v):
                        pool.append((t + k, v[t + k]["P"], v[t + k]["L"]))
            if t < N * 2 or not pool:
                corr = v[t]["P"]
            else:
                use = pool[-L:]
                T = np.array([p[0] for p in use], dtype=float)
                Rp = np.array([p[1] for p in use])
                Rl = np.array([p[2] for p in use])
                D = Rp - Rl
                if sel == "time":
                    take = np.argsort(T)[-min(C, len(use)):]
                else:
                    take = np.argsort(((Rl - v[t]["L"]) ** 2).sum(1))[:min(C, len(use))]
                corr = v[t]["P"] - D[take].mean(0)
            e = float(np.hypot((corr[0] - v[t]["L"][0]) * v[t]["S"][0],
                               (corr[1] - v[t]["L"][1]) * v[t]["S"][1]))
            if t >= N * 2:
                errs.append(e)
    return np.array(errs)


all_err = np.concatenate([[float(np.hypot((r["P"][0] - r["L"][0]) * r["S"][0],
                                          (r["P"][1] - r["L"][1]) * r["S"][1]))
                           for r in sess[s]] for s in sess])
print("=" * 104)
print(f"【可部署标定协议】{Path(CSV).stem}")
print("=" * 104)
print(f"  不修正全量基线： EM {all_err.mean():.3f}  中位 {np.median(all_err):.3f}  "
      f">3cm {100 * (all_err > 3).mean():.2f}%")
print(f"\n  {'协议（每 N 帧 1 个已知点）':<34}{'池':>4}{'选C':>5}{'准则':>7}"
      f"{'EM均值':>9}{'>3cm':>8}{'已知点占比':>11}")
for N, m in ((30, 1), (60, 1), (150, 1), (300, 1), (600, 1), (300, 5), (600, 5)):
    for L, C, sel, tag in ((20, 10, "space", "空间"), (20, 10, "time", "时间"),
                           (100, 10, "space", "空间"), (40, 20, "space", "空间")):
        e = run(N=N, L=L, C=C, sel=sel, m=m)
        print(f"  {'每 %d 帧 1 点 (m=%d)' % (N, m):<34}{L:>4}{C:>5}{tag:>7}"
              f"{e.mean():>9.3f}{100 * (e > 3).mean():>7.2f}%{m / N * 100:>10.1f}%")

# 速度 / 自相关
V, E = [], []
for s, v in sess.items():
    L = np.array([r["L"] for r in v]); P = np.array([r["P"] for r in v])
    S = np.array([r["S"] for r in v])
    if len(v) < 3:
        continue
    d = np.diff(L, axis=0) * S[1:]
    V.append(np.hypot(d[:, 0], d[:, 1])); E.append(np.hypot((P - L)[1:, 0] * S[1:, 0],
                                                            (P - L)[1:, 1] * S[1:, 1]))
V = np.concatenate(V); E = np.concatenate(E)
print(f"\n  误差 vs 注视速度： corr = {np.corrcoef(V, E)[0, 1]:+.3f}")
qs = np.percentile(V, [0, 25, 50, 75, 100])
for a, b in zip(qs[:-1], qs[1:]):
    mm = (V >= a) & (V <= b if b == qs[-1] else V < b)
    if mm.sum():
        print(f"    速度 {a:.3f}~{b:.3f} cm/帧: n={mm.sum():>5} EM {E[mm].mean():.3f}")

print("\n  误差向量的自相关（按 session 平均，单位：帧）:")
for k in (1, 10, 30, 90, 300, 900, 1800):
    cs = []
    for s, v in sess.items():
        if len(v) <= k + 10:
            continue
        L = np.array([r["L"] for r in v]); P = np.array([r["P"] for r in v])
        S = np.array([r["S"] for r in v]); D = (P - L) * S
        cs.append(np.corrcoef(D[:-k, 1], D[k:, 1])[0, 1])
    print(f"    lag={k:>4} 帧 ({k / 30:>5.1f}s): corr(dy) = {np.nanmean(cs):+.3f}")
