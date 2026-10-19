"""覆盖度门控：让仿射"只在标定确实铺开时"生效。

分工（本轮测清）：
  · 基础层 V3：按原始预测归格 + 不用仿射 + 偏移钳制 ⇒ 任何 regime 都不伤害 ✓✓
  · 增益层 V1：加全局仿射（按 p1 归格）⇒ 标签铺开时多赚 5~8 个点 ✓，铺不开时是灾难 ✗✗
⇒ 门控量必须与"注视位置"无关 ✗ 均值漂移会被位置混淆；改用**覆盖度**：
      标定标签的预测位置覆盖到多少个网格（位置无关 ✓）
  coverage = 命中格数 / G²；≥θ ⇒ 启用仿射(V1 约定)，否则退回 V3
变体 θ = 1.0(=永不) / 0.55 / 0.40 / 0.30 / 0.0(=总是)
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/xieliangcai/code/gazeformer-exps")
CSV = str(ROOT / "out/rgbdgaze/analysis/eye224_test.csv")
G, MIN, NZ = 8, 8, 12
CL_OFF = 0.05

sc = {}
with open(ROOT / "data/RGBDGaze/index/index.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        sc[r["rgb_path"]] = (float(r["screen_w"]) * float(r["cm_px_x"]),
                             float(r["screen_h"]) * float(r["cm_px_y"]))

subj = defaultdict(list)
for r in csv.DictReader(open(CSV, encoding="utf-8")):
    subj[r["subject"]].append(dict(fi=int(r["frame_index"]), path=r["path"],
                                   P=np.array([float(r["pred_x"]), float(r["pred_y"])]),
                                   L=np.array([float(r["label_x"]), float(r["label_y"])])))
for s in subj:
    subj[s].sort(key=lambda r: r["fi"])
SUB = sorted(subj)


def cidx(pt):
    return min(G - 1, max(0, int(pt[0] * G))) * G + min(G - 1, max(0, int(pt[1] * G)))


def em(pred, lab, path):
    w, h = sc[path]
    return float(np.hypot((pred[0] - lab[0]) * w, (pred[1] - lab[1]) * h))


def affine_of(glob):
    if glob[0] < MIN:
        return None
    M = np.array([[glob[0], glob[1], glob[2]], [glob[1], glob[3], glob[4]],
                  [glob[2], glob[4], glob[5]]]) + 0.01 * glob[0] * np.diag([0, 1, 1])
    try:
        return (np.linalg.solve(M, glob[6:9]),
                np.linalg.solve(M, np.array([glob[9], glob[10], glob[11]])))
    except np.linalg.LinAlgError:
        return None


def run(mode, K, theta):
    A = B = 0.0; B3 = 0.0; n = 0; use_cnt = 0
    for held in SUB:
        v = subj[held]; N = len(v)
        if N < 300:
            continue
        if mode == "all":
            cal = set(range(N))
        elif mode == "head":
            cal = set(np.arange(min(K, N // 2)).tolist())
        else:
            cal = set(np.unique(np.linspace(0, N - 1, min(K, N)).astype(int)).tolist())
        glob = np.zeros(NZ); grid = defaultdict(lambda: [0.0, 0.0, 0.0])
        cells = set()
        for i, r in enumerate(v):
            P, L = r["P"], r["L"]
            cov = len(cells) / (G * G)
            on = cov >= theta
            use_cnt += 1 if on else 0
            if on:
                b = affine_of(glob)
                if b:
                    x, y = float(P[0]), float(P[1])
                    bv = np.array([np.array([1.0, x, y]) @ b[0], np.array([1.0, x, y]) @ b[1]])
                else:
                    bv = np.zeros(2)
                keypt = P - bv
            else:
                bv = np.zeros(2)
                keypt = P
            c = cidx(keypt); i0, j0 = c // G, c % G
            acc = np.zeros(2); cnt = 0.0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ii, jj = i0 + di, j0 + dj
                    if 0 <= ii < G and 0 <= jj < G:
                        cc = grid[ii * G + jj]
                        cnt += cc[0]; acc += cc[1:]
            corr = keypt - np.clip(acc / cnt, -CL_OFF, CL_OFF) if cnt >= MIN else keypt
            A += em(P, L, r["path"]); B += em(corr, L, r["path"])
            B3 += em(corr, L, r["path"]) > 3; n += 1
            if i in cal:
                x, y = float(P[0]), float(P[1])
                glob[0] += 1
                glob[1] += x; glob[2] += y
                glob[3] += x * x; glob[4] += x * y; glob[5] += y * y
                glob[6] += P[0] - L[0]; glob[7] += (P[0] - L[0]) * x
                glob[8] += (P[0] - L[0]) * y
                glob[9] += P[1] - L[1]; glob[10] += (P[1] - L[1]) * x
                glob[11] += (P[1] - L[1]) * y
                cells.add(cidx(P))
                cg = grid[cidx(keypt)]
                cg[0] += 1
                cg[1] += float(P[0] - L[0]) - float(bv[0])
                cg[2] += float(P[1] - L[1]) - float(bv[1])
    return A / n, B / n, B3 / n


print("=" * 120)
print("【覆盖度门控】仿射仅在 标定覆盖度≥θ 时启用；否则纯 V3（键P、无仿射、钳制0.05）")
print("=" * 120)
print(f"  {'模式':>7}{'K':>6}{'不修正':>8}{'θ=1.0(纯V3)':>17}{'θ=.55':>17}"
      f"{'θ=.40':>17}{'θ=.30':>17}{'θ=0(纯V1)':>17}")
for mode, Ks in (("all", (0,)), ("spread", (200, 20)), ("head", (200, 50))):
    for K in Ks:
        rs = [run(mode, K, t) for t in (1.01, 0.55, 0.40, 0.30, 0.0)]
        base = rs[0][0]
        line = f"  {mode:>7}{(K if K else 0):>6}{base:>8.3f}"
        for r in rs:
            line += f"{r[1]:>10.3f}({100 * (r[1] / base - 1):>+4.1f}%)"
        print(line)
print()
print("  >3cm 占比（同样的列序）：")
for mode, Ks in (("all", (0,)), ("spread", (200, 20)), ("head", (200, 50))):
    for K in Ks:
        rs = [run(mode, K, t) for t in (1.01, 0.55, 0.40, 0.30, 0.0)]
        print(f"    {mode:>7}{(K if K else 0):>6}   " +
              "   ".join(f"{r[2] * 100:>5.1f}%" for r in rs))
