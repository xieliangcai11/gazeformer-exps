"""统一估计器：全局仿射 × 局部每格场 的「收缩混合」(partial pooling)。

矛盾（本轮测清）：
  · 全局仿射 = 低方差/高偏差：标定有代表性时强（spread −15.5%）✗ 但 regime 漂移时灾难（head +51.7%）
  · 局部每格场 = 低偏差/高方差：任何 regime 都不伤害 ✓ 但稀疏标签下欠拟合
  · 开关式门控 ✗（"近30帧均值"被注视位置混淆 ⇒ 误杀 spread）

⇒ 正确做法：按每格样本量 n 做收缩
      corr = p1 − [ w·field_cell + (1−w)·affine(p1) ],   w = n_cell / (n_cell + K0)
  标签少 ⇒ 仿射主导（稳）；标签多 ⇒ 局部场主导（准）
变体：K0=∞(纯仿射) / K0=40 / 15 / 5 / 0(纯局部)
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
        return np.linalg.solve(M, glob[6:9]), np.linalg.solve(M, np.array([glob[9], glob[10], glob[11]]))
    except np.linalg.LinAlgError:
        return None


def run(mode, K, K0):
    """K0: 收缩尺度（∞=纯仿射，0=纯局部）"""
    A = B = 0.0; B3 = 0.0; n = 0
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
        for i, r in enumerate(v):
            P, L = r["P"], r["L"]
            x, y = float(P[0]), float(P[1])
            b = affine_of(glob)
            if b:
                bv = np.array([np.array([1.0, x, y]) @ b[0], np.array([1.0, x, y]) @ b[1]])
            else:
                bv = np.zeros(2)
            p1 = P - bv
            # 局部场（3x3 池化）
            c = cidx(p1); i0, j0 = c // G, c % G
            acc = np.zeros(2); cnt = 0.0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ii, jj = i0 + di, j0 + dj
                    if 0 <= ii < G and 0 <= jj < G:
                        cc = grid[ii * G + jj]
                        cnt += cc[0]; acc += cc[1:]
            f_loc = acc / cnt if cnt > 0 else np.zeros(2)
            f_glob = bv
            if K0 == 0:
                d = f_loc
            elif np.isinf(K0):
                d = f_glob
            else:
                w = cnt / (cnt + K0)
                d = w * f_loc + (1 - w) * f_glob
            corr = p1 - np.clip(d, -CL_OFF, CL_OFF)
            A += em(P, L, r["path"]); B += em(corr, L, r["path"])
            B3 += em(corr, L, r["path"]) > 3; n += 1
            if i in cal:
                glob[0] += 1
                glob[1] += x; glob[2] += y
                glob[3] += x * x; glob[4] += x * y; glob[5] += y * y
                glob[6] += P[0] - L[0]; glob[7] += (P[0] - L[0]) * x
                glob[8] += (P[0] - L[0]) * y
                glob[9] += P[1] - L[1]; glob[10] += (P[1] - L[1]) * x
                glob[11] += (P[1] - L[1]) * y
                cg = grid[cidx(p1)]
                cg[0] += 1
                cg[1] += float(P[0] - L[0]) - float(bv[0])
                cg[2] += float(P[1] - L[1]) - float(bv[1])
    return A / n, B / n, B3 / n


print("=" * 116)
print("【收缩混合】corr = p1 − [w·局部场 + (1−w)·全局仿射], w = n/(n+K0)   (偏移钳制0.05 常开)")
print("=" * 116)
print(f"  {'模式':>7}{'K':>6}{'不修正':>8}{'纯仿射':>15}{'K0=40':>15}{'K0=15':>15}"
      f"{'K0=5':>15}{'纯局部':>15}")
for mode, Ks in (("all", (0,)), ("spread", (200, 50, 20)), ("head", (200, 50, 20))):
    for K in Ks:
        rs = [run(mode, K, k0) for k0 in (np.inf, 40, 15, 5, 0)]
        base = rs[0][0]
        line = f"  {mode:>7}{(K if K else 0):>6}{base:>8.3f}"
        for r in rs:
            line += f"{r[1]:>10.3f}({100 * (r[1] / base - 1):>+4.1f}%)"
        print(line)
print()
print("  每行对应的 >3cm 占比（纯仿射 / K0=15 / 纯局部）：")
for mode, Ks in (("all", (0,)), ("spread", (200, 50, 20)), ("head", (200, 50, 20))):
    for K in Ks:
        rs = [run(mode, K, k0) for k0 in (np.inf, 15, 0)]
        print(f"    {mode:>7}{(K if K else 0):>6}   " +
              "   ".join(f"{r[2] * 100:>5.1f}%" for r in rs))
