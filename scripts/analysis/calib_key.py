"""索引键对照：局部场该按「原始预测 P」还是「仿射后 p1」归格？

发现（本轮）：head 模式下
  · 按 P 归格 ⇒ 几乎中性（−1%~−6% ✓✓，见 calib_safety.py）
  · 按 p1 归格 ⇒ +85%（场被脆弱仿射的坐标扭曲喂坏 ✗✗，见 calib_blend.py）
而主结果 field_final.py 用的是 p1 归格（密标签下 −25.7% ✓）

⇒ 变体（偏移钳制0.05 常开）：
  V1 键=p1，目标=仿射后残差          （= 现行主结果）
  V2 键=P ，目标=仿射后残差          （稳健键 + 去相关目标 ← 候选）
  V3 键=P ，目标=原始残差，无仿射     （纯局部场的稳健形式）
  V4 = V2 + 收缩混合 K0=15
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


def run(mode, K, variant):
    key_raw = variant in ("V2", "V3", "V4")
    no_affine = variant == "V3"
    K0 = 15 if variant == "V4" else 0
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
            b = None if no_affine else affine_of(glob)
            bv = np.array([np.array([1.0, x, y]) @ b[0], np.array([1.0, x, y]) @ b[1]]) \
                if b else np.zeros(2)
            p1 = P - bv
            # 场查询
            cell_pt = P if key_raw else p1
            c = cidx(cell_pt); i0, j0 = c // G, c % G
            acc = np.zeros(2); cnt = 0.0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ii, jj = i0 + di, j0 + dj
                    if 0 <= ii < G and 0 <= jj < G:
                        cc = grid[ii * G + jj]
                        cnt += cc[0]; acc += cc[1:]
            f_loc = acc / cnt if cnt > 0 else np.zeros(2)
            f_glob = bv if key_raw else np.zeros(2)
            if K0 and cnt > 0:
                w = cnt / (cnt + K0)
                d = w * f_loc + (1 - w) * (bv if key_raw else bv)
            elif K0:
                d = bv if key_raw else np.zeros(2)
            else:
                d = f_loc if (not key_raw) else (f_loc + 0.0)
            base = P if key_raw else p1
            corr = base - np.clip(d, -CL_OFF, CL_OFF) if (cnt >= MIN or K0) else p1
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
                cg = grid[cidx(P if key_raw else p1)]
                cg[0] += 1
                cg[1] += float(P[0] - L[0]) - float(bv[0])
                cg[2] += float(P[1] - L[1]) - float(bv[1])
    return A / n, B / n, B3 / n


print("=" * 118)
print("【索引键对照】V1=键p1(现行) / V2=键P+仿射后目标 / V3=键P无仿射 / V4=V2+收缩K0=15")
print("=" * 118)
print(f"  {'模式':>7}{'K':>6}{'不修正':>8}{'V1键p1':>16}{'V2键P':>16}{'V3键P无仿射':>16}"
      f"{'V4键P+收缩':>16}")
for mode, Ks in (("all", (0,)), ("spread", (200, 20)), ("head", (200, 50))):
    for K in Ks:
        rs = [run(mode, K, v) for v in ("V1", "V2", "V3", "V4")]
        base = rs[0][0]
        line = f"  {mode:>7}{(K if K else 0):>6}{base:>8.3f}"
        for r in rs:
            line += f"{r[1]:>10.3f}({100 * (r[1] / base - 1):>+4.1f}%)"
        print(line)
print()
print("  >3cm 占比（V1 / V2 / V3 / V4）：")
for mode, Ks in (("all", (0,)), ("spread", (200, 20)), ("head", (200, 50))):
    for K in Ks:
        rs = [run(mode, K, v) for v in ("V1", "V2", "V3", "V4")]
        print(f"    {mode:>7}{(K if K else 0):>6}   " +
              "   ".join(f"{r[2] * 100:>5.1f}%" for r in rs))
