"""可部署配方：① 因果在线 + 钳制（验证钳制不损失主结果）
              ② 一次性标定：仅每格场 / +仿射 / +仿射&漂移门控

已确立的分工：
  · 每格场（局部）：抗 regime 漂移 ✓ ⇒ 稀疏/前段标定下也安全
  · 全局仿射：标定是否"有代表性"时才对 ⇒ 无代表性时是主要伤害源 ✗（head 模式 +26~+78%）
⇒ 用「预测位置漂移」做门控（零额外标注）：当前流与标定窗口的预测均值差异过大 ⇒ 关掉仿射，只留场
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/xieliangcai/code/gazeformer-exps")
CSV = str(ROOT / "out/rgbdgaze/analysis/eye224_test.csv")
G, MIN, NZ = 8, 8, 12
CL_OFF, CL_SLOPE = 0.05, 0.20

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


def solve_affine(glob, slope_clip=np.inf):
    if glob[0] < MIN:
        return None
    M = np.array([[glob[0], glob[1], glob[2]], [glob[1], glob[3], glob[4]],
                  [glob[2], glob[4], glob[5]]]) + 0.01 * glob[0] * np.diag([0, 1, 1])
    try:
        b1 = np.linalg.solve(M, glob[6:9])
        b2 = np.linalg.solve(M, np.array([glob[9], glob[10], glob[11]]))
    except np.linalg.LinAlgError:
        return None
    if np.isfinite(slope_clip):
        b1 = np.array([b1[0], np.clip(b1[1], -slope_clip, slope_clip),
                       np.clip(b1[2], -slope_clip, slope_clip)])
        b2 = np.array([b2[0], np.clip(b2[1], -slope_clip, slope_clip),
                       np.clip(b2[2], -slope_clip, slope_clip)])
    return b1, b2


def field_corr(grid, p1, clip):
    c = cidx(p1); i0, j0 = c // G, c % G
    acc = np.zeros(2); cnt = 0.0
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            ii, jj = i0 + di, j0 + dj
            if 0 <= ii < G and 0 <= jj < G:
                cc = grid[ii * G + jj]
                cnt += cc[0]; acc += cc[1:]
    if cnt < MIN:
        return p1
    return p1 - np.clip(acc / cnt, -clip, clip)


# ---------------- ① 因果在线 + 钳制 ----------------
def causal(K, clip):
    A = B = 0.0; B3 = 0.0; n = 0
    for held in SUB:
        v = subj[held]; N = len(v)
        if N < 300:
            continue
        cal = (set(range(N)) if K == 0
               else set(np.unique(np.linspace(0, N - 1, min(K, N)).astype(int)).tolist()))
        glob = np.zeros(NZ); grid = defaultdict(lambda: [0.0, 0.0, 0.0])
        for i, r in enumerate(v):
            P, L = r["P"], r["L"]
            x, y = float(P[0]), float(P[1])
            b = solve_affine(glob, CL_SLOPE)
            bv = np.array([np.array([1.0, x, y]) @ b[0], np.array([1.0, x, y]) @ b[1]]) \
                if b else np.zeros(2)
            p1 = P - bv
            corr = field_corr(grid, p1, clip)
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


print("=" * 112)
print("① 因果在线（主结果所在形式）：钳制是否损失收益？  [K=0 表示每帧都有标签=主结果]")
print("=" * 112)
print(f"  {'K':>7}{'不修正':>10}{'无钳制':>17}{'钳偏移':>17}{'钳斜率':>17}{'两者':>17}"
      f"{'无钳制>3cm':>12}{'两者>3cm':>12}")
for K in (0, 200, 50):
    r0 = causal(K, np.inf)
    r1 = causal(K, CL_OFF)
    r2 = causal(K, CL_OFF * 2)
    r3 = causal(K, CL_OFF * 4)
    print(f"  {K:>7}{r0[0]:>10.3f}{r0[1]:>10.3f}({100 * (r0[1] / r0[0] - 1):>+4.1f}%)"
          f"{r1[1]:>10.3f}({100 * (r1[1] / r0[0] - 1):>+4.1f}%)"
          f"{r2[1]:>10.3f}({100 * (r2[1] / r0[0] - 1):>+4.1f}%)"
          f"{r3[1]:>10.3f}({100 * (r3[1] / r0[0] - 1):>+4.1f}%)"
          f"{r0[2] * 100:>11.1f}%{r3[2] * 100:>11.1f}%")


# ---------------- ② 一次性标定 + 漂移门控 ----------------
def oneshot(mode, K, use_affine, theta=np.inf):
    """theta: 预测位置漂移阈值；超过则关掉仿射（仍用每格场）。"""
    A = B = 0.0; B3 = 0.0; n = 0
    for held in SUB:
        v = subj[held]; N = len(v)
        if N < 300:
            continue
        cal = (np.arange(min(K, N // 2)) if mode == "head"
               else np.unique(np.linspace(0, N - 1, min(K, N)).astype(int)))
        ev = np.setdiff1d(np.arange(N), cal)
        calr = [v[i] for i in cal]
        glob = np.zeros(NZ); grid = defaultdict(lambda: [0.0, 0.0, 0.0])
        for r in calr:
            P, L = r["P"], r["L"]
            x, y = float(P[0]), float(P[1])
            glob[0] += 1
            glob[1] += x; glob[2] += y
            glob[3] += x * x; glob[4] += x * y; glob[5] += y * y
            glob[6] += P[0] - L[0]; glob[7] += (P[0] - L[0]) * x
            glob[8] += (P[0] - L[0]) * y
            glob[9] += P[1] - L[1]; glob[10] += (P[1] - L[1]) * x
            glob[11] += (P[1] - L[1]) * y
        b = solve_affine(glob, CL_SLOPE) if use_affine else None
        for r in calr:
            P, L = r["P"], r["L"]
            x, y = float(P[0]), float(P[1])
            bv = np.array([np.array([1.0, x, y]) @ b[0], np.array([1.0, x, y]) @ b[1]]) \
                if b else np.zeros(2)
            p1 = P - bv
            cg = grid[cidx(p1)]
            cg[0] += 1
            cg[1] += float(P[0] - L[0]) - float(bv[0])
            cg[2] += float(P[1] - L[1]) - float(bv[1])
        cal_mean = np.array([np.mean([r["P"][0] for r in calr]),
                             np.mean([r["P"][1] for r in calr])])
        hist = []
        for i in ev:
            r = v[i]
            P, L = r["P"], r["L"]
            x, y = float(P[0]), float(P[1])
            drift = np.linalg.norm(np.mean(hist[-30:], axis=0) - cal_mean) if len(hist) >= 30 else 0.0
            bb = b if drift <= theta else None
            bv = np.array([np.array([1.0, x, y]) @ bb[0], np.array([1.0, x, y]) @ bb[1]]) \
                if bb else np.zeros(2)
            p1 = P - bv
            corr = field_corr(grid, p1, CL_OFF)
            A += em(P, L, r["path"]); B += em(corr, L, r["path"])
            B3 += em(corr, L, r["path"]) > 3; n += 1
            hist.append(P)
    return A / n, B / n, B3 / n


print()
print("=" * 112)
print("② 一次性标定（钳偏移0.05 常开）：仿射是否该开？漂移门控(θ)能否保住仿射的收益？")
print("=" * 112)
print(f"  {'模式':>7}{'K':>6}{'仅场':>17}{'场+仿射':>17}{'θ=0.04':>17}{'θ=0.08':>17}{'θ=0.15':>17}")
for mode in ("head", "spread"):
    for K in (20, 50, 200):
        r0 = oneshot(mode, K, False)
        r1 = oneshot(mode, K, True)
        g = [oneshot(mode, K, True, t) for t in (0.04, 0.08, 0.15)]
        base = r1[0]
        line = f"  {mode:>7}{K:>6}{r0[1]:>10.3f}({100 * (r0[1] / r0[0] - 1):>+4.1f}%)"
        line += f"{r1[1]:>10.3f}({100 * (r1[1] / base - 1):>+4.1f}%)"
        for r in g:
            line += f"{r[1]:>10.3f}({100 * (r[1] / base - 1):>+4.1f}%)"
        print(line)
