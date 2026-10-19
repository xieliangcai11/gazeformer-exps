"""标定安全层 v2：钳制"仿射斜率"（拉伸倍数）+ 钳制偏移，检验
   「标定永不伤害」且「不损失最好成绩（每帧都有标签时 −25.7%）」。

依据（上一步测出）：
  · 用"开头 K 帧"标定时，真正的灾难来自**全局仿射**（被 regime 错位污染 ⇒ 过度拉伸）✗
  · 每格场本身稳健 ✓；钳制修正幅度可白拿改进 ✓

变体：无钳制 / 钳偏移 / 钳斜率(±s) / 两者
模式：head（前 K 帧，可部署）/ spread（交错 K 帧，最好情况）/ all（每帧都有标签 = 本数据集主结果）
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/xieliangcai/code/gazeformer-exps")
CSV = str(ROOT / "out/rgbdgaze/analysis/eye224_test.csv")
G, MIN, NZ = 8, 8, 12

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


def clamp_slope(b1, b2, s):
    """把仿射斜率限制在 ±s（b=[β0, β1, β2]，β1/β2 是斜率）。"""
    return (np.array([b1[0], np.clip(b1[1], -s, s), np.clip(b1[2], -s, s)]),
            np.array([b2[0], np.clip(b2[1], -s, s), np.clip(b2[2], -s, s)]))


def run(mode, K, cl_off=np.inf, cl_slope=np.inf):
    A = B = 0.0; B3 = 0.0; n = 0
    for held in SUB:
        v = subj[held]
        N = len(v)
        if N < 300:
            continue
        if mode == "all":
            cal = np.arange(N)
        elif mode == "head":
            cal = np.arange(min(K, N // 2))
        else:
            cal = np.unique(np.linspace(0, N - 1, min(K, N)).astype(int))
        ev = np.setdiff1d(np.arange(N), cal)
        if len(ev) == 0:
            continue
        calset = set(cal.tolist())
        glob = np.zeros(NZ); grid = defaultdict(lambda: [0.0, 0.0, 0.0])
        # 先把标定窗口的统计量灌进去（一次性标定；all 模式等价于逐帧在线 ✓）
        for i in cal:
            r = v[i]
            x, y = float(r["P"][0]), float(r["P"][1])
            glob[0] += 1
            glob[1] += x; glob[2] += y
            glob[3] += x * x; glob[4] += x * y; glob[5] += y * y
            glob[6] += r["P"][0] - r["L"][0]; glob[7] += (r["P"][0] - r["L"][0]) * x
            glob[8] += (r["P"][0] - r["L"][0]) * y
            glob[9] += r["P"][1] - r["L"][1]; glob[10] += (r["P"][1] - r["L"][1]) * x
            glob[11] += (r["P"][1] - r["L"][1]) * y
        # 仿射（含斜率钳制）
        b = None
        if glob[0] >= MIN:
            M = np.array([[glob[0], glob[1], glob[2]], [glob[1], glob[3], glob[4]],
                          [glob[2], glob[4], glob[5]]]) + 0.01 * glob[0] * np.diag([0, 1, 1])
            try:
                b = (np.linalg.solve(M, glob[6:9]),
                     np.linalg.solve(M, np.array([glob[9], glob[10], glob[11]])))
                if np.isfinite(cl_slope):
                    b = clamp_slope(b[0], b[1], cl_slope)
            except np.linalg.LinAlgError:
                b = None
        # 每格残差（用带钳制仿射后的位置归格）
        for r in [v[i] for i in cal]:
            P, L = r["P"], r["L"]
            x, y = float(P[0]), float(P[1])
            bv = np.array([np.array([1.0, x, y]) @ b[0], np.array([1.0, x, y]) @ b[1]]) \
                if b is not None else np.zeros(2)
            p1 = P - bv
            cg = grid[cidx(p1)]
            cg[0] += 1
            cg[1] += float(P[0] - L[0]) - float(bv[0])
            cg[2] += float(P[1] - L[1]) - float(bv[1])
        # 评估
        for i in ev:
            r = v[i]
            P, L = r["P"], r["L"]
            x, y = float(P[0]), float(P[1])
            bv = np.array([np.array([1.0, x, y]) @ b[0], np.array([1.0, x, y]) @ b[1]]) \
                if b is not None else np.zeros(2)
            p1 = P - bv
            c = cidx(p1); i0, j0 = c // G, c % G
            acc = np.zeros(2); cnt = 0.0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    ii, jj = i0 + di, j0 + dj
                    if 0 <= ii < G and 0 <= jj < G:
                        cc = grid[ii * G + jj]
                        cnt += cc[0]; acc += cc[1:]
            corr = p1
            if cnt >= MIN:
                d = acc / cnt
                d = np.clip(d, -cl_off, cl_off)
                corr = p1 - d
            A += em(P, L, r["path"]); B += em(corr, L, r["path"])
            B3 += em(corr, L, r["path"]) > 3; n += 1
    return A / n, B / n, B3 / n


print("=" * 110)
print("【标定安全层 v2】斜率钳制 / 偏移钳制（数值=评估帧 EM 均值，括号=相对不修正）")
print("=" * 110)
print(f"  {'模式':>7}{'K':>6}{'A':>8}{'裸标定':>16}{'钳偏移.05':>16}"
      f"{'钳斜率.20':>16}{'两者':>16}")
for mode, Ks in (("head", (20, 50, 200)), ("spread", (20, 50, 200)), ("all", (0,))):
    for K in Ks:
        r0 = run(mode, K)
        r1 = run(mode, K, cl_off=0.05)
        r2 = run(mode, K, cl_slope=0.20)
        r3 = run(mode, K, cl_off=0.05, cl_slope=0.20)
        tag = f"{mode:>7}{K if K else -1:>6}{r0[0]:>8.3f}"
        for r in (r0, r1, r2, r3):
            tag += f"{r[1]:>9.3f}({100 * (r[1] / r0[0] - 1):>+4.1f}%)"
        print("  " + tag)
print()
print("  参考：主结果（每帧标签、无钳制）= 1.093（−25.7%）；>3cm 3.56%")
