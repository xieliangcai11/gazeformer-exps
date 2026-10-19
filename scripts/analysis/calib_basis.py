"""真值效率实验：误差场的「人群形状基 + 少量系数」 vs 现有「每用户自估每格场」。

文献方向：Meta Prompt(AAAI24)/Alfa(AAAI26)/TTAGaze/ELF-UA 都是"从人群学先验 + 极少标签适配"。
我们此前只否证了"整张场直接迁移"✗ 与"跨被试尺度"✗，未测过"低维基 + 少系数" ✓。

方法 C 的模型（每个基向量是一条完整二维误差场）：
    Δx(p) ≈ 仿射_x(p) + Σ_j c_j · V_j[cell(p)]
    Δy(p) ≈ 仿射_y(p) + Σ_j c_j · V_j[G² + cell(p)]
  其中 V 由**其他被试**的场做 SVD 得到（k 维），仿射与 c 由**该用户 K 个标签**一次性最小二乘解出。

对照（同预算 K）：A 不修正 / B 现有每格场（key=预测，3x3 池化 + 全局仿射，仅用这 K 个标签更新）
采样模式：head（该用户前 K 帧，可部署）/ spread（交错 K 帧，最好情况）
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/xieliangcai/code/gazeformer-exps")
CSV = str(ROOT / "out/rgbdgaze/analysis/eye224_test.csv")
G, MIN, NZ = 8, 8, 12
D2 = G * G

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


def field_of(v):
    acc = np.zeros((D2, 2)); cnt = np.zeros(D2)
    for r in v:
        c = cidx(r["P"]); d = r["P"] - r["L"]
        acc[c] += d; cnt[c] += 1
    f = np.zeros((D2, 2))
    m = cnt > 0
    f[m] = acc[m] / cnt[m][:, None]
    return np.concatenate([f[:, 0], f[:, 1]])          # (2G²,)


def em(pred, lab, path):
    w, h = sc[path]
    return float(np.hypot((pred[0] - lab[0]) * w, (pred[1] - lab[1]) * h))


F = np.stack([field_of(subj[s]) for s in SUB])
mu = F.mean(0)
_, S, Vt = np.linalg.svd(F - mu, full_matrices=False)
print("=" * 108)
print(f"【真值效率】基由其他 {len(SUB)} 个被试的场学得；每个测试用户只用 K 个标签")
print(f"  场奇异值前 4 = {np.round(S[:4], 4)}；前 2 个分量占总能量 "
      f"{np.round((S[:2] ** 2).sum() / (S ** 2).sum() * 100, 1)}%")
print("=" * 108)


def run(k_dim, mode, K):
    V = Vt[:k_dim]                                     # (k, 2G²)
    A0 = B0 = C0 = 0.0
    B3 = C3 = 0.0
    n = 0
    for held in SUB:
        v = subj[held]
        N = len(v)
        if N < 300:
            continue
        cal = (np.arange(N) if mode == "spread" else np.arange(N))[:0]  # 占位
        if mode == "head":
            cal = np.arange(min(K, N // 2))
        else:
            cal = np.unique(np.linspace(0, N - 1, min(K, N)).astype(int))
        ev = np.setdiff1d(np.arange(N), cal)
        Pc = np.array([v[i]["P"] for i in cal]); Lc = np.array([v[i]["L"] for i in cal])
        Dc = Pc - Lc
        # ---- 方法 C：一次性解 仿射(12) + 基系数(k) ----
        X = np.zeros((2 * len(cal), 12 + k_dim))
        y = np.concatenate([Dc[:, 0], Dc[:, 1]])
        for t, i in enumerate(cal):
            p = v[i]["P"]; c = cidx(p)
            X[t, 0:3] = [p[0], p[1], 1.0]                 # 仿射_x
            X[t, 3:6] = 0.0
            X[t, 6:9] = 0.0                               # 仿射_y 不用此行
            X[t, 9:12] = 0.0
            X[t, 12:] = V[:, c]                           # 基对 x 的贡献
            X[len(cal) + t, 0:3] = 0.0
            X[len(cal) + t, 6:9] = [p[0], p[1], 1.0]      # 仿射_y
            X[len(cal) + t, 12:] = V[:, D2 + c]           # 基对 y 的贡献
        lam = 1e-2 * max(len(cal), 1)
        try:
            coef = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
        except np.linalg.LinAlgError:
            coef = None
        # ---- 方法 B：仅用 cal 标签在线更新每格场 + 全局仿射 ----
        glob = np.zeros(NZ); grid = defaultdict(lambda: [0.0, 0.0, 0.0])
        calset = set(cal.tolist())
        for i, r in enumerate(v):
            P, L = r["P"], r["L"]
            x, y_ = float(P[0]), float(P[1])
            b = None
            if glob[0] >= 4:
                M = np.array([[glob[0], glob[1], glob[2]], [glob[1], glob[3], glob[4]],
                              [glob[2], glob[4], glob[5]]]) + 0.01 * glob[0] * np.diag([0, 1, 1])
                try:
                    b = (np.linalg.solve(M, glob[6:9]),
                         np.linalg.solve(M, np.array([glob[9], glob[10], glob[11]])))
                except np.linalg.LinAlgError:
                    b = None
            bv = np.array([np.array([1.0, x, y_]) @ b[0], np.array([1.0, x, y_]) @ b[1]]) \
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
            corrB = p1 - (acc / cnt if cnt >= MIN else 0.0)
            if coef is not None:
                xx = np.concatenate([[p[0], p[1], 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     V[:, cidx(P)]])
                yy = np.concatenate([[0, 0, 0, 0, 0, 0, p[0], p[1], 1.0, 0, 0, 0],
                                     V[:, D2 + cidx(P)]])
                corrC = P - np.array([xx @ coef, yy @ coef])
            else:
                corrC = P
            if i in ev:
                A0 += em(P, L, r["path"]); B0 += em(corrB, L, r["path"])
                C0 += em(corrC, L, r["path"])
                B3 += em(corrB, L, r["path"]) > 3
                C3 += em(corrC, L, r["path"]) > 3
                n += 1
            if i in calset:
                glob[0] += 1
                glob[1] += x; glob[2] += y_
                glob[3] += x * x; glob[4] += x * y_; glob[5] += y_ * y_
                glob[6] += P[0] - L[0]; glob[7] += (P[0] - L[0]) * x
                glob[8] += (P[0] - L[0]) * y_
                glob[9] += P[1] - L[1]; glob[10] += (P[1] - L[1]) * x
                glob[11] += (P[1] - L[1]) * y_
                cg = grid[cidx(p1)]
                cg[0] += 1
                cg[1] += float(P[0] - L[0]) - float(bv[0])
                cg[2] += float(P[1] - L[1]) - float(bv[1])
    return (A0 / n, B0 / n, C0 / n, B3 / n, C3 / n, n) if n else None


print(f"  {'模式':>7}{'K':>6}{'A不修正':>9}{'B每格场':>9}{'C基+系数':>9}"
      f"{'B>3cm':>8}{'C>3cm':>8}{'C vs B':>9}")
for mode in ("head", "spread"):
    for K in (5, 10, 20, 50, 100, 200):
        r = run(2, mode, K)
        if r:
            print(f"  {mode:>7}{K:>6}{r[0]:>9.3f}{r[1]:>9.3f}{r[2]:>9.3f}"
                  f"{r[3] * 100:>7.1f}%{r[4] * 100:>7.1f}%{100 * (r[2] / r[1] - 1):>8.1f}%")
print()
print(f"  {'基维k':>6}{'模式':>7}{'K':>6}{'A':>9}{'B':>9}{'C':>9}{'C vs B':>9}")
for mode in ("head", "spread"):
    for k in (1, 2, 4, 8):
        r = run(k, mode, 20)
        if r:
            print(f"  {k:>6}{mode:>7}{20:>6}{r[0]:>9.3f}{r[1]:>9.3f}{r[2]:>9.3f}"
                  f"{100 * (r[2] / r[1] - 1):>8.1f}%")
