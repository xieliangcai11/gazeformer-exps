"""同模型同划分下，四条标定线的正面对照（此前各测各的 ✗ 不可比）。

数据：out/rgbdgaze/analysis/feats_eye224_test.npz（旗舰 eye224_mgpu，test 11031 帧，
      基线 EM=1.472，特征 1536 维 = [融合CLS ⊕ 骨干CLS ⊕ patch均值 ⊕ 双眼]）

方法（全部只用该被试的 K 个标签，其余帧评估）：
  A 不标定
  B 输出空间误差场：按原始预测归格 + 3x3 池化 + 偏移钳制 0.05（本轮确立的最稳形式）
  C1 特征空间：岭回归(features -> 注视点)        （= calibrate_final.py 的形式，改自实现）
  C2 特征空间：岭回归([features, pred] -> 残差) + pred（以模型预测为起点，信息更多）
  D  级联：先 C2，再在 C2 残差上套 B 的误差场

岭回归：纯 numpy，样本数 n≤600 << 维数 1536 ⇒ 用 n×n Gram 的对偶形式；
        α 由留一误差闭式（H 对角）在 17 个候选里自动选，无 sklearn 依赖。
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/xieliangcai/code/gazeformer-exps")
NPZ = ROOT / "out/rgbdgaze/analysis/feats_eye224_test.npz"
G, MIN, CL_OFF = 8, 8, 0.05
ALPHAS = np.logspace(-4, 6, 21)

sc = {}
with open(ROOT / "data/RGBDGaze/index/index.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        sc[r["rgb_path"]] = (float(r["screen_w"]) * float(r["cm_px_x"]),
                             float(r["screen_h"]) * float(r["cm_px_y"]))

d = np.load(NPZ, allow_pickle=True)
F = d["feats"].astype(np.float64)
P = d["pred"].astype(np.float64)
L = d["label"].astype(np.float64)
paths = d["path"]
subj = d["subject"]
fi = d["frame_index"]

by_subj = defaultdict(list)
for i in range(len(P)):
    by_subj[str(subj[i])].append(i)
for s in by_subj:
    by_subj[s].sort(key=lambda i: fi[i])
SUB = sorted(by_subj)
print(f"数据：{len(P)} 帧，被试 {SUB}，各 {[len(by_subj[s]) for s in SUB]} 帧")


def cidx(pt):
    return min(G - 1, max(0, int(pt[0] * G))) * G + min(G - 1, max(0, int(pt[1] * G)))


def em(pred, lab, path):
    w, h = sc[path]
    return float(np.hypot((pred[0] - lab[0]) * w, (pred[1] - lab[1]) * h))


def field_apply(grid, key_pt, base):
    c = cidx(key_pt); i0, j0 = c // G, c % G
    acc = np.zeros(2); cnt = 0.0
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            ii, jj = i0 + di, j0 + dj
            if 0 <= ii < G and 0 <= jj < G:
                cc = grid[ii * G + jj]
                cnt += cc[0]; acc += cc[1:]
    if cnt < MIN:
        return base
    return base - np.clip(acc / cnt, -CL_OFF, CL_OFF)


class RidgeLOO:
    """对偶形式岭回归；α 由留一误差（H 对角线闭式）在候选集里选。"""

    def __init__(self, alphas=ALPHAS):
        self.alphas = alphas

    def fit(self, X, Y):
        self.mx_ = X.mean(0); self.sx_ = X.std(0) + 1e-6
        Xs = (X - self.mx_) / self.sx_
        self.my_ = Y.mean(0)
        Yc = Y - self.my_
        K = Xs @ Xs.T                                   # n x n
        w, V = np.linalg.eigh(K)                         # K = V diag(w) V^T
        self.Xs_, self.V_, self.w_ = Xs, V, w
        # 留一误差：H = V diag(w/(w+a)) V^T，diag(H)_i = sum_j V_ij^2 * w_j/(w_j+a)
        V2 = V ** 2
        best, best_a = np.inf, None
        for a in self.alphas:
            h = V2 @ (w / (w + a))
            num = V @ ((V.T @ Yc) / (w + a)[:, None])      # 拟合值（对偶）
            r = (Yc - num) / np.clip(1.0 - h, 1e-6, None)[:, None]
            mse = float((r ** 2).mean())
            if mse < best:
                best, best_a = mse, a
        self.alpha_ = best_a
        self.coef_ = V @ ((V.T @ Yc) / (w + best_a)[:, None])   # 对偶系数 (n,2)
        return self

    def predict(self, X):
        Xs = (X - self.mx_) / self.sx_
        return self.my_ + (Xs @ self.Xs_.T) @ self.coef_


def run(mode, K):
    keys = ("A", "B", "C1", "C2", "D")
    acc = {k: [0.0, 0] for k in keys}
    thr = {k: [0, 0, 0] for k in keys}      # >2 >3 >5
    alphas_used = []
    for s in SUB:
        idx = by_subj[s]
        N = len(idx)
        if N < 400:
            continue
        if mode == "head":
            cal = idx[:K]
        else:
            cal = [idx[t] for t in np.unique(np.linspace(0, N - 1, K).astype(int))]
        calset = set(cal)
        ev = [i for i in idx if i not in calset]
        Xc = F[cal]
        # C1：特征 -> 注视点
        m1 = RidgeLOO().fit(Xc, L[cal])
        gazeC1 = m1.predict(F)
        alphas_used.append(m1.alpha_)
        # C2：特征(含预测) -> 残差
        Pb = np.linalg.norm(P - 0.5, axis=1, keepdims=True)
        X2 = np.concatenate([F, P, Pb], 1)
        m2 = RidgeLOO().fit(X2[cal], L[cal] - P[cal])
        gazeC2 = P + m2.predict(X2)
        # 误差场（B：原始预测；D：C2 预测）
        gB = defaultdict(lambda: [0.0, 0.0, 0.0])
        gD = defaultdict(lambda: [0.0, 0.0, 0.0])
        for i in cal:
            cB = gB[cidx(P[i])]
            cB[0] += 1; cB[1] += P[i][0] - L[i][0]; cB[2] += P[i][1] - L[i][1]
            cD = gD[cidx(gazeC2[i])]
            cD[0] += 1; cD[1] += gazeC2[i][0] - L[i][0]; cD[2] += gazeC2[i][1] - L[i][1]
        for i in ev:
            out = {"A": P[i], "B": field_apply(gB, P[i], P[i]), "C1": gazeC1[i],
                   "C2": gazeC2[i], "D": field_apply(gD, gazeC2[i], gazeC2[i])}
            for k, v in out.items():
                e = em(v, L[i], str(paths[i]))
                acc[k][0] += e; acc[k][1] += 1
                thr[k][0] += e > 2; thr[k][1] += e > 3; thr[k][2] += e > 5
    res = {k: (acc[k][0] / acc[k][1], [t / acc[k][1] for t in thr[k]]) for k in keys}
    return res, alphas_used


print()
print("=" * 110)
print("【四条标定线正面对照】同一模型/划分/标签预算（EM 均值 cm，括号=相对不标定）")
print("=" * 110)
print(f"  {'协议':>7}{'K':>6}{'A不标定':>9}{'B输出场':>17}{'C1特征→注视':>17}"
      f"{'C2特征→残差':>17}{'D级联(C2∘B)':>17}")
for mode in ("head", "spread"):
    for K in (100, 200, 600):
        r, al = run(mode, K)
        base = r["A"][0]
        line = f"  {mode:>7}{K:>6}{base:>9.3f}"
        for k in ("B", "C1", "C2", "D"):
            line += f"{r[k][0]:>10.3f}({100 * (r[k][0] / base - 1):>+4.1f}%)"
        print(line)
print()
print("  >3cm 占比（A / B / C1 / C2 / D）：")
for mode in ("head", "spread"):
    for K in (100, 200, 600):
        r, _ = run(mode, K)
        print(f"    {mode:>7}{K:>6}   " +
              "   ".join(f"{r[k][1][1] * 100:>5.2f}%" for k in ("A", "B", "C1", "C2", "D")))
