"""文献口径的标定对照：只用少量真值（K 个标定点），评估期**零真值**。

口径（= 文献同类协议）：每会话只允许用开头一段（frac）里的 K 个带真值帧标定，
评估 = 其后全部帧，严格因果。

对照的方法（都是"标定映射"，闭式解，无训练）：
  A bias    ：只修常量偏置（最经典的 offset calibration）
  B affine  ：每轴 尺度+平移（经典线性标定）
  C poly2   ：每轴 2 阶多项式（经典多项式标定）
  D tps     ：RBF/TPS 平滑修正映射（在**预测位置**上拟合残差，商业眼动仪常用；
              γ 与 α 由标定集内部留一误差选）
  E knn     ：自适应带宽局部线性回归（取最近 k 个标定点做加权局部线性拟合）
  F field   ：我们此前的做法（全局仿射 + 8×8 核加权空间场）
  G c2      ：我们当前最好（投影特征 ⊕ 预测 ⊕ 离线心距离 → 残差 的线性岭）

说明（避免把"用了逐帧真值"的方法混进来）：本脚本**所有**方法的真值只出现在标定集，
评估帧的真值只用于打分。参照的文献口径来源：本仓库 docs/ 内的
RGBDGaze(ICMI'22，免标定基线 1.89/2.26)、MAC-Gaze(2025，少量带真值样本 + 触发式再标定)、
MetaPrompt(AAAI'24，无标签测试时个性化)、Alfa(AAAI'26，少量无标签样本)、
以及经典 offset/decomposition 标定（Chen & Shi, 2020）。
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/xieliangcai/code/gazeformer-exps")
NPZ = ROOT / "out/rgbdgaze/analysis/feats_eye224_test.npz"
CKPT = ROOT / "out/rgbdgaze/ablation/checkpoints/eye224_mgpu/best.pt"
G, MIN_A, MIN_C, CLIP = 8, 3, 8, 0.05
ALPHAS = np.logspace(-6, 3, 19)

sc = {}
with open(ROOT / "data/RGBDGaze/index/index.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        sc[r["rgb_path"]] = (float(r["screen_w"]) * float(r["cm_px_x"]),
                             float(r["screen_h"]) * float(r["cm_px_y"]))

d = np.load(NPZ, allow_pickle=True)
F, P, L = d["feats"].astype(np.float64), d["pred"].astype(np.float64), d["label"].astype(np.float64)
paths, fi = d["path"], d["frame_index"]
nseg = F.shape[1] // 4
st = torch.load(CKPT, map_location="cpu", weights_only=False)


def _get(n):
    for k in (n, f"module.{n}", f"model.{n}"):
        if k in st:
            return st[k].float().numpy()
    raise KeyError(n)


Wc, bc = _get("proj_rgb_cls.weight"), _get("proj_rgb_cls.bias")
Wp, bp = _get("proj_rgb_patch.weight"), _get("proj_rgb_patch.bias")
XFEAT = np.concatenate([F[:, nseg:2 * nseg] @ Wc.T + bc,
                        F[:, 2 * nseg:3 * nseg] @ Wp.T + bp, P,
                        np.linalg.norm(P - 0.5, axis=1, keepdims=True)], 1)

sess = defaultdict(list)
for i in range(len(P)):
    sess[str(paths[i]).rsplit("/", 1)[0]].append(i)
for s in sess:
    sess[s].sort(key=lambda i: fi[i])
SESS = sorted(sess)


def emv(pred, lab, path):
    w, h = sc[path]
    return float(np.hypot((pred[0] - lab[0]) * w, (pred[1] - lab[1]) * h))


class KRR:
    """核岭（线性/RBF），α 由留一误差闭式选（n=K 很小 ⇒ 便宜）。"""

    def __init__(self, kind="linear", gamma=1.0):
        self.kind, self.gamma = kind, gamma

    def _k(self, A, B):
        if self.kind == "linear":
            return A @ B.T
        d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
        return np.exp(-self.gamma * np.clip(d2, 0, None))

    def fit(self, X, Y):
        self.mx_, self.sx_ = X.mean(0), X.std(0) + 1e-6
        self.my_ = Y.mean(0)
        Xs = (X - self.mx_) / self.sx_
        self.Xs_ = Xs
        Yc = Y - self.my_
        K = self._k(Xs, Xs)
        w, V = np.linalg.eigh(K)
        V2 = V ** 2
        best, ba = np.inf, float(ALPHAS[0])
        for a in ALPHAS:
            h = V2 @ (w / (w + a))
            num = V @ ((V.T @ Yc) / (w + a)[:, None])
            rr = (Yc - num) / np.clip(1.0 - h, 1e-6, None)[:, None]
            m = float((rr ** 2).mean())
            if m < best:
                best, ba = m, a
        self.coef_ = V @ ((V.T @ Yc) / (w + ba)[:, None])
        return self

    def predict(self, X):
        return self.my_ + self._k((X - self.mx_) / self.sx_, self.Xs_) @ self.coef_


def fit_tps(cal, y):
    """在预测位置上拟合残差的 RBF(+线性) 映射；γ、α 由标定集留一误差选。"""
    best = None
    for gm in (2.0, 5.0, 10.0, 20.0, 50.0):
        for kind in ("rbf", "linear"):
            m = KRR(kind, gm).fit(P[cal], y)
            err = float(((y - m.predict(P[cal])) ** 2).mean())
            if best is None or err < best[0]:
                best = (err, m)
    return best[1]


def fit_knn(cal, k=10):
    """自适应带宽局部线性：预测点附近最近 k 个标定点做加权线性拟合。"""
    Xc, Yc = P[cal], L[cal] - P[cal]

    def pred(pe):
        dist = np.linalg.norm(Xc - pe[None, :], axis=1)
        idx = np.argsort(dist)[:min(k, len(Xc))]
        h = max(dist[idx[-1]], 1e-3)
        w = np.exp(-(dist[idx] ** 2) / (2 * h * h)) + 1e-6
        A = np.stack([np.ones_like(w), Xc[idx, 0], Xc[idx, 1]], 1) * w[:, None]
        try:
            coef = np.linalg.solve(A.T @ A + 1e-4 * np.eye(3), A.T @ (Yc[idx] * w[:, None]))
        except np.linalg.LinAlgError:
            return np.zeros(2)
        return np.array([1.0, pe[0], pe[1]]) @ coef

    return pred


def field_fit(cal):
    n = 0.0; M = np.zeros((3, 3)); Rx = np.zeros(3); Ry = np.zeros(3)
    grid = defaultdict(lambda: [0.0, 0.0, 0.0])

    def beta():
        if n < MIN_A:
            return None
        A = M + max(0.01 * n, 0.05) * np.diag([0.0, 1.0, 1.0])
        try:
            return np.linalg.solve(A, Rx), np.linalg.solve(A, Ry)
        except np.linalg.LinAlgError:
            return None

    for i in cal:
        Pi, Li = P[i], L[i]
        b = beta()
        u = np.array([1.0, Pi[0], Pi[1]])
        bv = np.array([u @ b[0], u @ b[1]]) if b else np.zeros(2)
        p1 = Pi - bv
        c = grid[min(G - 1, max(0, int(p1[0] * G))) * G + min(G - 1, max(0, int(p1[1] * G)))]
        c[0] += 1.0; c[1] += float(Pi[0] - Li[0]) - float(bv[0]); c[2] += float(Pi[1] - Li[1]) - float(bv[1])
        n += 1.0; M += np.outer(u, u)
        Rx += (Pi[0] - Li[0]) * u; Ry += (Pi[1] - Li[1]) * u

    def corr(Pi):
        b = beta()
        u = np.array([1.0, Pi[0], Pi[1]])
        bv = np.array([u @ b[0], u @ b[1]]) if b else np.zeros(2)
        p1 = Pi - bv
        i0 = min(G - 1, max(0, int(p1[0] * G))); j0 = min(G - 1, max(0, int(p1[1] * G)))
        acc = np.zeros(2); cnt = 0.0
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ii, jj = i0 + di, j0 + dj
                if 0 <= ii < G and 0 <= jj < G:
                    cc = grid[ii * G + jj]
                    if cc[0] > 0:
                        wt = float(np.exp(-(di * di + dj * dj) / (2 * 3.0 ** 2)))
                        acc += wt * np.array([cc[1], cc[2]]); cnt += wt * cc[0]
        return (p1 - np.clip(acc / cnt, -CLIP, CLIP)) if cnt >= MIN_C else p1
    return corr


def run(frac, K, methods=("A", "B", "C", "D", "E", "F", "G"), knn_k=10):
    res = {m: [] for m in methods}
    for s in SESS:
        idx = sess[s]
        cut = max(int(len(idx) * frac), 30)
        pool, ev = idx[:cut], idx[cut:]
        if len(ev) < 40 or len(pool) < 20:
            continue
        cal = [pool[t] for t in np.unique(
            np.linspace(0, len(pool) - 1, min(K, len(pool))).astype(int))]
        Yc = L[cal] - P[cal]
        fc = field_fit(cal)
        mD = fit_tps(cal, Yc)
        fE = fit_knn(cal, knn_k) if "E" in methods else None
        mB, mC, mG = None, None, None
        if "B" in methods:
            mB = []
            for ax in (0, 1):
                X = np.stack([P[cal, ax], np.ones(len(cal))], 1)
                mB.append(np.linalg.lstsq(X, L[cal, ax], rcond=None)[0])
        if "C" in methods:
            mC = []
            for ax in (0, 1):
                p = P[cal, ax]
                X = np.stack([np.ones_like(p), p, p ** 2], 1)
                mC.append(np.linalg.lstsq(X, L[cal, ax], rcond=None)[0])
        if "G" in methods:
            mG = KRR("linear").fit(XFEAT[cal], Yc)
        for i in ev:
            pi = P[i]
            if "A" in methods:
                res["A"].append(emv(pi - Yc.mean(0), L[i], str(paths[i])))
            if "B" in methods:
                q = np.array([mB[0][0] + mB[0][1] * pi[0], mB[1][0] + mB[1][1] * pi[1]])
                res["B"].append(emv(q, L[i], str(paths[i])))
            if "C" in methods:
                q = np.array([mC[0] @ np.array([1, pi[0], pi[0] ** 2]),
                              mC[1] @ np.array([1, pi[1], pi[1] ** 2])])
                res["C"].append(emv(q, L[i], str(paths[i])))
            if "D" in methods:
                res["D"].append(emv(pi + mD.predict(pi[None, :])[0], L[i], str(paths[i])))
            if "E" in methods:
                res["E"].append(emv(pi + fE(pi), L[i], str(paths[i])))
            if "F" in methods:
                res["F"].append(emv(fc(pi), L[i], str(paths[i])))
            if "G" in methods:
                res["G"].append(emv(pi + mG.predict(XFEAT[i:i + 1])[0], L[i], str(paths[i])))
    return {k: np.array(v) for k, v in res.items()}


NM = {"A": "offset", "B": "affine", "C": "poly2", "D": "TPS/RBF", "E": "kNN局部",
      "F": "误差场(旧)", "G": "C2特征(我们)"}
print("=" * 116)
print("文献口径标定对照：K 个标定点、评估期零真值（每会话开头 frac 的比例内取 K 帧）")
print("=" * 116)
for frac in (0.10, 0.20, 0.70):
    print(f"\n  标定窗口 = 会话前 {int(frac*100)}%")
    print(f"    {'K':>5}" + "".join(f"{NM[m]:>16}" for m in ("A", "B", "C", "D", "E", "F", "G")))
    for K in (20, 50, 100, 200):
        r = run(frac, K)
        b = run(frac, K, methods=("A",))["A"].mean()   # 以 offset 为参考基线口径统一
        line = f"    {K:>5}"
        for m in ("A", "B", "C", "D", "E", "F", "G"):
            line += f"{r[m].mean():>16.3f}"
        print(line)
    r = run(frac, 200)
    print("     相对 offset：" + "  ".join(
        f"{NM[m]}={100*(r[m].mean()/r['A'].mean()-1):+.1f}%" for m in ("B", "C", "D", "E", "F", "G")))
