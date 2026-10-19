"""文献口径标定对照 v2：**每个方法都配安全层（α 内部验证）**，并给两条参照。

v1 的教训：不加安全层时，经典方法在短窗口会"炸"（affine 10.8cm、offset 2.0cm ✗✗）——
  短窗口里标定点分布退化 ⇒ 最小二乘斜率爆炸；偏置取到"早期 regime"的值。
  这正是我们此前发现过的病（全局映射在错配标定下灾难），不是方法本身差。
所以公平对照必须给**每个**方法同样的安全层：
  标定帧交错对半 A/B ⇒ A 拟合、B 上选收缩力度 α ∈ {0, .25, .5, .75, 1}（选不出就 α=0）⇒
  用全部标定帧重拟合、按 α 施加。⇒ 结构上"永不伤害"。

方法（都不是我们发明的，来自文献/通用眼动校准）：
  A offset（Chen & Shi 2020 式偏置标定） B affine（经典线性） C poly2（经典多项式）
  D TPS/RBF 平滑修正映射（商业眼动仪常用；γ、α 由标定集留一误差选）
  E kNN 自适应带宽局部线性   F 我们的 8×8 误差场   G 我们的 C2（投影特征⊕预测→残差）
两条参照：0 不标定；A offset。
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
AS = (0.0, 0.25, 0.5, 0.75, 1.0)
METHODS = ("A", "B", "C", "D", "E", "F", "G")

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
XF = np.concatenate([F[:, nseg:2 * nseg] @ Wc.T + bc,
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
    def __init__(self, kind="rbf", gamma=5.0):
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
        self.Xs_, Yc = Xs, Y - self.my_
        K = self._k(Xs, Xs)
        w, V = np.linalg.eigh(K)
        V2, best, ba = V ** 2, np.inf, float(ALPHAS[0])
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


def make_A(method, cal, knn_k=10):
    """在 cal 上拟合方法，返回 corr(P_i) -> 修正后坐标。"""
    y = L[cal] - P[cal]
    if method == "A":
        d0 = y.mean(0)
        return lambda pi: pi - d0
    if method == "B":
        c = [np.linalg.lstsq(np.stack([P[cal, ax], np.ones(len(cal))], 1),
                             L[cal, ax], rcond=None)[0] for ax in (0, 1)]
        return lambda pi: np.array([c[0][0] + c[0][1] * pi[0], c[1][0] + c[1][1] * pi[1]])
    if method == "C":
        c = [np.linalg.lstsq(np.stack([np.ones(len(cal)), P[cal, ax], P[cal, ax] ** 2], 1),
                             L[cal, ax], rcond=None)[0] for ax in (0, 1)]
        return lambda pi: np.array([c[0] @ np.array([1, pi[0], pi[0] ** 2]),
                                    c[1] @ np.array([1, pi[1], pi[1] ** 2])])
    if method == "D":
        best = None
        for gm in (2.0, 5.0, 10.0, 20.0, 50.0):
            for kind in ("rbf", "linear"):
                m = KRR(kind, gm).fit(P[cal], y)
                e = float(((y - m.predict(P[cal])) ** 2).mean())
                if best is None or e < best[0]:
                    best = (e, m)
        m = best[1]
        return lambda pi: pi + m.predict(pi[None, :])[0]
    if method == "E":
        Xc, Yc = P[cal], y
        def f(pi):
            dist = np.linalg.norm(Xc - pi[None, :], axis=1)
            idx = np.argsort(dist)[:min(knn_k, len(Xc))]
            h = max(dist[idx[-1]], 1e-3)
            wt = np.exp(-(dist[idx] ** 2) / (2 * h * h)) + 1e-6
            A = np.stack([np.ones_like(wt), Xc[idx, 0], Xc[idx, 1]], 1) * wt[:, None]
            try:
                coef = np.linalg.solve(A.T @ A + 1e-4 * np.eye(3), A.T @ (Yc[idx] * wt[:, None]))
            except np.linalg.LinAlgError:
                return pi.copy()
            return pi + np.array([1.0, pi[0], pi[1]]) @ coef
        return f
    if method == "F":
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
        def fc(pi):
            b = beta()
            u = np.array([1.0, pi[0], pi[1]])
            bv = np.array([u @ b[0], u @ b[1]]) if b else np.zeros(2)
            p1 = pi - bv
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
        return fc
    if method == "G":
        m = KRR("linear").fit(XF[cal], y)
        return lambda pi_ignored=None, _i=None: None   # 占位（G 需要帧索引，见下）
    raise KeyError(method)


def run(frac, K, method):
    """返回 (EM数组, α数组)。方法自带安全层（不含 G，G 走特征索引）。"""
    errs, alphas = [], []
    for s in SESS:
        idx = sess[s]
        cut = max(int(len(idx) * frac), 30)
        pool, ev = idx[:cut], idx[cut:]
        if len(ev) < 40 or len(pool) < 24:
            continue
        cal = [pool[t] for t in np.unique(
            np.linspace(0, len(pool) - 1, min(K, len(pool))).astype(int))]
        A_, B_ = cal[::2], cal[1::2]
        # 统一约定：修正函数接受**帧索引**，返回修正后的屏幕坐标
        if method == "G":
            gA = (lambda i: P[i] + KRR("linear").fit(XF[A_], L[A_] - P[A_]
                                                    ).predict(XF[i:i + 1])[0])
        else:
            _fA = make_A(method, A_)
            gA = lambda i: _fA(P[i])
        cb = np.array([gA(i) - P[i] for i in B_])
        # 用 B 半选 α（B 半属于标定集 ⇒ 不违反"评估期零真值"）
        best_a, best_e = 0.0, 1e9
        for a in AS:
            e = np.mean([emv(P[i] + a * cb[j], L[i], str(paths[i])) for j, i in enumerate(B_)])
            if e < best_e:
                best_e, best_a = e, a
        alphas.append(best_a)
        if method == "G":
            _mF = KRR("linear").fit(XF[cal], L[cal] - P[cal])
            gfull = lambda i: P[i] + _mF.predict(XF[i:i + 1])[0]
        else:
            _fF = make_A(method, cal)
            gfull = lambda i: _fF(P[i])
        for i in ev:
            errs.append(emv(P[i] + best_a * (gfull(i) - P[i]), L[i], str(paths[i])))
    return np.array(errs), np.array(alphas)


def run_base(frac, K):
    errs = []
    for s in SESS:
        idx = sess[s]
        cut = max(int(len(idx) * frac), 30)
        for i in idx[cut:]:
            errs.append(emv(P[i], L[i], str(paths[i])))
    return np.array(errs)


NM = {"A": "offset", "B": "affine", "C": "poly2", "D": "TPS/RBF", "E": "kNN局部",
      "F": "误差场", "G": "C2特征"}
print("=" * 118)
print("v2：所有方法均配 α 内部验证（结构上永不伤害）；参照 = 不标定 / offset")
print("=" * 118)
for frac in (0.10, 0.20, 0.70):
    for K in (50, 200):
        base = run_base(frac, K).mean()
        print(f"\n  窗口{int(frac*100)}% K={K} 不标定={base:.3f}")
        for m in METHODS:
            e, al = run(frac, K, m)
            print(f"    {NM[m]:<9}{e.mean():>8.3f}  相对不标定 {100*(e.mean()/base-1):>+6.1f}%"
                  f"   ᾱ={al.mean():.2f}")
