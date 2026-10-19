"""可比口径冲刺 v2：一次性标定到底要"多长 + 怎么选"，才能像文献那样拿到收益。

v1 的负结果（重要）：标定池只取「开头前 10% 帧」时，所有方法都只有 0~+6% ✗；
而池取「前 70%」时同样方法有 −17~−20% ✓ ⇒ 决定因素不是估计器，而是**标定覆盖了注视空间的多少**。

本脚本回答落地问题：
  · 标定窗口 frac ∈ {5%,10%,20%,50%}（会话开头，评估=其余，严格因果、零评估期真值）
  · 选取规则：spread（窗内等距，现状） vs **cover（按「预测位置」做 3x3 分层，每格轮替取最近帧）**
    ⇒ cover 是部署可用的（只用预测，不用真值 ✓；等价于"让用户看屏幕各处"的引导式标定）
  · 方法：V1 场 / C2 线性岭 / 核岭(RBF×扩展特征) / 核岭+场级联
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/xieliangcai/code/gazeformer-exps")
NPZ = ROOT / "out/rgbdgaze/analysis/feats_eye224_test.npz"
CKPT = ROOT / "out/rgbdgaze/ablation/checkpoints/eye224_mgpu/best.pt"
G, MIN_A, MIN_C, CLIP, SIG = 8, 3, 8, 0.05, 3.0
ALPHAS = np.logspace(-3, 5, 17)
GAMMAS = (0.5, 1.0, 2.0)
ALPHA_S = (0.0, 0.25, 0.5, 0.75, 1.0)

sc = {}
with open(ROOT / "data/RGBDGaze/index/index.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        sc[r["rgb_path"]] = (float(r["screen_w"]) * float(r["cm_px_x"]),
                             float(r["screen_h"]) * float(r["cm_px_y"]))
d = np.load(NPZ, allow_pickle=True)
F, P, L = d["feats"].astype(np.float64), d["pred"].astype(np.float64), d["label"].astype(np.float64)
paths, fi = d["path"], d["frame_index"]
nseg = F.shape[1] // 4
Fused, Fcls, Fpat, Feye = (F[:, 0:nseg], F[:, nseg:2 * nseg],
                           F[:, 2 * nseg:3 * nseg], F[:, 3 * nseg:4 * nseg])
st = torch.load(CKPT, map_location="cpu", weights_only=False)


def _get(n):
    for k in (n, f"module.{n}", f"model.{n}"):
        if k in st:
            return st[k].float().numpy()
    raise KeyError(n)


Wc, bc = _get("proj_rgb_cls.weight"), _get("proj_rgb_cls.bias")
Wp, bp = _get("proj_rgb_patch.weight"), _get("proj_rgb_patch.bias")
CLS = Fcls @ Wc.T + bc
PAT = Fpat @ Wp.T + bp
R = np.linalg.norm(P - 0.5, axis=1, keepdims=True)
QUAD = np.concatenate([P, P ** 2, (P[:, 0:1] * P[:, 1:2]), R, R ** 2], 1)
XLIN = np.concatenate([CLS, PAT, P, R], 1)
XEXT = np.concatenate([CLS, PAT, Feye, Fused, QUAD], 1)

sess = defaultdict(list)
for i in range(len(P)):
    sess[str(paths[i]).rsplit("/", 1)[0]].append(i)
for s in sess:
    sess[s].sort(key=lambda i: fi[i])
SESS = sorted(sess)


def cidx(pt):
    return min(G - 1, max(0, int(pt[0] * G))) * G + min(G - 1, max(0, int(pt[1] * G)))


def emv(pred, lab, path):
    w, h = sc[path]
    return float(np.hypot((pred[0] - lab[0]) * w, (pred[1] - lab[1]) * h))


class KRR:
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
        self.alpha_ = ba
        self.coef_ = V @ ((V.T @ Yc) / (w + ba)[:, None])
        return self

    def predict(self, X):
        Xs = (X - self.mx_) / self.sx_
        return self.my_ + self._k(Xs, self.Xs_) @ self.coef_


def pick(rule, pool, K):
    if rule == "spread":
        return [pool[t] for t in np.unique(np.linspace(0, len(pool) - 1, min(K, len(pool))).astype(int))]
    # cover：按预测位置的 3x3 网格分层，逐格轮替取（每格内取最接近该格中心者，部署可用）
    order = list(P[:, 0] > 0.5), list(P[:, 1] > 0.5)
    buckets = defaultdict(list)
    for i in pool:
        key = (int(P[i, 0] * 3), int(P[i, 1] * 3))
        buckets[key].append(i)
    for k in buckets:
        cx = (k[0] + 0.5) / 3.0; cy = (k[1] + 0.5) / 3.0
        buckets[k].sort(key=lambda i: (P[i, 0] - cx) ** 2 + (P[i, 1] - cy) ** 2)
    sel, keys = [], sorted(buckets)
    ptr = {k: 0 for k in keys}
    while len(sel) < min(K, len(pool)) and any(ptr[k] < len(buckets[k]) for k in keys):
        for k in keys:
            if ptr[k] < len(buckets[k]) and len(sel) < min(K, len(pool)):
                sel.append(buckets[k][ptr[k]]); ptr[k] += 1
    del order
    return sorted(sel)


def field_fit(cal, base_pred):
    """V1：仿射 + 核加权场（键 p1）。base_pred(i) 给出基座预测（默认用 P）。"""
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
        Pi = base_pred[i]
        Li = L[i]
        b = beta()
        u = np.array([1.0, Pi[0], Pi[1]])
        bv = np.array([u @ b[0], u @ b[1]]) if b else np.zeros(2)
        p1 = Pi - bv
        c = grid[cidx(p1)]
        c[0] += 1.0
        c[1] += float(Pi[0] - Li[0]) - float(bv[0])
        c[2] += float(Pi[1] - Li[1]) - float(bv[1])
        dx, dy = Pi[0] - Li[0], Pi[1] - Li[1]
        n += 1.0
        M += np.outer(u, u)
        Rx += dx * u; Ry += dy * u

    def correct(Pi):
        b = beta()
        u = np.array([1.0, Pi[0], Pi[1]])
        bv = np.array([u @ b[0], u @ b[1]]) if b else np.zeros(2)
        p1 = Pi - bv
        c = cidx(p1); i0, j0 = c // G, c % G
        acc = np.zeros(2); cnt = 0.0
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ii, jj = i0 + di, j0 + dj
                if 0 <= ii < G and 0 <= jj < G:
                    cc = grid[ii * G + jj]
                    if cc[0] <= 0:
                        continue
                    wt = float(np.exp(-(di * di + dj * dj) / (2 * SIG ** 2)))
                    acc += wt * np.array([cc[1], cc[2]])
                    cnt += wt * cc[0]
        return (p1 - np.clip(acc / cnt, -CLIP, CLIP)) if cnt >= MIN_C else p1

    return correct


def run(frac, K, rule="spread", which=("M0", "M2", "M2v")):
    res = {m: [] for m in which}
    for s in SESS:
        idx = sess[s]
        cut = max(int(len(idx) * frac), 40)
        pool, ev = idx[:cut], idx[cut:]
        if len(pool) < 20 or len(ev) < 50:
            continue
        cal = pick(rule, pool, K)
        Xev_lin, Xev_ext = XLIN[ev], XEXT[ev]
        Yc = L[cal] - P[cal]
        if "M2v" in which:                      # 交错对半：A 拟合 / B 选 α
            Ac, Bc = cal[::2], cal[1::2]
            m_ = KRR("linear").fit(XLIN[Ac], L[Ac] - P[Ac])
            corrB = m_.predict(XLIN[Bc])
            best_a, best_e = 0.0, 1e9
            for a in ALPHA_S:
                e = np.mean([np.hypot((P[Bc[j]][0] + a * corrB[j][0] - L[Bc[j]][0]) * sc[str(paths[Bc[j]])][0],
                                      (P[Bc[j]][1] + a * corrB[j][1] - L[Bc[j]][1]) * sc[str(paths[Bc[j]])][1])
                             for j in range(len(Bc))])
                if e < best_e:
                    best_e, best_a = e, a
            m_full = KRR("linear").fit(XLIN[cal], Yc)
            p2v = P[ev] + best_a * m_full.predict(Xev_lin)
        fc = field_fit(cal, P) if ("M1" in which or "M7" in which) else None
        if "M2" in which:
            p2 = P[ev] + KRR("linear").fit(XLIN[cal], Yc).predict(Xev_lin)
        if "M6" in which or "M7" in which:
            best = None
            for g in GAMMAS:
                m = KRR("rbf", g).fit(XEXT[cal], Yc)
                err = float(((Yc - m.predict(XEXT[cal])) ** 2).mean())
                if best is None or err < best[0]:
                    best = (err, m)
            seg = list(cal) + list(ev)
            p6_seg = P[seg] + best[1].predict(XEXT[seg])
            p6 = p6_seg[len(cal):]                      # 评估帧上的 M6 预测
            base6 = {i: np.array(p6_seg[k]) for k, i in enumerate(seg)}
            fc2 = field_fit(cal, base6) if "M7" in which else None
        for k_, i in enumerate(ev):
            res["M0"].append(emv(P[i], L[i], str(paths[i])))
            if "M1" in which:
                res["M1"].append(emv(fc(P[i]), L[i], str(paths[i])))
            if "M2" in which:
                res["M2"].append(emv(p2[k_], L[i], str(paths[i])))
            if "M2v" in which:
                res["M2v"].append(emv(p2v[k_], L[i], str(paths[i])))
            if "M6" in which:
                res["M6"].append(emv(p6[k_], L[i], str(paths[i])))
            if "M7" in which:
                res["M7"].append(emv(fc2(base6[i]), L[i], str(paths[i])))
    return {k: np.array(v) for k, v in res.items()}


NM = {"M0": "不标定", "M1": "V1 场", "M2": "C2 线性岭", "M2v": "C2+内部验证α",
      "M6": "核岭×扩展", "M7": "核岭+场级联"}
print("=" * 112)
print("最终版：窗口内等距抽样 + 内部验证选收缩力度 α（评估期零真值；A/B 交错对半）")
print("=" * 112)
print(f"    {'frac':>6}{'K':>5}{'不标定':>9}{'α=1':>16}{'α 内部验证':>18}{'V1场':>16}{'核岭×扩展':>16}")
for frac in (0.05, 0.10, 0.20, 0.50, 0.70):
    for K in (50, 200):
        r = run(frac, K, "spread", ("M0", "M2", "M2v", "M1", "M6"))
        b = r["M0"].mean()
        print(f"    {frac:>6.2f}{K:>5}{b:>9.3f}"
              f"{r['M2'].mean():>10.3f}({100 * (r['M2'].mean() / b - 1):>+4.1f}%)"
              f"{r['M2v'].mean():>12.3f}({100 * (r['M2v'].mean() / b - 1):>+4.1f}%)"
              f"{r['M1'].mean():>10.3f}({100 * (r['M1'].mean() / b - 1):>+4.1f}%)"
              f"{r['M6'].mean():>10.3f}({100 * (r['M6'].mean() / b - 1):>+4.1f}%)")
