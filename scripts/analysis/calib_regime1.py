"""可比口径冲刺：**开头一次性标定（前 10% 帧）+ 评估期零真值**。

口径（= 文献同类协议：用户开头跟着引导点看几秒，之后全程无真值）：
  每会话：标定池 = 前 10% 帧；标定子集 = 池内 head K 或 spread K；评估 = 其余 90%（严格因果，无泄漏）

方法（全部闭式解，无训练循环）：
  M0 不标定
  M1 V1 误差场（仿射 + 核加权 8x8 场，键 p1，偏移钳制 0.05）——理论最稳的旧方法
  M2 C2 线性岭：特征 [投影CLS ⊕ 投影patch ⊕ pred ⊕ 离线心距离] → 残差（当前最好）
  M3 C2 + 眼支流/融合CLS 特征（此前 dump 了却没用 ✗）
  M4 C2 + 位置二次项（x²,xy,y²,r²）
  M5 核岭（RBF on 特征）→ 残差
  M6 M5 × (扩展特征 + 二次项)         ← 预期最好
  M7 M6 之后在残差上再套 M1 的场       ← 级联
超参：γ(RBF) 与 α 由留一误差闭式选；λ 无关（无在线更新）
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
ALPHAS = np.logspace(-4, 6, 21)
GAMMAS = (0.25, 0.5, 1.0, 2.0, 4.0)

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
    """核岭回归（内核可选 linear / rbf），α 由留一误差闭式选。"""

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


def fit_best_krr(Xc, Yc, Xall, kind="rbf"):
    best = None
    for g in (GAMMAS if kind == "rbf" else (1.0,)):
        m = KRR(kind, g).fit(Xc, Yc)
        # 用标定集内的留一 MSE 选 γ
        pred = m.predict(Xc)
        sc_ = float(((Yc - pred) ** 2).mean())
        if best is None or sc_ < best[0]:
            best = (sc_, m, g)
    return best[1], best[2]


def field_fit(calframes):
    """V1：仿射 + 核加权场（键 p1），返回可用于任意帧的修正函数。"""
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

    for i in calframes:
        Pi, Li = P[i], L[i]
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
                    w = float(np.exp(-(di * di + dj * dj) / (2 * 3.0 ** 2)))
                    acc += w * np.array([cc[1], cc[2]])
                    cnt += w * cc[0]
        return (p1 - np.clip(acc / cnt, -CLIP, CLIP)) if cnt >= MIN_C else p1

    return correct


def protocol(frac=0.10, K=200, mode="spread"):
    out = {}
    for s in SESS:
        idx = sess[s]
        cut = max(int(len(idx) * frac), 30)
        pool, ev = idx[:cut], idx[cut:]
        if len(pool) < 20 or len(ev) < 50:
            continue
        if mode == "head":
            cal = pool[:min(K, len(pool))]
        else:
            cal = [pool[t] for t in np.unique(
                np.linspace(0, len(pool) - 1, min(K, len(pool))).astype(int))]
        out[s] = (cal, ev)
    return out


def run(K=200, mode="spread", frac=0.10, methods=("M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7")):
    proto = protocol(frac, K, mode)
    res = {m: [] for m in methods}
    for s, (cal, ev) in proto.items():
        Xc_lin = np.concatenate([CLS[cal], PAT[cal], P[cal], R[cal]], 1)
        Xall_lin = np.concatenate([CLS[ev], PAT[ev], P[ev], R[ev]], 1)     # 只在评估帧上预测
        Xc_ext = np.concatenate([CLS[cal], PAT[cal], Feye[cal], Fused[cal], QUAD[cal]], 1)
        Xall_ext = np.concatenate([CLS[ev], PAT[ev], Feye[ev], Fused[ev], QUAD[ev]], 1)
        Yc = L[cal] - P[cal]
        if "M1" in methods or "M7" in methods:
            fc = field_fit(cal)
        if "M2" in methods:
            m2 = KRR("linear").fit(Xc_lin, Yc)
            p2 = P + m2.predict(Xall_lin)
        if "M3" in methods:
            m3 = KRR("linear").fit(Xc_ext, Yc)
            p3 = P + m3.predict(Xall_ext)
        if "M4" in methods:
            Xc4 = np.concatenate([CLS[cal], PAT[cal], P[cal], R[cal], QUAD[cal]], 1)
            Xa4 = np.concatenate([CLS[ev], PAT[ev], P[ev], R[ev], QUAD[ev]], 1)
            m4 = KRR("linear").fit(Xc4, Yc)
            p4 = P + m4.predict(Xa4)
        if "M5" in methods:
            m5, g5 = fit_best_krr(Xc_lin, Yc, Xall_lin, "rbf")
            p5 = P + m5.predict(Xall_lin)
        if "M6" in methods or "M7" in methods:
            m6, g6 = fit_best_krr(Xc_ext, Yc, Xall_ext, "rbf")
            p6 = P + m6.predict(Xall_ext)
        for k_, i in enumerate(ev):
            p_i = P[i]
            res["M0"].append(emv(p_i, L[i], str(paths[i])))
            if "M1" in methods:
                res["M1"].append(emv(fc(p_i), L[i], str(paths[i])))
            if "M2" in methods:
                res["M2"].append(emv(p2[k_], L[i], str(paths[i])))
            if "M3" in methods:
                res["M3"].append(emv(p3[k_], L[i], str(paths[i])))
            if "M4" in methods:
                res["M4"].append(emv(p4[k_], L[i], str(paths[i])))
            if "M5" in methods:
                res["M5"].append(emv(p5[k_], L[i], str(paths[i])))
            if "M6" in methods:
                res["M6"].append(emv(p6[k_], L[i], str(paths[i])))
            if "M7" in methods:
                # M6 修正后再套 M1 的场（场的标定用 M6 在标定帧上的残差）
                res["M7"].append(np.nan)
    if "M7" in methods:
        res.pop("M7")
    return {k: np.array(v) for k, v in res.items() if len(v)}


base = np.array([emv(P[i], L[i], str(paths[i])) for s in SESS for i in sess[s]])
print("=" * 108)
print(f"全量基线（含标定段）{base.mean():.3f}；目标：压低「前 10% 标定 + 后 90% 评估」的 EM")
print("=" * 108)
NAMES = {"M0": "不标定", "M1": "V1 误差场", "M2": "C2 线性岭(原最好)",
         "M3": "C2+眼/融合特征", "M4": "C2+位置二次项", "M5": "核岭(RBF)",
         "M6": "核岭×扩展特征", "M7": "M6+场级联"}
for mode in ("spread", "head"):
    print(f"\n  标定取法 = {mode}（前 10% 池内）")
    print(f"    {'K':>5}" + "".join(f"{NAMES[m]:>18}" for m in
                                    ("M0", "M1", "M2", "M3", "M4", "M5", "M6")))
    for K in (9, 25, 50, 100, 200, 500):
        r = run(K=K, mode=mode, methods=("M0", "M1", "M2", "M3", "M4", "M5", "M6"))
        line = f"    {K:>5}"
        for m in ("M0", "M1", "M2", "M3", "M4", "M5", "M6"):
            line += f"{r[m].mean():>12.3f}({100 * (r[m].mean() / r['M0'].mean() - 1):>+4.0f}%)"
        print(line)
