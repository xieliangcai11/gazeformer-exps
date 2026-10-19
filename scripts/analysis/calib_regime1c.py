"""可比口径冲刺 v3：让一次性标定**永不伤害** —— 标定窗口内部验证 + 收缩力度 α。

v2 事实：标定窗口 = 会话前 5~10% 时，所有方法都伤害（+5~+14%）✗；
         20%~70% 才有 −11~−15% ✓；核岭在少样本下过拟合、线性岭最稳 ✓。

思路（部署可用，额外真值 0）：
  把「标定窗口」按时间对半切：A 半拟合估计器，B 半**验证**；
  在候选收缩力度 α ∈ {0, 0.25, 0.5, 0.75, 1.0} 中选使 B 半 EM 最小的 α 应用到评估段；
  ⇒ 若修正连 B 半都改善不了，自动取 α=0（= 不修正）⇒ **结构上永不伤害** ✓
对照：α=1（= 现状，不做内部验证）。
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
Fcls, Fpat = F[:, nseg:2 * nseg], F[:, 2 * nseg:3 * nseg]
st = torch.load(CKPT, map_location="cpu", weights_only=False)


def _get(n):
    for k in (n, f"module.{n}", f"model.{n}"):
        if k in st:
            return st[k].float().numpy()
    raise KeyError(n)


Wc, bc = _get("proj_rgb_cls.weight"), _get("proj_rgb_cls.bias")
Wp, bp = _get("proj_rgb_patch.weight"), _get("proj_rgb_patch.bias")
R = np.linalg.norm(P - 0.5, axis=1, keepdims=True)
XLIN = np.concatenate([Fcls @ Wc.T + bc, Fpat @ Wp.T + bp, P, R], 1)

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
    def __init__(self, kind="linear"):
        self.kind = kind

    def _k(self, A, B):
        return A @ B.T

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
        return self.my_ + self._k((X - self.mx_) / self.sx_, self.Xs_) @ self.coef_


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
        c = grid[cidx(p1)]
        c[0] += 1.0
        c[1] += float(Pi[0] - Li[0]) - float(bv[0])
        c[2] += float(Pi[1] - Li[1]) - float(bv[1])
        dx, dy = Pi[0] - Li[0], Pi[1] - Li[1]
        n += 1.0
        M += np.outer(u, u)
        Rx += dx * u; Ry += dy * u

    def corr(Pi):
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
        if cnt < MIN_C:
            return np.zeros(2)
        return -np.clip(acc / cnt, -CLIP, CLIP)

    return corr


def run(frac, method="C2", K=None):
    out = {"M0": [], "naive": [], "validated": [], "alpha": []}
    for s in SESS:
        idx = sess[s]
        cut = max(int(len(idx) * frac), 40)
        pool, ev = idx[:cut], idx[cut:]
        if len(pool) < 24 or len(ev) < 50:
            continue
        h = len(pool) // 2
        A_half, B_half = pool[:h], pool[h:]
        if method == "C2":
            mA = KRR().fit(XLIN[A_half], L[A_half] - P[A_half])
            mB = KRR().fit(XLIN[B_half], L[B_half] - P[B_half])
            corr_eval = mA.predict(XLIN[ev])
            def corr_val(vals):                      # 在 B 半上验证：用 A 半的模型
                return mA.predict(XLIN[vals])
            base_ev, base_B = P[ev], P[B_half]
            corr_B = mA.predict(XLIN[B_half])
        else:
            fA = field_fit(A_half)
            fB = field_fit(B_half)
            corr_eval = np.array([fA(P[i]) for i in ev])
            corr_B = np.array([fA(P[i]) for i in B_half])
            base_ev, base_B = P[ev], P[B_half]
        # α 由 B 半（有真值）选出
        best_a, best_e = 0.0, 1e9
        for a in ALPHA_S:
            e = np.mean([emv(base_B[k] + a * corr_B[k], L[B_half[k]], str(paths[B_half[k]]))
                         for k in range(len(B_half))])
            if e < best_e:
                best_e, best_a = e, a
        out["alpha"].append(best_a)
        for k_, i in enumerate(ev):
            out["M0"].append(emv(base_ev[k_], L[i], str(paths[i])))
            out["naive"].append(emv(base_ev[k_] + corr_eval[k_], L[i], str(paths[i])))
            out["validated"].append(emv(base_ev[k_] + best_a * corr_eval[k_], L[i], str(paths[i])))
    return {k: np.array(v) for k, v in out.items()}


print("=" * 112)
print("内部验证 + 收缩力度 α（B 半选 α，应用于评估段；α=0 即等于不修正）")
print("=" * 112)
print(f"  {'方法':>6}{'frac':>6}{'不标定':>9}{'α=1(现状)':>16}{'α 内部验证':>18}"
      f"{'选出α均值':>11}{'α=0 的会话占比':>15}")
for method in ("C2", "V1"):
    for frac in (0.05, 0.10, 0.20, 0.50, 0.70):
        r = run(frac, method)
        b = r["M0"].mean()
        print(f"  {method:>6}{frac:>6.2f}{b:>9.3f}"
              f"{r['naive'].mean():>10.3f}({100 * (r['naive'].mean() / b - 1):>+4.1f}%)"
              f"{r['validated'].mean():>12.3f}({100 * (r['validated'].mean() / b - 1):>+4.1f}%)"
              f"{r['alpha'].mean():>11.2f}{100 * (r['alpha'] == 0).mean():>14.0f}%")
