"""修正版 2：一次性标定下的索引键重测（会话口径）+ 真正的级联装法。

教训（本轮踩过两次）：静默空操作 —— 传错参数/改名不匹配 ⇒ 分支永不触发 ⇒ 结果看似"没差别" ✗。
故本脚本对每个关键分支都打印"实际生效帧数"自检 ✓。

A 一次性标定（每会话各取 head/spread K 帧构建状态，冻结后评估其余帧）：键 p1 vs 键 P
B 级联（dense 因果在线）：C2 每 500 帧重拟合（refit=True 已确认生效），输出喂 V1 场
C 级联（一次性 spread K）：C2 在 cal 帧上拟合 → 输出喂 V1 场
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/xieliangcai/code/gazeformer-exps")
NPZ = ROOT / "out/rgbdgaze/analysis/feats_eye224_test.npz"
CKPT = ROOT / "out/rgbdgaze/ablation/checkpoints/eye224_mgpu/best.pt"
G, MIN, NZ = 8, 8, 12
ALPHAS = np.logspace(-2, 4, 7)

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
XC2 = np.concatenate([F[:, nseg:2 * nseg] @ Wc.T + bc,
                      F[:, 2 * nseg:3 * nseg] @ Wp.T + bp,
                      P, np.linalg.norm(P - 0.5, axis=1, keepdims=True)], 1)

sess = defaultdict(list)
for i in range(len(P)):
    sess[str(paths[i]).rsplit("/", 1)[0]].append(i)
for s in sess:
    sess[s].sort(key=lambda i: fi[i])
SESS = sorted(sess)


def cidx(pt):
    return min(G - 1, max(0, int(pt[0] * G))) * G + min(G - 1, max(0, int(pt[1] * G)))


def em(pred, lab, path):
    w, h = sc[path]
    return float(np.hypot((pred[0] - lab[0]) * w, (pred[1] - lab[1]) * h))


def affine_of(g, slope=np.inf):
    if g[0] < MIN:
        return None
    A = np.array([[g[0], g[1], g[2]], [g[1], g[3] + 0.01 * g[0], g[4]],
                  [g[2], g[4], g[5] + 0.01 * g[0]]])
    try:
        b1, b2 = np.linalg.solve(A, g[6:9]), np.linalg.solve(A, np.array([g[9], g[10], g[11]]))
    except np.linalg.LinAlgError:
        return None
    if np.isfinite(slope):
        b1 = np.array([b1[0], np.clip(b1[1], -slope, slope), np.clip(b1[2], -slope, slope)])
        b2 = np.array([b2[0], np.clip(b2[1], -slope, slope), np.clip(b2[2], -slope, slope)])
    return b1, b2


def lookup(grid, key):
    c = cidx(key); i0, j0 = c // G, c % G
    acc = np.zeros(2); cnt = 0.0
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            ii, jj = i0 + di, j0 + dj
            if 0 <= ii < G and 0 <= jj < G:
                cc = grid[ii * G + jj]
                cnt += cc[0]; acc += cc[1:]
    return acc / cnt if cnt >= MIN else None


def upd(g, grid, base, lab, p1, bv):
    x, y = float(base[0]), float(base[1])
    g[0] += 1
    g[1] += x; g[2] += y
    g[3] += x * x; g[4] += x * y; g[5] += y * y
    dx, dy = base[0] - lab[0], base[1] - lab[1]
    g[6] += dx; g[7] += dx * x; g[8] += dx * y
    g[9] += dy; g[10] += dy * x; g[11] += dy * y
    c = grid[cidx(p1)]
    c[0] += 1
    c[1] += float(dx) - float(bv[0])
    c[2] += float(dy) - float(bv[1])


class Ridge:
    def __init__(self, alpha=None, alphas=ALPHAS):
        self.alpha, self.alphas = alpha, alphas

    def fit(self, X, Y):
        self.mx_, self.sx_ = X.mean(0), X.std(0) + 1e-6
        self.my_ = Y.mean(0)
        Xs, Yc = (X - self.mx_) / self.sx_, Y - self.my_
        K = Xs @ Xs.T
        w, V = np.linalg.eigh(K)
        self.Xs_, self.V_, self.w_ = Xs, V, w
        a = self.alpha
        if a is None:
            V2, best, a = V ** 2, np.inf, float(self.alphas[0])
            for cand in self.alphas:
                h = V2 @ (w / (w + cand))
                num = V @ ((V.T @ Yc) / (w + cand)[:, None])
                r = (Yc - num) / np.clip(1.0 - h, 1e-6, None)[:, None]
                m = float((r ** 2).mean())
                if m < best:
                    best, a = m, cand
        self.alpha_ = a
        self.coef_ = V @ ((V.T @ Yc) / (w + a)[:, None])
        return self

    def predict(self, X):
        return self.my_ + (((X - self.mx_) / self.sx_) @ self.Xs_.T) @ self.coef_


def run(mode, K=200, clip=0.05, key_raw=False, use_c2=False, refit=False, check=False):
    """mode: all（全标签在线）/ head / spread（一次性：仅 cal 帧更新状态）"""
    errs, base_ok, c2_used = [], [], 0
    mode = "all" if mode == "all" else mode
    for s in SESS:
        idx = sess[s]
        if mode == "all":
            cal_set, ev = set(idx), idx
        elif mode == "head":
            cal = set(idx[:min(K, len(idx) // 2)])
            cal_set = cal
            ev = [i for i in idx if i not in cal_set]
        else:
            cal = set(np.unique(np.linspace(0, len(idx) - 1, min(K, len(idx))).astype(int)).tolist())
            cal_set = {idx[t] for t in cal}
            ev = [i for i in idx if i not in cal_set]
        g, grid, m2 = np.zeros(NZ), defaultdict(lambda: [0.0, 0.0, 0.0]), None
        if use_c2 and mode != "all":
            cl = [i for i in idx if i in cal_set]
            m2 = Ridge(alpha=None).fit(XC2[cl], L[cl] - P[cl])
        for n, i in enumerate(idx):
            if use_c2 and mode == "all" and n >= 200 and n % 500 == 0:
                seen = idx[:n]
                sel = [seen[t] for t in np.unique(
                    np.linspace(0, len(seen) - 1, min(1000, len(seen))).astype(int))]
                m2 = Ridge(alpha=None).fit(XC2[sel], L[sel] - P[sel])
            if m2 is not None:
                c2_used += 1
                base = P[i] + m2.predict(XC2[i:i + 1])[0]
            else:
                base = P[i]
            b = affine_of(g, np.inf)
            bv = np.array([np.array([1.0, base[0], base[1]]) @ b[0],
                           np.array([1.0, base[0], base[1]]) @ b[1]]) if b else np.zeros(2)
            p1 = base - bv
            key = base if key_raw else p1
            dd = lookup(grid, key)
            corr = key - np.clip(dd, -clip, clip) if dd is not None else p1
            if i in ev:
                errs.append(em(corr, L[i], str(paths[i])))
                base_ok.append(em(P[i], L[i], str(paths[i])))
            if mode == "all" or i in cal_set:
                upd(g, grid, base, L[i], p1, bv)
    if check:
        print(f"    [自检] 评估帧 {len(errs)}；C2 生效帧 {c2_used}"
              f"（{'✓ 非零' if c2_used else '✗ 零 ⇒ 静默空操作!'}）")
    return np.array(errs), np.array(base_ok)


print("=" * 100)
print("A 一次性标定（每会话内取 K 帧建状态、冻结，评估其余帧）：键 p1 vs 键 P")
print("=" * 100)
print(f"  {'协议':>8}{'K':>6}{'不修正':>10}{'键p1(V1)':>18}{'键P(V3)':>18}")
for mode in ("head", "spread"):
    for K in (100, 200, 500):
        e1, b = run(mode, K, key_raw=False)
        e3, _ = run(mode, K, key_raw=True)
        print(f"  {mode:>8}{K:>6}{b.mean():>10.3f}{e1.mean():>12.3f}"
              f"({100 * (e1.mean() / b.mean() - 1):>+4.1f}%)"
              f"{e3.mean():>12.3f}({100 * (e3.mean() / b.mean() - 1):>+4.1f}%)")

print()
print("=" * 100)
print("B 级联（全标签因果在线）：C2 每 500 帧重拟合 ⇒ 输出喂 V1 场")
print("=" * 100)
e_v1, b_all = run("all", key_raw=False)
e_cas, _ = run("all", key_raw=False, use_c2=True, check=True)
print(f"  {'方案':<26}{'EM均值':>9}{'vs基线':>9}{'>3cm':>8}{'max':>8}")
for nm, e in (("A 不修正", b_all), ("V1 场（1.093 口径）", e_v1), ("C2 + V1 场（级联）", e_cas)):
    print(f"  {nm:<26}{e.mean():>9.3f}{100 * (e.mean() / b_all.mean() - 1):>8.1f}%"
          f"{100 * (e > 3).mean():>7.2f}%{e.max():>8.2f}")
print(f"  级联相对 1.093 口径：{100 * (e_cas.mean() / e_v1.mean() - 1):+.1f}%")

print()
print("=" * 100)
print("C 级联（一次性 spread K）：C2 在 cal 帧上拟合 ⇒ 输出喂 V1 场")
print("=" * 100)
print(f"  {'K':>6}{'不修正':>9}{'V1场':>17}{'C2(仅特征)':>17}{'C2+V1场(级联)':>19}")
for K in (200, 500, 1000):
    e1, b = run("spread", K, key_raw=False)
    ec, _ = run("spread", K, key_raw=False, use_c2=True, check=False)
    # C2 单独（不走场）：用 key_raw 关掉场不易；此处给出 C2 输出直接对照
    e_c2only = []
    for s in SESS:
        idx = sess[s]
        cal = set(np.unique(np.linspace(0, len(idx) - 1, min(K, len(idx))).astype(int)).tolist())
        cl = [idx[t] for t in cal]
        ev = [i for i in idx if idx.index(i) not in cal]
        m2 = Ridge(alpha=None).fit(XC2[cl], L[cl] - P[cl])
        for i in ev:
            e_c2only.append(em(P[i] + m2.predict(XC2[i:i + 1])[0], L[i], str(paths[i])))
    e_c2only = np.array(e_c2only)
    print(f"  {K:>6}{b.mean():>9.3f}{e1.mean():>11.3f}({100 * (e1.mean() / b.mean() - 1):>+4.1f}%)"
          f"{e_c2only.mean():>11.3f}({100 * (e_c2only.mean() / b.mean() - 1):>+4.1f}%)"
          f"{ec.mean():>13.3f}({100 * (ec.mean() / b.mean() - 1):>+4.1f}%)")
