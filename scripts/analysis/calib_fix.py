"""修正版：按「会话」分组（= field_hier.py 口径），重做本轮关键实验。

修正的 bug：此前按被试分组再按 frame_index 排序 ⇒ 各 activity 会话交错拼接 ✗
（frame_index 每个会话从 0 重开）⇒ 1.093 复现失败（得 1.218）。
权威口径：sess = path 去掉文件名；会话内按 frame_index 排序 ✓

T1 索引键 × 协议（重验 F7）
T2 复现 1.093 并检验钳制是否免费（全标签、因果在线、G=8）
T3 装进 1.093：C2(特征+预测→残差) 每 500 帧重拟合 ⇒ 其输出喂 V1 场（全因果在线）
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
F = d["feats"].astype(np.float64)
P = d["pred"].astype(np.float64)
L = d["label"].astype(np.float64)
paths, subj, fi = d["path"], d["subject"], d["frame_index"]

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
    sess[s].sort(key=lambda i: fi[i])            # 会话内按帧号；跨会话不混 ✓
SESS = sorted(sess)
print(f"会话数 {len(SESS)}，帧数 {len(P)}（按会话分组 ✓）")


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
        b1 = np.linalg.solve(A, g[6:9])
        b2 = np.linalg.solve(A, np.array([g[9], g[10], g[11]]))
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


def upd(g, grid, base, lab, key, bv):
    x, y = float(base[0]), float(base[1])
    g[0] += 1
    g[1] += x; g[2] += y
    g[3] += x * x; g[4] += x * y; g[5] += y * y
    dx, dy = base[0] - lab[0], base[1] - lab[1]
    g[6] += dx; g[7] += dx * x; g[8] += dx * y
    g[9] += dy; g[10] += dy * x; g[11] += dy * y
    c = grid[cidx(key)]
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


def pipeline(base_of, clip=np.inf, slope=np.inf, key_raw=False, refit=False, eval_ids=None):
    """V1 流水线（按会话因果在线）。base_of(i, m2) 给出基座预测。"""
    has_ft = getattr(base_of, "needs_m2", False)
    errs = []
    for s in SESS:
        idx = sess[s]
        g, grid, m2 = np.zeros(NZ), defaultdict(lambda: [0.0, 0.0, 0.0]), None
        for n, i in enumerate(idx):
            if has_ft and refit and n >= 100 and n % 500 == 0:
                seen = idx[:n]
                sel = [seen[t] for t in np.unique(
                    np.linspace(0, len(seen) - 1, min(1000, len(seen))).astype(int))]
                m2 = Ridge(alpha=None).fit(XC2[sel], L[sel] - P[sel])
            base = np.asarray(base_of(i, m2), dtype=float)
            b = affine_of(g, slope)
            bv = np.array([np.array([1.0, base[0], base[1]]) @ b[0],
                           np.array([1.0, base[0], base[1]]) @ b[1]]) if b else np.zeros(2)
            p1 = base - bv
            key = base if key_raw else p1
            dd = lookup(grid, key)
            corr = key - np.clip(dd, -clip, clip) if dd is not None else p1
            if eval_ids is None or i in eval_ids:
                errs.append(em(corr, L[i], str(paths[i])))
            upd(g, grid, base, L[i], p1, bv)
    return np.array(errs)


def raw_base(i, _m=None):
    return P[i]


# ---------------- T1 索引键 × 协议（按会话） ----------------
print()
print("=" * 100)
print("T1 索引键 × 协议（按会话分组；偏移钳制 0.05）")
print("=" * 100)
print(f"  {'协议':>10}{'不修正':>10}{'键p1(V1)':>18}{'键P(V3)':>18}")
for tag, kw in (("all", {}), ("head200", {}), ("spread200", {})):
    ev_ids = None
    if tag != "all":
        ev_ids = set()
        K = 200
        for s in SESS:
            idx = sess[s]
            cal = (idx[:min(K, len(idx) // 2)] if tag == "head200"
                   else [idx[t] for t in np.unique(np.linspace(0, len(idx) - 1, min(K, len(idx))).astype(int))])
            calset = set(cal)
            ev_ids |= {i for i in idx if i not in calset}
        # 标定帧的场：仅用 cal 帧构建（一次性），评估其外推
        r1 = pipeline(lambda i, m=None: P[i], 0.05, np.inf, key_raw=False, eval_ids=ev_ids)
        r3 = pipeline(lambda i, m=None: P[i], 0.05, np.inf, key_raw=True, eval_ids=ev_ids)
        base = np.array([em(P[i], L[i], str(paths[i])) for i in range(len(P)) if i in ev_ids])
    else:
        r1 = pipeline(raw_base, 0.05, np.inf, key_raw=False)
        r3 = pipeline(raw_base, 0.05, np.inf, key_raw=True)
        base = np.array([em(P[i], L[i], str(paths[i])) for i in range(len(P))])
    print(f"  {tag:>10}{base.mean():>10.3f}{r1.mean():>12.3f}"
          f"({100 * (r1.mean() / base.mean() - 1):>+4.1f}%)"
          f"{r3.mean():>12.3f}({100 * (r3.mean() / base.mean() - 1):>+4.1f}%)")

# ---------------- T2 复现 1.093 + 钳制 ----------------
print()
print("=" * 100)
print("T2 复现 1.093（全标签/因果在线/G=8/按会话）并检验钳制")
print("=" * 100)
base_all = np.array([em(P[i], L[i], str(paths[i])) for i in range(len(P))])
print(f"  {'配置':<22}{'EM均值':>9}{'vs基线':>9}{'>3cm':>8}{'max':>8}")
print(f"  {'A 不修正':<22}{base_all.mean():>9.3f}{0.0:>8.1f}%"
      f"{100 * (base_all > 3).mean():>7.2f}%{base_all.max():>8.2f}")
for tag, kw in (("V1 键p1（基准）", dict(key_raw=False, clip=np.inf)),
                ("V1 + 偏移钳制0.05", dict(key_raw=False, clip=0.05)),
                ("V1 + 偏移钳制0.10", dict(key_raw=False, clip=0.10)),
                ("V1 + 斜率钳制0.50", dict(key_raw=False, clip=np.inf, slope=0.5)),
                ("V3 键P + 钳制0.05", dict(key_raw=True, clip=0.05))):
    e = pipeline(raw_base, **kw)
    print(f"  {tag:<22}{e.mean():>9.3f}{100 * (e.mean() / base_all.mean() - 1):>8.1f}%"
          f"{100 * (e > 3).mean():>7.2f}%{e.max():>8.2f}")


# ---------------- T3 装进 1.093：级联 ----------------
def c2_base(i, m2):
    if m2 is None:
        return P[i]
    return P[i] + m2.predict(XC2[i:i + 1])[0]


c2_base.needs_m2 = True
c2_only = np.array([em(c2_base(i, None) if True else P[i], L[i], str(paths[i]))
                    for i in range(len(P))])
print()
print("=" * 100)
print("T3 装进 1.093：C2 每 500 帧用「迄今标签(≤1000)」重拟合，输出再喂 V1 场")
print("=" * 100)
e_v1 = pipeline(raw_base, 0.05)
e_c2 = pipeline(c2_base, 0.05)          # C2 输出的场（= 级联）
e_c2_only = pipeline(c2_base, 0.05, key_raw=True, refit=False)
print(f"  {'方案':<30}{'EM均值':>9}{'vs基线':>9}{'>2cm':>8}{'>3cm':>8}{'>5cm':>8}{'max':>8}")
for nm, e in (("A 不修正", base_all), ("V1 场（1.093 口径）", e_v1),
              ("C2 + V1 场（级联）", e_c2)):
    print(f"  {nm:<30}{e.mean():>9.3f}{100 * (e.mean() / base_all.mean() - 1):>8.1f}%"
          f"{100 * (e > 2).mean():>7.2f}%{100 * (e > 3).mean():>7.2f}%"
          f"{100 * (e > 5).mean():>7.2f}%{e.max():>8.2f}")
print(f"\n  级联相对 1.093 口径：{100 * (e_c2.mean() / e_v1.mean() - 1):+.1f}%")
