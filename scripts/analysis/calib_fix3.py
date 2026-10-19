"""落定版：固定评估集（每会话后 30%），标定只用前 70% ⇒ 各方法严格可比。

背景纠正（本轮）：
  · 会话分组修正后，1.093 精确复现（1.093 = −25.7% ✓），偏移钳制免费（1.091 ✓）
  · 「键P(V3) 更安全」结论作废 ✗（是跨会话交错拼接 bug 的产物）；键p1 全面更优 ✓
  · 全标签在线（=1.093 regime）下，特征级联反而 +2.8% ✗（C2 每 500 帧重拟合有滞后/噪声）
⇒ 本脚本只回答落地问题：**给定一段靠前的标定、评估其后段**，各方法谁强。

协议：每会话前 70% 作标定来源，后 30% 作评估（不重叠、因果）；
      标定取 K 个（head=前 K；spread=在前 70% 内等距）；
      另给"评估段内在线更新"作上界参考（= 该 regime 的信息上限）。
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


def affine_of(g):
    if g[0] < MIN:
        return None
    A = np.array([[g[0], g[1], g[2]], [g[1], g[3] + 0.01 * g[0], g[4]],
                  [g[2], g[4], g[5] + 0.01 * g[0]]])
    try:
        return np.linalg.solve(A, g[6:9]), np.linalg.solve(A, np.array([g[9], g[10], g[11]]))
    except np.linalg.LinAlgError:
        return None


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
    def __init__(self, alphas=ALPHAS):
        self.alphas = alphas

    def fit(self, X, Y):
        self.mx_, self.sx_ = X.mean(0), X.std(0) + 1e-6
        self.my_ = Y.mean(0)
        Xs, Yc = (X - self.mx_) / self.sx_, Y - self.my_
        K = Xs @ Xs.T
        w, V = np.linalg.eigh(K)
        self.Xs_, self.V_, self.w_ = Xs, V, w
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


def run(tag, K, online_in_eval=False):
    errs = []
    for s in SESS:
        idx = sess[s]
        cut = int(len(idx) * 0.7)
        cal_pool, ev = idx[:cut], idx[cut:]
        if len(ev) < 30 or len(cal_pool) < 30:
            continue
        if tag == "head":
            cal = cal_pool[:min(K, len(cal_pool))]
        else:
            cal = [cal_pool[t] for t in np.unique(
                np.linspace(0, len(cal_pool) - 1, min(K, len(cal_pool))).astype(int))]
        m2 = Ridge().fit(XC2[cal], L[cal] - P[cal])
        g, grid = np.zeros(NZ), defaultdict(lambda: [0.0, 0.0, 0.0])
        for i in cal:                       # 用标定帧建场（基座=C2 输出）
            base = P[i] + m2.predict(XC2[i:i + 1])[0]
            b = affine_of(g)
            bv = np.array([np.array([1.0, base[0], base[1]]) @ b[0],
                           np.array([1.0, base[0], base[1]]) @ b[1]]) if b else np.zeros(2)
            upd(g, grid, base, L[i], base - bv, bv)
        for i in ev:
            base = P[i] + m2.predict(XC2[i:i + 1])[0]
            b = affine_of(g)
            bv = np.array([np.array([1.0, base[0], base[1]]) @ b[0],
                           np.array([1.0, base[0], base[1]]) @ b[1]]) if b else np.zeros(2)
            p1 = base - bv
            dd = lookup(grid, p1)
            corr = p1 - dd if dd is not None else p1
            errs.append(em(corr, L[i], str(paths[i])))
            if online_in_eval:              # 上界参考：评估段内也用标签更新
                upd(g, grid, base, L[i], p1, bv)
    return np.array(errs)


def run_field(tag, K, key_raw=False):
    """同协议下：纯 V1 误差场（键 p1 或 P，含仿射），无特征。"""
    errs = []
    for s in SESS:
        idx = sess[s]
        cut = int(len(idx) * 0.7)
        cal_pool, ev = idx[:cut], idx[cut:]
        if len(ev) < 30 or len(cal_pool) < 30:
            continue
        cal = (cal_pool[:min(K, len(cal_pool))] if tag == "head" else
               [cal_pool[t] for t in np.unique(
                   np.linspace(0, len(cal_pool) - 1, min(K, len(cal_pool))).astype(int))])
        g, grid = np.zeros(NZ), defaultdict(lambda: [0.0, 0.0, 0.0])
        for i in cal:
            b = affine_of(g)
            bv = np.array([np.array([1.0, P[i][0], P[i][1]]) @ b[0],
                           np.array([1.0, P[i][0], P[i][1]]) @ b[1]]) if b else np.zeros(2)
            upd(g, grid, P[i], L[i], P[i] - bv, bv)
        for i in ev:
            b = affine_of(g)
            bv = np.array([np.array([1.0, P[i][0], P[i][1]]) @ b[0],
                           np.array([1.0, P[i][0], P[i][1]]) @ b[1]]) if b else np.zeros(2)
            p1 = P[i] - bv
            key = P[i] if key_raw else p1
            dd = lookup(grid, key)
            corr = key - dd if dd is not None else p1
            errs.append(em(corr, L[i], str(paths[i])))
    return np.array(errs)


print("=" * 108)
print("固定协议：每会话前 70% 可标定，后 30% 评估（各方法同帧集，严格因果）")
print("=" * 108)
print(f"  {'标定':>7}{'K':>6}{'不修正':>9}{'V1场(键p1)':>17}{'V3场(键P)':>16}"
      f"{'C2(仅特征)':>17}{'C2+V1场':>17}{'+评估段在线':>16}")
for tag in ("head", "spread"):
    for K in (50, 100, 200, 500):
        r_cas = run(tag, K)
        r_on = run(tag, K, online_in_eval=True)
        # 不修正 / 纯 C2 的对照（同帧集）
        base_e, c2_e = [], []
        for s in SESS:
            idx = sess[s]
            cut = int(len(idx) * 0.7)
            cal_pool, ev = idx[:cut], idx[cut:]
            if len(ev) < 30 or len(cal_pool) < 30:
                continue
            cal = (cal_pool[:min(K, len(cal_pool))] if tag == "head" else
                   [cal_pool[t] for t in np.unique(
                       np.linspace(0, len(cal_pool) - 1, min(K, len(cal_pool))).astype(int))])
            m2 = Ridge().fit(XC2[cal], L[cal] - P[cal])
            for i in ev:
                base_e.append(em(P[i], L[i], str(paths[i])))
                c2_e.append(em(P[i] + m2.predict(XC2[i:i + 1])[0], L[i], str(paths[i])))
        base_e, c2_e = np.array(base_e), np.array(c2_e)
        r_v1 = run_field(tag, K)
        r_v3 = run_field(tag, K, key_raw=True)
        line = f"  {tag:>7}{K:>6}{base_e.mean():>9.3f}"
        for r in (r_v1, r_v3, c2_e, r_cas, r_on):
            line += f"{r.mean():>11.3f}({100 * (r.mean() / base_e.mean() - 1):>+4.1f}%)"
        print(line)
