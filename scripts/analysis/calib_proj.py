"""用「任务训练过的投影」特征重跑四条标定线对照（解释 C1 灾难 vs G5 记录的 −16%）。

关键差异：calibrator.py / calibrate_final.py 用的是
    [ proj_rgb_cls(骨干CLS) ⊕ proj_rgb_patch(patch均值) ]   （768 维，投影矩阵是训练学到的 ⇒ 与注视任务对齐 ✓）
而上一轮我 dump 的是**未投影**的原始骨干特征（norm≈50，未与任务对齐 ✗）⇒ 岭回归在其上无约束外推 ⇒ head 协议 +325% ✗✗。

投影是线性映射 ⇒ 可由 ckpt 里的 W/b 事后施加，无需重新前向。
变体：C1p 投影特征→注视点（= calibrate_final 形式）/ C2p 投影特征+预测→残差 / Dp=C2p∘B
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data1/xieliangcai/code/gazeformer-exps")
NPZ = ROOT / "out/rgbdgaze/analysis/feats_eye224_test.npz"
CKPT = ROOT / "out/rgbdgaze/ablation/checkpoints/eye224_mgpu/best.pt"
G, MIN, CL_OFF = 8, 8, 0.05
ALPHAS = np.logspace(-4, 6, 21)

sc = {}
with open(ROOT / "data/RGBDGaze/index/index.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        sc[r["rgb_path"]] = (float(r["screen_w"]) * float(r["cm_px_x"]),
                             float(r["screen_h"]) * float(r["cm_px_y"]))

st = torch.load(CKPT, map_location="cpu", weights_only=False)


def _get(name):
    for k in (name, f"module.{name}", f"model.{name}"):
        if k in st:
            return st[k].float().numpy()
    raise KeyError(name)


Wc, bc = _get("proj_rgb_cls.weight"), _get("proj_rgb_cls.bias")
Wp, bp = _get("proj_rgb_patch.weight"), _get("proj_rgb_patch.bias")

d = np.load(NPZ, allow_pickle=True)
F, P, L = d["feats"].astype(np.float64), d["pred"].astype(np.float64), d["label"].astype(np.float64)
paths, subj, fi = d["path"], d["subject"], d["frame_index"]
# 段：fused(384) | face_cls(384) | patch_mean(384) | eye(384)
nseg = F.shape[1] // 4
face_cls = F[:, nseg:2 * nseg]
patch_mean = F[:, 2 * nseg:3 * nseg]
Fp = np.concatenate([face_cls @ Wc.T + bc, patch_mean @ Wp.T + bp], 1)     # 768 维投影特征

by_subj = defaultdict(list)
for i in range(len(P)):
    by_subj[str(subj[i])].append(i)
for s in by_subj:
    by_subj[s].sort(key=lambda i: fi[i])
SUB = sorted(by_subj)
print(f"投影特征 {Fp.shape}（= calibrate_final 口径）；被试各帧数 "
      f"{[len(by_subj[s]) for s in SUB]}")


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
    def __init__(self, alphas=ALPHAS):
        self.alphas = alphas

    def fit(self, X, Y):
        self.mx_ = X.mean(0); self.sx_ = X.std(0) + 1e-6
        Xs = (X - self.mx_) / self.sx_
        self.my_ = Y.mean(0)
        Yc = Y - self.my_
        K = Xs @ Xs.T
        w, V = np.linalg.eigh(K)
        self.Xs_, self.V_, self.w_ = Xs, V, w
        V2 = V ** 2
        best, best_a = np.inf, None
        for a in self.alphas:
            h = V2 @ (w / (w + a))
            num = V @ ((V.T @ Yc) / (w + a)[:, None])
            r = (Yc - num) / np.clip(1.0 - h, 1e-6, None)[:, None]
            mse = float((r ** 2).mean())
            if mse < best:
                best, best_a = mse, a
        self.alpha_ = best_a
        self.coef_ = V @ ((V.T @ Yc) / (w + best_a)[:, None])
        return self

    def predict(self, X):
        return self.my_ + (((X - self.mx_) / self.sx_) @ self.Xs_.T) @ self.coef_


ALPHA = []


def run(mode, K):
    keys = ("A", "B", "C1p", "C2p", "Dp", "C2p_c", "Dp_c")
    acc = {k: [0.0, 0] for k in keys}
    thr = {k: [0, 0, 0] for k in keys}
    for s in SUB:
        idx = by_subj[s]; N = len(idx)
        if N < 400:
            continue
        cal = idx[:K] if mode == "head" else [
            idx[t] for t in np.unique(np.linspace(0, N - 1, K).astype(int))]
        calset = set(cal)
        ev = [i for i in idx if i not in calset]
        m1 = RidgeLOO().fit(Fp[cal], L[cal])
        g1 = m1.predict(Fp)
        ALPHA.append((mode, K, s, float(m1.alpha_)))
        Pb = np.linalg.norm(P - 0.5, axis=1, keepdims=True)
        X2 = np.concatenate([Fp, P, Pb], 1)
        m2 = RidgeLOO().fit(X2[cal], L[cal] - P[cal])
        g2 = P + m2.predict(X2)
        ALPHA.append((mode, K, s + "-res", round(float(m2.alpha_), 2)))
        m2c = RidgeLOO(alphas=[1e6]).fit(X2[cal], L[cal] - P[cal])
        g2c = P + m2c.predict(X2)
        gB = defaultdict(lambda: [0.0, 0.0, 0.0])
        gD = defaultdict(lambda: [0.0, 0.0, 0.0])
        gC = defaultdict(lambda: [0.0, 0.0, 0.0])
        for i in cal:
            cB = gB[cidx(P[i])]
            cB[0] += 1; cB[1] += P[i][0] - L[i][0]; cB[2] += P[i][1] - L[i][1]
            cD = gD[cidx(g2[i])]
            cD[0] += 1; cD[1] += g2[i][0] - L[i][0]; cD[2] += g2[i][1] - L[i][1]
            cC = gC[cidx(g2c[i])]
            cC[0] += 1; cC[1] += g2c[i][0] - L[i][0]; cC[2] += g2c[i][1] - L[i][1]
        for i in ev:
            out = {"A": P[i], "B": field_apply(gB, P[i], P[i]), "C1p": g1[i],
                   "C2p": g2[i], "Dp": field_apply(gD, g2[i], g2[i]),
                   "C2p_c": g2c[i], "Dp_c": field_apply(gC, g2c[i], g2c[i])}
            for k, v in out.items():
                e = em(v, L[i], str(paths[i]))
                acc[k][0] += e; acc[k][1] += 1
                thr[k][0] += e > 2; thr[k][1] += e > 3; thr[k][2] += e > 5
    return ({k: (acc[k][0] / acc[k][1], [t / acc[k][1] for t in thr[k]]) for k in keys})


print()
print("=" * 112)
print("【投影特征版】C1p=投影特征→注视点(=calibrate_final) / C2p=+预测→残差 / Dp=C2p∘B")
print("=" * 112)
print(f"  {'协议':>7}{'K':>6}{'A不标定':>9}{'B输出场':>17}{'C1p':>17}{'C2p':>17}{'Dp':>17}")
for mode in ("head", "spread"):
    for K in (100, 200, 600):
        r = run(mode, K)
        base = r["A"][0]
        line = f"  {mode:>7}{K:>6}{base:>9.3f}"
        for k in ("B", "C1p", "C2p", "Dp", "C2p_c", "Dp_c"):
            line += f"{r[k][0]:>10.3f}({100 * (r[k][0] / base - 1):>+4.1f}%)"
        print(line)
print()
print("  >3cm 占比（A / B / C1p / C2p / Dp）：")
for mode in ("head", "spread"):
    for K in (100, 200, 600):
        r = run(mode, K)
        print(f"    {mode:>7}{K:>6}   " +
              "   ".join(f"{r[k][1][1] * 100:>5.2f}%" for k in ("A", "B", "C1p", "C2p", "Dp", "C2p_c", "Dp_c")))
