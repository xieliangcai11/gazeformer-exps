"""MVO-SVR 校准（复现 MGazeNet 论文的“群智能 + SVR”思路，适配我们的模型）。

设置：
- 校准样本 = 锚点帧（展示坐标已知，无需用户真值）
- 特征 = 预测坐标的多项式 [x, y, x², y², xy]（"输出非线性映射"，不依赖深层特征）
- 模型 = 每会话每轴一个 RBF-SVR
- 超参 (C, γ, ε) 用 MVO（Multi-Verse Optimizer）在 3 折 CV 上搜索
- 评估 = 同一会话全部帧的 EM（逐帧 cm 与 dump 自洽）

对照：不校准 / 锚点仿射 α=0.75 / SVR 默认超参 / SVR+MVO

用法：
  python -m scripts.analysis.mvo_svr_calibration \
      --preds out/gazecapture/analysis/preds_gc_test_m2.csv \
      --index-csv data/GazeCapture/index/index.csv --limit-sessions 0
"""
import argparse
import sys
from collections import defaultdict

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path[:0] = ["src", "."]
from scripts.analysis.anchor_calibration import (  # noqa: E402
    load_preds, load_meta_and_anchors, derive_cm, apply_affine, sess_em)


def feats_poly(p):
    x, y = p[:, 0], p[:, 1]
    return np.stack([x, y, x * x, y * y, x * y], 1)


def mvo_search(X, y, n_universes=8, iters=12, seed=0, folds=3):
    """MVO 搜索 (C, gamma, epsilon)。返回 (best_params, best_cv_rmse)。"""
    lo = np.array([0.0, -4.0, 0.001])   # log10C, log10gamma, eps
    hi = np.array([3.0, 0.0, 0.2])
    rng = np.random.RandomState(seed)
    U = lo + rng.rand(n_universes, 3) * (hi - lo)
    # 固定折，避免每次评价的划分不同（否则适应度有噪声）
    idx = rng.permutation(len(X))
    folds_idx = np.array_split(idx, folds)

    def decode(u):
        return 10 ** u[0], 10 ** u[1], float(u[2])

    def fitness(u):
        C, g, eps = decode(u)
        errs = []
        for k in range(folds):
            te = folds_idx[k]
            tr = np.concatenate([folds_idx[j] for j in range(folds) if j != k])
            mdl = make_pipeline(StandardScaler(),
                                SVR(C=C, gamma=g, epsilon=eps))
            mdl.fit(X[tr], y[tr])
            errs.append(np.sqrt(np.mean((mdl.predict(X[te]) - y[te]) ** 2)))
        return float(np.mean(errs))

    fit = np.array([fitness(u) for u in U])
    best_u, best = U[np.argmin(fit)].copy(), float(fit.min())
    for it in range(iters):
        order = np.argsort(fit)
        U, fit = U[order], fit[order]
        WEP = 0.2 + (1.0 - 0.2) * it / max(1, iters)
        TDR = 1.0 - (it ** (1.0 / max(1, iters)))
        fmin, fmax = fit.min(), fit.max()
        NI = (fmax - fit) / (fmax - fmin + 1e-12)     # 越大越好
        NIc = np.cumsum(NI / (NI.sum() + 1e-12))
        for i in range(n_universes):
            if rng.rand() < NI[i]:
                j = int(np.searchsorted(NIc, rng.rand()))
                j = min(j, n_universes - 1)
                if j != i:
                    U[i] = U[j]
            if rng.rand() < WEP:
                sign = 1.0 if rng.rand() < 0.5 else -1.0
                U[i] = best_u + sign * TDR * ((hi - lo) * rng.rand(3) + lo)
            U[i] = np.clip(U[i], lo, hi)
            fit[i] = fitness(U[i])
            if fit[i] < best:
                best, best_u = fit[i], U[i].copy()
    return decode(best_u), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--min-anchors", type=int, default=40)
    ap.add_argument("--limit-sessions", type=int, default=0)
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--universes", type=int, default=8)
    args = ap.parse_args()

    rows = load_preds(args.preds)
    meta, anchors = load_meta_and_anchors(args.index_csv, [r["path"] for r in rows])
    tab, _ = derive_cm(rows, meta)
    sess = defaultdict(list)
    for r in rows:
        sess[r["subject"]].append(r)

    names = [s for s in sess if len([1 for r in sess[s] if
             (meta[r["path"]][4], meta[r["path"]][5]) in anchors.get(meta[r["path"]][:4], set())]) >= args.min_anchors]
    names = sorted(names)
    if args.limit_sessions:
        names = names[:args.limit_sessions]
    print(f"[data] 可用会话 {len(names)}；MVO={args.universes}宇宙×{args.iters}轮")

    out = defaultdict(list)
    params = []
    for si, s in enumerate(names):
        rs = sorted(sess[s], key=lambda r: int(r["frame_index"]))
        p = np.stack([r["_pred"] for r in rs])
        l = np.stack([r["_label"] for r in rs])
        c = np.stack([tab[meta[r["path"]][:4]] for r in rs])
        A = anchors.get(meta[rs[0]["path"]][:4], set())
        m = np.array([(meta[r["path"]][4], meta[r["path"]][5]) in A for r in rs])
        if m.sum() < args.min_anchors:
            continue
        base = float(sess_em(p, l, c).mean())
        # --- 仿射基线 ---
        cf = [np.linalg.lstsq(np.stack([p[m, ax], np.ones(m.sum())], 1),
                              l[m, ax], rcond=None)[0] for ax in (0, 1)]
        aff = float(sess_em(apply_affine(p, cf, 0.75), l, c).mean())

        # --- SVR ---
        Xc = feats_poly(p[m])
        pred_poly = feats_poly(p)
        preds = {}
        # 默认超参
        y0 = np.empty_like(p)
        for ax in (0, 1):
            mdl = make_pipeline(StandardScaler(), SVR())
            mdl.fit(Xc, l[m, ax])
            y0[:, ax] = mdl.predict(pred_poly)
        svr_def = float(sess_em(y0, l, c).mean())

        # MVO（两轴共用一个适应度：CV RMSE 平均）
        best = None
        for ax in (0, 1):
            (C, g, eps), cv = mvo_search(Xc, l[m, ax], args.universes, args.iters,
                                         seed=ax, folds=3)
            mdl = make_pipeline(StandardScaler(), SVR(C=C, gamma=g, epsilon=eps))
            mdl.fit(Xc, l[m, ax])
            if ax == 0:
                y1 = np.empty_like(p)
                params.append((C, g, eps, cv))
            y1[:, ax] = mdl.predict(pred_poly)
        svr_mvo = float(sess_em(y1, l, c).mean())

        out["base"].append((base, len(p)))
        out["affine"].append((aff, len(p)))
        out["svr_def"].append((svr_def, len(p)))
        out["svr_mvo"].append((svr_mvo, len(p)))
        if (si + 1) % 20 == 0:
            print(f"  {si+1}/{len(names)} ...")

    def agg(k):
        v = np.array(out[k])
        return float((v[:, 0] * v[:, 1]).sum() / v[:, 1].sum())

    base = agg("base")
    print(f"\n会话 {len(out['base'])} 帧 {int(sum(x[1] for x in out['base']))}")
    for k, nm in (("base", "不校准"), ("affine", "锚点仿射α.75"),
                  ("svr_def", "SVR默认"), ("svr_mvo", "SVR+MVO")):
        em = agg(k)
        rel = 100 * (em / base - 1)
        worst = 100 * max(x[0] / out["base"][i][0] - 1 for i, x in enumerate(out[k]))
        print(f"  {nm:<12} {em:.4f} ({rel:+.1f}%) 最差 {worst:+.0f}%")
    P = np.array(params)
    print(f"  MVO 参数中位: log10C={np.median(np.log10(P[:,0])):.2f} "
          f"log10γ={np.median(np.log10(P[:,1])):.2f} ε={np.median(P[:,2]):.3f}")


if __name__ == "__main__":
    main()
