"""全量锚点 + 深层特征 MVO-SVR 校准（对齐 MGazeNet 论文设置）。

与 mvo_svr_feats.py 的差别：
- 使用**全量**锚点帧（每会话约 250 帧，而非采样后的 ~60）
- 特征可选 "fused"（最后融合 CLS，384 维，对应论文的 256 维最后层）
- MVO 默认 population=30 / iterations=10（与论文一致）
- 先拟合锚点仿射，再按 λ 混入 SVR 修正；λ 在源域被试上选，目标域只评一次

用法：
  python -m scripts.analysis.mvo_svr_full \
      --feats out/gazecapture/analysis/feats_gc_test_m2_full.npz \
      --index-csv data/GazeCapture/index/index.csv --feat-mode fused --pca 32
"""
import argparse
import sys
from collections import defaultdict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

sys.path[:0] = ["src", "."]
from scripts.analysis.anchor_calibration import (  # noqa: E402
    load_meta_and_anchors, apply_affine)
from scripts.analysis.mvo_svr_calibration import mvo_search  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--feat-mode", choices=["fused", "concat"], default="fused")
    ap.add_argument("--pca", type=int, default=32, help="0=不降维")
    ap.add_argument("--pop", type=int, default=30)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--min-anchors", type=int, default=60)
    ap.add_argument("--limit-sessions", type=int, default=0)
    ap.add_argument("--lams", default="0,0.2,0.3,0.5,0.7,1.0")
    ap.add_argument("--max-worst", type=float, default=0.05,
                    help="源域选 λ 时要求的最差会话上限（相对基线）")
    args = ap.parse_args()

    z = np.load(args.feats, allow_pickle=True)
    feats = z["feats"].astype(np.float32)
    pred = z["pred"].astype(np.float64); label = z["label"].astype(np.float64)
    err = np.stack([z["cm_x"], z["cm_y"]], 1)
    paths = z["path"]; subj = z["subject"]; fidx = z["frame_index"]
    if args.feat_mode == "fused":
        feats = feats[:, :384]          # 融合 CLS（head 输入）
    print(f"[load] 帧 {len(pred)} 特征 {feats.shape} 被试 {len(set(subj.tolist()))}")

    meta, anchors = load_meta_and_anchors(args.index_csv, paths.tolist())

    # cm：err/(pred-label) 按 (device,orientation,screen) 取中位
    pairs = defaultdict(lambda: [[], []])
    d = pred - label
    for i in range(len(pred)):
        k = meta[paths[i]][:4]
        if abs(d[i, 0]) > 0.01:
            pairs[k][0].append(err[i, 0] / d[i, 0])
        if abs(d[i, 1]) > 0.01:
            pairs[k][1].append(err[i, 1] / d[i, 1])
    tab = {k: (float(np.median(v[0])), float(np.median(v[1]))) for k, v in pairs.items()}
    cm = np.stack([np.array([tab[meta[p][:4]][0] for p in paths]),
                   np.array([tab[meta[p][:4]][1] for p in paths])], 1)
    e0 = np.hypot(d[:, 0] * cm[:, 0], d[:, 1] * cm[:, 1])
    print(f"[check] 基线 EM={e0.mean():.4f} cm")

    sess = defaultdict(list)
    for i in range(len(pred)):
        sess[str(subj[i])].append(i)
    names = sorted(sess)
    if args.limit_sessions:
        names = names[:args.limit_sessions]

    def em(p, l, c):
        return np.hypot((p[:, 0] - l[:, 0]) * c[:, 0], (p[:, 1] - l[:, 1]) * c[:, 1])

    lams = [float(x) for x in args.lams.split(",")]
    recs = []
    for si, s in enumerate(names):
        idx = sorted(sess[s], key=lambda i: int(fidx[i]))
        p = pred[idx]; l = label[idx]; c = cm[idx]; f = feats[idx]
        A = anchors.get(meta[paths[idx[0]]][:4], set())
        m = np.array([(meta[paths[i]][4], meta[paths[i]][5]) in A for i in idx])
        if m.sum() < args.min_anchors:
            continue
        base = float(em(p, l, c).mean())
        cf = [np.linalg.lstsq(np.stack([p[m, ax], np.ones(m.sum())], 1),
                              l[m, ax], rcond=None)[0] for ax in (0, 1)]
        aff = apply_affine(p, cf, 0.75)

        fm = f[m]
        if args.pca > 0:
            k = int(min(args.pca, m.sum() - 1, fm.shape[1]))
            pca = PCA(n_components=k, random_state=0).fit(fm)
            X = pca.transform(f); Xc = X[m]
        else:
            X = f; Xc = fm
        sv = np.empty_like(p)
        for ax in (0, 1):
            (C, g, eps), _ = mvo_search(Xc, l[m, ax], args.pop, args.iters,
                                        seed=si * 2 + ax, folds=3)
            mdl = make_pipeline(StandardScaler(), SVR(C=C, gamma=g, epsilon=eps))
            mdl.fit(Xc, l[m, ax]); sv[:, ax] = mdl.predict(X)

        ems = {f"lam{lam}": float(em(aff + lam * (sv - aff), l, c).mean()) for lam in lams}
        ems["svr"] = float(em(sv, l, c).mean())
        ems["aff"] = float(em(aff, l, c).mean())
        ems["base"] = base
        # 样本内（锚点帧上的误差），用于对照论文口径
        ems["insample_svr"] = float(em(sv[m], l[m], c[m]).mean())
        recs.append(dict(s=s, n=len(idx), **ems))
        if (si + 1) % 30 == 0:
            print(f"  {si+1}/{len(names)} ...")

    n_all = sum(r["n"] for r in recs)
    base_all = sum(r["base"] * r["n"] for r in recs) / n_all
    print(f"\n会话 {len(recs)} 帧 {n_all} 基线 {base_all:.4f}")

    def agg(key):
        return sum(r[key] * r["n"] for r in recs) / n_all

    def worst(key):
        return max(r[key] / r["base"] - 1 for r in recs)

    print("[全量会话]")
    for key in ("aff", "svr", "insample_svr") + tuple(f"lam{x}" for x in lams):
        print(f"  {str(key):<12} {agg(key):.4f} ({100*(agg(key)/base_all-1):+.1f}%) "
              f"最差 {100*worst(key):+.0f}%")

    src = [r for r in recs if int(r["s"]) % 2 == 0]
    tgt = [r for r in recs if int(r["s"]) % 2 == 1]
    if len(src) >= 5 and len(tgt) >= 5:
        def ev(rs, key):
            den = sum(r["n"] for r in rs)
            e = sum(r[key] * r["n"] for r in rs) / den
            b = sum(r["base"] * r["n"] for r in rs) / den
            w = max(r[key] / r["base"] - 1 for r in rs)
            return e, b, w

        best = None
        for lam in lams:
            e, b, w = ev(src, f"lam{lam}")
            if w <= args.max_worst and (best is None or e < best[0]):
                best = (e, w, lam)
        print(f"\n[源域选 λ] 约束最差≤{100*args.max_worst:.0f}% ⇒ {best}")
        if best:
            lam = best[2]
            for nm, rs in (("源", src), ("目标", tgt)):
                e, b, w = ev(rs, f"lam{lam}")
                print(f"  {nm}: 基线 {b:.4f} → {e:.4f} ({100*(e/b-1):+.1f}%) 最差 {100*w:+.0f}%")


if __name__ == "__main__":
    main()
