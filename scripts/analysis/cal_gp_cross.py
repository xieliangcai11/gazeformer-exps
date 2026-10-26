"""GP 局部校准 + 共形门控的跨集验证：源域（val）选参/定阈值 → 目标域（test）一次性评估。

用法：
  python -m scripts.analysis.cal_gp_cross \
    --source out/gazecapture/analysis/feats_gc_val_m2_600.npz \
    --target out/gazecapture/analysis/feats_gc_test_m2_600.npz \
    --index-csv data/GazeCapture/index/index.csv
"""
import argparse
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import beta
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

sys.path[:0] = ["src", "."]
from scripts.analysis.anchor_calibration import (  # noqa: E402
    load_meta_and_anchors, apply_affine)


def cp_ub(k, n, alpha):
    if n == 0 or k >= n:
        return 1.0
    return float(beta.ppf(1 - alpha, k + 1, n - k))


def load_session_data(feats_path, meta, anchors):
    z = np.load(feats_path, allow_pickle=True)
    feats = z["feats"].astype(np.float32)[:, :384]
    pred = z["pred"].astype(np.float64); label = z["label"].astype(np.float64)
    err = np.stack([z["cm_x"], z["cm_y"]], 1)
    paths = z["path"]; subj = z["subject"]; fidx = z["frame_index"]

    pairs = defaultdict(lambda: [[], []])
    d = pred - label
    for i in range(len(pred)):
        k = meta[paths[i]][:4]
        if abs(d[i, 0]) > 0.01: pairs[k][0].append(err[i, 0] / d[i, 0])
        if abs(d[i, 1]) > 0.01: pairs[k][1].append(err[i, 1] / d[i, 1])
    tab = {k: (float(np.median(v[0])), float(np.median(v[1]))) for k, v in pairs.items()}
    cm = np.stack([np.array([tab[meta[p][:4]][0] for p in paths]),
                   np.array([tab[meta[p][:4]][1] for p in paths])], 1)

    def em(p, l, c):
        return np.hypot((p[:, 0] - l[:, 0]) * c[:, 0], (p[:, 1] - l[:, 1]) * c[:, 1])

    sess = defaultdict(list)
    for i in range(len(pred)): sess[str(subj[i])].append(i)

    CFG = [(0.15, 0.1), (0.15, 0.03), (0.25, 0.1), (0.25, 0.03)]
    rows = []
    for s in sorted(sess):
        idx = sorted(sess[s], key=lambda i: int(fidx[i]))
        p = pred[idx]; l = label[idx]; c = cm[idx]; f = feats[idx]
        A = anchors.get(meta[paths[idx[0]]][:4], set())
        m = np.array([(meta[paths[i]][4], meta[paths[i]][5]) in A for i in idx])
        if m.sum() < 60: continue
        base = float(em(p, l, c).mean())
        aerr = max(float(em(p[m], l[m], c[m]).mean()), 1e-6)
        cf = [np.linalg.lstsq(np.stack([p[m, ax], np.ones(m.sum())], 1),
                              l[m, ax], rcond=None)[0] for ax in (0, 1)]
        aff = apply_affine(p, cf, 0.75)
        k = int(min(8, m.sum() - 1, 384))
        pca = PCA(n_components=k, random_state=0).fit(f[m])
        X = pca.transform(f); Xc = X[m]
        cals, ratios = {}, {}
        for ln, nz in CFG:
            out = []
            for ax in (0, 1):
                g = GaussianProcessRegressor(
                    kernel=ConstantKernel(1.0) * RBF(length_scale=ln) + WhiteKernel(noise_level=nz),
                    normalize_y=True, random_state=0).fit(p[m], l[m, ax])
                out.append(g.predict(p, return_std=True))
            (mx, sx), (my, sy) = out
            conf = np.stack([0.05 ** 2 / (0.05 ** 2 + sx ** 2),
                             0.05 ** 2 / (0.05 ** 2 + sy ** 2)], 1)
            y = aff + conf * (np.stack([mx, my], 1) - aff)
            sv = np.empty_like(y)
            for ax in (0, 1):
                mdl = make_pipeline(StandardScaler(),
                                    LinearSVR(C=1.0, epsilon=0.01, max_iter=20000))
                mdl.fit(Xc, l[m, ax] - y[m, ax])
                sv[:, ax] = y[:, ax] + mdl.predict(X)
            y = y + 0.2 * (sv - y)
            cals[(ln, nz)] = float(em(y, l, c).mean())
            ratios[(ln, nz)] = float(np.linalg.norm((y - p) * c, axis=1).mean()) / aerr
        rows.append(dict(s=s, n=len(idx), base=base, cals=cals, ratios=ratios))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--alpha", type=float, default=0.10)
    args = ap.parse_args()

    # 索引只扫一次
    import numpy as np  # noqa: F811
    meta_paths = []
    for f in (args.source, args.target):
        z = np.load(f, allow_pickle=True)
        meta_paths += z["path"].tolist()
    meta, anchors = load_meta_and_anchors(args.index_csv, meta_paths)

    src = load_session_data(args.source, meta, anchors)
    tgt = load_session_data(args.target, meta, anchors)
    print(f"[data] 源 {len(src)} 会话 / 目标 {len(tgt)} 会话")

    CFG = list(src[0]["cals"].keys())
    # 1) 源域选超参（均值最优，约束最差 ≤15%）
    best = None
    for key in CFG:
        e = sum(r["cals"][key] * r["n"] for r in src) / sum(r["n"] for r in src)
        b = sum(r["base"] * r["n"] for r in src) / sum(r["n"] for r in src)
        w = max(r["cals"][key] / r["base"] - 1 for r in src)
        if w <= 0.15 and (best is None or e < best[0]):
            best = (e, b, w, key)
    print(f"[源域] 选超参 {best[3]}：{100*(best[0]/best[1]-1):+.1f}%，最差 {100*best[2]:+.0f}%")
    key = best[3]

    # 2) 源域定门控阈值（CP 上界 ≤5%，alpha 可调）
    cands = []
    for t in np.arange(0.1, 0.9, 0.05):
        acc = [r for r in src if r["ratios"][key] > t]
        if len(acc) < 10: continue
        bad = sum(1 for r in acc if r["cals"][key] / r["base"] - 1 > args.eps)
        ub = cp_ub(bad, len(acc), args.alpha)
        cands.append((ub, -len(acc), float(t), len(acc), bad))
    if not cands:
        print("[源域] 无可行阈值"); return
    cands.sort()
    ub, negacc, t, nacc, bad = cands[0]
    print(f"[源域] 门控：τ={t:.2f} 接受 {nacc}，退化 {bad}，"
          f"{100*(1-args.alpha):.0f}% 置信 CP 上界 {100*ub:.1f}%")

    # 3) 目标域一次性评估
    for nm, rs in (("源", src), ("目标", tgt)):
        den = sum(r["n"] for r in rs)
        e = sum((r["cals"][key] if r["ratios"][key] > t else r["base"]) * r["n"] for r in rs) / den
        b = sum(r["base"] * r["n"] for r in rs) / den
        w = max((r["cals"][key] if r["ratios"][key] > t else r["base"]) / r["base"] - 1 for r in rs)
        bad = sum(1 for r in rs if ((r["cals"][key] if r["ratios"][key] > t else r["base"]) / r["base"] - 1) > args.eps)
        acc = sum(1 for r in rs if r["ratios"][key] > t)
        print(f"  {nm}: 基线 {b:.4f} → {e:.4f} ({100*(e/b-1):+.1f}%) 最差 {100*w:+.0f}% "
              f"退化>{100*args.eps:.0f}%: {bad}/{len(rs)}（接受校准 {acc}）")


if __name__ == "__main__":
    main()
