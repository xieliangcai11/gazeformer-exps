"""校准的方差控制对照：Ridge / LinearSVR / 保守 RBF + 修正裁剪 + 分歧收缩。

数据：dump_feats 的 npz（含 fused CLS 384 维、pred、label、cm）
流程（逐会话）：
  affine = 锚点仿射(α=0.75)
  各模型 = 在锚点上拟合（特征 PCA 到 k 维），预测全帧
  融合 = affine + λ·(model - affine)，并可选：
        · clip：把修正量按轴裁剪到 ±clip_cm
        · disagree：逐帧 λ_i = clip(tau_cm/||d_i||_cm, 0, λmax)（分歧大 ⇒ 少信）
输出均值/最差；并在源域（被试奇偶）选 λ 后评估目标域。

用法：
  python -m scripts.analysis.cal_variance_control \
    --feats out/gazecapture/analysis/feats_gc_test_m2_600.npz \
    --index-csv data/GazeCapture/index/index.csv --pca 8
"""
import argparse
import sys
from collections import defaultdict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR, LinearSVR

sys.path[:0] = ["src", "."]
from scripts.analysis.anchor_calibration import (  # noqa: E402
    load_meta_and_anchors, apply_affine)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--pca", type=int, default=8)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--clip-cm", type=float, default=2.0)
    ap.add_argument("--tau-cm", type=float, default=2.0)
    ap.add_argument("--lamax", type=float, default=0.5)
    ap.add_argument("--min-anchors", type=int, default=60)
    args = ap.parse_args()

    z = np.load(args.feats, allow_pickle=True)
    feats = z["feats"].astype(np.float32)[:, :384]
    pred = z["pred"].astype(np.float64); label = z["label"].astype(np.float64)
    err = np.stack([z["cm_x"], z["cm_y"]], 1)
    paths = z["path"]; subj = z["subject"]; fidx = z["frame_index"]
    meta, anchors = load_meta_and_anchors(args.index_csv, paths.tolist())

    pairs = defaultdict(lambda: [[], []])
    d = pred - label
    for i in range(len(pred)):
        k = meta[paths[i]][:4]
        if abs(d[i, 0]) > 0.01: pairs[k][0].append(err[i, 0] / d[i, 0])
        if abs(d[i, 1]) > 0.01: pairs[k][1].append(err[i, 1] / d[i, 1])
    tab = {k: (float(np.median(v[0])), float(np.median(v[1]))) for k, v in pairs.items()}
    cm = np.stack([np.array([tab[meta[p][:4]][0] for p in paths]),
                   np.array([tab[meta[p][:4]][1] for p in paths])], 1)

    sess = defaultdict(list)
    for i in range(len(pred)): sess[str(subj[i])].append(i)

    def em(p, l, c):
        return np.hypot((p[:, 0] - l[:, 0]) * c[:, 0], (p[:, 1] - l[:, 1]) * c[:, 1])

    MODELS = {
        "ridge": lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "linsvr": lambda: make_pipeline(StandardScaler(), LinearSVR(C=1.0, epsilon=0.01, max_iter=5000)),
        "rbf_cons": lambda: make_pipeline(StandardScaler(), SVR(C=10.0, gamma=0.01, epsilon=0.1)),
        "rbf_scale": lambda: make_pipeline(StandardScaler(), SVR()),
    }
    recs = []
    for si, s in enumerate(sorted(sess)):
        idx = sorted(sess[s], key=lambda i: int(fidx[i]))
        p = pred[idx]; l = label[idx]; c = cm[idx]; f = feats[idx]
        A = anchors.get(meta[paths[idx[0]]][:4], set())
        m = np.array([(meta[paths[i]][4], meta[paths[i]][5]) in A for i in idx])
        if m.sum() < args.min_anchors: continue
        base = float(em(p, l, c).mean())
        cf = [np.linalg.lstsq(np.stack([p[m, ax], np.ones(m.sum())], 1),
                              l[m, ax], rcond=None)[0] for ax in (0, 1)]
        aff = apply_affine(p, cf, 0.75)
        k = int(min(args.pca, m.sum() - 1, 384))
        pca = PCA(n_components=k, random_state=0).fit(f[m])
        X = pca.transform(f); Xc = X[m]
        r = dict(s=s, n=len(idx), base=base, aff=float(em(aff, l, c).mean()))
        for name, ctor in MODELS.items():
            y = np.empty_like(p)
            for ax in (0, 1):
                mdl = ctor(); mdl.fit(Xc, l[m, ax]); y[:, ax] = mdl.predict(X)
            yl = aff + args.lam * (y - aff)
            r[f"{name}_lam"] = float(em(yl, l, c).mean())
            # 修正裁剪（按 cm）
            dcm = (y - aff) * c
            dcl = np.clip(dcm, -args.clip_cm, args.clip_cm) / c
            r[f"{name}_clip"] = float(em(aff + dcl, l, c).mean())
            # 分歧收缩：逐帧 λ_i
            dmag = np.hypot(dcm[:, 0], dcm[:, 1])
            li = np.clip(args.tau_cm / (dmag + 1e-6), 0, args.lamax)[:, None]
            r[f"{name}_dis"] = float(em(aff + li * (y - aff), l, c).mean())
        recs.append(r)
        if (si + 1) % 40 == 0: print(f"  {si+1} ...")

    n = sum(r["n"] for r in recs)
    base = sum(r["base"] * r["n"] for r in recs) / n
    print(f"\n会话 {len(recs)} 帧 {n} 基线 {base:.4f}")
    keys = ["aff"] + [f"{m}_{k}" for m in MODELS for k in ("lam", "clip", "dis")]
    for key in keys:
        e = sum(r[key] * r["n"] for r in recs) / n
        w = max(r[key] / r["base"] - 1 for r in recs)
        print(f"  {key:<16} {e:.4f} ({100*(e/base-1):+.1f}%) 最差 {100*w:+.0f}%")


if __name__ == "__main__":
    main()
