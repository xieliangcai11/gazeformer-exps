"""在“深层特征”上复现 MVO-SVR 校准（对应 MGazeNet 论文口径）。

输入：scripts/analysis/dump_feats.py 导出的 npz（feats / pred / label / cm_x / cm_y /
      path / subject / frame_index）。
流程（逐会话）：
  1) 锚点帧（展示坐标已知）作为校准样本
  2) 特征 = 模型深层特征（1536 维）→ 仅在锚点上拟合 PCA 降到 k 维（无标签）
  3) 每轴一个 RBF-SVR；超参 (C, γ, ε) 由 MVO 在锚点 3 折 CV 上搜索
  4) 在该会话全部帧上评估 EM（cm 由 npz 提供，自洽）

对照：不校准 / 锚点仿射 α=0.75 / 深度特征 SVR 默认 / 深度特征 SVR+MVO

用法：
  python -m scripts.analysis.mvo_svr_feats \
      --feats out/gazecapture/analysis/feats_gc_test_m2.npz \
      --index-csv data/GazeCapture/index/index.csv --pca 16
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
    ap.add_argument("--min-anchors", type=int, default=40)
    ap.add_argument("--pca", type=int, default=16)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--universes", type=int, default=6)
    ap.add_argument("--limit-sessions", type=int, default=0)
    args = ap.parse_args()

    z = np.load(args.feats, allow_pickle=True)
    feats = z["feats"]; pred = z["pred"]; label = z["label"]
    err_cm = np.stack([z["cm_x"], z["cm_y"]], 1)   # 注意：这是“误差(cm)”，不是尺度
    paths = z["path"]; subj = z["subject"]; fidx = z["frame_index"]
    print(f"[load] 帧 {len(pred)} 特征 {feats.shape} 被试 {len(set(subj.tolist()))}")

    meta, anchors = load_meta_and_anchors(args.index_csv, paths.tolist())

    # 由 err/(pred-label) 反推“每归一化单位对应的 cm”，按 (device,orientation,screen) 取中位
    from collections import defaultdict as _dd
    pairs = _dd(lambda: [[], []])
    d = pred - label
    for i in range(len(pred)):
        key = meta[paths[i]][:4]
        if abs(d[i, 0]) > 0.01:
            pairs[key][0].append(err_cm[i, 0] / d[i, 0])
        if abs(d[i, 1]) > 0.01:
            pairs[key][1].append(err_cm[i, 1] / d[i, 1])
    tab = {k: (float(np.median(v[0])), float(np.median(v[1]))) for k, v in pairs.items()}
    cm = np.stack([np.array([tab[meta[p][:4]][0] for p in paths]),
                   np.array([tab[meta[p][:4]][1] for p in paths])], 1)
    em0 = np.hypot(d[:, 0] * cm[:, 0], d[:, 1] * cm[:, 1])
    print(f"[check] 基线 EM={em0.mean():.4f} cm；cm 中位=({np.median(cm[:,0]):.2f},{np.median(cm[:,1]):.2f})")
    sess = defaultdict(list)
    for i in range(len(paths)):
        sess[str(subj[i])].append(i)
    names = sorted([s for s in sess if sum(
        1 for i in sess[s] if (meta[paths[i]][4], meta[paths[i]][5]) in anchors.get(meta[paths[i]][:4], set())) >= args.min_anchors])
    if args.limit_sessions:
        names = names[:args.limit_sessions]
    print(f"[data] 可用会话 {len(names)}")

    def em_of(p, l, c):
        return np.hypot((p[:, 0] - l[:, 0]) * c[:, 0], (p[:, 1] - l[:, 1]) * c[:, 1])

    agg = defaultdict(list)
    for si, s in enumerate(names):
        idx = sorted(sess[s], key=lambda i: int(fidx[i]))
        p = pred[idx].astype(np.float64); l = label[idx].astype(np.float64); c = cm[idx].astype(np.float64)
        f = feats[idx].astype(np.float64)
        A = anchors.get(meta[paths[idx[0]]][:4], set())
        m = np.array([(meta[paths[i]][4], meta[paths[i]][5]) in A for i in idx])
        if m.sum() < args.min_anchors:
            continue
        base = float(em_of(p, l, c).mean())
        cf = [np.linalg.lstsq(np.stack([p[m, ax], np.ones(m.sum())], 1),
                              l[m, ax], rcond=None)[0] for ax in (0, 1)]
        aff = float(em_of(apply_affine(p, cf, 0.75), l, c).mean())

        # PCA on anchor features only (无标签)
        k = int(min(args.pca, m.sum() - 1))
        pca = PCA(n_components=k, random_state=0).fit(f[m])
        Xa = pca.transform(f)
        Xc = Xa[m]

        def svr_predict(params=None):
            y = np.empty_like(p)
            for ax in (0, 1):
                if params is None:
                    mdl = make_pipeline(StandardScaler(), SVR())
                else:
                    C, g, eps = params[ax]
                    mdl = make_pipeline(StandardScaler(), SVR(C=C, gamma=g, epsilon=eps))
                mdl.fit(Xc, l[m, ax]); y[:, ax] = mdl.predict(Xa)
            return y

        svr_def = float(em_of(svr_predict(None), l, c).mean())
        best_params = []
        for ax in (0, 1):
            best_params.append(mvo_search(Xc, l[m, ax], args.universes, args.iters,
                                          seed=si * 2 + ax, folds=3)[0])
        svr_mvo = float(em_of(svr_predict(best_params), l, c).mean())

        agg["base"].append((base, len(idx)))
        agg["affine"].append((aff, len(idx)))
        agg["svr_def"].append((svr_def, len(idx)))
        agg["svr_mvo"].append((svr_mvo, len(idx)))
        if (si + 1) % 20 == 0:
            print(f"  {si+1}/{len(names)} ...")

    def A(k):
        v = np.array(agg[k]); return float((v[:, 0] * v[:, 1]).sum() / v[:, 1].sum())

    base = A("base")
    print(f"\n会话 {len(agg['base'])} 帧 {int(sum(x[1] for x in agg['base']))}")
    for k, nm in (("base", "不校准"), ("affine", "锚点仿射α.75"),
                  ("svr_def", "深层SVR默认"), ("svr_mvo", "深层SVR+MVO")):
        e = A(k)
        worst = 100 * max(x[0] / agg["base"][i][0] - 1 for i, x in enumerate(agg[k]))
        print(f"  {nm:<13} {e:.4f} ({100*(e/base-1):+.1f}%) 最差 {worst:+.0f}%")


if __name__ == "__main__":
    main()
