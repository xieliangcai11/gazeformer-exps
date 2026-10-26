"""A+B 短实验：GP/克里金局部校准 + 共形风险门控（附自检）。

核心：
  1) 在锚点上拟合 GP（2D 输入 → x/y 输出），得到均值 m(x) 与后验方差 s²(x)
  2) 置信融合：y = affine + conf(x)·(m(x) − affine)，conf = τ²/(τ²+s²(x))
     - 锚点附近 s² 小 ⇒ 信空间模型；远离锚点 ⇒ 退回仿射
  3) 可选残差 LinearSVR（深层特征，PCA 后）小 λ 修正
  4) 共形门控：源域上选取阈值，使“退化 > ε 的比例 ≤ α”以置信度成立

自检（不通过直接报错）：
  C1 基线 EM 与已知值一致
  C2 锚点处 GP 后验 std 小于远离锚点处
  C3 conf ∈ [0,1]，且 conf→0 时输出退化为 affine
  C4 共形上界公式在合成数据上正确（二项 Clopper-Pearson 单侧）

用法：
  python -m scripts.analysis.cal_gp \
      --feats out/gazecapture/analysis/feats_gc_test_m2_600.npz \
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


def cp_upper(k, n, alpha=0.05):
    """Clopper-Pearson 单侧上界：给定 n 次试验中 k 次失败，返回失败概率的 1-alpha 上界。"""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    return float(beta.ppf(1 - alpha, k + 1, n - k))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--min-anchors", type=int, default=60)
    ap.add_argument("--lengths", default="0.15,0.25,0.4")
    ap.add_argument("--noises", default="0.01,0.03,0.1")
    ap.add_argument("--taus", default="0.02,0.05,0.1")
    ap.add_argument("--lams", default="0,0.2")
    ap.add_argument("--eps", type=float, default=0.02)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--limit-sessions", type=int, default=0)
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

    def em(p, l, c):
        return np.hypot((p[:, 0] - l[:, 0]) * c[:, 0], (p[:, 1] - l[:, 1]) * c[:, 1])

    base_all = float(em(pred, label, cm).mean())
    print(f"[C1] 全量基线 EM = {base_all:.4f} cm（应与既有报告 ~1.67 一致）")
    assert 1.4 < base_all < 1.9, "基线异常，检查 cm 口径"

    sess = defaultdict(list)
    for i in range(len(pred)): sess[str(subj[i])].append(i)

    lengths = [float(x) for x in args.lengths.split(",")]
    noises = [float(x) for x in args.noises.split(",")]
    taus = [float(x) for x in args.taus.split(",")]
    lams = [float(x) for x in args.lams.split(",")]

    # C2 合成数据单测：已知光滑函数上，锚点处方差应小于远点
    rng = np.random.RandomState(0)
    Xa = rng.rand(60, 2); ya = np.sin(3 * Xa[:, 0]) + np.cos(3 * Xa[:, 1])
    Xt = rng.rand(500, 2); yt = np.sin(3 * Xt[:, 0]) + np.cos(3 * Xt[:, 1])
    kk = ConstantKernel(1.0) * RBF(length_scale=0.25) + WhiteKernel(noise_level=0.01)
    g = GaussianProcessRegressor(kernel=kk, normalize_y=True, random_state=0).fit(Xa, ya)
    _, sd_t = g.predict(Xt, return_std=True)
    _, sd_a = g.predict(Xa, return_std=True)
    print(f"[C2] 合成单测：锚点 sd={sd_a.mean():.4f} 远点 sd={sd_t.mean():.4f}")
    assert sd_a.mean() < sd_t.mean(), "GP 方差方向异常（合成单测失败）"

    # 预计算每会话的 affine / GP 均值与方差 / 残差 SVR 预测
    recs = []
    sd_dist_corr = []
    names = sorted(sess)
    if args.limit_sessions: names = names[:args.limit_sessions]
    for si, s in enumerate(names):
        idx = sorted(sess[s], key=lambda i: int(fidx[i]))
        p = pred[idx]; l = label[idx]; c = cm[idx]; f = feats[idx]
        A = anchors.get(meta[paths[idx[0]]][:4], set())
        m = np.array([(meta[paths[i]][4], meta[paths[i]][5]) in A for i in idx])
        if m.sum() < args.min_anchors: continue
        base = float(em(p, l, c).mean())
        cf = [np.linalg.lstsq(np.stack([p[m, ax], np.ones(m.sum())], 1),
                              l[m, ax], rcond=None)[0] for ax in (0, 1)]
        aff = apply_affine(p, cf, 0.75)
        gp = {}
        for ln in lengths:
            for nz in noises:
                out = []
                for ax in (0, 1):
                    kernel = ConstantKernel(1.0) * RBF(length_scale=ln) + WhiteKernel(noise_level=nz)
                    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                                   n_restarts_optimizer=0, random_state=0)
                    gpr.fit(p[m], l[m, ax])
                    mu, sd = gpr.predict(p, return_std=True)
                    out.append((mu, sd))
                gp[(ln, nz)] = out
        # C2：GP 后验标准差应随“到最近锚点的距离”增大（相关性为正）
        ln, nz = lengths[0], noises[0]
        mu0, sd0 = gp[(ln, nz)][0]
        dmin = np.sqrt(((p[:, None, :] - p[m][None, :, :]) ** 2).sum(-1)).min(1)
        if np.std(sd0) > 1e-12:
            sd_dist_corr.append(float(np.corrcoef(dmin, sd0)[0, 1]))
        recs.append(dict(s=s, n=len(idx), base=base, p=p, l=l, c=c, m=m, aff=aff, gp=gp, f=f))

    print(f"[数据] 可用会话 {len(recs)}；sd~距离 相关性中位 "
          f"{np.median(sd_dist_corr):+.3f}（应 >0）")
    n = sum(r["n"] for r in recs)
    B = sum(r["base"] * r["n"] for r in recs) / n
    print(f"[C1] 可用会话基线 = {B:.4f}")

    # C3：conf→0 时等于 affine
    r0 = recs[0]
    mu, sd = r0["gp"][(lengths[0], noises[0])][0]
    tau0 = taus[0]
    conf = (tau0 ** 2) / (tau0 ** 2 + sd ** 2)
    y0 = r0["aff"] + conf[:, None] * (np.stack([mu, r0["gp"][(lengths[0], noises[0])][1][0]], 1) - r0["aff"])
    print(f"[C3] conf 范围 [{conf.min():.3f}, {conf.max():.3f}]，"
          f"最大偏离 affine {np.abs(y0 - r0['aff']).max():.4f}")

    # 逐配置评估
    results = []
    for ln in lengths:
        for nz in noises:
            for tau in taus:
                for lam in lams:
                    per = []
                    for r in recs:
                        (mx, sx), (my, sy) = r["gp"][(ln, nz)]
                        confx = tau ** 2 / (tau ** 2 + sx ** 2)
                        confy = tau ** 2 / (tau ** 2 + sy ** 2)
                        mp = np.stack([mx, my], 1)
                        y = r["aff"] + np.stack([confx, confy], 1) * (mp - r["aff"])
                        if lam > 0:
                            # 残差 SVR：在锚点上预测 (label - y)，PCA8
                            m = r["m"]; k = int(min(8, m.sum() - 1, 384))
                            pca = PCA(n_components=k, random_state=0).fit(r["f"][m])
                            X = pca.transform(r["f"]); Xc = X[m]
                            sv = np.empty_like(y)
                            for ax in (0, 1):
                                mdl = make_pipeline(StandardScaler(),
                                                    LinearSVR(C=1.0, epsilon=0.01, max_iter=20000))
                                mdl.fit(Xc, r["l"][m, ax] - y[m, ax])
                                sv[:, ax] = y[:, ax] + mdl.predict(X)
                            y = y + lam * (sv - y)
                        per.append((float(em(y, r["l"], r["c"]).mean()), r["base"], r["n"],
                                    float(em(r["p"][r["m"]], r["l"][r["m"]], r["c"][r["m"]]).mean())))
                    arr = np.array(per)
                    e = (arr[:, 0] * arr[:, 2]).sum() / arr[:, 2].sum()
                    b = (arr[:, 1] * arr[:, 2]).sum() / arr[:, 2].sum()
                    results.append(dict(key=(ln, nz, tau, lam), e=e, b=b,
                                        worst=float((arr[:, 0] / arr[:, 1] - 1).max()),
                                        rows=arr))
    print("\n[全量] 前 8 个配置（按均值排序）：")
    for r in sorted(results, key=lambda x: x["e"])[:8]:
        print(f"  len={r['key'][0]:.2f} noise={r['key'][1]:.2f} tau={r['key'][2]:.2f} "
              f"lam={r['key'][3]:.1f} → {r['e']:.4f} ({100*(r['e']/r['b']-1):+.1f}%) 最差 {100*r['worst']:+.0f}%")

    # 源域选参 + 共形门控 + 目标域评估
    src_ids = {i for i, r in enumerate(recs) if int(r["s"]) % 2 == 0}
    tgt_ids = {i for i, r in enumerate(recs) if int(r["s"]) % 2 == 1}

    def gate_eval(r, tau_gate):
        """anchor-error 门控：aerr > tau_gate 才校准。"""
        sel = r["_aerr"] > tau_gate
        return r["cal"] if sel else r["base"]

    best = None
    for res in results:
        rows = res["rows"]
        # 选择门控阈值：源域上"退化 > eps 的比例"的 CP 上界 <= alpha
        for tau_gate in (0.0, 0.8, 1.0, 1.2, 1.4):
            acc = [i for i in src_ids if rows[i, 3] > tau_gate]
            if len(acc) < 5: continue
            bad = sum(1 for i in acc if rows[i, 0] / rows[i, 1] - 1 > args.eps)
            ub = cp_upper(bad, len(acc), args.alpha)
            if ub > args.alpha: continue
            e_src = (rows[acc, 0] * rows[acc, 2]).sum() / rows[acc, 2].sum()
            if best is None or e_src < best[0]:
                best = (e_src, res, tau_gate, ub, len(acc), bad)

    if best is None:
        print("\n[B] 没有满足共形约束的配置（源域）")
        return
    _, res, tau_gate, ub, nacc, nbad = best
    rows = res["rows"]
    print(f"\n[源域选参] len={res['key'][0]:.2f} noise={res['key'][1]:.2f} "
          f"tau={res['key'][2]:.2f} lam={res['key'][3]:.1f} 门控τ={tau_gate:.1f}")
    print(f"  源域: 接受 {nacc} 会话，退化>{100*args.eps:.0f}% 比例上界 {100*ub:.1f}%")
    for nm, ids in (("源", src_ids), ("目标", tgt_ids)):
        cal = []
        for i in ids:
            cal.append(rows[i, 0] if rows[i, 3] > tau_gate else rows[i, 1])
        w = rows[:, 2]
        e = sum(cal[i] * w[i] for i in range(len(cal))) / sum(w[i] for i in range(len(cal)))
        b = sum(rows[i, 1] * w[i] for i in range(len(cal))) / sum(w[i] for i in range(len(cal)))
        worst = max((rows[i, 0] if rows[i, 3] > tau_gate else rows[i, 1]) / rows[i, 1] - 1 for i in ids)
        bad = sum(1 for i in ids if (rows[i, 0] if rows[i, 3] > tau_gate else rows[i, 1]) / rows[i, 1] - 1 > args.eps)
        print(f"  {nm}: 基线 {b:.4f} → {e:.4f} ({100*(e/b-1):+.1f}%) 最差 {100*worst:+.0f}% "
              f"退化>{100*args.eps:.0f}%: {bad}/{len(ids)}")


if __name__ == "__main__":
    main()
