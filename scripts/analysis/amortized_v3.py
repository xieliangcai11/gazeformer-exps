"""B1 扩容版：用 GC train 子集训练摊销模型，val 验证迁移，test 最终评估。

- 先验：从 val（模型未训练过的被试）估计每朝向仿射参数的均值/协方差
- B1：在 train 子集（模型训练过的被试）上生成 episodes，MLP 学 (K 个校准样本 → θ)
- 验证：val 上检查迁移；最终：test
- 对照：先验 MAP（闭式）、无先验 LS

用法：
  python -m scripts.analysis.amortized_v3 \
    --train out/gazecapture/analysis/feats_gc_train300.npz \
    --val out/gazecapture/analysis/feats_gc_val_m2_full.npz \
    --test out/gazecapture/analysis/feats_gc_test_m2_full.npz \
    --index-csv data/GazeCapture/index/index.csv --K 52 --seeds 5
"""
import argparse
import csv
import sys
from collections import defaultdict

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.path[:0] = ["src", "."]
from scripts.analysis.anchor_calibration import apply_affine  # noqa: E402


def load_npz(f):
    z = np.load(f, allow_pickle=True)
    return dict(pred=z["pred"].astype(np.float64), label=z["label"].astype(np.float64),
                err=np.stack([z["cm_x"], z["cm_y"]], 1),
                subject=z["subject"], fidx=z["frame_index"], path=z["path"])


def cm_of(p, l, e):
    d = p - l
    rx = [e[i, 0] / d[i, 0] for i in range(len(p)) if abs(d[i, 0]) > 0.01]
    ry = [e[i, 1] / d[i, 1] for i in range(len(p)) if abs(d[i, 1]) > 0.01]
    return np.tile([float(np.median(rx)) if rx else 1.0,
                    float(np.median(ry)) if ry else 1.0], (len(p), 1))


def em(p, l, c):
    return np.hypot((p[:, 0] - l[:, 0]) * c[:, 0], (p[:, 1] - l[:, 1]) * c[:, 1])


def fit_theta(x, u, y, v, lam=0.0, mu=None):
    out = []
    for j, (a, b) in enumerate(((x, u), (y, v))):
        X = np.stack([a, np.ones(len(a))], 1)
        A = X.T @ X + (lam * np.eye(2) if lam > 0 else 0)
        rhs = X.T @ b + (lam * mu[j] if (lam > 0 and mu is not None) else 0)
        out.append(np.linalg.solve(A, rhs))
    return out


def sample_coverage(pred, k, rng, nbin=3):
    cells = defaultdict(list)
    for j in range(len(pred)):
        cells[(min(nbin - 1, int(pred[j, 0] * nbin)),
               min(nbin - 1, int(pred[j, 1] * nbin)))].append(j)
    per = max(1, k // max(1, len(cells)))
    sel = []
    for v in cells.values():
        v = list(v); rng.shuffle(v); sel += v[:per]
    if len(sel) < k:
        rest = [j for j in range(len(pred)) if j not in set(sel)]
        sel += list(rng.choice(rest, min(k - len(sel), len(rest)), replace=False))
    return sorted(set(sel))[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--K", type=int, default=52)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    tr, va, te = load_npz(args.train), load_npz(args.val), load_npz(args.test)
    want = set(tr["path"].tolist()) | set(va["path"].tolist()) | set(te["path"].tolist())
    ori = {}
    with open(args.index_csv, encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        pcol = "rgb_path" if "rgb_path" in rd.fieldnames else "frame_path"
        for r in rd:
            if r[pcol] in want:
                ori[r[pcol]] = int(float(r["orientation"]))

    def group(d):
        s = defaultdict(list)
        for i in range(len(d["pred"])):
            s[(str(d["subject"][i]), ori.get(d["path"][i], 1))].append(i)
        return {k: sorted(v, key=lambda i: int(d["fidx"][i])) for k, v in s.items()}

    gtr, gva, gte = group(tr), group(va), group(te)
    print(f"[data] train 会话朝向 {len(gtr)} / val {len(gva)} / test {len(gte)}")

    # 先验：val（干净）
    per = defaultdict(list)
    for (sub, o), idx in gva.items():
        if len(idx) < 100: continue
        p, l = va["pred"][idx], va["label"][idx]
        th = fit_theta(p[:, 0], l[:, 0], p[:, 1], l[:, 1])
        per[o].append([th[0][0], th[1][0], th[0][1], th[1][1]])
    pri = {o: (np.mean(np.array(v), 0), np.cov(np.array(v).T) + 1e-6 * np.eye(4))
           for o, v in per.items()}
    print(f"[prior] 朝向 {sorted(pri)}，样本数 {[len(v) for v in per.values()]}")

    kper = max(2, int(np.ceil(args.K / 4)))

    def feats_of(p, l, sel, o):
        mu = pri.get(o, pri[list(pri)[0]])[0]
        return np.concatenate([p[sel].mean(0), l[sel].mean(0), (p[sel] - l[sel]).mean(0),
                               p[sel].std(0), l[sel].std(0), [len(sel)], mu])

    # 训练 B1：train 子集 episodes
    X, Y = [], []
    for (sub, o), idx in gtr.items():
        if len(idx) < 200: continue
        p, l = tr["pred"][idx], tr["label"][idx]
        th = fit_theta(p[:, 0], l[:, 0], p[:, 1], l[:, 1])
        y = np.array([th[0][0], th[1][0], th[0][1], th[1][1]])
        for t in range(20):
            rng = np.random.RandomState(abs(hash((sub, o, t))) % 2 ** 31)
            sel = sample_coverage(p, kper, rng)
            X.append(feats_of(p, l, sel, o)); Y.append(y)
    X = np.array(X); Y = np.array(Y)
    print(f"[B1] 训练样本 {len(X)}（{len(gtr)} 个会话朝向）")
    sc = StandardScaler().fit(X)
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-2, max_iter=3000,
                       random_state=0).fit(sc.transform(X), Y)

    def evaluate(d, gd, name, seeds):
        bysub = defaultdict(dict)
        for (sub, o), idx in gd.items(): bysub[sub][o] = idx
        rows = defaultdict(list)
        for seed in range(seeds):
            for sub, od in bysub.items():
                allidx = [i for v in od.values() for i in v]
                if len(allidx) < args.K + 200: continue
                rng = np.random.RandomState(abs(hash((sub, seed))) % 2 ** 31)
                n_or = len(od); kk = int(np.ceil(args.K / n_or))
                sel = []
                for o, idx in od.items():
                    sel += [idx[j] for j in sample_coverage(d["pred"][idx], kk, rng)]
                sel = list(dict.fromkeys(sel))
                ev = [i for i in allidx if i not in set(sel)]
                if len(ev) < 100: continue
                p, l = d["pred"], d["label"]
                c = cm_of(p[ev], l[ev], d["err"][ev])
                base = float(em(p[ev], l[ev], c).mean())
                evo = np.array([ori.get(d["path"][i], 1) for i in ev])
                y_map = p[ev].copy(); y_mlp = p[ev].copy()
                for o in np.unique(evo):
                    m = evo == o
                    so = [i for i in sel if ori.get(d["path"][i], 1) == o]
                    mu = pri.get(o, pri[list(pri)[0]])[0]
                    mu2 = [np.array([mu[0], mu[2]]), np.array([mu[1], mu[3]])]
                    th = fit_theta(p[so][:, 0], l[so][:, 0], p[so][:, 1], l[so][:, 1], 1.0, mu2) \
                        if len(so) >= 2 else [mu2[0], mu2[1]]
                    y_map[m] = apply_affine(p[ev][m], th, 0.75)
                    th4 = mlp.predict(sc.transform(feats_of(p, l, so, o)[None]))[0]
                    thm = [np.array([th4[0], th4[2]]), np.array([th4[1], th4[3]])]
                    y_mlp[m] = apply_affine(p[ev][m], thm, 0.75)
                rows["base"].append((base, len(ev)))
                rows["map"].append((float(em(y_map, l[ev], c).mean()), base, len(ev)))
                rows["mlp"].append((float(em(y_mlp, l[ev], c).mean()), base, len(ev)))
        n = sum(x[1] for x in rows["base"]); B = sum(x[0] * x[1] for x in rows["base"]) / n
        print(f"\n[{name}] seeds={seeds} 帧 {n} 基线 {B:.4f}")
        for k in ("map", "mlp"):
            v = np.array(rows[k]); E = (v[:, 0] * v[:, 2]).sum() / v[:, 2].sum()
            print(f"  {k:<4} {E:.4f} ({100*(E/B-1):+.1f}%) 最差 {100*max(v[:,0]/v[:,1]-1):+.0f}% "
                  f"退化 {int(((v[:,0]/v[:,1]-1)>0.02).sum())}/{len(v)}")

    evaluate(va, gva, "val", 2)
    evaluate(te, gte, "test", args.seeds)


if __name__ == "__main__":
    main()
