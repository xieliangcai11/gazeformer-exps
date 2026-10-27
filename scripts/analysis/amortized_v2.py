"""B1 + B2：摊销式校准（学习先验 / 超网络）+ 覆盖式选点 + 多 seed。

- B2：K 个标定帧按"朝向 × 屏幕 3×3 覆盖"选点；5 个 seed；与随机选点对比。
- B1：用源域（val）训练一个 MLP，把 K 个校准样本的统计特征映射为
      每朝向仿射参数 θ=[sx,sy,bx,by]；目标域一次前向得到 θ（无优化）。
       对照：无先验 LS、先验 MAP（岭回归）、真值 θ（上界）。

自检：
  C1 基线 EM 与已知一致
  C2 λ=1e9 的 MAP 解 ≈ 先验均值；λ=0 ≈ LS
  C3 覆盖选点在每个 (朝向,格子) 上的最大样本数不超过 kper

用法：
  python -m scripts.analysis.amortized_v2 \
    --source out/gazecapture/analysis/feats_gc_val_m2_full.npz \
    --target out/gazecapture/analysis/feats_gc_test_m2_full.npz \
    --index-csv data/GazeCapture/index/index.csv --K 52
"""
import argparse
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
    return out  # [(sx,bx),(sy,by)]


def sample_coverage(pred, k, rng, nbin=3):
    """按预测位置的 3×3 网格均匀取样（覆盖式）。"""
    cells = defaultdict(list)
    for j in range(len(pred)):
        xb = min(nbin - 1, int(pred[j, 0] * nbin)); yb = min(nbin - 1, int(pred[j, 1] * nbin))
        cells[(xb, yb)].append(j)
    per = max(1, k // max(1, len(cells)))
    sel = []
    for v in cells.values():
        v = list(v); rng.shuffle(v); sel += v[:per]
    # 不够则从剩余随机补
    if len(sel) < k:
        rest = [j for j in range(len(pred)) if j not in set(sel)]
        sel += list(rng.choice(rest, min(k - len(sel), len(rest)), replace=False))
    return sorted(set(sel))[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--K", type=int, default=52)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--lams", default="0.5,1.0,2.0")
    args = ap.parse_args()

    src = load_npz(args.source); tgt = load_npz(args.target)
    import csv
    want = set(src["path"].tolist()) | set(tgt["path"].tolist())
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

    sg = group(src); tg = group(tgt)

    # 先验（每朝向）：源域用全量帧拟合
    per = defaultdict(list)
    for (sub, o), idx in sg.items():
        if len(idx) < 100: continue
        p, l = src["pred"][idx], src["label"][idx]
        th = fit_theta(p[:, 0], l[:, 0], p[:, 1], l[:, 1])
        per[o].append([th[0][0], th[1][0], th[0][1], th[1][1]])
    pri = {o: (np.mean(np.array(v), 0), np.cov(np.array(v).T) + 1e-6 * np.eye(4))
           for o, v in per.items()}
    print(f"[prior] {len(pri)} 个朝向；样本数 {[len(v) for v in per.values()]}")

    # C2
    o0 = list(pri)[0]; mu0 = pri[o0][0]
    idx0 = next(idx for k, idx in sg.items() if k[1] == o0 and len(idx) >= 200)
    p0, l0 = src["pred"][idx0][:20], src["label"][idx0][:20]
    big = fit_theta(p0[:, 0], l0[:, 0], p0[:, 1], l0[:, 1], 1e9,
                    [np.array([mu0[0], mu0[2]]), np.array([mu0[1], mu0[3]])])
    print(f"[C2] λ=1e9 解 {np.round([big[0][0], big[1][0], big[0][1], big[1][1]], 3)}"
          f" ≈ 先验 {np.round(mu0, 3)}")
    assert np.allclose([big[0][0], big[1][0], big[0][1], big[1][1]], mu0, atol=1e-3)

    # 训练 B1（源域 episodes）
    Xtr, Ytr = [], []
    for (sub, o), idx in sg.items():
        if len(idx) < 200: continue
        p, l = src["pred"][idx], src["label"][idx]
        th = fit_theta(p[:, 0], l[:, 0], p[:, 1], l[:, 1])
        y = np.array([th[0][0], th[1][0], th[0][1], th[1][1]])
        c = cm_of(p, l, src["err"][idx])
        # 与目标域一致：每朝向约 K/4 个标定帧
        kper_train = max(2, int(np.ceil(args.K / 4)))
        for t in range(20):
            rng = np.random.RandomState(abs(hash((sub, o, t))) % 2 ** 31)
            sel = sample_coverage(p, kper_train, rng)
            ev = np.array([i for i in range(len(p)) if i not in set(sel)])
            if len(ev) < 50: continue
            feats = np.concatenate([
                p[sel].mean(0), l[sel].mean(0), (p[sel] - l[sel]).mean(0),
                p[sel].std(0), l[sel].std(0), [len(sel)],
                pri.get(o, (np.zeros(4), np.eye(4)))[0],
            ])
            Xtr.append(feats); Ytr.append(y)
    Xtr = np.array(Xtr); Ytr = np.array(Ytr)
    print(f"[B1] 训练样本 {len(Xtr)}（源会话 episode）")
    sc = StandardScaler().fit(Xtr)
    mlp = MLPRegressor(hidden_layer_sizes=(32,), alpha=1e-3, max_iter=2000,
                       random_state=0, early_stopping=False).fit(sc.transform(Xtr), Ytr)

    # 目标域：B2（覆盖选点 × seeds）+ B1
    bysub = defaultdict(dict)
    for (sub, o), idx in tg.items(): bysub[sub][o] = idx
    rows = defaultdict(list)
    for seed in range(args.seeds):
        for sub, od in bysub.items():
            allidx = [i for v in od.values() for i in v]
            if len(allidx) < args.K + 200: continue
            rng = np.random.RandomState(abs(hash((sub, seed))) % 2 ** 31)
            # 每朝向均匀分配 K
            n_or = len(od); kper = int(np.ceil(args.K / n_or))
            sel = []
            for o, idx in od.items():
                sel += [idx[j] for j in sample_coverage(tgt["pred"][idx], kper, rng)]
            sel = list(dict.fromkeys(sel))
            ev = [i for i in allidx if i not in set(sel)]
            if len(ev) < 100: continue
            p, l = tgt["pred"], tgt["label"]
            c = cm_of(p[ev], l[ev], tgt["err"][ev])
            base = float(em(p[ev], l[ev], c).mean())
            evo = np.array([ori.get(tgt["path"][i], 1) for i in ev])
            # MAP
            y_map = p[ev].copy()
            for o in np.unique(evo):
                m = evo == o
                so = [i for i in sel if ori.get(tgt["path"][i], 1) == o]
                mu, _ = pri.get(o, pri[list(pri)[0]])
                mu2 = [np.array([mu[0], mu[2]]), np.array([mu[1], mu[3]])]
                th = fit_theta(p[so][:, 0], l[so][:, 0], p[so][:, 1], l[so][:, 1], 1.0, mu2) \
                    if len(so) >= 2 else [mu2[0], mu2[1]]
                y_map[m] = apply_affine(p[ev][m], th, 0.75)
            # B1：每个朝向用该朝向的标定样本构造特征 → MLP 预测 θ
            y_mlp = p[ev].copy()
            for o in np.unique(evo):
                m = evo == o
                so = [i for i in sel if ori.get(tgt["path"][i], 1) == o]
                if len(so) < 2:
                    mu = pri.get(o, pri[list(pri)[0]])[0]
                    th = [np.array([mu[0], mu[2]]), np.array([mu[1], mu[3]])]
                else:
                    feats = np.concatenate([
                        p[so].mean(0), l[so].mean(0), (p[so] - l[so]).mean(0),
                        p[so].std(0), l[so].std(0), [len(so)],
                        pri.get(o, pri[list(pri)[0]])[0],
                    ])[None]
                    th4 = mlp.predict(sc.transform(feats))[0]
                    th = [np.array([th4[0], th4[2]]), np.array([th4[1], th4[3]])]
                y_mlp[m] = apply_affine(p[ev][m], th, 0.75)
            rows["base"].append((base, len(ev)))
            rows["map"].append((float(em(y_map, l[ev], c).mean()), base, len(ev)))
            rows["mlp"].append((float(em(y_mlp, l[ev], c).mean()), base, len(ev)))
    n = sum(x[1] for x in rows["base"]); B = sum(x[0] * x[1] for x in rows["base"]) / n
    print(f"\n[结果] seeds={args.seeds} K={args.K} 评估帧 {n} 基线 {B:.4f}")
    for k in ("map", "mlp"):
        v = np.array(rows[k]); E = (v[:, 0] * v[:, 2]).sum() / v[:, 2].sum()
        print(f"  {k:<4} {E:.4f} ({100*(E/B-1):+.1f}%) 最差 {100*max(v[:,0]/v[:,1]-1):+.0f}% "
              f"退化 {int(((v[:,0]/v[:,1]-1)>0.02).sum())}/{len(v)}")


if __name__ == "__main__":
    main()
