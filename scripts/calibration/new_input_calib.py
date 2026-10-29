"""Huber + ridge/MLP correction from shared landmark/eye features.

Base:  q0 = Huber_theta(p)  (K support labels)
Corr:  q  = q0 + W^T z(features)

Features are the shared MediaPipe landmark/eye/face/quality features extracted
for both datasets.  The correction strength is chosen by leave-one-out on the
K support points, so no extra labels are used.
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path[:0] = ["src", "."]
from scripts.calibration.seq_bayes_calib import load_index, em  # noqa: E402
from scripts.calibration.meta_residual_calib import (  # noqa: E402
    fit_source_mu, fit_theta_huber, apply_theta_np, support_indices,
)


def log(msg):
    print(msg, flush=True)


def load_sessions_rows(feat_npz, index_csv, context_key, path_to_row, invalid_row):
    z = np.load(feat_npz, allow_pickle=True)
    pred = z["pred"].astype(np.float64)
    label = z["label"].astype(np.float64)
    path = [str(p) for p in z["path"].tolist()]
    fidx = z["frame_index"].astype(np.int64)
    meta = load_index(index_csv, np.array(path, object))
    groups = defaultdict(list)
    for i, p in enumerate(path):
        if p in meta:
            ctx = meta[p][4] if context_key != "none" else ""
            groups[f"{p.rsplit('/', 1)[0]}||{ctx}"].append(i)
    out = {}
    for s, idx in groups.items():
        idx = np.array(sorted(idx, key=lambda i: int(fidx[i])), np.int64)
        if len(idx) < 40:
            continue
        out[s] = dict(
            pred=pred[idx], label=label[idx],
            sx=np.array([meta[path[i]][2] for i in idx]),
            sy=np.array([meta[path[i]][3] for i in idx]),
            rows=np.array([path_to_row.get(path[i], invalid_row) for i in idx], np.int64),
        )
    return out


def select_support(d, K, rng, F, V, rows, mode):
    if mode == "coverage":
        return support_indices(d, K, rng)
    n = len(d["pred"])
    if mode == "random":
        return np.sort(rng.choice(n, min(K, n), replace=False))
    # greedy max-min diversity in standardized feature space
    X = F[rows].astype(np.float64).copy()
    ok = V[rows] & np.isfinite(X).all(1)
    if ok.sum() < K:
        return support_indices(d, K, rng)
    mu = X[ok].mean(0)
    sd = X[ok].std(0) + 1e-6
    X = np.clip((X - mu) / sd, -5.0, 5.0)
    X[~np.isfinite(X)] = 0.0
    X = np.concatenate([X, d["pred"] * 3.0], 1)
    cand = np.where(ok)[0]
    sel = [int(cand[rng.randint(len(cand))])]
    dist = np.linalg.norm(X[cand] - X[sel[0]], axis=1)
    while len(sel) < min(K, len(cand)):
        j = int(np.argmax(dist))
        sel.append(int(cand[j]))
        d2 = np.linalg.norm(X[cand] - X[sel[-1]], axis=1)
        dist = np.minimum(dist, d2)
    return np.array(sorted(sel), np.int64)


def region_feats(P, mode):
    if mode == "none":
        return np.zeros((len(P), 0), np.float64)
    R = 2 if mode == "2x2" else 3
    rx = np.clip((P[:, 0] * R).astype(int), 0, R - 1)
    ry = np.clip((P[:, 1] * R).astype(int), 0, R - 1)
    idx = ry * R + rx
    oh = np.zeros((len(P), R * R), np.float64)
    oh[np.arange(len(P)), idx] = 1.0
    inter = np.concatenate([oh * P[:, 0:1], oh * P[:, 1:2]], 1)
    return np.concatenate([oh, inter], 1)


def prep_features(F, V, idx):
    """Standardize on the support subset, fill missing with 0."""
    X = F[idx].astype(np.float64).copy()
    m = V[idx] & np.isfinite(X).all(1)
    if m.sum() < 5:
        return None, None, None
    mu = X[m].mean(0)
    sd = X[m].std(0) + 1e-6
    X = np.clip((X - mu) / sd, -5.0, 5.0)
    X[~np.isfinite(X)] = 0.0
    X = np.concatenate([X, np.ones((len(X), 1))], 1)
    return X, mu, sd


def fit_ridge(Z, y, lam):
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    return np.linalg.solve(A, Z.T @ y)


def fit_robust(Z, y, lam, delta=1.345, iters=6):
    w = fit_ridge(Z, y, lam)
    for _ in range(iters):
        r = y - Z @ w
        scale = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        u = np.abs(r) / (delta * scale)
        wt = np.where(u <= 1.0, 1.0, 1.0 / np.maximum(u, 1e-9))
        A = Z.T @ (wt[:, None] * Z) + lam * np.eye(Z.shape[1])
        w = np.linalg.solve(A, Z.T @ (wt * y))
    return w


def loo_lam(Z, y, lams):
    n = len(y)
    best = (lams[0], float("inf"))
    for lam in lams:
        errs = []
        for j in range(n):
            m = np.ones(n, bool)
            m[j] = False
            if m.sum() < 3:
                continue
            w = fit_ridge(Z[m], y[m], lam)
            errs.append(float((Z[j] @ w - y[j]) ** 2))
        if errs:
            e = float(np.mean(errs))
            if e < best[1]:
                best = (lam, e)
    return best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--val-npz", required=True)
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--pose-npz", required=True)
    ap.add_argument("--landmark-npz", required=True)
    ap.add_argument("--context-key", choices=["none", "orientation", "activity"], default="none")
    ap.add_argument("--K", type=int, default=52)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--delta", type=float, default=1.345)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lams", type=float, nargs="+",
                    default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--groups", nargs="+", default=["all"])
    ap.add_argument("--method", choices=["ridge", "mlp"], default="ridge")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--regions", choices=["none", "2x2", "3x3"], default="none")
    ap.add_argument("--robust", action="store_true")
    ap.add_argument("--use-context", action="store_true")
    ap.add_argument("--select", choices=["coverage", "random", "dopt"], default="coverage")
    ap.add_argument("--robust-iters", type=int, default=6)
    ap.add_argument("--use-pose", action="store_true")
    ap.add_argument("--mlp-hidden", type=int, default=32)
    ap.add_argument("--mlp-steps", type=int, default=300)
    ap.add_argument("--mlp-lr", type=float, default=1e-2)
    ap.add_argument("--mlp-wd", type=float, default=1e-2)
    ap.add_argument("--detail", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    log(f"[ncal] args={vars(a)}")
    rows = list(csv.DictReader(open(a.index_csv, encoding="utf-8")))
    pcol = "frame_path" if rows and "frame_path" in rows[0] else "rgb_path"
    path_to_row = {r[pcol]: i for i, r in enumerate(rows)}
    z = np.load(a.landmark_npz, allow_pickle=True)
    names = [str(n) for n in z["names"].tolist()]
    GROUPS = {
        "iris": ["iris_lx", "iris_ly", "iris_rx", "iris_ry"],
        "eye": ["eye_l_w", "eye_l_h", "eye_r_w", "eye_r_h",
                "ear_l", "ear_r", "iod_face"],
        "face": ["face_cx", "face_cy", "face_w", "face_h", "face_ratio"],
        "quality": ["bright", "contrast", "blur"],
        "irisp": ["iris_px_l", "iris_px_r"],
    }
    use = np.zeros(len(names), bool)
    for g in a.groups:
        if g == "all":
            use[:] = True
        else:
            for n in GROUPS.get(g, []):
                if n in names:
                    use[names.index(n)] = True
    if not use.any():
        raise SystemExit("empty feature selection")
    log(f"[ncal] groups={a.groups} used={int(use.sum())}/{len(names)}")
    F = np.full((len(rows), len(names)), np.nan, np.float32)
    V = np.zeros(len(rows), bool)
    F[z["row_idx"].astype(np.int64)] = z["feats"]
    V[z["row_idx"].astype(np.int64)] = z["valid"]
    invalid_row = len(rows)
    if a.use_pose:
        pz = np.load(a.pose_npz, allow_pickle=True)
        pf = np.full((len(rows), pz["feats"].shape[1]), np.nan, np.float32)
        pv = np.zeros(len(rows), bool)
        pr = pz["row_idx"].astype(np.int64)
        pf[pr] = pz["feats"]
        pv[pr] = pz["valid"]
        F = np.concatenate([F, pf], 1)
        V = V & pv
        names = names + [f"pose{i}" for i in range(pz["feats"].shape[1])]
        use = np.concatenate([use, np.ones(pz["feats"].shape[1], bool)])
        log(f"[ncal] appended pose dims={pz['feats'].shape[1]}")
    if a.use_context:
        ccol = "activity" if "activity" in rows[0] else (
            "orientation" if "orientation" in rows[0] else None)
        if ccol is None:
            raise SystemExit("no context column in index")
        vals = sorted({r[ccol] for r in rows})
        cmap = {v: i for i, v in enumerate(vals)}
        oh = np.zeros((len(rows), len(vals)), np.float32)
        for i, r in enumerate(rows):
            oh[i, cmap[r[ccol]]] = 1.0
        F = np.concatenate([F, oh], 1)
        names = names + [f"ctx_{v}" for v in vals]
        use = np.concatenate([use, np.ones(len(vals), bool)])
        log(f"[ncal] appended context one-hot={vals}")
    F = np.vstack([F, np.zeros((1, F.shape[1]), np.float32)])
    V = np.append(V, False)

    D = load_sessions_rows(a.train_npz, a.index_csv, a.context_key,
                           path_to_row, invalid_row)
    S = load_sessions_rows(a.val_npz if a.split == "val" else a.test_npz,
                           a.index_csv, a.context_key, path_to_row, invalid_row)
    mu_aff, _ = fit_source_mu({k: {"pred": d["pred"], "label": d["label"]}
                               for k, d in D.items()})
    log(f"[ncal] source={len(D)} eval={len(S)} K={a.K} dims={len(names)}")

    acc0 = acc1 = nn = 0.0
    ratios = []
    lams_used = []
    detail = []
    for si, (s, d) in enumerate(S.items()):
        r0 = r1 = rn = 0.0
        for seed in range(a.seeds):
            sel = select_support(d, a.K, np.random.RandomState(90000 + seed),
                                 F, V, d["rows"], a.select)
            q = np.setdiff1d(np.arange(len(d["pred"])), sel)
            if len(q) < 20:
                continue
            th, _ = fit_theta_huber(d["pred"][sel], d["label"][sel],
                                    a.lam, mu_aff, a.delta)
            q0 = apply_theta_np(d["pred"][q], th)
            e0 = float(em(q0, d["label"][q], d["sx"][q], d["sy"][q]).mean())
            Zs, fmu, fsd = prep_features(F[:, use], V, d["rows"][sel])
            if Zs is None:
                e1 = e0
            else:
                Rs = d["label"][sel] - apply_theta_np(d["pred"][sel], th)
                Ze_raw = F[d["rows"][q]][:, use].astype(np.float64).copy()
                Ze_raw = np.clip((Ze_raw - fmu) / fsd, -5.0, 5.0)
                Ze_raw[~np.isfinite(Ze_raw)] = 0.0
                if a.method == "ridge":
                    Zs = np.concatenate([Zs, region_feats(d["pred"][sel], a.regions)], 1)
                    Ze_raw = np.concatenate(
                        [Ze_raw, region_feats(d["pred"][q], a.regions)], 1)
                    w = []
                    lam_used = []
                    for ax in range(2):
                        lam_ax = loo_lam(Zs, Rs[:, ax], a.lams)
                        lam_used.append(lam_ax)
                        if a.robust:
                            w.append(fit_robust(Zs, Rs[:, ax], lam_ax,
                                                iters=a.robust_iters))
                        else:
                            w.append(fit_ridge(Zs, Rs[:, ax], lam_ax))
                    if a.gate:
                        # baseline LOO: Huber fitted on support-minus-j
                        base_loo = []
                        for j in range(len(d["pred"][sel])):
                            m = np.ones(len(d["pred"][sel]), bool)
                            m[j] = False
                            if m.sum() < 3:
                                continue
                            thj, _ = fit_theta_huber(d["pred"][sel][m],
                                                     d["label"][sel][m],
                                                     a.lam, mu_aff, a.delta)
                            qj = apply_theta_np(d["pred"][sel][j:j + 1], thj)
                            base_loo.append(float(
                                em(qj, d["label"][sel][j:j + 1],
                                   d["sx"][sel][j:j + 1],
                                   d["sy"][sel][j:j + 1])[0]) ** 2)
                        base_err = float(np.mean(base_loo)) if base_loo else 0.0
                        loo_errs = []
                        for j in range(len(Zs)):
                            m = np.ones(len(Zs), bool)
                            m[j] = False
                            if m.sum() < 3:
                                continue
                            pred = []
                            for ax in range(2):
                                wj = fit_ridge(Zs[m], Rs[m, ax], lam_used[ax])
                                pred.append(Zs[j] @ wj)
                            loo_errs.append((pred[0] - Rs[j, 0]) ** 2 +
                                            (pred[1] - Rs[j, 1]) ** 2)
                        if loo_errs and float(np.mean(loo_errs)) > base_err:
                            w = [np.zeros_like(w[0]), np.zeros_like(w[1])]
                    Ze = np.concatenate([Ze_raw, np.ones((len(Ze_raw), 1))], 1)
                    corr = np.stack([Ze @ w[0], Ze @ w[1]], 1)
                    lams_used.append(float(np.mean(lam_used)))
                else:
                    import torch
                    import torch.nn as torch_nn
                    torch.set_num_threads(4)
                    Xs = torch.tensor(Zs[:, :-1], dtype=torch.float32)
                    Ys = torch.tensor(Rs, dtype=torch.float32)
                    net = torch_nn.Sequential(
                        torch_nn.Linear(Xs.shape[1], a.mlp_hidden), torch_nn.GELU(),
                        torch_nn.Linear(a.mlp_hidden, 2),
                    )
                    torch_nn.init.zeros_(net[-1].weight)
                    torch_nn.init.zeros_(net[-1].bias)
                    opt = torch.optim.AdamW(net.parameters(), lr=a.mlp_lr,
                                            weight_decay=a.mlp_wd)
                    for _ in range(a.mlp_steps):
                        loss = ((net(Xs) - Ys) ** 2).mean()
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                    with torch.no_grad():
                        corr = net(torch.tensor(Ze_raw, dtype=torch.float32)).numpy()
                    lams_used.append(0.0)
                corr = np.clip(corr, -0.15, 0.15)
                q1 = q0 + corr
                e1 = float(em(q1, d["label"][q], d["sx"][q], d["sy"][q]).mean())
            r0 += e0 * len(q)
            r1 += e1 * len(q)
            rn += len(q)
        if rn:
            b, e = r0 / rn, r1 / rn
            acc0 += b * rn
            acc1 += e * rn
            nn += rn
            ratios.append(e / max(b, 1e-12) - 1)
            detail.append([s, rn, b, e, 100 * (e / b - 1)])
            if si % 5 == 0 or si == len(S) - 1:
                log(f"[ncal] session={s} n={rn:.0f} base={b:.4f} "
                    f"lm={e:.4f} ({100*(e/b-1):+.2f}%)")
    B, E = acc0 / nn, acc1 / nn
    rr = np.array(ratios)
    rng = np.random.RandomState(12345)
    br, bd = [], []
    nvec = np.array([x[1] for x in detail], float)
    dvec = np.array([x[3] - x[2] for x in detail], float)
    for _ in range(10000):
        idx = rng.choice(len(detail), len(detail), replace=True)
        w = nvec[idx]
        br.append(float(np.sum(rr[idx] * w) / w.sum()))
        bd.append(float(np.sum(dvec[idx] * w) / w.sum()))
    lo, hi = np.percentile(br, [2.5, 97.5])
    log(f"[ncal] RESULT K={a.K} base={B:.4f} lm={E:.4f} "
        f"({100*(E/B-1):+.2f}%) sessions={len(rr)} "
        f"improved={int((rr<0).sum())} worst={100*rr.max():+.1f}% "
        f"boot95=[{100*lo:+.2f},{100*hi:+.2f}]% "
        f"p(delta>=0)={np.mean(np.array(bd)>=0):.3f} "
        f"mean_lam={np.mean(lams_used) if lams_used else -1:g}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["session", "n", "base", "lm", "pct"])
        w.writerows(detail)
    if a.detail:
        dp = Path(a.detail)
        dp.parent.mkdir(parents=True, exist_ok=True)
        with open(dp, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["session", "n", "base", "lm", "pct"])
            w.writerows(detail)
    log(f"[ncal] saved {out}")


if __name__ == "__main__":
    main()
