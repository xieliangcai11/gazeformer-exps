"""FS-TTC with shared landmark/eye features.

FS-TTC estimates screen coordinates directly by an identity-regularized ridge
on features, with LOO-selected correction strength.  Here the feature vector is

    [p_x, p_y, 1]  +  standardized landmark/eye/face/quality features

so the new information is inside the FS-TTC estimator itself.
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
            rows=np.array([path_to_row.get(path[i], invalid_row) for i in idx],
                          np.int64),
        )
    return out


def build_Z(P, LM, use, mu, sd):
    X = LM.astype(np.float64).copy()
    X = np.clip((X - mu) / sd, -5.0, 5.0)
    X[~np.isfinite(X)] = 0.0
    return np.concatenate([P, X[:, use], np.ones((len(P), 1))], 1)


def prior_vec(dim, axis, n_p=2):
    w = np.zeros(dim)
    if axis == 0:
        w[0] = 1.0
    else:
        w[1] = 1.0
    return w


def fit_ridge(Z, y, lam, w0):
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    return np.linalg.solve(A, Z.T @ y + lam * w0)


def loo_lam(Z, y, w0, lams):
    n = len(y)
    best = (lams[0], float("inf"))
    for lam in lams:
        errs = []
        for j in range(n):
            m = np.ones(n, bool)
            m[j] = False
            if m.sum() < 3:
                continue
            w = fit_ridge(Z[m], y[m], lam, w0)
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
                    default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])
    ap.add_argument("--groups", nargs="+", default=["all"])
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--out", required=True)
    ap.add_argument("--detail", default="")
    a = ap.parse_args()

    log(f"[fsttclm] args={vars(a)}")
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
    log(f"[fsttclm] groups={a.groups} used={int(use.sum())}/{len(names)}")
    LM = np.full((len(rows), len(names)), np.nan, np.float32)
    V = np.zeros(len(rows), bool)
    LM[z["row_idx"].astype(np.int64)] = z["feats"]
    V[z["row_idx"].astype(np.int64)] = z["valid"]
    invalid_row = len(rows)
    LM = np.vstack([LM, np.zeros((1, LM.shape[1]), np.float32)])
    V = np.append(V, False)

    D = load_sessions_rows(a.train_npz, a.index_csv, a.context_key,
                           path_to_row, invalid_row)
    S = load_sessions_rows(a.val_npz if a.split == "val" else a.test_npz,
                           a.index_csv, a.context_key, path_to_row, invalid_row)
    mu_aff, _ = fit_source_mu({k: {"pred": d["pred"], "label": d["label"]}
                               for k, d in D.items()})
    log(f"[fsttclm] source={len(D)} eval={len(S)} K={a.K}")

    acc0 = acc1 = nn = 0.0
    ratios = []
    lams_used = []
    detail = []
    for si, (s, d) in enumerate(S.items()):
        r0 = r1 = rn = 0.0
        for seed in range(a.seeds):
            sel = support_indices(d, a.K, np.random.RandomState(90000 + seed))
            q = np.setdiff1d(np.arange(len(d["pred"])), sel)
            if len(q) < 20:
                continue
            th, _ = fit_theta_huber(d["pred"][sel], d["label"][sel],
                                    a.lam, mu_aff, a.delta)
            q0 = apply_theta_np(d["pred"][q], th)
            e0 = float(em(q0, d["label"][q], d["sx"][q], d["sy"][q]).mean())
            Ls = LM[d["rows"][sel]]
            m = V[d["rows"][sel]] & np.isfinite(Ls).all(1)
            if m.sum() < 5:
                e1 = e0
            else:
                fmu = Ls[m].mean(0)
                fsd = Ls[m].std(0) + 1e-6
                Zs = build_Z(d["pred"][sel], Ls, use, fmu, fsd)
                w = []
                lam_used = []
                for ax in range(2):
                    w0 = prior_vec(Zs.shape[1], ax)
                    lam_ax = loo_lam(Zs, d["label"][sel, ax], w0, a.lams)
                    lam_used.append(lam_ax)
                    w.append(fit_ridge(Zs, d["label"][sel, ax], lam_ax, w0))
                Le = LM[d["rows"][q]]
                Ze = build_Z(d["pred"][q], Le, use, fmu, fsd)
                q1 = np.stack([Ze @ w[0], Ze @ w[1]], 1)
                e1 = float(em(q1, d["label"][q], d["sx"][q], d["sy"][q]).mean())
                lams_used.append(float(np.mean(lam_used)))
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
                log(f"[fsttclm] session={s} n={rn:.0f} base={b:.4f} "
                    f"fsttc={e:.4f} ({100*(e/b-1):+.2f}%)")
    B, E = acc0 / nn, acc1 / nn
    rr = np.array(ratios)
    nvec = np.array([x[1] for x in detail], float)
    dvec = np.array([x[3] - x[2] for x in detail], float)
    rng = np.random.RandomState(12345)
    br, bd = [], []
    for _ in range(10000):
        idx = rng.choice(len(detail), len(detail), replace=True)
        w = nvec[idx]
        br.append(float(np.sum(rr[idx] * w) / w.sum()))
        bd.append(float(np.sum(dvec[idx] * w) / w.sum()))
    lo, hi = np.percentile(br, [2.5, 97.5])
    log(f"[fsttclm] RESULT K={a.K} base={B:.4f} fsttc={E:.4f} "
        f"({100*(E/B-1):+.2f}%) sessions={len(rr)} "
        f"improved={int((rr<0).sum())} worst={100*rr.max():+.1f}% "
        f"boot95=[{100*lo:+.2f},{100*hi:+.2f}]% "
        f"p(delta>=0)={np.mean(np.array(bd)>=0):.3f} "
        f"mean_lam={np.mean(lams_used) if lams_used else -1:g}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["session", "n", "base", "fsttc", "pct"])
        w.writerows(detail)
    if a.detail:
        dp = Path(a.detail)
        dp.parent.mkdir(parents=True, exist_ok=True)
        with open(dp, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["session", "n", "base", "fsttc", "pct"])
            w.writerows(detail)
    log(f"[fsttclm] saved {out}")


if __name__ == "__main__":
    main()
