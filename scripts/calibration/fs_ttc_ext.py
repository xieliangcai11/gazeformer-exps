"""FS-TTC extended with radial / quadratic features.

The original FS-TTC is an identity-regularized affine ridge with LOO-selected
correction strength.  Here the same estimator is extended to richer features:

    affine : [x, y, 1]
    radial : [x, y, 1, r]
    quad   : [x, y, 1, r, x^2, y^2, xy, r^2]

so FS-TTC itself can model the spatial structure found by the residual
diagnostic, without switching to the Huber baseline.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path[:0] = ["src", "."]
from scripts.calibration.seq_bayes_calib import load_sessions, em  # noqa: E402
from scripts.calibration.meta_residual_calib import (  # noqa: E402
    fit_source_mu, fit_theta_huber, apply_theta_np, support_indices,
)


def log(msg):
    print(msg, flush=True)


def feats(P, mode):
    x, y = P[:, 0], P[:, 1]
    r = np.hypot(x - 0.5, y - 0.5)
    if mode == "affine":
        return np.stack([x, y, np.ones_like(x)], 1)
    if mode == "radial":
        return np.stack([x, y, np.ones_like(x), r], 1)
    if mode == "quad":
        return np.stack([x, y, np.ones_like(x), r,
                         x * x, y * y, x * y, r * r], 1)
    raise ValueError(mode)


def prior_vec(mode, axis):
    d = {"affine": 3, "radial": 4, "quad": 8}[mode]
    w = np.zeros(d)
    if axis == 0:
        w[0] = 1.0
    else:
        w[1] = 1.0
    return w


def fit_ridge(Z, y, lam, w0):
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    b = Z.T @ y + lam * w0
    return np.linalg.solve(A, b)


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


def eval_case(d, sel, mode, mu, lam, delta, lams):
    q = np.setdiff1d(np.arange(len(d["pred"])), sel)
    if len(q) < 20:
        return None
    P_s, L_s = d["pred"][sel], d["label"][sel]
    P_e, L_e = d["pred"][q], d["label"][q]
    # Huber baseline
    th, _ = fit_theta_huber(P_s, L_s, lam, mu, delta)
    q0 = apply_theta_np(P_e, th)
    e0 = float(em(q0, L_e, d["sx"][q], d["sy"][q]).mean())
    if mode == "huber":
        return e0, e0, 0.0, len(q)
    Zs = feats(P_s, mode)
    Ze = feats(P_e, mode)
    w = []
    lam_used = []
    for axis in range(2):
        w0 = prior_vec(mode, axis)
        lam_ax = loo_lam(Zs, L_s[:, axis], w0, lams)
        lam_used.append(lam_ax)
        w.append(fit_ridge(Zs, L_s[:, axis], lam_ax, w0))
    q1 = np.stack([Ze @ w[0], Ze @ w[1]], 1)
    e1 = float(em(q1, L_e, d["sx"][q], d["sy"][q]).mean())
    return e0, e1, float(np.mean(lam_used)), len(q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--val-npz", required=True)
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--pose-npz", required=True)
    ap.add_argument("--context-key", choices=["none", "orientation", "activity"], default="none")
    ap.add_argument("--K", type=int, default=52)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--delta", type=float, default=1.345)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--lams", type=float, nargs="+",
                    default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
    ap.add_argument("--methods", nargs="+",
                    default=["huber", "affine", "radial", "quad"])
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    log(f"[fsttcx] start={time.strftime('%F %T')} args={vars(a)}")
    D = load_sessions(a.train_npz, a.index_csv, a.pose_npz, a.context_key)
    S = load_sessions(a.val_npz if a.split == "val" else a.test_npz,
                      a.index_csv, a.pose_npz, a.context_key)
    mu, _ = fit_source_mu(D)
    log(f"[fsttcx] source={len(D)} eval={len(S)} K={a.K} "
        f"prior={np.round(mu,4).tolist()}")

    rows = []
    for method in a.methods:
        acc0 = acc1 = nn = 0.0
        ratios = []
        lams_used = []
        t0 = time.time()
        for si, (s, d) in enumerate(S.items()):
            r0 = r1 = rn = 0.0
            for seed in range(a.seeds):
                rng = np.random.RandomState(90000 + seed)
                sel = support_indices(d, a.K, rng)
                r = eval_case(d, sel, method, mu, a.lam, a.delta, a.lams)
                if r is None:
                    continue
                e0, e1, lu, n = r
                r0 += e0 * n
                r1 += e1 * n
                rn += n
                if method != "huber":
                    lams_used.append(lu)
            if rn:
                b, e = r0 / rn, r1 / rn
                acc0 += b * rn
                acc1 += e * rn
                nn += rn
                ratios.append(e / max(b, 1e-12) - 1)
                if si % 5 == 0 or si == len(S) - 1:
                    log(f"[fsttcx] {method} session={s} n={rn:.0f} "
                        f"base={b:.4f} calib={e:.4f} ({100*(e/b-1):+.2f}%) "
                        f"elapsed={time.time()-t0:.1f}s")
        if not nn:
            continue
        B, E = acc0 / nn, acc1 / nn
        rr = np.array(ratios)
        log(f"[fsttcx] RESULT method={method} K={a.K} base={B:.4f} "
            f"calib={E:.4f} ({100*(E/B-1):+.2f}%) sessions={len(rr)} "
            f"improved={int((rr<0).sum())} worst={100*rr.max():+.1f}% "
            f"mean_lam={np.mean(lams_used) if lams_used else -1:g}")
        rows.append([method, a.K, B, E, 100 * (E / B - 1), len(rr),
                     int((rr < 0).sum()), float(rr.max()),
                     float(np.mean(lams_used)) if lams_used else -1])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "K", "base", "calib", "pct", "sessions",
                    "improved", "worst", "mean_lam"])
        w.writerows(rows)
    log(f"[fsttcx] saved {out}")


if __name__ == "__main__":
    main()
