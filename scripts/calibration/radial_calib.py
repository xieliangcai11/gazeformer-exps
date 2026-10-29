"""Radial / quadratic correction on top of the Huber affine calibration.

The residual diagnostic showed structure:
  - GazeCapture: |residual| grows with radial distance from screen centre.
  - RGBDGaze: residual depends on screen position, activity, eye openness.

This script keeps the Huber affine as the base and fits a small ridge
correction on the K support points:

    q = theta_huber(p) + W^T z(p)

where z contains radial and/or quadratic terms.  The correction strength is
chosen by leave-one-out on the support set.  Nothing is retrained.
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


def features(P, mode):
    x = P[:, 0]
    y = P[:, 1]
    r = np.hypot(x - 0.5, y - 0.5)
    if mode == "radial":
        Z = np.stack([x, y, np.ones_like(x), r], 1)
    elif mode == "quad":
        Z = np.stack([x, y, np.ones_like(x), r,
                      x * x, y * y, x * y, r * r], 1)
    else:
        raise ValueError(mode)
    return Z


def fit_corr(Z, R, lam):
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    return np.linalg.solve(A, Z.T @ R)


def loo_choose(Z, R, lams):
    n = len(R)
    best = (None, float("inf"))
    for lam in lams:
        errs = []
        for j in range(n):
            m = np.ones(n, bool)
            m[j] = False
            if m.sum() < 3:
                continue
            w = fit_corr(Z[m], R[m], lam)
            errs.append(float(np.linalg.norm(Z[j] @ w - R[j])))
        if errs:
            e = float(np.mean(errs))
            if e < best[1]:
                best = (lam, e)
    return best[0]


def eval_case(d, sel, mode, mu, lam, delta, lams):
    q = np.setdiff1d(np.arange(len(d["pred"])), sel)
    if len(q) < 20:
        return None
    th, _ = fit_theta_huber(d["pred"][sel], d["label"][sel], lam, mu, delta)
    q0 = apply_theta_np(d["pred"][q], th)
    base = float(em(q0, d["label"][q], d["sx"][q], d["sy"][q]).mean())
    if mode == "huber":
        return base, base, 0.0, len(q)
    Zs = features(d["pred"][sel], mode)
    Rs = d["label"][sel] - apply_theta_np(d["pred"][sel], th)
    best_lam = loo_choose(Zs, Rs, lams)
    w = fit_corr(Zs, Rs, best_lam)
    Ze = features(d["pred"][q], mode)
    q1 = q0 + Ze @ w
    e1 = float(em(q1, d["label"][q], d["sx"][q], d["sy"][q]).mean())
    return base, e1, float(best_lam), len(q)


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
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lams", type=float, nargs="+",
                    default=[1e-3, 1e-2, 1e-1, 1.0, 10.0])
    ap.add_argument("--methods", nargs="+", default=["huber", "radial", "quad"])
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    log(f"[rad] start={time.strftime('%F %T')} args={vars(a)}")
    D = load_sessions(a.train_npz, a.index_csv, a.pose_npz, a.context_key)
    S = load_sessions(a.val_npz if a.split == "val" else a.test_npz,
                      a.index_csv, a.pose_npz, a.context_key)
    mu, _ = fit_source_mu(D)
    log(f"[rad] source={len(D)} eval={len(S)} K={a.K} "
        f"prior={np.round(mu,4).tolist()}")

    rows = []
    for method in a.methods:
        acc0 = acc1 = nn = 0.0
        ratios = []
        chosen = []
        t0 = time.time()
        for si, (s, d) in enumerate(S.items()):
            r0 = r1 = rn = 0.0
            for seed in range(a.seeds):
                rng = np.random.RandomState(90000 + seed)
                sel = support_indices(d, a.K, rng)
                r = eval_case(d, sel, method, mu, a.lam, a.delta, a.lams)
                if r is None:
                    continue
                b, e, lam_used, n = r
                r0 += b * n
                r1 += e * n
                rn += n
                if method != "huber":
                    chosen.append(lam_used)
            if rn:
                b, e = r0 / rn, r1 / rn
                acc0 += b * rn
                acc1 += e * rn
                nn += rn
                ratios.append(e / max(b, 1e-12) - 1)
                if si % 5 == 0 or si == len(S) - 1:
                    log(f"[rad] {method} session={s} n={rn:.0f} "
                        f"base={b:.4f} calib={e:.4f} ({100*(e/b-1):+.2f}%) "
                        f"elapsed={time.time()-t0:.1f}s")
        if not nn:
            continue
        B, E = acc0 / nn, acc1 / nn
        rr = np.array(ratios)
        log(f"[rad] RESULT method={method} K={a.K} base={B:.4f} "
            f"calib={E:.4f} ({100*(E/B-1):+.2f}%) sessions={len(rr)} "
            f"improved={int((rr<0).sum())} worst={100*rr.max():+.1f}% "
            f"mean_lam={np.mean(chosen) if chosen else -1:g}")
        rows.append([method, a.K, B, E, 100 * (E / B - 1), len(rr),
                     int((rr < 0).sum()), float(rr.max()),
                     float(np.mean(chosen)) if chosen else -1])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "K", "base", "calib", "pct", "sessions",
                    "improved", "worst", "mean_lam"])
        w.writerows(rows)
    log(f"[rad] saved {out}")


if __name__ == "__main__":
    main()
