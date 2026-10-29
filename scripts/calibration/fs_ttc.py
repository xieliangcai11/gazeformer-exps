"""FS-TTC-style identity-regularized ridge calibration with LOO selection.

A modern (2026) output-correction baseline: fit a per-axis affine ridge toward
an identity/prior mapping, choose the correction strength by leave-one-out on
the K support points, then evaluate held-out frames.

This is intentionally *only* a calibration baseline; no prompt/model changes.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path[:0] = ["src", "."]
from scripts.calibration.seq_bayes_calib import load_sessions, em  # noqa: E402
from scripts.calibration.robust_calib import fit_ridge  # noqa: E402
from scripts.calibration.meta_residual_calib import (  # noqa: E402
    fit_source_mu, fit_theta_huber, apply_theta_np, support_indices,
)


def log(msg):
    print(msg, flush=True)


def fit_affine(P, L, lam, mu):
    tx = fit_ridge(P[:, 0], L[:, 0], lam, np.asarray(mu[:2]))
    ty = fit_ridge(P[:, 1], L[:, 1], lam, np.asarray(mu[2:]))
    return np.array([tx[0], tx[1], ty[0], ty[1]])


def loo_em(P, L, sx, sy, lam, mu):
    n = len(P)
    errs = []
    for j in range(n):
        m = np.ones(n, bool)
        m[j] = False
        th = fit_affine(P[m], L[m], lam, mu)
        q = apply_theta_np(P[j:j + 1], th)
        errs.append(float(em(q, L[j:j + 1], sx[j:j + 1], sy[j:j + 1])[0]))
    return float(np.mean(errs))


def eval_preds(d, sel, th, method):
    q = np.setdiff1d(np.arange(len(d["pred"])), sel)
    if len(q) < 20:
        return None
    q1 = apply_theta_np(d["pred"][q], th)
    e = em(q1, d["label"][q], d["sx"][q], d["sy"][q]).mean()
    e0 = em(d["pred"][q], d["label"][q],
            d["sx"][q], d["sy"][q]).mean()
    return float(e0), float(e), q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--val-npz", required=True)
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--pose-npz", required=True)
    ap.add_argument("--context-key", choices=["none", "orientation", "activity"], default="none")
    ap.add_argument("--K", type=int, default=9)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--lams", type=float, nargs="+",
                    default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
    ap.add_argument("--prior", choices=["source", "identity"], default="identity")
    ap.add_argument("--huber-lam", type=float, default=0.1)
    ap.add_argument("--huber-delta", type=float, default=1.345)
    ap.add_argument("--detail", default="", help="optional per-session CSV")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    log(f"[fsttc] start={time.strftime('%F %T')} args={vars(a)}")
    D = load_sessions(a.train_npz, a.index_csv, a.pose_npz, a.context_key)
    V = load_sessions(a.val_npz, a.index_csv, a.pose_npz, a.context_key)
    T = load_sessions(a.test_npz, a.index_csv, a.pose_npz, a.context_key)
    mu_src, _ = fit_source_mu(D)
    mu = np.array([1.0, 0.0, 1.0, 0.0]) if a.prior == "identity" else mu_src
    log(f"[fsttc] sessions={len(D)}/{len(V)}/{len(T)} prior={np.round(mu,4).tolist()}")

    rows = []
    detail = []
    for method in ("huber", "fsttc"):
        acc0 = acc1 = nn = 0.0
        ratios = []
        chosen = []
        t0 = time.time()
        for si, (s, d) in enumerate(T.items()):
            r0 = r1 = rn = 0.0
            lams_used = []
            for seed in range(a.seeds):
                rng = np.random.RandomState(90000 + seed)
                sel = support_indices(d, a.K, rng)
                if method == "fsttc":
                    vals = [loo_em(d["pred"][sel], d["label"][sel],
                                   d["sx"][sel], d["sy"][sel], lam, mu)
                            for lam in a.lams]
                    lam = a.lams[int(np.argmin(vals))]
                    lams_used.append(lam)
                    th = fit_affine(d["pred"][sel], d["label"][sel], lam, mu)
                else:
                    th, _ = fit_theta_huber(d["pred"][sel], d["label"][sel],
                                            a.huber_lam, mu, a.huber_delta)
                r = eval_preds(d, sel, th, method)
                if r is None:
                    continue
                e0, e1, q = r
                r0 += e0 * len(q)
                r1 += e1 * len(q)
                rn += len(q)
            if rn:
                b, e = r0 / rn, r1 / rn
                acc0 += b * rn
                acc1 += e * rn
                nn += rn
                ratios.append(e / max(b, 1e-12) - 1)
                if lams_used:
                    chosen.append(float(np.mean(lams_used)))
                detail.append([method, s, rn, b, e, 100 * (e / b - 1),
                               float(np.mean(lams_used)) if lams_used else -1])
                if si % 5 == 0 or si == len(T) - 1:
                    log(f"[fsttc] {method} session={s} n={rn:.0f} "
                        f"base={b:.4f} calib={e:.4f} ({100*(e/b-1):+.2f}%) "
                        f"lam={np.mean(lams_used) if lams_used else -1:g} "
                        f"elapsed={time.time()-t0:.1f}s")
        if not nn:
            continue
        B, E = acc0 / nn, acc1 / nn
        rr = np.array(ratios)
        log(f"[fsttc] RESULT method={method} K={a.K} base={B:.4f} "
            f"calib={E:.4f} ({100*(E/B-1):+.2f}%) sessions={len(rr)} "
            f"improved={int((rr<0).sum())} worst={100*rr.max():+.1f}% "
            f"mean_lam={np.mean(chosen) if chosen else -1:g}")
        rows.append([method, a.K, B, E, 100 * (E / B - 1),
                     len(rr), int((rr < 0).sum()), float(rr.max()),
                     float(np.mean(chosen)) if chosen else -1])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "K", "base", "calib", "pct", "sessions",
                    "improved", "worst", "mean_lam"])
        w.writerows(rows)
    if a.detail:
        dp = Path(a.detail)
        dp.parent.mkdir(parents=True, exist_ok=True)
        with open(dp, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["method", "session", "n", "base", "calib", "pct", "lam"])
            w.writerows(detail)
    log(f"[fsttc] saved {out}")


if __name__ == "__main__":
    main()
