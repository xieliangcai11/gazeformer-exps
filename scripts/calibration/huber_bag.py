"""Small improvements to the Huber calibration baseline.

Methods:
  huber        : one support set, Huber per axis.
  huber_bag    : average predictions from several support sets.
  huber_bagth  : average the affine parameters from several support sets.
  huber_med    : coordinate-wise median over several support sets.

The goal is variance reduction with almost no new assumptions.
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


def run_method(d, method, K, mu, lam, delta, bag, seed0, bag_mode="pool"):
    n = len(d["pred"])
    rng = np.random.RandomState(seed0)
    sel0 = support_indices(d, K, rng)
    q = np.setdiff1d(np.arange(n), sel0)
    if len(q) < 20:
        return None
    if method == "huber":
        th, _ = fit_theta_huber(d["pred"][sel0], d["label"][sel0], lam, mu, delta)
        q1 = apply_theta_np(d["pred"][q], th)
    else:
        thetas = []
        preds = []
        for s in range(bag):
            if bag_mode == "bootstrap":
                sel = rng.choice(sel0, len(sel0), replace=True)
            else:
                sel = support_indices(d, K, np.random.RandomState(seed0 + 1000 * s))
            th, _ = fit_theta_huber(d["pred"][sel], d["label"][sel], lam, mu, delta)
            thetas.append(th)
            preds.append(apply_theta_np(d["pred"][q], th))
        if method == "huber_bag":
            q1 = np.mean(preds, axis=0)
        elif method == "huber_bagth":
            th = np.mean(thetas, axis=0)
            q1 = apply_theta_np(d["pred"][q], th)
        elif method == "huber_med":
            q1 = np.median(preds, axis=0)
        else:
            raise ValueError(method)
    e0 = em(d["pred"][q], d["label"][q], d["sx"][q], d["sy"][q]).mean()
    e1 = em(q1, d["label"][q], d["sx"][q], d["sy"][q]).mean()
    return float(e0), float(e1), len(q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--val-npz", required=True)
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--pose-npz", required=True)
    ap.add_argument("--context-key", choices=["none", "orientation", "activity"], default="none")
    ap.add_argument("--K", type=int, default=9)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--delta", type=float, default=1.345)
    ap.add_argument("--bag", type=int, default=9)
    ap.add_argument("--bag-mode", choices=["pool", "bootstrap"], default="bootstrap")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    log(f"[bag] start={time.strftime('%F %T')} args={vars(a)}")
    D = load_sessions(a.train_npz, a.index_csv, a.pose_npz, a.context_key)
    T = load_sessions(a.test_npz, a.index_csv, a.pose_npz, a.context_key)
    mu, _ = fit_source_mu(D)
    log(f"[bag] sessions={len(D)}/{len(T)} prior={np.round(mu,4).tolist()}")

    rows = []
    for method in ("huber", "huber_bag", "huber_bagth", "huber_med"):
        acc0 = acc1 = nn = 0.0
        ratios = []
        t0 = time.time()
        for si, (s, d) in enumerate(T.items()):
            r = run_method(d, method, a.K, mu, a.lam, a.delta, a.bag, 90000,
                           a.bag_mode)
            if r is None:
                continue
            e0, e1, n = r
            acc0 += e0 * n
            acc1 += e1 * n
            nn += n
            ratios.append(e1 / max(e0, 1e-12) - 1)
            if si % 10 == 0 or si == len(T) - 1:
                log(f"[bag] {method} session={s} n={n} base={e0:.4f} "
                    f"calib={e1:.4f} ({100*(e1/e0-1):+.2f}%) "
                    f"elapsed={time.time()-t0:.1f}s")
        if not nn:
            continue
        B, E = acc0 / nn, acc1 / nn
        rr = np.array(ratios)
        log(f"[bag] RESULT method={method} K={a.K} base={B:.4f} "
            f"calib={E:.4f} ({100*(E/B-1):+.2f}%) sessions={len(rr)} "
            f"improved={int((rr<0).sum())} worst={100*rr.max():+.1f}%")
        rows.append([method, a.K, B, E, 100 * (E / B - 1), len(rr),
                     int((rr < 0).sum()), float(rr.max())])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "K", "base", "calib", "pct", "sessions",
                    "improved", "worst"])
        w.writerows(rows)
    log(f"[bag] saved {out}")


if __name__ == "__main__":
    main()
