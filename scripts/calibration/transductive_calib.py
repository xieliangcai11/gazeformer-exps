"""Target-unlabeled transductive calibration experiments.

Only K support labels are supervised.  Unlabeled target frames are used by
three transductive rules:

  quantile_src : match each axis' prediction quantiles to source label quantiles.
  quantile_uni : match each axis' prediction quantiles to uniform [0,1].
  temporal     : optimize the affine parameters with a small temporal
                 smoothness penalty on the unlabeled sequence.

These are deliberately strong "free lunch" baselines: if they fail, unlabeled
frames alone do not identify the calibration.  All methods start from the same
per-session Huber affine fit.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path[:0] = ["src", "."]
from scripts.calibration.seq_bayes_calib import load_sessions, em  # noqa: E402
from scripts.calibration.meta_residual_calib import (  # noqa: E402
    fit_source_mu, fit_theta_huber, apply_theta_np, support_indices,
)


def log(msg):
    print(msg, flush=True)


def quantile_map(x, src_q, dst_q):
    src_q = np.asarray(src_q)
    dst_q = np.asarray(dst_q)
    order = np.argsort(src_q)
    src_q = src_q[order]
    dst_q = dst_q[order]
    return np.interp(x, src_q, dst_q)


def temporal_theta(d, sel, theta0, lam_s, steps=50):
    P = torch.tensor(d["pred"], dtype=torch.float64)
    L = torch.tensor(d["label"], dtype=torch.float64)
    S = torch.tensor(sel)
    sx = torch.tensor(d["sx"], dtype=torch.float64)
    sy = torch.tensor(d["sy"], dtype=torch.float64)
    th = torch.tensor(theta0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([th], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        q = P.clone()
        q[:, 0] = th[0] * P[:, 0] + th[1]
        q[:, 1] = th[2] * P[:, 1] + th[3]
        e = ((q[S, 0] - L[S, 0]) * sx[S]) ** 2 + \
            ((q[S, 1] - L[S, 1]) * sy[S]) ** 2
        sup = e.mean()
        dif = q[1:] - q[:-1]
        dist = torch.sqrt(dif[:, 0] ** 2 + dif[:, 1] ** 2 + 1e-12)
        # Huber-like cap: a real saccade should not dominate.
        smooth = torch.where(dist < 0.05, dist, 0.05 + 0.1 * (dist - 0.05)).mean()
        loss = sup + lam_s * smooth + 1e-3 * ((th - torch.tensor(theta0)) ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return th.detach().numpy()


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
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=1.345)
    ap.add_argument("--smooth-lam", type=float, default=0.1)
    ap.add_argument("--methods", nargs="+",
                    default=["huber", "quantile_src", "quantile_uni", "temporal"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    log(f"[trans] start={time.strftime('%F %T')} args={vars(a)}")
    D = load_sessions(a.train_npz, a.index_csv, a.pose_npz, a.context_key)
    V = load_sessions(a.val_npz, a.index_csv, a.pose_npz, a.context_key)
    T = load_sessions(a.test_npz, a.index_csv, a.pose_npz, a.context_key)
    mu, TH = fit_source_mu(D)
    srcL = np.concatenate([d["label"] for d in D.values()], axis=0)
    qs = np.linspace(0, 1, 21)
    src_q = [np.quantile(srcL[:, a], qs) for a in range(2)]
    log(f"[trans] groups={len(D)}/{len(V)}/{len(T)} K={a.K} seeds={a.seeds} "
        f"source_labels={len(srcL)} src_mean={np.round(srcL.mean(0),4).tolist()}")
    log(f"[trans] source theta mean={np.round(mu,4).tolist()}")

    rows = []
    for method in a.methods:
        acc0 = acc1 = nn = 0.0
        ratios = []
        t0 = time.time()
        for si, (s, d) in enumerate(T.items()):
            r0 = r1 = rn = 0.0
            for seed in range(a.seeds):
                rng = np.random.RandomState(90000 + seed)
                sel = support_indices(d, a.K, rng)
                q = np.setdiff1d(np.arange(len(d["pred"])), sel)
                if len(q) < 20:
                    continue
                theta0, _ = fit_theta_huber(d["pred"][sel], d["label"][sel],
                                            a.lam, mu, a.delta)
                q0 = apply_theta_np(d["pred"][q], theta0)
                if method == "huber":
                    q1 = q0
                elif method == "quantile_src":
                    q1 = q0.copy()
                    for ax in range(2):
                        q1[:, ax] = quantile_map(
                            q0[:, ax], np.quantile(q0[:, ax], qs), src_q[ax]
                        )
                elif method == "quantile_uni":
                    q1 = q0.copy()
                    for ax in range(2):
                        q1[:, ax] = quantile_map(q0[:, ax], np.quantile(q0[:, ax], qs), qs)
                elif method == "temporal":
                    th = temporal_theta(d, sel, theta0, a.smooth_lam)
                    q1 = apply_theta_np(d["pred"][q], th)
                else:
                    raise ValueError(method)
                e0 = em(q0, d["label"][q], d["sx"][q], d["sy"][q]).mean()
                e1 = em(q1, d["label"][q], d["sx"][q], d["sy"][q]).mean()
                r0 += e0 * len(q)
                r1 += e1 * len(q)
                rn += len(q)
                ratios.append(e1 / max(e0, 1e-12) - 1)
            if rn:
                b, e = r0 / rn, r1 / rn
                acc0 += b * rn
                acc1 += e * rn
                nn += rn
                if si % 5 == 0 or si == len(T) - 1:
                    log(f"[trans] {method} session={s} n={rn:.0f} "
                        f"base={b:.4f} calib={e:.4f} ({100*(e/b-1):+.2f}%) "
                        f"elapsed={time.time()-t0:.1f}s")
        if not nn:
            log(f"[trans] WARNING empty method={method}")
            continue
        B, E = acc0 / nn, acc1 / nn
        rr = np.array(ratios)
        log(f"[trans] RESULT method={method} base={B:.4f} calib={E:.4f} "
            f"({100*(E/B-1):+.2f}%) sessions={len(rr)} "
            f"improved={int((rr<0).sum())} degrade2={int((rr>0.02).sum())} "
            f"worst={100*rr.max():+.1f}%")
        rows.append([method, a.K, B, E, 100 * (E / B - 1), len(rr),
                     int((rr < 0).sum()), int((rr > 0.02).sum()), float(rr.max())])

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "K", "base", "calib", "pct", "sessions",
                    "improved", "degrade2", "worst"])
        w.writerows(rows)
    log(f"[trans] saved {out}")


if __name__ == "__main__":
    main()
