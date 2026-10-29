"""Route 2: calibrate 2D screen gaze with a 3D gaze prior.

Inputs are the 3D directions produced by ``gaze3d_prior.py`` and the same 2D
predictions/labels used by the Huber calibration baseline.

Methods:
  huber    : 2D affine, per-axis Huber (baseline).
  linear3d : affine map [gx, gy, gz, 1] -> (screen_x, screen_y).
  proj     : differentiable screen projection with per-session screen pose.

Only K support labels are used for fitting; the remaining frames are used for
evaluation.  Test support seeds are independent, and every session logs its
base/calibrated error.
"""
import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path[:0] = ["src", "."]
from scripts.calibration.seq_bayes_calib import load_index, em  # noqa: E402
from scripts.calibration.meta_residual_calib import (  # noqa: E402
    fit_source_mu, fit_theta_huber, apply_theta_np, support_indices,
)
from scripts.calibration.seq_bayes_calib import load_sessions  # noqa: E402


def log(msg):
    print(msg, flush=True)


def load_paired(feat_npz, gaze_npz, index_csv, context_key):
    z2 = np.load(feat_npz, allow_pickle=True)
    z3 = np.load(gaze_npz, allow_pickle=True)
    g3 = {str(p): (z3["pred3d"][i].astype(np.float64),
                   z3["origin"][i].astype(np.float64))
          for i, p in enumerate(z3["path"].tolist())}
    pred = z2["pred"].astype(np.float64)
    label = z2["label"].astype(np.float64)
    path = [str(p) for p in z2["path"].tolist()]
    fidx = z2["frame_index"].astype(np.int64)
    meta = load_index(index_csv, np.array(path, object))
    groups = defaultdict(list)
    miss_meta = miss_3d = 0
    for i, p in enumerate(path):
        if p not in meta:
            miss_meta += 1
            continue
        if p not in g3:
            miss_3d += 1
            continue
        ctx = meta[p][4] if context_key != "none" else ""
        groups[f"{p.rsplit('/', 1)[0]}||{ctx}"].append(i)
    out = {}
    for s, idx in groups.items():
        idx = np.array(sorted(idx, key=lambda i: int(fidx[i])), np.int64)
        if len(idx) < 40:
            continue
        gg = np.stack([g3[path[i]][0] for i in idx])
        nn = np.linalg.norm(gg, axis=1, keepdims=True)
        good = nn[:, 0] > 1e-8
        if good.sum() < 40:
            continue
        idx = idx[good]
        gg = gg[good] / nn[good]
        out[s] = dict(
            pred=pred[idx], label=label[idx], g=gg.astype(np.float32),
            origin=np.stack([g3[path[i]][1] for i in idx]).astype(np.float32),
            sx=np.array([meta[path[i]][2] for i in idx]),
            sy=np.array([meta[path[i]][3] for i in idx]),
            ctx=np.array([meta[path[i]][4] for i in idx]),
        )
    log(f"[screen] paired={len(out)} sessions miss_meta={miss_meta} miss_3d={miss_3d}")
    return out


def fit_linear3d(P3, L, lam):
    A = np.concatenate([P3, np.ones((len(P3), 1))], axis=1)
    W = np.linalg.solve(A.T @ A + lam * np.eye(4), A.T @ L)
    return W


def apply_linear3d(P3, W):
    A = np.concatenate([P3, np.ones((len(P3), 1))], axis=1)
    return A @ W


def rodrigues(r):
    th = torch.linalg.norm(r)
    if float(th) < 1e-12:
        return torch.eye(3, dtype=r.dtype, device=r.device)
    k = r / th
    K = torch.zeros(3, 3, dtype=r.dtype, device=r.device)
    K[0, 1], K[0, 2] = -k[2], k[1]
    K[1, 0], K[1, 2] = k[2], -k[0]
    K[2, 0], K[2, 1] = -k[1], k[0]
    return torch.eye(3, dtype=r.dtype, device=r.device) + \
        torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


def project(g, o, r, t, scale, off):
    R = rodrigues(r)
    n = R[:, 2]
    denom = (g * n).sum(1)
    denom = torch.where(denom.abs() < 1e-6,
                        torch.full_like(denom, 1e-6), denom)
    p3 = o + (((t - o) * n).sum(1) / denom)[:, None] * g
    p = (p3 - t) @ R
    return scale * p[:, :2] + off


def fit_projection(g, o, label, g_all, o_all, steps=300, lr=0.05, reg=1e-4):
    """Fit screen pose + 2D scale/offset on support; project all frames."""
    g = torch.tensor(g, dtype=torch.float64)
    o = torch.tensor(o, dtype=torch.float64)
    y = torch.tensor(label, dtype=torch.float64)
    ga = torch.tensor(g_all, dtype=torch.float64)
    oa = torch.tensor(o_all, dtype=torch.float64)
    r = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    t = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64, requires_grad=True)
    scale = torch.ones(2, dtype=torch.float64, requires_grad=True)
    off = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([r, t, scale, off], lr=lr)
    for _ in range(steps):
        q = project(g, o, r, t, scale, off)
        loss = ((q - y) ** 2).mean() + reg * (r ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        q = project(ga, oa, r, t, scale, off)
    return q.numpy()


def evaluate(method, d, sel, mu, lam, delta, proj_steps):
    if len(sel) < 2:
        return None
    q = np.setdiff1d(np.arange(len(d["pred"])), sel)
    if len(q) < 20:
        return None
    theta0, _ = fit_theta_huber(d["pred"][sel], d["label"][sel], lam, mu, delta)
    q0 = apply_theta_np(d["pred"][q], theta0)
    e0 = em(q0, d["label"][q], d["sx"][q], d["sy"][q])
    if method == "huber":
        q1 = q0
    elif method == "linear3d":
        W = fit_linear3d(d["g"][sel], d["label"][sel], 1e-3)
        q1 = apply_linear3d(d["g"][q], W)
    elif method == "proj":
        q1 = fit_projection(d["g"][sel], d["origin"][sel], d["label"][sel],
                            d["g"], d["origin"], steps=proj_steps)
        q1 = q1[q]
    else:
        raise ValueError(method)
    e1 = em(q1, d["label"][q], d["sx"][q], d["sy"][q])
    return float(e0.mean()), float(e1.mean()), len(q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--gaze3d-npz", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--pose-npz", required=True)
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--context-key", choices=["none", "orientation", "activity"], default="none")
    ap.add_argument("--K", type=int, default=9)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=1.345)
    ap.add_argument("--proj-steps", type=int, default=300)
    ap.add_argument("--methods", nargs="+", default=["huber", "linear3d"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    log(f"[screen] start={time.strftime('%F %T')} args={vars(a)}")
    D = load_sessions(a.train_npz, a.index_csv, a.pose_npz, a.context_key)
    mu, TH = fit_source_mu(D)
    log(f"[screen] source groups={len(D)} theta_mean={np.round(mu,4).tolist()}")
    T = load_paired(a.test_npz, a.gaze3d_npz, a.index_csv, a.context_key)
    if not T:
        raise RuntimeError("no paired sessions")
    ns = [len(d["pred"]) for d in T.values()]
    log(f"[screen] test sessions={len(T)} frames={sum(ns)} "
        f"n[min/med/max]={min(ns)}/{int(np.median(ns))}/{max(ns)}")

    rows = []
    for method in a.methods:
        acc0 = acc1 = nn = 0.0
        ratios = []
        t0 = time.time()
        for si, (s, d) in enumerate(T.items()):
            r0 = r1 = rn = 0.0
            for seed in range(a.seeds):
                rng = np.random.RandomState(90000 + seed + 1000 * hash(s) % 1000)
                sel = support_indices(d, a.K, rng)
                r = evaluate(method, d, sel, mu, a.lam, a.delta, a.proj_steps)
                if r is None:
                    continue
                r0 += r[0] * r[2]
                r1 += r[1] * r[2]
                rn += r[2]
            if rn:
                b, e = r0 / rn, r1 / rn
                acc0 += b * rn
                acc1 += e * rn
                nn += rn
                ratios.append(e / max(b, 1e-12) - 1.0)
                if si % 10 == 0 or si == len(T) - 1:
                    log(f"[screen] {method} session={s} n={rn:.0f} "
                        f"base={b:.4f} calib={e:.4f} ({100*(e/b-1):+.2f}%) "
                        f"elapsed={time.time()-t0:.1f}s")
        if not nn:
            log(f"[screen] WARNING no results method={method}")
            continue
        B, E = acc0 / nn, acc1 / nn
        rr = np.array(ratios)
        log(f"[screen] RESULT method={method} base={B:.4f} calib={E:.4f} "
            f"({100*(E/B-1):+.2f}%) sessions={len(rr)} "
            f"improved={int((rr<0).sum())} degrade2={int((rr>0.02).sum())} "
            f"worst={100*rr.max():+.1f}%")
        rows.append([method, B, E, 100 * (E / B - 1), len(rr),
                     int((rr < 0).sum()), int((rr > 0.02).sum()), float(rr.max())])
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "base", "calib", "pct", "sessions",
                    "improved", "degrade2", "worst"])
        w.writerows(rows)
    log(f"[screen] saved {out}")


if __name__ == "__main__":
    main()
