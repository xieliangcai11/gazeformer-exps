"""Diagnose whether Huber residual has spatial / pose structure.

For each held-out session:
  - fit Huber affine on K coverage support points
  - compute signed residual in cm on the remaining frames
  - relate residual to predicted screen position, radial distance, pose,
    screen size, and context

If residual is structured, local / conditional calibration can help; if it is
white noise, a 4-parameter affine is already near the information limit.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path[:0] = ["src", "."]
from scripts.calibration.seq_bayes_calib import load_sessions  # noqa: E402
from scripts.calibration.meta_residual_calib import (  # noqa: E402
    fit_source_mu, fit_theta_huber, apply_theta_np, support_indices,
)


def log(msg):
    print(msg, flush=True)


def collect(sessions, K, mu, lam, delta, seed=90000):
    rows = []
    for s, d in sessions.items():
        sel = support_indices(d, K, np.random.RandomState(seed))
        q = np.setdiff1d(np.arange(len(d["pred"])), sel)
        if len(q) < 20:
            continue
        th, _ = fit_theta_huber(d["pred"][sel], d["label"][sel], lam, mu, delta)
        p = apply_theta_np(d["pred"][q], th)
        rx = (p[:, 0] - d["label"][q, 0]) * d["sx"][q]
        ry = (p[:, 1] - d["label"][q, 1]) * d["sy"][q]
        pose = d["pose"][q]
        rows.append(dict(
            session=s, rx=rx, ry=ry,
            px=p[:, 0], py=p[:, 1],
            lx=d["label"][q, 0], ly=d["label"][q, 1],
            pose=pose,
            sw=d["w"][q], sh=d["h"][q],
            ctx=np.array([str(c) for c in d["ctx"][q]]),
        ))
    return rows


def summarize(rows, name):
    if not rows:
        log(f"[res] {name}: no data")
        return
    rx = np.concatenate([r["rx"] for r in rows])
    ry = np.concatenate([r["ry"] for r in rows])
    px = np.concatenate([r["px"] for r in rows])
    py = np.concatenate([r["py"] for r in rows])
    lx = np.concatenate([r["lx"] for r in rows])
    ly = np.concatenate([r["ly"] for r in rows])
    pose = np.concatenate([r["pose"] for r in rows])
    sw = np.concatenate([r["sw"] for r in rows])
    sh = np.concatenate([r["sh"] for r in rows])
    log(f"[res] {name} frames={len(rx)} "
        f"|r| mean={np.hypot(rx,ry).mean():.4f} "
        f"rx mean/std={rx.mean():+.4f}/{rx.std():.4f} "
        f"ry mean/std={ry.mean():+.4f}/{ry.std():.4f}")
    feats = {
        "px": px, "py": py, "lx": lx, "ly": ly,
        "radial": np.hypot(lx - 0.5, ly - 0.5),
        "pose0": pose[:, 0], "pose1": pose[:, 1], "pose2": pose[:, 2],
        "pose4": pose[:, 4], "sw": sw, "sh": sh,
    }
    for fn, f in feats.items():
        c_x = np.corrcoef(f, rx)[0, 1] if f.std() > 1e-12 else 0.0
        c_y = np.corrcoef(f, ry)[0, 1] if f.std() > 1e-12 else 0.0
        c_n = np.corrcoef(f, np.hypot(rx, ry))[0, 1] if f.std() > 1e-12 else 0.0
        log(f"[res] corr({fn}) rx={c_x:+.3f} ry={c_y:+.3f} |r|={c_n:+.3f}")
    # Binned radial magnitude
    rad = feats["radial"]
    qs = np.quantile(rad, np.linspace(0, 1, 6))
    for i in range(5):
        m = (rad >= qs[i]) & (rad <= qs[i + 1])
        if m.sum() < 10:
            continue
        log(f"[res] radial_bin{i} n={int(m.sum())} "
            f"rad[{qs[i]:.3f},{qs[i+1]:.3f}] "
            f"|r|={np.hypot(rx[m],ry[m]).mean():.4f} "
            f"rx={rx[m].mean():+.4f} ry={ry[m].mean():+.4f}")
    # Context split
    ctx = np.concatenate([r["ctx"] for r in rows])
    for c in sorted(set(ctx.tolist())):
        m = ctx == c
        if m.sum() < 10:
            continue
        log(f"[res] ctx={c} n={int(m.sum())} "
            f"|r|={np.hypot(rx[m],ry[m]).mean():.4f} "
            f"rx={rx[m].mean():+.4f} ry={ry[m].mean():+.4f}")
    # Linear model R^2 on residual from features
    X = np.stack([np.ones_like(px), px, py, lx, ly, rad,
                  pose[:, 0], pose[:, 1], pose[:, 2], pose[:, 4]], 1)
    for tgt, nm in ((rx, "rx"), (ry, "ry")):
        w = np.linalg.lstsq(X, tgt, rcond=None)[0]
        pred = X @ w
        r2 = 1 - ((tgt - pred) ** 2).sum() / max(((tgt - tgt.mean()) ** 2).sum(), 1e-12)
        log(f"[res] linear_{nm}_R2={r2:.4f}")


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
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    log(f"[res] start={time.strftime('%F %T')} args={vars(a)}")
    D = load_sessions(a.train_npz, a.index_csv, a.pose_npz, a.context_key)
    S = load_sessions(a.val_npz if a.split == "val" else a.test_npz,
                      a.index_csv, a.pose_npz, a.context_key)
    mu, _ = fit_source_mu(D)
    log(f"[res] source={len(D)} target={len(S)} K={a.K} "
        f"prior={np.round(mu,4).tolist()}")
    rows = collect(S, a.K, mu, a.lam, a.delta)
    summarize(rows, f"{a.split} K={a.K}")
    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["session", "n", "rx_mean", "ry_mean",
                        "rx_std", "ry_std", "abs_mean"])
            for r in rows:
                w.writerow([r["session"], len(r["rx"]), r["rx"].mean(),
                            r["ry"].mean(), r["rx"].std(), r["ry"].std(),
                            np.hypot(r["rx"], r["ry"]).mean()])
        log(f"[res] saved {out}")


if __name__ == "__main__":
    main()
