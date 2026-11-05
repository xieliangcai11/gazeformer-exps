"""Method 1: source-prior posterior adapter + closed-loop active selection.

Source: learn residual adapter prior W0 from photo features.
Target: start with anchors, then update posterior adapter after every selected point;
choose next point by predicted residual + leverage + feature/space novelty.
Val development only; test is not touched here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np

sys.path[:0] = ["src", "."]
from gazelab.datasets.table_index import TableIndex  # noqa: E402
from scripts.calibration.active_calib_sim import (  # noqa: E402
    build_dot_table,
    select_dot_coverage,
    subject_of,
)
from scripts.calibration.cm_residual_calib import (  # noqa: E402
    apply_affine,
    build_feature_matrix,
    fit_huber,
    load_feature_source,
    load_sessions,
    to_cm,
)

_T0 = time.time()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time()-_T0:7.1f}s] {msg}", flush=True)


def base_theta(dots):
    P = np.stack([x["pred"] for x in dots])
    U = np.stack([x["label"] for x in dots])
    th = fit_huber(P, U, 1.0, np.array([1., 0., 1., 0.]), 1.345,
                   scale_mode="axis")
    return th


def cm(d, dots, pred):
    pos = np.array([x["pos"][0] for x in dots], np.int64)
    q = to_cm(pred, d["scm"][pos], d["phi"][pos])
    t = to_cm(np.stack([x["label"] for x in dots]), d["scm"][pos], d["phi"][pos])
    return float(np.linalg.norm(q - t, axis=1).mean())


def fit_w(Z, R, W0, lam):
    A = Z.T @ Z + float(lam) * np.eye(Z.shape[1])
    return np.linalg.solve(A, Z.T @ R + float(lam) * W0), np.linalg.inv(A)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-npz", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--dino-npz", required=True)
    ap.add_argument("--landmark-npz", required=True)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--anchor", type=int, default=4)
    ap.add_argument("--source-lam", type=float, default=10.0)
    ap.add_argument("--target-lam", type=float, default=1.0)
    ap.add_argument("--feat-weight", type=float, default=0.2)
    ap.add_argument("--cov-weight", type=float, default=0.1)
    ap.add_argument("--lev-weight", type=float, default=0.2)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    log(f"args={vars(a)}")
    table = TableIndex(a.index_csv)
    D = load_sessions(a.val_npz, table, "orientation", 40)
    dsrc = load_feature_source(a.dino_npz, "dino")
    lsrc = load_feature_source(a.landmark_npz, "lm")
    F, V, names = build_feature_matrix([dsrc, lsrc], len(table))
    dotargs = SimpleNamespace(min_dot_frames=2, dot_split=0.65, split_seed=0)
    train, valid = {}, {}
    for s, d in D.items():
        tab = build_dot_table(s, d, table, F, V, dotargs)
        if tab is None:
            continue
        if zlib.crc32(subject_of(s).encode("utf-8")) % 2 == 0:
            train[s] = (d, tab)
        else:
            valid[s] = (d, tab)
    log(f"features={len(names)} train={len(train)} valid={len(valid)}")

    allf = []
    for d, tab in train.values():
        for x in tab["cand"]:
            allf.append(x["feat"])
    allf = np.asarray(allf, np.float64)
    fmu = allf.mean(0); fsd = allf.std(0) + 1e-6

    Zs, Rs = [], []
    for d, tab in train.values():
        dots, _ = select_dot_coverage(tab["cand"], a.K, 0)
        if len(dots) < 4:
            continue
        P = np.stack([x["pred"] for x in dots])
        U = np.stack([x["label"] for x in dots])
        Z = (np.stack([x["feat"] for x in dots]) - fmu) / fsd
        th = base_theta(dots)
        Zs.append(Z); Rs.append(U - apply_affine(P, th))
    Zs = np.concatenate(Zs, 0); Rs = np.concatenate(Rs, 0)
    W0, _ = fit_w(Zs, Rs, np.zeros((Zs.shape[1], 2)), a.source_lam)
    log(f"source rows={len(Zs)} W0={W0.shape}")

    rows = []
    for s, (d, tab) in valid.items():
        # start anchors
        anchors, _ = select_dot_coverage(tab["cand"], a.anchor, 0)
        sel = list(anchors)
        while len(sel) < a.K:
            P = np.stack([x["pred"] for x in sel])
            U = np.stack([x["label"] for x in sel])
            Z = (np.stack([x["feat"] for x in sel]) - fmu) / fsd
            th = base_theta(sel)
            R = U - apply_affine(P, th)
            Wt, Ainv = fit_w(Z, R, W0, a.target_lam)
            best = None; best_dot = None
            for dot in tab["cand"]:
                if any(dot is x for x in sel):
                    continue
                z = (dot["feat"] - fmu) / fsd
                rn = float(np.linalg.norm(z @ Wt))
                lev = float(z @ Ainv @ z)
                fd = min(float(np.linalg.norm(z - ((x["feat"] - fmu) / fsd)))
                         for x in sel)
                pd = min(float(np.linalg.norm(dot["pred"] - x["pred"])) for x in sel)
                u = rn + a.lev_weight * lev + a.feat_weight * fd + a.cov_weight * pd
                if best is None or u > best:
                    best, best_dot = u, dot
            if best_dot is None:
                break
            sel.append(best_dot)
        P = np.stack([x["pred"] for x in sel])
        U = np.stack([x["label"] for x in sel])
        Z = (np.stack([x["feat"] for x in sel]) - fmu) / fsd
        th = base_theta(sel)
        Wt, _ = fit_w(Z, U - apply_affine(P, th), W0, a.target_lam)
        QP = np.stack([x["pred"] for x in tab["held"]])
        QF = (np.stack([x["feat"] for x in tab["held"]]) - fmu) / fsd
        base_pred = apply_affine(QP, th)
        pred = base_pred + QF @ Wt
        rows.append((s, cm(d, tab["held"], base_pred), cm(d, tab["held"], pred)))
    b = float(np.mean([x[1] for x in rows]))
    c = float(np.mean([x[2] for x in rows]))
    win = float(np.mean([x[2] < x[1] for x in rows]))
    log(f"RESULT sessions={len(rows)} base={b:.4f} adapter={c:.4f} "
        f"delta={c-b:+.4f} win={win:.3f}")
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(dict(args=vars(a), sessions=len(rows), base=b, adapter=c,
                       delta=c-b, win=win, source_rows=len(Zs)),
                  fh, ensure_ascii=False, indent=2)
    with open(out / "detail.csv", "w", encoding="utf-8") as fh:
        fh.write("session,base_cm,adapter_cm\n")
        for s, x, y in rows:
            fh.write(f"{s},{x:.6f},{y:.6f}\n")


if __name__ == "__main__":
    main()
