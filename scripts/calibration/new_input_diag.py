"""Correlate the new landmark/eye features with Huber residual.

Aligns landmark features (row index in index.csv) with the same sessions used
by the calibration scripts, fits Huber on K support points, and reports:
  - per-feature correlation with signed/absolute residual
  - binned residual vs iris relative position / eye aspect / blur
"""
import argparse
import csv
import sys
from collections import defaultdict

import numpy as np

sys.path[:0] = ["src", "."]
from scripts.calibration.seq_bayes_calib import load_index  # noqa: E402
from scripts.calibration.meta_residual_calib import (  # noqa: E402
    fit_source_mu, fit_theta_huber, apply_theta_np, support_indices,
)


def log(msg):
    print(msg, flush=True)


def load_sessions_rows(feat_npz, index_csv, context_key, path_to_row):
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
            rows=np.array([path_to_row.get(path[i], -1) for i in idx], np.int64),
        )
    return out


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
    ap.add_argument("--split", choices=["val", "test"], default="val")
    a = ap.parse_args()

    log(f"[ndiag] args={vars(a)}")
    rows = list(csv.DictReader(open(a.index_csv, encoding="utf-8")))
    pcol = "frame_path" if rows and "frame_path" in rows[0] else "rgb_path"
    path_to_row = {r[pcol]: i for i, r in enumerate(rows)}
    z = np.load(a.landmark_npz, allow_pickle=True)
    names = [str(n) for n in z["names"].tolist()]
    row_feat = np.full((len(rows), len(names)), np.nan, np.float32)
    row_valid = np.zeros(len(rows), bool)
    row_feat[z["row_idx"].astype(np.int64)] = z["feats"]
    row_valid[z["row_idx"].astype(np.int64)] = z["valid"]
    log(f"[ndiag] index_rows={len(rows)} landmark_valid={int(row_valid.sum())} "
        f"({100*row_valid.mean():.2f}%) dims={len(names)}")

    D = load_sessions_rows(a.train_npz, a.index_csv, a.context_key, path_to_row)
    S = load_sessions_rows(a.val_npz if a.split == "val" else a.test_npz,
                           a.index_csv, a.context_key, path_to_row)
    mu, _ = fit_source_mu({k: {"pred": d["pred"], "label": d["label"]}
                           for k, d in D.items()})
    log(f"[ndiag] source={len(D)} eval={len(S)} K={a.K}")

    # collect residual + features on held-out frames
    all_r = []
    all_f = []
    all_v = []
    for s, d in S.items():
        sel = support_indices(d, a.K, np.random.RandomState(90000))
        q = np.setdiff1d(np.arange(len(d["pred"])), sel)
        if len(q) < 20:
            continue
        th, _ = fit_theta_huber(d["pred"][sel], d["label"][sel],
                                a.lam, mu, a.delta)
        p = apply_theta_np(d["pred"][q], th)
        rx = (p[:, 0] - d["label"][q, 0]) * d["sx"][q]
        ry = (p[:, 1] - d["label"][q, 1]) * d["sy"][q]
        f = row_feat[d["rows"][q]]
        v = row_valid[d["rows"][q]]
        all_r.append(np.stack([rx, ry, np.hypot(rx, ry)], 1))
        all_f.append(f)
        all_v.append(v)
    R = np.concatenate(all_r)
    F = np.concatenate(all_f)
    V = np.concatenate(all_v)
    log(f"[ndiag] frames={len(R)} valid_feat={int(V.sum())} "
        f"|r| mean={R[:, 2].mean():.4f}")

    for j, n in enumerate(names):
        m = V & np.isfinite(F[:, j])
        if m.sum() < 50 or F[m, j].std() < 1e-9:
            continue
        c0 = np.corrcoef(F[m, j], R[m, 0])[0, 1]
        c1 = np.corrcoef(F[m, j], R[m, 1])[0, 1]
        c2 = np.corrcoef(F[m, j], R[m, 2])[0, 1]
        log(f"[ndiag] corr {n:12s} rx={c0:+.3f} ry={c1:+.3f} |r|={c2:+.3f}")

    # binned analysis for key features
    for key in ("iris_lx", "iris_ly", "iris_rx", "iris_ry",
                "eye_l_h", "ear_l", "blur", "face_w"):
        if key not in names:
            continue
        j = names.index(key)
        m = V & np.isfinite(F[:, j])
        if m.sum() < 100:
            continue
        x = F[m, j]
        qs = np.quantile(x, np.linspace(0, 1, 5))
        for b in range(4):
            mm = (x >= qs[b]) & (x <= qs[b + 1])
            if mm.sum() < 20:
                continue
            log(f"[ndiag] bin {key}[{qs[b]:+.3f},{qs[b+1]:+.3f}] "
                f"n={int(mm.sum())} |r|={R[m][mm,2].mean():.4f} "
                f"rx={R[m][mm,0].mean():+.4f} ry={R[m][mm,1].mean():+.4f}")


if __name__ == "__main__":
    main()
