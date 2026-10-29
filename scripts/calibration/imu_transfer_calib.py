"""IMU-based cross-orientation calibration transfer.

Goal: calibrate in ONE natural device orientation (anchor), then transfer the
calibration to all other orientations using the IMU rotation, without any
labels from those orientations.
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


def rot_to_canon(p, phi):
    x = p[:, 0] - 0.5
    y = p[:, 1] - 0.5
    c, s = np.cos(phi), np.sin(phi)
    return np.stack([c * x + s * y + 0.5, -s * x + c * y + 0.5], 1)


def rot_from_canon(p, phi):
    return rot_to_canon(p, -phi)


def build_features(rows, lm_npz, pose_npz):
    z = np.load(lm_npz, allow_pickle=True)
    names = [str(n) for n in z["names"].tolist()]
    F = np.full((len(rows), len(names)), np.nan, np.float32)
    V = np.zeros(len(rows), bool)
    F[z["row_idx"].astype(np.int64)] = z["feats"]
    V[z["row_idx"].astype(np.int64)] = z["valid"]
    pz = np.load(pose_npz, allow_pickle=True)
    pf = np.full((len(rows), pz["feats"].shape[1]), np.nan, np.float32)
    pv = np.zeros(len(rows), bool)
    pr = pz["row_idx"].astype(np.int64)
    pf[pr] = pz["feats"]
    pv[pr] = pz["valid"]
    F = np.concatenate([F, pf], 1)
    V = V & pv
    F = np.vstack([F, np.zeros((1, F.shape[1]), np.float32)])
    V = np.append(V, False)
    return F, V, len(rows)


def fit_ridge(Z, y, lam):
    return np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ y)


def loo_lam(Z, y, lams):
    best = (lams[0], float("inf"))
    for lam in lams:
        errs = []
        for j in range(len(y)):
            m = np.ones(len(y), bool)
            m[j] = False
            if m.sum() < 3:
                continue
            w = fit_ridge(Z[m], y[m], lam)
            errs.append(float((Z[j] @ w - y[j]) ** 2))
        if errs and float(np.mean(errs)) < best[1]:
            best = (lam, float(np.mean(errs)))
    return best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--landmark-npz", required=True)
    ap.add_argument("--pose-npz", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--anchor", default="1", help="anchor orientation")
    ap.add_argument("--K", type=int, default=52)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lams", type=float, nargs="+",
                    default=[1e-3, 1e-2, 1e-1, 1.0, 10.0])
    ap.add_argument("--clip", type=float, default=0.15)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    log(f"[imut] args={vars(a)}")
    rows = list(csv.DictReader(open(a.index_csv, encoding="utf-8")))
    pcol = "frame_path" if rows and "frame_path" in rows[0] else "rgb_path"
    path_to_row = {r[pcol]: i for i, r in enumerate(rows)}
    orient_row = {}
    phi_row = np.zeros(len(rows), np.float64)
    for i, r in enumerate(rows):
        orient_row[r[pcol]] = str(r.get("orientation", "0"))
        try:
            gx, gy = float(r["imu_x"]), float(r["imu_y"])
            phi_row[i] = np.arctan2(gx, -gy)
        except Exception:
            phi_row[i] = 0.0
    F, V, inv = build_features(rows, a.landmark_npz, a.pose_npz)

    z = np.load(a.test_npz, allow_pickle=True)
    pred = z["pred"].astype(np.float64)
    label = z["label"].astype(np.float64)
    path = [str(p) for p in z["path"].tolist()]
    fidx = z["frame_index"].astype(np.int64)
    meta = load_index(a.index_csv, np.array(path, object))
    groups = defaultdict(list)
    for i, p in enumerate(path):
        if p in meta:
            groups[p.rsplit("/", 1)[0]].append(i)
    D = np.load(a.train_npz, allow_pickle=True)
    mu_aff, _ = fit_source_mu({"all": {"pred": D["pred"].astype(np.float64),
                                       "label": D["label"].astype(np.float64)}})

    acc_b = acc_c = nn = 0.0
    by_ori = defaultdict(lambda: [0.0, 0.0, 0])
    for s, idx in groups.items():
        idx = np.array(sorted(idx, key=lambda i: int(fidx[i])), np.int64)
        if len(idx) < 40:
            continue
        ori = np.array([orient_row.get(path[i], "0") for i in idx])
        anchor = np.where(ori == a.anchor)[0]
        if len(anchor) < a.K + 20:
            # fall back to the largest orientation
            vals, cnt = np.unique(ori, return_counts=True)
            anchor = np.where(ori == vals[np.argmax(cnt)])[0]
        if len(anchor) < 20:
            continue
        phi = phi_row[[path_to_row[path[i]] for i in idx]]
        pc = rot_to_canon(pred[idx], phi)
        lc = rot_to_canon(label[idx], phi)
        for seed in range(a.seeds):
            sel = support_indices(dict(pred=pc[anchor]), a.K,
                                  np.random.RandomState(90000 + seed))
            sel = anchor[sel]
            th, _ = fit_theta_huber(pc[sel], lc[sel], a.lam, mu_aff, a.delta)
            X = F[[path_to_row[path[i]] for i in idx]].astype(np.float64)
            ok = V[[path_to_row[path[i]] for i in idx]] & np.isfinite(X).all(1)
            if ok[sel].sum() < 5:
                continue
            fmu = X[sel][ok[sel]].mean(0)
            fsd = X[sel][ok[sel]].std(0) + 1e-6
            X = np.clip((X - fmu) / fsd, -5.0, 5.0)
            X[~np.isfinite(X)] = 0.0
            Z = np.concatenate([X, np.ones((len(X), 1))], 1)
            Rs = lc[sel] - apply_theta_np(pc[sel], th)
            w = []
            for ax in range(2):
                lam = loo_lam(Z[sel], Rs[:, ax], a.lams)
                w.append(fit_ridge(Z[sel], Rs[:, ax], lam))
            q = np.setdiff1d(np.arange(len(idx)), sel)
            corr = np.clip(np.stack([Z[q] @ w[0], Z[q] @ w[1]], 1),
                           -a.clip, a.clip)
            qc = apply_theta_np(pc[q], th) + corr
            q0o = rot_from_canon(apply_theta_np(pc[q], th), phi[q])
            qo = rot_from_canon(qc, phi[q])
            sx = np.array([meta[path[i]][2] for i in idx])[q]
            sy = np.array([meta[path[i]][3] for i in idx])[q]
            e0 = float(em(q0o, label[idx][q], sx, sy).mean())
            e1 = float(em(qo, label[idx][q], sx, sy).mean())
            acc_b += e0 * len(q)
            acc_c += e1 * len(q)
            nn += len(q)
            for o in np.unique(ori[q]):
                m = ori[q] == o
                by_ori[o][0] += e0 * m.sum()
                by_ori[o][1] += e1 * m.sum()
                by_ori[o][2] += int(m.sum())
    B, C = acc_b / nn, acc_c / nn
    log(f"[imut] RESULT anchor={a.anchor} K={a.K} base={B:.4f} "
        f"transfer={C:.4f} ({100*(C/B-1):+.2f}%)")
    for o, (b, c, n) in sorted(by_ori.items()):
        if n:
            log(f"[imut] ori={o} n={n} base={b/n:.4f} transfer={c/n:.4f} "
                f"({100*((c/n)/(b/n)-1):+.2f}%)")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        fh.write("result,anchor,K,base,transfer\n")
        fh.write(f"RESULT,{a.anchor},{a.K},{B},{C}\n")
    log(f"[imut] saved {out}")


if __name__ == "__main__":
    main()
