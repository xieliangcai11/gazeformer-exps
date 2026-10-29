"""IMU-canonical calibration: rotate screen coordinates to a canonical
orientation using gravity, calibrate once, then rotate predictions back.

This removes the need to calibrate separately for each device orientation.
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
    xc = c * x + s * y
    yc = -s * x + c * y
    return np.stack([xc + 0.5, yc + 0.5], 1)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--landmark-npz", required=True)
    ap.add_argument("--pose-npz", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--K", type=int, default=52)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lams", type=float, nargs="+",
                    default=[1e-3, 1e-2, 1e-1, 1.0, 10.0])
    ap.add_argument("--clip", type=float, default=0.15)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    log(f"[canon] args={vars(a)}")
    rows = list(csv.DictReader(open(a.index_csv, encoding="utf-8")))
    pcol = "frame_path" if rows and "frame_path" in rows[0] else "rgb_path"
    path_to_row = {r[pcol]: i for i, r in enumerate(rows)}
    phi_row = np.zeros(len(rows), np.float64)
    for i, r in enumerate(rows):
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
    sessions = {}
    for s, idx in groups.items():
        idx = np.array(sorted(idx, key=lambda i: int(fidx[i])), np.int64)
        if len(idx) >= 40:
            sessions[s] = idx
    D = np.load(a.train_npz, allow_pickle=True)
    mu_aff, _ = fit_source_mu({"all": {"pred": D["pred"].astype(np.float64),
                                       "label": D["label"].astype(np.float64)}})
    log(f"[canon] sessions={len(sessions)} K={a.K}")

    acc_b = acc_c = nn = 0.0
    ratios = []
    for s, ii in sessions.items():
        r0 = r1 = rn = 0.0
        for seed in range(a.seeds):
            phi = phi_row[[path_to_row[path[i]] for i in ii]]
            pc = rot_to_canon(pred[ii], phi)
            lc = rot_to_canon(label[ii], phi)
            sel = support_indices(dict(pred=pc), a.K,
                                  np.random.RandomState(90000 + seed))
            q = np.setdiff1d(np.arange(len(ii)), sel)
            if len(q) < 20:
                continue
            th, _ = fit_theta_huber(pc[sel], lc[sel], a.lam, mu_aff, a.delta)
            # features standardized on support
            X = F[[path_to_row[path[i]] for i in ii]][:, :].astype(np.float64)
            ok = V[[path_to_row[path[i]] for i in ii]] & np.isfinite(X).all(1)
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
                # simple ridge with LOO-selected lambda
                best = (a.lams[0], float("inf"))
                for lam in a.lams:
                    errs = []
                    for j in range(len(sel)):
                        m = np.ones(len(sel), bool)
                        m[j] = False
                        if m.sum() < 3:
                            continue
                        wj = np.linalg.solve(Z[sel][m].T @ Z[sel][m] +
                                             lam * np.eye(Z.shape[1]),
                                             Z[sel][m].T @ Rs[m, ax])
                        errs.append((Z[sel][j] @ wj - Rs[j, ax]) ** 2)
                    if errs and float(np.mean(errs)) < best[1]:
                        best = (lam, float(np.mean(errs)))
                lam = best[0]
                w.append(np.linalg.solve(Z[sel].T @ Z[sel] + lam * np.eye(Z.shape[1]),
                                         Z[sel].T @ Rs[:, ax]))
            corr = np.stack([Z[q] @ w[0], Z[q] @ w[1]], 1)
            corr = np.clip(corr, -a.clip, a.clip)
            qc = apply_theta_np(pc[q], th) + corr
            # rotate back to original orientation for evaluation
            qo = rot_from_canon(qc, phi[q])
            q0o = rot_from_canon(apply_theta_np(pc[q], th), phi[q])
            sx = np.array([meta[path[i]][2] for i in ii])[q]
            sy = np.array([meta[path[i]][3] for i in ii])[q]
            e0 = float(em(q0o, label[ii][q], sx, sy).mean())
            e1 = float(em(qo, label[ii][q], sx, sy).mean())
            r0 += e0 * len(q)
            r1 += e1 * len(q)
            rn += len(q)
        if rn:
            b, e = r0 / rn, r1 / rn
            acc_b += b * rn
            acc_c += e * rn
            nn += rn
            ratios.append(e / max(b, 1e-12) - 1)
    B, C = acc_b / nn, acc_c / nn
    rr = np.array(ratios)
    log(f"[canon] RESULT K={a.K} base={B:.4f} canon={C:.4f} "
        f"({100*(C/B-1):+.2f}%) sessions={len(rr)} "
        f"improved={int((rr<0).sum())} worst={100*rr.max():+.1f}%")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        fh.write("result,K,base,canon\n")
        fh.write(f"RESULT,{a.K},{B},{C}\n")
    log(f"[canon] saved {out}")


if __name__ == "__main__":
    main()
