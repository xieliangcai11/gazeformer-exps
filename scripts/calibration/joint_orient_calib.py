"""Orientation-conditioned joint calibration.

Instead of fitting one affine per device orientation (needs K labels in every
orientation), fit a *single* model that takes the IMU-derived orientation
one-hot as input:

    residual = W @ [p_canon, onehot(orientation), f, 1]

Labels from all orientations are pooled, so each orientation only needs a few
labels.  Compares against per-orientation calibration at the same total label
budget.
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
    fit_source_mu, fit_theta_huber, apply_theta_np,
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
    return F, V


def fit_ridge(Z, y, lam):
    return np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ y)


def make_feats(pc, ori, Xn, oris, oidx, mode):
    """Build the joint design matrix for a given pooling mode."""
    one = np.eye(len(oris))[[oidx.get(o, 0) for o in ori]]
    if mode == "shared_slope":
        return np.concatenate([pc, one, np.ones((len(pc), 1))], 1)
    if mode == "shared_full":
        return np.concatenate([pc, one * pc[:, :1], one * pc[:, 1:2], one,
                               np.ones((len(pc), 1))], 1)
    return np.concatenate([pc, Xn, one, np.ones((len(pc), 1))], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--landmark-npz", required=True)
    ap.add_argument("--pose-npz", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--orientations", nargs="+", default=["1", "2", "3", "4"])
    ap.add_argument("--K", type=int, default=10, help="labels per orientation")
    ap.add_argument("--mode", default="shared_slope",
                    choices=["shared_slope", "shared_full", "lm"])
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--delta", type=float, default=1.0)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    log(f"[joc] args={vars(a)}")
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
    F, V = build_features(rows, a.landmark_npz, a.pose_npz)
    oris = list(a.orientations)
    oidx = {o: i for i, o in enumerate(oris)}

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

    acc_j = acc_p = nn = 0.0
    by_ori = defaultdict(lambda: [0.0, 0.0, 0])
    for s, idx in groups.items():
        idx = np.array(sorted(idx, key=lambda i: int(fidx[i])), np.int64)
        if len(idx) < 80:
            continue
        ori = np.array([orient_row.get(path[i], "0") for i in idx])
        phi = phi_row[[path_to_row[path[i]] for i in idx]]
        pc = rot_to_canon(pred[idx], phi)
        lc = rot_to_canon(label[idx], phi)
        X = F[[path_to_row[path[i]] for i in idx]].astype(np.float64)
        ok = V[[path_to_row[path[i]] for i in idx]] & np.isfinite(X).all(1)
        if ok.sum() < 10:
            continue
        fmu = X[ok].mean(0)
        fsd = X[ok].std(0) + 1e-6
        Xn = np.clip((X - fmu) / fsd, -5.0, 5.0)
        Xn[~np.isfinite(Xn)] = 0.0
        for seed in range(a.seeds):
            rng = np.random.RandomState(70000 + seed)
            sel = []
            for o in oris:
                cand = np.where((ori == o) & ok)[0]
                if len(cand) == 0:
                    continue
                kk = min(a.K, len(cand))
                sel.append(rng.choice(cand, kk, replace=False))
            if not sel:
                continue
            sel = np.concatenate(sel)
            sel = np.unique(sel)
            Zs = make_feats(pc[sel], ori[sel], Xn[sel], oris, oidx, a.mode)
            w = []
            for ax in range(2):
                w.append(fit_ridge(Zs, lc[sel][:, ax], a.lam))
            Z = make_feats(pc, ori, Xn, oris, oidx, a.mode)
            qc = np.stack([Z @ w[0], Z @ w[1]], 1)
            qo = rot_from_canon(qc, phi)
            sx = np.array([meta[path[i]][2] for i in idx])
            sy = np.array([meta[path[i]][3] for i in idx])
            e1 = float(em(qo, label[idx], sx, sy).mean())
            # per-orientation baseline at the same labels-per-orientation budget
            qp = np.zeros_like(pc)
            for o in oris:
                so = np.where((ori == o) & ok)[0]
                if len(so) < 5:
                    continue
                kk = min(a.K, len(so))
                sup = rng.choice(so, kk, replace=False)
                th, _ = fit_theta_huber(pc[sup], lc[sup], a.lam,
                                        mu_aff, a.delta)
                qp[so] = apply_theta_np(pc[so], th)
            e0 = float(em(rot_from_canon(qp, phi), label[idx], sx, sy).mean())
            acc_j += e1 * len(idx)
            acc_p += e0 * len(idx)
            nn += len(idx)
            for o in np.unique(ori):
                m = ori == o
                by_ori[o][0] += e0 * m.sum()
                by_ori[o][1] += e1 * m.sum()
                by_ori[o][2] += int(m.sum())
    B, C = acc_p / nn, acc_j / nn
    log(f"[joc] RESULT mode={a.mode} K_per_ori={a.K} per_ori={B:.4f} "
        f"joint={C:.4f} ({100*(C/B-1):+.2f}%) sessions={len(groups)}")
    for o, (b, c, n) in sorted(by_ori.items()):
        if n:
            log(f"[joc] ori={o} n={n} per_ori={b/n:.4f} joint={c/n:.4f}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        fh.write("result,K,per_ori,joint\n")
        fh.write(f"RESULT,{a.K},{B},{C}\n")
    log(f"[joc] saved {out}")


if __name__ == "__main__":
    main()
