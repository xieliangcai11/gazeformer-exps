"""Physical-space (centimetre) calibration across device orientations.

Motivation
----------
Earlier cross-orientation transfer rotated *normalised* screen coordinates
(``x / screen_w``, ``y / screen_h``) by the IMU gravity angle.  That is only
geometrically correct for a square screen: a non-square screen is stretched
by the normalisation, so a 90-degree device rotation is **not** a 90-degree
rotation of normalised coordinates.

Here we convert predictions/labels to centimetres on the physical screen,
rotate them about the screen centre by the IMU angle, calibrate once in that
canonical physical frame, and rotate back.  This is a single, dataset-agnostic
procedure: the only input is screen size + pixel pitch + IMU gravity.

Modes
-----
  per_ctx : calibrate with K labels from every context (baseline, per orientation)
  anchor  : calibrate with K labels from one anchor orientation, transfer to all
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path[:0] = ["src", "."]
from scripts.calibration.seq_bayes_calib import load_index, em  # noqa: E402


def log(msg):
    print(msg, flush=True)


def fit_ridge(Z, y, lam):
    return np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ y)


def fit_huber(Z, y, lam, delta, iters=5):
    """Iteratively reweighted ridge (Huber) on a design matrix."""
    D = Z.shape[1]
    w = fit_ridge(Z, y, lam)
    for _ in range(iters):
        r = y - Z @ w
        a = np.abs(r)
        wt = np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-9))
        A = Z.T @ (Z * wt[:, None]) + lam * np.eye(D)
        b = Z.T @ (wt * y)
        w = np.linalg.solve(A, b)
    return w


def to_cm(p, meta):
    """normalised [0,1] screen point -> centimetres about screen centre."""
    sw, sh, cmx, cmy = meta
    wcm, hcm = sw * cmx, sh * cmy
    return np.stack([p[:, 0] * wcm - wcm / 2.0,
                     p[:, 1] * hcm - hcm / 2.0], 1), (wcm, hcm)


def from_cm(q, meta):
    sw, sh, cmx, cmy = meta
    wcm, hcm = sw * cmx, sh * cmy
    return np.stack([(q[:, 0] + wcm / 2.0) / wcm,
                     (q[:, 1] + hcm / 2.0) / hcm], 1)


def rot(p, phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.stack([c * p[:, 0] + s * p[:, 1],
                     -s * p[:, 0] + c * p[:, 1]], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--anchor", default="1")
    ap.add_argument("--mode", choices=["per_ctx", "anchor"], default="anchor")
    ap.add_argument("--K", type=int, default=52)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--delta", type=float, default=1.0, help="Huber delta in cm")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    log(f"[phys] args={vars(a)}")
    rows = list(csv.DictReader(open(a.index_csv, encoding="utf-8")))
    pcol = "frame_path" if rows and "frame_path" in rows[0] else "rgb_path"
    info = {}
    for r in rows:
        try:
            meta = (float(r["screen_w"]), float(r["screen_h"]),
                    float(r["cm_px_x"]), float(r["cm_px_y"]))
            gx, gy = float(r["imu_x"]), float(r["imu_y"])
            phi = float(np.arctan2(gx, -gy))
        except Exception:
            continue
        info[r[pcol]] = (meta, phi, str(r.get("orientation", "0")))

    z = np.load(a.test_npz, allow_pickle=True)
    pred = z["pred"].astype(np.float64)
    label = z["label"].astype(np.float64)
    path = [str(p) for p in z["path"].tolist()]
    fidx = z["frame_index"].astype(np.int64)
    meta = load_index(a.index_csv, np.array(path, object))
    groups = defaultdict(list)
    for i, p in enumerate(path):
        if p in meta and p in info:
            groups[p.rsplit("/", 1)[0]].append(i)

    acc_b = acc_c = nn = 0.0
    by_ori = defaultdict(lambda: [0.0, 0.0, 0])
    for s, idx in groups.items():
        idx = np.array(sorted(idx, key=lambda i: int(fidx[i])), np.int64)
        if len(idx) < 80:
            continue
        ori = np.array([info[path[i]][2] for i in idx])
        phi = np.array([info[path[i]][1] for i in idx])
        mets = [info[path[i]][0] for i in idx]
        # per-frame physical sizes can differ; convert each frame separately
        pc = np.zeros((len(idx), 2), np.float64)
        lc = np.zeros((len(idx), 2), np.float64)
        for j, i in enumerate(idx):
            pcm, _ = to_cm(pred[i:i + 1], mets[j])
            lcm, _ = to_cm(label[i:i + 1], mets[j])
            pc[j] = rot(pcm, -phi[j])[0]
            lc[j] = rot(lcm, -phi[j])[0]
        for seed in range(a.seeds):
            rng = np.random.RandomState(60000 + seed)
            if a.mode == "anchor":
                cand = np.where(ori == a.anchor)[0]
            else:
                cand = np.arange(len(idx))
            if len(cand) < max(8, a.K // 4):
                continue
            kk = min(a.K, len(cand))
            sel = cand[np.sort(rng.choice(len(cand), kk, replace=False))]
            Z = np.concatenate([pc[sel], np.ones((len(sel), 1))], 1)
            w = []
            for ax in range(2):
                w.append(fit_huber(Z, lc[sel][:, ax], a.lam, a.delta))
            Zq = np.concatenate([pc, np.ones((len(pc), 1))], 1)
            qc = np.stack([Zq @ w[0], Zq @ w[1]], 1)
            # back to each frame's original normalised screen frame
            qo = np.zeros_like(qc)
            for j, i in enumerate(idx):
                qo[j] = from_cm(rot(qc[j:j + 1], phi[j]), mets[j])[0]
            sx = np.array([meta[path[i]][2] for i in idx])
            sy = np.array([meta[path[i]][3] for i in idx])
            e1 = float(em(qo, label[idx], sx, sy).mean())
            e0 = 0.0
            if a.mode == "per_ctx":
                # same model without physical canonicalisation (plain normalised)
                for o in np.unique(ori):
                    so = np.where(ori == o)[0]
                    if len(so) < 5:
                        continue
                    kk = min(a.K, len(so))
                    sup = rng.choice(so, kk, replace=False)
                    Zs = np.concatenate([pred[idx][sup], np.ones((kk, 1))], 1)
                    ws = [fit_ridge(Zs, label[idx][sup][:, ax], 1e-3)
                          for ax in range(2)]
                    Zqo = np.concatenate([pred[idx][so], np.ones((len(so), 1))], 1)
                    qp = np.stack([Zqo @ ws[0], Zqo @ ws[1]], 1)
                    e0 += float(em(qp, label[idx][so], sx[so], sy[so]).sum())
                e0 /= len(idx)
            acc_c += e1 * len(idx)
            acc_b += e0 * len(idx)
            nn += len(idx)
            for o in np.unique(ori):
                m = ori == o
                by_ori[o][1] += e1 * m.sum()
                by_ori[o][2] += int(m.sum())
    C = acc_c / nn
    log(f"[phys] RESULT mode={a.mode} anchor={a.anchor} K={a.K} "
        f"cal={C:.4f} sessions={len(groups)}")
    for o, (_, c, n) in sorted(by_ori.items()):
        if n:
            log(f"[phys] ori={o} n={n} cal={c/n:.4f}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        fh.write("result,mode,anchor,K,cal\n")
        fh.write(f"RESULT,{a.mode},{a.anchor},{a.K},{C}\n")
    log(f"[phys] saved {out}")


if __name__ == "__main__":
    main()
