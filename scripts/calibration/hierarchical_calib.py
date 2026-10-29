"""Hierarchical calibration: one shared correction across contexts plus a
small per-context deviation.

For each target session:
  - per context c: Huber affine fitted on K_c labels
  - shared correction W_global fitted on the pooled support of all contexts
  - per-context deviation dW_c fitted with a ridge prior toward W_global

Compared against the independent per-context correction (no sharing).
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


def build_feat(rows, lm_npz, pose_npz, use_pose, ccol, ctx_vals, groups):
    z = np.load(lm_npz, allow_pickle=True)
    names = [str(n) for n in z["names"].tolist()]
    F = np.full((len(rows), len(names)), np.nan, np.float32)
    V = np.zeros(len(rows), bool)
    F[z["row_idx"].astype(np.int64)] = z["feats"]
    V[z["row_idx"].astype(np.int64)] = z["valid"]
    G = {
        "iris": ["iris_lx", "iris_ly", "iris_rx", "iris_ry"],
        "eye": ["eye_l_w", "eye_l_h", "eye_r_w", "eye_r_h",
                "ear_l", "ear_r", "iod_face"],
        "face": ["face_cx", "face_cy", "face_w", "face_h", "face_ratio"],
        "quality": ["bright", "contrast", "blur"],
    }
    use = np.zeros(len(names), bool)
    for g in groups:
        if g == "all":
            use[:] = True
        else:
            for n in G.get(g, []):
                if n in names:
                    use[names.index(n)] = True
    F = F[:, use]
    if use_pose:
        pz = np.load(pose_npz, allow_pickle=True)
        pf = np.full((len(rows), pz["feats"].shape[1]), np.nan, np.float32)
        pv = np.zeros(len(rows), bool)
        pr = pz["row_idx"].astype(np.int64)
        pf[pr] = pz["feats"]
        pv[pr] = pz["valid"]
        F = np.concatenate([F, pf], 1)
        V = V & pv
    if ccol is not None:
        cmap = {v: i for i, v in enumerate(ctx_vals)}
        oh = np.zeros((len(rows), len(ctx_vals)), np.float32)
        for i, r in enumerate(rows):
            oh[i, cmap[r[ccol]]] = 1.0
        F = np.concatenate([F, oh], 1)
    F = np.vstack([F, np.zeros((1, F.shape[1]), np.float32)])
    V = np.append(V, False)
    return F, V, len(rows)


def load_groups(feat_npz, index_csv, context_key, path_to_row, invalid_row):
    z = np.load(feat_npz, allow_pickle=True)
    pred = z["pred"].astype(np.float64)
    label = z["label"].astype(np.float64)
    path = [str(p) for p in z["path"].tolist()]
    fidx = z["frame_index"].astype(np.int64)
    meta = load_index(index_csv, np.array(path, object))
    groups = defaultdict(lambda: defaultdict(list))
    for i, p in enumerate(path):
        if p not in meta:
            continue
        ctx = meta[p][4] if context_key != "none" else "all"
        groups[p.rsplit("/", 1)[0]][ctx].append(i)
    out = {}
    for s, ctxd in groups.items():
        dd = {}
        for c, idx in ctxd.items():
            idx = np.array(sorted(idx, key=lambda i: int(fidx[i])), np.int64)
            if len(idx) < 40:
                continue
            dd[c] = dict(
                pred=pred[idx], label=label[idx],
                sx=np.array([meta[path[i]][2] for i in idx]),
                sy=np.array([meta[path[i]][3] for i in idx]),
                rows=np.array([path_to_row.get(path[i], invalid_row)
                               for i in idx], np.int64),
            )
        if dd:
            out[s] = dd
    return out


def fit_ridge_prior(Z, y, lam, w0):
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    return np.linalg.solve(A, Z.T @ y + lam * w0)


def loo_lam(Z, y, w0, lams):
    n = len(y)
    best = (lams[0], float("inf"))
    for lam in lams:
        errs = []
        for j in range(n):
            m = np.ones(n, bool)
            m[j] = False
            if m.sum() < 3:
                continue
            w = fit_ridge_prior(Z[m], y[m], lam, w0)
            errs.append(float((Z[j] @ w - y[j]) ** 2))
        if errs:
            e = float(np.mean(errs))
            if e < best[1]:
                best = (lam, e)
    return best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--landmark-npz", required=True)
    ap.add_argument("--pose-npz", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--context-key", choices=["none", "orientation", "activity"], default="activity")
    ap.add_argument("--K", type=int, default=26)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--delta", type=float, default=1.345)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lams", type=float, nargs="+",
                    default=[1e-3, 1e-2, 1e-1, 1.0, 10.0])
    ap.add_argument("--global-lam", type=float, default=1e-2)
    ap.add_argument("--groups", nargs="+", default=["all"])
    ap.add_argument("--use-pose", action="store_true")
    ap.add_argument("--clip", type=float, default=0.15)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    log(f"[hier] args={vars(a)}")
    rows = list(csv.DictReader(open(a.index_csv, encoding="utf-8")))
    pcol = "frame_path" if rows and "frame_path" in rows[0] else "rgb_path"
    path_to_row = {r[pcol]: i for i, r in enumerate(rows)}
    ccol = "activity" if "activity" in rows[0] else (
        "orientation" if "orientation" in rows[0] else None)
    ctx_vals = sorted({r[ccol] for r in rows}) if ccol else []
    F, V, inv = build_feat(rows, a.landmark_npz, a.pose_npz, a.use_pose,
                           ccol, ctx_vals, a.groups)
    D = load_groups(a.train_npz, a.index_csv, "none", path_to_row, inv)
    T = load_groups(a.test_npz, a.index_csv, a.context_key, path_to_row, inv)
    mu_aff, _ = fit_source_mu({f"{s}|{c}": {"pred": d["pred"], "label": d["label"]}
                               for s, dd in D.items() for c, d in dd.items()})
    log(f"[hier] sessions={len(T)} K={a.K} contexts={ctx_vals}")

    def Zof(idx):
        X = F[idx].astype(np.float64).copy()
        X = np.clip((X - fmu) / fsd, -5.0, 5.0)
        X[~np.isfinite(X)] = 0.0
        return np.concatenate([X, np.ones((len(X), 1))], 1)

    acc_i = acc_h = acc_b = nn = 0.0
    ratios_i, ratios_h = [], []
    detail = []
    for s, dd in T.items():
        # standardize on pooled support features
        all_rows = []
        for c, d in dd.items():
            all_rows.append(d["rows"][:a.K])
        all_rows = np.concatenate(all_rows)
        Xs = F[all_rows].astype(np.float64)
        ok = V[all_rows] & np.isfinite(Xs).all(1)
        if ok.sum() < 5:
            continue
        fmu = Xs[ok].mean(0)
        fsd = Xs[ok].std(0) + 1e-6
        Zpool, Rpool = [], []
        info = {}
        for c, d in dd.items():
            sel = support_indices(d, a.K, np.random.RandomState(90000))
            q = np.setdiff1d(np.arange(len(d["pred"])), sel)
            if len(q) < 20:
                continue
            th, _ = fit_theta_huber(d["pred"][sel], d["label"][sel],
                                    a.lam, mu_aff, a.delta)
            q0 = apply_theta_np(d["pred"][q], th)
            e0 = float(em(q0, d["label"][q], d["sx"][q], d["sy"][q]).mean())
            Zs = Zof(d["rows"][sel])
            Rs = d["label"][sel] - apply_theta_np(d["pred"][sel], th)
            info[c] = dict(sel=sel, q=q, th=th, e0=e0, Zs=Zs, Rs=Rs)
            Zpool.append(Zs)
            Rpool.append(Rs)
        if not info:
            continue
        Zp = np.concatenate(Zpool, 0)
        Rp = np.concatenate(Rpool, 0)
        wg = np.stack([fit_ridge_prior(Zp, Rp[:, ax], a.global_lam,
                                       np.zeros(Zp.shape[1]))
                       for ax in range(2)], 1)
        # independent
        ei = eh = eb = 0.0
        n_i = 0
        for c, inf in info.items():
            d = dd[c]
            q = inf["q"]
            q0 = apply_theta_np(d["pred"][q], inf["th"])
            e0 = inf["e0"]
            # independent per-context
            wi = []
            for ax in range(2):
                lam = loo_lam(inf["Zs"], inf["Rs"][:, ax],
                              np.zeros(inf["Zs"].shape[1]), a.lams)
                wi.append(fit_ridge_prior(inf["Zs"], inf["Rs"][:, ax], lam,
                                          np.zeros(inf["Zs"].shape[1])))
            Ze = Zof(d["rows"][q])
            ci = np.clip(np.stack([Ze @ wi[0], Ze @ wi[1]], 1),
                         -a.clip, a.clip)
            ei += float(em(q0 + ci, d["label"][q], d["sx"][q],
                           d["sy"][q]).mean()) * len(q)
            # hierarchical
            wh = []
            for ax in range(2):
                lam = loo_lam(inf["Zs"], inf["Rs"][:, ax], wg[:, ax], a.lams)
                wh.append(fit_ridge_prior(inf["Zs"], inf["Rs"][:, ax], lam,
                                          wg[:, ax]))
            ch = np.clip(np.stack([Ze @ wh[0], Ze @ wh[1]], 1),
                         -a.clip, a.clip)
            eh += float(em(q0 + ch, d["label"][q], d["sx"][q],
                           d["sy"][q]).mean()) * len(q)
            eb += e0 * len(q)
            n_i += len(q)
        if n_i:
            acc_i += ei
            acc_h += eh
            acc_b += eb
            nn += n_i
            ratios_i.append(ei / max(eb, 1e-12) - 1)
            ratios_h.append(eh / max(eb, 1e-12) - 1)
            detail.append([s, n_i, eb / n_i, ei / n_i, eh / n_i])
    B, IND, HIE = acc_b / nn, acc_i / nn, acc_h / nn
    ri = np.array(ratios_i)
    log(f"[hier] RESULT K={a.K} base={B:.4f} independent={IND:.4f} "
        f"hierarchical={HIE:.4f} ind_vs_base={100*(IND/B-1):+.2f}% "
        f"hier_vs_base={100*(HIE/B-1):+.2f}% "
        f"hier_vs_ind={100*(HIE/IND-1):+.2f}% sessions={len(ri)}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["session", "n", "base", "independent", "hierarchical"])
        w.writerows(detail)
    log(f"[hier] saved {out}")


if __name__ == "__main__":
    main()
