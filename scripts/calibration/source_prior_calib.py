"""Source-prior correction: learn a global correction on source sessions and
fine-tune it on K target labels.

Model:
    q = Huber_theta(p) + W^T z
    W = W_src + dW

W_src is learned on source sessions with full labels (features -> Huber
residual).  Target sessions fit only dW with a ridge prior toward W_src,
selected by leave-one-out on the K support points.
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


def load_sessions_rows(feat_npz, index_csv, context_key, path_to_row, invalid_row):
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
            rows=np.array([path_to_row.get(path[i], invalid_row) for i in idx],
                          np.int64),
        )
    return out


def build_feat(rows, lm_npz, pose_npz, use_pose, context_col, ctx_vals, groups):
    z = np.load(lm_npz, allow_pickle=True)
    names = [str(n) for n in z["names"].tolist()]
    F = np.full((len(rows), len(names)), np.nan, np.float32)
    V = np.zeros(len(rows), bool)
    F[z["row_idx"].astype(np.int64)] = z["feats"]
    V[z["row_idx"].astype(np.int64)] = z["valid"]
    use = np.zeros(len(names), bool)
    G = {
        "iris": ["iris_lx", "iris_ly", "iris_rx", "iris_ry"],
        "eye": ["eye_l_w", "eye_l_h", "eye_r_w", "eye_r_h",
                "ear_l", "ear_r", "iod_face"],
        "face": ["face_cx", "face_cy", "face_w", "face_h", "face_ratio"],
        "quality": ["bright", "contrast", "blur"],
    }
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
    if context_col is not None:
        oh = np.zeros((len(rows), len(ctx_vals)), np.float32)
        cmap = {v: i for i, v in enumerate(ctx_vals)}
        for i, r in enumerate(rows):
            oh[i, cmap[r[context_col]]] = 1.0
        F = np.concatenate([F, oh], 1)
    invalid_row = len(rows)
    F = np.vstack([F, np.zeros((1, F.shape[1]), np.float32)])
    V = np.append(V, False)
    return F, V, invalid_row


def fit_ridge_prior(Z, y, lam, w0):
    A = Z.T @ Z + lam * np.eye(Z.shape[1])
    return np.linalg.solve(A, Z.T @ y + lam * w0)


def loo_lam_prior(Z, y, w0, lams):
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
    ap.add_argument("--source-npz", required=True)
    ap.add_argument("--source-landmark-npz", required=True)
    ap.add_argument("--source-pose-npz", required=True)
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--test-landmark-npz", required=True)
    ap.add_argument("--test-pose-npz", required=True)
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--context-key", choices=["none", "orientation", "activity"], default="none")
    ap.add_argument("--source-context-key", choices=["none", "orientation", "activity"], default="none")
    ap.add_argument("--K", type=int, default=26)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--delta", type=float, default=1.345)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lams", type=float, nargs="+",
                    default=[1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0])
    ap.add_argument("--src-lam", type=float, default=1e-2)
    ap.add_argument("--groups", nargs="+", default=["all"])
    ap.add_argument("--use-pose", action="store_true")
    ap.add_argument("--clip", type=float, default=0.15)
    ap.add_argument("--mode", choices=["full", "scalar"], default="full")
    ap.add_argument("--per-context-src", action="store_true")
    ap.add_argument("--alpha-lam", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--detail", default="")
    a = ap.parse_args()

    log(f"[srcprior] args={vars(a)}")
    rows = list(csv.DictReader(open(a.index_csv, encoding="utf-8")))
    pcol = "frame_path" if rows and "frame_path" in rows[0] else "rgb_path"
    path_to_row = {r[pcol]: i for i, r in enumerate(rows)}
    ccol = "activity" if "activity" in rows[0] else (
        "orientation" if "orientation" in rows[0] else None)
    ctx_vals = sorted({r[ccol] for r in rows}) if ccol else []

    Fs, Vs, inv_s = build_feat(rows, a.source_landmark_npz, a.source_pose_npz,
                               a.use_pose, ccol, ctx_vals, a.groups)
    Ft, Vt, inv_t = build_feat(rows, a.test_landmark_npz, a.test_pose_npz,
                               a.use_pose, ccol, ctx_vals, a.groups)
    D = load_sessions_rows(a.source_npz, a.index_csv, a.source_context_key,
                           path_to_row, inv_s)
    T = load_sessions_rows(a.test_npz, a.index_csv, a.context_key,
                           path_to_row, inv_t)
    D0 = load_sessions_rows(a.train_npz, a.index_csv, a.source_context_key,
                            path_to_row, inv_s)
    mu_aff, _ = fit_source_mu({k: {"pred": d["pred"], "label": d["label"]}
                               for k, d in D0.items()})
    log(f"[srcprior] source={len(D)} target={len(T)} dims={Fs.shape[1]}")

    # global source feature statistics
    m = Vs & np.isfinite(Fs).all(1)
    fmu = Fs[m].mean(0)
    fsd = Fs[m].std(0) + 1e-6

    def Z_of(F, V, idx):
        X = F[idx].astype(np.float64).copy()
        X = np.clip((X - fmu) / fsd, -5.0, 5.0)
        X[~np.isfinite(X)] = 0.0
        return np.concatenate([X, np.ones((len(X), 1))], 1)

    # learn source correction (global or per-context)
    by_ctx = defaultdict(lambda: [[], []])
    for s, d in D.items():
        if len(d["pred"]) < 20:
            continue
        th, _ = fit_theta_huber(d["pred"], d["label"], a.lam, mu_aff, a.delta)
        Z = Z_of(Fs, Vs, d["rows"])
        R = d["label"] - apply_theta_np(d["pred"], th)
        ctx = s.split("||")[-1] if "||" in s else "all"
        by_ctx[ctx][0].append(Z)
        by_ctx[ctx][1].append(R)
    w_src_by_ctx = {}
    for ctx, (Zl, Rl) in by_ctx.items():
        Zc = np.concatenate(Zl, 0)
        Rc = np.concatenate(Rl, 0)
        w_src_by_ctx[ctx] = np.stack(
            [fit_ridge_prior(Zc, Rc[:, ax], a.src_lam,
                             np.zeros(Zc.shape[1])) for ax in range(2)], 1)
    Zall = np.concatenate([x for c in by_ctx for x in by_ctx[c][0]], 0)
    Rall = np.concatenate([x for c in by_ctx for x in by_ctx[c][1]], 0)
    w_src = np.stack(
        [fit_ridge_prior(Zall, Rall[:, ax], a.src_lam,
                         np.zeros(Zall.shape[1])) for ax in range(2)], 1)
    log(f"[srcprior] source correction contexts={list(w_src_by_ctx)} "
        f"global_norm={np.linalg.norm(w_src):.4f}")

    acc0 = acc1 = nn = 0.0
    ratios = []
    lams_used = []
    detail = []
    for si, (s, d) in enumerate(T.items()):
        r0 = r1 = rn = 0.0
        for seed in range(a.seeds):
            sel = support_indices(d, a.K, np.random.RandomState(90000 + seed))
            q = np.setdiff1d(np.arange(len(d["pred"])), sel)
            if len(q) < 20:
                continue
            th, _ = fit_theta_huber(d["pred"][sel], d["label"][sel],
                                    a.lam, mu_aff, a.delta)
            q0 = apply_theta_np(d["pred"][q], th)
            e0 = float(em(q0, d["label"][q], d["sx"][q], d["sy"][q]).mean())
            w_use = w_src
            if a.per_context_src:
                tctx = s.split("||")[-1] if "||" in s else "all"
                w_use = w_src_by_ctx.get(tctx, w_src)
            Zs = Z_of(Ft, Vt, d["rows"][sel])
            Rs = d["label"][sel] - apply_theta_np(d["pred"][sel], th)
            Ze = Z_of(Ft, Vt, d["rows"][q])
            if a.mode == "scalar":
                Gs = Zs @ w_use
                Ge = Ze @ w_use
                alphas = []
                for ax in range(2):
                    denom = float(Gs[:, ax] @ Gs[:, ax]) + a.alpha_lam
                    al = float(Gs[:, ax] @ Rs[:, ax]) / max(denom, 1e-9)
                    al = float(np.clip(al, -0.5, 1.5))
                    alphas.append(al)
                corr = np.stack([alphas[0] * Ge[:, 0], alphas[1] * Ge[:, 1]], 1)
                lam_used = [float(np.mean(alphas))]
            else:
                w = []
                lam_used = []
                for ax in range(2):
                    lam_ax = loo_lam_prior(Zs, Rs[:, ax], w_use[:, ax], a.lams)
                    lam_used.append(lam_ax)
                    w.append(fit_ridge_prior(Zs, Rs[:, ax], lam_ax, w_use[:, ax]))
                corr = np.stack([Ze @ w[0], Ze @ w[1]], 1)
            corr = np.clip(corr, -a.clip, a.clip)
            q1 = q0 + corr
            e1 = float(em(q1, d["label"][q], d["sx"][q], d["sy"][q]).mean())
            lams_used.append(float(np.mean(lam_used)))
            r0 += e0 * len(q)
            r1 += e1 * len(q)
            rn += len(q)
        if rn:
            b, e = r0 / rn, r1 / rn
            acc0 += b * rn
            acc1 += e * rn
            nn += rn
            ratios.append(e / max(b, 1e-12) - 1)
            detail.append([s, rn, b, e, 100 * (e / b - 1)])
            if si % 5 == 0 or si == len(T) - 1:
                log(f"[srcprior] session={s} n={rn:.0f} mode={a.mode} base={b:.4f} "
                    f"srcprior={e:.4f} ({100*(e/b-1):+.2f}%)")
    B, E = acc0 / nn, acc1 / nn
    rr = np.array(ratios)
    nvec = np.array([x[1] for x in detail], float)
    dvec = np.array([x[3] - x[2] for x in detail], float)
    rng = np.random.RandomState(12345)
    br, bd = [], []
    for _ in range(10000):
        idx = rng.choice(len(detail), len(detail), replace=True)
        w = nvec[idx]
        br.append(float(np.sum(rr[idx] * w) / w.sum()))
        bd.append(float(np.sum(dvec[idx] * w) / w.sum()))
    lo, hi = np.percentile(br, [2.5, 97.5])
    log(f"[srcprior] RESULT K={a.K} base={B:.4f} srcprior={E:.4f} "
        f"({100*(E/B-1):+.2f}%) sessions={len(rr)} "
        f"improved={int((rr<0).sum())} worst={100*rr.max():+.1f}% "
        f"boot95=[{100*lo:+.2f},{100*hi:+.2f}]% "
        f"p(delta>=0)={np.mean(np.array(bd)>=0):.3f} "
        f"mean_lam={np.mean(lams_used) if lams_used else -1:g}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["session", "n", "base", "srcprior", "pct"])
        w.writerows(detail)
    log(f"[srcprior] saved {out}")


if __name__ == "__main__":
    main()
