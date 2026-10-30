"""Hierarchical (empirical-Bayes) calibration with a context prior.

One estimator for every dataset/context
---------------------------------------
For a context ``c`` (device orientation, activity, or the whole session) the
calibration is an affine map

    y = A_c p + b_c + noise

Instead of fitting each context independently (which needs many labels in
every context), we learn a **prior over affine parameters** from the source
split, where labels are plentiful:

    theta_c ~ N(mu, diag(sigma^2))

At test time only K labels per context are used and the MAP estimate is

    min_theta  Huber(y - Z theta) + sum_j (theta_j - mu_j)^2 / sigma_j^2

which shrinks poorly-determined contexts towards the population calibration.
This is a single algorithm; ``c`` is just metadata (orientation/activity).
"""
import argparse
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


def fit_huber(Z, y, lam, delta, iters=5, prior_mu=None, prior_prec=None):
    D = Z.shape[1]
    w = fit_ridge(Z, y, lam)
    for _ in range(iters):
        r = y - Z @ w
        a = np.abs(r)
        wt = np.where(a <= delta, 1.0, delta / np.maximum(a, 1e-9))
        A = Z.T @ (Z * wt[:, None]) + lam * np.eye(D)
        b = Z.T @ (wt * y)
        if prior_mu is not None:
            A = A + np.diag(prior_prec)
            b = b + prior_prec * prior_mu
        w = np.linalg.solve(A, b)
    return w


def design(p):
    return np.concatenate([p, np.ones((len(p), 1))], 1)


def context_of(path, index_csv, key):
    return load_index(index_csv, np.array(path, object))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--context", choices=["orientation", "activity", "none"],
                    default="orientation")
    ap.add_argument("--K", type=int, default=52)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--delta", type=float, default=0.02)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--prior-scale", type=float, default=1.0,
                    help="multiply prior precision (0 disables prior)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    log(f"[hier] args={vars(a)}")
    tr = np.load(a.train_npz, allow_pickle=True)
    tp = tr["pred"].astype(np.float64)
    tl = tr["label"].astype(np.float64)
    tpath = [str(p) for p in tr["path"].tolist()]
    tmeta = load_index(a.index_csv, np.array(tpath, object))
    src = defaultdict(lambda: ([], []))
    for i, p in enumerate(tpath):
        if p not in tmeta:
            continue
        m = tmeta[p]
        ctx = m[4] if a.context != "none" else "all"
        src[f"{p.rsplit('/', 1)[0]}||{ctx}"][0].append(tp[i])
        src[f"{p.rsplit('/', 1)[0]}||{ctx}"][1].append(tl[i])
    theta_list = []
    for s, (pp, ll) in src.items():
        if len(pp) < 20:
            continue
        Z = design(np.array(pp))
        y = np.array(ll)
        th = np.stack([fit_huber(Z, y[:, ax], a.lam, a.delta) for ax in range(2)], 1)
        theta_list.append(th)
    if not theta_list:
        raise SystemExit("no source contexts")
    TH = np.array(theta_list)
    mu = TH.mean(0)
    sd = TH.std(0) + 1e-3
    prec = a.prior_scale / (sd ** 2)
    log(f"[hier] source contexts={len(TH)} mu={np.round(mu, 3).tolist()} "
        f"sd={np.round(sd, 3).tolist()}")

    z = np.load(a.test_npz, allow_pickle=True)
    pred = z["pred"].astype(np.float64)
    label = z["label"].astype(np.float64)
    path = [str(p) for p in z["path"].tolist()]
    fidx = z["frame_index"].astype(np.int64)
    meta = load_index(a.index_csv, np.array(path, object))
    groups = defaultdict(list)
    for i, p in enumerate(path):
        if p in meta:
            m = meta[p]
            ctx = m[4] if a.context != "none" else "all"
            groups[f"{p.rsplit('/', 1)[0]}||{ctx}"].append(i)

    acc_h = acc_b = nn = 0.0
    by_ctx = defaultdict(lambda: [0.0, 0.0, 0])
    for s, idx in groups.items():
        idx = np.array(sorted(idx, key=lambda i: int(fidx[i])), np.int64)
        if len(idx) < 40:
            continue
        Z = design(pred[idx])
        y = label[idx]
        sx = np.array([meta[path[i]][2] for i in idx])
        sy = np.array([meta[path[i]][3] for i in idx])
        ctx = s.rsplit("||", 1)[-1]
        for seed in range(a.seeds):
            rng = np.random.RandomState(50000 + seed)
            kk = min(a.K, len(idx) - 1)
            sel = rng.choice(len(idx), kk, replace=False)
            wh = [fit_huber(Z[sel], y[sel][:, ax], a.lam, a.delta,
                            prior_mu=mu[:, ax], prior_prec=prec[:, ax])
                  for ax in range(2)]
            qh = np.stack([Z @ wh[0], Z @ wh[1]], 1)
            eh = float(em(qh, y, sx, sy).mean())
            # no-prior baseline at the same K
            wb = [fit_huber(Z[sel], y[sel][:, ax], a.lam, a.delta)
                  for ax in range(2)]
            qb = np.stack([Z @ wb[0], Z @ wb[1]], 1)
            eb = float(em(qb, y, sx, sy).mean())
            acc_h += eh * len(idx)
            acc_b += eb * len(idx)
            nn += len(idx)
            by_ctx[ctx][0] += eb * len(idx)
            by_ctx[ctx][1] += eh * len(idx)
            by_ctx[ctx][2] += len(idx)
    B, H = acc_b / nn, acc_h / nn
    log(f"[hier] RESULT context={a.context} K={a.K} no_prior={B:.4f} "
        f"hier={H:.4f} ({100*(H/B-1):+.2f}%) sessions={len(groups)}")
    for c, (b, h, n) in sorted(by_ctx.items()):
        log(f"[hier] ctx={c} n={n} no_prior={b/n:.4f} hier={h/n:.4f}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        fh.write("result,context,K,no_prior,hier\n")
        fh.write(f"RESULT,{a.context},{a.K},{B},{H}\n")
    log(f"[hier] saved {out}")


if __name__ == "__main__":
    main()
