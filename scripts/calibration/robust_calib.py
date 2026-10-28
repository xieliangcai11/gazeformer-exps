"""B2 vs Huber 鲁棒标定（仿射 + ridge 先验）。

Huber IRLS：对残差大的标定点降权，减少“没看准”的点对参数的破坏。

用法：
  python -m scripts.calibration.robust_calib \
    --train-npz ... --val-npz ... --test-npz ... \
    --index-csv ... --pose-npz ... --context-key orientation
"""
import argparse
import csv
try:
    import resource
except ImportError:  # Windows
    resource = None


def _maxrss():
    if resource is None:
        return 0.0
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = ["src", "."]
from scripts.calibration.seq_bayes_calib import load_sessions, em  # noqa: E402
from scripts.calibration.cc_eb import select_coverage  # noqa: E402


def fit_ridge(P, U, lam, mu):
    X = np.stack([P, np.ones(len(P))], 1)
    A = X.T @ X + lam * np.eye(2)
    return np.linalg.solve(A, X.T @ U + lam * mu)


def fit_huber(P, U, lam, mu, delta=1.345, iters=10):
    th = fit_ridge(P, U, lam, mu)
    X = np.stack([P, np.ones(len(P))], 1)
    for _ in range(iters):
        r = U - X @ th
        scale = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-6
        z = np.abs(r) / (delta * scale + 1e-12)
        w = np.where(z <= 1.0, 1.0, 1.0 / np.maximum(z, 1e-12))
        A = X.T @ (w[:, None] * X) + lam * np.eye(2)
        b = X.T @ (w * U) + lam * mu
        th = np.linalg.solve(A, b)
    return th, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--val-npz", required=True)
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--pose-npz", required=True)
    ap.add_argument("--context-key", default="orientation")
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 9, 13, 26])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    ap.add_argument("--deltas", type=float, nargs="+", default=[1.0, 1.345, 2.0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    D = load_sessions(a.train_npz, a.index_csv, a.pose_npz, a.context_key)
    V = load_sessions(a.val_npz, a.index_csv, a.pose_npz, a.context_key)
    T = load_sessions(a.test_npz, a.index_csv, a.pose_npz, a.context_key)
    print(f"[rob] groups train/val/test={len(D)}/{len(V)}/{len(T)}")

    # 源域先验（仿射）
    TH = []
    for s, d in D.items():
        tx = fit_ridge(d["pred"][:, 0], d["label"][:, 0], 1e-6, 0.0)
        ty = fit_ridge(d["pred"][:, 1], d["label"][:, 1], 1e-6, 0.0)
        TH.append(np.concatenate([tx, ty]))
    TH = np.array(TH)
    mu = TH.mean(0)
    print(f"[rob] theta mean={np.round(mu,4).tolist()}")

    def eval_case(d, sel, lam, delta):
        P = d["pred"][sel]; L = d["label"][sel]
        if delta is None:
            tx = fit_ridge(P[:, 0], L[:, 0], lam, mu[0:2])
            ty = fit_ridge(P[:, 1], L[:, 1], lam, mu[2:4])
            frac = 0.0
        else:
            tx, wx = fit_huber(P[:, 0], L[:, 0], lam, mu[0:2], delta)
            ty, wy = fit_huber(P[:, 1], L[:, 1], lam, mu[2:4], delta)
            frac = float(np.mean(np.concatenate([wx, wy]) < 0.999))
        ev = np.setdiff1d(np.arange(len(d["pred"])), sel)
        if len(ev) < 20:
            return None
        q = d["pred"][ev].copy()
        q[:, 0] = tx[0] * q[:, 0] + tx[1]
        q[:, 1] = ty[0] * q[:, 1] + ty[1]
        e0 = em(d["pred"][ev], d["label"][ev], d["sx"][ev], d["sy"][ev]).mean()
        e1 = em(q, d["label"][ev], d["sx"][ev], d["sy"][ev]).mean()
        return e0, e1, len(ev), frac

    def choose(method, k):
        rows = []
        delta_iter = [None] if method == "b2" else a.deltas
        for lam in a.lams:
            for delta in delta_iter:
                acc0 = acc1 = nn = 0.0
                for s, d in V.items():
                    rng = np.random.RandomState(abs(hash((s, method, k, lam, delta))) % 2**31)
                    sel = select_coverage(d["pred"], k, rng)
                    r = eval_case(d, sel, lam, delta)
                    if r is None:
                        continue
                    acc0 += r[0] * r[2]; acc1 += r[1] * r[2]; nn += r[2]
                if nn:
                    rows.append((acc1 / nn, lam, delta))
        rows.sort()
        return rows[0]

    out_rows = []
    for method in ("b2", "huber"):
        for k in a.ks:
            val, lam, delta = choose(method, k)
            print(f"[rob] val method={method} K={k} lam={lam} delta={delta} EM={val:.4f}")
            acc0 = acc1 = nn = 0.0
            fracs = []
            ratios = []
            for s, d in T.items():
                r0 = r1 = rn = 0.0
                for seed in range(a.seeds):
                    rng = np.random.RandomState(abs(hash((s, k, seed, method, lam, delta))) % 2**31)
                    sel = select_coverage(d["pred"], k, rng)
                    r = eval_case(d, sel, lam, delta)
                    if r is None:
                        continue
                    r0 += r[0] * r[2]; r1 += r[1] * r[2]; rn += r[2]
                    fracs.append(r[3])
                if rn:
                    acc0 += r0; acc1 += r1; nn += rn
                    ratios.append((r1 / rn) / (r0 / rn) - 1)
            if not nn:
                continue
            B, E = acc0 / nn, acc1 / nn
            sr = np.array(ratios)
            print(f"[rob] method={method:5s} K={k:2d} lam={lam} delta={delta} "
                  f"base={B:.4f} calib={E:.4f} ({100*(E/B-1):+.2f}%) "
                  f"改善 {int((sr<0).sum())}/{len(sr)} 退化>2% {int((sr>0.02).sum())} "
                  f"最差 {100*sr.max():+.0f}% downweight={np.mean(fracs):.2f}")
            out_rows.append([method, k, lam, delta, B, E, len(sr), int((sr<0).sum()),
                             int((sr>0.02).sum()), float(sr.max()), float(np.mean(fracs))])
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "K", "lam", "delta", "base", "calib", "sessions",
                    "improved", "degrade", "worst", "downweight"])
        w.writerows(out_rows)
    print(f"[rob] saved {out} maxrss="
          f"{_maxrss():.2f}GB")


if __name__ == "__main__":
    main()
