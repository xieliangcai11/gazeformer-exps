"""仿射 vs 二次（带先验）校准对比。

模型（逐上下文：GC 朝向 / RGD 活动）：
  u = phi(p)^T theta_x,  v = phi(p)^T theta_y
  affine: phi = [1, x, y]
  quad  : phi = [1, x, y, x^2, y^2, xy]

源域：每组拟合系数；用组间均值/协方差做先验。
测试：覆盖选点 K 个；ridge MAP： (Phi^T Phi + lam I) theta = Phi^T u + lam mu
lambda 在验证集上选；测试集只评一次；多 seed 选点。

用法：
  python -m scripts.analysis.poly_calib \
    --train-npz ... --val-npz ... --test-npz ... \
    --index-csv ... --pose-npz ... --context-key orientation
"""
import argparse
import csv
import resource
import sys

import numpy as np

sys.path[:0] = ["src", "."]
from scripts.analysis.cc_eb import load_sessions, em, select_coverage  # noqa: E402


def phi(p, model):
    x, y = p[:, 0], p[:, 1]
    if model == "affine":
        return np.stack([np.ones(len(p)), x, y], 1)
    return np.stack([np.ones(len(p)), x, y, x * x, y * y, x * y], 1)


def fit_theta(P, U, model, lam=1e-6):
    X = phi(P, model)
    A = X.T @ X + lam * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ U)


def apply_theta(P, tx, ty, model):
    X = phi(P, model)
    return np.stack([X @ tx, X @ ty], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--val-npz", required=True)
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--pose-npz", required=True)
    ap.add_argument("--context-key", default="orientation")
    ap.add_argument("--models", nargs="+", default=["affine", "quad"])
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 9, 13, 26])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.1, 1.0, 10.0, 100.0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    D = load_sessions(a.train_npz, a.index_csv, a.pose_npz, a.context_key)
    V = load_sessions(a.val_npz, a.index_csv, a.pose_npz, a.context_key)
    T = load_sessions(a.test_npz, a.index_csv, a.pose_npz, a.context_key)
    print(f"[poly] groups train/val/test={len(D)}/{len(V)}/{len(T)} "
          f"maxrss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024:.2f}GB")

    # 源域系数与先验
    priors = {}
    for model in a.models:
        TH = []
        for s, d in D.items():
            tx = fit_theta(d["pred"], d["label"][:, 0], model)
            ty = fit_theta(d["pred"], d["label"][:, 1], model)
            TH.append(np.concatenate([tx, ty]))
        TH = np.array(TH)
        mu = TH.mean(0)
        S = np.cov(TH.T) + 1e-6 * np.eye(TH.shape[1])
        priors[model] = (mu, S)
        print(f"[poly] model={model} dim={TH.shape[1]} "
              f"theta_mean={np.round(mu,3).tolist()}")
        print(f"[poly]   theta_std={np.round(np.sqrt(np.diag(S)),4).tolist()}")

    def map_fit(d, sel, model, lam):
        mu, S = priors[model]
        X = phi(d["pred"][sel], model)
        A = X.T @ X + lam * np.eye(X.shape[1])
        tx = np.linalg.solve(A, X.T @ d["label"][sel, 0] + lam * mu[:X.shape[1]])
        ty = np.linalg.solve(A, X.T @ d["label"][sel, 1] + lam * mu[X.shape[1]:])
        return tx, ty

    def eval_case(d, sel, model, lam):
        tx, ty = map_fit(d, sel, model, lam)
        ev = np.setdiff1d(np.arange(len(d["pred"])), sel)
        if len(ev) < 20:
            return None
        e0 = em(d["pred"][ev], d["label"][ev], d["sx"][ev], d["sy"][ev]).mean()
        e1 = em(apply_theta(d["pred"][ev], tx, ty, model), d["label"][ev],
                d["sx"][ev], d["sy"][ev]).mean()
        return e0, e1, len(ev)

    # 验证集选 lambda（K=13，coverage，1 seed）
    best = {}
    for model in a.models:
        rows = []
        for lam in a.lams:
            acc0 = acc1 = nn = 0.0
            for s, d in V.items():
                rng = np.random.RandomState(abs(hash((s, model, lam))) % 2**31)
                sel = select_coverage(d["pred"], 13, rng)
                r = eval_case(d, sel, model, lam)
                if r is None:
                    continue
                acc0 += r[0] * r[2]; acc1 += r[1] * r[2]; nn += r[2]
            if nn:
                val = acc1 / nn
                rows.append((val, lam))
                print(f"[poly] val model={model} lam={lam} EM={val:.4f}")
        rows.sort()
        best[model] = rows[0][1]
        print(f"[poly] chosen model={model} lam={best[model]} val={rows[0][0]:.4f}")

    out_rows = []
    for model in a.models:
        lam = best[model]
        for k in a.ks:
            acc0 = acc1 = nn = 0.0
            sess_ratio = []
            for s, d in T.items():
                r0 = r1 = rn = 0.0
                for seed in range(a.seeds):
                    rng = np.random.RandomState(abs(hash((s, k, seed, model))) % 2**31)
                    sel = select_coverage(d["pred"], k, rng)
                    r = eval_case(d, sel, model, lam)
                    if r is None:
                        continue
                    r0 += r[0] * r[2]; r1 += r[1] * r[2]; rn += r[2]
                if rn:
                    acc0 += r0; acc1 += r1; nn += rn
                    sess_ratio.append((r1 / rn) / (r0 / rn) - 1)
            if not nn:
                continue
            B, E = acc0 / nn, acc1 / nn
            sr = np.array(sess_ratio)
            print(f"[poly] model={model:6s} K={k:2d} lam={lam} base={B:.4f} "
                  f"calib={E:.4f} ({100*(E/B-1):+.2f}%) 改善 {int((sr<0).sum())}/{len(sr)} "
                  f"退化>2% {int((sr>0.02).sum())} 最差 {100*sr.max():+.0f}%")
            out_rows.append([model, k, lam, B, E, len(sr), int((sr<0).sum()),
                             int((sr>0.02).sum()), float(sr.max())])

    from pathlib import Path
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "K", "lam", "base", "calib", "sessions",
                    "improved", "degrade", "worst"])
        w.writerows(out_rows)
    print(f"[poly] saved {out} maxrss="
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024:.2f}GB")


if __name__ == "__main__":
    main()
