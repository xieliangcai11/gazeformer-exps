"""B 的最小验证：摊销式校准（Amortized Calibration）——闭式版本。

思路：
  源域（val）上统计"仿射参数 θ=[sx,sy,bx,by]"的**先验分布**（按朝向）。
  目标域（test）用 K 个标定帧（真值可见），求 θ 的 **MAP**：
      θ* = argmin ||Xθ - y||²/σ² + (θ-μ)ᵀΣ⁻¹(θ-μ)
  一次闭式求解，不做迭代优化，不训练网络（后续可替换为 hypernetwork）。

对照：不校准 / 无先验 LS（只用 K 点）/ 有先验 MAP / 全标定段 LS（上界）

自检：
  C1 基线 EM 与已知一致
  C2 λ→∞ 时 MAP 解 → 先验均值；λ→0 时 → LS 解
  C3 K 增大时 LS 与 MAP 应收敛

用法：
  python -m scripts.analysis.amortized_calibration \
    --source out/gazecapture/analysis/feats_gc_val_m2_full.npz \
    --target out/gazecapture/analysis/feats_gc_test_m2_full.npz \
    --index-csv data/GazeCapture/index/index.csv
"""
import argparse
import sys
from collections import defaultdict

import numpy as np

sys.path[:0] = ["src", "."]
from scripts.analysis.anchor_calibration import apply_affine  # noqa: E402


def load_npz(f):
    z = np.load(f, allow_pickle=True)
    return dict(pred=z["pred"].astype(np.float64), label=z["label"].astype(np.float64),
                err=np.stack([z["cm_x"], z["cm_y"]], 1),
                subject=z["subject"], fidx=z["frame_index"], path=z["path"])


def cm_of(p, l, e):
    d = p - l
    rx = [e[i, 0] / d[i, 0] for i in range(len(p)) if abs(d[i, 0]) > 0.01]
    ry = [e[i, 1] / d[i, 1] for i in range(len(p)) if abs(d[i, 1]) > 0.01]
    return np.tile([float(np.median(rx)) if rx else 1.0,
                    float(np.median(ry)) if ry else 1.0], (len(p), 1))


def em(p, l, c):
    return np.hypot((p[:, 0] - l[:, 0]) * c[:, 0], (p[:, 1] - l[:, 1]) * c[:, 1])


def fit_theta(x, u, y, v, lam=0.0, mu=None):
    """逐轴拟合 u = sx*x + bx, v = sy*y + by；lam 为先验强度（0=LS）。

    mu: [(sx,bx),(sy,by)] 每轴各自的先验均值。
    """
    out = []
    for j, (a, b) in enumerate(((x, u), (y, v))):
        X = np.stack([a, np.ones(len(a))], 1)
        A = X.T @ X
        rhs = X.T @ b
        if lam > 0 and mu is not None:
            A = A + lam * np.eye(2)
            rhs = rhs + lam * mu[j]
        out.append(np.linalg.solve(A, rhs))
    return out  # [(sx,bx),(sy,by)]


def group_sessions(d):
    sess = defaultdict(list)
    for i in range(len(d["pred"])):
        sess[str(d["subject"][i])].append(i)
    return {s: sorted(v, key=lambda i: int(d["fidx"][i])) for s, v in sess.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--ks", default="3,5,9,13,26")
    ap.add_argument("--lams", default="0,0.1,1,10")
    ap.add_argument("--trials", type=int, default=5)
    args = ap.parse_args()

    src = load_npz(args.source); tgt = load_npz(args.target)
    # C1
    allp = np.vstack([src["pred"], tgt["pred"]])
    alll = np.vstack([src["label"], tgt["label"]])
    print(f"[C1] 源 {len(src['pred'])} 帧 / 目标 {len(tgt['pred'])} 帧")

    # 源域先验：每会话拟合全量仿射 → 参数分布
    ss = group_sessions(src)
    params = []
    for s, idx in ss.items():
        p, l = src["pred"][idx], src["label"][idx]
        if len(p) < 100: continue
        th = fit_theta(p[:, 0], l[:, 0], p[:, 1], l[:, 1])
        params.append(np.array([th[0][0], th[1][0], th[0][1], th[1][1]]))
    P = np.array(params)
    mu = P.mean(0); cov = np.cov(P.T) + 1e-6 * np.eye(4)
    print(f"[prior] μ={np.round(mu,3)}  σ={np.round(np.sqrt(np.diag(cov)),3)}  n={len(P)}")

    # C2：λ 极端值检查（单会话）
    s0 = list(ss)[0]; idx0 = ss[s0]
    p0, l0 = src["pred"][idx0], src["label"][idx0]
    k0 = 10
    X0 = p0[:k0, 0]; U0 = l0[:k0, 0]
    ls0 = fit_theta(X0, U0, p0[:k0, 1], l0[:k0, 1], lam=0.0)[0]
    big = fit_theta(X0, U0, p0[:k0, 1], l0[:k0, 1], lam=1e9,
                    mu=[np.array([mu[0], mu[2]]), np.array([mu[1], mu[3]])])[0]
    print(f"[C2] LS 解 {np.round(ls0,3)} vs λ=1e9（应≈先验均值）"
          f"{np.round(big,3)} vs μ={np.round([mu[0],mu[2]],3)}")
    assert abs(big[1] - mu[2]) < 1e-3 and abs(big[0] - mu[0]) < 1e-3, "先验收缩不生效"

    ts = group_sessions(tgt)
    RES = defaultdict(list)
    for K in [int(x) for x in args.ks.split(",")]:
        for lam in [float(x) for x in args.lams.split(",")]:
            e_base = []; e_ls = []; e_map = []
            for s, idx in ts.items():
                p, l = tgt["pred"][idx], tgt["label"][idx]
                c = cm_of(p, l, tgt["err"][idx])
                n = len(idx)
                if n < K + 100: continue
                # 随机 K 个标定帧（多次试验取平均）
                for _ in range(args.trials):
                    sel = np.random.RandomState(hash(s) % 1000 + _).choice(n, K, replace=False)
                    ev = np.array([i for i in range(n) if i not in set(sel)])
                    mu2 = [np.array([mu[0], mu[2]]), np.array([mu[1], mu[3]])]
                    th_ls = fit_theta(p[sel, 0], l[sel, 0], p[sel, 1], l[sel, 1], 0.0)
                    th_map = fit_theta(p[sel, 0], l[sel, 0], p[sel, 1], l[sel, 1], lam, mu2)
                    e_base.append(em(p[ev], l[ev], c[ev]).mean())
                    e_ls.append(em(apply_affine(p[ev], th_ls, 0.75), l[ev], c[ev]).mean())
                    e_map.append(em(apply_affine(p[ev], th_map, 0.75), l[ev], c[ev]).mean())
            RES[(K, lam)] = (np.mean(e_base), np.mean(e_ls), np.mean(e_map))
            print(f"  K={K:>2} λ={lam:<5}: base {RES[(K,lam)][0]:.4f} | "
                  f"LS {RES[(K,lam)][1]:.4f} ({100*(RES[(K,lam)][1]/RES[(K,lam)][0]-1):+.1f}%) | "
                  f"MAP {RES[(K,lam)][2]:.4f} ({100*(RES[(K,lam)][2]/RES[(K,lam)][0]-1):+.1f}%)")


if __name__ == "__main__":
    main()
