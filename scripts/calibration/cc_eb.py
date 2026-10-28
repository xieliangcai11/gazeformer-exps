"""CC-EB：上下文条件先验 + 少量标定点 + 主动选点。

源域（训练阶段，可用标签）：
  每个会话用全部标签拟合仿射 θ=(sx,bx,sy,by)；
  用上下文 c（屏幕宽高、头姿统计）学 μ(c) 与残差协方差 Σ。

测试（只用 K 个标定点）：
  θ* = (XᵀX + λΣ⁻¹)⁻¹ (Xᵀy + λΣ⁻¹ μ(c))
  选点三种：coverage（3×3）、active（D-optimal）、random。

用法：
  python -m scripts.calibration.cc_eb \
    --train-npz ... --val-npz ... --test-npz ... \
    --index-csv ... --pose-npz ...
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
from collections import defaultdict

import numpy as np

sys.path[:0] = ["src", "."]


def load_index(index_csv, needed):
    need = set(needed.tolist())
    out = {}
    with open(index_csv, encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        pcol = "frame_path" if "frame_path" in rd.fieldnames else "rgb_path"
        for r in rd:
            p = r[pcol]
            if p not in need:
                continue
            ctx = r.get("orientation")
            if ctx is None or ctx == "":
                ctx = r.get("activity", "")
            out[p] = (float(r["screen_w"]) * float(r["cm_px_x"]),
                      float(r["screen_h"]) * float(r["cm_px_y"]),
                      float(r["screen_w"]) * float(r["cm_px_x"]),
                      float(r["screen_h"]) * float(r["cm_px_y"]),
                      str(ctx))
    return out


def load_pose(pose_npz, index_csv, needed):
    pz = np.load(pose_npz, allow_pickle=True)
    row_idx = pz["row_idx"].astype(np.int64)
    feats = pz["feats"].astype(np.float32)
    valid = pz["valid"].astype(bool)
    pose_row = np.full((int(row_idx.max()) + 1, 8), np.nan, np.float32)
    valid_row = np.zeros(int(row_idx.max()) + 1, bool)
    pose_row[row_idx] = feats
    valid_row[row_idx] = valid
    need = set(needed.tolist())
    out = {}
    with open(index_csv, encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        pcol = "frame_path" if "frame_path" in rd.fieldnames else "rgb_path"
        for i, r in enumerate(rd):
            p = r[pcol]
            if p not in need:
                continue
            if i < len(pose_row) and valid_row[i]:
                out[p] = pose_row[i]
    return out


def fit_theta(px, ux, py, uy):
    out = []
    for a, b in ((px, ux), (py, uy)):
        X = np.stack([a, np.ones(len(a))], 1)
        out.append(np.linalg.solve(X.T @ X + 1e-8 * np.eye(2), X.T @ b))
    return out


def apply_theta(p, tx, ty):
    q = p.copy()
    q[:, 0] = tx[0] * p[:, 0] + tx[1]
    q[:, 1] = ty[0] * p[:, 1] + ty[1]
    return q


def load_sessions(npz_path, index_csv, pose_npz):
    z = np.load(npz_path, allow_pickle=True)
    pred = z["pred"].astype(np.float32)
    label = z["label"].astype(np.float32)
    path = z["path"]
    fidx = z["frame_index"].astype(np.int64)
    meta = load_index(index_csv, path)
    pose = load_pose(pose_npz, index_csv, path)
    sess = defaultdict(list)
    for i in range(len(pred)):
        if path[i] in meta and path[i] in pose:
            sess[path[i].rsplit("/", 1)[0]].append(i)
    out = {}
    for s, idx in sess.items():
        idx = np.array(sorted(idx, key=lambda i: int(fidx[i])), np.int64)
        if len(idx) < 40:
            continue
        out[s] = dict(idx=idx, pred=pred[idx], label=label[idx],
                      pose=np.stack([pose[path[i]] for i in idx]),
                      sx=np.array([meta[path[i]][2] for i in idx]),
                      sy=np.array([meta[path[i]][3] for i in idx]),
                      w=np.array([meta[path[i]][0] for i in idx]),
                      h=np.array([meta[path[i]][1] for i in idx]),
                      ctx=np.array([meta[path[i]][4] for i in idx]))
    return out


def expand_groups(S, key):
    if key == "none":
        return S
    out = {}
    for s, d in S.items():
        for c in sorted(set(d["ctx"].tolist())):
            m = d["ctx"] == c
            if m.sum() < 40:
                continue
            dd = {k: (v[m] if isinstance(v, np.ndarray) and len(v) == len(m) else v)
                  for k, v in d.items()}
            out[f"{s}||{c}"] = dd
    return out


def context_of(d):
    p = d["pose"]
    return np.concatenate([[d["w"][0], d["h"][0], d["w"][0] / max(d["h"][0], 1e-6)],
                           p.mean(0), p.std(0)])


def em(p, l, sx, sy):
    return np.hypot((p[:, 0] - l[:, 0]) * sx, (p[:, 1] - l[:, 1]) * sy)


def select_coverage(pred, k, rng, nbin=3):
    cells = defaultdict(list)
    for j in range(len(pred)):
        cells[(min(nbin - 1, int(pred[j, 0] * nbin)),
               min(nbin - 1, int(pred[j, 1] * nbin)))].append(j)
    per = max(1, k // max(1, len(cells)))
    sel = []
    for v in cells.values():
        v = list(v); rng.shuffle(v); sel += v[:per]
    rest = [j for j in range(len(pred)) if j not in set(sel)]
    if len(sel) < k:
        sel += list(rng.choice(rest, min(k - len(sel), len(rest)), replace=False))
    return sorted(set(sel))[:k]


def select_random(pred, k, rng):
    return sorted(rng.choice(len(pred), min(k, len(pred)), replace=False).tolist())


def select_active(pred, k, prior_prec, rng=None, pool=400):
    """D-optimal 贪心：每个点贡献两行 [px,0,1,0] 与 [0,py,0,1]。"""
    n = len(pred)
    idxs = np.arange(n)
    if n > pool:
        idxs = np.sort(rng.choice(n, pool, replace=False))
    A = prior_prec.copy()
    chosen = []
    for _ in range(min(k, len(idxs))):
        best_j, best_score = None, -np.inf
        for j in idxs:
            if j in chosen:
                continue
            px, py = pred[j]
            X = np.array([[px, 0, 1, 0], [0, py, 0, 1]], np.float64)
            score = np.linalg.slogdet(A + X.T @ X)[1]
            if score > best_score:
                best_j, best_score = int(j), score
        if best_j is None:
            break
        chosen.append(best_j)
        px, py = pred[best_j]
        X = np.array([[px, 0, 1, 0], [0, py, 0, 1]], np.float64)
        A += X.T @ X
    return sorted(chosen)


def map_theta(pred, label, sel, mu, prec):
    P = pred[sel]; L = label[sel]
    out = []
    for ax in range(2):
        X = np.stack([P[:, ax], np.ones(len(sel))], 1)
        A = X.T @ X + prec[ax]
        b = X.T @ L[:, ax] + prec[ax] @ mu[ax]
        out.append(np.linalg.solve(A, b))
    return out


def apply_map(pred, theta):
    q = pred.copy()
    q[:, 0] = theta[0][0] * pred[:, 0] + theta[0][1]
    q[:, 1] = theta[1][0] * pred[:, 1] + theta[1][1]
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--val-npz", required=True)
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--pose-npz", required=True)
    ap.add_argument("--test-index-csv", default="")
    ap.add_argument("--test-pose-npz", default="")
    ap.add_argument("--ks", type=int, nargs="+", default=[3, 5, 9, 13])
    ap.add_argument("--context-key", choices=["none", "orientation", "activity"],
                    default="none")
    ap.add_argument("--prior", choices=["ridge", "cov"], default="ridge")
    ap.add_argument("--ridge", type=float, default=1e-2)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.1, 0.3, 1.0, 3.0, 10.0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=1, help="选点重复次数，结果取平均")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    D = load_sessions(a.train_npz, a.index_csv, a.pose_npz)
    V = load_sessions(a.val_npz, a.index_csv, a.pose_npz)
    ti = a.test_index_csv or a.index_csv
    tp = a.test_pose_npz or a.pose_npz
    T = load_sessions(a.test_npz, ti, tp)
    D = expand_groups(D, a.context_key)
    V = expand_groups(V, a.context_key)
    T = expand_groups(T, a.context_key)
    print(f"[cc] context_key={a.context_key} groups train/val/test={len(D)}/{len(V)}/{len(T)}")
    print(f"[cc] sessions train/val/test={len(D)}/{len(V)}/{len(T)} "
          f"maxrss={_maxrss():.2f}GB")

    # 源域 θ 与上下文
    S = list(D)
    C = np.stack([context_of(D[s]) for s in S])
    TH = []
    for s in S:
        d = D[s]
        th = fit_theta(d["pred"][:, 0], d["label"][:, 0],
                       d["pred"][:, 1], d["label"][:, 1])
        TH.append([th[0][0], th[0][1], th[1][0], th[1][1]])
    TH = np.array(TH)
    cmu, csd = C.mean(0), C.std(0) + 1e-6
    Cn = (C - cmu) / csd
    X = np.concatenate([Cn, np.ones((len(Cn), 1))], 1)
    W = np.linalg.solve(X.T @ X + a.ridge * np.eye(X.shape[1]), X.T @ TH)
    pred_mu = X @ W
    resid = TH - pred_mu
    SIG = np.cov(resid.T) + 1e-6 * np.eye(4)
    MU_G = TH.mean(0)
    SIG_G = np.cov(TH.T) + 1e-6 * np.eye(4)
    print(f"[cc] theta mean={np.round(TH.mean(0),4).tolist()} "
          f"std={np.round(TH.std(0),4).tolist()}")
    print(f"[cc] context dim={C.shape[1]} ridge={a.ridge} "
          f"mu_resid_std={np.round(resid.std(0),4).tolist()}")
    print(f"[cc] sigma diag={np.round(np.diag(SIG),5).tolist()}")

    def prior_for(d):
        c = (context_of(d) - cmu) / csd
        x = np.concatenate([c, [1.0]])
        th = x @ W
        return np.array([th[0], th[1]]), np.array([th[2], th[3]]), th

    def prec_from(lam, cov):
        if a.prior == "cov":
            P = lam * np.linalg.inv(cov)
        else:
            P = lam * np.eye(4)
        return np.array([P[0:2, 0:2], P[2:4, 2:4]]), P

    def prec_for(lam):
        return prec_from(lam, SIG)

    def prec_global(lam):
        return prec_from(lam, SIG_G)

    # 在 val 上选 lambda（context prior + coverage，K=9）
    rng = np.random.RandomState(a.seed)
    best_lam, best_val = None, np.inf
    for lam in a.lams:
        prec, _ = prec_for(lam)
        vals = []
        for s, d in V.items():
            mu_x, mu_y, _ = prior_for(d)
            sel = select_coverage(d["pred"], 9, rng)
            th = map_theta(d["pred"], d["label"], sel, [mu_x, mu_y], prec)
            ev = np.setdiff1d(np.arange(len(d["pred"])), sel)
            e = em(apply_map(d["pred"][ev], th), d["label"][ev], d["sx"][ev], d["sy"][ev])
            vals.append((e.mean(), len(e)))
        val = sum(v * n for v, n in vals) / sum(n for _, n in vals)
        print(f"[cc] val lambda={lam} EM={val:.4f}")
        if val < best_val:
            best_val, best_lam = val, lam
    print(f"[cc] chosen lambda={best_lam} val={best_val:.4f}")
    prec, P = prec_for(best_lam)

    def run_split(ST, name):
        print(f"\n=== {name} (seeds={a.seeds}) ===")
        out_rows = []
        def eval_case(d, sel, pname):
            if pname == "ls":
                mu_x, mu_y = np.zeros(2), np.zeros(2)
                prec_l = np.array([np.zeros((2, 2)), np.zeros((2, 2))])
            elif pname == "global":
                mu_x, mu_y = MU_G[0:2], MU_G[2:4]
                prec_l, _ = prec_global(best_lam)
            else:
                mu_x, mu_y, _ = prior_for(d)
                prec_l, _ = prec_for(best_lam)
            th = map_theta(d["pred"], d["label"], sel, [mu_x, mu_y], prec_l)
            ev = np.setdiff1d(np.arange(len(d["pred"])), sel)
            if len(ev) < 20:
                return None
            e0 = em(d["pred"][ev], d["label"][ev], d["sx"][ev], d["sy"][ev]).mean()
            e1 = em(apply_map(d["pred"][ev], th), d["label"][ev],
                    d["sx"][ev], d["sy"][ev]).mean()
            return e0, e1, len(ev)

        # A. 先验对照：coverage，多 seed 同一选择
        for k in a.ks:
            for pname in ("ls", "global", "context"):
                acc0 = acc1 = nn = 0.0
                sess_ratio = []
                for s, d in ST.items():
                    r0 = r1 = rn = 0.0
                    for seed in range(a.seeds):
                        rng_s = np.random.RandomState(
                            abs(hash((s, k, seed, "cov", a.context_key))) % 2**31)
                        sel = select_coverage(d["pred"], k, rng_s)
                        r = eval_case(d, sel, pname)
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
                print(f"  prior={pname:7s} K={k:2d} base={B:.4f} calib={E:.4f} "
                      f"({100*(E/B-1):+.2f}%) 改善 {int((sr<0).sum())}/{len(sr)} "
                      f"退化>2% {int((sr>0.02).sum())} 最差 {100*sr.max():+.0f}%")
                out_rows.append([name, "prior", pname, k, B, E, len(sr),
                                 int((sr<0).sum()), int((sr>0.02).sum()), float(sr.max())])
        # B. 选点对照：context 先验
        for k in a.ks:
            for mode in ("coverage", "active", "random"):
                acc0 = acc1 = nn = 0.0
                sess_ratio = []
                for s, d in ST.items():
                    r0 = r1 = rn = 0.0
                    for seed in range(a.seeds):
                        rng_s = np.random.RandomState(
                            abs(hash((s, k, seed, mode, a.context_key))) % 2**31)
                        if mode == "coverage":
                            sel = select_coverage(d["pred"], k, rng_s)
                        elif mode == "random":
                            sel = select_random(d["pred"], k, rng_s)
                        else:
                            sel = select_active(d["pred"], k, P, rng_s)
                        r = eval_case(d, sel, "context")
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
                print(f"  mode={mode:8s} K={k:2d} base={B:.4f} calib={E:.4f} "
                      f"({100*(E/B-1):+.2f}%) 改善 {int((sr<0).sum())}/{len(sr)} "
                      f"退化>2% {int((sr>0.02).sum())} 最差 {100*sr.max():+.0f}%")
                out_rows.append([name, "select", mode, k, B, E, len(sr),
                                 int((sr<0).sum()), int((sr>0.02).sum()), float(sr.max())])
        import csv as _csv
        from pathlib import Path as _P
        out = _P(a.out); out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = _csv.writer(fh)
            w.writerow(["split", "type", "name", "K", "base", "calib", "sessions",
                        "improved", "degrade", "worst"])
            w.writerows(out_rows)
        print(f"[cc] saved {out}")
        print(f"[mem] maxrss={_maxrss():.2f}GB")

    run_split(T, "test")


if __name__ == "__main__":
    main()
