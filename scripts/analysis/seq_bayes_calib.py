"""噪声感知 + 序贯 + 自适应 K 的贝叶斯标定。

模型（逐上下文：GC 朝向 / RGD 活动）：
  u = s * p + b + eps,   eps ~ N(0, sigma^2(p, pose, screen))
  theta = (sx,bx,sy,by) ~ N(mu, Sigma)   （源域学）

流程：
  1. 源域：每个 group 拟合 theta；用前后半段估计残差；拟合 log sigma^2；
  2. 测试：从候选屏幕位置 q 中，按信息增益/噪声选点；
     测量 (p,u) 后更新后验；后验足够小则停止；
  3. 用后验均值校正剩余帧，与 LS / B2（固定 K 覆盖选点）对比。

用法：
  python -m scripts.analysis.seq_bayes_calib \
    --train-npz ... --val-npz ... --test-npz ... \
    --index-csv ... --pose-npz ... --context-key orientation
"""
import argparse
import csv
import resource
import sys
from collections import defaultdict

import numpy as np

sys.path[:0] = ["src", "."]


# ---------- 数据 ----------
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


def load_sessions(npz_path, index_csv, pose_npz, context_key):
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
        if context_key == "none":
            groups = [("", idx)]
        else:
            ctxs = np.array([meta[path[i]][4] for i in idx])
            groups = [(c, idx[ctxs == c]) for c in sorted(set(ctxs.tolist()))]
        for c, gi in groups:
            if len(gi) < 40:
                continue
            out[f"{s}||{c}"] = dict(
                pred=pred[gi], label=label[gi], pose=np.stack([pose[path[i]] for i in gi]),
                sx=np.array([meta[path[i]][2] for i in gi]),
                sy=np.array([meta[path[i]][3] for i in gi]),
                w=np.array([meta[path[i]][0] for i in gi]),
                h=np.array([meta[path[i]][1] for i in gi]),
                ctx=np.array([meta[path[i]][4] for i in gi]))
    return out


# ---------- 基础 ----------
def fit_theta(px, ux, py, uy):
    out = []
    for a, b in ((px, ux), (py, uy)):
        X = np.stack([a, np.ones(len(a))], 1)
        out.append(np.linalg.solve(X.T @ X + 1e-8 * np.eye(2), X.T @ b))
    return out


def apply_theta(p, th):
    q = p.copy()
    q[:, 0] = th[0][0] * p[:, 0] + th[0][1]
    q[:, 1] = th[1][0] * p[:, 1] + th[1][1]
    return q


def em(p, l, sx, sy):
    return np.hypot((p[:, 0] - l[:, 0]) * sx, (p[:, 1] - l[:, 1]) * sy)


def noise_feats(p, pose, w, h):
    return np.array([
        abs(p[0] - 0.5), abs(p[1] - 0.5),
        abs(pose[0]), abs(pose[1]), abs(pose[2]),
        pose[4],  # openness
        w / 10.0, h / 15.0,
    ], np.float64)


# ---------- 主 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--val-npz", required=True)
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--pose-npz", required=True)
    ap.add_argument("--context-key", choices=["none", "orientation", "activity"],
                    default="none")
    ap.add_argument("--kmin", type=int, default=3)
    ap.add_argument("--kmax", type=int, default=20)
    ap.add_argument("--stop-std", type=float, default=0.01)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--grid", type=int, default=9)
    ap.add_argument("--min-dist", type=float, default=0.18)
    ap.add_argument("--margin", type=float, default=0.08)
    ap.add_argument("--ridge-noise", type=float, default=1e-2)
    ap.add_argument("--select-mode", choices=["info", "coverage"], default="info")
    ap.add_argument("--coverage-cells", action="store_true",
                    help="选点时强制覆盖 3×3 屏幕格子")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--max-sessions", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    D = load_sessions(a.train_npz, a.index_csv, a.pose_npz, a.context_key)
    V = load_sessions(a.val_npz, a.index_csv, a.pose_npz, a.context_key)
    T = load_sessions(a.test_npz, a.index_csv, a.pose_npz, a.context_key)
    if a.max_sessions:
        T = dict(list(T.items())[:a.max_sessions])
    print(f"[seq] groups train/val/test={len(D)}/{len(V)}/{len(T)} "
          f"maxrss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024:.2f}GB")

    # 源域 theta 与噪声
    TH = []; NF = []; LR = []
    for s, d in D.items():
        n = len(d["pred"])
        cut = n // 2
        th = fit_theta(d["pred"][:cut, 0], d["label"][:cut, 0],
                       d["pred"][:cut, 1], d["label"][:cut, 1])
        TH.append([th[0][0], th[0][1], th[1][0], th[1][1]])
        for i in range(cut, n):
            p = d["pred"][i]
            r = d["label"][i] - np.array([th[0][0] * p[0] + th[0][1],
                                          th[1][0] * p[1] + th[1][1]])
            NF.append(noise_feats(p, d["pose"][i], d["w"][i], d["h"][i]))
            LR.append(np.log(np.mean(r ** 2) + 1e-6))
    TH = np.array(TH); NF = np.array(NF); LR = np.array(LR)
    mu = TH.mean(0); S = np.cov(TH.T) + 1e-6 * np.eye(4)
    nmu = NF.mean(0); nsd = NF.std(0) + 1e-6
    Xn = np.concatenate([(NF - nmu) / nsd, np.ones((len(NF), 1))], 1)
    Wn = np.linalg.solve(Xn.T @ Xn + a.ridge_noise * np.eye(Xn.shape[1]), Xn.T @ LR)
    print(f"[seq] theta mean={np.round(mu,4).tolist()} std={np.round(np.sqrt(np.diag(S)),4).tolist()}")
    print(f"[seq] noise logvar mean={LR.mean():.3f} std={LR.std():.3f} "
          f"features={NF.shape[1]}")

    def sigma2(p, pose, w, h):
        f = (noise_feats(p, pose, w, h) - nmu) / nsd
        x = np.concatenate([f, [1.0]])
        lv = float(x @ Wn)
        return float(np.clip(np.exp(lv), 1e-4, 5e-2))

    def prior_prec():
        return np.linalg.inv(S)

    # 候选位置网格
    qs = []
    for i in range(a.grid):
        for j in range(a.grid):
            qx = a.margin + (1 - 2 * a.margin) * i / (a.grid - 1)
            qy = a.margin + (1 - 2 * a.margin) * j / (a.grid - 1)
            qs.append((qx, qy))
    qs = np.array(qs)

    def nearest_frame(d, q, used):
        dist = np.hypot(d["label"][:, 0] - q[0], d["label"][:, 1] - q[1])
        dist[list(used)] = 1e9
        i = int(np.argmin(dist))
        return i, float(dist[i])

    def info_score(d, q, used, Lam):
        i, dist = nearest_frame(d, q, used)
        if dist > 0.15:
            return -1e9, None
        pose = d["pose"][i]; w = d["w"][i]; h = d["h"][i]
        s2 = sigma2(q, pose, w, h)
        x1 = np.array([q[0], 1, 0, 0]); x2 = np.array([0, 0, q[1], 1])
        M = Lam + (x1[:, None] @ x1[None, :] + x2[:, None] @ x2[None, :]) / s2
        return float(np.linalg.slogdet(M)[1]), i

    def coverage_order(d):
        cells = defaultdict(list)
        for j in range(len(d["pred"])):
            cells[(min(2, int(d["pred"][j, 0] * 3)),
                   min(2, int(d["pred"][j, 1] * 3)))].append(j)
        for v in cells.values():
            np.random.RandomState(len(v)).shuffle(v)
        order = []
        keys = sorted(cells)
        while any(cells[k] for k in keys):
            for k in keys:
                if cells[k]:
                    order.append(cells[k].pop())
        return order

    def seq_calib(d):
        Lam0 = prior_prec()
        chosen = []
        used_cells = set()
        stop_reason = "kmax"
        th = mu.copy()
        sd = np.sqrt(np.diag(np.linalg.inv(Lam0)))

        def build():
            n = len(chosen)
            X = np.zeros((2 * n, 4)); y = np.zeros(2 * n); w = np.zeros(2 * n)
            for j, i in enumerate(chosen):
                p = d["pred"][i]; u = d["label"][i]
                s2 = sigma2(p, d["pose"][i], d["w"][i], d["h"][i])
                X[2 * j] = [p[0], 1, 0, 0]; y[2 * j] = u[0]
                X[2 * j + 1] = [0, 0, p[1], 1]; y[2 * j + 1] = u[1]
                w[2 * j] = w[2 * j + 1] = 1.0 / s2
            Lam = Lam0 + X.T @ (w[:, None] * X)
            eta = Lam0 @ mu + X.T @ (w * y)
            if a.debug and len(chosen) <= 2:
                print(f"      [build] n={n} X[:4]={np.round(X[:4],3).tolist()} "
                      f"y[:4]={np.round(y[:4],3).tolist()} w[:4]={np.round(w[:4],5).tolist()} "
                      f"mu={np.round(mu,3).tolist()}")
                try:
                    Xw = X * np.sqrt(w)[:, None]
                    th_ls = np.linalg.lstsq(Xw, y * np.sqrt(w), rcond=None)[0]
                    print(f"      [build] weighted_ls={np.round(th_ls,3).tolist()}")
                except Exception as e:
                    print("      [build] ls err", e)
            return Lam, eta

        order = coverage_order(d) if a.select_mode == "coverage" else None
        for step in range(a.kmax):
            Lam, _ = build()
            best = (-1e9, None, None)
            if a.select_mode == "coverage":
                nxt = None
                for i in order:
                    if i not in chosen:
                        nxt = i
                        break
                if nxt is None:
                    stop_reason = "no_candidate"
                    break
                best = (0.0, None, nxt)
            else:
                for q in qs:
                    sc, i = info_score(d, q, set(chosen), Lam)
                    if i is None or i in chosen:
                        continue
                    if any(np.hypot(q[0] - d["label"][j, 0], q[1] - d["label"][j, 1]) < a.min_dist
                           for j in chosen):
                        continue
                    cell = (min(2, int(q[0] * 3)), min(2, int(q[1] * 3)))
                    if a.coverage_cells and cell in used_cells and len(used_cells) < 9:
                        continue
                    if sc > best[0]:
                        best = (sc, q, i)
            if best[2] is None:
                stop_reason = "no_candidate"
                break
            chosen.append(best[2])
            qq = best[1] if best[1] is not None else d["pred"][best[2]]
            used_cells.add((min(2, int(qq[0] * 3)), min(2, int(qq[1] * 3))))
            Lam, eta = build()
            th = np.linalg.solve(Lam, eta)
            sd = np.sqrt(np.diag(np.linalg.inv(Lam)))
            if len(chosen) >= a.kmin and sd.max() < a.stop_std:
                stop_reason = "std"
                break
        return th, chosen, stop_reason, sd

    # 基线：B2 覆盖选点固定 K
    def coverage_sel(d, k, seed):
        rng = np.random.RandomState(abs(hash((d["label"][0, 0], k, seed))) % 2**31)
        pred = d["pred"]
        cells = defaultdict(list)
        for j in range(len(pred)):
            cells[(min(2, int(pred[j, 0] * 3)), min(2, int(pred[j, 1] * 3)))].append(j)
        per = max(1, k // max(1, len(cells)))
        sel = []
        for v in cells.values():
            v = list(v); rng.shuffle(v); sel += v[:per]
        rest = [j for j in range(len(pred)) if j not in set(sel)]
        if len(sel) < k:
            sel += list(rng.choice(rest, min(k - len(sel), len(rest)), replace=False))
        return sorted(set(sel))[:k]

    def map_ridge(d, sel):
        # B2: ridge lambda I around global mu
        th = []
        for ax in range(2):
            X = np.stack([d["pred"][sel, ax], np.ones(len(sel))], 1)
            A = X.T @ X + a.lam * np.eye(2)
            b = X.T @ d["label"][sel, ax] + a.lam * mu[ax * 2:ax * 2 + 2]
            th.append(np.linalg.solve(A, b))
        return th

    def eval_case(d, sel, th):
        ev = np.setdiff1d(np.arange(len(d["pred"])), sel)
        if len(ev) < 20:
            return None
        e0 = em(d["pred"][ev], d["label"][ev], d["sx"][ev], d["sy"][ev]).mean()
        e1 = em(apply_theta(d["pred"][ev], th), d["label"][ev],
                d["sx"][ev], d["sy"][ev]).mean()
        return e0, e1, len(ev)

    def eval_split(ST, name):
        print(f"\n=== {name} ===")
        rows = []
        for s, d in ST.items():
            th_seq, chosen, reason, sd = seq_calib(d)
            if a.debug:
                th_ls = fit_theta(d["pred"][chosen, 0], d["label"][chosen, 0],
                                  d["pred"][chosen, 1], d["label"][chosen, 1])
                print(f"    [debug] {s} theta={np.round(th_seq,4).tolist()} "
                      f"sd={np.round(sd,4).tolist()} "
                      f"ls=[{th_ls[0][0]:.3f},{th_ls[0][1]:.3f},{th_ls[1][0]:.3f},{th_ls[1][1]:.3f}]")
                for j in chosen[:3]:
                    print(f"      frame={j} pred={np.round(d['pred'][j],4).tolist()} "
                          f"label={np.round(d['label'][j],4).tolist()} "
                          f"pose={np.round(d['pose'][j][:5],3).tolist()}")
            r = eval_case(d, chosen, [th_seq[0:2], th_seq[2:4]])
            if r is None:
                continue
            rows.append(("seq", s, len(chosen), reason, sd.max(), *r))
            print(f"  seq  {s} K={len(chosen):2d} stop={reason:12s} "
                  f"sd_max={sd.max():.4f} base={r[0]:.4f} calib={r[1]:.4f} "
                  f"({100*(r[1]/r[0]-1):+.1f}%)")
        # 固定 K 覆盖基线
        for k in (3, 5, 9, 13):
            for s, d in ST.items():
                sel = coverage_sel(d, k, 0)
                th = map_ridge(d, sel)
                r = eval_case(d, sel, th)
                if r is None:
                    continue
                rows.append((f"b2_k{k}", s, len(sel), "fixed", 0.0, *r))
        df = defaultdict(list)
        for r in rows:
            df[r[0]].append(r)
        print("\n[summary]")
        for meth, rr in df.items():
            B = sum(x[5] * x[7] for x in rr) / sum(x[7] for x in rr)
            E = sum(x[6] * x[7] for x in rr) / sum(x[7] for x in rr)
            ratios = np.array([x[6] / x[5] - 1 for x in rr])
            ks = np.array([x[2] for x in rr])
            print(f"  {meth:8s} n={len(rr)} base={B:.4f} calib={E:.4f} "
                  f"({100*(E/B-1):+.2f}%) Kmean={ks.mean():.1f} "
                  f"改善 {int((ratios<0).sum())}/{len(ratios)} "
                  f"退化>2% {int((ratios>0.02).sum())} 最差 {100*ratios.max():+.0f}%")
        import csv as _csv
        from pathlib import Path as _P
        out = _P(a.out); out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = _csv.writer(fh)
            w.writerow(["method", "session", "K", "stop", "sd_max", "base", "calib", "n"])
            w.writerows(rows)
        print(f"[seq] saved {out} maxrss="
              f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024/1024:.2f}GB")

    eval_split(T, "test")


if __name__ == "__main__":
    main()
