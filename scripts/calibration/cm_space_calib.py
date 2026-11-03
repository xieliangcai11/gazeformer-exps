"""直接在 cm 空间做校准的干净实现。

流程：
  pred_norm/label_norm + index -> 转 canonical cm
  -> 每个 session/context 选 K 个支持帧
  -> 源域拟合 affine cm 先验 mu/Sigma
  -> 测试域 robust affine cm + 可选 DINO residual
  -> 在剩余帧上算 cm 误差

不依赖 new_input_calib 的多分支实验选项。
"""
import argparse, csv, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path[:0] = ["src", "."]
from gazelab.datasets.table_index import TableIndex  # noqa: E402


def log(m): print(m, flush=True)


def to_cm(p, scm, phi):
    x = (p[:, 0] - 0.5) * scm[:, 0]
    y = (p[:, 1] - 0.5) * scm[:, 1]
    c, s = np.cos(phi), np.sin(phi)
    return np.stack([c * x + s * y, -s * x + c * y], 1)


def to_norm(q, scm, phi):
    c, s = np.cos(phi), np.sin(phi)
    x = c * q[:, 0] - s * q[:, 1]
    y = s * q[:, 0] + c * q[:, 1]
    return np.stack([x / scm[:, 0] + 0.5, y / scm[:, 1] + 0.5], 1)


def fit_affine(P, U, lam, mu):
    out = []
    for ax in range(2):
        X = np.stack([P[:, ax], np.ones(len(P))], 1)
        A = X.T @ X + lam * np.eye(2)
        b = X.T @ U[:, ax] + lam * mu[ax * 2:ax * 2 + 2]
        out.append(np.linalg.solve(A, b))
    return out


def apply_affine(P, th):
    q = P.copy()
    q[:, 0] = th[0][0] * P[:, 0] + th[0][1]
    q[:, 1] = th[1][0] * P[:, 1] + th[1][1]
    return q


def fit_huber(P, U, lam, mu, delta=1.345, iters=6):
    th = fit_affine(P, U, lam, mu)
    for _ in range(iters):
        q = apply_affine(P, th)
        r = U - q
        scale = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        z = np.abs(r) / (delta * scale)
        w = np.where(z <= 1.0, 1.0, 1.0 / np.maximum(z, 1e-12))
        out = []
        for ax in range(2):
            X = np.stack([P[:, ax], np.ones(len(P))], 1)
            A = X.T @ (w[:, ax, None] * X) + lam * np.eye(2)
            b = X.T @ (w[:, ax] * U[:, ax]) + lam * mu[ax * 2:ax * 2 + 2]
            out.append(np.linalg.solve(A, b))
        th = out
    return th


def coverage_select(P, K, seed=0):
    rng = np.random.RandomState(seed)
    cells = defaultdict(list)
    for i in range(len(P)):
        cells[(min(2, int(P[i, 0] * 3)), min(2, int(P[i, 1] * 3)))].append(i)
    per = max(1, K // max(1, len(cells)))
    sel = []
    for v in cells.values():
        v = list(v); rng.shuffle(v); sel += v[:per]
    rest = [i for i in range(len(P)) if i not in set(sel)]
    if len(sel) < K:
        sel += list(rng.choice(rest, min(K - len(sel), len(rest)), replace=False))
    return np.array(sorted(set(sel))[:K], np.int64)


def load_npz(path, index_csv, context_key):
    z = np.load(path, allow_pickle=True)
    pred = z["pred"].astype(np.float64)
    label = z["label"].astype(np.float64)
    paths = [str(p) for p in z["path"].tolist()]
    ti = TableIndex(index_csv)
    lut = ti.lookup("frame_path" if "frame_path" in ti.fieldnames else "rgb_path")
    rows = np.array([lut.get(p, -1) for p in paths], np.int64)
    ok = rows >= 0
    rows = rows[ok]
    meta = []
    for r in rows:
        row = ti[int(r)]
        scm = np.array([float(row["screen_w"]) * float(row["cm_px_x"]),
                        float(row["screen_h"]) * float(row["cm_px_y"])])
        phi = float(np.arctan2(float(row["imu_x"]), -float(row["imu_y"])))
        if context_key == "orientation":
            ctx = str(row.get("orientation", "0"))
        elif context_key == "activity":
            ctx = str(row.get("activity", ""))
        else:
            ctx = ""
        meta.append((scm, phi, ctx, paths[len(meta)]))
    pred, label = pred[ok], label[ok]
    paths = [paths[i] for i in np.where(ok)[0]]
    scm = np.stack([m[0] for m in meta])
    phi = np.array([m[1] for m in meta])
    ctx = np.array([m[2] for m in meta])
    pc = to_cm(pred, scm, phi)
    lc = to_cm(label, scm, phi)
    groups = defaultdict(list)
    for i, p in enumerate(paths):
        groups[f"{p.rsplit('/', 1)[0]}||{ctx[i]}"].append(i)
    out = {}
    for k, idx in groups.items():
        idx = np.array(sorted(idx, key=lambda i: paths[i]), np.int64)
        if len(idx) < 40:
            continue
        out[k] = dict(pred=pred[idx], label=label[idx], pc=pc[idx], lc=lc[idx],
                      scm=scm[idx], phi=phi[idx], idx=idx, rows=rows[idx],
                      paths=[paths[i] for i in idx])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--extra-npz", default="")
    ap.add_argument("--context-key", default="orientation")
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--delta", type=float, default=1.345)
    ap.add_argument("--dino-lam", type=float, default=1.0)
    ap.add_argument("--base-space", choices=["norm", "cm"], default="cm",
                    help="基础校准所在空间；norm=原始归一化坐标，cm=canonical cm")
    ap.add_argument("--resid-space", choices=["norm", "cm"], default="cm",
                    help="DINO 残差拟合/裁剪所在空间")
    ap.add_argument("--resid-clip", type=float, default=0.0,
                    help="残差裁剪阈值（resid-space 单位）；0=自动 norm 0.02 / cm 0.15")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    log(f"[cmcal] args={vars(a)}")
    D = load_npz(a.train_npz, a.index_csv, a.context_key)
    T = load_npz(a.test_npz, a.index_csv, a.context_key)
    log(f"[cmcal] groups source={len(D)} test={len(T)}")

    extra = None
    if a.extra_npz:
        z = np.load(a.extra_npz, allow_pickle=True)
        er = z["row_idx"].astype(np.int64); ef = z["feats"].astype(np.float64)
        ev = z["valid"].astype(bool)
        extra = (er, ef, ev)
        log(f"[cmcal] extra dims={ef.shape[1]} valid={int(ev.sum())}")

    # source prior in cm and norm
    TH = []; THn = []
    for s, d in D.items():
        th = fit_affine(d["pc"], d["lc"], 1e-6, np.array([1., 0., 1., 0.]))
        TH.append(np.array([th[0][0], th[0][1], th[1][0], th[1][1]]))
        thn = fit_affine(d["pred"], d["label"], 1e-6, np.array([1., 0., 1., 0.]))
        THn.append(np.array([thn[0][0], thn[0][1], thn[1][0], thn[1][1]]))
    TH = np.array(TH); mu = TH.mean(0); S = np.cov(TH.T) + 1e-6 * np.eye(4)
    THn = np.array(THn); mun = THn.mean(0); Sn = np.cov(THn.T) + 1e-6 * np.eye(4)
    log(f"[cmcal] prior_cm mu={np.round(mu,4).tolist()} sd={np.round(np.sqrt(np.diag(S)),4).tolist()}")
    log(f"[cmcal] prior_norm mu={np.round(mun,4).tolist()} sd={np.round(np.sqrt(np.diag(Sn)),4).tolist()}")

    clip = a.resid_clip if a.resid_clip > 0 else (0.02 if a.resid_space == "norm" else 0.15)
    log(f"[cmcal] base_space={a.base_space} resid_space={a.resid_space} clip={clip}")
    rows = []
    Bn = Cn = Nn = 0.0
    Bn_n = Cn_n = 0.0
    Rn = Rn_n = 0.0
    for s, d in T.items():
        raw_errs = []; base_errs = []; cal_errs = []
        raw_errs_n = []; base_errs_n = []; cal_errs_n = []
        for seed in range(a.seeds):
            sel = coverage_select(d["pred"], a.K, seed)
            q = np.setdiff1d(np.arange(len(d["pred"])), sel)
            if len(q) < 20:
                continue
            raw_errs.append(float(np.linalg.norm(d["pc"][q] - d["lc"][q], axis=1).mean()))
            raw_errs_n.append(float(np.linalg.norm(d["pred"][q] - d["label"][q], axis=1).mean()))
            if a.base_space == "cm":
                th = fit_huber(d["pc"][sel], d["lc"][sel], a.lam, mu, a.delta)
                q0_cm = apply_affine(d["pc"][q], th)
                q0_cm_sel = apply_affine(d["pc"][sel], th)
                q0_norm = to_norm(q0_cm, d["scm"][q], d["phi"][q])
                q0_norm_sel = to_norm(q0_cm_sel, d["scm"][sel], d["phi"][sel])
            else:
                th = fit_huber(d["pred"][sel], d["label"][sel], a.lam, mun, a.delta)
                q0_norm = apply_affine(d["pred"][q], th)
                q0_norm_sel = apply_affine(d["pred"][sel], th)
                q0_cm = to_cm(q0_norm, d["scm"][q], d["phi"][q])
                q0_cm_sel = to_cm(q0_norm_sel, d["scm"][sel], d["phi"][sel])
            e0 = float(np.linalg.norm(q0_cm - d["lc"][q], axis=1).mean())
            e0n = float(np.linalg.norm(q0_norm - d["label"][q], axis=1).mean())
            base_errs.append(e0); base_errs_n.append(e0n)
            q1_cm = q0_cm.copy(); q1_norm = q0_norm.copy()
            if extra is not None:
                er, ef, ev = extra
                fmap = {int(r): i for i, r in enumerate(er) if bool(ev[i])}
                srows = [int(x) for x in d["rows"][sel]]
                qrows = [int(x) for x in d["rows"][q]]
                si = [fmap[r] for r in srows if r in fmap]
                qi = [fmap[r] for r in qrows if r in fmap]
                if len(si) == len(sel) and len(qi) == len(q):
                    Xs = ef[si]
                    muX = Xs.mean(0); sdX = Xs.std(0) + 1e-6
                    Zs = np.clip((Xs - muX) / sdX, -5, 5)
                    Zs = np.concatenate([Zs, np.ones((len(Zs), 1))], 1)
                    Zq = np.clip((ef[qi] - muX) / sdX, -5, 5)
                    Zq = np.concatenate([Zq, np.ones((len(Zq), 1))], 1)
                    if a.resid_space == "cm":
                        Rs = d["lc"][sel] - q0_cm_sel
                    else:
                        Rs = d["label"][sel] - q0_norm_sel
                    ws = []
                    for ax in range(2):
                        A = Zs.T @ Zs + a.dino_lam * np.eye(Zs.shape[1])
                        ws.append(np.linalg.solve(A, Zs.T @ Rs[:, ax]))
                    corr = np.stack([Zq @ ws[0], Zq @ ws[1]], 1)
                    corr = np.clip(corr, -clip, clip)
                    if a.resid_space == "cm":
                        q1_cm = q0_cm + corr
                        q1_norm = to_norm(q1_cm, d["scm"][q], d["phi"][q])
                    else:
                        q1_norm = q0_norm + corr
                        q1_cm = to_cm(q1_norm, d["scm"][q], d["phi"][q])
            cal_errs.append(float(np.linalg.norm(q1_cm - d["lc"][q], axis=1).mean()))
            cal_errs_n.append(float(np.linalg.norm(q1_norm - d["label"][q], axis=1).mean()))
        if base_errs:
            n = len(d["pred"]) - int(a.K)
            rm = float(np.mean(raw_errs)); rmn = float(np.mean(raw_errs_n))
            bm = float(np.mean(base_errs)); cmv = float(np.mean(cal_errs))
            bmn = float(np.mean(base_errs_n)); cmn = float(np.mean(cal_errs_n))
            rows.append([s, len(d["pred"]), int(a.K), a.seeds,
                         rm, bm, cmv, cmv / max(rm, 1e-9) - 1,
                         rmn, bmn, cmn, cmn / max(rmn, 1e-9) - 1])
            Rn += rm * n; Bn += bm * n; Cn += cmv * n
            Rn_n += rmn * n; Bn_n += bmn * n; Cn_n += cmn * n; Nn += n
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["session", "n", "K", "seeds",
                    "raw_cm", "base_cm", "calib_cm", "gain_vs_raw_cm",
                    "raw_norm", "base_norm", "calib_norm", "gain_vs_raw_norm"])
        w.writerows(rows)
    R = Rn / max(Nn, 1); B = Bn / max(Nn, 1); C = Cn / max(Nn, 1)
    Rn_ = Rn_n / max(Nn, 1); Bn_ = Bn_n / max(Nn, 1); Cn_ = Cn_n / max(Nn, 1)
    log(f"[cmcal] RESULT sessions={len(rows)} raw_cm={R:.4f} base_cm={B:.4f} "
        f"calib_cm={C:.4f} ({100*(C/R-1):+.2f}%) raw_norm={Rn_:.4f} "
        f"base_norm={Bn_:.4f} calib_norm={Cn_:.4f} ({100*(Cn_/Rn_-1):+.2f}%)")


if __name__ == "__main__": main()
