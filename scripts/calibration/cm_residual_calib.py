"""cm 模型专用校准 v2：多源特征残差 + 目标标准化 + LOO λ + support gate。

设计目标
--------
1. 先能逐位复现 N1 baseline（norm base + DINO16 残差，fixed λ=1.0，clip=0.02 norm）；
2. 在同一套 support / 同一 base 上，逐步打开：
   - 多源特征（DINO16 + landmark21 + pose8）；
   - 残差目标按轴标准化（让 λ 跨空间可比）；
   - LOO 选 ridge λ；
   - support LOO 选收缩系数 alpha，并在 LOO 不改善时 gate 掉残差；
   - 数据驱动 clip（support 残差分位数）。
3. 日志必须足够排错：输入形状/有效数/丢弃原因/每会话/每 seed/最终汇总/资源。

坐标约定
--------
- `pred`/`label` 是归一化坐标 [0,1]；
- canonical cm 用 `phi=atan2(imu_x,-imu_y)` 旋转；
- `scm=[screen_w*cm_px_x, screen_h*cm_px_y]`；
- 评价主指标是 cm，norm 只作辅助。

不做什么
--------
- 不改模型、不训练；
- 不并行多进程；一次一个进程；
- 不自动调参到 test：所有默认值来自既有口径或 val 诊断。
"""
from __future__ import annotations

import os

# 在 import numpy 之前限制 BLAS 线程，避免单进程把 CPU 打满。
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path[:0] = ["src", "."]
from gazelab.datasets.table_index import TableIndex  # noqa: E402

try:  # resource 只在 Linux 可用；Windows 本地编译时跳过。
    import resource  # type: ignore
except Exception:  # pragma: no cover
    resource = None  # type: ignore


_T0 = time.time()
_LOG_LEVEL = 1  # 0=warn, 1=info, 2=debug


def log(msg: str, level: int = 1) -> None:
    if level > _LOG_LEVEL:
        return
    tag = {0: "WARN", 1: "INFO", 2: "DBG"}.get(level, "INFO")
    print(f"[{tag} {time.strftime('%H:%M:%S')} +{time.time() - _T0:8.1f}s] {msg}",
          flush=True)


def warn(msg: str) -> None:
    log("!! " + msg, level=0)


def maxrss_gb() -> float:
    if resource is None:
        return float("nan")
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# 坐标变换
# ---------------------------------------------------------------------------

def to_cm(p: np.ndarray, scm: np.ndarray, phi: np.ndarray) -> np.ndarray:
    x = (p[:, 0] - 0.5) * scm[:, 0]
    y = (p[:, 1] - 0.5) * scm[:, 1]
    c, s = np.cos(phi), np.sin(phi)
    return np.stack([c * x + s * y, -s * x + c * y], 1)


def to_norm(q: np.ndarray, scm: np.ndarray, phi: np.ndarray) -> np.ndarray:
    c, s = np.cos(phi), np.sin(phi)
    x = c * q[:, 0] - s * q[:, 1]
    y = s * q[:, 0] + c * q[:, 1]
    return np.stack([x / scm[:, 0] + 0.5, y / scm[:, 1] + 0.5], 1)


# ---------------------------------------------------------------------------
# 仿射 + Huber（与 cm_space_calib.py 的 baseline 保持逐行一致）
# ---------------------------------------------------------------------------

def fit_affine(P: np.ndarray, U: np.ndarray, lam: float, mu: np.ndarray):
    out = []
    for ax in range(2):
        X = np.stack([P[:, ax], np.ones(len(P))], 1)
        A = X.T @ X + lam * np.eye(2)
        b = X.T @ U[:, ax] + lam * mu[ax * 2:ax * 2 + 2]
        out.append(np.linalg.solve(A, b))
    return out


def apply_affine(P: np.ndarray, th) -> np.ndarray:
    q = P.copy()
    q[:, 0] = th[0][0] * P[:, 0] + th[0][1]
    q[:, 1] = th[1][0] * P[:, 1] + th[1][1]
    return q


def fit_huber(P: np.ndarray, U: np.ndarray, lam: float, mu: np.ndarray,
              delta: float = 1.345, iters: int = 6,
              scale_mode: str = "joint"):
    """Per-axis affine + Huber IRLS.

    scale_mode="joint": baseline 行为，两轴共享一个 MAD scale（复现 N1）。
    scale_mode="axis" : 每轴独立 MAD scale（cm x/y 尺度不同，P3 用）。
    """
    th = fit_affine(P, U, lam, mu)
    for _ in range(iters):
        q = apply_affine(P, th)
        r = U - q
        if scale_mode == "axis":
            scales = np.array([
                1.4826 * np.median(np.abs(r[:, ax] - np.median(r[:, ax]))) + 1e-9
                for ax in range(2)
            ])
        else:
            scales = np.full(2, 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9)
        out = []
        for ax in range(2):
            z = np.abs(r[:, ax]) / (delta * scales[ax])
            w = np.where(z <= 1.0, 1.0, 1.0 / np.maximum(z, 1e-12))
            X = np.stack([P[:, ax], np.ones(len(P))], 1)
            A = X.T @ (w[:, None] * X) + lam * np.eye(2)
            b = X.T @ (w * U[:, ax]) + lam * mu[ax * 2:ax * 2 + 2]
            out.append(np.linalg.solve(A, b))
        th = out
    return th


def coverage_select(P: np.ndarray, K: int, seed: int = 0) -> np.ndarray:
    """与 cm_space_calib.coverage_select 完全一致：norm 3x3 + 随机补足。"""
    rng = np.random.RandomState(seed)
    cells = defaultdict(list)
    for i in range(len(P)):
        cells[(min(2, int(P[i, 0] * 3)), min(2, int(P[i, 1] * 3)))].append(i)
    per = max(1, K // max(1, len(cells)))
    sel = []
    for v in cells.values():
        v = list(v)
        rng.shuffle(v)
        sel += v[:per]
    rest = [i for i in range(len(P)) if i not in set(sel)]
    if len(sel) < K:
        sel += list(rng.choice(rest, min(K - len(sel), len(rest)), replace=False))
    return np.array(sorted(set(sel))[:K], np.int64)


def region_feats(P: np.ndarray, mode: str) -> np.ndarray:
    """屏幕分区 one-hot + 与预测坐标的交互；列顺序由调用方统一处理。"""
    if mode == "none":
        return np.zeros((len(P), 0), np.float64)
    R = 2 if mode == "2x2" else 3
    rx = np.clip((P[:, 0] * R).astype(int), 0, R - 1)
    ry = np.clip((P[:, 1] * R).astype(int), 0, R - 1)
    idx = ry * R + rx
    oh = np.zeros((len(P), R * R), np.float64)
    oh[np.arange(len(P)), idx] = 1.0
    inter = np.concatenate([oh * P[:, 0:1], oh * P[:, 1:2]], 1)
    return np.concatenate([oh, inter], 1)


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_sessions(npz_path: str, table: TableIndex, context_key: str,
                  min_frames: int = 40, drop_invalid_imu: bool = False):
    """把 npz 按 session dir + context 分组；返回原始 norm/cm 数据和元信息。"""
    z = np.load(npz_path, allow_pickle=True)
    for k in ("pred", "label", "path"):
        if k not in z:
            raise SystemExit(f"{npz_path} 缺少键 {k}；现有键={list(z.keys())}")
    pred = z["pred"].astype(np.float64)
    label = z["label"].astype(np.float64)
    paths = [str(p) for p in z["path"].tolist()]
    if len(pred) != len(label) or len(pred) != len(paths):
        raise SystemExit(f"{npz_path} 长度不一致 pred={len(pred)} label={len(label)} "
                         f"path={len(paths)}")

    pcol = "frame_path" if "frame_path" in table.fieldnames else "rgb_path"
    lut = table.lookup(pcol)
    rows = np.array([lut.get(p, -1) for p in paths], np.int64)
    ok = rows >= 0
    n_missing = int((~ok).sum())
    if n_missing:
        warn(f"{npz_path}: {n_missing}/{len(paths)} 条 path 在 index 中找不到")
    pred, label = pred[ok], label[ok]
    paths = [paths[i] for i in np.where(ok)[0]]
    rows = rows[ok]

    n = len(paths)
    scm = np.zeros((n, 2), np.float64)
    phi = np.zeros(n, np.float64)
    ctx = np.empty(n, object)
    dev = np.empty(n, object)
    imu_ok = np.ones(n, bool)
    bad_scm = 0
    bad_imu = 0
    for i, r in enumerate(rows):
        row = table[int(r)]
        w = float(row["screen_w"]) * float(row["cm_px_x"])
        h = float(row["screen_h"]) * float(row["cm_px_y"])
        scm[i] = (w, h)
        try:
            phi[i] = math.atan2(float(row["imu_x"]), -float(row["imu_y"]))
        except Exception:
            phi[i] = 0.0
            imu_ok[i] = False
        gxy = 0.0
        try:
            gxy = math.hypot(float(row["imu_x"]), float(row["imu_y"]))
        except Exception:
            pass
        if not (math.isfinite(gxy) and gxy >= 0.05):
            imu_ok[i] = False
        if context_key == "orientation":
            ctx[i] = str(row.get("orientation", "0"))
        elif context_key == "activity":
            ctx[i] = str(row.get("activity", ""))
        elif context_key == "subject":
            ctx[i] = str(row.get("subject", ""))
        elif context_key == "session":
            ctx[i] = "session"
        else:
            ctx[i] = ""
        dev[i] = str(row.get("device", "?"))

    finite = np.isfinite(scm).all(1) & np.isfinite(phi)
    if (~finite).any():
        bad_scm = int((~finite).sum())
        warn(f"{npz_path}: {bad_scm} 条 scm/phi 非有限，已丢弃")
    if drop_invalid_imu:
        keep = finite & imu_ok
    else:
        keep = finite
    bad_imu = int((~imu_ok).sum())
    if bad_imu:
        warn(f"{npz_path}: {bad_imu}/{n} 条 IMU |g_xy|<0.05 或缺失"
             f"（drop_invalid_imu={drop_invalid_imu}）")
    if not keep.all():
        pred, label = pred[keep], label[keep]
        paths = [paths[i] for i in np.where(keep)[0]]
        rows, scm, phi = rows[keep], scm[keep], phi[keep]
        ctx, dev, imu_ok = ctx[keep], dev[keep], imu_ok[keep]

    pc = to_cm(pred, scm, phi)
    lc = to_cm(label, scm, phi)

    groups = defaultdict(list)
    for i, p in enumerate(paths):
        groups[f"{p.rsplit('/', 1)[0]}||{ctx[i]}"].append(i)

    out = {}
    dropped = 0
    for s, idx in groups.items():
        idx = np.array(sorted(idx, key=lambda i: paths[i]), np.int64)
        if len(idx) < min_frames:
            dropped += 1
            continue
        out[s] = dict(
            pred=pred[idx], label=label[idx], pc=pc[idx], lc=lc[idx],
            scm=scm[idx], phi=phi[idx], rows=rows[idx],
            paths=[paths[i] for i in idx], ori=str(ctx[idx[0]]),
            device=str(dev[idx[0]]), imu_ok=imu_ok[idx],
        )
    log(f"[load] {npz_path}: frames={len(paths)} groups={len(groups)} "
        f"kept={len(out)} dropped<{min_frames}={dropped} "
        f"missing_path={n_missing} bad_scm={bad_scm} bad_imu={bad_imu}")
    return out


def load_feature_source(path: str, prefix: str):
    z = np.load(path, allow_pickle=True)
    if "row_idx" not in z or "feats" not in z:
        raise SystemExit(f"{path} 缺少 row_idx/feats；现有={list(z.keys())}")
    r = z["row_idx"].astype(np.int64)
    f = np.asarray(z["feats"], np.float32)
    v = np.asarray(z["valid"], bool) if "valid" in z else np.ones(len(r), bool)
    if len(r) != len(f) or len(v) != len(r):
        raise SystemExit(f"{path} 长度不一致 row_idx={len(r)} feats={len(f)} "
                         f"valid={len(v)}")
    if "names" in z:
        names = [str(x) for x in z["names"].tolist()]
    else:
        names = [f"{prefix}{i}" for i in range(f.shape[1])]
    if len(names) != f.shape[1]:
        warn(f"{path}: names 数 {len(names)} != dims {f.shape[1]}，用序号补齐")
        names = [f"{prefix}{i}" for i in range(f.shape[1])]
    names = [f"{prefix}:{x}" for x in names]
    dup = len(r) - len(np.unique(r))
    if dup:
        warn(f"{path}: row_idx 有 {dup} 个重复；后写覆盖前写")
    log(f"[feat] {path}: rows={len(r)} dims={f.shape[1]} valid={int(v.sum())} "
        f"dup={dup} prefix={prefix}")
    return dict(row=r, feat=f, valid=v, names=names, path=path, prefix=prefix)


def build_feature_matrix(sources, n_rows: int):
    dims = sum(s["feat"].shape[1] for s in sources)
    F = np.zeros((n_rows, dims), np.float32)
    V = np.ones(n_rows, bool)
    covered = np.zeros(n_rows, bool)
    off = 0
    names = []
    for s in sources:
        r = s["row"]
        d = s["feat"].shape[1]
        inb = (r >= 0) & (r < n_rows)
        bad = int((~inb).sum())
        if bad:
            warn(f"{s['path']}: {bad} 个 row_idx 越界，已忽略")
        r, f, v = r[inb], s["feat"][inb], s["valid"][inb]
        F[r, off:off + d] = f
        V[r] &= v
        covered[r] = True
        names += s["names"]
        off += d
    V &= covered
    log(f"[feat] matrix={F.shape} covered={int(covered.sum())}/{n_rows} "
        f"all_valid={int(V.sum())} names={len(names)}")
    return F, V, names


# ---------------------------------------------------------------------------
# 特征标准化 + ridge / LOO
# ---------------------------------------------------------------------------

def prepare_features(Xs: np.ndarray, Xq: np.ndarray,
                     Vs: np.ndarray, Vq: np.ndarray):
    """support 上标准化，clip ±5，缺失/无效填 0，最后加截距。"""
    Xs = np.asarray(Xs, np.float64)
    Xq = np.asarray(Xq, np.float64)
    ok = Vs & np.isfinite(Xs).all(1)
    need = max(10, Xs.shape[1] // 2)
    if ok.sum() < need:
        return None
    mu = Xs[ok].mean(0)
    sd = Xs[ok].std(0) + 1e-6
    Zs = np.clip((Xs - mu) / sd, -5.0, 5.0)
    Zq = np.clip((Xq - mu) / sd, -5.0, 5.0)
    Zs[~np.isfinite(Zs)] = 0.0
    Zq[~np.isfinite(Zq)] = 0.0
    # 无效行按均值（标准化后的 0）处理，并记录。
    Zs[~Vs] = 0.0
    Zq[~Vq] = 0.0
    Zs = np.concatenate([Zs, np.ones((len(Zs), 1))], 1)
    Zq = np.concatenate([Zq, np.ones((len(Zq), 1))], 1)
    return Zs, Zq, mu, sd, int(ok.sum())


def ridge_pick(Z: np.ndarray, y: np.ndarray, lams):
    """闭式 LOO 选 λ；返回最优 lam/mse/w/loo_residual。"""
    best = None
    D = Z.shape[1]
    for lam in lams:
        A = Z.T @ Z + float(lam) * np.eye(D)
        A = A + 1e-10 * np.eye(D)
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            continue
        w = A_inv @ (Z.T @ y)
        h = np.einsum("ij,jk,ik->i", Z, A_inv, Z)
        h = np.clip(h, 0.0, 0.999999)
        loo = (y - Z @ w) / np.maximum(1.0 - h, 1e-9)
        mse = float(np.mean(loo ** 2))
        if not np.isfinite(mse):
            continue
        if best is None or mse < best["mse"]:
            best = dict(lam=float(lam), mse=mse, w=w, loo=loo)
    return best


def fit_residual(Zs: np.ndarray, Zq: np.ndarray, Rs: np.ndarray, a):
    """拟合 support 残差；返回 (corr, info)。"""
    nq = len(Zq)
    corr = np.zeros((nq, 2), np.float64)
    info = dict(
        lam_x=float("nan"), lam_y=float("nan"),
        alpha_x=float("nan"), alpha_y=float("nan"),
        mse0_x=float("nan"), mse0_y=float("nan"),
        mse1_x=float("nan"), mse1_y=float("nan"),
        clip_x=float("nan"), clip_y=float("nan"),
        clip_frac_x=0.0, clip_frac_y=0.0,
    )
    if a.target_standardize:
        med = np.median(Rs, 0)
        scale = 1.4826 * np.median(np.abs(Rs - med), 0) + 1e-9
        scale = np.maximum(scale, 1e-6)
    else:
        scale = np.ones(2)
    info["scale_x"] = float(scale[0])
    info["scale_y"] = float(scale[1])

    for ax in range(2):
        y = Rs[:, ax] / scale[ax]
        if a.loo_lam:
            best = ridge_pick(Zs, y, a.lams)
        else:
            best = ridge_pick(Zs, y, [a.fixed_lam])
        if best is None:
            warn(f"ridge 无解 ax={ax}；该轴残差置 0")
            continue
        w = best["w"]
        lam = best["lam"]
        loo = best["loo"]
        pred_loo = y - loo

        if a.hard_gate:
            mse0 = float(np.mean(y ** 2))
            mse1 = float(np.mean((pred_loo - y) ** 2))
            alpha = 1.0 if mse1 < mse0 * (1.0 - a.gate_min_improve) else 0.0
        elif a.shrink or a.gate:
            denom = float(pred_loo @ pred_loo)
            alpha = float((pred_loo @ y) / denom) if denom > 1e-12 else 0.0
            alpha = float(np.clip(alpha, 0.0, 1.0))
            mse0 = float(np.mean(y ** 2))
            mse1 = float(np.mean((alpha * pred_loo - y) ** 2))
            if a.gate and mse1 >= mse0 * (1.0 - a.gate_min_improve):
                alpha = 0.0
        else:
            alpha = 1.0
            mse0 = float(np.mean(y ** 2))
            mse1 = float(np.mean((pred_loo - y) ** 2))

        raw = (Zq @ w) * scale[ax]
        if a.clip_mode == "quantile":
            qv = float(np.quantile(np.abs(Rs[:, ax]), a.clip_q))
            clip = max(qv * a.clip_factor, 1e-9)
        else:
            clip = a.clip_norm if a.resid_space == "norm" else a.clip_cm
        c = np.clip(alpha * raw, -clip, clip)
        corr[:, ax] = c
        info.update({
            f"lam_{'x' if ax == 0 else 'y'}": float(lam),
            f"alpha_{'x' if ax == 0 else 'y'}": float(alpha),
            f"mse0_{'x' if ax == 0 else 'y'}": float(mse0),
            f"mse1_{'x' if ax == 0 else 'y'}": float(mse1),
            f"clip_{'x' if ax == 0 else 'y'}": float(clip),
            f"clip_frac_{'x' if ax == 0 else 'y'}": float(np.mean(np.abs(c) >= clip * 0.999)),
        })
    return corr, info


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------

def wmean(values, weights):
    values = np.asarray(values, np.float64)
    weights = np.asarray(weights, np.float64)
    return float((values * weights).sum() / max(weights.sum(), 1e-12))


def boot_ci(values, weights, reps=10000, seed=12345):
    values = np.asarray(values, np.float64)
    weights = np.asarray(weights, np.float64)
    rng = np.random.RandomState(seed)
    n = len(values)
    out = []
    for _ in range(reps):
        idx = rng.randint(0, n, n)
        out.append(float((values[idx] * weights[idx]).sum() /
                         max(weights[idx].sum(), 1e-12)))
    return np.percentile(out, [2.5, 97.5]).tolist()


def summarize(rows, args):
    """rows: 每会话 mean-over-seeds 指标。"""
    keys = ["raw_cm", "base_cm", "calib_cm", "raw_norm", "base_norm", "calib_norm"]
    w = np.array([r["n_heldout"] for r in rows], np.float64)
    out = dict(sessions=len(rows), heldout_frames=int(w.sum()))
    for k in keys:
        out[k] = wmean([r[k] for r in rows], w)
    out["gain_vs_raw_cm"] = out["calib_cm"] / max(out["raw_cm"], 1e-12) - 1
    out["gain_vs_base_cm"] = out["calib_cm"] / max(out["base_cm"], 1e-12) - 1
    out["gain_vs_raw_norm"] = out["calib_norm"] / max(out["raw_norm"], 1e-12) - 1
    out["gain_vs_base_norm"] = out["calib_norm"] / max(out["base_norm"], 1e-12) - 1
    out["calib_minus_raw_cm_ci95"] = boot_ci(
        [r["calib_cm"] - r["raw_cm"] for r in rows], w)
    out["calib_minus_raw_norm_ci95"] = boot_ci(
        [r["calib_norm"] - r["raw_norm"] for r in rows], w)
    out["improved_cm"] = int(sum(r["calib_cm"] < r["base_cm"] for r in rows))
    out["improved_norm"] = int(sum(r["calib_norm"] < r["base_norm"] for r in rows))
    out["worst_gain_cm"] = max((r["calib_cm"] / max(r["base_cm"], 1e-12) - 1)
                               for r in rows)
    out["median_n_heldout"] = float(np.median([r["n_heldout"] for r in rows]))
    # 按 orientation / device
    for group_key in ("ori", "device"):
        tab = {}
        for g in sorted({str(r[group_key]) for r in rows}):
            sub = [r for r in rows if str(r[group_key]) == g]
            ww = np.array([r["n_heldout"] for r in sub], np.float64)
            tab[g] = dict(
                sessions=len(sub), frames=int(ww.sum()),
                raw_cm=wmean([r["raw_cm"] for r in sub], ww),
                base_cm=wmean([r["base_cm"] for r in sub], ww),
                calib_cm=wmean([r["calib_cm"] for r in sub], ww),
                raw_norm=wmean([r["raw_norm"] for r in sub], ww),
                base_norm=wmean([r["base_norm"] for r in sub], ww),
                calib_norm=wmean([r["calib_norm"] for r in sub], ww),
            )
        out[group_key] = tab
    out["args"] = vars(args)
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="cm 模型校准 v2（多源特征 + LOO + gate）")
    ap.add_argument("--train-npz", required=True, help="源域（val）预测/标签 npz")
    ap.add_argument("--test-npz", required=True, help="目标域（test）预测/标签 npz")
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--dino-npz", default="")
    ap.add_argument("--landmark-npz", default="")
    ap.add_argument("--pose-npz", default="")
    ap.add_argument("--features", default="dino",
                    help="逗号分隔：dino,landmark,pose；必须提供对应 npz")
    ap.add_argument("--context-key", default="orientation")
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--lam", type=float, default=1.0, help="base Huber 先验强度")
    ap.add_argument("--delta", type=float, default=1.345, help="base Huber delta")
    ap.add_argument("--huber-scale", choices=["joint", "axis"], default="joint")
    ap.add_argument("--prior-context", choices=["global", "orientation", "device"],
                    default="global", help="source prior 按什么上下文分组")
    ap.add_argument("--min-ctx-sessions", type=int, default=5,
                    help="某上下文源会话少于此数则回退 global")
    ap.add_argument("--no-residual", action="store_true",
                    help="只评估 base（跳过特征残差），用于先验/鲁棒消融")
    ap.add_argument("--base-space", choices=["norm", "cm"], default="norm")
    ap.add_argument("--resid-space", choices=["norm", "cm"], default="norm")
    ap.add_argument("--target-standardize", action="store_true")
    ap.add_argument("--loo-lam", action="store_true")
    ap.add_argument("--lams", type=float, nargs="+",
                    default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0])
    ap.add_argument("--fixed-lam", type=float, default=1.0)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--shrink", action="store_true",
                    help="用 support LOO 估计收缩系数 alpha；与 --gate 分开")
    ap.add_argument("--hard-gate", action="store_true",
                    help="LOO 有改善则 alpha=1，否则 alpha=0（不做连续收缩）")
    ap.add_argument("--gate-min-improve", type=float, default=0.001)
    ap.add_argument("--clip-mode", choices=["fixed", "quantile"], default="fixed")
    ap.add_argument("--clip-norm", type=float, default=0.02)
    ap.add_argument("--clip-cm", type=float, default=0.15)
    ap.add_argument("--clip-q", type=float, default=0.95)
    ap.add_argument("--clip-factor", type=float, default=2.0)
    ap.add_argument("--regions", choices=["none", "2x2", "3x3"], default="none",
                    help="在残差特征中加入屏幕分区 one-hot + 与预测坐标的交互")
    ap.add_argument("--min-frames", type=int, default=40)
    ap.add_argument("--min-session-frames", type=int, default=0,
                    help="只评估总帧数 >= N 的会话（0=全部；用于固定 K 曲线会话集）")
    ap.add_argument("--limit-sessions", type=int, default=0,
                    help="只跑前 N 个会话（冒烟）；0=全部")
    ap.add_argument("--log-sessions", action="store_true", default=True)
    ap.add_argument("--no-log-sessions", dest="log_sessions", action="store_false")
    ap.add_argument("--log-level", type=int, default=1)
    ap.add_argument("--detail", default="", help="每 seed 明细 CSV")
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", default="", help="汇总 JSON；默认由 out 派生")
    ap.add_argument("--args-out", default="", help="参数 JSON；默认由 out 派生")
    a = ap.parse_args()
    global _LOG_LEVEL
    _LOG_LEVEL = a.log_level

    log("=" * 78)
    log("cm_residual_calib v2")
    log(f"host={platform.node()} python={sys.version.split()[0]} "
        f"numpy={np.__version__} pid={os.getpid()}")
    log(f"cwd={Path.cwd()}")
    script = Path(__file__).resolve()
    log(f"script={script} sha256={hashlib.sha256(script.read_bytes()).hexdigest()[:16]}")
    log(f"args={json.dumps(vars(a), ensure_ascii=False, default=str)}")
    log(f"maxrss_start={maxrss_gb():.3f}GB")
    for p in (a.train_npz, a.test_npz, a.index_csv,
              a.dino_npz, a.landmark_npz, a.pose_npz):
        if p:
            pp = Path(p)
            log(f"input {p}: exists={pp.exists()} "
                f"size={pp.stat().st_size if pp.exists() else -1} "
                f"mtime={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(pp.stat().st_mtime)) if pp.exists() else '-'}")
    log(f"flags base_space={a.base_space} resid_space={a.resid_space} "
        f"target_std={a.target_standardize} loo_lam={a.loo_lam} gate={a.gate} "
        f"clip={a.clip_mode} features={a.features}")
    if a.seeds < 1:
        raise SystemExit("--seeds 必须 >=1")
    if a.K < 2:
        raise SystemExit("--K 必须 >=2")
    if a.base_space == "cm" and a.resid_space == "norm":
        log("[note] cm base + norm residual：会做显式 to_norm/to_cm")
    if a.base_space == "norm" and a.resid_space == "cm":
        log("[note] norm base + cm residual：会做显式 to_cm")

    table = TableIndex(a.index_csv)
    log(f"[index] rows={len(table)} fields={table.fieldnames}")
    D = load_sessions(a.train_npz, table, a.context_key, a.min_frames,
                      drop_invalid_imu=False)
    T = load_sessions(a.test_npz, table, a.context_key, a.min_frames,
                      drop_invalid_imu=False)
    if not D or not T:
        raise SystemExit(f"空会话：source={len(D)} test={len(T)}")
    log(f"[sessions] source={len(D)} test={len(T)} "
        f"test_ori={dict(Counter(d['ori'] for d in T.values()))} "
        f"test_dev={dict(Counter(d['device'] for d in T.values()).most_common(8))}")
    log(f"[sessions] test frames={sum(len(d['pred']) for d in T.values())} "
        f"source frames={sum(len(d['pred']) for d in D.values())}")

    # 源域先验（norm/cm 两套；global + 可选 context）
    THn, THc = [], []
    byn, byc = defaultdict(list), defaultdict(list)
    ctx_counts = Counter()
    for d in D.values():
        thn = fit_affine(d["pred"], d["label"], 1e-6, np.array([1., 0., 1., 0.]))
        thc = fit_affine(d["pc"], d["lc"], 1e-6, np.array([1., 0., 1., 0.]))
        vn = [thn[0][0], thn[0][1], thn[1][0], thn[1][1]]
        vc = [thc[0][0], thc[0][1], thc[1][0], thc[1][1]]
        THn.append(vn)
        THc.append(vc)
        if a.prior_context == "orientation":
            key = d["ori"]
        elif a.prior_context == "device":
            key = d["device"]
        else:
            key = "global"
        ctx_counts[key] += 1
        byn[key].append(vn)
        byc[key].append(vc)
    THn, THc = np.array(THn), np.array(THc)
    mu_norm, mu_cm = THn.mean(0), THc.mean(0)
    mu_norm_ctx = {k: np.mean(v, 0) for k, v in byn.items()
                   if k != "global" and len(v) >= a.min_ctx_sessions}
    mu_cm_ctx = {k: np.mean(v, 0) for k, v in byc.items()
                 if k != "global" and len(v) >= a.min_ctx_sessions}
    log(f"[prior] context={a.prior_context} global_counts={dict(ctx_counts)} "
        f"usable_ctx={len(mu_norm_ctx)} min_sessions={a.min_ctx_sessions}")
    if a.prior_context != "global":
        for k in sorted(ctx_counts):
            if k in mu_norm_ctx:
                log(f"[prior] ctx={k} n={ctx_counts[k]} "
                    f"norm_mu={np.round(mu_norm_ctx[k], 4).tolist()} "
                    f"cm_mu={np.round(mu_cm_ctx[k], 4).tolist()}")
            else:
                log(f"[prior] ctx={k} n={ctx_counts[k]} -> fallback global")
    log(f"[prior] norm mu={np.round(mu_norm, 5).tolist()} "
        f"sd={np.round(THn.std(0), 5).tolist()}")
    log(f"[prior] cm   mu={np.round(mu_cm, 5).tolist()} "
        f"sd={np.round(THc.std(0), 5).tolist()}")

    # 特征
    feats = [x.strip() for x in a.features.split(",") if x.strip()]
    sources = []
    if "dino" in feats:
        if not a.dino_npz:
            raise SystemExit("features 含 dino 但 --dino-npz 为空")
        sources.append(load_feature_source(a.dino_npz, "dino"))
    if "landmark" in feats:
        if not a.landmark_npz:
            raise SystemExit("features 含 landmark 但 --landmark-npz 为空")
        sources.append(load_feature_source(a.landmark_npz, "lm"))
    if "pose" in feats:
        if not a.pose_npz:
            raise SystemExit("features 含 pose 但 --pose-npz 为空")
        sources.append(load_feature_source(a.pose_npz, "pose"))
    unknown = set(feats) - {"dino", "landmark", "pose"}
    if unknown:
        raise SystemExit(f"未知特征组：{unknown}")
    if not sources:
        raise SystemExit("没有特征源")
    F, V, fnames = build_feature_matrix(sources, len(table))
    log(f"[feat] dims={len(fnames)} names={fnames[:8]}"
        f"{'...' if len(fnames) > 8 else ''}")

    sessions = list(T.items())
    if a.min_session_frames:
        before = len(sessions)
        sessions = [(s, d) for s, d in sessions
                    if len(d["pred"]) >= a.min_session_frames]
        log(f"[filter] min_session_frames={a.min_session_frames}: "
            f"{before}->{len(sessions)} sessions")
    if a.limit_sessions:
        sessions = sessions[:a.limit_sessions]
        log(f"[limit] 只跑前 {len(sessions)} 个会话（冒烟）")
    if not sessions:
        raise SystemExit("没有可评估会话")

    rows = []
    detail = []
    skipped = Counter()
    t_loop = time.time()
    for si, (s, d) in enumerate(sessions, 1):
        n_full = len(d["pred"])
        seed_rows = []
        for seed in range(a.seeds):
            if n_full - a.K < 20:
                skipped["n-K<20"] += 1
                continue
            sel = coverage_select(d["pred"], a.K, seed)
            q = np.setdiff1d(np.arange(n_full), sel)
            if len(q) < 20:
                skipped["q<20"] += 1
                continue

            if a.prior_context == "orientation":
                mn = mu_norm_ctx.get(d["ori"], mu_norm)
                mc = mu_cm_ctx.get(d["ori"], mu_cm)
            elif a.prior_context == "device":
                mn = mu_norm_ctx.get(d["device"], mu_norm)
                mc = mu_cm_ctx.get(d["device"], mu_cm)
            else:
                mn, mc = mu_norm, mu_cm
            if a.base_space == "norm":
                th = fit_huber(d["pred"][sel], d["label"][sel], a.lam, mn,
                               a.delta, scale_mode=a.huber_scale)
                q0_norm = apply_affine(d["pred"][q], th)
                q0_norm_sel = apply_affine(d["pred"][sel], th)
                q0_cm = to_cm(q0_norm, d["scm"][q], d["phi"][q])
                q0_cm_sel = to_cm(q0_norm_sel, d["scm"][sel], d["phi"][sel])
            else:
                th = fit_huber(d["pc"][sel], d["lc"][sel], a.lam, mc,
                               a.delta, scale_mode=a.huber_scale)
                q0_cm = apply_affine(d["pc"][q], th)
                q0_cm_sel = apply_affine(d["pc"][sel], th)
                q0_norm = to_norm(q0_cm, d["scm"][q], d["phi"][q])
                q0_norm_sel = to_norm(q0_cm_sel, d["scm"][sel], d["phi"][sel])

            raw_cm = float(np.linalg.norm(d["pc"][q] - d["lc"][q], axis=1).mean())
            raw_norm = float(np.linalg.norm(d["pred"][q] - d["label"][q], axis=1).mean())
            base_cm = float(np.linalg.norm(q0_cm - d["lc"][q], axis=1).mean())
            base_norm = float(np.linalg.norm(q0_norm - d["label"][q], axis=1).mean())

            if a.no_residual:
                corr = np.zeros((len(q), 2), np.float64)
                info = dict(lam_x=float("nan"), lam_y=float("nan"),
                            alpha_x=0.0, alpha_y=0.0,
                            mse0_x=float("nan"), mse0_y=float("nan"),
                            mse1_x=float("nan"), mse1_y=float("nan"),
                            clip_x=float("nan"), clip_y=float("nan"),
                            clip_frac_x=0.0, clip_frac_y=0.0,
                            feat_valid=0, note="no_residual")
            else:
                Xs = F[d["rows"][sel]]
                Xq = F[d["rows"][q]]
                Vs = V[d["rows"][sel]]
                Vq = V[d["rows"][q]]
                prep = prepare_features(Xs, Xq, Vs, Vq)
                if prep is None:
                    corr = np.zeros((len(q), 2), np.float64)
                    info = dict(lam_x=float("nan"), lam_y=float("nan"),
                                alpha_x=0.0, alpha_y=0.0,
                                mse0_x=float("nan"), mse0_y=float("nan"),
                                mse1_x=float("nan"), mse1_y=float("nan"),
                                clip_x=float("nan"), clip_y=float("nan"),
                                clip_frac_x=0.0, clip_frac_y=0.0,
                                feat_valid=int(Vs.sum()), note="prep_none")
                else:
                    Zs, Zq, fmu, fsd, ok_s = prep
                    if a.regions != "none":
                        # 正确列顺序：features + regions + intercept。
                        # 旧 new_input_calib 在这里把 regions 放在 intercept 之后，
                        # 训练/测试列顺序不一致，region 实验不可信。
                        Rs_feat = region_feats(d["pred"][sel], a.regions)
                        Rq_feat = region_feats(d["pred"][q], a.regions)
                        Zs = np.concatenate(
                            [Zs[:, :-1], Rs_feat, np.ones((len(Zs), 1))], 1)
                        Zq = np.concatenate(
                            [Zq[:, :-1], Rq_feat, np.ones((len(Zq), 1))], 1)
                    if a.resid_space == "norm":
                        Rs = d["label"][sel] - q0_norm_sel
                    else:
                        Rs = d["lc"][sel] - q0_cm_sel
                    corr, info = fit_residual(Zs, Zq, Rs, a)
                    info["feat_valid"] = ok_s
                    info["note"] = ""

            if a.resid_space == "norm":
                q1_norm = q0_norm + corr
                q1_cm = to_cm(q1_norm, d["scm"][q], d["phi"][q])
            else:
                q1_cm = q0_cm + corr
                q1_norm = to_norm(q1_cm, d["scm"][q], d["phi"][q])
            cal_cm = float(np.linalg.norm(q1_cm - d["lc"][q], axis=1).mean())
            cal_norm = float(np.linalg.norm(q1_norm - d["label"][q], axis=1).mean())

            if not all(np.isfinite([raw_cm, raw_norm, base_cm, base_norm,
                                    cal_cm, cal_norm])):
                raise SystemExit(f"非有限指标 session={s} seed={seed} "
                                 f"raw={raw_cm}/{raw_norm} base={base_cm}/{base_norm} "
                                 f"cal={cal_cm}/{cal_norm}")
            rec = dict(
                session=s, seed=seed, ori=d["ori"], device=d["device"],
                n_full=n_full, n_heldout=len(q), K=a.K,
                raw_cm=raw_cm, base_cm=base_cm, calib_cm=cal_cm,
                raw_norm=raw_norm, base_norm=base_norm, calib_norm=cal_norm,
                gain_vs_raw_cm=cal_cm / max(raw_cm, 1e-12) - 1,
                gain_vs_base_cm=cal_cm / max(base_cm, 1e-12) - 1,
                gain_vs_raw_norm=cal_norm / max(raw_norm, 1e-12) - 1,
                gain_vs_base_norm=cal_norm / max(base_norm, 1e-12) - 1,
                imu_valid_frac=float(d["imu_ok"][q].mean()),
                **{k: v for k, v in info.items() if k != "note"},
            )
            rec["note"] = info.get("note", "")
            seed_rows.append(rec)
            detail.append(rec)
            if a.log_level >= 2:
                log(f"[seed] {s} seed={seed} n={len(q)} "
                    f"raw={raw_cm:.4f} base={base_cm:.4f} cal={cal_cm:.4f} "
                    f"gain={100*rec['gain_vs_base_cm']:+.2f}% "
                    f"lam=({info['lam_x']:.3g},{info['lam_y']:.3g}) "
                    f"alpha=({info['alpha_x']:.3f},{info['alpha_y']:.3f}) "
                    f"valid={info.get('feat_valid','?')}", 2)

        if not seed_rows:
            log(f"[skip] {s} n={n_full} 全部 seed 被跳过")
            continue
        # 每会话对 seed 求平均（各 seed 的 heldout 帧数相同）
        m = {}
        for k in ("raw_cm", "base_cm", "calib_cm", "raw_norm", "base_norm",
                  "calib_norm", "gain_vs_raw_cm", "gain_vs_base_cm",
                  "gain_vs_raw_norm", "gain_vs_base_norm", "imu_valid_frac",
                  "feat_valid", "clip_frac_x", "clip_frac_y"):
            vals = [r[k] for r in seed_rows if k in r and np.isfinite(r[k])]
            m[k] = float(np.mean(vals)) if vals else float("nan")
        for k in ("lam_x", "lam_y", "alpha_x", "alpha_y"):
            vals = [r[k] for r in seed_rows if k in r and np.isfinite(r[k])]
            m[k] = float(np.mean(vals)) if vals else float("nan")
        m.update(session=s, ori=d["ori"], device=d["device"],
                 n_full=n_full, n_heldout=seed_rows[0]["n_heldout"],
                 K=a.K, n_seeds=len(seed_rows))
        rows.append(m)
        if a.log_sessions:
            log(f"[session] {si}/{len(sessions)} {s} ori={d['ori']} dev={d['device']} "
                f"n={n_full} held={m['n_heldout']} "
                f"raw={m['raw_cm']:.4f} base={m['base_cm']:.4f} "
                f"cal={m['calib_cm']:.4f} "
                f"gain_base={100*m['gain_vs_base_cm']:+.2f}% "
                f"gain_raw={100*m['gain_vs_raw_cm']:+.2f}% "
                f"lam=({m['lam_x']:.3g},{m['lam_y']:.3g}) "
                f"alpha=({m['alpha_x']:.3f},{m['alpha_y']:.3f})")

    if not rows:
        raise SystemExit("没有任何会话被评估")
    log(f"[loop] evaluated={len(rows)}/{len(sessions)} "
        f"detail_rows={len(detail)} skipped={dict(skipped)} "
        f"elapsed={time.time()-t_loop:.1f}s maxrss={maxrss_gb():.2f}GB")

    summary = summarize(rows, a)
    summary["skipped"] = dict(skipped)
    summary["runtime_s"] = time.time() - _T0
    summary["maxrss_gb"] = maxrss_gb()
    summary["feature_names"] = fnames
    summary["n_features"] = len(fnames)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["session", "ori", "device", "n_full", "n_heldout", "K", "n_seeds",
            "raw_cm", "base_cm", "calib_cm", "gain_vs_raw_cm", "gain_vs_base_cm",
            "raw_norm", "base_norm", "calib_norm", "gain_vs_raw_norm",
            "gain_vs_base_norm", "imu_valid_frac", "feat_valid",
            "lam_x", "lam_y", "alpha_x", "alpha_y",
            "mse0_x", "mse0_y", "mse1_x", "mse1_y",
            "clip_x", "clip_y", "clip_frac_x", "clip_frac_y"]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    log(f"[save] {out} rows={len(rows)}")

    if a.detail:
        dp = Path(a.detail)
        dp.parent.mkdir(parents=True, exist_ok=True)
        dcols = ["session", "seed", "ori", "device", "n_full", "n_heldout", "K",
                 "raw_cm", "base_cm", "calib_cm", "gain_vs_raw_cm",
                 "gain_vs_base_cm", "raw_norm", "base_norm", "calib_norm",
                 "gain_vs_raw_norm", "gain_vs_base_norm", "imu_valid_frac",
                 "feat_valid", "lam_x", "lam_y", "alpha_x", "alpha_y",
                 "mse0_x", "mse0_y", "mse1_x", "mse1_y",
                 "clip_x", "clip_y", "clip_frac_x", "clip_frac_y", "note"]
        with open(dp, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=dcols, extrasaction="ignore")
            w.writeheader()
            w.writerows(detail)
        log(f"[save] {dp} rows={len(detail)}")

    sp = Path(a.summary) if a.summary else out.with_suffix(".summary.json")
    sp.parent.mkdir(parents=True, exist_ok=True)
    with open(sp, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)
    log(f"[save] {sp}")

    apath = Path(a.args_out) if a.args_out else out.with_suffix(".args.json")
    with open(apath, "w", encoding="utf-8") as fh:
        json.dump(vars(a), fh, ensure_ascii=False, indent=2, default=str)
    log(f"[save] {apath}")

    log("=" * 78)
    log(f"[RESULT] sessions={summary['sessions']} heldout={summary['heldout_frames']} "
        f"raw_cm={summary['raw_cm']:.4f} base_cm={summary['base_cm']:.4f} "
        f"calib_cm={summary['calib_cm']:.4f} "
        f"({100*summary['gain_vs_raw_cm']:+.2f}% vs raw, "
        f"{100*summary['gain_vs_base_cm']:+.2f}% vs base) "
        f"raw_norm={summary['raw_norm']:.4f} base_norm={summary['base_norm']:.4f} "
        f"calib_norm={summary['calib_norm']:.4f} "
        f"({100*summary['gain_vs_raw_norm']:+.2f}% vs raw)")
    log(f"[RESULT] improved_cm={summary['improved_cm']}/{summary['sessions']} "
        f"improved_norm={summary['improved_norm']}/{summary['sessions']} "
        f"worst_gain_cm={100*summary['worst_gain_cm']:+.2f}% "
        f"median_n={summary['median_n_heldout']:.0f} "
        f"CI95_cm_vs_raw={summary['calib_minus_raw_cm_ci95']} "
        f"runtime={summary['runtime_s']:.1f}s maxrss={summary['maxrss_gb']:.2f}GB")
    for g, t in summary["ori"].items():
        log(f"[ori] {g}: sessions={t['sessions']} frames={t['frames']} "
            f"raw={t['raw_cm']:.4f} base={t['base_cm']:.4f} cal={t['calib_cm']:.4f}")
    log("[done]")


if __name__ == "__main__":
    main()
