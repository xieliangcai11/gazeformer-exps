"""可切换的 per-session base 模型。

axis     : 4 参数 affine（当前基线）
regional : 全局 affine + 3x3 区域 bias（层次 ridge）
rbf      : 全局 affine + screen 坐标 RBF 残差
robust   : axis + 去掉最差 trim 比例后重拟合
"""
from __future__ import annotations

import numpy as np

from scripts.calibration.cm_residual_calib import apply_affine, fit_huber


def _region_onehot(P: np.ndarray, n: int = 3) -> np.ndarray:
    P = np.asarray(P, np.float64)
    rx = np.clip((P[:, 0] * n).astype(int), 0, n - 1)
    ry = np.clip((P[:, 1] * n).astype(int), 0, n - 1)
    idx = ry * n + rx
    oh = np.zeros((len(P), n * n), np.float64)
    oh[np.arange(len(P)), idx] = 1.0
    return oh


def fit_axis(P, U, mu, lam=1.0, **kwargs):
    th = fit_huber(P, U, lam, mu, 1.345, scale_mode="axis")
    return {"kind": "axis", "th": th}


def apply_axis(P, model):
    return apply_affine(P, model["th"])


def fit_regional(P, U, mu, lam=1.0, regional_lam=5.0,
                 delta=1.345, iters=6):
    P = np.asarray(P, np.float64)
    U = np.asarray(U, np.float64)
    k = 2 + 9
    oh = _region_onehot(P, 3)
    out = []
    for ax in range(2):
        X = np.concatenate([P[:, ax:ax + 1], np.ones((len(P), 1)), oh], 1)
        mu_ax = np.concatenate([[mu[ax * 2], mu[ax * 2 + 1]],
                                np.zeros(9)])
        prec = np.diag(np.concatenate([
            [float(lam), float(lam)],
            np.full(9, float(regional_lam)),
        ]))
        A = X.T @ X + prec
        b = X.T @ U[:, ax] + prec @ mu_ax
        th = np.linalg.solve(A, b)
        for _ in range(int(iters)):
            r = U[:, ax] - X @ th
            scale = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
            z = np.abs(r) / (delta * scale)
            w = np.where(z <= 1.0, 1.0, 1.0 / np.maximum(z, 1e-12))
            A = X.T @ (w[:, None] * X) + prec
            b = X.T @ (w * U[:, ax]) + prec @ mu_ax
            th = np.linalg.solve(A, b)
        out.append(th)
    return {"kind": "regional", "th": out}


def apply_regional(P, model):
    P = np.asarray(P, np.float64)
    oh = _region_onehot(P, 3)
    cols = []
    for ax in range(2):
        X = np.concatenate([P[:, ax:ax + 1], np.ones((len(P), 1)), oh], 1)
        cols.append(X @ model["th"][ax])
    return np.stack(cols, 1)


def _rbf(X, Y, length):
    d2 = ((X[:, None, :] - Y[None, :, :]) ** 2).sum(2)
    return np.exp(-0.5 * d2 / max(length * length, 1e-9))


def fit_rbf(P, U, mu, lam=1.0, length=0.25, rbf_lam=1e-2):
    P = np.asarray(P, np.float64)
    U = np.asarray(U, np.float64)
    affine = fit_huber(P, U, lam, mu, 1.345, scale_mode="axis")
    q = apply_affine(P, affine)
    R = U - q
    K = _rbf(P, P, length)
    A = K + float(rbf_lam) * np.eye(len(P)) + 1e-9 * np.eye(len(P))
    alpha = np.linalg.solve(A, R)
    return {"kind": "rbf", "th": affine, "X": P, "alpha": alpha,
            "length": float(length)}


def apply_rbf(P, model):
    P = np.asarray(P, np.float64)
    q = apply_affine(P, model["th"])
    K = _rbf(P, model["X"], model["length"])
    return q + K @ model["alpha"]


def fit_robust(P, U, mu, lam=1.0, trim=0.1):
    P = np.asarray(P, np.float64)
    U = np.asarray(U, np.float64)
    th = fit_huber(P, U, lam, mu, 1.345, scale_mode="axis")
    q = apply_affine(P, th)
    err = np.linalg.norm(U - q, axis=1)
    keep = err <= np.quantile(err, max(0.0, 1.0 - float(trim)))
    if keep.sum() >= max(4, int(0.5 * len(P))):
        th = fit_huber(P[keep], U[keep], lam, mu, 1.345, scale_mode="axis")
    return {"kind": "robust", "th": th, "trim": float(trim),
            "keep": int(keep.sum())}


def apply_robust(P, model):
    return apply_affine(P, model["th"])


def fit_similarity(P, U, mu, lam=1.0, delta=1.345, iters=6):
    """统一缩放 + 旋转 + 平移（4 参数，线性形式）。

    x' = a*x - b*y + tx
    y' = b*x + a*y + ty
    """
    P = np.asarray(P, np.float64)
    U = np.asarray(U, np.float64)
    x = P[:, 0]
    y = P[:, 1]
    Xx = np.stack([x, -y, np.ones(len(P)), np.zeros(len(P))], 1)
    Xy = np.stack([y, x, np.zeros(len(P)), np.ones(len(P))], 1)
    X = np.concatenate([Xx, Xy], 0)
    Y = np.concatenate([U[:, 0], U[:, 1]], 0)
    mu4 = np.array([1.0, 0.0, 0.0, 0.0], np.float64)
    prec = float(lam) * np.eye(4)
    A = X.T @ X + prec
    b = X.T @ Y + prec @ mu4
    th = np.linalg.solve(A, b)
    for _ in range(int(iters)):
        r = Y - X @ th
        rx, ry = r[:len(P)], r[len(P):]
        sx = 1.4826 * np.median(np.abs(rx - np.median(rx))) + 1e-9
        sy = 1.4826 * np.median(np.abs(ry - np.median(ry))) + 1e-9
        wx = np.where(np.abs(rx) / (delta * sx) <= 1.0, 1.0,
                      1.0 / np.maximum(np.abs(rx) / (delta * sx), 1e-12))
        wy = np.where(np.abs(ry) / (delta * sy) <= 1.0, 1.0,
                      1.0 / np.maximum(np.abs(ry) / (delta * sy), 1e-12))
        w = np.concatenate([wx, wy])
        A = X.T @ (w[:, None] * X) + prec
        b = X.T @ (w * Y) + prec @ mu4
        th = np.linalg.solve(A, b)
    return {"kind": "similarity", "th": th}


def apply_similarity(P, model):
    P = np.asarray(P, np.float64)
    a, b, tx, ty = model["th"]
    return np.stack([a * P[:, 0] - b * P[:, 1] + tx,
                     b * P[:, 0] + a * P[:, 1] + ty], 1)



def fit_student(P, U, mu, lam=1.0, nu=5.0, iters=8):
    """Student-t 稳健 per-axis affine（4 参数）。"""
    P = np.asarray(P, np.float64)
    U = np.asarray(U, np.float64)
    out = []
    for ax in range(2):
        X = np.stack([P[:, ax], np.ones(len(P))], 1)
        A = X.T @ X + float(lam) * np.eye(2)
        b = X.T @ U[:, ax] + float(lam) * mu[ax * 2:ax * 2 + 2]
        th = np.linalg.solve(A, b)
        for _ in range(int(iters)):
            r = U[:, ax] - X @ th
            scale = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
            z = r / scale
            w = (float(nu) + 1.0) / (float(nu) + z * z)
            A = X.T @ (w[:, None] * X) + float(lam) * np.eye(2)
            b = X.T @ (w * U[:, ax]) + float(lam) * mu[ax * 2:ax * 2 + 2]
            th = np.linalg.solve(A, b)
        out.append(th)
    return {"kind": "student", "th": out, "nu": float(nu)}


def apply_student(P, model):
    return apply_affine(P, model["th"])


def fit_auto(P, U, mu, kinds=("axis", "similarity", "student")):
    """用 support 的 leave-one-out 误差选择校准器。"""
    P = np.asarray(P, np.float64)
    U = np.asarray(U, np.float64)
    best = None
    best_kind = None
    for kind in kinds:
        errs = []
        for i in range(len(P)):
            keep = np.ones(len(P), bool)
            keep[i] = False
            if keep.sum() < 4:
                continue
            try:
                th = fit_base(P[keep], U[keep], mu, kind=kind)
                q = apply_base(P[i:i + 1], th)[0]
                errs.append(float(np.linalg.norm(U[i] - q)))
            except Exception:
                errs = []
                break
        if not errs:
            continue
        e = float(np.mean(errs))
        if best is None or e < best:
            best, best_kind = e, kind
    if best_kind is None:
        best_kind = "axis"
    model = fit_base(P, U, mu, kind=best_kind)
    model["selected_kind"] = best_kind
    model["loo_error"] = best
    return model

def fit_base(P, U, mu, kind="axis", **kwargs):
    if kind == "auto":
        return fit_auto(P, U, mu, **kwargs)
    if kind == "axis":
        return fit_axis(P, U, mu, **kwargs)
    if kind == "regional":
        return fit_regional(P, U, mu, **kwargs)
    if kind == "rbf":
        return fit_rbf(P, U, mu, **kwargs)
    if kind == "robust":
        return fit_robust(P, U, mu, **kwargs)
    if kind == "similarity":
        return fit_similarity(P, U, mu, **kwargs)
    if kind == "student":
        return fit_student(P, U, mu, **kwargs)
    raise ValueError(f"未知 base kind={kind}")


def apply_base(P, model):
    kind = model.get("kind", "axis")
    if kind == "axis":
        return apply_axis(P, model)
    if kind == "regional":
        return apply_regional(P, model)
    if kind == "rbf":
        return apply_rbf(P, model)
    if kind == "robust":
        return apply_robust(P, model)
    if kind == "similarity":
        return apply_similarity(P, model)
    if kind == "student":
        return apply_student(P, model)
    raise ValueError(f"未知 base kind={kind}")
