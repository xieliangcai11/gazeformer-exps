"""Dot-level 主动校准离线模拟器（第一版）。

目标
----
回答：如果系统主动选择注视点（dot），能否用更少的 dot 达到与当前
frame-level K=50 校准相当甚至更好的 cm 误差？

设计
----
- 每个会话（subject+orientation）把帧按 `dot_num` 聚成 dot；
- dot 的目标坐标用该 dot 所有帧的 label 均值近似（真实系统里显示点坐标已知）；
- 每个会话把 dot 固定切成 candidate / heldout 两部分：
  candidate 用于主动选点，heldout 只用于评估；两者不重叠，避免标签泄漏；
- 所有方法都评估同一批 heldout dot 的帧；
- 校准模型沿用 A9：norm base + per-axis Huber + DINO/landmark/3x3 region 残差；
- 选择策略：
  * random
  * dot coverage（3x3）
  * Bayesian D-optimal（affine 参数信息增益）
  * Bayesian D-optimal + coverage/diversity
  * frame coverage（当前 frame-level 基线，按帧选，报告唯一 dot 数）
- 资源：单进程、纯 CPU、2 个 BLAS 线程、nice 10。

日志
----
- 启动：args、输入文件、shape、dot 统计、prior、feature 统计；
- 每会话（debug）：method/K/seed 的 dots/frames、base/calib、λ/alpha、选中的 dot；
- detail CSV：每会话/method/K/seed 一行；
- summary JSON：method/K 聚合、分 orientation/device、best、runtime/maxrss。
"""
from __future__ import annotations

import os

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
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path[:0] = ["src", "."]
from gazelab.datasets.table_index import TableIndex  # noqa: E402
from scripts.calibration.cm_residual_calib import (  # noqa: E402
    build_feature_matrix,
    coverage_select,
    fit_affine,
    fit_huber,
    apply_affine,
    fit_residual,
    load_feature_source,
    load_sessions,
    prepare_features,
    region_feats,
    to_cm,
)
from scripts.calibration.base_models import (  # noqa: E402
    apply_base,
    fit_base,
)

try:
    import resource  # type: ignore
except Exception:  # pragma: no cover
    resource = None  # type: ignore


_T0 = time.time()
_LOG_LEVEL = 1


def log(msg: str, level: int = 1) -> None:
    if level > _LOG_LEVEL:
        return
    tag = {0: "WARN", 1: "INFO", 2: "DBG"}.get(level, "INFO")
    print(f"[{tag} {time.strftime('%H:%M:%S')} +{time.time() - _T0:8.1f}s] {msg}",
          flush=True)


def warn(msg: str) -> None:
    log("!! " + msg, level=0)


def subject_of(sess) -> str:
    parts = str(sess).split("/")
    return parts[2] if len(parts) > 2 else str(sess)


def maxrss_gb() -> float:
    if resource is None:
        return float("nan")
    try:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0
    except Exception:
        return float("nan")


def stable_seed(*parts) -> int:
    h = zlib.crc32("|".join(str(p) for p in parts).encode("utf-8"))
    return int(h & 0xFFFFFFFF)


# ---------------------------------------------------------------------------
# dot 表
# ---------------------------------------------------------------------------

def build_dot_table(sname: str, d: dict, table: TableIndex, F: np.ndarray,
                    V: np.ndarray, args) -> dict | None:
    """把一个会话的帧按 dot_num 聚成 dot，并固定切 candidate/heldout。"""
    groups = defaultdict(list)
    frame_dot = np.empty(len(d["pred"]), object)
    frame_dot[:] = ""
    for pos, row in enumerate(d["rows"]):
        rowd = table[int(row)]
        dn = str(rowd.get("dot_num", "") or "")
        frame_dot[pos] = dn
        if dn == "" or dn == "-1":
            continue
        groups[dn].append(pos)

    dots = []
    n_single = 0
    for dn, pos in groups.items():
        pos = np.array(sorted(pos), np.int64)
        if len(pos) < args.min_dot_frames:
            n_single += 1
            continue
        rows = d["rows"][pos]
        feat = F[rows]
        valid = V[rows]
        if valid.any():
            feat_mean = feat[valid].mean(0)
        else:
            feat_mean = np.zeros(F.shape[1], np.float64)
        dots.append(dict(
            dot=dn, pos=pos, n=len(pos),
            pred=d["pred"][pos].mean(0),
            label=d["label"][pos].mean(0),
            feat=feat_mean,
            valid=float(valid.mean()),
        ))

    if len(dots) < 4:
        log(f"[dot-skip] {sname}: dots={len(dots)} (<4)", level=2)
        return None

    rng = np.random.RandomState(stable_seed(args.split_seed, sname))
    order = rng.permutation(len(dots))
    n_cand = int(round(args.dot_split * len(dots)))
    n_cand = max(2, min(len(dots) - 2, n_cand))
    cand = [dots[i] for i in order[:n_cand]]
    held = [dots[i] for i in order[n_cand:]]
    cand_pos = np.concatenate([x["pos"] for x in cand]) if cand else np.array([], np.int64)
    held_pos = np.concatenate([x["pos"] for x in held]) if held else np.array([], np.int64)
    if len(cand) == 0 or len(held) == 0:
        log(f"[dot-skip] {sname}: cand={len(cand)} held={len(held)}", level=2)
        return None
    return dict(dots=dots, cand=cand, held=held,
                cand_pos=cand_pos, held_pos=held_pos,
                frame_dot=frame_dot, n_single=n_single)


# ---------------------------------------------------------------------------
# 选择策略
# ---------------------------------------------------------------------------

def select_random(cand, K, seed):
    rng = np.random.RandomState(seed)
    n = min(int(K), len(cand))
    idx = rng.choice(len(cand), n, replace=False)
    return [cand[i] for i in idx], []


def select_dot_coverage(cand, K, seed):
    rng = np.random.RandomState(seed)
    cells = defaultdict(list)
    for i, dot in enumerate(cand):
        cx = min(2, max(0, int(dot["pred"][0] * 3)))
        cy = min(2, max(0, int(dot["pred"][1] * 3)))
        cells[(cx, cy)].append(i)
    per = max(1, int(K) // max(1, len(cells)))
    sel = []
    for v in cells.values():
        v = list(v)
        rng.shuffle(v)
        sel += v[:per]
    rest = [i for i in range(len(cand)) if i not in set(sel)]
    if len(sel) < int(K) and rest:
        sel += list(rng.choice(rest, min(int(K) - len(sel), len(rest)), replace=False))
    sel = sorted(set(sel))[:int(K)]
    return [cand[i] for i in sel], []


def _phi(p):
    return np.array([[p[0], 1.0, 0.0, 0.0],
                     [0.0, 0.0, p[1], 1.0]], np.float64)


def _cell(p):
    return (min(2, max(0, int(p[0] * 3))), min(2, max(0, int(p[1] * 3))))


def select_dopt(cand, K, seed, Lambda0, sigma2, cov_weight):
    """贪心 Bayesian D-optimal；可选 coverage/diversity 惩罚。"""
    n = min(int(K), len(cand))
    if n >= len(cand):
        return list(cand), []
    selected = []
    Lambda = Lambda0.copy()
    covered = set()
    utils = []
    while len(selected) < n:
        best = None
        best_i = None
        best_M = None
        for i, dot in enumerate(cand):
            if any(dot is x for x in selected):
                continue
            p = dot["pred"]
            Phi = _phi(p)
            M = (Phi.T @ Phi) / max(sigma2, 1e-12)
            try:
                gain = 0.5 * (np.linalg.slogdet(Lambda + M)[1] -
                              np.linalg.slogdet(Lambda)[1])
            except np.linalg.LinAlgError:
                continue
            cov = 0.0
            if cov_weight > 0:
                if selected:
                    dists = [float(np.linalg.norm(p - x["pred"])) for x in selected]
                    cov += min(dists)
                c = _cell(p)
                if c not in covered:
                    cov += 0.25
            u = float(gain + cov_weight * cov)
            if best is None or u > best:
                best, best_i, best_M = u, i, M
        if best_i is None:
            break
        dot = cand[best_i]
        selected.append(dot)
        covered.add(_cell(dot["pred"]))
        Lambda = Lambda + best_M
        utils.append(best)
    return selected, utils


def _featdiv_complete(cand, first, K, cov_weight):
    """在已有 first 的基础上，按特征空间 max-min 距离补点。"""
    n = len(cand)
    if len(first) >= min(K, n):
        return first[:min(K, n)], []
    X = np.stack([d["feat"] for d in cand]).astype(np.float64)
    P = np.stack([d["pred"] for d in cand]).astype(np.float64)
    mu = X.mean(0)
    sd = X.std(0) + 1e-6
    Z = np.clip((X - mu) / sd, -5.0, 5.0)
    idx_of = {id(d): i for i, d in enumerate(cand)}
    sel = list(first)
    sel_set = {idx_of[id(d)] for d in first}
    min_f = np.full(n, np.inf)
    min_p = np.full(n, np.inf)
    for j in sel_set:
        min_f = np.minimum(min_f, np.linalg.norm(Z - Z[j], axis=1))
        min_p = np.minimum(min_p, np.linalg.norm(P - P[j], axis=1))
    utils = []
    while len(sel) < min(K, n):
        u = min_f + cov_weight * min_p
        if sel_set:
            u[list(sel_set)] = -np.inf
        i = int(np.argmax(u))
        if not np.isfinite(u[i]):
            break
        sel.append(cand[i])
        sel_set.add(i)
        utils.append(float(u[i]))
        min_f = np.minimum(min_f, np.linalg.norm(Z - Z[i], axis=1))
        min_p = np.minimum(min_p, np.linalg.norm(P - P[i], axis=1))
    return sel, utils


def select_featdiv(cand, K, seed, n_affine, cov_weight):
    first, _ = select_dot_coverage(cand, min(int(K), int(n_affine)), seed)
    return _featdiv_complete(cand, first, K, cov_weight)


def select_hybrid(cand, K, seed, Lambda0, sigma2, cov_weight, n_affine):
    first, _ = select_dopt(cand, min(int(K), int(n_affine)), seed,
                           Lambda0, sigma2, cov_weight)
    return _featdiv_complete(cand, first, K, cov_weight)


def _grid_phi(n: int, margin: float = 0.05):
    xs = np.linspace(margin, 1.0 - margin, int(n))
    grid = np.array([(x, y) for x in xs for y in xs], np.float64)
    G = np.zeros((len(grid), 2, 4), np.float64)
    G[:, 0, 0] = grid[:, 0]
    G[:, 0, 1] = 1.0
    G[:, 1, 2] = grid[:, 1]
    G[:, 1, 3] = 1.0
    return G


def _trace_err(Lambda: np.ndarray, G: np.ndarray) -> float:
    Sigma = np.linalg.inv(Lambda)
    return float(np.einsum("tij,jk,tik->t", G, Sigma, G).mean())


def _post_update(Lambda, mu, dot, sigma2):
    Phi = _phi(dot["pred"])
    M = Phi.T @ Phi / max(sigma2, 1e-12)
    Lnew = Lambda + M
    Sig = np.linalg.inv(Lnew)
    mu_new = Sig @ (Lambda @ mu + Phi.T @ dot["label"] / max(sigma2, 1e-12))
    return Lnew, mu_new


def _expected_after(dot, Lambda, mu, sigma2, G, rng, n_mc):
    """Expected cm/norm error at target grid after observing candidate dot."""
    Sigma = np.linalg.inv(Lambda)
    try:
        L = np.linalg.cholesky(Sigma + 1e-10 * np.eye(4))
    except np.linalg.LinAlgError:
        return float("inf")
    theta = mu[None, :] + rng.randn(n_mc, 4) @ L.T
    Phi = _phi(dot["pred"])
    y = theta @ Phi.T + rng.randn(n_mc, 2) * np.sqrt(max(sigma2, 1e-12))
    M = Phi.T @ Phi / max(sigma2, 1e-12)
    Lnew = Lambda + M
    Sig_new = np.linalg.inv(Lnew)
    m_new = (Sig_new @ (Lambda @ mu)[:, None] +
             (Sig_new @ Phi.T) @ y.T / max(sigma2, 1e-12)).T
    pred = np.einsum("tij,nj->nti", G, m_new)
    true = np.einsum("tij,nj->nti", G, theta)
    return float(((pred - true) ** 2).sum(2).mean())


def select_expected(cand, K, seed, mu, Lambda0, sigma2, G, n_mc,
                    n_affine, cov_weight, feat_div=False, stop=False,
                    stop_tol=0.001, stop_min_dots=4):
    """Sequential Bayesian active selection; optionally feature-div after affine."""
    rng = np.random.RandomState(seed)
    sel = []
    Lambda = Lambda0.copy()
    mu_cur = mu.copy()
    utils = []
    reason = ""
    K = min(int(K), len(cand))
    while len(sel) < K:
        if feat_div and len(sel) >= n_affine:
            rest, u = _featdiv_complete(cand, sel, K, cov_weight)
            sel = rest
            utils += u
            reason = "featdiv"
            break
        err0 = _trace_err(Lambda, G)
        best = None
        best_i = None
        for i, dot in enumerate(cand):
            if any(dot is x for x in sel):
                continue
            e = _expected_after(dot, Lambda, mu_cur, sigma2, G, rng, n_mc)
            if best is None or e < best:
                best, best_i = e, i
        if best_i is None:
            break
        reduction = err0 - best
        if stop and len(sel) >= stop_min_dots and reduction < stop_tol * max(err0, 1e-12):
            reason = "tol"
            break
        dot = cand[best_i]
        sel.append(dot)
        utils.append(float(reduction))
        Lambda, mu_cur = _post_update(Lambda, mu_cur, dot, sigma2)
    return sel, utils, reason


def _standardize_dot_features(cand, regions="none"):
    X = np.stack([d["feat"] for d in cand]).astype(np.float64)
    V = np.array([d["valid"] > 0 for d in cand], bool)
    ok = V & np.isfinite(X).all(1)
    if ok.sum() < 2:
        return np.zeros_like(X), V
    mu = X[ok].mean(0)
    sd = X[ok].std(0) + 1e-6
    Z = np.clip((X - mu) / sd, -5.0, 5.0)
    Z[~np.isfinite(Z)] = 0.0
    Z[~V] = 0.0
    if regions != "none":
        P = np.stack([d["pred"] for d in cand]).astype(np.float64)
        Z = np.concatenate([Z, region_feats(P, regions)], 1)
    return Z, V


def select_closedloop(cand, K, seed, mu, Lambda0, sigma2, G, n_mc,
                      n_anchor=4, feat_weight=0.2, cov_weight=0.1):
    """两阶段闭环：覆盖锚点 -> 每步更新后验 -> 选预期误差+特征新异性最优。

    只用已选点的 label 更新后验；未选候选点不读取 label。
    """
    K = min(int(K), len(cand))
    n_anchor = min(max(2, int(n_anchor)), K)
    anchors, _ = select_dot_coverage(cand, n_anchor, seed)
    sel = list(anchors)
    rng = np.random.RandomState(seed)
    Lambda = Lambda0.copy()
    mu_cur = mu.copy()
    for dot in sel:
        Lambda, mu_cur = _post_update(Lambda, mu_cur, dot, sigma2)
    Z, V = _standardize_dot_features(cand, "none")
    idx = {id(d): i for i, d in enumerate(cand)}
    sel_idx = [idx[id(d)] for d in sel]
    utils = []
    while len(sel) < K:
        best = None
        best_i = None
        for i, dot in enumerate(cand):
            if i in set(sel_idx) or not V[i]:
                continue
            e = _expected_after(dot, Lambda, mu_cur, sigma2, G, rng, n_mc)
            fd = min(float(np.linalg.norm(Z[i] - Z[j])) for j in sel_idx)
            pd = min(float(np.linalg.norm(dot["pred"] - cand[j]["pred"]))
                     for j in sel_idx)
            u = -e + feat_weight * fd + cov_weight * pd
            if best is None or u > best:
                best, best_i = u, i
        if best_i is None:
            break
        dot = cand[best_i]
        sel.append(dot)
        sel_idx.append(best_i)
        utils.append(float(best))
        Lambda, mu_cur = _post_update(Lambda, mu_cur, dot, sigma2)
    return sel, utils


def select_closedloop_resid(cand, K, seed, mu, n_anchor=4,
                            resid_start=8, feat_weight=0.2,
                            cov_weight=0.1, lev_weight=0.2, ridge=1.0):
    """闭环：每步用已选点反馈重拟合 base+residual，选预测残差/不确定性高的点。"""
    K = min(int(K), len(cand))
    n_anchor = min(max(2, int(n_anchor)), K)
    anchors, _ = select_dot_coverage(cand, n_anchor, seed)
    sel = list(anchors)
    Z, V = _standardize_dot_features(cand, "none")
    idx = {id(d): i for i, d in enumerate(cand)}
    sel_idx = [idx[id(d)] for d in sel]
    utils = []
    while len(sel) < K:
        P = np.stack([d["pred"] for d in sel]).astype(np.float64)
        U = np.stack([d["label"] for d in sel]).astype(np.float64)
        th = fit_huber(P, U, 1.0, mu, 1.345, scale_mode="axis")
        R = U - apply_affine(P, th)
        # 残差模型：标准化特征上的 ridge；K 太少时只用多样性
        have_resid = len(sel) >= max(4, int(resid_start)) and Z.shape[1] > 0
        if have_resid:
            Zs = np.concatenate([Z[sel_idx], np.ones((len(sel_idx), 1))], 1)
            A = Zs.T @ Zs + float(ridge) * np.eye(Zs.shape[1])
            try:
                W = np.linalg.solve(A, Zs.T @ R)
                Ainv = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                have_resid = False
        best = None
        best_i = None
        for i, dot in enumerate(cand):
            if i in set(sel_idx) or not V[i]:
                continue
            z = np.concatenate([Z[i], [1.0]])
            fd = min(float(np.linalg.norm(Z[i] - Z[j])) for j in sel_idx)
            pd = min(float(np.linalg.norm(dot["pred"] - cand[j]["pred"]))
                     for j in sel_idx)
            lev = float(z @ Ainv @ z) if have_resid else 0.0
            if have_resid:
                r = z @ W
                rn = float(np.linalg.norm(r))
            else:
                rn = 0.0
            u = rn + feat_weight * fd + cov_weight * pd + lev_weight * lev
            if best is None or u > best:
                best, best_i = u, i
        if best_i is None:
            break
        dot = cand[best_i]
        sel.append(dot)
        sel_idx.append(best_i)
        utils.append(float(best))
    return sel, utils


def select_closedloop_map(cand, K, seed, mu, n_anchor=4,
                          length=0.2, resid_weight=1.0,
                          unc_weight=0.5, feat_weight=0.2,
                          cov_weight=0.1):
    """闭环误差图：用已选点残差做屏幕局部平滑，选高误差/高不确定区域。"""
    K = min(int(K), len(cand))
    n_anchor = min(max(2, int(n_anchor)), K)
    anchors, _ = select_dot_coverage(cand, n_anchor, seed)
    sel = list(anchors)
    Z, V = _standardize_dot_features(cand, "none")
    idx = {id(d): i for i, d in enumerate(cand)}
    sel_idx = [idx[id(d)] for d in sel]
    utils = []
    while len(sel) < K:
        P = np.stack([d["pred"] for d in sel]).astype(np.float64)
        U = np.stack([d["label"] for d in sel]).astype(np.float64)
        th = fit_huber(P, U, 1.0, mu, 1.345, scale_mode="axis")
        R = U - apply_affine(P, th)
        rn = np.linalg.norm(R, axis=1)
        # 稳健裁剪：避免单个坏点主导误差图
        clip = float(np.median(rn)) * 2.5 + 1e-6
        best = None
        best_i = None
        for i, dot in enumerate(cand):
            if i in set(sel_idx) or not V[i]:
                continue
            d2 = ((P - dot["pred"][None, :]) ** 2).sum(1)
            w = np.exp(-0.5 * d2 / max(float(length), 1e-6) ** 2)
            wsum = float(w.sum())
            if wsum <= 1e-12:
                continue
            local = (R * w[:, None]).sum(0) / wsum
            loc = min(float(np.linalg.norm(local)), clip)
            unc = 1.0 / (wsum + 1e-6)
            fd = min(float(np.linalg.norm(Z[i] - Z[j])) for j in sel_idx)
            pd = min(float(np.linalg.norm(dot["pred"] - cand[j]["pred"]))
                     for j in sel_idx)
            u = (resid_weight * loc + unc_weight * unc
                 + feat_weight * fd + cov_weight * pd)
            if best is None or u > best:
                best, best_i = u, i
        if best_i is None:
            break
        dot = cand[best_i]
        sel.append(dot)
        sel_idx.append(best_i)
        utils.append(float(best))
    return sel, utils


def select_closedloop_final(cand, K, seed, mu, Lambda0, sigma2, G, n_mc,
                            n_anchor=4, resid_start=8, feat_weight=0.2,
                            cov_weight=0.1, resid_weight=1.0,
                            lev_weight=0.2, ridge=1.0):
    """最终误差驱动：基础校准期望误差 + 残差预测/不确定性 + 覆盖/新异性。"""
    K = min(int(K), len(cand))
    n_anchor = min(max(2, int(n_anchor)), K)
    anchors, _ = select_dot_coverage(cand, n_anchor, seed)
    sel = list(anchors)
    rng = np.random.RandomState(seed)
    Lambda = Lambda0.copy()
    mu_cur = mu.copy()
    for dot in sel:
        Lambda, mu_cur = _post_update(Lambda, mu_cur, dot, sigma2)
    Z, V = _standardize_dot_features(cand, "none")
    idx = {id(d): i for i, d in enumerate(cand)}
    sel_idx = [idx[id(d)] for d in sel]
    utils = []
    while len(sel) < K:
        P = np.stack([d["pred"] for d in sel]).astype(np.float64)
        U = np.stack([d["label"] for d in sel]).astype(np.float64)
        th = fit_huber(P, U, 1.0, mu, 1.345, scale_mode="axis")
        R = U - apply_affine(P, th)
        have_resid = len(sel) >= max(4, int(resid_start)) and Z.shape[1] > 0
        if have_resid:
            Zs = np.concatenate([Z[sel_idx], np.ones((len(sel_idx), 1))], 1)
            A = Zs.T @ Zs + float(ridge) * np.eye(Zs.shape[1])
            try:
                W = np.linalg.solve(A, Zs.T @ R)
                Ainv = np.linalg.inv(A)
            except np.linalg.LinAlgError:
                have_resid = False
        best = None
        best_i = None
        for i, dot in enumerate(cand):
            if i in set(sel_idx) or not V[i]:
                continue
            e_base = _expected_after(dot, Lambda, mu_cur, sigma2, G, rng, n_mc)
            fd = min(float(np.linalg.norm(Z[i] - Z[j])) for j in sel_idx)
            pd = min(float(np.linalg.norm(dot["pred"] - cand[j]["pred"]))
                     for j in sel_idx)
            if have_resid:
                z = np.concatenate([Z[i], [1.0]])
                rn = float(np.linalg.norm(z @ W))
                lev = float(z @ Ainv @ z)
            else:
                rn = 0.0
                lev = 0.0
            u = (-e_base + resid_weight * (rn + lev_weight * lev)
                 + feat_weight * fd + cov_weight * pd)
            if best is None or u > best:
                best, best_i = u, i
        if best_i is None:
            break
        dot = cand[best_i]
        sel.append(dot)
        sel_idx.append(best_i)
        utils.append(float(best))
        Lambda, mu_cur = _post_update(Lambda, mu_cur, dot, sigma2)
    return sel, utils


def _map_candidate_score(cand, sel_idx, V, Z, P, U, th,
                         length=0.2, feat_weight=0.2, cov_weight=0.1):
    R = U - apply_affine(P, th)
    rn = np.linalg.norm(R, axis=1)
    clip = float(np.median(rn)) * 2.5 + 1e-6
    out = {}
    for i, dot in enumerate(cand):
        if i in set(sel_idx) or not V[i]:
            continue
        d2 = ((P - dot["pred"][None, :]) ** 2).sum(1)
        w = np.exp(-0.5 * d2 / max(float(length), 1e-6) ** 2)
        wsum = float(w.sum())
        if wsum <= 1e-12:
            continue
        local = (R * w[:, None]).sum(0) / wsum
        loc = min(float(np.linalg.norm(local)), clip)
        unc = 1.0 / (wsum + 1e-6)
        fd = min(float(np.linalg.norm(Z[i] - Z[j])) for j in sel_idx)
        pd = min(float(np.linalg.norm(dot["pred"] - cand[j]["pred"]))
                 for j in sel_idx)
        out[i] = loc + 0.5 * unc + feat_weight * fd + cov_weight * pd
    return out


def select_closedloop_pair(cand, K, seed, mu, n_anchor=4,
                           top_m=8, n_top=5, length=0.2,
                           feat_weight=0.2, cov_weight=0.1):
    """2 步 lookahead：候选点成对评估，加入后剩余误差越小越好。"""
    from itertools import combinations
    K = min(int(K), len(cand))
    n_anchor = min(max(2, int(n_anchor)), K)
    anchors, _ = select_dot_coverage(cand, n_anchor, seed)
    sel = list(anchors)
    Z, V = _standardize_dot_features(cand, "none")
    idx = {id(d): i for i, d in enumerate(cand)}
    sel_idx = [idx[id(d)] for d in sel]
    utils = []
    while len(sel) < K:
        P = np.stack([d["pred"] for d in sel]).astype(np.float64)
        U = np.stack([d["label"] for d in sel]).astype(np.float64)
        th = fit_huber(P, U, 1.0, mu, 1.345, scale_mode="axis")
        single = _map_candidate_score(cand, sel_idx, V, Z, P, U, th,
                                      length, feat_weight, cov_weight)
        if not single:
            break
        order = sorted(single, key=lambda i: single[i], reverse=True)[:int(top_m)]
        if len(sel) == K - 1:
            best_i = order[0]
            sel.append(cand[best_i]); sel_idx.append(best_i)
            utils.append(float(single[best_i])); break
        best = None
        best_pair = None
        for a, b in combinations(order, 2):
            trial_idx = sel_idx + [a, b]
            Pt = np.concatenate([P, cand[a]["pred"][None, :],
                                 cand[b]["pred"][None, :]], 0)
            Ut = np.concatenate([U, cand[a]["label"][None, :],
                                 cand[b]["label"][None, :]], 0)
            tht = fit_huber(Pt, Ut, 1.0, mu, 1.345, scale_mode="axis")
            sc = _map_candidate_score(cand, trial_idx, V, Z, Pt, Ut, tht,
                                      length, feat_weight, cov_weight)
            if not sc:
                continue
            vals = sorted(sc.values(), reverse=True)[:int(n_top)]
            score = float(np.mean(vals))
            if best is None or score > best:
                best, best_pair = score, (a, b)
        if best_pair is None:
            break
        for i in best_pair:
            sel.append(cand[i]); sel_idx.append(i)
        utils.append(float(best))
    return sel[:K], utils


def select_residual(cand, K, seed, mu, Lambda0, sigma2, G, n_affine,
                    stop=False, stop_tol=0.01, noise_frac=0.1,
                    min_support=10, regions="none", base_model="axis",
                    base_kwargs=None):
    """前 n_affine 个 dot 用 affine 期望误差；其余用残差特征的后验 D-optimal。

    残差特征在 candidate 上标准化（不看 target），目标分布用未选 candidate
    dot 的特征；每选一个 dot 用实际残差更新一次后验。support 少于
    min_support 时不允许停止，保证残差模型可拟合。
    """
    K = min(int(K), len(cand))
    n_aff = min(int(n_affine), K)
    aff, _, _ = select_expected(
        cand, n_aff, seed, mu, Lambda0, sigma2, G, 32, n_aff, 0.0,
        feat_div=False, stop=False,
    )
    sel = list(aff)
    if len(sel) >= K:
        return sel, [], "", dict(resid_tau2=float("nan"),
                                 resid_sigma2=float("nan"),
                                 resid_err0=float("nan"))

    # 用 affine 子集拟合 base，计算 dot-level 残差
    P = np.stack([d["pred"] for d in sel]).astype(np.float64)
    U = np.stack([d["label"] for d in sel]).astype(np.float64)
    if base_kwargs is None:
        base_kwargs = {}
    th = fit_base(P, U, mu, kind=base_model, **base_kwargs)
    R = np.stack([d["label"] - apply_base(d["pred"][None, :], th)[0]
                  for d in sel]).astype(np.float64)

    Z, V = _standardize_dot_features(cand, regions)
    idx = {id(d): i for i, d in enumerate(cand)}
    sel_idx = [idx[id(d)] for d in sel]
    tau2 = float(np.mean(np.var(R, axis=0)))
    tau2 = max(tau2, 1e-3)
    sig2 = max(float(noise_frac) * tau2, 1e-6)

    D = Z.shape[1]
    Lambda = np.eye(D) / tau2
    for j in sel_idx:
        if V[j]:
            Lambda = Lambda + np.outer(Z[j], Z[j]) / sig2
    Inv = np.linalg.inv(Lambda + 1e-9 * np.eye(D))
    tgt = [i for i in range(len(cand)) if i not in set(sel_idx) and V[i]]
    if len(tgt) < 2:
        tgt = [i for i in range(len(cand)) if V[i]]
    Zt = Z[tgt]
    err0 = float(np.einsum("ij,jk,ik->", Zt, Inv, Zt))
    err_init = err0
    utils = []
    reason = ""
    stop_min = max(int(min_support), n_aff + 2)
    while len(sel) < K:
        best = None
        best_i = None
        for i, dot in enumerate(cand):
            if i in set(sel_idx) or not V[i]:
                continue
            z = Z[i]
            u = Inv @ z
            denom = sig2 + float(z @ u)
            if denom <= 1e-12:
                continue
            v = Zt @ u
            red = float((v @ v) / denom)
            if best is None or red > best:
                best, best_i = red, i
        if best_i is None:
            break
        if stop and len(sel) >= stop_min and best < stop_tol * max(err_init, 1e-12):
            reason = "tol"
            break
        dot = cand[best_i]
        sel.append(dot)
        sel_idx.append(best_i)
        utils.append(float(best))
        z = Z[best_i]
        u = Inv @ z
        denom = sig2 + float(z @ u)
        Inv = Inv - np.outer(u, u) / max(denom, 1e-12)
        err0 = max(err0 - best, 0.0)
    info = dict(resid_tau2=tau2, resid_sigma2=sig2, resid_err0=err_init)
    return sel, utils, reason, info


def select_oracle(d, tab, K, args, F, V, mu_norm, RA, metric="base",
                  base_model="axis", base_kwargs=None):
    """Greedy oracle on heldout dots; 仅用于 headroom/upper-bound，不是可用方法。

    metric='base': 用 heldout base error 选点（快）。
    metric='calib': 用 heldout 最终 calib error 选点（慢，但更接近该 K 的上界）。
    """
    K = min(int(K), len(tab["cand"]))
    args_eval = SimpleNamespace(**vars(args))
    args_eval.no_residual = (metric == "base")
    sel = []
    utils = []
    for _ in range(K):
        best = None
        best_dot = None
        for dot in tab["cand"]:
            if any(dot is x for x in sel):
                continue
            trial = sel + [dot]
            rec = fit_eval(d, tab, trial, None, "oracle", len(trial), 0,
                           args_eval, F, V, mu_norm, RA,
                           base_model=base_model, base_kwargs=base_kwargs)
            if rec is None:
                continue
            score = rec["base_cm"] if metric == "base" else rec["calib_cm"]
            if best is None or score < best:
                best = score
                best_dot = dot
        if best_dot is None:
            break
        sel.append(best_dot)
        utils.append(float(best))
    return sel, utils


def select_baseaware(cand, K, seed, mu, base_model="axis",
                     base_kwargs=None, n_init=4):
    """贪心 uncertainty sampling：用当前 base 预测误差最大的候选 dot。

    只用已选 dot 的 label 拟合 base；候选 dot 的 target 是屏幕点坐标，
    在真实校准中也是已知的，不需要 heldout 标签。
    """
    K = min(int(K), len(cand))
    n_init = min(max(2, int(n_init)), K)
    init, _ = select_dot_coverage(cand, n_init, seed)
    sel = list(init)
    utils = []
    if base_kwargs is None:
        base_kwargs = {}
    while len(sel) < K:
        P = np.stack([d["pred"] for d in sel]).astype(np.float64)
        U = np.stack([d["label"] for d in sel]).astype(np.float64)
        th = fit_base(P, U, mu, kind=base_model, **base_kwargs)
        best = None
        best_dot = None
        for dot in cand:
            if any(dot is x for x in sel):
                continue
            q = apply_base(dot["pred"][None, :], th)[0]
            err = float(np.linalg.norm(dot["label"] - q))
            if best is None or err > best:
                best = err
                best_dot = dot
        if best_dot is None:
            break
        sel.append(best_dot)
        utils.append(float(best))
    return sel, utils


def select_frame_coverage(d, cand_pos, K, seed):
    if len(cand_pos) == 0:
        return np.array([], np.int64), []
    kk = min(int(K), len(cand_pos))
    idx = coverage_select(d["pred"][cand_pos], kk, seed)
    return cand_pos[np.array(idx, np.int64)], []


# ---------------------------------------------------------------------------
# 拟合 + 评估
# ---------------------------------------------------------------------------

def _dot_target_sel(selected_dots):
    pos = []
    tgt = []
    for dot in selected_dots:
        pos.append(dot["pos"])
        tgt.append(np.repeat(dot["label"][None, :], len(dot["pos"]), 0))
    if not pos:
        return np.array([], np.int64), np.zeros((0, 2), np.float64)
    return np.concatenate(pos), np.concatenate(tgt, 0)



def _low_rank_projection(Zs, Zq, rank):
    """低秩适配：在 support 特征上做 SVD，保留前 rank 个方向。"""
    r = int(rank)
    if r <= 0:
        return Zs, Zq
    Xs = Zs[:, :-1]
    Xq = Zq[:, :-1]
    mu = Xs.mean(0)
    U, S, Vt = np.linalg.svd(Xs - mu, full_matrices=False)
    r = min(r, Vt.shape[0])
    P = Vt[:r]
    Zs2 = np.concatenate([(Xs - mu) @ P.T, np.ones((len(Xs), 1))], 1)
    Zq2 = np.concatenate([(Xq - mu) @ P.T, np.ones((len(Xq), 1))], 1)
    return Zs2, Zq2

def _fit_dot_residual(d, selected_dots, held_pos, th, args, F, V, RA,
                      base_model="axis"):
    """按 dot 均值拟合残差：有效样本数=dot 数，避免同 dot 多帧被当作独立样本。"""
    rows_held = d["rows"][held_pos]
    Xs = np.stack([dot["feat"] for dot in selected_dots]).astype(np.float64)
    Ps = np.stack([dot["pred"] for dot in selected_dots]).astype(np.float64)
    Vs = np.array([dot["valid"] > 0 for dot in selected_dots], bool)
    Rs = np.empty((len(selected_dots), 2), np.float64)
    for i, dot in enumerate(selected_dots):
        q0 = apply_base(dot["pred"][None, :], th)[0]
        Rs[i] = dot["label"] - q0
    prep = prepare_features(Xs, F[rows_held], Vs, V[rows_held])
    if prep is None:
        return np.zeros((len(held_pos), 2), np.float64), dict(
            lam_x=float("nan"), lam_y=float("nan"), alpha_x=0.0, alpha_y=0.0,
            feat_valid=int(Vs.sum()), note="dot_prep_none")
    Zs, Zq, fmu, fsd, ok_s = prep
    if args.regions != "none":
        Rs_feat = region_feats(Ps, args.regions)
        Rq_feat = region_feats(d["pred"][held_pos], args.regions)
        Zs = np.concatenate([Zs[:, :-1], Rs_feat, np.ones((len(Zs), 1))], 1)
        Zq = np.concatenate([Zq[:, :-1], Rq_feat, np.ones((len(Zq), 1))], 1)
    Zs, Zq = _low_rank_projection(Zs, Zq, getattr(args, "residual_rank", 0))
    corr, info = fit_residual(Zs, Zq, Rs, RA)
    info["feat_valid"] = ok_s
    info["note"] = ""
    return corr, info


def fit_eval(d, tab, selected_dots, sel_pos_frame, method, K, seed,
             args, F, V, mu_norm, RA, base_model="axis", base_kwargs=None):
    if base_kwargs is None:
        base_kwargs = {}
    if method == "frame":
        sel_pos = np.asarray(sel_pos_frame, np.int64)
        target_sel = d["label"][sel_pos]
        n_sel_dots = len(set(tab["frame_dot"][sel_pos].tolist()))
        selected_ids = ",".join(sorted(set(tab["frame_dot"][sel_pos].tolist())))
    else:
        sel_pos, target_sel = _dot_target_sel(selected_dots)
        n_sel_dots = len(selected_dots)
        selected_ids = ",".join(x["dot"] for x in selected_dots)

    held_pos = tab["held_pos"]
    if len(sel_pos) < 2 or len(held_pos) < 2:
        return None

    # base: 可切换 axis/regional/rbf/robust；source prior 只用于 global affine
    th = fit_base(d["pred"][sel_pos], target_sel, mu_norm,
                  kind=base_model, **base_kwargs)
    q0_sel = apply_base(d["pred"][sel_pos], th)
    q0_held = apply_base(d["pred"][held_pos], th)
    base_info = th.get("kind", base_model)
    if "keep" in th:
        base_info += f":keep={th['keep']}"

    # residual
    corr = np.zeros((len(held_pos), 2), np.float64)
    info = dict(lam_x=float("nan"), lam_y=float("nan"),
                alpha_x=0.0, alpha_y=0.0, feat_valid=0, note="no_residual")
    if not args.no_residual:
        if args.residual_level == "dot" and method != "frame":
            corr, info = _fit_dot_residual(d, selected_dots, held_pos,
                                           th, args, F, V, RA,
                                           base_model=base_model)
        else:
            prep = prepare_features(F[d["rows"][sel_pos]], F[d["rows"][held_pos]],
                                    V[d["rows"][sel_pos]], V[d["rows"][held_pos]])
            if prep is None:
                info["note"] = "prep_none"
            else:
                Zs, Zq, fmu, fsd, ok_s = prep
                if args.regions != "none":
                    Rs_feat = region_feats(d["pred"][sel_pos], args.regions)
                    Rq_feat = region_feats(d["pred"][held_pos], args.regions)
                    Zs = np.concatenate(
                        [Zs[:, :-1], Rs_feat, np.ones((len(Zs), 1))], 1)
                    Zq = np.concatenate(
                        [Zq[:, :-1], Rq_feat, np.ones((len(Zq), 1))], 1)
                Rs = target_sel - q0_sel
                corr, info = fit_residual(Zs, Zq, Rs, RA)
                info["feat_valid"] = ok_s
                info["note"] = ""

    q1_norm = q0_held + corr
    q1_cm = to_cm(q1_norm, d["scm"][held_pos], d["phi"][held_pos])
    q0_cm = to_cm(q0_held, d["scm"][held_pos], d["phi"][held_pos])
    raw_cm = to_cm(d["pred"][held_pos], d["scm"][held_pos], d["phi"][held_pos])
    lc = to_cm(d["label"][held_pos], d["scm"][held_pos], d["phi"][held_pos])

    raw = float(np.linalg.norm(raw_cm - lc, axis=1).mean())
    base = float(np.linalg.norm(q0_cm - lc, axis=1).mean())
    cal = float(np.linalg.norm(q1_cm - lc, axis=1).mean())
    raw_n = float(np.linalg.norm(d["pred"][held_pos] - d["label"][held_pos], axis=1).mean())
    base_n = float(np.linalg.norm(q0_held - d["label"][held_pos], axis=1).mean())
    cal_n = float(np.linalg.norm(q1_norm - d["label"][held_pos], axis=1).mean())
    if not all(np.isfinite([raw, base, cal, raw_n, base_n, cal_n])):
        raise SystemExit(f"非有限指标 {method} K={K} seed={seed}")

    return dict(
        method=method, K=int(K), seed=int(seed),
        base_model=base_model, base_info=base_info,
        n_selected_dots=int(n_sel_dots), n_frames_used=int(len(sel_pos)),
        n_heldout_frames=int(len(held_pos)),
        raw_cm=raw, base_cm=base, calib_cm=cal,
        raw_norm=raw_n, base_norm=base_n, calib_norm=cal_n,
        gain_vs_base_cm=cal / max(base, 1e-12) - 1,
        gain_vs_raw_cm=cal / max(raw, 1e-12) - 1,
        gain_vs_base_norm=cal_n / max(base_n, 1e-12) - 1,
        selected=selected_ids,
        feat_valid=int(info.get("feat_valid", 0)),
        lam_x=float(info.get("lam_x", float("nan"))),
        lam_y=float(info.get("lam_y", float("nan"))),
        alpha_x=float(info.get("alpha_x", 0.0)),
        alpha_y=float(info.get("alpha_y", 0.0)),
        note=str(info.get("note", "")),
    )


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------

def wmean(v, w):
    v = np.asarray(v, np.float64)
    w = np.asarray(w, np.float64)
    return float((v * w).sum() / max(w.sum(), 1e-12))


def boot_ci(v, w, reps=2000, seed=12345):
    v = np.asarray(v, np.float64)
    w = np.asarray(w, np.float64)
    rng = np.random.RandomState(seed)
    out = []
    n = len(v)
    for _ in range(reps):
        idx = rng.randint(0, n, n)
        out.append(float((v[idx] * w[idx]).sum() / max(w[idx].sum(), 1e-12)))
    return np.percentile(out, [2.5, 97.5]).tolist()


def summarize(rows, args):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["K"])].append(r)
    out = {}
    for (method, K), rs in sorted(groups.items()):
        w = np.array([r["n_heldout_frames"] for r in rs], np.float64)
        d = dict(
            method=method, K=int(K), sessions=len(rs),
            heldout_frames=int(w.sum()),
            n_selected_dots_mean=float(np.mean([r["n_selected_dots"] for r in rs])),
            n_selected_dots_median=float(np.median([r["n_selected_dots"] for r in rs])),
            n_frames_used_mean=float(np.mean([r["n_frames_used"] for r in rs])),
            raw_cm=wmean([r["raw_cm"] for r in rs], w),
            base_cm=wmean([r["base_cm"] for r in rs], w),
            calib_cm=wmean([r["calib_cm"] for r in rs], w),
            raw_norm=wmean([r["raw_norm"] for r in rs], w),
            base_norm=wmean([r["base_norm"] for r in rs], w),
            calib_norm=wmean([r["calib_norm"] for r in rs], w),
            improved=int(sum(r["calib_cm"] < r["base_cm"] for r in rs)),
            worst=float(max(r["gain_vs_base_cm"] for r in rs)),
            ci95_cm=boot_ci([r["calib_cm"] for r in rs], w),
        )
        d["gain_vs_base_cm"] = d["calib_cm"] / max(d["base_cm"], 1e-12) - 1
        d["gain_vs_raw_cm"] = d["calib_cm"] / max(d["raw_cm"], 1e-12) - 1
        d["gain_vs_base_norm"] = d["calib_norm"] / max(d["base_norm"], 1e-12) - 1
        d["gain_vs_raw_norm"] = d["calib_norm"] / max(d["raw_norm"], 1e-12) - 1
        out[f"{method}@{K}"] = d
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="dot-level 主动校准离线模拟")
    ap.add_argument("--train-npz", required=True)
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--dino-npz", default="")
    ap.add_argument("--landmark-npz", default="")
    ap.add_argument("--features", default="dino,landmark")
    ap.add_argument("--regions", choices=["none", "2x2", "3x3"], default="3x3")
    ap.add_argument("--no-residual", action="store_true")
    ap.add_argument("--methods", default="random,dotcov,dopt,doptcov,featdiv,frame")
    ap.add_argument("--residual-level", choices=["frame", "dot"], default="dot",
                    help="dot 方法默认按 dot 均值拟合残差，避免同 dot 多帧被当作独立样本")
    ap.add_argument("--n-affine", type=int, default=4,
                    help="featdiv/hybrid 先用多少个点确定仿射")
    ap.add_argument("--session-split", choices=["all", "select", "eval"],
                    default="all",
                    help="按 subject 划 selection/eval；用于避免在 test 上调参")
    ap.add_argument("--session-split-seed", type=int, default=0)
    ap.add_argument("--source-split", choices=["all", "select", "eval"],
                    default="all",
                    help="按 subject 过滤 source sessions；val 开发时用 select，避免 prior 使用 eval 用户")
    ap.add_argument("--expected-grid", type=int, default=5,
                    help="expected 方法的屏幕目标网格 n×n")
    ap.add_argument("--mc-samples", type=int, default=32,
                    help="expected 方法的 posterior MC 样本数")
    ap.add_argument("--stop-tol", type=float, default=0.001,
                    help="expected_stop 的相对期望误差下降阈值")
    ap.add_argument("--stop-min-dots", type=int, default=4,
                    help="expected_stop 至少选多少个 dot")
    ap.add_argument("--resid-noise-frac", type=float, default=0.1,
                    help="residual D-optimal 的观测噪声/先验方差比")
    ap.add_argument("--resid-stop-tol", type=float, default=0.01,
                    help="resid_stop 的相对残差期望误差下降阈值")
    ap.add_argument("--resid-min-support", type=int, default=10,
                    help="resid_stop 至少保留多少个 dot 以拟合残差")
    ap.add_argument("--residual-rank", type=int, default=0,
                    help="低秩残差适配维度；0=全维 ridge")
    ap.add_argument("--K-dots", type=int, nargs="+", default=[4, 6, 8, 10, 15, 20])
    ap.add_argument("--K-frames", type=int, nargs="+", default=[20, 50])
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--dot-split", type=float, default=0.65)
    ap.add_argument("--min-dot-frames", type=int, default=2)
    ap.add_argument("--min-frames", type=int, default=40)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--dopt-sigma", type=float, default=0.02,
                    help="D-optimal 观测噪声标准差（norm 空间）")
    ap.add_argument("--base-model", choices=["axis", "regional", "rbf", "robust", "similarity", "student", "auto"],
                    default="axis")
    ap.add_argument("--prior-context", choices=["global", "orientation"],
                    default="global")
    ap.add_argument("--student-nu", type=float, default=5.0)
    ap.add_argument("--closed-anchor", type=int, default=4,
                    help="closedloop 先选多少个覆盖锚点")
    ap.add_argument("--closed-feat-weight", type=float, default=0.2,
                    help="closedloop 的特征新异性权重")
    ap.add_argument("--closed-cov-weight", type=float, default=0.1,
                    help="closedloop 的屏幕覆盖权重")
    ap.add_argument("--closed-resid-start", type=int, default=8,
                    help="闭环残差模型从第几个点开始启用")
    ap.add_argument("--closed-lev-weight", type=float, default=0.2,
                    help="闭环残差不确定性权重")
    ap.add_argument("--map-length", type=float, default=0.2,
                    help="误差图局部平滑长度（norm 坐标）")
    ap.add_argument("--map-resid-weight", type=float, default=1.0)
    ap.add_argument("--map-unc-weight", type=float, default=0.5)
    ap.add_argument("--final-resid-weight", type=float, default=1.0)
    ap.add_argument("--final-lev-weight", type=float, default=0.2)
    ap.add_argument("--pair-top", type=int, default=8)
    ap.add_argument("--pair-ntop", type=int, default=5)
    ap.add_argument("--regional-lam", type=float, default=5.0,
                    help="regional base 的 9 个区域 bias 的 ridge 强度")
    ap.add_argument("--rbf-length", type=float, default=0.25,
                    help="RBF base 的 lengthscale（norm 坐标）")
    ap.add_argument("--rbf-lam", type=float, default=1e-2,
                    help="RBF base 的 kernel ridge")
    ap.add_argument("--robust-trim", type=float, default=0.1,
                    help="robust base 去掉最差 dot 的比例")
    ap.add_argument("--cov-weight", type=float, default=0.1,
                    help="D-optimal+coverage 的覆盖/多样性权重")
    ap.add_argument("--limit-sessions", type=int, default=0)
    ap.add_argument("--log-level", type=int, default=1)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    global _LOG_LEVEL
    _LOG_LEVEL = a.log_level
    methods = [x.strip() for x in a.methods.split(",") if x.strip()]
    if not methods:
        raise SystemExit("--methods 为空")
    if a.seeds < 1:
        raise SystemExit("--seeds >=1")
    if any(m.startswith("oracle") for m in methods):
        log("!! ORACLE uses heldout labels; only for headroom/upper-bound", level=0)

    log("=" * 78)
    log("active_calib_sim")
    log(f"host={platform.node()} python={sys.version.split()[0]} "
        f"numpy={np.__version__} pid={os.getpid()}")
    log(f"args={json.dumps(vars(a), ensure_ascii=False, default=str)}")
    script = Path(__file__).resolve()
    log(f"script={script} sha256={hashlib.sha256(script.read_bytes()).hexdigest()[:16]}")
    for p in (a.train_npz, a.test_npz, a.index_csv, a.dino_npz, a.landmark_npz):
        if p:
            pp = Path(p)
            log(f"input {p}: exists={pp.exists()} "
                f"size={pp.stat().st_size if pp.exists() else -1}")

    table = TableIndex(a.index_csv)
    log(f"[index] rows={len(table)} fields={table.fieldnames}")
    D = load_sessions(a.train_npz, table, "orientation", a.min_frames)
    T = load_sessions(a.test_npz, table, "orientation", a.min_frames)
    if not D or not T:
        raise SystemExit(f"空会话 source={len(D)} test={len(T)}")
    if a.source_split != "all":
        want = 0 if a.source_split == "select" else 1
        before = len(D)
        D = {s: d for s, d in D.items()
             if (zlib.crc32((subject_of(s) + "|" + str(a.session_split_seed)).encode("utf-8")) % 2) == want}
        log(f"[source-split] {a.source_split} before={before} after={len(D)} "
            f"subjects={len({subject_of(s) for s in D})}")
    log(f"[sessions] source={len(D)} test={len(T)}")

    # source prior
    THn = []
    by_ori = defaultdict(list)
    for d in D.values():
        th = fit_affine(d["pred"], d["label"], 1e-6, np.array([1., 0., 1., 0.]))
        v = [th[0][0], th[0][1], th[1][0], th[1][1]]
        THn.append(v)
        by_ori[str(d["ori"])].append(v)
    THn = np.array(THn)
    mu_norm = THn.mean(0)
    mu_by_ori = {k: np.asarray(v).mean(0) for k, v in by_ori.items() if len(v) >= 3}
    log(f"[prior-context] {a.prior_context} groups=" + str({k: len(v) for k, v in by_ori.items()}))
    Sigma = np.cov(THn.T) + 1e-4 * np.eye(4)
    Lambda0 = np.linalg.inv(Sigma)
    log(f"[prior] mu={np.round(mu_norm, 5).tolist()} "
        f"sd={np.round(np.sqrt(np.diag(Sigma)), 5).tolist()} "
        f"sigma_dopt={a.dopt_sigma}")

    if a.base_model == "regional":
        base_kwargs = {"regional_lam": a.regional_lam}
    elif a.base_model == "rbf":
        base_kwargs = {"length": a.rbf_length, "rbf_lam": a.rbf_lam}
    elif a.base_model == "robust":
        base_kwargs = {"trim": a.robust_trim}
    elif a.base_model == "similarity":
        base_kwargs = {}
    elif a.base_model == "student":
        base_kwargs = {"nu": a.student_nu}
    elif a.base_model == "auto":
        base_kwargs = {}
    else:
        base_kwargs = {}
    log(f"[base] model={a.base_model} kwargs={base_kwargs}")

    def fit_eval_base(*fargs, **fkwargs):
        if a.prior_context == "orientation":
            fa = list(fargs)
            ori = str(fa[0]["ori"])
            fa[10] = mu_by_ori.get(ori, mu_norm)
            fargs = tuple(fa)
        return fit_eval(*fargs, base_model=a.base_model,
                        base_kwargs=base_kwargs, **fkwargs)

    # features
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
    unknown = set(feats) - {"dino", "landmark"}
    if unknown:
        raise SystemExit(f"未知特征组 {unknown}")
    F, V, fnames = build_feature_matrix(sources, len(table))
    log(f"[feat] dims={len(fnames)}")

    RA = SimpleNamespace(
        target_standardize=True, loo_lam=True,
        lams=[1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0],
        fixed_lam=1.0, shrink=False, gate=False, hard_gate=False,
        gate_min_improve=0.001, clip_mode="fixed",
        clip_norm=0.02, clip_cm=0.15, clip_q=0.95, clip_factor=2.0,
        resid_space="norm",
    )

    # dot tables
    tabs = {}
    dot_stats = []
    for s, d in T.items():
        tab = build_dot_table(s, d, table, F, V, a)
        if tab is None:
            continue
        tabs[s] = tab
        dot_stats.append((len(tab["dots"]), len(tab["cand"]), len(tab["held"])))
    if not tabs:
        raise SystemExit("没有可用 dot 表")
    log(f"[dots] sessions={len(tabs)} mean_total={np.mean([x[0] for x in dot_stats]):.1f} "
        f"mean_cand={np.mean([x[1] for x in dot_stats]):.1f} "
        f"mean_held={np.mean([x[2] for x in dot_stats]):.1f} "
        f"min_total={min(x[0] for x in dot_stats)}")

    sessions = list(tabs.items())
    if a.session_split != "all":
        want = 0 if a.session_split == "select" else 1

        def _subj(sess):
            parts = str(sess).split("/")
            return parts[2] if len(parts) > 2 else str(sess)

        before = len(sessions)
        sessions = [(s, d) for s, d in sessions
                    if (zlib.crc32(
                        (_subj(s) + "|" + str(a.session_split_seed)).encode("utf-8")
                    ) % 2) == want]
        log(f"[split] {a.session_split} before={before} after={len(sessions)} "
            f"subjects={len({_subj(s) for s, _ in sessions})}")
    if a.limit_sessions:
        sessions = sessions[:a.limit_sessions]
        log(f"[limit] {len(sessions)} sessions")

    G = _grid_phi(a.expected_grid)
    log(f"[expected] target_grid={G.shape[0]} mc={a.mc_samples} "
        f"stop_tol={a.stop_tol} stop_min={a.stop_min_dots}")

    rows = []
    t0 = time.time()
    for si, (s, tab) in enumerate(sessions, 1):
        d = T[s]
        task_lists = []
        for method in methods:
            if method in ("random", "dotcov", "dopt", "doptcov",
                          "featdiv", "hybrid", "expected", "expected_feat",
                          "expected_stop", "resid", "resid_stop",
                          "baseaware", "closedloop", "closedloop_resid",
                          "closedloop_map", "closedloop_final",
                          "closedloop_pair"):
                for K in a.K_dots:
                    for seed in range(a.seeds):
                        task_lists.append((method, K, seed))
            elif method in ("oracle_base", "oracle_final"):
                for K in a.K_dots:
                    for seed in range(a.seeds):
                        task_lists.append((method, K, seed))
            elif method == "allcand":
                for seed in range(a.seeds):
                    task_lists.append((method, -1, seed))
            elif method == "frame":
                for K in a.K_frames:
                    for seed in range(a.seeds):
                        task_lists.append((method, K, seed))
            else:
                warn(f"未知 method={method}，跳过")

        for method, K, seed in task_lists:
            utils = []
            if method == "random":
                sel_dots, utils = select_random(tab["cand"], K, seed)
                rec = fit_eval_base(d, tab, sel_dots, None, method, K, seed,
                               a, F, V, mu_norm, RA)
            elif method == "dotcov":
                sel_dots, utils = select_dot_coverage(tab["cand"], K, seed)
                rec = fit_eval_base(d, tab, sel_dots, None, method, K, seed,
                               a, F, V, mu_norm, RA)
            elif method in ("dopt", "doptcov"):
                cw = a.cov_weight if method == "doptcov" else 0.0
                sel_dots, utils = select_dopt(tab["cand"], K, seed, Lambda0,
                                              a.dopt_sigma ** 2, cw)
                rec = fit_eval_base(d, tab, sel_dots, None, method, K, seed,
                               a, F, V, mu_norm, RA)
            elif method == "featdiv":
                sel_dots, utils = select_featdiv(tab["cand"], K, seed,
                                                 a.n_affine, a.cov_weight)
                rec = fit_eval_base(d, tab, sel_dots, None, method, K, seed,
                               a, F, V, mu_norm, RA)
            elif method == "hybrid":
                sel_dots, utils = select_hybrid(tab["cand"], K, seed, Lambda0,
                                                a.dopt_sigma ** 2,
                                                a.cov_weight, a.n_affine)
                rec = fit_eval_base(d, tab, sel_dots, None, method, K, seed,
                               a, F, V, mu_norm, RA)
            elif method == "closedloop":
                sel_dots, utils = select_closedloop(
                    tab["cand"], K, seed, mu_norm, Lambda0,
                    a.dopt_sigma ** 2, G, a.mc_samples,
                    n_anchor=a.closed_anchor,
                    feat_weight=a.closed_feat_weight,
                    cov_weight=a.closed_cov_weight)
                rec = fit_eval_base(d, tab, sel_dots, None, method, K, seed,
                               a, F, V, mu_norm, RA)
                if rec is not None:
                    rec["stop_reason"] = "closed_loop"
            elif method == "closedloop_resid":
                sel_dots, utils = select_closedloop_resid(
                    tab["cand"], K, seed, mu_norm,
                    n_anchor=a.closed_anchor,
                    resid_start=a.closed_resid_start,
                    feat_weight=a.closed_feat_weight,
                    cov_weight=a.closed_cov_weight,
                    lev_weight=a.closed_lev_weight)
                rec = fit_eval_base(d, tab, sel_dots, None, method, K, seed,
                               a, F, V, mu_norm, RA)
                if rec is not None:
                    rec["stop_reason"] = "closed_loop_resid"
            elif method == "closedloop_map":
                sel_dots, utils = select_closedloop_map(
                    tab["cand"], K, seed, mu_norm,
                    n_anchor=a.closed_anchor,
                    length=a.map_length,
                    resid_weight=a.map_resid_weight,
                    unc_weight=a.map_unc_weight,
                    feat_weight=a.closed_feat_weight,
                    cov_weight=a.closed_cov_weight)
                rec = fit_eval_base(d, tab, sel_dots, None, method, K, seed,
                               a, F, V, mu_norm, RA)
                if rec is not None:
                    rec["stop_reason"] = "closed_loop_map"
            elif method == "closedloop_final":
                sel_dots, utils = select_closedloop_final(
                    tab["cand"], K, seed, mu_norm, Lambda0,
                    a.dopt_sigma ** 2, G, a.mc_samples,
                    n_anchor=a.closed_anchor,
                    resid_start=a.closed_resid_start,
                    feat_weight=a.closed_feat_weight,
                    cov_weight=a.closed_cov_weight,
                    resid_weight=a.final_resid_weight,
                    lev_weight=a.final_lev_weight)
                rec = fit_eval_base(d, tab, sel_dots, None, method, K, seed,
                               a, F, V, mu_norm, RA)
                if rec is not None:
                    rec["stop_reason"] = "closed_loop_final"
            elif method == "closedloop_pair":
                sel_dots, utils = select_closedloop_pair(
                    tab["cand"], K, seed, mu_norm,
                    n_anchor=a.closed_anchor,
                    top_m=a.pair_top,
                    n_top=a.pair_ntop,
                    length=a.map_length,
                    feat_weight=a.closed_feat_weight,
                    cov_weight=a.closed_cov_weight)
                rec = fit_eval_base(d, tab, sel_dots, None, method, K, seed,
                               a, F, V, mu_norm, RA)
                if rec is not None:
                    rec["stop_reason"] = "closed_loop_pair"
            elif method in ("expected", "expected_feat", "expected_stop"):
                sel_dots, utils, reason = select_expected(
                    tab["cand"], K, seed, mu_norm, Lambda0,
                    a.dopt_sigma ** 2, G, a.mc_samples, a.n_affine,
                    a.cov_weight,
                    feat_div=(method == "expected_feat"),
                    stop=(method == "expected_stop"),
                    stop_tol=a.stop_tol,
                    stop_min_dots=a.stop_min_dots,
                )
                rec = fit_eval_base(d, tab, sel_dots, None, method, K, seed,
                               a, F, V, mu_norm, RA)
                if rec is not None:
                    rec["stop_reason"] = reason
            elif method in ("resid", "resid_stop"):
                sel_dots, utils, reason, rinfo = select_residual(
                    tab["cand"], K, seed, mu_norm, Lambda0,
                    a.dopt_sigma ** 2, G, a.n_affine,
                    stop=(method == "resid_stop"),
                    stop_tol=a.resid_stop_tol,
                    noise_frac=a.resid_noise_frac,
                    min_support=a.resid_min_support,
                    regions=a.regions,
                    base_model=a.base_model,
                    base_kwargs=base_kwargs,
                )
                rec = fit_eval_base(d, tab, sel_dots, None, method, K, seed,
                               a, F, V, mu_norm, RA)
                if rec is not None:
                    rec["stop_reason"] = reason
                    rec.update(rinfo)
            elif method == "baseaware":
                sel_dots, utils = select_baseaware(
                    tab["cand"], K, seed, mu_norm,
                    base_model=a.base_model, base_kwargs=base_kwargs,
                    n_init=a.n_affine)
                rec = fit_eval_base(d, tab, sel_dots, None, method, K, seed,
                               a, F, V, mu_norm, RA)
                if rec is not None:
                    rec["stop_reason"] = "base_uncertainty"
            elif method in ("oracle_base", "oracle_final"):
                metric = "base" if method == "oracle_base" else "calib"
                sel_dots, utils = select_oracle(
                    d, tab, K, a, F, V, mu_norm, RA, metric=metric,
                    base_model=a.base_model, base_kwargs=base_kwargs)
                rec = fit_eval_base(d, tab, sel_dots, None, method,
                               K, seed, a, F, V, mu_norm, RA)
                if rec is not None:
                    rec["oracle_metric"] = metric
                    rec["stop_reason"] = "oracle_%s" % metric
            elif method == "allcand":
                sel_dots = list(tab["cand"])
                rec = fit_eval_base(d, tab, sel_dots, None, method,
                               -1, seed, a, F, V, mu_norm, RA)
                if rec is not None:
                    rec["stop_reason"] = "all_candidate"
            else:  # frame
                sel_pos, utils = select_frame_coverage(d, tab["cand_pos"], K, seed)
                rec = fit_eval_base(d, tab, None, sel_pos, method, K, seed,
                               a, F, V, mu_norm, RA)
            if rec is None:
                warn(f"[skip] {s} {method} K={K} seed={seed}: 样本不足")
                continue
            rec.update(session=s, ori=d["ori"], device=d["device"],
                       n_candidate_dots=len(tab["cand"]),
                       n_heldout_dots=len(tab["held"]),
                       stop_reason=rec.get("stop_reason", ""),
                       utility=";".join(f"{u:.5f}" for u in utils) if utils else "")
            rows.append(rec)
            if a.log_level >= 2:
                log(f"[rec] {s} {method} K={K} seed={seed} "
                    f"dots={rec['n_selected_dots']} frames={rec['n_frames_used']} "
                    f"base={rec['base_cm']:.4f} cal={rec['calib_cm']:.4f} "
                    f"gain={100*rec['gain_vs_base_cm']:+.2f}% "
                    f"feat={rec['feat_valid']} note={rec['note']}", 2)
        if si % 25 == 0 or si == len(sessions):
            log(f"[progress] {si}/{len(sessions)} sessions, "
                f"rows={len(rows)}, elapsed={time.time()-t0:.1f}s")

    if not rows:
        raise SystemExit("没有结果")
    log(f"[loop] rows={len(rows)} sessions={len(sessions)} "
        f"elapsed={time.time()-t0:.1f}s maxrss={maxrss_gb():.2f}GB")

    summary = summarize(rows, a)
    summary.update(
        args=vars(a),
        runtime_s=time.time() - _T0,
        maxrss_gb=maxrss_gb(),
        n_features=len(fnames),
        feature_names=fnames,
        dot_stats=dict(
            sessions=len(tabs),
            mean_total=float(np.mean([x[0] for x in dot_stats])),
            mean_cand=float(np.mean([x[1] for x in dot_stats])),
            mean_held=float(np.mean([x[2] for x in dot_stats])),
        ),
    )

    outdir = Path(a.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    detail = outdir / "active_calib_detail.csv"
    cols = ["session", "ori", "device", "method", "K", "seed",
            "base_model", "base_info",
            "n_candidate_dots", "n_heldout_dots", "n_selected_dots",
            "n_frames_used", "n_heldout_frames",
            "raw_cm", "base_cm", "calib_cm",
            "gain_vs_base_cm", "gain_vs_raw_cm",
            "raw_norm", "base_norm", "calib_norm",
            "gain_vs_base_norm", "feat_valid", "note",
            "lam_x", "lam_y", "alpha_x", "alpha_y", "selected", "utility",
            "stop_reason", "resid_tau2", "resid_sigma2", "resid_err0",
            "oracle_metric"]
    with open(detail, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    log(f"[save] {detail} rows={len(rows)}")

    sp = outdir / "active_calib_summary.json"
    with open(sp, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, default=str)
    log(f"[save] {sp}")

    apath = outdir / "active_calib_args.json"
    with open(apath, "w", encoding="utf-8") as fh:
        json.dump(vars(a), fh, ensure_ascii=False, indent=2, default=str)
    log(f"[save] {apath}")

    log("=" * 78)
    for key, d in sorted(summary.items()):
        if not isinstance(d, dict) or "calib_cm" not in d:
            continue
        log(f"[RESULT] {key}: dots={d['n_selected_dots_mean']:.1f} "
            f"frames={d['n_frames_used_mean']:.0f} "
            f"base={d['base_cm']:.4f} cal={d['calib_cm']:.4f} "
            f"({100*d['gain_vs_base_cm']:+.2f}% vs base) "
            f"raw={d['raw_cm']:.4f} improved={d['improved']}/{d['sessions']} "
            f"worst={100*d['worst']:+.2f}%")
    log("[done]")


if __name__ == "__main__":
    main()
