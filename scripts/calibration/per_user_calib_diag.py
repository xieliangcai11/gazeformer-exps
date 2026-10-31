"""离线诊断：按用户校准的上限与 K 曲线。

使用当前厘米模型在测试集上的预测（已转换为归一化坐标），
在“标准物理厘米”空间做每用户仿射校准：

  y_cm = A_user p_cm + b_user

对每个用户：
  - 随机选 K 个支持帧拟合；
  - 在其余帧上评估；
  - 多种子平均。

同时给出：
  - 无校准；
  - 每用户 oracle（用全部帧拟合并在全部帧上评估，偏乐观）；
  - K = 5/10/20/50/100。

指标同时报告归一化 L2 与物理厘米误差。
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path[:0] = ["src", "."]


def log(msg):
    print(msg, flush=True)


def to_canon(p, scm, phi):
    """normalised screen -> canonical cm (screen centre, portrait axes)."""
    v = (p - 0.5) * scm
    c, s = np.cos(phi), np.sin(phi)
    return np.stack([c * v[:, 0] + s * v[:, 1],
                     -s * v[:, 0] + c * v[:, 1]], 1)


def from_canon(q, scm, phi):
    c, s = np.cos(phi), np.sin(phi)
    v = np.stack([c * q[:, 0] - s * q[:, 1],
                  s * q[:, 0] + c * q[:, 1]], 1)
    return v / scm + 0.5


def fit_ridge(Z, y, lam):
    return np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-npz", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k-list", type=int, nargs="+",
                    default=[5, 10, 20, 50, 100])
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--model", choices=["affine", "bias"], default="affine")
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    log(f"[usr] args={vars(a)}")

    z = np.load(a.test_npz, allow_pickle=True)
    pred = z["pred"].astype(np.float64)
    label = z["label"].astype(np.float64)
    paths = [str(p) for p in z["path"].tolist()]
    need = set(paths)
    meta = {}
    with open(a.index_csv, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            p = r["frame_path"]
            if p not in need:
                continue
            try:
                scm = np.array([float(r["screen_w"]) * float(r["cm_px_x"]),
                                float(r["screen_h"]) * float(r["cm_px_y"])])
                phi = float(np.arctan2(float(r["imu_x"]),
                                       -float(r["imu_y"])))
            except Exception:
                continue
            if np.hypot(float(r["imu_x"]), float(r["imu_y"])) < 0.05:
                continue
            meta[p] = dict(scm=scm, phi=phi, subject=r["subject"],
                           ori=str(r.get("orientation", "")))
    keep = [i for i, p in enumerate(paths) if p in meta]
    log(f"[usr] frames={len(paths)} joined={len(keep)}")
    p = pred[keep]; l = label[keep]
    scm = np.array([meta[paths[i]]["scm"] for i in keep])
    phi = np.array([meta[paths[i]]["phi"] for i in keep])
    subj = [meta[paths[i]]["subject"] for i in keep]
    pc = to_canon(p, scm, phi)
    lc = to_canon(l, scm, phi)
    groups = defaultdict(list)
    for i, s in enumerate(subj):
        groups[s].append(i)
    log(f"[usr] subjects={len(groups)}")

    def metrics(qc, idx):
        qn = from_canon(qc, scm[idx], phi[idx])
        cm = np.linalg.norm((qn - l[idx]) * scm[idx], axis=1)
        nm = np.linalg.norm(qn - l[idx], axis=1)
        return cm, nm

    # no calibration
    cm0, nm0 = metrics(pc, np.arange(len(pc)))
    log(f"[usr] RESULT no_cal cm={cm0.mean():.4f} norm={nm0.mean():.4f}")

    # oracle: fit all, evaluate all (optimistic)
    oracle_c = np.zeros_like(pc)
    for s, idx in groups.items():
        idx = np.array(idx)
        if len(idx) < 4:
            continue
        Z = np.column_stack([pc[idx], np.ones(len(idx))])
        w = [fit_ridge(Z, lc[idx, ax], a.lam) for ax in range(2)]
        oracle_c[idx] = np.column_stack([Z @ w[0], Z @ w[1]])
    cmo, nmo = metrics(oracle_c, np.arange(len(pc)))
    log(f"[usr] RESULT oracle_in_sample cm={cmo.mean():.4f} "
        f"norm={nmo.mean():.4f}")

    # K curve, held-out evaluation
    results = {str(k): dict(cm=[], norm=[]) for k in a.k_list}
    for k in a.k_list:
        vals_cm = []
        vals_nm = []
        for seed in range(a.seeds):
            rng = np.random.RandomState(1234 + seed)
            all_cm = []
            all_nm = []
            for s, idx in groups.items():
                idx = np.array(idx)
                if len(idx) < k + 5:
                    continue
                support = rng.choice(idx, k, replace=False)
                rest = np.setdiff1d(idx, support)
                if len(rest) == 0:
                    continue
                if a.model == "bias":
                    b = (lc[support] - pc[support]).mean(0)
                    qc = pc[rest] + b
                else:
                    Z = np.column_stack([pc[support], np.ones(k)])
                    w = [fit_ridge(Z, lc[support, ax], a.lam) for ax in range(2)]
                    Zq = np.column_stack([pc[rest], np.ones(len(rest))])
                    qc = np.column_stack([Zq @ w[0], Zq @ w[1]])
                cm, nm = metrics(qc, rest)
                all_cm.append(cm); all_nm.append(nm)
            if all_cm:
                vals_cm.append(np.concatenate(all_cm))
                vals_nm.append(np.concatenate(all_nm))
        if vals_cm:
            c = np.concatenate(vals_cm); n = np.concatenate(vals_nm)
            results[str(k)] = dict(cm=float(c.mean()), norm=float(n.mean()),
                                   n=int(len(c)))
            log(f"[usr] RESULT K={k} cm={c.mean():.4f} "
                f"norm={n.mean():.4f} eval_frames={len(c)}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".json"), "w", encoding="utf-8") as fh:
        json.dump(dict(args=vars(a), no_cal=dict(cm=float(cm0.mean()),
                                                 norm=float(nm0.mean())),
                       oracle=dict(cm=float(cmo.mean()),
                                   norm=float(nmo.mean())),
                       k=results), fh, indent=2)
    log(f"[usr] saved {out}.json")


if __name__ == "__main__":
    main()
