"""诊断：按“注视点(dot)”校准 vs 按“帧”校准。

同一注视点通常有 6–7 帧。真实标定中，用户看 K 个点，而不是提供 K 个独立帧。
本脚本比较：
  A. 随机选 K 帧拟合仿射；
  B. 选 K 个不同 dot，用每个 dot 的多帧平均拟合仿射；
在剩余帧上评估，指标为归一化 L2 和物理厘米。
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
    ap.add_argument("--k-list", type=int, nargs="+", default=[10, 20, 50])
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    log(f"[dot] args={vars(a)}")

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
                dot = str(r.get("dot_num", ""))
            except Exception:
                continue
            if dot == "":
                continue
            meta[p] = dict(scm=scm, phi=phi, dot=dot,
                           session=p.rsplit("/", 1)[0],
                           subject=r["subject"],
                           ori=str(r.get("orientation", "")))
    keep = [i for i, p in enumerate(paths) if p in meta]
    log(f"[dot] frames={len(paths)} joined={len(keep)}")
    p = pred[keep]; l = label[keep]
    scm = np.array([meta[paths[i]]["scm"] for i in keep])
    phi = np.array([meta[paths[i]]["phi"] for i in keep])
    pc = to_canon(p, scm, phi)
    lc = to_canon(l, scm, phi)

    sess = defaultdict(lambda: defaultdict(list))
    for j, i in enumerate(keep):
        m = meta[paths[i]]
        sess[m["session"]][m["dot"]].append(j)
    log(f"[dot] sessions={len(sess)} dots={sum(len(v) for v in sess.values())}")

    def metrics(qc, idx):
        qn = from_canon(qc, scm[idx], phi[idx])
        cm = np.linalg.norm((qn - l[idx]) * scm[idx], axis=1)
        nm = np.linalg.norm(qn - l[idx], axis=1)
        return cm, nm

    cm0, nm0 = metrics(pc, np.arange(len(pc)))
    log(f"[dot] RESULT no_cal cm={cm0.mean():.4f} norm={nm0.mean():.4f}")

    out = {}
    for k in a.k_list:
        frame_cm, frame_nm = [], []
        dot_cm, dot_nm = [], []
        support_frames = []
        for seed in range(a.seeds):
            rng = np.random.RandomState(7000 + seed)
            fc, fn, dc, dn = [], [], [], []
            for s, dots in sess.items():
                idx = np.array([j for v in dots.values() for j in v])
                if len(idx) < k + 20:
                    continue
                # frame-level
                sup = rng.choice(idx, min(k, len(idx)), replace=False)
                rest = np.setdiff1d(idx, sup)
                if len(rest) < 10 or len(sup) < 4:
                    continue
                Z = np.column_stack([pc[sup], np.ones(len(sup))])
                w = [fit_ridge(Z, lc[sup, ax], a.lam) for ax in range(2)]
                Zq = np.column_stack([pc[rest], np.ones(len(rest))])
                qc = np.column_stack([Zq @ w[0], Zq @ w[1]])
                c, n = metrics(qc, rest); fc.append(c); fn.append(n)
                # dot-level
                keys = list(dots.keys())
                if len(keys) < k:
                    continue
                choose = rng.choice(len(keys), k, replace=False)
                sel_frames = [j for ci in choose for j in dots[keys[ci]]]
                rest_dots = [ci for ci in range(len(keys)) if ci not in set(choose)]
                rest_frames = [j for ci in rest_dots for j in dots[keys[ci]]]
                if len(sel_frames) < 4 or len(rest_frames) < 10:
                    continue
                # dot mean as one support sample
                mp = np.array([pc[dots[keys[ci]]].mean(0) for ci in choose])
                ml = np.array([lc[dots[keys[ci]]].mean(0) for ci in choose])
                Z = np.column_stack([mp, np.ones(len(mp))])
                w = [fit_ridge(Z, ml[:, ax], a.lam) for ax in range(2)]
                rf = np.array(rest_frames)
                Zq = np.column_stack([pc[rf], np.ones(len(rf))])
                qc = np.column_stack([Zq @ w[0], Zq @ w[1]])
                c, n = metrics(qc, rf); dc.append(c); dn.append(n)
                support_frames.append(len(sel_frames))
            if fc:
                frame_cm.append(np.concatenate(fc)); frame_nm.append(np.concatenate(fn))
            if dc:
                dot_cm.append(np.concatenate(dc)); dot_nm.append(np.concatenate(dn))
        if frame_cm:
            c = np.concatenate(frame_cm); n = np.concatenate(frame_nm)
            out[f"frame_{k}"] = dict(cm=float(c.mean()), norm=float(n.mean()),
                                     n=int(len(c)))
            log(f"[dot] RESULT frame K={k} cm={c.mean():.4f} "
                f"norm={n.mean():.4f} eval_frames={len(c)}")
        if dot_cm:
            c = np.concatenate(dot_cm); n = np.concatenate(dot_nm)
            out[f"dot_{k}"] = dict(cm=float(c.mean()), norm=float(n.mean()),
                                   n=int(len(c)),
                                   support_frames=float(np.mean(support_frames)))
            log(f"[dot] RESULT dot   K={k} cm={c.mean():.4f} "
                f"norm={n.mean():.4f} eval_frames={len(c)} "
                f"support_frames≈{np.mean(support_frames):.1f}")
    o = Path(a.out)
    o.parent.mkdir(parents=True, exist_ok=True)
    with open(o.with_suffix(".json"), "w", encoding="utf-8") as fh:
        json.dump(dict(args=vars(a), no_cal=dict(cm=float(cm0.mean()),
                                                 norm=float(nm0.mean())),
                       results=out), fh, indent=2)
    log(f"[dot] saved {o}.json")


if __name__ == "__main__":
    main()
