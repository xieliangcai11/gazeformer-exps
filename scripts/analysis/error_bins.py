"""误差分箱分析：屏幕、设备、活动/朝向、头姿、注视分布。

只做分析，不改方法；标签只用于打分。
输出每个箱的 n / 平均 EM / 中位 / p95，以及 Spearman 相关。

用法：
  python -m scripts.analysis.error_bins \
    --feats out/.../feats_gc_test_m2_full.npz \
    --index-csv data/GazeCapture/index/index.csv \
    --pose-npz out/gazecapture/analysis/pose_features.npz
"""
import argparse
import csv
import sys
from collections import defaultdict

import numpy as np

sys.path[:0] = ["src", "."]


def load_index(index_csv, needed):
    need = set(needed.tolist())
    meta = {}
    with open(index_csv, encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        pcol = "frame_path" if "frame_path" in rd.fieldnames else "rgb_path"
        for r in rd:
            p = r[pcol]
            if p not in need:
                continue
            meta[p] = dict(
                device=r.get("device", ""),
                orientation=r.get("orientation", ""),
                activity=r.get("activity", ""),
                w_cm=float(r["screen_w"]) * float(r["cm_px_x"]),
                h_cm=float(r["screen_h"]) * float(r["cm_px_y"]),
                sx=float(r["screen_w"]) * float(r["cm_px_x"]),
                sy=float(r["screen_h"]) * float(r["cm_px_y"]),
            )
    return meta


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


def spearman(x, y):
    x = np.asarray(x); y = np.asarray(y)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return float("nan")
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def report(name, values, em, bins=6):
    values = np.asarray(values); em = np.asarray(em)
    ok = np.isfinite(values) & np.isfinite(em)
    values, em = values[ok], em[ok]
    if len(values) == 0:
        print(f"[{name}] no data")
        return
    qs = np.quantile(values, np.linspace(0, 1, bins + 1))
    qs = np.unique(qs)
    print(f"[{name}] n={len(values)} spear={spearman(values, em):+.3f}")
    for j in range(len(qs) - 1):
        m = (values >= qs[j]) & (values < qs[j + 1])
        if m.sum() < 10:
            continue
        print(f"  [{qs[j]:+.4f},{qs[j+1]:+.4f}) n={int(m.sum())} "
              f"EM={em[m].mean():.4f} med={np.median(em[m]):.4f} "
              f"p95={np.percentile(em[m],95):.4f}")


def report_cat(name, cat, em):
    cat = np.asarray(cat); em = np.asarray(em)
    ok = np.isfinite(em)
    cat, em = cat[ok], em[ok]
    print(f"[{name}]")
    for c in sorted(set(cat.tolist())):
        m = cat == c
        if m.sum() == 0:
            continue
        print(f"  {str(c):12s} n={int(m.sum())} EM={em[m].mean():.4f} "
              f"med={np.median(em[m]):.4f} p95={np.percentile(em[m],95):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--pose-npz", default="")
    a = ap.parse_args()

    z = np.load(a.feats, allow_pickle=True)
    pred = z["pred"].astype(np.float32)
    label = z["label"].astype(np.float32)
    path = z["path"]
    meta = load_index(a.index_csv, path)
    pose = load_pose(a.pose_npz, a.index_csv, path) if a.pose_npz else {}
    print(f"[bins] frames={len(pred)} meta={len(meta)} pose={len(pose)}")

    em = np.full(len(pred), np.nan, np.float32)
    for i in range(len(pred)):
        m = meta.get(path[i])
        if m is None:
            continue
        em[i] = np.hypot((pred[i, 0] - label[i, 0]) * m["sx"],
                         (pred[i, 1] - label[i, 1]) * m["sy"])
    ok = np.isfinite(em)
    print(f"[bins] overall EM={em[ok].mean():.4f} n={int(ok.sum())}")

    # 会话注视分布（标签标准差，cm）
    sess = defaultdict(list)
    for i in range(len(pred)):
        sess[path[i].rsplit("/", 1)[0]].append(i)
    spread = {}
    for s, idx in sess.items():
        idx = np.array(idx)
        m0 = meta.get(path[idx[0]])
        if m0 is None:
            continue
        lx = label[idx, 0] * m0["sx"]
        ly = label[idx, 1] * m0["sy"]
        spread[s] = float(np.hypot(lx.std(), ly.std()))
    sp_frames = np.array([spread.get(path[i].rsplit("/", 1)[0], np.nan)
                          for i in range(len(pred))])

    report("screen_width_cm", np.array([meta.get(p, {}).get("w_cm", np.nan) for p in path]), em)
    report("screen_height_cm", np.array([meta.get(p, {}).get("h_cm", np.nan) for p in path]), em)
    report("gaze_spread_cm", sp_frames, em)
    report_cat("device", [meta.get(p, {}).get("device", "") for p in path], em)
    if any(meta.get(p, {}).get("orientation", "") for p in path):
        report_cat("orientation", [meta.get(p, {}).get("orientation", "") for p in path], em)
    if any(meta.get(p, {}).get("activity", "") for p in path):
        report_cat("activity", [meta.get(p, {}).get("activity", "") for p in path], em)

    if pose:
        P = np.stack([pose.get(p, np.full(8, np.nan)) for p in path])
        names = ["yaw", "pitch", "roll", "iod", "openness", "edge_l", "edge_r", "face_ratio"]
        for j, n in enumerate(names):
            report(f"pose_{n}", P[:, j], em)


if __name__ == "__main__":
    main()
