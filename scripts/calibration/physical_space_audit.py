"""物理空间坐标审计（不训练）。

检查：
  1. 坐标自检：物理点 -> 屏幕坐标 -> 标准厘米 -> 物理点；
  2. 真实配对在标准厘米空间的残差（mm）；
  3. 各朝向的标签分布在标准厘米空间是否重合；
  4. GC / RGD 设备在官方设备表中的覆盖率与相机偏移可用性。

标准厘米空间定义：
  p = ((x_px/screen_w - 0.5) * screen_w_cm,
       (y_px/screen_h - 0.5) * screen_h_cm)
  p_canon = R(-phi) p,  phi = atan2(imu_x, -imu_y)
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path[:0] = ["src", "."]


def log(msg):
    print(msg, flush=True)


def norm_dev(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def rot(v, phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.stack([c * v[..., 0] + s * v[..., 1],
                     -s * v[..., 0] + c * v[..., 1]], -1)


def canon(gx, gy, sw, sh, cmx, cmy, phi):
    w, h = sw * cmx, sh * cmy
    p = np.array([(gx / sw - 0.5) * w, (gy / sh - 0.5) * h])
    return rot(p, phi)


def self_check():
    rng = np.random.RandomState(0)
    err = 0.0
    for _ in range(200):
        w, h = rng.uniform(5, 13), rng.uniform(8, 25)
        p = rng.uniform([-w / 2, -h / 2], [w / 2, h / 2])
        phi = rng.uniform(-np.pi, np.pi)
        v = rot(p, -phi)          # canonical -> current orientation
        gx, gy = v[0] / w + 0.5, v[1] / h + 0.5
        q = canon(gx, gy, 1.0, 1.0, w, h, phi)
        err = max(err, float(np.abs(q - p).max()))
    log(f"[audit] self_check_max_err={err:.3e}")
    if err > 1e-9:
        raise RuntimeError("coordinate self-check failed")


def load_devices(path):
    spec = {}
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                spec[norm_dev(r["DeviceName"])] = dict(
                    W=float(r["DeviceScreenWidthMm"]),
                    H=float(r["DeviceScreenHeightMm"]),
                    cx=float(r["DeviceCameraXMm"]),
                    cy=float(r["DeviceCameraYMm"]),
                    cxs=float(r["DeviceCameraToScreenXMm"]),
                    cys=float(r["DeviceCameraToScreenYMm"]),
                )
            except Exception:
                continue
    return spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gc-index", required=True)
    ap.add_argument("--rgd-index", default="")
    ap.add_argument("--device-csv", required=True)
    ap.add_argument("--pairs", default="")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    log(f"[audit] args={vars(a)}")
    self_check()
    spec = load_devices(a.device_csv)
    log(f"[audit] device specs={len(spec)}")

    # ---- GC rows ----
    gc = {}
    dev_count = Counter()
    dev_ok = Counter()
    ori_stats = defaultdict(list)
    with open(a.gc_index, encoding="utf-8") as fh:
        for i, r in enumerate(csv.DictReader(fh)):
            if a.split and r.get("split", a.split) != a.split:
                continue
            try:
                sw, sh = float(r["screen_w"]), float(r["screen_h"])
                cmx, cmy = float(r["cm_px_x"]), float(r["cm_px_y"])
                gx, gy = float(r["gaze_x"]), float(r["gaze_y"])
                phi = float(np.arctan2(float(r["imu_x"]),
                                       -float(r["imu_y"])))
            except Exception:
                continue
            p = canon(gx, gy, sw, sh, cmx, cmy, phi)
            gc[i] = p
            d = norm_dev(r["device"])
            dev_count[d] += 1
            if d in spec:
                dev_ok[d] += 1
            ori_stats[str(r.get("orientation", "0"))].append(p)
    log(f"[audit] GC frames={len(gc)} devices={len(dev_count)}")
    missing = [d for d in dev_count if d not in spec]
    log(f"[audit] GC devices missing from spec: {len(missing)} "
        f"frames={sum(dev_count[d] for d in missing)} "
        f"top={sorted(((dev_count[d], d) for d in missing), reverse=True)[:5]}")

    for o, ps in sorted(ori_stats.items()):
        p = np.array(ps)
        log(f"[audit] GC ori={o} n={len(p)} "
            f"mean=({p[:,0].mean():+.2f},{p[:,1].mean():+.2f})cm "
            f"std=({p[:,0].std():.2f},{p[:,1].std():.2f})cm "
            f"p5=({np.percentile(p[:,0],5):+.2f},"
            f"{np.percentile(p[:,1],5):+.2f}) "
            f"p95=({np.percentile(p[:,0],95):+.2f},"
            f"{np.percentile(p[:,1],95):+.2f})")

    # ---- pair residuals in canonical cm ----
    if a.pairs:
        z = np.load(a.pairs, allow_pickle=True)
        src = z["src_row"].astype(np.int64)
        tgt = z["tgt_row"].astype(np.int64)
        d = []
        by = defaultdict(list)
        for s, t in zip(src, tgt):
            if int(s) not in gc or int(t) not in gc:
                continue
            ds = float(np.linalg.norm(gc[int(s)] - gc[int(t)])) * 10.0
            d.append(ds)
            by["pair"].append(ds)
        d = np.array(d)
        log(f"[audit] pair residual mm: n={len(d)} mean={d.mean():.3f} "
            f"median={np.median(d):.3f} p90={np.percentile(d,90):.3f} "
            f"p95={np.percentile(d,95):.3f} max={d.max():.3f}")

    # ---- RGD device coverage ----
    if a.rgd_index:
        rcount = Counter()
        with open(a.rgd_index, encoding="utf-8") as fh:
            for i, r in enumerate(csv.DictReader(fh)):
                rcount[r["device"]] += 1
        log(f"[audit] RGD devices={len(rcount)}")
        for dev, n in rcount.most_common(20):
            log(f"[audit] RGD {dev} frames={n} in_spec={norm_dev(dev) in spec}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".json"), "w", encoding="utf-8") as fh:
        json.dump(dict(
            args=vars(a),
            gc_frames=len(gc),
            gc_devices=len(dev_count),
            gc_missing_devices={d: dev_count[d] for d in missing},
            gc_ori={o: dict(n=len(v)) for o, v in ori_stats.items()},
        ), fh, indent=2)
    log(f"[audit] saved {out}.json")


if __name__ == "__main__":
    main()
