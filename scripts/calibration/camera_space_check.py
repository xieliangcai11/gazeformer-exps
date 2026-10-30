"""验证 GazeCapture 官方「相机厘米空间」的坐标约定。

论文明确：iTracker 预测的是「相对相机的厘米坐标」，不是归一化屏幕坐标。
本脚本用真实配对（同一被试/不同朝向/同一注视点）检验：
  在哪种相机偏移约定下，配对点在相机厘米空间最接近。
"""
import argparse
import csv
import sys

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--device-csv", required=True)
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--limit", type=int, default=20000)
    a = ap.parse_args()
    log(f"[cam] args={vars(a)}")

    spec = {}
    with open(a.device_csv, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            try:
                spec[norm_dev(r["DeviceName"])] = dict(
                    cx=float(r["DeviceCameraXMm"]),
                    cy=float(r["DeviceCameraYMm"]),
                    cxs=float(r["DeviceCameraToScreenXMm"]),
                    cys=float(r["DeviceCameraToScreenYMm"]),
                    W=float(r["DeviceScreenWidthMm"]),
                    H=float(r["DeviceScreenHeightMm"]),
                )
            except Exception:
                continue
    log(f"[cam] devices={len(spec)}")

    rows = {}
    with open(a.index_csv, encoding="utf-8") as fh:
        for i, r in enumerate(csv.DictReader(fh)):
            if r.get("split") != "train":
                continue
            d = spec.get(norm_dev(r["device"]))
            if d is None:
                continue
            try:
                sw, sh = float(r["screen_w"]), float(r["screen_h"])
                cmx, cmy = float(r["cm_px_x"]), float(r["cm_px_y"])
                gx, gy = float(r["gaze_x"]), float(r["gaze_y"])
                phi = float(np.arctan2(float(r["imu_x"]), -float(r["imu_y"])))
            except Exception:
                continue
            rows[i] = (d, sw, sh, cmx, cmy, gx, gy, phi,
                       str(r["orientation"]))
    log(f"[cam] frames={len(rows)}")

    z = np.load(a.pairs, allow_pickle=True)
    src = z["src_row"].astype(np.int64)[:a.limit]
    tgt = z["tgt_row"].astype(np.int64)[:a.limit] if "tgt_row" in z else None
    # build_pairs outputs src_row/tgt_row; fall back to src/tgt
    if tgt is None:
        src = z["src"].astype(np.int64)[:a.limit]
        tgt = z["tgt"].astype(np.int64)[:a.limit]
    log(f"[cam] pairs={len(src)}")

    def point(row, mode):
        d, sw, sh, cmx, cmy, gx, gy, phi, ori = row
        wcm, hcm = sw * cmx * 10.0, sh * cmy * 10.0
        p = np.array([(gx / sw - 0.5) * wcm,
                      (gy / sh - 0.5) * hcm])
        p = rot(p, phi)  # current screen -> portrait canonical
        if mode == "none":
            off = (0.0, 0.0)
        elif mode == "cam":
            off = (d["cx"] - d["W"] / 2, d["cy"] - d["H"] / 2)
        elif mode == "cams":
            off = (d["cxs"] - d["W"] / 2, d["cys"] - d["H"] / 2)
        elif mode == "cams_inv":
            off = (d["W"] / 2 - d["cxs"], d["H"] / 2 - d["cys"])
        elif mode == "cam_tl":
            off = (d["cx"], d["cy"])
        else:
            raise ValueError(mode)
        return p - np.array(off)

    modes = ["none", "cam", "cams", "cams_inv", "cam_tl"]
    log("[cam] ---- pair distance in candidate camera spaces (mm) ----")
    for mode in modes:
        ds = []
        for s, t in zip(src, tgt):
            ps = point(rows[int(s)], mode)
            pt = point(rows[int(t)], mode)
            ds.append(float(np.linalg.norm(ps - pt)))
        ds = np.array(ds)
        log(f"[cam] {mode}: n={len(ds)} mean={ds.mean():.3f} "
            f"median={np.median(ds):.3f} p90={np.percentile(ds,90):.3f}")


if __name__ == "__main__":
    main()
