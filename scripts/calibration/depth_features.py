"""Depth-image proxy features for RGBDGaze (shared with IMU/landmark features).

The released depth images are 8-bit colourised maps, not metric depth, but
their intensity/hue/gradient still carry face-distance and geometry cues.
Features are computed on the face bbox.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path[:0] = ["src", "."]

NAMES = ["d_mean", "d_std", "d_p10", "d_p50", "d_p90",
         "d_grad", "d_hue", "d_sat"]


def log(msg):
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=1000)
    ap.add_argument("--rgb-w", type=float, default=1080.0)
    ap.add_argument("--rgb-h", type=float, default=1440.0)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.index_csv, encoding="utf-8")))
    if a.limit:
        rows = rows[:a.limit]
    feats = np.zeros((len(rows), len(NAMES)), np.float32)
    valid = np.zeros(len(rows), bool)
    for i, r in enumerate(rows):
        p = r.get("depth_path")
        if not p:
            continue
        im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if im is None:
            continue
        if im.ndim == 3:
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        else:
            hsv = None
            gray = im
        scx = im.shape[1] / a.rgb_w
        scy = im.shape[0] / a.rgb_h
        x, y = float(r["bbox_x"]) * scx, float(r["bbox_y"]) * scy
        w, h = float(r["bbox_w"]) * scx, float(r["bbox_h"]) * scy
        x0, y0 = int(max(0, x)), int(max(0, y))
        x1, y1 = int(min(gray.shape[1], x + w)), int(min(gray.shape[0], y + h))
        if x1 <= x0 + 2 or y1 <= y0 + 2:
            continue
        c = gray[y0:y1, x0:x1].astype(np.float32)
        gx = cv2.Sobel(c, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(c, cv2.CV_32F, 0, 1, ksize=3)
        grad = float(np.hypot(gx, gy).mean())
        if hsv is not None:
            hh = hsv[y0:y1, x0:x1, 0].astype(np.float32)
            ss = hsv[y0:y1, x0:x1, 1].astype(np.float32)
            hue = float(hh.mean() / 180.0)
            sat = float(ss.mean() / 255.0)
        else:
            hue = sat = 0.0
        feats[i] = [c.mean() / 255.0, c.std() / 255.0,
                    np.percentile(c, 10) / 255.0,
                    np.percentile(c, 50) / 255.0,
                    np.percentile(c, 90) / 255.0,
                    grad / 255.0, hue, sat]
        valid[i] = True
        if i % a.log_every == 0:
            log(f"[depth] {i}/{len(rows)} valid={int(valid.sum())}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, row_idx=np.arange(len(rows)), feats=feats,
                        valid=valid, names=np.array(NAMES, object))
    log(f"[depth] saved {out} rows={len(rows)} valid={int(valid.sum())} "
        f"({100*valid.mean():.2f}%)")


if __name__ == "__main__":
    main()
