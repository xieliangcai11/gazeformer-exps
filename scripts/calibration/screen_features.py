"""Screen-geometry features shared by all 2D-screen datasets.

These features describe the device screen and the face position on it; they
are available without depth and therefore work on GazeCapture and RGBDGaze.

Features
--------
  cm_px_x, cm_px_y       physical centimetres per screen pixel
  screen_w, screen_h     screen size in pixels
  aspect                 screen_w / screen_h
  phys_w, phys_h         physical screen size (cm)
  log_phys_w, log_phys_h log physical size
  fg_x, fg_y, fg_w, fg_h face-grid position normalised by screen size
  fg_missing             1 if face-grid columns are absent
"""
import argparse
import csv
from pathlib import Path

import numpy as np

NAMES = ["cm_px_x", "cm_px_y", "screen_w", "screen_h", "aspect",
         "phys_w", "phys_h", "log_phys_w", "log_phys_h",
         "fg_x", "fg_y", "fg_w", "fg_h", "fg_missing"]


def log(msg):
    print(msg, flush=True)


def fnum(r, key, default=None):
    v = r.get(key)
    if v is None or v == "":
        return default
    try:
        return float(v)
    except ValueError:
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.index_csv, encoding="utf-8")))
    feats = np.zeros((len(rows), len(NAMES)), np.float32)
    valid = np.zeros(len(rows), bool)
    missing_fg = 0
    for i, r in enumerate(rows):
        cmx = fnum(r, "cm_px_x")
        cmy = fnum(r, "cm_px_y", cmx)
        sw = fnum(r, "screen_w")
        sh = fnum(r, "screen_h")
        if cmx is None or cmy is None or sw is None or sh is None or sw <= 0 or sh <= 0:
            continue
        pw, ph = sw * cmx, sh * cmy
        gx = fnum(r, "face_grid_x")
        gy = fnum(r, "face_grid_y")
        gw = fnum(r, "face_grid_w")
        gh = fnum(r, "face_grid_h")
        has_fg = None not in (gx, gy, gw, gh)
        if not has_fg:
            missing_fg += 1
            gx = gy = gw = gh = 0.0
        feats[i] = [cmx, cmy, sw, sh, sw / sh, pw, ph,
                    np.log(max(pw, 1e-3)), np.log(max(ph, 1e-3)),
                    gx / sw, gy / sh, gw / sw, gh / sh, 0.0 if has_fg else 1.0]
        valid[i] = True
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, row_idx=np.arange(len(rows)), feats=feats,
                        valid=valid, names=np.array(NAMES, object))
    log(f"[screen] saved {out} rows={len(rows)} valid={int(valid.sum())} "
        f"({100*valid.mean():.2f}%) face_grid_missing={missing_fg}")


if __name__ == "__main__":
    main()
