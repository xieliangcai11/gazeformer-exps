"""Add IMU/orientation columns to the RGBDGaze index from per-session label.csv.

RGBDGaze label.csv contains imu_x/y/z and orientation, but the processed
index.csv dropped them.  This script restores them, aligned by frame number.
"""
import argparse
import csv
import os
from pathlib import Path

import numpy as np


def log(msg):
    print(msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.index_csv, encoding="utf-8")))
    fields = list(rows[0].keys())
    extra = ["imu_x", "imu_y", "imu_z", "orientation"]
    fields = fields + [f for f in extra if f not in fields]
    cache = {}
    n_ok = n_bad = 0
    imus = []
    for r in rows:
        p = r.get("rgb_path") or r.get("frame_path")
        d = os.path.dirname(os.path.dirname(p))
        lab = os.path.join(d, "label.csv")
        if lab not in cache:
            try:
                cache[lab] = list(csv.DictReader(open(lab, encoding="utf-8")))
            except Exception as e:
                cache[lab] = None
                log(f"[imu] cannot read {lab}: {e}")
        L = cache[lab]
        frame = int(os.path.basename(p).split(".")[0])
        if L is None or frame >= len(L):
            r["imu_x"] = r["imu_y"] = r["imu_z"] = ""
            r["orientation"] = ""
            n_bad += 1
            continue
        labrow = L[frame]
        r["imu_x"] = labrow["imu_x"]
        r["imu_y"] = labrow["imu_y"]
        r["imu_z"] = labrow["imu_z"]
        r["orientation"] = labrow["orientation"]
        imus.append([float(labrow["imu_x"]), float(labrow["imu_y"]),
                     float(labrow["imu_z"])])
        n_ok += 1
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    if imus:
        im = np.array(imus)
        log(f"[imu] rows={len(rows)} ok={n_ok} bad={n_bad} "
            f"imu_mean={im.mean(0).round(4).tolist()} "
            f"imu_std={im.std(0).round(4).tolist()}")
    log(f"[imu] saved {out}")


if __name__ == "__main__":
    main()
