"""构建真实跨朝向配对表（R1）。

同一被试、不同朝向：
  - 注视点转到物理厘米标准系；
  - 双向最近邻 + 互为最近（mutual nearest）；
  - 距离 <= tol 才保留；
  - 每个源帧最多保留 max-per-source 个目标。

输出 npz：
  src_row, tgt_row, dist, subject, src_ori, tgt_ori, src_path, tgt_path
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path[:0] = ["src", "."]


def log(msg):
    print(msg, flush=True)


def rot_canon(v, phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.stack([c * v[..., 0] + s * v[..., 1],
                     -s * v[..., 0] + c * v[..., 1]], -1)


def canon(gx, gy, sw, sh, cmx, cmy, phi):
    wcm, hcm = sw * cmx, sh * cmy
    return rot_canon(np.array([(gx / sw - 0.5) * wcm,
                               (gy / sh - 0.5) * hcm]), phi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--tol", type=float, default=2.0)
    ap.add_argument("--max-per-source", type=int, default=3)
    ap.add_argument("--max-total", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    log(f"[build] args={vars(a)}")

    # row index -> metadata; grouped by subject/orientation
    rows = defaultdict(lambda: defaultdict(list))
    with open(a.index_csv, encoding="utf-8") as fh:
        for i, r in enumerate(csv.DictReader(fh)):
            if a.split and r.get("split", a.split) != a.split:
                continue
            try:
                sw, sh = float(r["screen_w"]), float(r["screen_h"])
                cmx, cmy = float(r["cm_px_x"]), float(r["cm_px_y"])
                gx, gy = float(r["gaze_x"]), float(r["gaze_y"])
                gxi, gyi = float(r["imu_x"]), float(r["imu_y"])
            except Exception:
                continue
            phi = float(np.arctan2(gxi, -gyi))
            p = canon(gx, gy, sw, sh, cmx, cmy, phi)
            rows[r["subject"]][str(r.get("orientation", "0"))].append(
                (i, p[0], p[1], r["frame_path"]))

    out = defaultdict(list)
    stats = defaultdict(lambda: dict(n=0, pairs=0, mean=0.0))
    for s, oris in rows.items():
        keys = sorted(oris)
        for ii in range(len(keys)):
            for jj in range(ii + 1, len(keys)):
                A, B = oris[keys[ii]], oris[keys[jj]]
                if len(A) < 5 or len(B) < 5:
                    continue
                pa = np.array([[x[1], x[2]] for x in A])
                pb = np.array([[x[1], x[2]] for x in B])
                ta, tb = cKDTree(pa), cKDTree(pb)
                dab, iab = tb.query(pa, k=1)   # A -> B
                dba, iba = ta.query(pb, k=1)   # B -> A
                # mutual nearest
                keep = []
                for ai, (bi, d) in enumerate(zip(iab, dab)):
                    if d <= a.tol and iba[bi] == ai:
                        keep.append((ai, bi, float(d)))
                # cap per source
                cnt = defaultdict(int)
                for ai, bi, d in keep:
                    if cnt[ai] >= a.max_per_source:
                        continue
                    cnt[ai] += 1
                    out["src_row"].append(A[ai][0])
                    out["tgt_row"].append(B[bi][0])
                    out["dist"].append(d)
                    out["subject"].append(s)
                    out["src_ori"].append(keys[ii])
                    out["tgt_ori"].append(keys[jj])
                    out["src_path"].append(A[ai][3])
                    out["tgt_path"].append(B[bi][3])
                st = stats[f"{keys[ii]}->{keys[jj]}"]
                st["n"] += len(A)
                st["pairs"] += len(keep)
                if keep:
                    st["mean"] += float(np.mean([x[2] for x in keep])) * len(keep)
                if a.max_total and len(out["src_row"]) >= a.max_total:
                    break
            if a.max_total and len(out["src_row"]) >= a.max_total:
                break
        if a.max_total and len(out["src_row"]) >= a.max_total:
            break

    n = len(out["src_row"])
    log(f"[build] pairs={n}")
    for k, st in sorted(stats.items()):
        mean = st["mean"] / max(st["pairs"], 1)
        log(f"[build] {k} frames={st['n']} pairs={st['pairs']} "
            f"per_src={st['pairs']/max(st['n'],1):.3f} mean_d={mean:.3f}cm")
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **{
        k: (np.array(v, object) if isinstance(v[0], str) else np.array(v))
        for k, v in out.items()
    })
    with open(p.with_suffix(".json"), "w", encoding="utf-8") as fh:
        json.dump(dict(args=vars(a), stats=stats, pairs=n), fh, indent=2)
    log(f"[build] saved {p} / {p.with_suffix('.json')}")


if __name__ == "__main__":
    main()
