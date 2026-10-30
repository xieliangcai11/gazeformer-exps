"""R1 可行性检查：真实跨朝向帧的空间配对率。

对同一被试的不同屏幕朝向：
  - 把人脸帧的屏幕上注视点转换到「物理厘米标准系」；
  - 在源朝向里找最近邻帧；
  - 统计 1/2/3 cm 容差内的匹配率。

只有匹配率足够高，才能用真实图片构造跨朝向配对损失；
同一张图的合成配对不能替代真实帧（见前期结论）。

坐标约定与训练/诊断一致：
  cm 点 = ((x_norm-0.5)*w_cm, (y_norm-0.5)*h_cm)
  标准系 = R(-phi) · cm 点，phi = atan2(imu_x, -imu_y)
"""
import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path[:0] = ["src", "."]


def log(msg):
    print(msg, flush=True)


def rot_canon(v, phi):
    """Rotate by -phi (same convention as the diagnostics)."""
    c, s = np.cos(phi), np.sin(phi)
    return np.stack([c * v[..., 0] + s * v[..., 1],
                     -s * v[..., 0] + c * v[..., 1]], -1)


def canon_point(gx, gy, sw, sh, cmx, cmy, phi):
    wcm, hcm = sw * cmx, sh * cmy
    v = np.array([(gx / sw - 0.5) * wcm, (gy / sh - 0.5) * hcm])
    return rot_canon(v, phi)


def self_check():
    """A physical point must map back to itself after canonicalisation."""
    rng = np.random.RandomState(0)
    err = 0.0
    for _ in range(100):
        wcm, hcm = rng.uniform(5, 12), rng.uniform(8, 20)
        p = rng.uniform([-wcm / 2, -hcm / 2], [wcm / 2, hcm / 2])
        phi = rng.uniform(-np.pi, np.pi)
        # inverse: canonical -> screen normalised
        v = rot_canon(p, -phi)
        gx = (v[0] / wcm + 0.5) * 1.0  # normalized x (sw=1)
        gy = (v[1] / hcm + 0.5) * 1.0
        q = canon_point(gx, gy, 1.0, 1.0, wcm, hcm, phi)
        err = max(err, float(np.abs(q - p).max()))
    log(f"[pair] self_check_max_err={err:.3e}")
    if err > 1e-9:
        raise RuntimeError("canonicalisation self-check failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-subjects", type=int, default=0)
    ap.add_argument("--tols", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-pairs-tol", type=float, default=0.0,
                    help=">0: save matched pairs at this tolerance")
    a = ap.parse_args()
    log(f"[pair] args={vars(a)}")
    self_check()

    rows = defaultdict(lambda: defaultdict(list))
    with open(a.index_csv, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
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
            p = canon_point(gx, gy, sw, sh, cmx, cmy, phi)
            rows[r["subject"]][str(r.get("orientation", "0"))].append(
                (p[0], p[1], r["frame_path"], int(float(r.get("frame_index", 0))),
                 str(r.get("dot_num", ""))))
    subjects = sorted(rows)
    if a.max_subjects:
        subjects = subjects[:a.max_subjects]
    log(f"[pair] subjects={len(subjects)} orientations="
        f"{sorted({o for s in subjects for o in rows[s]})}")

    pair_stats = defaultdict(lambda: dict(
        n=0, sum_nn=0.0, dist=[], match={f"{t}": 0 for t in a.tols},
        unique2=set()))
    per_subject = {}
    saved = []
    t0 = time.time()
    for si, s in enumerate(subjects):
        oris = sorted(rows[s])
        for i in range(len(oris)):
            for j in range(len(oris)):
                if i == j:
                    continue
                src, tgt = oris[i], oris[j]
                S = rows[s][src]
                T = rows[s][tgt]
                if len(S) < 5 or len(T) < 5:
                    continue
                sp = np.array([[x[0], x[1]] for x in S])
                tp = np.array([[x[0], x[1]] for x in T])
                tree = cKDTree(sp)
                d, idx = tree.query(tp, k=1)
                st = pair_stats[(src, tgt)]
                st["n"] += len(T)
                st["sum_nn"] += float(d.sum())
                st["dist"].append(d)
                for tol in a.tols:
                    st["match"][str(tol)] += int((d <= tol).sum())
                st["unique2"].update(np.unique(idx[d <= 2.0]).tolist())
                if a.save_pairs_tol > 0:
                    for k in np.where(d <= a.save_pairs_tol)[0]:
                        saved.append((s, src, tgt, S[idx[k]][2], T[k][2],
                                      float(d[k])))
        if si % 50 == 0:
            log(f"[pair] subject {si}/{len(subjects)} s={s} "
                f"elapsed={time.time()-t0:.1f}s")

    log("[pair] ---- per orientation pair ----")
    summary = {}
    for (src, tgt), st in sorted(pair_stats.items()):
        n = st["n"]
        d = np.concatenate(st["dist"]) if st["dist"] else np.array([])
        out = dict(n=n, mean_nn=float(d.mean()) if len(d) else 0.0,
                   median_nn=float(np.median(d)) if len(d) else 0.0,
                   unique_source_at_2cm=len(st["unique2"]))
        for tol in a.tols:
            k = st["match"][str(tol)]
            out[f"match_{tol}cm"] = k
            out[f"rate_{tol}cm"] = k / max(n, 1)
        summary[f"{src}->{tgt}"] = out
        log(f"[pair] {src}->{tgt} n={n} mean_nn={out['mean_nn']:.3f}cm "
            f"median={out['median_nn']:.3f}cm "
            f"match1={out.get('rate_1.0cm', 0):.3f} "
            f"match2={out.get('rate_2.0cm', 0):.3f} "
            f"match3={out.get('rate_3.0cm', 0):.3f} "
            f"uniq@2={out['unique_source_at_2cm']}")

    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp.with_suffix(".json"), "w", encoding="utf-8") as fh:
        json.dump(dict(args=vars(a), pairs=summary, per_subject=per_subject),
                  fh, indent=2)
    if saved:
        np.savez_compressed(outp.with_suffix(".pairs.npz"),
                            pairs=np.array(saved, object))
    log(f"[pair] saved {outp}.json" + (" + .pairs.npz" if saved else ""))


if __name__ == "__main__":
    main()
