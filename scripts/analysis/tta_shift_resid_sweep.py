"""二维扫描：整体平移(λ_b) × 逐帧残差(λ_s)（通用 RGBDGaze / GazeCapture）。

y = base + λ_b·Δmean + λ_s·(tta − base − Δmean)
  λ_b=0,λ_s=0 ⇒ 不校准        λ_b=1,λ_s=1 ⇒ 原样 TTA
  λ_b=1,λ_s=0 ⇒ 只用 TTA 的会话级平移（"bias-only"）

用法：
  python -m scripts.analysis.tta_shift_resid_sweep --dump ... --index-csv ... --label GC-val
"""
import argparse
import csv
from collections import defaultdict

import numpy as np


def load_cm(index_csv):
    out = {}
    with open(index_csv, encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        fields = rd.fieldnames or []
        pcol = "rgb_path" if "rgb_path" in fields else "frame_path"
        for r in rd:
            try:
                out[r[pcol]] = (float(r["cm_px_x"]) * float(r["screen_w"]),
                                float(r["cm_px_y"]) * float(r["screen_h"]))
            except (KeyError, ValueError):
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--grid", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--fix", default=None, help="只评估固定点，如 0.75,0.25（λ_b,λ_s）")
    a = ap.parse_args()

    cm_map = load_cm(a.index_csv)
    rows = defaultdict(list)
    with open(a.dump, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows[r["session"]].append(r)
    recs = []
    for s, rs in rows.items():
        rs.sort(key=lambda r: int(float(r["frame_no"])))
        b = np.array([[float(r["base_px"]), float(r["base_py"])] for r in rs])
        t = np.array([[float(r["tta_px"]), float(r["tta_py"])] for r in rs])
        lab = np.array([[float(r["label_x"]), float(r["label_y"])] for r in rs])
        cm = np.array([cm_map.get(r["path"], (np.nan, np.nan)) for r in rs])
        if np.isnan(cm).any():
            continue
        recs.append((s, b, t, lab, cm))
    n = sum(len(r[1]) for r in recs)

    def em(y, lab, cm):
        return float(np.hypot((y[:, 0] - lab[:, 0]) * cm[:, 0],
                              (y[:, 1] - lab[:, 1]) * cm[:, 1]).mean())

    base_fw = sum(em(b, lab, cm) * len(b) for _, b, t, lab, cm in recs) / n
    print(f"=== 平移×残差扫描 {a.label} ===")
    print(f"会话={len(recs)} 帧={n} base={base_fw:.4f}")
    if a.fix:
        lb, ls = (float(x) for x in a.fix.split(","))
        tot, rel = 0.0, []
        for s, b, t, lab, cm in recs:
            dm = t.mean(0) - b.mean(0)
            y = b + lb * dm + ls * (t - b - dm)
            e, e0 = em(y, lab, cm), em(b, lab, cm)
            tot += e * len(b)
            if e0 > 1e-9:
                rel.append((e / e0 - 1, s))
        rel.sort(reverse=True)
        print(f"  fix λ_b={lb:.2f} λ_s={ls:.2f}: EM={tot/n:.4f} "
              f"({100*(tot/n/base_fw-1):+.2f}%) 最差={100*rel[0][0]:+.1f}% "
              f"({'/'.join(rel[0][1].split('/')[-3:-1])})")
        return

    grid = [float(x) for x in a.grid.split(",")]
    print("  λ_b\\λ_s |" + "".join(f" {g:>6.2f} " for g in grid))
    best = (1e9, None)
    for lb in grid:
        cells = []
        for ls in grid:
            tot = 0.0
            for _, b, t, lab, cm in recs:
                dm = t.mean(0) - b.mean(0)
                y = b + lb * dm + ls * (t - b - dm)
                tot += em(y, lab, cm) * len(b)
            e = tot / n
            cells.append(f"{100*(e/base_fw-1):+6.1f}")
            if e < best[0]:
                best = (e, (lb, ls))
        print(f"  {lb:>5.2f}  |" + "".join(f" {c:>6} " for c in cells))
    e, (lb, ls) = best
    print(f"  最佳全局: λ_b={lb:.2f} λ_s={ls:.2f} → {e:.4f} ({100*(e/base_fw-1):+.2f}%)")

    # 逐会话最优（oracle 上界）
    orc = 0.0
    for _, b, t, lab, cm in recs:
        dm = t.mean(0) - b.mean(0)
        be = min(em(b + lb2 * dm + ls2 * (t - b - dm), lab, cm)
                 for lb2 in grid for ls2 in grid)
        orc += be * len(b)
    print(f"  [oracle 逐会话] {orc/n:.4f} ({100*(orc/n/base_fw-1):+.2f}%)")


if __name__ == "__main__":
    main()
