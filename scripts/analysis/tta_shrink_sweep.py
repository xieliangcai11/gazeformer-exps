"""扫描 TTA 平移收缩系数 λ（通用：RGBDGaze / GazeCapture）。

做法：y = base + λ·(mean_tta − mean_base)，逐会话；λ=0 等价于不校准，λ=1 等价于"只用 TTA 的均值偏移"。
输出：帧加权 EM、相对基线的变化、最差会话、以及**逐会话最优 λ（oracle，仅供上界参考）**。

用法：
  python -m scripts.analysis.tta_shrink_sweep \
      --dump out/gazecapture/analysis/tta_val_s100_dump.csv \
      --index-csv data/GazeCapture/index/index.csv
"""
import argparse
import csv
from collections import defaultdict

import numpy as np


def load_cm(index_csv):
    """path -> (cm/归一化单位) 逐帧：cm_px·screen 宽高。"""
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
    ap.add_argument("--lambdas", default="0,0.25,0.5,0.75,1")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    cm_map = load_cm(a.index_csv)
    rows = defaultdict(list)
    with open(a.dump, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows[r["session"]].append(r)
    miss = 0
    recs = []
    for s, rs in rows.items():
        rs.sort(key=lambda r: int(float(r["frame_no"])))
        b = np.array([[float(r["base_px"]), float(r["base_py"])] for r in rs])
        t = np.array([[float(r["tta_px"]), float(r["tta_py"])] for r in rs])
        l = np.array([[float(r["label_x"]), float(r["label_y"])] for r in rs])
        cm = np.array([cm_map.get(r["path"], (np.nan, np.nan)) for r in rs])
        if np.isnan(cm).any():
            miss += 1
            continue
        recs.append((s, b, t, l, cm, len(rs)))
    if not recs:
        print("无可用会话")
        return
    n = sum(r[5] for r in recs)

    def em(y, lab, cm):
        return float(np.hypot((y[:, 0] - lab[:, 0]) * cm[:, 0],
                              (y[:, 1] - lab[:, 1]) * cm[:, 1]).mean())

    def total(fn):
        return sum(em(fn(b, t, l), l, cm) * m for _, b, t, l, cm, m in recs) / n

    base = total(lambda b, t, l: b)
    print(f"=== 收缩扫描 {a.label} ===")
    print(f"会话={len(recs)} 帧={n} 缺 cm 会话={miss} base={base:.4f}")
    lams = [float(x) for x in a.lambdas.split(",")]
    for lam in lams:
        def fn(b, t, l, lam=lam):
            return b + lam * (t.mean(0) - b.mean(0))
        e = total(fn)
        rel = []
        for s, b, t, l, cm, _m in recs:
            e0 = em(b, l, cm)
            if e0 > 1e-9:
                rel.append((em(fn(b, t, l), l, cm) / e0 - 1, s))
        rel.sort(reverse=True)
        w, ws = rel[0]
        print(f"  shift×{lam:.2f}: EM={e:.4f} ({100*(e/base-1):+.2f}%)  "
              f"最差={100*w:+.1f}% ({'/'.join(ws.split('/')[-3:-1])})")

    # 逐会话 oracle λ（只用真值选，仅供上界参考，不可声明为方法）
    orc = 0.0
    for s, b, t, l, cm, _m in recs:
        best = min((em(b + lam * (t.mean(0) - b.mean(0)), l, cm), lam) for lam in
                   np.linspace(0, 1, 21))
        orc += best[0] * len(b)
    print(f"  [oracle 逐会话 λ] EM={orc/n:.4f} ({100*(orc/n/base-1):+.2f}%)  ← 上界，非方法")


if __name__ == "__main__":
    main()
