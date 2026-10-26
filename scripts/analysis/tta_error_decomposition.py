"""阶段 0：TTA 增益的**误差分解**（不训练、不占 GPU）。

数据：`ttadump_budget_s100.csv`（逐帧 base/tta 预测 + 标签）。
用途：回答“TTA 到底修了什么”，为后续方法设计提供依据：
  · 均值偏移（bias） vs 尺度（scale） vs 非线性；
  · 被 TTA 弄坏的会话有什么共同特征；
  · 若只允许改其中一项，哪一项最值钱。

用法：
    python -m scripts.analysis.tta_error_decomposition \
        --dump out/rgbdgaze/analysis/ttadump_budget_s100.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _bootstrap() -> Path:
    here = Path(__file__).resolve()
    for d in (here.parent, *here.parents):
        if (d / "src" / "gazelab" / "__init__.py").is_file():
            for p in (str(d), str(d / "src")):
                if p not in sys.path:
                    sys.path.insert(0, p)
            return d
    raise RuntimeError("找不到项目根")


_ROOT = _bootstrap()
from scripts.analysis.context_calib_l2 import load_index  # noqa: E402


def load_dump(path):
    rows = defaultdict(list)
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows[r["session"]].append(r)
    for k in rows:
        rows[k].sort(key=lambda r: int(float(r["frame_no"])))
    return rows


def em(y, lab, cm):
    return np.hypot((y[:, 0] - lab[:, 0]) * cm[:, 0],
                    (y[:, 1] - lab[:, 1]) * cm[:, 1]).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="out/rgbdgaze/analysis/ttadump_budget_s100.csv")
    args = ap.parse_args()

    meta = load_index()
    rows = load_dump(args.dump)
    print(f"[decomp] 会话 {len(rows)}")

    recs = []
    for k, rs in sorted(rows.items()):
        pred = np.array([[float(r["base_px"]), float(r["base_py"])] for r in rs])
        tta = np.array([[float(r["tta_px"]), float(r["tta_py"])] for r in rs])
        lab = np.array([[float(r["label_x"]), float(r["label_y"])] for r in rs])
        m = meta.get(rs[0]["path"])
        if m is None:
            continue
        cm = np.tile(np.array([m[3], m[4]]), (len(rs), 1))

        mb, mt, ml = pred.mean(0), tta.mean(0), lab.mean(0)
        sb = pred.std(0) + 1e-6
        st = tta.std(0)
        sl = lab.std(0) + 1e-6
        s_tta = st / sb            # TTA 引入的尺度
        s_or = sl / sb             # oracle 需要的尺度

        variants = {
            "base": pred,
            "tta": tta,
            # 只保留 TTA 的均值偏移（形状用 base）
            "bias_only": pred + (mt - mb),
            # 只保留 TTA 的尺度（均值用 base）
            "scale_only": mb + (pred - mb) * s_tta,
            # oracle：只改均值 / 只改尺度 / 两者
            "oracle_bias": pred + (ml - mb),
            "oracle_scale": mb + (pred - mb) * s_or,
            "oracle_both": (pred - mb) * s_or + ml,
            # TTA 的尺度 + base 的均值（检验“尺度部分单独是否有效”）
            "tta_scale_base_mean": mb + (pred - mb) * s_tta,
        }
        e = {name: em(y, lab, cm) for name, y in variants.items()}
        # 会话特征（无标签）
        d = np.abs(np.diff(pred, axis=0))
        dwell = float((d.max(1) < 0.002).mean()) if len(d) else 0.0
        recs.append(dict(key=k, n=len(rs), dwell=dwell,
                         s_tta_x=s_tta[0], s_tta_y=s_tta[1],
                         bias_mag=float(np.abs(mt - mb).mean()),
                         **e))

    n = np.array([r["n"] for r in recs])
    fw = lambda key: float(np.sum([r[key] * r["n"] for r in recs]) / n.sum())

    print(f"\n{'变体':>22} {'被试等权':>9} {'帧加权':>8} {'相对 base':>10}")
    base_fw = fw("base")
    for name in ("base", "tta", "bias_only", "scale_only",
                 "oracle_bias", "oracle_scale", "oracle_both"):
        eq = float(np.mean([r[name] for r in recs]))
        v = fw(name)
        print(f"{name:>22} {eq:9.3f} {v:8.3f} {100*(v/base_fw-1):+9.1f}%")

    print(f"\n=== 逐会话（相对 base 的变化）===")
    print(f"{'会话':>14} {'base':>6} {'tta':>6} {'bias':>6} {'scale':>6} | "
          f"{'s_x':>5} {'s_y':>5} {'|Δmean|':>7} {'dwell':>5}")
    for r in sorted(recs, key=lambda r: (r["key"].split("/")[-2], r["key"].split("/")[-3])):
        name = "/".join(r["key"].split("/")[-3:-1])
        print(f"{name:>14} {r['base']:6.2f} {r['tta']:6.2f} {r['bias_only']:6.2f} "
              f"{r['scale_only']:6.2f} | {r['s_tta_x']:5.2f} {r['s_tta_y']:5.2f} "
              f"{r['bias_mag']:7.3f} {r['dwell']:5.3f}")

    # 相关：TTA 增益 vs 可观测特征
    g = np.array([r["tta"] / r["base"] - 1 for r in recs])
    print(f"\n=== 相关性（TTA 相对增益 vs 可观测特征）===")
    for name in ("s_tta_x", "s_tta_y", "bias_mag", "dwell"):
        x = np.array([r[name] for r in recs])
        print(f"  corr(gain, {name}) = {np.corrcoef(g, x)[0,1]:+.3f}")

    # 关键分解：TTA 的收益里，均值 vs 尺度各占多少
    print(f"\n=== 结论线索 ===")
    print(f"  TTA 总变化         : {100*(fw('tta')/base_fw-1):+.1f}%")
    print(f"  仅均值偏移(bias)   : {100*(fw('bias_only')/base_fw-1):+.1f}%")
    print(f"  仅尺度(scale)      : {100*(fw('scale_only')/base_fw-1):+.1f}%")
    print(f"  oracle 仅均值      : {100*(fw('oracle_bias')/base_fw-1):+.1f}%")
    print(f"  oracle 仅尺度      : {100*(fw('oracle_scale')/base_fw-1):+.1f}%")
    print(f"  oracle 两者        : {100*(fw('oracle_both')/base_fw-1):+.1f}%")


if __name__ == "__main__":
    main()
