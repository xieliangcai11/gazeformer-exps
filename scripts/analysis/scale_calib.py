"""D1（系统性收缩）的可修复性检验：在**其他被试**上拟合尺度，评估留出被试。

收缩证据：pred std / label std = 0.885/0.898（RGBDGaze ②）、0.505/0.592（GazeCapture）
⇒ 若收缩是**模型/损失的性质**（跨被试稳定）⇒ 用别的被试拟合的尺度应能修好留出被试 ✓
⇒ 若收缩是**被试特异的** ⇒ 只能自身标定 ✗

三档：
  A 不修正
  S 尺度校正（只用 a，per-axis：a = std_label/std_pred，在拟合集上算）
  SB 尺度+偏置（a, b 最小二乘）
  O 同被试 oracle（在同一被试上拟合 ⇒ 上限参考）
分别在 RGBDGaze（LOSO）与 GazeCapture（用 1262 训练被试拟合 → 150 测试被试）上跑。
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/xieliangcai/code/gazeformer-exps")


def load_scale():
    out = {}
    for idx in (ROOT / "data/RGBDGaze/index/index.csv",
                ROOT / "data/GazeCapture/index/index.csv"):
        if not idx.exists():
            continue
        with open(idx, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                pk = "rgb_path" if "rgb_path" in r else "frame_path"
                out[r[pk]] = (float(r["screen_w"]) * float(r["cm_px_x"]),
                              float(r["screen_h"]) * float(r["cm_px_y"]))
    return out


SC = load_scale()


def load(p):
    rows = []
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r["path"] not in SC:
            continue
        rows.append((r["path"], r["subject"],
                     np.array([float(r["pred_x"]), float(r["pred_y"])]),
                     np.array([float(r["label_x"]), float(r["label_y"])])))
    return rows


def em(pred, lab, path):
    w, h = SC[path]
    return float(np.hypot((pred[0] - lab[0]) * w, (pred[1] - lab[1]) * h))


def fit_scale(rows, use_bias):
    """返回 (a[2], b[2])：pred' = a*pred + b 拟合到 label。"""
    P = np.array([r[2] for r in rows])
    L = np.array([r[3] for r in rows])
    a, b = np.ones(2), np.zeros(2)
    for k in range(2):
        if use_bias:
            X = np.stack([P[:, k], np.ones(len(P))], 1)
            coef, *_ = np.linalg.lstsq(X, L[:, k], rcond=None)
            a[k], b[k] = coef[0], coef[1]
        else:
            a[k] = L[:, k].std() / max(P[:, k].std(), 1e-9)
    return a, b


def run(tag, fit_rows, eval_rows, group_fit, loso=True):
    print("=" * 92)
    print(f"【{tag}】拟合 {len(fit_rows)} 帧 / {len({r[1] for r in fit_rows})} 被试"
          f" → 评估 {len(eval_rows)} 帧 / {len({r[1] for r in eval_rows})} 被试")
    print("=" * 92)
    base = np.array([em(r[2], r[3], r[0]) for r in eval_rows])
    a, b = fit_scale(fit_rows, False)
    e_s = np.array([em(r[2] * a + 0 * b, r[3], r[0]) for r in eval_rows])
    a2, b2 = fit_scale(fit_rows, True)
    e_sb = np.array([em(r[2] * a2 + b2, r[3], r[0]) for r in eval_rows])
    # 同被试 oracle
    by = defaultdict(list)
    for r in eval_rows:
        by[r[1]].append(r)
    e_o = []
    for s, v in by.items():
        a3, b3 = fit_scale(v, True)
        e_o += [em(r[2] * a3 + b3, r[3], r[0]) for r in v]
    e_o = np.array(e_o)
    P = np.array([r[2] for r in fit_rows]); L = np.array([r[3] for r in fit_rows])
    print(f"  拟合集收缩比: x {L[:, 0].std() / P[:, 0].std():.3f} | "
          f"y {L[:, 1].std() / P[:, 1].std():.3f}   拟合尺度 a={a.round(3)} b={b.round(3)}")
    print(f"  {'方案':<26}{'EM均值':>9}{'vs基线':>9}{'中位':>8}{'>3cm':>8}")
    for nm, arr in (("A 不修正", base), ("S 仅尺度 a（跨被试）", e_s),
                    ("SB 尺度+偏置（跨被试）", e_sb), ("O 同被试 oracle（上限）", e_o)):
        print(f"  {nm:<26}{arr.mean():>9.3f}{100 * (arr.mean() / base.mean() - 1):>8.1f}%"
              f"{np.median(arr):>8.3f}{100 * (arr > 3).mean():>7.2f}%")
    return base.mean(), e_s.mean(), e_sb.mean(), e_o.mean()


if __name__ == "__main__":
    if sys.argv[1] == "rgbd":
        rows = load(str(ROOT / "out/rgbdgaze/analysis/eye224_test.csv"))
        subs = sorted({r[1] for r in rows})
        for held in subs:                       # LOSO
            fit = [r for r in rows if r[1] != held]
            ev = [r for r in rows if r[1] == held]
            run(f"RGBDGaze LOSO（留出 {held}）", fit, ev, held)
    else:
        fit = load(str(ROOT / "out/gazecapture/analysis/gc_train_sub.csv"))
        ev = load(str(ROOT / "out/gazecapture/analysis/gc_test.csv"))
        run("GazeCapture（1262 训练被试 → 150 测试被试）", fit, ev, None)
