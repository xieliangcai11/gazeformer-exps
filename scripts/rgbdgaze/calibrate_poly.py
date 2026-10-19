"""一次性标定工具：对逐帧预测 CSV 做「多边形标定 + LOO α + 覆盖检查」。

用 `gazelab.models.dinov2_gaze.calibrator.PolyCalibrator`（模块化实现）。
标定口径：**只用 K 个标定点的真值，评估期零真值**；评估集 = 各会话除标定帧外的帧。

两种选点（与 docs/experiments_2026-10-15.md §8 与 B 组实验一致）：
  --mode prefix --k 50   ：会话开头 10% 内等距取 K 帧（当前"默默用开头"的协议 ✗）
  --mode grid  --grid 3  ：按屏幕 3x3 网格格心选最近帧（= 引导用户看满屏的**模拟** ✓）
                           ⚠️ grid 模式用真值选点，属"引导协议上界"的模拟，不是可部署结果

输出：逐会话 + 总体的 不标定 EM / 标定后 EM / α / 覆盖(bbox面积占比) / 回退会话数，
并把修正后的预测写成 CSV（新增 calib_alpha 列）。

用法：
    python -m scripts.rgbdgaze.calibrate_poly --preds out/rgbdgaze/analysis/eye224_test.csv \
        --mode grid --grid 3 --out out/rgbdgaze/analysis/eye224_test_calib_grid3.csv
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
    raise RuntimeError(f"找不到项目根：从 {here} 向上未发现 src/gazelab/__init__.py")


_PROJECT_ROOT = _bootstrap()

from configs.rgbdgaze_config import RGBDGaze_INDEX_DIR  # noqa: E402
from gazelab.models.dinov2_gaze.calibrator import PolyCalibrator  # noqa: E402


def _pick_grid(idx, P, L, n):
    """按 n×n 网格格心选最近帧（引导式标定的模拟）。"""
    cents = [((i + 0.5) / n, (j + 0.5) / n) for i in range(n) for j in range(n)]
    sel, used = [], set()
    for cx, cy in cents:
        dd = np.hypot(L[idx, 0] - cx, L[idx, 1] - cy)
        for o in np.argsort(dd):
            k = int(o)
            if k not in used:
                used.add(k)
                sel.append(idx[k])
                break
    return sorted(sel)


def _pick_prefix(idx, P, L, k, frac=0.10):
    pool = idx[:max(int(len(idx) * frac), 30)]
    return [pool[t] for t in np.unique(
        np.linspace(0, len(pool) - 1, min(k, len(pool))).astype(int))]


def main():
    ap = argparse.ArgumentParser(description="一次性标定（多项式 + LOO α + 覆盖检查）")
    ap.add_argument("--preds", required=True, help="逐帧预测 CSV（含 pred_x/y 与 label_x/y）")
    ap.add_argument("--index-csv", default=str(Path(RGBDGaze_INDEX_DIR) / "index.csv"))
    ap.add_argument("--mode", choices=["prefix", "grid"], default="grid")
    ap.add_argument("--k", type=int, default=50, help="prefix 模式的标定帧数")
    ap.add_argument("--grid", type=int, default=3, help="grid 模式的 n（n×n 个标定点）")
    ap.add_argument("--out", default=None, help="修正后的 CSV 路径（可选）")
    args = ap.parse_args()

    sc = {}
    with open(args.index_csv, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            sc[r["rgb_path"]] = (float(r["screen_w"]) * float(r["cm_px_x"]),
                                 float(r["screen_h"]) * float(r["cm_px_y"]))

    rows = list(csv.DictReader(open(args.preds, encoding="utf-8")))
    if not rows:
        raise SystemExit(f"空 CSV：{args.preds}")
    sess = defaultdict(list)
    for r in rows:
        sess[str(r["path"]).rsplit("/", 1)[0]].append(r)
    for s in sess:
        sess[s].sort(key=lambda r: int(r["frame_index"]))
    SESS = sorted(sess)
    if args.mode == "grid":
        print("[calib] ⚠️ mode=grid 用真值选点 ⇒ 这是**引导协议上界的模拟**，不是可部署结果")

    per_session, all_e, all_b, alphas, areas, fallbacks, used_cal = [], [], [], [], [], 0, set()
    corrected = {}
    for s in SESS:
        v = sess[s]
        P = np.array([[float(r["pred_x"]), float(r["pred_y"])] for r in v])
        L = np.array([[float(r["label_x"]), float(r["label_y"])] for r in v])
        idx = np.arange(len(v))
        cal = _pick_grid(idx, P, L, args.grid) if args.mode == "grid" else _pick_prefix(idx, P, L, args.k)
        if len(cal) < 8:                       # 与 PolyCalibrator.min_points 对齐
            continue
        cal = np.asarray(sorted(set(cal)))
        used_cal |= {s + f"#{c}" for c in cal.tolist()}
        ev = np.setdiff1d(idx, cal)
        screen_cm = sc[v[0]["path"]]
        pc = PolyCalibrator().fit(P[cal], L[cal], screen_cm=screen_cm)
        Q = pc.apply(P)
        for i in ev:
            all_b.append(float(np.hypot((P[i, 0] - L[i, 0]) * screen_cm[0],
                                        (P[i, 1] - L[i, 1]) * screen_cm[1])))
            all_e.append(float(np.hypot((Q[i, 0] - L[i, 0]) * screen_cm[0],
                                        (Q[i, 1] - L[i, 1]) * screen_cm[1])))
        alphas.append(pc.alpha)
        areas.append(pc.info["span"]["area"])
        fallbacks += 0 if pc.ok else 1
        per_session.append((s, np.mean([float(np.hypot((P[i, 0] - L[i, 0]) * screen_cm[0],
                                                     (P[i, 1] - L[i, 1]) * screen_cm[1]))
                                        for i in ev]) if len(ev) else float("nan"),
                            len(cal), pc.alpha, pc.info["span"]["area"], pc.ok))
        for i in idx:
            corrected[(v[i]["path"], v[i]["frame_index"])] = (float(Q[i, 0]), float(Q[i, 1]))

    e, b = np.array(all_e), np.array(all_b)
    print("=" * 104)
    print(f"口径：{args.mode}" + (f"（{args.grid}×{args.grid} 网格，模拟引导标定）"
                                 if args.mode == "grid" else f"（开头 10% 内 K={args.k}）"))
    print("=" * 104)
    print(f"  {'会话':<52}{'不标定':>9}{'标定后':>9}{'点数':>6}{'α':>6}{'bbox面积':>9}{'状态':>6}")
    for s, be, n, a, ar, ok in per_session:
        print(f"  {s[-46:]:<52}{be:>9.3f}{'':>9}{n:>6}{a:>6.2f}{ar:>9.2f}{'  ok' if ok else '  回退':>6}")
    print(f"\n  总体：不标定 {b.mean():.3f} → 标定后 {e.mean():.3f}"
          f"（相对 {100 * (e.mean() / b.mean() - 1):+.1f}%）")
    print(f"  平均 α={np.mean(alphas):.2f}  平均 bbox 面积占比={np.mean(areas):.2f}"
          f"  回退会话 {fallbacks}/{len(per_session)}  标定帧 {len(used_cal)}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        cols = list(rows[0].keys())
        if "calib_alpha" not in cols:
            cols.append("calib_alpha")
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in rows:
                key = (r["path"], r["frame_index"])
                r = dict(r)
                a = np.nan
                if key in corrected:
                    qx, qy = corrected[key]
                    scr = sc[r["path"]]
                    r["pred_x"], r["pred_y"] = f"{qx:.6f}", f"{qy:.6f}"
                    r["em_cm"] = f"{np.hypot((qx - float(r['label_x'])) * scr[0], (qy - float(r['label_y'])) * scr[1]):.4f}"
                    r["err_x_cm"] = f"{(qx - float(r['label_x'])) * scr[0]:.4f}"
                    r["err_y_cm"] = f"{(qy - float(r['label_y'])) * scr[1]:.4f}"
                r["calib_alpha"] = "" if np.isnan(a) else f"{a:.2f}"
                w.writerow(r)
        print(f"  修正后 CSV: {out}")


if __name__ == "__main__":
    main()
