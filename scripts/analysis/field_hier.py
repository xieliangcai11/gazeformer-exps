"""层级化误差场（干净版）：全局仿射（吃尺度/收缩）→ 每格残差偏移（吃空间结构）。

设计（严格因果：每帧先用、后用其标签更新）：
  1) 全局仿射  β = LS(pred → (pred-label))，只用本 session 过去的帧（岭只罚斜率 ✓）
  2) 用全局校正后的位置查"每格残差均值"（3x3 池化 ✓），从残差里再减掉
  最终: pred_corr = pred - β(pred) - cellmean( pred - β(pred) )

对照：仅每格偏移 / 仅全局仿射 / 层级（本方法）
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/xieliangcai/code/gazeformer-exps")
CSV = sys.argv[1]
MIN = 8
NZ = 12

sc = {}
for _idx in (ROOT / "data/RGBDGaze/index", ROOT / "data/GazeCapture/index"):
    if not _idx.exists():
        continue
    with open(_idx / "index.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            pk = "rgb_path" if "rgb_path" in r else "frame_path"
            sc[r[pk]] = (float(r["screen_w"]) * float(r["cm_px_x"]),
                         float(r["screen_h"]) * float(r["cm_px_y"]))

sess = defaultdict(list)
for r in csv.DictReader(open(CSV, encoding="utf-8")):
    sess[r["path"].rsplit("/", 1)[0]].append(
        dict(fi=int(r["frame_index"]),
             P=np.array([float(r["pred_x"]), float(r["pred_y"])]),
             L=np.array([float(r["label_x"]), float(r["label_y"])]),
             C=str(r.get("activity") or ""), S=sc[r["path"]]))
for s in sess:
    sess[s].sort(key=lambda r: r["fi"])


def fit_affine(a, lam):
    """a: [n, Sx, Sy, Sxx, Sxy, Syy, Sdx, Sdxx, Sdxy, Sdy, Sdxy2, Sdyy]"""
    n = a[0]
    if n < MIN:
        return None
    pen = lam * n if np.isfinite(lam) else 0.0
    A = np.array([[a[0], a[1], a[2]],
                  [a[1], a[3] + pen, a[4]],
                  [a[2], a[4], a[5] + pen]])
    try:
        b1 = np.linalg.solve(A, a[6:9])
        b2 = np.linalg.solve(A, np.array([a[9], a[10], a[11]]))
    except np.linalg.LinAlgError:
        return None
    return b1, b2


def apply_affine(b, x, y):
    v = np.array([1.0, x, y])
    return np.array([v @ b[0], v @ b[1]])


def add(st, x, y, dx, dy):
    st[0] += 1
    st[1] += x; st[2] += y
    st[3] += x * x; st[4] += x * y; st[5] += y * y
    st[6] += dx; st[7] += dx * x; st[8] += dx * y
    st[9] += dy; st[10] += dy * x; st[11] += dy * y


def run(mode="hier", G=4, lam=0.01, radius=3):
    """mode: none | cell | glob | hier"""
    errs = []
    for s, v in sess.items():
        glob = defaultdict(lambda: np.zeros(NZ))
        grid = defaultdict(lambda: [0, 0.0, 0.0])
        for r in v:
            P, L, cond = r["P"], r["L"], r["C"]
            x, y = float(P[0]), float(P[1])
            b = fit_affine(glob[cond], lam) if mode in ("glob", "hier") else None
            g = apply_affine(b, x, y) if b is not None else np.zeros(2)
            p1 = P - g
            # 每格残差偏移（用校正后位置建索引）
            est_cell = None
            if mode in ("cell", "hier"):
                base_pt = p1 if mode == "hier" else P
                cx, cy = float(base_pt[0]), float(base_pt[1])
                i0 = min(G - 1, max(0, int(cx * G)))
                j0 = min(G - 1, max(0, int(cy * G)))
                cnt = sx = sy = 0
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        i, j = i0 + di, j0 + dj
                        if 0 <= i < G and 0 <= j < G:
                            c = grid[(cond, i, j)]
                            cnt += c[0]; sx += c[1]; sy += c[2]
                if cnt >= MIN:
                    est_cell = np.array([sx / cnt, sy / cnt])
            corr = p1 - est_cell if est_cell is not None else p1
            if mode == "cell":
                corr = P - est_cell if est_cell is not None else P
            errs.append(float(np.hypot((corr[0] - L[0]) * r["S"][0],
                                       (corr[1] - L[1]) * r["S"][1])))
            # ---- 更新（严格因果）----
            add(glob[cond], x, y, float(P[0] - L[0]), float(P[1] - L[1]))
            if mode == "hier":
                cx, cy = float(p1[0]), float(p1[1])
                res = (P - L) - g
            elif mode == "cell":
                cx, cy = x, y
                res = P - L
            else:
                continue
            c = grid[(cond, min(G - 1, max(0, int(cx * G))),
                      min(G - 1, max(0, int(cy * G))))]
            c[0] += 1
            c[1] += float(res[0])
            c[2] += float(res[1])
    return np.array(errs)


base = np.array([float(np.hypot((r["P"][0] - r["L"][0]) * r["S"][0],
                                (r["P"][1] - r["L"][1]) * r["S"][1]))
                 for s in sess for r in sess[s]])
print("=" * 100)
print(f"【层级化误差场】{Path(CSV).stem}  基线 EM {base.mean():.3f}  "
      f"中位 {np.median(base):.3f}  >3cm {100 * (base > 3).mean():.2f}%")
print("=" * 100)
print(f"  {'配置':<40}{'EM均值':>9}{'vs基线':>9}{'中位':>8}{'>3cm':>8}")
for tag, kw in (("A 不修正", None),
                ("仅全局仿射（每 session）", dict(mode="glob")),
                ("仅每格偏移 + 3x3", dict(mode="cell")),
                ("层级：全局仿射 → 每格偏移", dict(mode="hier")),
                ("层级 G=8", dict(mode="hier", G=8)),
                ("层级 G=4, 半径1(无池化)", dict(mode="hier", radius=1))):
    if kw is None:
        print(f"  {tag:<40}{base.mean():>9.3f}{0.0:>8.1f}%"
              f"{np.median(base):>8.3f}{100 * (base > 3).mean():>7.2f}%")
        continue
    e = run(**kw)
    print(f"  {tag:<40}{e.mean():>9.3f}{100 * (e.mean() / base.mean() - 1):>8.1f}%"
          f"{np.median(e):>8.3f}{100 * (e > 3).mean():>7.2f}%")
