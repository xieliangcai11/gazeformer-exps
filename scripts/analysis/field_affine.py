"""误差场·容量空间化：每格从"常数偏移"升级为"小仿射"，并扫描容量（岭）与平滑半径。

动机：
  · 位移误差（offset）只解释一部分；另一大块是**尺度/形状**误差（预测收缩 0.88）✗
    ⇒ 每格若能拟合 6 参数小仿射（pred→label），应能吃掉"收缩"部分 ✓
  · 但样本少时会过拟合（容量定律 ✗）⇒ 用**岭正则**把容量连续地从"仅偏移"调到"完整仿射"：
      λ→∞  ⇒ 退化为每格常数偏移（= 上一轮 −24.2% 的配置，作为内部对照 ✓）
      λ→0  ⇒ 完整每格仿射
  · 严格因果：先用后更新；邻域平滑用"统计量池化"实现（比平均估计更正确 ✓）

用法: python field_affine.py <预测CSV>
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/xieliangcai/code/gazeformer-exps")
CSV = sys.argv[1]
MIN = 8

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

# 每格统计量: n, Sx, Sy, Sxx, Sxy, Syy, <dx: S, Sx, Sy>, <dy: S, Sx, Sy>
NZ = 12


def newacc():
    return defaultdict(lambda: np.zeros(NZ))


def add(acc, key, x, y, dx, dy):
    a = acc[key]
    a[0] += 1
    a[1] += x; a[2] += y
    a[3] += x * x; a[4] += x * y; a[5] += y * y
    a[6] += dx; a[7] += dx * x; a[8] += dx * y
    a[9] += dy; a[10] += dy * x; a[11] += dy * y


def solve(a, lam, x, y):
    """用池化统计量 + 岭解出 β，并返回修正量 [β0+β1x+β2y] × 2。"""
    n = a[0]
    if n < MIN:
        return None
    A = np.array([[a[0], a[1], a[2]],
                  [a[1], a[3], a[4]],
                  [a[2], a[4], a[5]]])
    A = A + lam * n * np.eye(3)
    b1 = np.array([a[6], a[7], a[8]])
    b2 = np.array([a[9], a[10], a[11]])
    v = np.array([1.0, x, y])
    try:
        r1 = np.linalg.solve(A, b1)
        r2 = np.linalg.solve(A, b2)
    except np.linalg.LinAlgError:
        return None
    return np.array([v @ r1, v @ r2])


def run(G=4, lam=1e-3, radius=1, use_cond=True, oracle=False):
    errs = []
    for s, v in sess.items():
        grid = newacc()
        glob = newacc()
        for r in v:
            cond = r["C"] if use_cond else ""
            x, y = float(r["P"][0]), float(r["P"][1])
            i0 = min(G - 1, max(0, int(x * G)))
            j0 = min(G - 1, max(0, int(y * G)))
            est = None
            if radius == 1:
                est = solve(grid[(cond, i0, j0)], lam, x, y)
            else:
                pooled = np.zeros(NZ)
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        i, j = i0 + di, j0 + dj
                        if 0 <= i < G and 0 <= j < G:
                            pooled += grid[(cond, i, j)]
                est = solve(pooled, lam, x, y)
            if est is None:
                est = solve(glob[cond], lam, x, y)
            corr = r["P"] - est if est is not None else r["P"]
            errs.append(float(np.hypot((corr[0] - r["L"][0]) * r["S"][0],
                                       (corr[1] - r["L"][1]) * r["S"][1])))
            # ---- 更新（严格因果）----
            d = r["P"] - r["L"]
            kt = r["L"] if oracle else r["P"]
            kx, ky = float(kt[0]), float(kt[1])
            add(grid, (cond, min(G - 1, max(0, int(kx * G))),
                       min(G - 1, max(0, int(ky * G)))), kx, ky, d[0], d[1])
            add(glob, cond, kx, ky, d[0], d[1])
    return np.array(errs)


base = np.array([float(np.hypot((r["P"][0] - r["L"][0]) * r["S"][0],
                                (r["P"][1] - r["L"][1]) * r["S"][1]))
                 for s in sess for r in sess[s]])
print("=" * 100)
print(f"【误差场·容量空间化】{Path(CSV).stem}  基线 EM {base.mean():.3f}")
print("=" * 100)
print(f"  {'配置':<40}{'EM均值':>9}{'vs基线':>9}{'中位':>8}{'>3cm':>8}")
print(f"  {'A 不修正':<40}{base.mean():>9.3f}{0.0:>8.1f}%"
      f"{np.median(base):>8.3f}{100 * (base > 3).mean():>7.2f}%")
for lam in (1e9, 1.0, 0.1, 0.03, 0.01, 0.003):
    e = run(G=4, lam=lam, radius=1)
    tag = "λ=∞（=每格常数偏移，上轮配置）" if lam > 1e8 else f"λ={lam} 每格仿射"
    print(f"  {tag:<40}{e.mean():>9.3f}{100 * (e.mean() / base.mean() - 1):>8.1f}%"
          f"{np.median(e):>8.3f}{100 * (e > 3).mean():>7.2f}%")
for lam in (0.03, 0.01):
    e = run(G=4, lam=lam, radius=3)
    print(f"  {'λ=%g + 3x3 统计量池化' % lam:<40}{e.mean():>9.3f}"
          f"{100 * (e.mean() / base.mean() - 1):>8.1f}%{np.median(e):>8.3f}"
          f"{100 * (e > 3).mean():>7.2f}%")
for G in (2, 3, 6, 8):
    e = run(G=G, lam=0.03, radius=3)
    print(f"  {'G=%d + 3x3 + λ=0.03' % G:<40}{e.mean():>9.3f}"
          f"{100 * (e.mean() / base.mean() - 1):>8.1f}%{np.median(e):>8.3f}"
          f"{100 * (e > 3).mean():>7.2f}%")
e = run(G=4, lam=0.03, radius=3, oracle=True)
print(f"  {'oracle 键（上限参考）':<40}{e.mean():>9.3f}"
      f"{100 * (e.mean() / base.mean() - 1):>8.1f}%{np.median(e):>8.3f}"
      f"{100 * (e > 3).mean():>7.2f}%")
