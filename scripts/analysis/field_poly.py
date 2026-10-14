"""误差场·高阶平滑表示（因果）：把"场"从"每格偏移/仿射"升级为**全局平滑多项式场**。

动机（这是我此前漏测的一类表示 ✗）：
  已知：误差是注视位置的平滑函数 ✓（离线留一 4x4 格解释 36%；跨半段相关 +0.86 ✓）
  已测：每格偏移/每格仿射/全局仿射 + 每格残差 ⇒ 全部收在 −24~−26% ✗
  未测：**全局高阶多项式场**（deg 2/3，6/10 项）⇒ 无覆盖度/预热问题 ✓✓
        （参数少、全局支撑 ⇒ 冷启动也不至于无效 ✓）

本脚本：
  第 1 层：全局多项式场 β（deg=1/2/3），只用本 session **过去**帧的矩累积、
          每 K 帧重解一次（模拟 App 周期性重拟合 ✓），岭只罚非常数项 ✓
  第 2 层（可选）：用校正后位置查"每格残差偏移"（3x3 池化 ✓）
并输出**预热分桶**（按 session 内已积累帧数）⇒ 量化"因果 vs 离线"的缺口里有多少来自冷启动 ✓
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/xieliangcai/code/gazeformer-exps")
CSV = sys.argv[1]
K_REFIT = 30     # 每 K 帧重解一次（模拟 App 周期重拟合）
MIN_CELL = 8


def basis(x, y, deg):
    if deg == 1:
        return np.array([1.0, x, y])
    if deg == 2:
        return np.array([1.0, x, y, x * x, x * y, y * y])
    return np.array([1.0, x, y, x * x, x * y, y * y,
                     x ** 3, x * x * y, x * y * y, y ** 3])


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
             S=sc[r["path"]]))
for s in sess:
    sess[s].sort(key=lambda r: r["fi"])


def run(deg=2, lam=1e-3, G=8, use_cell=True, radius=3):
    errs, elapsed = [], []
    d = len(basis(0.0, 0.0, deg))
    for s, v in sess.items():
        M = np.zeros((d, d))          # Σ bb^T
        V = np.zeros((d, 2))          # Σ b·(pred-label)
        beta = None
        grid = defaultdict(lambda: [0, 0.0, 0.0])
        for t, r in enumerate(v):
            P, L = r["P"], r["L"]
            x, y = float(P[0]), float(P[1])
            if t % K_REFIT == 0 and M[0, 0] >= 20:
                A = M + lam * M[0, 0] * np.diag([0.0] + [1.0] * (d - 1))
                try:
                    beta = np.linalg.solve(A, V)
                except np.linalg.LinAlgError:
                    beta = None
            g = basis(x, y, deg) @ beta if beta is not None else np.zeros(2)
            p1 = P - g
            est_cell = None
            if use_cell:
                cx, cy = float(p1[0]), float(p1[1])
                i0 = min(G - 1, max(0, int(cx * G)))
                j0 = min(G - 1, max(0, int(cy * G)))
                cnt = sx = sy = 0.0
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        i, j = i0 + di, j0 + dj
                        if 0 <= i < G and 0 <= j < G:
                            c = grid[(i, j)]
                            cnt += c[0]; sx += c[1]; sy += c[2]
                if cnt >= MIN_CELL:
                    est_cell = np.array([sx / cnt, sy / cnt])
            corr = p1 - est_cell if est_cell is not None else p1
            errs.append(float(np.hypot((corr[0] - L[0]) * r["S"][0],
                                       (corr[1] - L[1]) * r["S"][1])))
            elapsed.append(t)
            # ---- 因果更新 ----
            b = basis(x, y, deg)
            M += np.outer(b, b)
            V += np.outer(b, P - L)
            if use_cell:
                res = (P - L) - g
                c = grid[(min(G - 1, max(0, int(float(p1[0]) * G))),
                          min(G - 1, max(0, int(float(p1[1]) * G))))]
                c[0] += 1
                c[1] += float(res[0])
                c[2] += float(res[1])
    return np.array(errs), np.array(elapsed)


base = np.array([float(np.hypot((r["P"][0] - r["L"][0]) * r["S"][0],
                                (r["P"][1] - r["L"][1]) * r["S"][1]))
                 for s in sess for r in sess[s]])
el0 = np.array([t for s in sess for t in range(len(sess[s]))])
print("=" * 104)
print(f"【高阶平滑误差场】{Path(CSV).stem}  基线 EM {base.mean():.3f}"
      f"  中位 {np.median(base):.3f}  >3cm {100 * (base > 3).mean():.2f}%")
print("=" * 104)
print(f"  {'配置':<42}{'EM均值':>9}{'vs基线':>9}{'中位':>8}{'>3cm':>8}")
print(f"  {'A 不修正':<42}{base.mean():>9.3f}{0.0:>8.1f}%"
      f"{np.median(base):>8.3f}{100 * (base > 3).mean():>7.2f}%")
best = None
for deg, lam in ((1, 1e-3), (2, 1e-3), (3, 1e-3), (2, 1e-4), (3, 1e-4)):
    for uc in (False, True):
        e, el = run(deg=deg, lam=lam, use_cell=uc)
        tag = f"deg={deg} λ={lam:g} " + ("+每格残差偏移" if uc else "（仅全局场）")
        print(f"  {tag:<42}{e.mean():>9.3f}{100 * (e.mean() / base.mean() - 1):>8.1f}%"
              f"{np.median(e):>8.3f}{100 * (e > 3).mean():>7.2f}%")
        if best is None or e.mean() < best[0].mean():
            best = (e, el, tag)
print(f"\n  【预热分桶】最优配置 = {best[2]}")
print(f"  {'session 内已积累帧数':<24}" + "".join(f"{k:>16}" for k in
                                              ("不修正", "本方法", "相对基线")))
for a, b in ((0, 200), (200, 600), (600, 1500), (1500, 10 ** 9)):
    m = (el0 >= a) & (el0 < b)
    mb = (best[1] >= a) & (best[1] < b)
    if m.sum() == 0:
        continue
    bs = base[m].mean()
    ms = best[0][mb].mean() if mb.sum() else float("nan")
    hi = "inf" if b >= 10 ** 9 else str(b)
    print(f"  {f'{a}-{hi}':<24}{bs:>16.3f}{ms:>16.3f}"
          f"{100 * (ms / bs - 1):>15.1f}%")
