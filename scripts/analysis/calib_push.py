"""冲击目标：把 1.093 压下去。

三条候选（各自独立、可组合）：
  ① 跨会话持久化 + 遗忘：被试的场/仿射在 activity 会话之间延续，边界处按 ρ 折扣
     （动机：场每会话从零开始 ⇒ 首段（误差最差的一段，EM≈1.68）完全没有修正；且
       场/仿射本身对被试是持久的，D1b 已证「同被试尺度可迁移 −5.4~−44.1%」✓）
  ② 核加权局部估计：把 3x3 硬盒平均换成高斯核（带宽 σ，单位=格）⇒ 消除盒平均/分箱偏差
  ③ 离线上限：场用「全 session（含未来）标签」一次性构建 ⇒ 给因果在线提供可达上界

自检：ρ=0 + 硬盒 必须精确复现 1.093（会话口径）✓
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/xieliangcai/code/gazeformer-exps")
CSV = ROOT / "out/rgbdgaze/analysis/eye224_test.csv"
G, MIN, NZ = 8, 8, 12

sc = {}
with open(ROOT / "data/RGBDGaze/index/index.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        sc[r["rgb_path"]] = (float(r["screen_w"]) * float(r["cm_px_x"]),
                             float(r["screen_h"]) * float(r["cm_px_y"]))

sess = defaultdict(list)
for r in csv.DictReader(open(CSV, encoding="utf-8")):
    sess[r["path"].rsplit("/", 1)[0]].append(
        dict(fi=int(r["frame_index"]), path=r["path"], subj=r["subject"],
             P=np.array([float(r["pred_x"]), float(r["pred_y"])]),
             L=np.array([float(r["label_x"]), float(r["label_y"])])))


def subj_of(s):
    return s.split("/")[-4] if s.count("/") >= 4 else s


# 被试内会话按「清单出现顺序」视为时间序（dataset manifest 顺序）
order_by_subj = defaultdict(list)
for s in sess:                                  # dict 保序 = 清单顺序
    order_by_subj[subj_of(s)].append(s)
for s in sess:
    sess[s].sort(key=lambda r: r["fi"])
SUBJ = sorted(order_by_subj)


def cidx(pt):
    return min(G - 1, max(0, int(pt[0] * G))) * G + min(G - 1, max(0, int(pt[1] * G)))


def em(pred, lab, path):
    w, h = sc[path]
    return float(np.hypot((pred[0] - lab[0]) * w, (pred[1] - lab[1]) * h))


def affine_of(g):
    if g[0] < MIN:
        return None
    A = np.array([[g[0], g[1], g[2]], [g[1], g[3] + 0.01 * g[0], g[4]],
                  [g[2], g[4], g[5] + 0.01 * g[0]]])
    try:
        return (np.linalg.solve(A, g[6:9]),
                np.linalg.solve(A, np.array([g[9], g[10], g[11]])))
    except np.linalg.LinAlgError:
        return None


def upd(g, grid, base, lab, p1, bv):
    x, y = float(base[0]), float(base[1])
    g[0] += 1
    g[1] += x; g[2] += y
    g[3] += x * x; g[4] += x * y; g[5] += y * y
    dx, dy = base[0] - lab[0], base[1] - lab[1]
    g[6] += dx; g[7] += dx * x; g[8] += dx * y
    g[9] += dy; g[10] += dy * x; g[11] += dy * y
    c = grid[cidx(p1)]
    c[0] += 1.0
    c[1] += float(dx) - float(bv[0])
    c[2] += float(dy) - float(bv[1])


def est(grid, key, sigma=None, rad=1):
    """σ=None ⇒ 3x3 硬盒；否则高斯核加权（单位=格）。"""
    c = cidx(key); i0, j0 = c // G, c % G
    acc = np.zeros(2); cnt = 0.0
    for di in range(-rad, rad + 1):
        for dj in range(-rad, rad + 1):
            ii, jj = i0 + di, j0 + dj
            if not (0 <= ii < G and 0 <= jj < G):
                continue
            cc = grid[ii * G + jj]
            if cc[0] <= 0:
                continue
            w = 1.0 if sigma is None else float(np.exp(-(di * di + dj * dj) / (2 * sigma ** 2)))
            acc += w * np.array([cc[1], cc[2]])
            cnt += w * cc[0]
    if cnt < MIN:
        return None
    return acc / cnt


def discount(g, grid, rho):
    g *= rho
    for k in list(grid):
        grid[k][0] *= rho
        grid[k][1] *= rho
        grid[k][2] *= rho


def run(rho=0.0, sigma=None, rad=1, offline=False, per_subject=False):
    errs, per = [], defaultdict(list)
    for sj in SUBJ:
        g, grid = np.zeros(NZ), defaultdict(lambda: [0.0, 0.0, 0.0])
        for s in order_by_subj[sj]:
            v = sess[s]
            if offline:                     # 用整段（含未来）一次性构建，再评估
                for r in v:
                    b = affine_of(g)
                    bv = np.array([np.array([1.0, r["P"][0], r["P"][1]]) @ b[0],
                                   np.array([1.0, r["P"][0], r["P"][1]]) @ b[1]]) if b else np.zeros(2)
                    p1 = r["P"] - bv
                    e = est(grid, p1, sigma, rad)
                    corr = p1 - e if e is not None else p1
                    errs.append(em(corr, r["L"], r["path"]))
                    per[sj].append(errs[-1])
                    upd(g, grid, r["P"], r["L"], p1, bv)
                discount(g, grid, rho)
                continue
            for r in v:
                b = affine_of(g)
                bv = np.array([np.array([1.0, r["P"][0], r["P"][1]]) @ b[0],
                               np.array([1.0, r["P"][0], r["P"][1]]) @ b[1]]) if b else np.zeros(2)
                p1 = r["P"] - bv
                e = est(grid, p1, sigma, rad)
                corr = p1 - e if e is not None else p1
                errs.append(em(corr, r["L"], r["path"]))
                per[sj].append(errs[-1])
                upd(g, grid, r["P"], r["L"], p1, bv)
            discount(g, grid, rho)          # 会话边界：按 ρ 保留历史
    e = np.array(errs)
    if per_subject:
        return e, {k: float(np.mean(v)) for k, v in per.items()}
    return e


base = np.array([em(r["P"], r["L"], r["path"]) for s in sess for r in sess[s]])
print("=" * 112)
print(f"基线（不标定）EM {base.mean():.3f}；目标：低于 1.093（−25.7%）")
print("=" * 112)


def show(tag, e, extra=""):
    print(f"  {tag:<34}{e.mean():>9.3f}{100 * (e.mean() / base.mean() - 1):>9.1f}%"
          f"{100 * (e > 2).mean():>8.2f}%{100 * (e > 3).mean():>8.2f}%"
          f"{100 * (e > 5).mean():>8.2f}%{e.max():>8.2f}  {extra}")


print(f"  {'方案':<34}{'EM均值':>9}{'vs基线':>9}{'>2cm':>9}{'>3cm':>9}{'>5cm':>9}{'max':>8}")
show("A 不修正", base)
e0 = run(rho=0.0)
show("V1 因果在线（自检应=1.093）", e0)
show("离线上限（含未来标签）", run(offline=True))
print()
print("  ① 跨会话持久化（遗忘因子 ρ）")
for rho in (0.3, 0.5, 0.7, 0.9, 1.0):
    show(f"  ρ={rho}", run(rho=rho))
print()
print("  ② 核加权局部估计（带宽 σ 格；rad=1）")
for sigma in (0.6, 0.9, 1.2, 2.0):
    show(f"  σ={sigma}", run(rho=0.0, sigma=sigma))
print("  ②b 核加权 + 更大邻域 rad=2")
for sigma in (1.2, 2.0, 3.0):
    show(f"  rad=2 σ={sigma}", run(rho=0.0, sigma=sigma, rad=2))
