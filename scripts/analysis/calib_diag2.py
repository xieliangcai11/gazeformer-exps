"""诊断：1.085/1.093 剩下的误差质量分布在哪？（决定下一步该打哪一块）

输出四张分解表（都用同一估计器：8x8 格 + 3x3 硬盒 + 仿射 + 因果在线）：
  · 按「标签角速度」分箱（cm/s）——含**误差质量占比** Σerr/Σerr（关键：谁占大头）
  · 按「离屏心距离」分箱
  · 按被试 / 按 activity
  · 真·两步离线上限（先用整段含未来标签建场，再评估）⇒ 因果滞后还有多少点可回收
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
             act=r["activity"],
             P=np.array([float(r["pred_x"]), float(r["pred_y"])]),
             L=np.array([float(r["label_x"]), float(r["label_y"])])))
for s in sess:
    sess[s].sort(key=lambda r: r["fi"])
SESS = sorted(sess, key=lambda s: (s.split("/")[-4], s))


def cidx(pt):
    return min(G - 1, max(0, int(pt[0] * G))) * G + min(G - 1, max(0, int(pt[1] * G)))


def emv(pred, lab, path):
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


def lookup(grid, key):
    c = cidx(key); i0, j0 = c // G, c % G
    acc = np.zeros(2); cnt = 0.0
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            ii, jj = i0 + di, j0 + dj
            if 0 <= ii < G and 0 <= jj < G:
                cc = grid[ii * G + jj]
                cnt += cc[0]; acc += cc[1:]
    return acc / cnt if cnt >= MIN else None


def correct(grid, g, P):
    b = affine_of(g)
    bv = np.array([np.array([1.0, P[0], P[1]]) @ b[0],
                   np.array([1.0, P[0], P[1]]) @ b[1]]) if b else np.zeros(2)
    p1 = P - bv
    dd = lookup(grid, p1)
    return (p1 - dd) if dd is not None else p1, p1, bv


rows = []          # (err_raw, err_corr, speed_cm_s, radius_cm, subj, act, sess)
off = []
FPS = 30.0
for s in SESS:
    v = sess[s]
    g, grid = np.zeros(NZ), defaultdict(lambda: [0.0, 0.0, 0.0])
    prev_L = None
    for r in v:
        P, L = r["P"], r["L"]
        w, h = sc[r["path"]]
        corr, p1, bv = correct(grid, g, P)
        spd = 0.0 if prev_L is None else float(np.hypot((L[0] - prev_L[0]) * w,
                                                       (L[1] - prev_L[1]) * h)) * FPS
        rad = float(np.hypot((L[0] - 0.5) * w, (L[1] - 0.5) * h))
        rows.append((emv(P, L, r["path"]), emv(corr, L, r["path"]), spd, rad,
                     r["subj"], r["act"], s))
        prev_L = L
        upd(g, grid, P, L, p1, bv)

# 真·两步离线上限：先整段建场，再整段评估
for s in SESS:
    v = sess[s]
    g, grid = np.zeros(NZ), defaultdict(lambda: [0.0, 0.0, 0.0])
    for r in v:
        _, p1, bv = correct(grid, g, r["P"])
        upd(g, grid, r["P"], r["L"], p1, bv)
    for r in v:
        corr, _, _ = correct(grid, g, r["P"])
        off.append(emv(corr, r["L"], r["path"]))

E = np.array([x[0] for x in rows]); C = np.array([x[1] for x in rows])
spd = np.array([x[2] for x in rows]); rad = np.array([x[3] for x in rows])

print("=" * 108)
print(f"总体：不修正 {E.mean():.3f} → 场修正 {C.mean():.3f}（{100 * (C.mean() / E.mean() - 1):+.1f}%）")
print(f"真·两步离线上限（含未来标签）：{np.mean(off):.3f}（{100 * (np.mean(off) / E.mean() - 1):+.1f}%）")
print(f"⇒ 因果滞后剩余可回收：{100 * (C.mean() / np.mean(off) - 1):+.1f}%")
print("=" * 108)


def table(name, key, bins):
    print(f"\n  {name}")
    print(f"    {'区间':>16}{'n':>8}{'占比':>8}{'原始EM':>9}{'修正EM':>9}{'增益':>8}"
          f"{'误差质量占比':>12}")
    idx = np.digitize(key, bins)
    tot = (E - 0).sum()
    for b in range(len(bins) + 1):
        m = idx == b
        if m.sum() == 0:
            continue
        lo = "-∞" if b == 0 else f"{bins[b - 1]:.2f}"
        hi = "+∞" if b == len(bins) else f"{bins[b]:.2f}"
        raw = E[m - 0][m].mean() if False else E[m].mean()
        print(f"    {lo+'~'+hi:>16}{int(m.sum()):>8}{100 * m.mean():>7.1f}%"
              f"{raw:>9.3f}{C[m].mean():>9.3f}{100 * (C[m].mean() / raw - 1):>+7.1f}%"
              f"{100 * C[m].sum() / C.sum():>11.1f}%")


table("按「标签角速度」(cm/s)：误差质量是否集中在快速运动？",
      spd, [2.0, 5.0, 10.0, 20.0])
table("按「离屏心距离」(cm)：质量是否集中在屏幕边缘？",
      rad, [1.0, 2.0, 3.0, 4.5])

print("\n  按被试 / activity")
print(f"    {'组':>12}{'n':>8}{'原始EM':>9}{'修正EM':>9}{'增益':>8}")
for field_i, nm in ((4, "subject"), (5, "activity")):
    keys = [x[field_i] for x in rows]
    for k in sorted(set(keys)):
        m = np.array([x == k for x in keys])
        print(f"    {str(k):>12}{int(m.sum()):>8}{E[m].mean():>9.3f}{C[m].mean():>9.3f}"
              f"{100 * (C[m].mean() / E[m].mean() - 1):>+7.1f}%")
