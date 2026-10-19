"""逐帧追踪：把"记录 → 修正"两步在真实数据上打印出来。

选一个 session（默认 p11/walking），按时间顺序跑因果在线流程，并在若干帧上打印：
  当帧的 预测 / 真值 / 原始误差
  第一级（全局仿射）：它估的偏差 d̂、修正后的位置
  第二级（网格残差）：落在哪个格、3x3 块里有多少样本、残差均值、再修正后的位置
  最终输出与该帧的误差（对比原始）
并统计"有足够样本的格数"随时间的变化（解释启动阶段为何无效）。
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/xieliangcai/code/gazeformer-exps")
G, MIN, NZ = 8, 8, 12
TARGET = sys.argv[1] if len(sys.argv) > 1 else "p11/decoded/walking"

sc = {}
with open(ROOT / "data/RGBDGaze/index/index.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        sc[r["rgb_path"]] = (float(r["screen_w"]) * float(r["cm_px_x"]),
                             float(r["screen_h"]) * float(r["cm_px_y"]))

rows = []
for r in csv.DictReader(open(ROOT / "out/rgbdgaze/analysis/eye224_test.csv",
                             encoding="utf-8")):
    if TARGET in r["path"]:
        rows.append((int(r["frame_index"]), r["path"],
                     float(r["pred_x"]), float(r["pred_y"]),
                     float(r["label_x"]), float(r["label_y"])))
rows.sort()
print(f"会话 {TARGET}：{len(rows)} 帧（测试集）")


def cell(x, y):
    return min(G - 1, max(0, int(x * G))), min(G - 1, max(0, int(y * G)))


def cm(pred, lab, path):
    w, h = sc[path]
    return float(np.hypot((pred[0] - lab[0]) * w, (pred[1] - lab[1]) * h))


def add(st, x, y, dx, dy):
    st[0] += 1
    st[1] += x; st[2] += y
    st[3] += x * x; st[4] += x * y; st[5] += y * y
    st[6] += dx; st[7] += dx * x; st[8] += dx * y
    st[9] += dy; st[10] += dy * x; st[11] += dy * y


SHOW = {0, 8, 30, 60, 150, 400, 1000, 1500}
glob = np.zeros(NZ)
grid = defaultdict(lambda: [0.0, 0.0, 0.0])
print(f"\n{'t':>5}{'原始误差':>9}{'全局仿射估的偏差 d̂':>22}{'p1后误差':>9}"
      f"{'格(i,j)':>10}{'3x3样本':>8}{'格残差':>18}{'最终误差':>9}")
for t, path, px, py, lx, ly in rows:
    P, L = np.array([px, py]), np.array([lx, ly])
    x, y = float(P[0]), float(P[1])
    e_raw = cm(P, L, path)
    # ---- 第一级：全局仿射（用过去帧拟合 误差 ≈ β0+β1x+β2y）----
    bv = np.zeros(2)
    if glob[0] >= MIN:
        A = np.array([[glob[0], glob[1], glob[2]], [glob[1], glob[3], glob[4]],
                      [glob[2], glob[4], glob[5]]]) \
            + 0.01 * glob[0] * np.diag([0.0, 1.0, 1.0])
        try:
            b1 = np.linalg.solve(A, glob[6:9])
            b2 = np.linalg.solve(A, np.array([glob[9], glob[10], glob[11]]))
            v = np.array([1.0, x, y])
            bv = np.array([v @ b1, v @ b2])
        except np.linalg.LinAlgError:
            bv = np.zeros(2)
    p1 = P - bv
    e1 = cm(p1, L, path)
    # ---- 第二级：网格残差（3x3 池化）----
    i0, j0 = cell(float(p1[0]), float(p1[1]))
    cnt = sx = sy = 0.0
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            i, j = i0 + di, j0 + dj
            if 0 <= i < G and 0 <= j < G:
                c = grid[(i, j)]
                cnt += c[0]; sx += c[1]; sy += c[2]
    if cnt >= MIN:
        res = np.array([sx / cnt, sy / cnt])
        p2 = p1 - res
    else:
        res = np.zeros(2)
        p2 = p1
    e_fin = cm(p2, L, path)
    if t in SHOW:
        print(f"{t:>5}{e_raw:>9.3f}{f'({bv[0]:+.3f},{bv[1]:+.3f})':>22}{e1:>9.3f}"
              f"{f'({i0},{j0})':>10}{int(cnt):>8}"
              f"{f'({res[0]:+.3f},{res[1]:+.3f})':>18}{e_fin:>9.3f}")
    # ---- 更新（在修正之后）----
    add(glob, x, y, float(P[0] - L[0]), float(P[1] - L[1]))
    c = grid[(min(G - 1, max(0, int(float(p1[0]) * G))),
              min(G - 1, max(0, int(float(p1[1]) * G))))]
    c[0] += 1
    c[1] += float(P[0] - L[0]) - float(bv[0])
    c[2] += float(P[1] - L[1]) - float(bv[1])
print("\n（d̂ = 模型在该位置的系统偏差估计；p1 = 预测 − d̂；最终 = p1 − 该格残差）")
