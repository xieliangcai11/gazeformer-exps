"""误差场的"可拿上界"与缺口分解（关键实验）。

三档对照，全部以**预测注视点**为索引键（与测试时可观测一致 ✓）：
  A 不修正
  B 离线场：只用 session **前 R 比例**帧拟合（无冷启动 ✓、无未来泄漏 ✗），在后半段评估
      ⇒ 这是"一个拟合良好的场"能拿到多少 = **可拿上界** ✓✓
  C 因果在线场（= 现有方法，含冷启动 ✗）在同一批评估帧上重算
若 B ≫ C ⇒ 缺口来自**冷启动/估计质量** ⇒ 可用"开局先校准 N 秒"直接补上 ✓✓
若 B ≈ C ⇒ 确实封顶 ✓
另给 D：用**整段**拟合的场（含未来 ⇒ 上界的上界，仅参考 ✗）
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/xieliangcai/code/gazeformer-exps")
CSV = sys.argv[1]
G, MIN = 8, 8

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


def cellg(pt):
    return (min(G - 1, max(0, int(pt[0] * G))), min(G - 1, max(0, int(pt[1] * G))))


def lookup(grid, pt):
    i0, j0 = cellg(pt)
    cnt = sx = sy = 0.0
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            i, j = i0 + di, j0 + dj
            if 0 <= i < G and 0 <= j < G:
                c = grid.get((i, j))
                if c:
                    cnt += c[0]; sx += c[1]; sy += c[2]
    if cnt < MIN:
        return None
    return np.array([sx / cnt, sy / cnt])


def em(v, idxs, corrector):
    out = []
    for i in idxs:
        r = v[i]
        c = corrector(r)
        out.append(float(np.hypot((c[0] - r["L"][0]) * r["S"][0],
                                  (c[1] - r["L"][1]) * r["S"][1])))
    return np.array(out)


print("=" * 100)
print(f"【可拿上界与缺口分解】{Path(CSV).stem}")
print("=" * 100)
print(f"  {'协议':<34}{'评估帧':>8}{'EM均值':>9}{'vs基线':>9}{'中位':>8}")
for R in (0.3, 0.5):
    for tag_mode in ("offline_fit_only",):
        # 评估集：后半段
        ev_idx = {}
        for s, v in sess.items():
            k = max(1, int(len(v) * R))
            ev_idx[s] = list(range(k, len(v)))
        base_e, off_e, caus_e, full_e = [], [], [], []
        for s, v in sess.items():
            k = max(1, int(len(v) * R))
            cal = list(range(k))
            ev = ev_idx[s]
            if not ev:
                continue
            # B 离线场：用 cal 全部帧拟合（含 cal 内的未来 ⇒ 但那是"拟合期"，无泄漏到评估期 ✓）
            grid = defaultdict(lambda: [0, 0.0, 0.0])
            for i in cal:
                r = v[i]
                c = grid[cellg(r["P"])]
                c[0] += 1
                c[1] += float(r["P"][0] - r["L"][0])
                c[2] += float(r["P"][1] - r["L"][1])
            # D 整段拟合（仅参考 ✗）
            gridf = defaultdict(lambda: [0, 0.0, 0.0])
            for i in range(len(v)):
                r = v[i]
                c = gridf[cellg(r["P"])]
                c[0] += 1
                c[1] += float(r["P"][0] - r["L"][0])
                c[2] += float(r["P"][1] - r["L"][1])
            # C 因果在线（在评估帧上逐步累积）
            grid_c = defaultdict(lambda: [0, 0.0, 0.0])
            for i in range(len(v)):
                r = v[i]
                if i in ev:
                    e = lookup(grid_c, r["P"])
                    c = r["P"] - e if e is not None else r["P"]
                    base_e.append(float(np.hypot((r["P"][0] - r["L"][0]) * r["S"][0],
                                                 (r["P"][1] - r["L"][1]) * r["S"][1])))
                    caus_e.append(float(np.hypot((c[0] - r["L"][0]) * r["S"][0],
                                                 (c[1] - r["L"][1]) * r["S"][1])))
                    e2 = lookup(grid, r["P"])
                    c2 = r["P"] - e2 if e2 is not None else r["P"]
                    off_e.append(float(np.hypot((c2[0] - r["L"][0]) * r["S"][0],
                                                (c2[1] - r["L"][1]) * r["S"][1])))
                    e3 = lookup(gridf, r["P"])
                    c3 = r["P"] - e3 if e3 is not None else r["P"]
                    full_e.append(float(np.hypot((c3[0] - r["L"][0]) * r["S"][0],
                                                 (c3[1] - r["L"][1]) * r["S"][1])))
                cg = grid_c[cellg(r["P"])]
                cg[0] += 1
                cg[1] += float(r["P"][0] - r["L"][0])
                cg[2] += float(r["P"][1] - r["L"][1])
        base_e = np.array(base_e)
        for tag, arr in (("A 不修正", base_e),
                         (f"B 离线场（前 {R*100:.0f}% 拟合）→ 后半段", np.array(off_e)),
                         ("C 因果在线场（含冷启动）", np.array(caus_e)),
                         ("D 整段拟合（参考 ✗ 含泄漏）", np.array(full_e))):
            print(f"  {tag:<34}{len(arr):>8}{arr.mean():>9.3f}"
                  f"{100 * (arr.mean() / base_e.mean() - 1):>8.1f}%"
                  f"{np.median(arr):>8.3f}")
        print()
