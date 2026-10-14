"""误差场可部署化·第二轮：网格粗细 × 邻域平滑 × 免费条件键（朝向/活动）。

动机：B（键=预测,查=预测）在 RGBDGaze 有效（−15.8%）但 GC 失效（−1.2%）✗，
因为 GC 的归一化预测误差（~0.35-0.6）远大于格宽（1/G）⇒ 查询必然落错格。

三条对策（本实验逐条 + 组合）：
  ① 加粗网格 G=3/2（格宽 0.33/0.5）
  ② 3x3 邻域加权平滑（把有效作用半径放大）
  ③ 免费条件键：把表按（朝向/活动 × 格）建 —— 传感器已知、零成本，且让同组更同质

回退层级：本格(≥MIN) → 3x3 邻域 → 条件全局均值 → 不修正。
严格因果：先用后更新（第 t 帧的标签在第 t 帧修正之后才进入表）。
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/xieliangcai/code/gazeformer-exps")
CSV = sys.argv[1]
MIN = 5

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
             C=str(r.get("activity") or ""),
             S=sc[r["path"]]))
for s in sess:
    sess[s].sort(key=lambda r: r["fi"])


def run(G=6, smooth=1, use_cond=False, oracle=False):
    """表: (cond, i, j) -> [n, sx, sy]；另有 cond -> [n, sx, sy] 作为全局回退。"""
    errs = []
    for s, v in sess.items():
        cell = defaultdict(lambda: [0, 0.0, 0.0])
        glob = defaultdict(lambda: [0, 0.0, 0.0])
        for r in v:
            def cidx(pt):
                return (min(G - 1, max(0, int(pt[0] * G))),
                        min(G - 1, max(0, int(pt[1] * G))))
            cond = r["C"] if use_cond else ""
            q = r["P"]
            i0, j0 = cidx(q)
            est = None
            # ① 本格
            k = (cond, i0, j0)
            if cell[k][0] >= MIN:
                est = [cell[k][1] / cell[k][0], cell[k][2] / cell[k][0]]
            elif smooth == 3:
                # ② 3x3 邻域加权
                sx = sy = sw = 0.0
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        i, j = i0 + di, j0 + dj
                        if 0 <= i < G and 0 <= j < G:
                            kk = (cond, i, j)
                            if cell[kk][0] > 0:
                                w = 1.0 / (1 + abs(di) + abs(dj))
                                sx += w * (cell[kk][1] / cell[kk][0])
                                sy += w * (cell[kk][2] / cell[kk][0])
                                sw += w
                if sw > 0:
                    est = [sx / sw, sy / sw]
            if est is None and glob[cond][0] >= MIN:
                # ③ 条件全局均值
                est = [glob[cond][1] / glob[cond][0], glob[cond][2] / glob[cond][0]]
            corr = r["P"] if est is None else r["P"] - np.array(est)
            e = float(np.hypot((corr[0] - r["L"][0]) * r["S"][0],
                               (corr[1] - r["L"][1]) * r["S"][1]))
            errs.append(e)
            # ---- 更新（严格因果：修正之后）----
            d = r["P"] - r["L"]
            kt = r["L"] if oracle else r["P"]          # 建表用的位置
            i2, j2 = cidx(kt)
            kk = (cond, i2, j2)
            cell[kk][0] += 1
            cell[kk][1] += d[0]
            cell[kk][2] += d[1]
            glob[cond][0] += 1
            glob[cond][1] += d[0]
            glob[cond][2] += d[1]
    return np.array(errs)


base = None
print("=" * 108)
print(f"【误差场·第二轮】{Path(CSV).stem}  回退: 本格(≥{MIN}) → 3x3 → 条件全局 → 不修正")
print("=" * 108)
print(f"  {'配置':<40}{'EM均值':>9}{'vs基线':>9}{'中位':>8}{'>3cm':>8}")
configs = [
    ("A 不修正", dict(G=6, smooth=1)),
    ("B 键=预测 G=6 无平滑（上轮）", dict(G=6, smooth=1)),
    ("G=6 + 3x3 平滑", dict(G=6, smooth=3)),
    ("G=4 + 3x3 平滑", dict(G=4, smooth=3)),
    ("G=3 + 3x3 平滑", dict(G=3, smooth=3)),
    ("G=3 无平滑", dict(G=3, smooth=1)),
    ("G=2 + 3x3 平滑", dict(G=2, smooth=3)),
    ("G=6 + 条件键", dict(G=6, smooth=1, use_cond=True)),
    ("G=4 + 条件键 + 3x3", dict(G=4, smooth=3, use_cond=True)),
    ("G=3 + 条件键 + 3x3", dict(G=3, smooth=3, use_cond=True)),
    ("E 键=真值 G=6 键不匹配（参考）", dict(G=6, smooth=1, oracle=True)),
]
for tag, kw in configs:
    if tag.startswith("A "):
        e = run(G=6, smooth=1) if False else np.array(
            [float(np.hypot((r["P"][0] - r["L"][0]) * r["S"][0],
                            (r["P"][1] - r["L"][1]) * r["S"][1]))
             for s in sess for r in sess[s]])
        base = e
    elif tag.startswith("B "):
        e = base  # 上轮已测，此处只作行标
        continue
    else:
        e = run(**kw)
    print(f"  {tag:<40}{e.mean():>9.3f}{100 * (e.mean() / base.mean() - 1):>8.1f}%"
          f"{np.median(e):>8.3f}{100 * (e > 3).mean():>7.2f}%")
