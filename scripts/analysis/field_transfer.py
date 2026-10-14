"""误差场**跨被试迁移**测试（大规模）：用训练被试建立误差场，在未见被试上评估。

背景：此前 LOSO（仅 4 个被试）显示"场不可迁移"✗，但那很可能是规模问题
（文献在千级被试上迁移成功 ✓）。GazeCapture 有 1263 个训练被试（抽样 10.4 万帧）
与 150 个独立测试被试 ⇒ 这是文献标准的检验。

对比三档：
  A 不修正
  T 迁移场：由**训练被试**建立（键=预测注视点格；可选叠加朝向键），查表修正测试帧
  S 自索引：由**本被试自身历史**建立（= 上一轮结果，供对照）

用法: python field_transfer.py <训练CSV> <测试CSV>
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/xieliangcai/code/gazeformer-exps")
TRAIN_CSV, TEST_CSV = sys.argv[1], sys.argv[2]
MIN = 5

sc = {}
for _idx in (ROOT / "data/GazeCapture/index", ROOT / "data/RGBDGaze/index"):
    if not _idx.exists():
        continue
    with open(_idx / "index.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            pk = "rgb_path" if "rgb_path" in r else "frame_path"
            sc[r[pk]] = (float(r["screen_w"]) * float(r["cm_px_x"]),
                         float(r["screen_h"]) * float(r["cm_px_y"]))


def load(p):
    out = []
    for r in csv.DictReader(open(p, encoding="utf-8")):
        if r["path"] not in sc:
            continue
        out.append(dict(sub=r["subject"], cond=str(r.get("activity") or ""),
                        P=np.array([float(r["pred_x"]), float(r["pred_y"])]),
                        L=np.array([float(r["label_x"]), float(r["label_y"])]),
                        S=sc[r["path"]]))
    return out


train, test = load(TRAIN_CSV), load(TEST_CSV)
print("=" * 104)
print(f"【误差场跨被试迁移】训练 {len(train)} 帧 / {len({r['sub'] for r in train})} 被试"
      f"  测试 {len(test)} 帧 / {len({r['sub'] for r in test})} 被试")
print("=" * 104)


def cell(v, G):
    return (min(G - 1, max(0, int(v[0] * G))), min(G - 1, max(0, int(v[1] * G))))


def build(G, use_cond):
    tab = defaultdict(lambda: [0, 0.0, 0.0])
    for r in train:
        c = (r["cond"] if use_cond else "",) + cell(r["P"], G)
        d = r["P"] - r["L"]
        tab[c][0] += 1
        tab[c][1] += d[0]
        tab[c][2] += d[1]
    return tab


def apply_field(G, smooth, use_cond):
    tab = build(G, use_cond)
    errs, hit = [], 0
    for r in test:
        cond = r["cond"] if use_cond else ""
        i0, j0 = cell(r["P"], G)
        est, wsum = None, 0.0
        if tab[(cond, i0, j0)][0] >= MIN:
            t = tab[(cond, i0, j0)]
            est = [t[1] / t[0], t[2] / t[0]]
            hit += 1
        elif smooth == 3:
            sx = sy = sw = 0.0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    i, j = i0 + di, j0 + dj
                    if 0 <= i < G and 0 <= j < G:
                        t = tab[(cond, i, j)]
                        if t[0] > 0:
                            w = 1.0 / (1 + abs(di) + abs(dj))
                            sx += w * (t[1] / t[0])
                            sy += w * (t[2] / t[0])
                            sw += w
            if sw > 0:
                est = [sx / sw, sy / sw]
                hit += 1
        corr = r["P"] if est is None else r["P"] - np.array(est)
        errs.append(float(np.hypot((corr[0] - r["L"][0]) * r["S"][0],
                                   (corr[1] - r["L"][1]) * r["S"][1])))
    return np.array(errs), hit / len(test)


base = np.array([float(np.hypot((r["P"][0] - r["L"][0]) * r["S"][0],
                                (r["P"][1] - r["L"][1]) * r["S"][1])) for r in test])
print(f"  A 不修正（测试集）: EM {base.mean():.3f}  中位 {np.median(base):.3f}"
      f"  >3cm {100 * (base > 3).mean():.2f}%")
print(f"\n  {'迁移场配置':<34}{'EM均值':>9}{'vs基线':>9}{'中位':>8}{'>3cm':>8}{'覆盖率':>8}")
for G in (4, 6, 12, 24):
    for sm in (1, 3):
        e, hv = apply_field(G, sm, False)
        print(f"  {'G=%d 平滑=%s' % (G, sm if sm == 3 else '关'):<34}"
              f"{e.mean():>9.3f}{100 * (e.mean() / base.mean() - 1):>8.1f}%"
              f"{np.median(e):>8.3f}{100 * (e > 3).mean():>7.2f}%{hv * 100:>7.1f}%")
for G in (6, 12):
    e, hv = apply_field(G, 3, True)
    print(f"  {'G=%d 平滑=3 + 朝向键' % G:<34}{e.mean():>9.3f}"
          f"{100 * (e.mean() / base.mean() - 1):>8.1f}%{np.median(e):>8.3f}"
          f"{100 * (e > 3).mean():>7.2f}%{hv * 100:>7.1f}%")
print("\n  对照：自身历史自索引（上一轮）= −3.2% ✗；oracle 上界（键=查=真值）= −76.7% ✓✓")
