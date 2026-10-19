"""诚实性定稿：把「标签流外推」这个平凡来源彻底分离，并给出标签连续性证据。

问题：在「有逐帧真值流」的 regime 下，任何因果在线方法都能逼近"复制上一帧"，甚至更好
      （注视=真值近乎常量；扫视=一步跳变 ⇒ 一步外推即可预测大多数帧）。
⇒ 必须量化：① 平凡因果基线（上一帧 / 一阶外推）各自的 EM；② 模型在它们之上的边际贡献；
          ③ 标签本身的时间连续性（|ΔL| 分布）——这是"指标被饱和"的直接证据。
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/xieliangcai/code/gazeformer-exps")
CSV = ROOT / "out/rgbdgaze/analysis/eye224_test.csv"
G, MIN_C = 8, 8

sc = {}
with open(ROOT / "data/RGBDGaze/index/index.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        sc[r["rgb_path"]] = (float(r["screen_w"]) * float(r["cm_px_x"]),
                             float(r["screen_h"]) * float(r["cm_px_y"]))
sess = defaultdict(list)
for r in csv.DictReader(open(CSV, encoding="utf-8")):
    sess[r["path"].rsplit("/", 1)[0]].append(
        dict(fi=int(r["frame_index"]), path=r["path"],
             P=np.array([float(r["pred_x"]), float(r["pred_y"])]),
             L=np.array([float(r["label_x"]), float(r["label_y"])])))
for s in sess:
    sess[s].sort(key=lambda r: r["fi"])
SESS = sorted(sess)


def emv(pred, lab, path):
    w, h = sc[path]
    return float(np.hypot((pred[0] - lab[0]) * w, (pred[1] - lab[1]) * h))


def cm_arr(a, p):
    w, h = sc[p]
    return np.array([a[0] * w, a[1] * h])


# ---------- 标签连续性证据 ----------
dl, jumps, prev_common = [], [], []
for s in SESS:
    v = sess[s]
    for i in range(1, len(v)):
        a = cm_arr(v[i]["L"] - v[i - 1]["L"], v[i]["path"])
        dl.append(float(np.hypot(a[0], a[1])))
dl = np.array(dl)
print("=" * 104)
print("【标签连续性】相邻帧真值位移（厘米）")
print(f"  均值 {dl.mean():.3f}  中位 {np.median(dl):.3f}  p90 {np.percentile(dl, 90):.3f}"
      f"  p99 {np.percentile(dl, 99):.3f}  max {dl.max():.2f}")
for th in (0.05, 0.1, 0.3, 0.5, 1.0):
    print(f"  位移 < {th:<4}cm 的帧占比：{100 * (dl < th).mean():>5.1f}%")

# ---------- 平凡因果基线 ----------
base = np.array([emv(r["P"], r["L"], r["path"]) for s in SESS for r in sess[s]])
res = defaultdict(list)
for s in SESS:
    v = sess[s]
    for i in range(1, len(v)):
        L = v[i]["L"]; p = v[i]["path"]
        prev, prev2 = v[i - 1]["L"], (v[i - 2]["L"] if i >= 2 else None)
        res["P1 上一帧真值"].append(emv(prev, L, p))
        if prev2 is not None:
            res["P4 一阶外推(γ=1)"].append(emv(prev + (prev - prev2), L, p))
            res["P5 外推(γ=0.5)"].append(emv(prev + 0.5 * (prev - prev2), L, p))
        # 模型 + 平凡外推 的固定比例融合（无自适应）——看模型是否真有边际贡献
        for a in (0.1, 0.2, 0.3):
            ex = prev if prev2 is None else prev + (prev - prev2)
            res[f"M 模型×{a} + 外推×(1-{a})"].append(
                emv(a * v[i]["P"] + (1 - a) * ex, L, p))
print()
print("【平凡因果基线与模型边际贡献】")
print(f"  模型原始                       {base.mean():>8.3f}")
for k in sorted(res):
    e = np.array(res[k])
    print(f"  {k:<28}{e.mean():>8.3f}   相对模型 {100 * (e.mean() / base.mean() - 1):>7.1f}%"
          f"   >3cm {100 * (e > 3).mean():>5.2f}%")
