"""D3（闭眼地板）的"时序输入上限"估算（纯 CPU，零训练）。

思路：闭眼帧没有可用信息 ⇒ 若允许用**邻帧**作输入（协议允许，标签仍逐帧有效 ✓），
      一个时序模型至少能做到"借用最近的非闭眼帧的预测"。
本脚本用该策略作为**下界的下界**（保守），量化闭眼帧误差能降到多少：
  A 现状：当前帧预测
  B 最近的非闭眼帧的预测（"复制邻帧"）
  C 邻帧窗口内（|Δframe|<=k）所有非闭眼帧预测的均值
并同时给出**干净帧**上的对照（说明"必须选择性使用"，否则会伤主体 ✓）。

用法: python blink_temporal.py <预测CSV> [blink阈值=0.3] [窗口k=15]
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/xieliangcai/code/gazeformer-exps")
AN = ROOT / "out/rgbdgaze/analysis"
CSV = sys.argv[1] if len(sys.argv) > 1 else str(AN / "eye224_test.csv")
THR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
KW = int(sys.argv[3]) if len(sys.argv) > 3 else 15

sc = {}
with open(ROOT / "data/RGBDGaze/index/index.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        sc[r["rgb_path"]] = (float(r["screen_w"]) * float(r["cm_px_x"]),
                             float(r["screen_h"]) * float(r["cm_px_y"]))
blink = {r["path"]: float(r["blink"]) for r in
         csv.DictReader(open(AN / "blink_scores.csv", encoding="utf-8"))}

sess = defaultdict(list)
for r in csv.DictReader(open(CSV, encoding="utf-8")):
    sess[r["path"].rsplit("/", 1)[0]].append(
        dict(path=r["path"], fi=int(r["frame_index"]),
             P=np.array([float(r["pred_x"]), float(r["pred_y"])]),
             L=np.array([float(r["label_x"]), float(r["label_y"])]),
             S=sc[r["path"]], b=blink.get(r["path"], 0.0)))
for s in sess:
    sess[s].sort(key=lambda r: r["fi"])


def em(pred, r):
    w, h = r["S"]
    return float(np.hypot((pred[0] - r["L"][0]) * w, (pred[1] - r["L"][1]) * h))


rows = []          # (bucket, base, copy_nearest, window_mean)
for s, v in sess.items():
    fi = np.array([r["fi"] for r in v])
    clean = np.array([i for i, r in enumerate(v) if r["b"] <= THR])
    if len(clean) < 5:
        continue
    for i, r in enumerate(v):
        bucket = "干净" if r["b"] <= THR else ("半闭" if r["b"] <= 0.7 else "闭眼")
        base = em(r["P"], r)
        if len(clean) == 0:
            rows.append((bucket, base, base, base))
            continue
        j = clean[np.argmin(np.abs(fi[clean] - fi[i]))]
        copy = em(v[j]["P"], r)
        w = clean[np.abs(fi[clean] - fi[i]) <= KW]
        wm = em(np.mean([v[k]["P"] for k in w], axis=0), r) if len(w) else copy
        rows.append((bucket, base, copy, wm))

arr = {}
for b in ("干净", "半闭", "闭眼", "全部"):
    sel = [r for r in rows if b == "全部" or r[0] == b]
    if not sel:
        continue
    arr[b] = np.array([[x[1], x[2], x[3]] for x in sel])
print("=" * 96)
print(f"【D3 时序上限估算】{Path(CSV).stem}  blink阈值 {THR}  窗口 ±{KW} 帧"
      f"（≈±{KW/30:.2f}s）")
print("=" * 96)
print(f"  {'帧类别':<8}{'n':>7}{'A 现状':>10}{'B 复制最近非闭眼帧':>20}"
      f"{'C 邻帧均值':>12}{'C相对A':>10}")
for b, a in arr.items():
    print(f"  {b:<8}{len(a):>7}{a[:, 0].mean():>10.3f}{a[:, 1].mean():>20.3f}"
          f"{a[:, 2].mean():>12.3f}{100 * (a[:, 2].mean() / a[:, 0].mean() - 1):>9.1f}%")
print(f"\n  尾部占比（A 现状 → C 邻帧均值）：")
for t in (2, 3, 5):
    a = arr["全部"]
    print(f"    >{t}cm : {100 * (a[:, 0] > t).mean():>6.2f}% → {100 * (a[:, 2] > t).mean():>6.2f}%")
print(f"\n  仅看闭眼帧（A → C）: EM {arr['闭眼'][:, 0].mean():.3f} → {arr['闭眼'][:, 2].mean():.3f}"
      f"  |  >3cm {100 * (arr['闭眼'][:, 0] > 3).mean():.1f}% → "
      f"{100 * (arr['闭眼'][:, 2] > 3).mean():.1f}%")
