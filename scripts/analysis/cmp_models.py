"""多模型对比：结构诊断（偏置/收缩）+ 尾部 + 闭眼条件误差。

用法: python cmp_models.py a_test.csv b_test.csv ...
每个 CSV 需含 path, subject, label_x, label_y, pred_x, pred_y, em_cm。
blink 分数从 out/rgbdgaze/analysis/blink_scores.csv 按 path 关联。

回答三个问题：
  Q1 每个模型的"每 session 常数偏置占误差比例"与"预测收缩(std比)"是否一致 ⇒ 泛化性
  Q2 尾部指标（>2/3/5cm、max）谁更小
  Q3 眼支流是否在**闭眼/半闭帧**(blink>0.3)上更好 ⇒ 它的本职
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/xieliangcai/code/gazeformer-exps")
AN = ROOT / "out/rgbdgaze/analysis"
IDX = ROOT / "data/RGBDGaze/index"
GC_IDX = ROOT / "data/GazeCapture/index"

blink = {r["path"]: float(r["blink"]) for r in
         csv.DictReader(open(AN / "blink_scores.csv", encoding="utf-8"))}
def load_scale():
    """把两个数据集的屏幕物理尺度合并成 {path: (w_cm, h_cm)}（数据集无关）。"""
    out = {}
    for idx in (IDX / "index.csv", GC_IDX / "index.csv"):
        if not idx.exists():
            continue
        with open(idx, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                pk = "rgb_path" if "rgb_path" in r else "frame_path"
                out[r[pk]] = (float(r["screen_w"]) * float(r["cm_px_x"]),
                              float(r["screen_h"]) * float(r["cm_px_y"]))
    return out


sc = load_scale()


def load(p):
    rows = []
    for r in csv.DictReader(open(p, encoding="utf-8")):
        rows.append((r["path"], r["subject"],
                     float(r["pred_x"]), float(r["pred_y"]),
                     float(r["label_x"]), float(r["label_y"]), float(r["em_cm"])))
    return rows


print("=" * 108)
print("【Q1/Q2 各模型：结构诊断 + 尾部】")
print("=" * 108)
print(f"  {'模型':<24}{'EM均值':>8}{'中位':>7}{'p95':>7}{'max':>7}"
      f"{'>2cm':>7}{'>3cm':>7}{'>5cm':>7}{'|bias|中位':>11}{'偏置占比':>9}"
      f"{'收缩x':>7}{'收缩y':>7}")
data = {}
for p in sys.argv[1:]:
    rows = load(p)
    em = np.array([r[6] for r in rows])
    sess = defaultdict(list)
    for r in rows:
        sess[(r[1], r[0].rsplit("/", 1)[0])].append(r)
    mags, shares = [], []
    for k, v in sess.items():
        if len(v) < 50:
            continue
        scv = np.array([sc[r[0]] for r in v])
        d = np.array([[r[2] - r[4], r[3] - r[5]] for r in v]) * scv
        mags.append(float(np.linalg.norm(d.mean(0))))
        shares.append(float(np.linalg.norm(d.mean(0)) / np.abs(d).sum(1).mean()))
    P = np.array([[r[2] * sc[r[0]][0], r[3] * sc[r[0]][1]] for r in rows])
    L = np.array([[r[4] * sc[r[0]][0], r[5] * sc[r[0]][1]] for r in rows])
    rx, ry = P[:, 0].std() / L[:, 0].std(), P[:, 1].std() / L[:, 1].std()
    data[p] = dict(rows=rows, em=em, mags=mags, shares=shares, rx=rx, ry=ry)
    nm = Path(p).stem.replace("_test", "")[:23]
    print(f"  {nm:<24}{em.mean():>8.3f}{np.median(em):>7.3f}{np.percentile(em, 95):>7.3f}"
          f"{em.max():>7.2f}{100 * (em > 2).mean():>6.2f}%{100 * (em > 3).mean():>6.2f}%"
          f"{100 * (em > 5).mean():>6.2f}%{np.median(mags):>11.3f}"
          f"{100 * np.median(shares):>8.1f}%{rx:>7.3f}{ry:>7.3f}")

print("\n" + "=" * 108)
print("【Q3 闭眼条件误差】blink 分桶的 EM 均值（眼支流的本职：高 blink 帧应更好）")
print("=" * 108)
buckets = [("睁眼 blink<=0.3", lambda b: b <= 0.3),
           ("半闭 0.3-0.7", lambda b: 0.3 < b <= 0.7),
           ("闭眼 >0.7", lambda b: b > 0.7)]
print(f"  {'模型':<24}{'睁眼 n/EM':>18}{'半闭 n/EM':>18}{'闭眼 n/EM':>18}")
for p in sys.argv[1:]:
    rows = data[p]["rows"]
    out = f"  {Path(p).stem.replace('_test', '')[:23]:<24}"
    for nm, f in buckets:
        v = [r[6] for r in rows if r[0] in blink and f(blink[r[0]])]
        out += f"{len(v):>8}/{np.mean(v) if v else float('nan'):>9.3f}" if v else \
               f"{0:>8}/{'--':>9}"
    print(out)
