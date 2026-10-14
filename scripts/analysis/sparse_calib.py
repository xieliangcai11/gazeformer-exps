"""稀疏标定协议：UX 友好版（经典"开头做一次多点标定"）能否拿到在线标定的收益？

关键疑问（必须分清两件事）：
  前面测出"必须≲1~2 秒刷新" —— 但那是在**只用最近帧**的估计器下得到的。
  如果同一被试内误差是**注视位置的平滑函数**（而非随时间漂移），那么：
    · 用"最近 1 秒"之所以好，是因为轨迹连续 ⇒ 最近帧恰好落在相近的注视位置
    · 那么"开头铺满屏幕的 5~9 点标定"就够了，且不随时间衰减 ⇒ **UX 问题不存在** ✓

做法（严格因果：只用标定段，绝不看评估帧）：
  标定集 = 每 session **开头 R 比例帧**（真实标定阶段）
  评估帧 t：从标定集里取**注视位置最近**的 C 帧，用其 (pred-label) 均值修正
对照：
  · 全局常数（= 此前失败的"前段标定"：被分布漂移污染）
  · 最近 K 帧（时间估计器，= 此前的在线方案）
  · oracle 选择（按真值位置选 ⇒ 上界参考）

sweep：C ∈ {1,3,5,9,25,全} ；R ∈ {0.05,0.10,0.30}
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

IDX = Path("/data1/xieliangcai/code/gazeformer-exps/data/RGBDGaze/index")
sc = {}
with open(IDX / "index.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        sc[r["rgb_path"]] = (float(r["screen_w"]) * float(r["cm_px_x"]),
                             float(r["screen_h"]) * float(r["cm_px_y"]))

CSV = sys.argv[1] if len(sys.argv) > 1 else \
    "/data1/xieliangcai/code/gazeformer-exps/out/rgbdgaze/analysis/eye224_test.csv"

sess = defaultdict(list)
for r in csv.DictReader(open(CSV, encoding="utf-8")):
    sess[r["path"].rsplit("/", 1)[0]].append(
        dict(path=r["path"], fi=int(r["frame_index"]),
             P=np.array([float(r["pred_x"]), float(r["pred_y"])]),
             L=np.array([float(r["label_x"]), float(r["label_y"])]),
             S=sc[r["path"]]))
for s in sess:
    sess[s].sort(key=lambda r: r["fi"])


def em_arr(corr, v, idxs):
    return np.array([float(np.hypot((corr[k][0] - v[i]["L"][0]) * v[i]["S"][0],
                                    (corr[k][1] - v[i]["L"][1]) * v[i]["S"][1]))
                     for k, i in enumerate(idxs)])


def run(R=0.10, C=9, sel="pred", weight="uniform"):
    """返回 (全部评估帧误差, 标定帧数占比)。sel: pred=按预测位置选, label=按真值选(oracle)。"""
    errs, n_cal, n_tot = [], 0, 0
    for s, v in sess.items():
        n = len(v)
        k = max(1, int(n * R))
        cal = list(range(k))
        ev = list(range(k, n))
        if not ev:
            continue
        n_cal += k
        n_tot += n
        Pc = np.array([v[i]["P"] for i in cal])
        Lc = np.array([v[i]["L"] for i in cal])
        Dc = Pc - Lc
        for t in ev:
            q = v[t]["L"] if sel == "label" else v[t]["P"]
            d2 = ((Lc - q) ** 2).sum(1)
            take = np.argsort(d2)[:min(C, len(cal))]
            if weight == "gauss":
                w = np.exp(-d2[take] / (2 * 0.10 ** 2))
                if w.sum() < 1e-9:
                    w = np.ones(len(take))
            else:
                w = np.ones(len(take))
            corr = v[t]["P"] - (Dc[take] * w[:, None]).sum(0) / w.sum()
            errs.append(float(np.hypot((corr[0] - v[t]["L"][0]) * v[t]["S"][0],
                                       (corr[1] - v[t]["L"][1]) * v[t]["S"][1])))
    return np.array(errs), n_cal / max(n_tot, 1)


base, _ = run(R=0.0, C=1, sel="pred")
base = np.array([float(np.hypot(0, 0))]) if len(base) == 0 else base
# 真正的基线：不修正
b_errs = []
for s, v in sess.items():
    for r in v:
        b_errs.append(float(np.hypot(0, 0))) if False else None
base = np.array([float(np.hypot((r["P"][0] - r["L"][0]) * r["S"][0],
                                (r["P"][1] - r["L"][1]) * r["S"][1]))
                 for s in sess for r in sess[s]])

print("=" * 108)
print(f"【稀疏标定协议】{Path(CSV).stem}")
print("=" * 108)
print(f"  不标定基线： EM {base.mean():.3f}  中位 {np.median(base):.3f}  "
      f">3cm {100 * (base > 3).mean():.2f}%")


def line(tag, e, frac=None):
    extra = f"{frac * 100:>8.1f}%" if frac is not None else f"{'—':>9}"
    print(f"  {tag:<40}{e.mean():>9.3f}{100 * (e.mean() / base.mean() - 1):>9.1f}%"
          f"{np.median(e):>8.3f}{100 * (e > 3).mean():>8.2f}%{extra}")


print(f"\n  {'协议':<40}{'EM均值':>9}{'vs基线':>10}{'中位':>8}{'>3cm':>9}{'标定占比':>9}")
print("  " + "-" * 86)
for R in (0.05, 0.10, 0.30):
    e, f = run(R=R, C=999999, sel="pred")
    line(f"R={R:.0%} 全局常数（旧失败方案）", e, f)
print("  " + "-" * 86)
for R in (0.10,):
    for C in (1, 3, 5, 9, 25, 100):
        e, f = run(R=R, C=C, sel="pred")
        line(f"R={R:.0%} 按注视位置取最近 C={C}", e, f)
print("  " + "-" * 86)
for R in (0.10,):
    for C in (5, 9):
        e, f = run(R=R, C=C, sel="label")
        line(f"R={R:.0%} C={C}（oracle 选择上界参考）", e, f)
print("  " + "-" * 86)
for C in (1, 3, 5, 9, 25):
    e, f = run(R=0.10, C=C, sel="pred", weight="gauss")
    line(f"R=10% C={C} 高斯核加权", e, f)
print("  " + "-" * 86)
print("  对照：整段都用、最近 30 帧（此前的在线方案）= 1.028（−30.2%）")
