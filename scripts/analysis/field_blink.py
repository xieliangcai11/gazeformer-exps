"""误差场（最优层级配置）修的是什么、修不动什么：按闭眼程度分桶 + 修正后最差帧构成。

目的：把"故事"收口 ——
  · 若场在闭眼帧上也有效 ⇒ 闭眼误差里也有系统性成分 ⇒ 可修 ✗
  · 若场在闭眼帧上无效 ⇒ 闭眼地板是"逐帧不可约"的（需时序输入，D3）✓
同时给出**修正后**的最差帧列表（看尾部构成是否改变：是否仍被闭眼帧占据）。

配置与 field_hier.py 最优一致：全局仿射（每帧重解）→ 每格残差偏移（G=8，3x3 池化），严格因果。
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/xieliangcai/code/gazeformer-exps")
AN = ROOT / "out/rgbdgaze/analysis"
CSV = sys.argv[1] if len(sys.argv) > 1 else str(AN / "eye224_test.csv")
G, MIN = 8, 8
NZ = 12

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
             S=sc[r["path"]]))
for s in sess:
    sess[s].sort(key=lambda r: r["fi"])


def fit_affine(a, lam=0.01):
    if a[0] < MIN:
        return None
    pen = lam * a[0]
    A = np.array([[a[0], a[1], a[2]],
                  [a[1], a[3] + pen, a[4]],
                  [a[2], a[4], a[5] + pen]])
    try:
        return (np.linalg.solve(A, a[6:9]), np.linalg.solve(A, np.array([a[9], a[10], a[11]])))
    except np.linalg.LinAlgError:
        return None


def add(st, x, y, dx, dy):
    st[0] += 1
    st[1] += x; st[2] += y
    st[3] += x * x; st[4] += x * y; st[5] += y * y
    st[6] += dx; st[7] += dx * x; st[8] += dx * y
    st[9] += dy; st[10] += dy * x; st[11] += dy * y


base_all, corr_all, meta_all = [], [], []
for s, v in sess.items():
    glob = defaultdict(lambda: np.zeros(NZ))
    grid = defaultdict(lambda: [0.0, 0.0, 0.0])
    for r in v:
        P, L = r["P"], r["L"]
        x, y = float(P[0]), float(P[1])
        b = fit_affine(glob[""])
        g = np.array([np.array([1.0, x, y]) @ b[0], np.array([1.0, x, y]) @ b[1]]) \
            if b is not None else np.zeros(2)
        p1 = P - g
        cx, cy = float(p1[0]), float(p1[1])
        i0 = min(G - 1, max(0, int(cx * G)))
        j0 = min(G - 1, max(0, int(cy * G)))
        cnt = sx = sy = 0.0
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                i, j = i0 + di, j0 + dj
                if 0 <= i < G and 0 <= j < G:
                    c = grid[(i, j)]
                    cnt += c[0]; sx += c[1]; sy += c[2]
        est = np.array([sx / cnt, sy / cnt]) if cnt >= MIN else None
        corr = p1 - est if est is not None else p1
        w, h = r["S"]
        base_all.append(float(np.hypot((P[0] - L[0]) * w, (P[1] - L[1]) * h)))
        corr_all.append(float(np.hypot((corr[0] - L[0]) * w, (corr[1] - L[1]) * h)))
        meta_all.append((r["path"], float(corr_all[-1])))
        add(glob[""], x, y, float(P[0] - L[0]), float(P[1] - L[1]))
        c = grid[(min(G - 1, max(0, int(cx * G))), min(G - 1, max(0, int(cy * G))))]
        c[0] += 1
        c[1] += float(P[0] - L[0]) - float(g[0])
        c[2] += float(P[1] - L[1]) - float(g[1])

base = np.array(base_all)
corr = np.array(corr_all)
print("=" * 92)
print(f"【误差场修的是什么】{Path(CSV).stem}  不修正 EM {base.mean():.3f} → "
      f"层级场 {corr.mean():.3f}（{100 * (corr.mean() / base.mean() - 1):+.1f}%）")
print("=" * 92)
print(f"  {'闭眼程度':<22}{'n':>7}{'不修正':>9}{'修正后':>9}{'相对增益':>10}")
bs = np.array([blink.get(p, np.nan) for p, _ in meta_all])
for tag, m in (("睁眼 blink<=0.3", bs <= 0.3), ("半闭 0.3~0.7", (bs > 0.3) & (bs <= 0.7)),
               ("闭眼 >0.7", bs > 0.7), ("全部", np.ones(len(bs), bool))):
    m = np.nan_to_num(m, nan=False).astype(bool)
    if m.sum() == 0:
        continue
    print(f"  {tag:<22}{m.sum():>7}{base[m].mean():>9.3f}{corr[m].mean():>9.3f}"
          f"{100 * (corr[m].mean() / base[m].mean() - 1):>9.1f}%")
print(f"\n  阈值占比（不修正 → 修正后）:")
for t in (2, 3, 5):
    print(f"    >{t}cm : {100 * (base > t).mean():>6.2f}% → {100 * (corr > t).mean():>6.2f}%")
print(f"\n  修正后最差 10 帧（看尾部构成是否仍被闭眼帧占据）:")
order = np.argsort(-corr)[:10]
for i in order:
    p = meta_all[i][0]
    print(f"    {corr[i]:>6.2f}cm  blink={blink.get(p, -1):.2f}  {p.split('/')[-1]}")
