"""最终系统数字与机制证据（纯 CPU）：
  ① 误差场的收益在**空间上**集中在哪里（按离屏心距离分桶）⇒ 若集中在边缘，说明它在抵消"收缩/径向放大"✓
  ② 场 + 闭眼帧"借用邻帧"（**仅用过去帧**，严格因果 ✓）的**混合系统**，给出最终指标（含 >3cm/>5cm 尾部）
供早上简报直接引用。

用法: python field_final.py <预测CSV> [blink阈值=0.3]
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
G, MIN, NZ = 8, 8, 12

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
        dict(path=r["path"], sub=r["subject"], fi=int(r["frame_index"]),
             P=np.array([float(r["pred_x"]), float(r["pred_y"])]),
             L=np.array([float(r["label_x"]), float(r["label_y"])]),
             S=sc[r["path"]], b=blink.get(r["path"], 0.0)))
for s in sess:
    sess[s].sort(key=lambda r: r["fi"])


def add(st, x, y, dx, dy):
    st[0] += 1
    st[1] += x; st[2] += y
    st[3] += x * x; st[4] += x * y; st[5] += y * y
    st[6] += dx; st[7] += dx * x; st[8] += dx * y
    st[9] += dy; st[10] += dy * x; st[11] += dy * y


# ---- 逐帧：原始 / 场修正 / 因果借用（仅过去干净帧）+ 场 ----
recs = []
for s, v in sess.items():
    glob = defaultdict(lambda: np.zeros(NZ))
    grid = defaultdict(lambda: [0.0, 0.0, 0.0])
    last_clean_corr = None
    for r in v:
        P, L, w, h = r["P"], r["L"], r["S"][0], r["S"][1]
        x, y = float(P[0]), float(P[1])
        a = glob[""]
        beta = None
        if a[0] >= MIN:
            A = np.array([[a[0], a[1], a[2]], [a[1], a[3], a[4]], [a[2], a[4], a[5]]]) \
                + 0.01 * a[0] * np.diag([0.0, 1.0, 1.0])
            try:
                beta = (np.linalg.solve(A, a[6:9]), np.linalg.solve(A, np.array([a[9], a[10], a[11]])))
            except np.linalg.LinAlgError:
                beta = None
        bv = np.array([np.array([1.0, x, y]) @ beta[0], np.array([1.0, x, y]) @ beta[1]]) \
            if beta is not None else np.zeros(2)
        p1 = P - bv
        i0 = min(G - 1, max(0, int(float(p1[0]) * G)))
        j0 = min(G - 1, max(0, int(float(p1[1]) * G)))
        cnt = sx = sy = 0.0
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                i, j = i0 + di, j0 + dj
                if 0 <= i < G and 0 <= j < G:
                    c = grid[(i, j)]
                    cnt += c[0]; sx += c[1]; sy += c[2]
        corr = p1 - np.array([sx / cnt, sy / cnt]) if cnt >= MIN else p1
        e_raw = float(np.hypot((P[0] - L[0]) * w, (P[1] - L[1]) * h))
        e_corr = float(np.hypot((corr[0] - L[0]) * w, (corr[1] - L[1]) * h))
        # 因果借用：若当前为闭眼帧且存在过去的干净帧的"已修正预测"，用它
        e_hyb = e_corr
        if r["b"] > THR and last_clean_corr is not None:
            e_hyb = float(np.hypot((last_clean_corr[0] - L[0]) * w,
                                   (last_clean_corr[1] - L[1]) * h))
        if r["b"] <= THR:
            last_clean_corr = corr
        recs.append(dict(raw=e_raw, corr=e_corr, hyb=e_hyb, b=r["b"], sub=r["sub"],
                         dc=float(np.hypot(L[0] - 0.5, L[1] - 0.5))))
        add(a, x, y, float(P[0] - L[0]), float(P[1] - L[1]))
        c = grid[(min(G - 1, max(0, int(float(p1[0]) * G))),
                  min(G - 1, max(0, int(float(p1[1]) * G))))]
        c[0] += 1
        c[1] += float(P[0] - L[0]) - float(bv[0])
        c[2] += float(P[1] - L[1]) - float(bv[1])

raw = np.array([r["raw"] for r in recs])
corr = np.array([r["corr"] for r in recs])
hyb = np.array([r["hyb"] for r in recs])
dc = np.array([r["dc"] for r in recs])
bv = np.array([r["b"] for r in recs])

print("=" * 100)
print(f"【最终系统】{Path(CSV).stem}")
print("=" * 100)
print(f"  {'方案':<24}{'EM均值':>9}{'中位':>8}{'>2cm':>8}{'>3cm':>8}{'>5cm':>8}{'max':>8}")
for nm, a in (("A 原始（未标定）", raw), ("B 误差场（因果在线）", corr),
              ("C 场 + 闭眼借用（因果）", hyb)):
    print(f"  {nm:<24}{a.mean():>9.3f}{np.median(a):>8.3f}{100 * (a > 2).mean():>7.2f}%"
          f"{100 * (a > 3).mean():>7.2f}%{100 * (a > 5).mean():>7.2f}%{a.max():>8.2f}")

print(f"\n  【机制证据】场修正的收益 vs 离屏心距离（若集中在边缘 ⇒ 它在抵消收缩/径向放大 ✓）")
print(f"  {'离屏心距离':<16}{'n':>7}{'原始':>9}{'场修正':>9}{'相对增益':>10}")
qs = np.percentile(dc, [0, 25, 50, 75, 100])
for a_, b_ in zip(qs[:-1], qs[1:]):
    m = (dc >= a_) & (dc <= b_ if b_ == qs[-1] else dc < b_)
    if m.sum() == 0:
        continue
    print(f"  {f'{a_:.3f}~{b_:.3f}':<16}{m.sum():>7}{raw[m].mean():>9.3f}"
          f"{corr[m].mean():>9.3f}{100 * (corr[m].mean() / raw[m].mean() - 1):>9.1f}%")
print(f"\n  【对照】闭眼帧上的三方案（A→B→C）:")
m = bv > THR
print(f"    n={m.sum()}  {raw[m].mean():.3f} → {corr[m].mean():.3f} → {hyb[m].mean():.3f}")

subs = sorted({r["sub"] for r in recs})
print(f"\n  【分被试】A 原始 → B 场修正（检验'最差被试获益是否最大'）")
print(f"  {'被试':<7}{'n':>6}{'A 原始':>9}{'B 场':>9}{'相对增益':>10}{'A>3cm':>9}{'B>3cm':>9}")
rows2 = []
for s in subs:
    m2 = np.array([r["sub"] == s for r in recs])
    if m2.sum() < 50:
        continue
    a0, a1 = raw[m2].mean(), corr[m2].mean()
    rows2.append((s, m2.sum(), a0, a1, 100 * (a1 / a0 - 1),
                  100 * (raw[m2] > 3).mean(), 100 * (corr[m2] > 3).mean()))
for s, n, a0, a1, g, t0, t1 in sorted(rows2, key=lambda x: -x[2]):
    print(f"  {s:<7}{n:>6}{a0:>9.3f}{a1:>9.3f}{g:>9.1f}%{t0:>8.2f}%{t1:>8.2f}%")
c0 = np.corrcoef([x[2] for x in rows2], [x[4] for x in rows2])[0, 1] if len(rows2) > 2 else float("nan")
print(f"  ⇒ corr(原始误差, 相对增益) = {c0:+.2f}（越接近 -1 表示'越差的被试获益越大'）")
