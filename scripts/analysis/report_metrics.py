"""测试集完整指标报告（单模型，或与若干对照并排）。

用法: python report_metrics.py <主模型.csv> [对照1.csv 对照2.csv ...]

主模型输出全部细分；对照只出现在"总览/尾部/分被试/闭眼"四张对照表里。
所有 CSV 需含: path, subject, activity, device, frame_index,
               pred_x, pred_y, label_x, label_y, em_cm, err_x_cm, err_y_cm, ang_deg
（由 scripts/analysis/dump_predictions.py 生成）
blink 分数从 analysis/blink_scores.csv 按 path 关联。
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
screencm = {}
for _idx in (IDX, GC_IDX):
    if not _idx.exists():
        continue
    with open(_idx / "index.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            pk = "rgb_path" if "rgb_path" in r else "frame_path"
            screencm[r[pk]] = (float(r["screen_w"]) * float(r["cm_px_x"]),
                               float(r["screen_h"]) * float(r["cm_px_y"]))


def load(p):
    rows = []
    for r in csv.DictReader(open(p, encoding="utf-8")):
        rows.append(dict(path=r["path"], sub=r["subject"], act=r.get("activity", "?"),
                         dev=r.get("device", "?"), fi=int(r.get("frame_index", 0)),
                         px=float(r["pred_x"]), py=float(r["pred_y"]),
                         lx=float(r["label_x"]), ly=float(r["label_y"]),
                         em=float(r["em_cm"]), ax=abs(float(r["err_x_cm"])),
                         ay=abs(float(r["err_y_cm"])), ang=float(r["ang_deg"])))
    return rows


def em_of(pred, lab, p):
    w, h = screencm[p]
    return float(np.hypot((pred[0] - lab[0]) * w, (pred[1] - lab[1]) * h))


def stat(name, em, ang=None, ind="  "):
    e = np.asarray(em)
    s = (f"{ind}{name:<26}{len(e):>6}{e.mean():>9.3f}{np.median(e):>9.3f}"
         f"{e.std():>8.3f}{np.percentile(e, 90):>8.3f}{np.percentile(e, 95):>8.3f}"
         f"{np.percentile(e, 99):>8.3f}{e.min():>7.3f}{e.max():>7.2f}")
    if ang is not None:
        a = np.asarray(ang)
        s += (f"{a.mean():>8.2f}{np.median(a):>8.2f}{(a <= 1.91).mean() * 100:>8.1f}%")
    return s


def main():
    main_csv = sys.argv[1]
    others = sys.argv[2:]
    R = load(main_csv)
    EM = np.array([r["em"] for r in R])
    ANG = np.array([r["ang"] for r in R])
    AX = np.array([r["ax"] for r in R])
    AY = np.array([r["ay"] for r in R])
    nm = Path(main_csv).stem

    print("=" * 140)
    print(f"【总览】{nm}   n={len(R)}")
    print("=" * 140)
    print(f"{'':<26}{'n':>6}{'EM均值':>9}{'EM中位':>9}{'标准差':>8}{'p90':>8}{'p95':>8}"
          f"{'p99':>8}{'min':>7}{'max':>7}{'角均值':>8}{'角中位':>8}{'≤1.91°':>9}")
    print(stat("主模型", EM, ANG))
    for o in others:
        Ro = load(o)
        print(stat(Path(o).stem[:25], [r["em"] for r in Ro], [r["ang"] for r in Ro]))

    print(f"\n  分轴误差：|ex| 均值 {AX.mean():.3f} 中位 {np.median(AX):.3f} | "
          f"|ey| 均值 {AY.mean():.3f} 中位 {np.median(AY):.3f} | ey/ex = {AY.mean() / AX.mean():.2f}")
    print("  分位数(cm)： " + "  ".join(f"p{q}={np.percentile(EM, q):.3f}"
                                       for q in (5, 10, 25, 50, 75, 90, 95, 99)))
    print("  角度分位(°)：" + "  ".join(f"p{q}={np.percentile(ANG, q):.2f}"
                                       for q in (50, 75, 90, 95, 99)))
    print("  角度阈值： " + "  ".join(
        f"≤{t}°={100 * (ANG <= t).mean():.1f}%" for t in (1.91, 2, 3, 5, 10)))
    print("\n  误差阈值堆积（帧数与占比）：")
    for t in (0.5, 1, 2, 3, 4, 5, 8, 10):
        print(f"     >{t:>4}cm : {(EM > t).sum():>6} 帧  {100 * (EM > t).mean():>6.2f}%")

    def group(key, title, sort_by_n=True):
        print("\n" + "=" * 140)
        print(f"【按{title}】")
        print("=" * 140)
        print(f"{'':<26}{'n':>6}{'EM均值':>9}{'EM中位':>9}{'标准差':>8}{'p90':>8}{'p95':>8}"
              f"{'p99':>8}{'min':>7}{'max':>7}{'角均值':>8}{'角中位':>8}{'≤1.91°':>9}")
        d = defaultdict(list)
        for r in R:
            d[r[key]].append(r)
        keys = sorted(d, key=(lambda k: -len(d[k])) if sort_by_n else (lambda k: k))
        for k in keys:
            v = d[k]
            print(stat(f"{k}", [x["em"] for x in v], [x["ang"] for x in v]))
        print(stat("— 全量 —", EM, ANG))

    group("sub", "被试")
    group("act", "活动")
    group("dev", "设备")

    print("\n" + "=" * 140)
    print("【按屏幕 3×3 区域】EM 均值 / 中位 / n")
    print("=" * 140)
    ix = np.clip((np.array([r["lx"] for r in R]) * 3).astype(int), 0, 2)
    iy = np.clip((np.array([r["ly"] for r in R]) * 3).astype(int), 0, 2)
    for j in range(3):
        line = "  "
        for i in range(3):
            c = (ix == i) & (iy == j)
            line += (f"x{i + 1}y{j + 1}: {EM[c].mean():5.3f}/{np.median(EM[c]):5.3f}"
                     f" (n={int(c.sum()):>4})   ")
        print(line)

    print("\n" + "=" * 140)
    print("【按离屏心距离】误差随注视点离中心距离的变化（几何放大效应）")
    print("=" * 140)
    dc = np.hypot(np.array([r["lx"] for r in R]) - 0.5, np.array([r["ly"] for r in R]) - 0.5)
    qs = np.percentile(dc, [0, 20, 40, 60, 80, 100])
    for a, b in zip(qs[:-1], qs[1:]):
        m = (dc >= a) & (dc <= b if b == qs[-1] else dc < b)
        if m.sum():
            print(f"  离中心 {a:.3f}~{b:.3f}: n={m.sum():>5}  EM均值 {EM[m].mean():.3f}  "
                  f"中位 {np.median(EM[m]):.3f}  角均值 {ANG[m].mean():.2f}°")

    print("\n" + "=" * 140)
    print("【按 blink 分桶】（眼支流的本职）")
    print("=" * 140)
    bv = np.array([blink.get(r["path"], np.nan) for r in R])
    for nmb, m in (("睁眼 ≤0.3", bv <= 0.3), ("半闭 0.3~0.7", (bv > 0.3) & (bv <= 0.7)),
                   ("闭眼 >0.7", bv > 0.7)):
        m = np.nan_to_num(m, nan=False).astype(bool)
        if m.sum():
            print(stat(nmb, EM[m], ANG[m]))

    print("\n" + "=" * 140)
    print("【会话内时序】误差随会话内相对位置的变化（漂移检查）")
    print("=" * 140)
    sess = defaultdict(list)
    for r in R:
        sess[r["path"].rsplit("/", 1)[0]].append(r)
    pos, e = [], []
    for s, v in sess.items():
        v.sort(key=lambda r: r["fi"])
        for k, r in enumerate(v):
            pos.append(k / max(len(v) - 1, 1))
            e.append(r["em"])
    pos, e = np.array(pos), np.array(e)
    for a, b in ((0, .1), (.1, .3), (.3, .7), (.7, .9), (.9, 1.01)):
        m = (pos >= a) & (pos < b)
        if m.sum():
            print(f"  会话进度 {a * 100:>3.0f}~{b * 100:>3.0f}%: n={m.sum():>5}  "
                  f"EM均值 {e[m].mean():.3f}  中位 {np.median(e[m]):.3f}")

    print("\n" + "=" * 140)
    print("【每 session 结构】|bias| / 偏置占比 / 预测收缩")
    print("=" * 140)
    print(f"{'':<24}{'n':>6}{'|bias|':>9}{'偏置占比':>10}{'收缩x':>8}{'收缩y':>8}"
          f"{'EM':>8}")
    mags, shares = [], []
    for s in sorted(sess, key=lambda k: -len(sess[k])):
        v = sorted(sess[s], key=lambda r: r["fi"])
        scv = np.array([screencm[r["path"]] for r in v])
        d = np.array([[r["px"] - r["lx"], r["py"] - r["ly"]] for r in v]) * scv
        P = np.array([[r["px"] * screencm[r["path"]][0],
                       r["py"] * screencm[r["path"]][1]] for r in v])
        L = np.array([[r["lx"] * screencm[r["path"]][0],
                       r["ly"] * screencm[r["path"]][1]] for r in v])
        m = float(np.linalg.norm(d.mean(0)))
        share = m / np.abs(d).sum(1).mean()
        mags.append(m)
        shares.append(share)
        print(f"  {s.rsplit('/', 2)[-2] + '/' + s.rsplit('/', 1)[-1]:<24}{len(v):>6}"
              f"{m:>9.3f}{100 * share:>9.1f}%{P[:, 0].std() / L[:, 0].std():>8.3f}"
              f"{P[:, 1].std() / L[:, 1].std():>8.3f}{np.abs(d).sum(1).mean():>8.3f}")
    print(f"  ⇒ 中位 |bias| {np.median(mags):.3f}cm | 中位偏置占比 "
          f"{100 * np.median(shares):.1f}%")

    print("\n" + "=" * 140)
    print("【最差 15 帧】")
    print("=" * 140)
    for i in np.argsort(-EM)[:15]:
        r = R[i]
        print(f"  {EM[i]:>7.2f}cm  {r['sub']}/{r['act']}  blink={blink.get(r['path'], -1):.2f}"
              f"  真值({r['lx']:.2f},{r['ly']:.2f}) 预测({r['px']:.2f},{r['py']:.2f})"
              f"  {r['path'].split('/')[-1]}")


if __name__ == "__main__":
    main()
