"""目标点选择模拟：只有 k 个标定目标时，4 参数（闭式矩匹配）校正还剩多少？

背景（本仓库实测）：
  · 逐会话误差主体是"收缩 + 偏移"（σ_label/σ_pred ≈ 1.2，最高 3.9）⇒ 4 参数就够 ✓
  · 用**本会话目标点的边际**估这 4 个数 ⇒ 追平"用真值拟合的 oracle 仿射" ✓（−23.3%）
  · 经典最小二乘标定用同样信息反而变差 ✗ ⇒ 估计器（矩匹配）本身有价值 ✓

本脚本只回答一个产品约束问题 ✓：
  **只能展示 k 个目标点时，选哪 k 个最好？** 对比：
    all       = 全部帧（上界）
    first     = 会话前 k 帧（= 经典标定协议，时序前缀）
    random    = 随机 k 帧（多次抽样取均值）
    farthest  = 目标空间**最远点**采样（覆盖最优 ✓）
  评估：把由这 k 帧估出的 (s,b) 应用到**整段会话**，算平均 EM。

用法：
  python -m scripts.analysis.target_select --preds out/rgbdgaze/analysis/preds_eye224_mgpu_test.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
from collections import defaultdict

C = 0.5          # 屏幕中心（归一化坐标）


def load(preds):
    rows = []
    for r in csv.DictReader(open(preds, newline="", encoding="utf-8")):
        rows.append({
            "sess": r["path"].rsplit("/", 1)[0],
            "fn": int(r["frame_index"]),
            "px": float(r["pred_x"]), "py": float(r["pred_y"]),
            "lx": float(r["label_x"]), "ly": float(r["label_y"]),
            "ex": float(r["err_x_cm"]), "ey": float(r["err_y_cm"]),
        })
    g = defaultdict(list)
    for r in rows:
        g[r["sess"]].append(r)
    for s in g:
        g[s].sort(key=lambda r: r["fn"])
    return g


def cm_scale(rs):
    """每会话 'cm / 归一化单位'（中位；剔除小分母）"""
    sx = statistics.median([r["ex"] / (r["px"] - r["lx"]) for r in rs if abs(r["px"] - r["lx"]) > 0.01])
    sy = statistics.median([r["ey"] / (r["py"] - r["ly"]) for r in rs if abs(r["py"] - r["ly"]) > 0.01])
    return sx, sy


def fit_sb(rs):
    """闭式 4 参数：s = σ_label/σ_pred（逐轴），b 使校正后均值 = 目标均值（绕屏心）"""
    n = len(rs)
    mpx = sum(r["px"] for r in rs) / n; mpy = sum(r["py"] for r in rs) / n
    mlx = sum(r["lx"] for r in rs) / n; mly = sum(r["ly"] for r in rs) / n
    spx = math.sqrt(sum((r["px"] - mpx) ** 2 for r in rs) / n)
    spy = math.sqrt(sum((r["py"] - mpy) ** 2 for r in rs) / n)
    slx = math.sqrt(sum((r["lx"] - mlx) ** 2 for r in rs) / n)
    sly = math.sqrt(sum((r["ly"] - mly) ** 2 for r in rs) / n)
    sx = slx / max(spx, 1e-6); sy = sly / max(spy, 1e-6)
    bx = mlx - (sx * (mpx - C) + C)
    by = mly - (sy * (mpy - C) + C)
    return sx, sy, bx, by


def apply_sb(rs, S, s):
    sx, sy, bx, by = s; Sx, Sy = S
    tot = 0.0
    for r in rs:
        dx = sx * (r["px"] - C) + C + bx - r["px"]
        dy = sy * (r["py"] - C) + C + by - r["py"]
        tot += math.hypot(r["ex"] + dx * Sx, r["ey"] + dy * Sy)
    return tot / len(rs)


def farthest_sampling(rs, k):
    """在**目标空间**（label）做最远点采样（首个点取离屏心最远者 ✓）"""
    if k >= len(rs):
        return list(rs)
    pts = [(r["lx"], r["ly"]) for r in rs]
    first = max(range(len(rs)), key=lambda i: math.hypot(pts[i][0] - C, pts[i][1] - C))
    chosen = [first]
    d = [math.hypot(pts[i][0] - pts[first][0], pts[i][1] - pts[first][1]) for i in range(len(rs))]
    while len(chosen) < k:
        i = max((j for j in range(len(rs)) if j not in chosen), key=lambda j: d[j])
        chosen.append(i)
        for j in range(len(rs)):
            dj = math.hypot(pts[j][0] - pts[i][0], pts[j][1] - pts[i][1])
            if dj < d[j]:
                d[j] = dj
    return [rs[i] for i in chosen]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="out/rgbdgaze/analysis/preds_eye224_mgpu_test.csv")
    ap.add_argument("--ks", default="3,5,10,20,50")
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    g = load(args.preds)
    ks = [int(x) for x in args.ks.split(",")]
    rng = random.Random(args.seed)

    print(f"{'会话（后段等权汇总 ✓）':<26}" + "".join(f"{k:>10}" for k in ks))
    acc = {m: {k: [] for k in ks} for m in ("first", "random", "farthest", "all")}
    base = []
    for s in sorted(g):
        rs = g[s]
        S = cm_scale(rs)
        b = sum(math.hypot(r["ex"], r["ey"]) for r in rs) / len(rs)
        base.append(b)
        all_s = fit_sb(rs)
        vals = {}
        for k in ks:
            vals[k] = {}
            vals[k]["first"] = apply_sb(rs, S, fit_sb(rs[:k]))
            rr = [apply_sb(rs, S, fit_sb(rng.sample(rs, k))) for _ in range(args.draws)]
            vals[k]["random"] = statistics.mean(rr)
            vals[k]["farthest"] = apply_sb(rs, S, fit_sb(farthest_sampling(rs, k)))
            vals[k]["all"] = apply_sb(rs, S, all_s)
            for m in ("first", "random", "farthest", "all"):
                acc[m][k].append(vals[k][m])
        # 打印该会话 farthest 行
        print(f"{s[-24:]:<26}" + "".join(f"{vals[k]['farthest']:>10.3f}" for k in ks))
    print(f"\n{'★ 平均（会话等权）':<26}" + "".join(" " * 10 for _ in ks))
    print(f"{'基线（不校正）':<26}{statistics.mean(base):>10.4f}")
    for m, name in (("first", "前 k 帧（经典 ✓）"), ("random", "随机 k"), ("farthest", "最远点 k（覆盖 ✓）")):
        print(f"{name:<26}" + "".join(f"{statistics.mean(acc[m][k]):>10.4f}" for k in ks))
    print(f"{'全部帧（上界 ✓）':<26}" + "".join(f"{statistics.mean(acc['all'][k]):>10.4f}" for k in ks))
    print(f"\n{'相对基线（最远点）':<26}" + "".join(
        f"{100*(statistics.mean(acc['farthest'][k])/statistics.mean(base)-1):>9.1f}%" for k in ks))
    print(f"{'相对基线（前 k 帧）':<26}" + "".join(
        f"{100*(statistics.mean(acc['first'][k])/statistics.mean(base)-1):>9.1f}%" for k in ks))


if __name__ == "__main__":
    main()
