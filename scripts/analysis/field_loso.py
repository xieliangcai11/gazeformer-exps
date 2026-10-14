"""误差场可转移性（LOSO）：把"预测注视点 → 误差"的映射在 4 个被试上拟合，
在第 5 个上评估 ⇒ 量出**免标注**可获得的收益（离线学到的系统性误差场）。

若可转移 ⇒ 模型存在与主体无关的系统性误差结构 ⇒ 测试时不需要任何标注即可修正；
若不可转移 ⇒ 误差场是被试特异的 ⇒ 标注不可避免（那就只能靠高频在线标定）。

场的形式按容量递增：常数 / 线性 / 二次 / 高斯核平滑（带宽σ），均带岭。
特征只用**预测值**（测试时可得），不用真值 ✗ ⇒ 无泄漏。
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

IDX = Path("/data1/xieliangcai/code/gazeformer-exps/data/RGBDGaze/index")
sc = {}
with open(IDX / "index.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        sc[r["rgb_path"]] = (float(r["screen_w"]) * float(r["cm_px_x"]),
                             float(r["screen_h"]) * float(r["cm_px_y"]))

data = defaultdict(list)
for r in csv.DictReader(open(
        "/data1/xieliangcai/code/gazeformer-exps/out/rgbdgaze/analysis/eye224_test.csv",
        encoding="utf-8")):
    data[r["subject"]].append(dict(
        P=np.array([float(r["pred_x"]), float(r["pred_y"])]),
        L=np.array([float(r["label_x"]), float(r["label_y"])]),
        S=np.array(sc[r["path"]])))

subs = sorted(data)
print(f"被试: {subs}  帧数: { {s: len(v) for s, v in data.items()} }")


def feats(P, kind):
    x, y = P[:, 0], P[:, 1]
    if kind == "const":
        return np.ones((len(P), 1))
    if kind == "lin":
        return np.stack([np.ones(len(P)), x - 0.5, y - 0.5], 1)
    if kind == "quad":
        return np.stack([np.ones(len(P)), x - 0.5, y - 0.5,
                         (x - 0.5) ** 2, (x - 0.5) * (y - 0.5), (y - 0.5) ** 2], 1)
    raise ValueError(kind)


def em(corr, v):
    e = [np.hypot((c[0] - d["L"][0]) * d["S"][0], (c[1] - d["L"][1]) * d["S"][1])
         for c, d in zip(corr, v)]
    return np.array(e)


def run(kind, ridge=1e-3, sigma=None):
    """LOSO：返回被留出被试的逐帧误差。"""
    out = []
    for held in subs:
        tr = [d for s in subs if s != held for d in data[s]]
        Ptr = np.array([d["P"] for d in tr])
        Dtr = np.array([d["P"] - d["L"] for d in tr])
        Pt = np.array([d["P"] for d in data[held]])
        corr = []
        if kind == "knn":
            step = 400                      # 训练点下采样，控显存/时间
            Pk, Dk = Ptr[::step], Dtr[::step]
            for p in Pt:
                w = np.exp(-((Pk - p) ** 2).sum(1) / (2 * sigma ** 2))
                if w.sum() < 1e-9:
                    corr.append(p)
                else:
                    corr.append(p - (Dk * w[:, None]).sum(0) / w.sum())
        else:
            X = feats(Ptr, kind)
            A = X.T @ X + ridge * np.eye(X.shape[1])
            coef = np.linalg.solve(A, X.T @ Dtr)
            corr = list(Pt - feats(Pt, kind) @ coef)
        out.append(em(corr, data[held]))
    return np.concatenate(out)


base = np.concatenate([em([d["P"] for d in data[s]], data[s]) for s in subs])
print("=" * 104)
print(f"【误差场可转移性 LOSO】基线 EM 均值 {base.mean():.3f}（中位 {np.median(base):.3f}）"
      f"  >3cm {100 * (base > 3).mean():.2f}%")
print("=" * 104)
print(f"  {'场的形式':<28}{'EM均值':>9}{'vs基线':>9}{'中位':>8}{'>3cm':>8}{'>5cm':>8}{'max':>7}")
for kind, tag in (("const", "常数（全局平均误差）"), ("lin", "线性（1阶）"),
                  ("quad", "二次（2阶）")):
    for ridge in (1e-3, 1e-1, 1.0):
        e = run(kind, ridge)
        print(f"  {tag + f' 岭={ridge}':<28}{e.mean():>9.3f}"
              f"{100 * (e.mean() / base.mean() - 1):>8.1f}%{np.median(e):>8.3f}"
              f"{100 * (e > 3).mean():>7.2f}%{100 * (e > 5).mean():>7.2f}%{e.max():>7.2f}")
for sg in (0.10, 0.20, 0.35):
    e = run("knn", sigma=sg)
    print(f"  {'核平滑 σ=%s' % sg:<28}{e.mean():>9.3f}"
          f"{100 * (e.mean() / base.mean() - 1):>8.1f}%{np.median(e):>8.3f}"
          f"{100 * (e > 3).mean():>7.2f}%{100 * (e > 5).mean():>7.2f}%{e.max():>7.2f}")
print("\n  参考：每帧已知的在线常数（上限）= 1.028（−30.2%）| 无标注时先看本表能否接近")
