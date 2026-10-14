"""校准容量定律验证（数据集无关）：同一 session 内用前 R% 帧拟合修正，评估其余帧。

容量阶梯（输出端，只改预测不改权重）：
  const : 常数偏移                      2 参数
  lin   : 仿射（2x2 + 偏置）            6 参数
  quad  : 二次多项式（含交叉项）        12 参数
  knn   : 注视位置最近 C 个校准帧均值    ~2C 有效自由度
若"低预算时低容量更稳、高预算时高容量反超" ⇒ 容量定律复现（与 RGBDGaze 一致）。
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/xieliangcai/code/gazeformer-exps")
CSV = sys.argv[1]
WARM = 0
sc = {}
for _idx in (ROOT / "data/RGBDGaze/index", ROOT / "data/GazeCapture/index"):
    if not _idx.exists():
        continue
    with open(_idx / "index.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            pk = "rgb_path" if "rgb_path" in r else "frame_path"
            sc[r[pk]] = (float(r["screen_w"]) * float(r["cm_px_x"]),
                         float(r["screen_h"]) * float(r["cm_px_y"]))

sess = defaultdict(list)
for r in csv.DictReader(open(CSV, encoding="utf-8")):
    # 条件粒度：若有 activity/朝向列（GC 的 dump 把它写成 activity=朝向），
    # 则按 被试×朝向 分组；否则按目录（= 被试×活动）分组。
    cond = r.get("activity") or ""
    key = (r["subject"], cond) if cond else r["path"].rsplit("/", 1)[0]
    sess[key].append(
        dict(fi=int(r["frame_index"]),
             P=np.array([float(r["pred_x"]), float(r["pred_y"])]),
             L=np.array([float(r["label_x"]), float(r["label_y"])]),
             S=sc[r["path"]]))
for s in sess:
    sess[s].sort(key=lambda r: r["fi"])


def basis(P, kind):
    x, y = P[:, 0] - 0.5, P[:, 1] - 0.5
    if kind == "const":
        return np.ones((len(P), 1))
    if kind == "lin":
        return np.stack([np.ones(len(P)), x, y], 1)
    if kind == "quad":
        return np.stack([np.ones(len(P)), x, y, x * x, x * y, y * y], 1)
    raise ValueError(kind)


def run(kind, R, ridge=1e-3, C=9):
    errs = []
    for s, v in sess.items():
        n = len(v)
        k = max(1, int(n * R))
        if n - k < 20:
            continue
        Pc = np.array([r["P"] for r in v[:k]])
        Lc = np.array([r["L"] for r in v[:k]])
        D = Pc - Lc
        if kind == "knn":
            pass
        else:
            X = basis(Pc, kind)
            A = X.T @ X + ridge * np.eye(X.shape[1])
            coef = np.linalg.solve(A, X.T @ D)
        for r in v[k:]:
            if kind == "knn":
                d = ((Lc - r["P"]) ** 2).sum(1)
                sel = np.argsort(d)[:min(C, len(D))]
                corr = r["P"] - D[sel].mean(0)
            else:
                corr = r["P"] - basis(r["P"][None, :], kind) @ coef
                corr = corr[0]
            errs.append(float(np.hypot((corr[0] - r["L"][0]) * r["S"][0],
                                       (corr[1] - r["L"][1]) * r["S"][1])))
    return np.array(errs)


base = np.array([float(np.hypot((r["P"][0] - r["L"][0]) * r["S"][0],
                                (r["P"][1] - r["L"][1]) * r["S"][1]))
                 for s in sess for r in sess[s]])
print("=" * 96)
print(f"【校准容量定律】{Path(CSV).stem}   基线 EM {base.mean():.3f}"
      f"  >3cm {100 * (base > 3).mean():.2f}%   sessions={len(sess)}")
print("=" * 96)
print(f"  {'协议':<26}{'EM均值':>9}{'vs基线':>9}{'中位':>8}{'>3cm':>8}")
for R in (0.05, 0.10, 0.30, 0.50):
    print(f"  {'--- R=%.0f%%（校准用前若干帧）' % (R * 100):<26}")
    for kind, tag in (("const", "常数 2 参数"), ("lin", "仿射 6 参数"),
                      ("quad", "二次 12 参数"), ("knn", "空间最近 9 帧")):
        e = run(kind, R)
        print(f"    {tag:<24}{e.mean():>9.3f}{100 * (e.mean() / base.mean() - 1):>8.1f}%"
              f"{np.median(e):>8.3f}{100 * (e > 3).mean():>7.2f}%")
