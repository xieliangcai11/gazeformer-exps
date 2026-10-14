"""顺着"误差是注视位置的平滑误差场"深挖：

E1 主动选点（coverage-driven） vs 周期性选点 —— 同等标注预算下谁更强？
   这同时回答"那 0.17 秒的已知答案该在什么时候取"。
   策略 periodic : 每 k 帧取一帧
   策略 coverage : 仅当"当前注视位置离已有池子最远"（超出阈值 θ）时才取 ⇒ 铺满屏幕
   估计器固定：池（最近 20 个已知点）中取注视位置最近 10 个，用其平均偏差修正。

E2 误差场的结构刻画：
   · 把屏幕划 4x4，用**留一法**的格内平均偏差做修正 ⇒ 该格场"可解释的误差份额"
   · 场的时间稳定性：前半段拟合的场 vs 后半段（格内相关）
   · 跨被试形状相似度：去掉各被试均值后，格向量之间的相关（决定能否学"场的先验"）
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

IDX = Path("/data1/xieliangcai/code/gazeformer-exps/data/RGBDGaze/index")
GC_IDX = Path("/data1/xieliangcai/code/gazeformer-exps/data/GazeCapture/index")
sc = {}
for _idx in (IDX, GC_IDX):
    if not _idx.exists():
        continue
    with open(_idx / "index.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            pk = "rgb_path" if "rgb_path" in r else "frame_path"
            sc[r[pk]] = (float(r["screen_w"]) * float(r["cm_px_x"]),
                         float(r["screen_h"]) * float(r["cm_px_y"]))
CSV = sys.argv[1]

sess = defaultdict(list)
for r in csv.DictReader(open(CSV, encoding="utf-8")):
    sess[r["path"].rsplit("/", 1)[0]].append(
        dict(P=np.array([float(r["pred_x"]), float(r["pred_y"])]),
             L=np.array([float(r["label_x"]), float(r["label_y"])]),
             S=sc[r["path"]]))
for s in sess:
    sess[s].sort(key=lambda r: 0)  # 保持文件顺序（已是帧序）

base = np.concatenate([np.array([float(np.hypot((r["P"][0] - r["L"][0]) * r["S"][0],
                                             (r["P"][1] - r["L"][1]) * r["S"][1]))
                                 for r in sess[s]]) for s in sess])
print("=" * 100)
print(f"【误差场深挖】{Path(CSV).stem}   不修正基线 EM {base.mean():.3f}"
      f"（>3cm {100 * (base > 3).mean():.2f}%）")
print("=" * 100)


def run(strategy, param, C=10, L=20, warm=60):
    """strategy: 'none' | 'periodic'(每 param 帧) | 'coverage'(阈值 param)。"""
    errs = []
    for s, v in sess.items():
        pool = []
        for t, r in enumerate(v):
            if t >= warm:
                if strategy == "periodic":
                    take = (t % param == 0)
                elif strategy == "coverage":
                    if pool:
                        d = min(float(np.hypot((q[1][0] - r["L"][0]) * r["S"][0],
                                               (q[1][1] - r["L"][1]) * r["S"][1]))
                                for q in pool)
                        take = d > param
                    else:
                        take = True
                else:
                    take = False
                if take:
                    pool.append((t, r["L"], r["P"]))
                    pool = pool[-L:]
                if not pool:
                    corr = r["P"]
                else:
                    D = np.array([q[2] - q[1] for q in pool])
                    Ll = np.array([q[1] for q in pool])
                    take_i = np.argsort(((Ll - r["L"]) ** 2).sum(1))[:min(C, len(pool))]
                    corr = r["P"] - D[take_i].mean(0)
                errs.append(float(np.hypot((corr[0] - r["L"][0]) * r["S"][0],
                                           (corr[1] - r["L"][1]) * r["S"][1])))
            else:
                pool.append((t, r["L"], r["P"])); pool = pool[-L:]
                errs.append(float(np.hypot((r["P"][0] - r["L"][0]) * r["S"][0],
                                           (r["P"][1] - r["L"][1]) * r["S"][1])))
    a = np.array(errs)
    return a.mean(), 100 * (a > 3).mean()


def duty(strategy, param, warm=60):
    """统计实际取点数占比。"""
    n, k = 0, 0
    for s, v in sess.items():
        pool = []
        for t, r in enumerate(v):
            if t < warm:
                pool.append((t, r["L"], r["P"])); pool = pool[-20:]; n += 1; k += 1; continue
            n += 1
            if strategy == "periodic":
                take = (t % param == 0)
            else:
                take = (not pool) or min(float(np.hypot((q[1][0] - r["L"][0]) * r["S"][0],
                                                        (q[1][1] - r["L"][1]) * r["S"][1]))
                                         for q in pool) > param
            if take:
                pool.append((t, r["L"], r["P"])); pool = pool[-20:]; k += 1
    return 100 * k / n


print(f"\n【E1】同等预算下：周期性 vs 主动铺点")
print(f"  {'策略':<30}{'预算(取点占比)':>16}{'EM均值':>9}{'vs基线':>9}{'>3cm':>8}")
for k in (600, 300, 150, 90, 60, 30):
    m, t3 = run("periodic", k)
    print(f"  {'periodic 每 %d 帧' % k:<30}{duty('periodic', k):>15.1f}%{m:>9.3f}"
          f"{100 * (m / base.mean() - 1):>8.1f}%{t3:>7.2f}%")
for th in (0.12, 0.20, 0.30, 0.45, 0.70):
    m, t3 = run("coverage", th)
    print(f"  {'coverage 阈值 %.2f' % th:<30}{duty('coverage', th):>15.1f}%{m:>9.3f}"
          f"{100 * (m / base.mean() - 1):>8.1f}%{t3:>7.2f}%")

print(f"\n【E2】误差场结构（4x4 格，留一法格内均值修正）")
G = 4


def cell(L):
    return min(G - 1, int(L[0] * G)), min(G - 1, int(L[1] * G))


loo, fields = [], defaultdict(list)
for s, v in sess.items():
    by = defaultdict(list)
    for i, r in enumerate(v):
        by[cell(r["L"])].append(i)
    for c, idxs in by.items():
        if len(idxs) < 5:
            continue
        D = np.array([[v[i]["P"][0] - v[i]["L"][0], v[i]["P"][1] - v[i]["L"][1]]
                      for i in idxs])
        S = np.array([v[i]["S"] for i in idxs])
        tot = D.sum(0)
        for k2, i in enumerate(idxs):
            mu = (tot - D[k2]) / (len(idxs) - 1)
            corr = v[i]["P"] - mu
            loo.append(float(np.hypot((corr[0] - v[i]["L"][0]) * v[i]["S"][0],
                                      (corr[1] - v[i]["L"][1]) * v[i]["S"][1])))
        fields[s].append((c, D.mean(0), len(idxs)))
loo = np.array(loo)
print(f"  留一格内均值修正的 EM = {loo.mean():.3f}"
      f"（vs 基线 {base.mean():.3f}，{100 * (loo.mean() / base.mean() - 1):+.1f}%）"
      f"  ⇒ 场可解释的份额")
print(f"  覆盖帧数 {len(loo)}/{len(base)}（{100 * len(loo) / len(base):.0f}% 的帧落在样本充足的格里）")

# 场的时间稳定性：前半 vs 后半
cs = []
for s, v in sess.items():
    H = len(v) // 2
    a, b = defaultdict(list), defaultdict(list)
    for i, r in enumerate(v):
        (a if i < H else b)[cell(r["L"])].append(i)
    common = [c for c in a if c in b and len(a[c]) >= 5 and len(b[c]) >= 5]
    if len(common) < 3:
        continue
    fa = np.array([[np.mean([v[i]["P"][0] - v[i]["L"][0] for i in a[c]]),
                    np.mean([v[i]["P"][1] - v[i]["L"][1] for i in a[c]])]
                   for c in common]).ravel()
    fb = np.array([[np.mean([v[i]["P"][0] - v[i]["L"][0] for i in b[c]]),
                    np.mean([v[i]["P"][1] - v[i]["L"][1] for i in b[c]])]
                   for c in common]).ravel()
    cs.append(np.corrcoef(fa, fb)[0, 1])
print(f"  场的时间稳定性（前半 vs 后半，{len(cs)} 个 session）："
      f"中位相关 {np.median(cs):+.2f}  范围 {min(cs):+.2f}~{max(cs):+.2f}")

# 跨被试形状相似度（去掉各自均值）
names = sorted(fields)
vecs = {}
for s in names:
    d = {c: m for c, m, n in fields[s] if n >= 20}
    vecs[s] = d
common = set.intersection(*[set(v) for v in vecs.values()]) if vecs else set()
if common:
    M = np.array([[vecs[s][c][0], vecs[s][c][1]] for s in names for c in sorted(common)])
    W = M.reshape(len(names), -1)
    Wc = W - W.mean(1, keepdims=True)
    C = np.corrcoef(Wc)
    print(f"  跨被试场形状相关（{len(common)} 个公共格）："
          f"非对角中位 {np.median(C[~np.eye(len(names), dtype=bool)]):+.2f}")
else:
    print("  跨被试：公共格不足，跳过")
