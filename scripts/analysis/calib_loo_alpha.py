"""B：小 K 时的 α 选择改为**内部留一(LOO)**，并公正复测 9/13 点引导式标定。

问题（§8 的方法学提醒）：此前 α 用"标定集对半切"，K=9 时验证集只剩 4~5 点 ⇒ 验证本身是噪声
⇒ K=9 的结果不可用于评价"真实 9 点标定" ✗。本脚本：
  · α 选择改为 LOO：对每个标定帧 j，用 cal\\{j} 拟合 poly2 预测 j，得到 K 个"留一残差"
    ⇒ 在其上调 α（用全部 K 点做最终拟合 ✓，验证样本数 = K 而非 K/2 ✓）
  · 选点规则增加 **grid-guided**：按屏幕 n×n 网格格心选最近的帧（= 真实"9/13 点引导标定"的模拟 ✓）
  · 覆盖指标增加 **bbox 面积占比**（跨 K 可比，直接刻画"是否铺满屏幕" ✓）
对比：half-split(旧) vs LOO(新) × {grid-guided, window-spread} × K∈{9,13,25,50}
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/xieliangcai/code/gazeformer-exps")
NPZ = ROOT / "out/rgbdgaze/analysis/feats_eye224_test.npz"
AS = (0.0, 0.25, 0.5, 0.75, 1.0)

sc = {}
with open(ROOT / "data/RGBDGaze/index/index.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        sc[r["rgb_path"]] = (float(r["screen_w"]) * float(r["cm_px_x"]),
                             float(r["screen_h"]) * float(r["cm_px_y"]))
d = np.load(NPZ, allow_pickle=True)
P, L = d["pred"].astype(np.float64), d["label"].astype(np.float64)
paths, fi = d["path"], d["frame_index"]
sess = defaultdict(list)
for i in range(len(P)):
    sess[str(paths[i]).rsplit("/", 1)[0]].append(i)
for s in sess:
    sess[s].sort(key=lambda i: fi[i])
SESS = sorted(sess)


def emv(pred, lab, path):
    w, h = sc[path]
    return float(np.hypot((pred[0] - lab[0]) * w, (pred[1] - lab[1]) * h))


def fit_poly2(cal):
    """min points 6（每轴 3 参）⇒ 少于 4 个点时退化为 affine（2 参/轴）。"""
    c = []
    deg = 2 if len(cal) >= 8 else 1
    for ax in (0, 1):
        p = P[cal, ax]
        X = np.stack([np.ones_like(p)] + [p ** k for k in range(1, deg + 1)], 1)
        c.append((deg, np.linalg.lstsq(X, L[cal, ax], rcond=None)[0]))

    def f(pi):
        return np.array([np.dot(np.array([1.0] + [pi[0] ** k for k in range(1, c[0][0] + 1)]), c[0][1]),
                         np.dot(np.array([1.0] + [pi[1] ** k for k in range(1, c[1][0] + 1)]), c[1][1])])
    return f


def pick_grid(idx, K):
    """按屏幕 n×n 网格格心选最近帧（模拟真实引导标定：点铺满屏幕）。"""
    n = max(2, int(round(np.sqrt(K))))
    cents = [((i + 0.5) / n, (j + 0.5) / n) for i in range(n) for j in range(n)]
    labels = L[idx]
    sel, used = [], set()
    for cx, cy in cents:
        dd = np.hypot(labels[:, 0] - cx, labels[:, 1] - cy)
        order = np.argsort(dd)
        for o in order:
            if int(o) not in used:
                used.add(int(o)); sel.append(idx[int(o)]); break
        if len(sel) >= K:
            break
    return sorted(sel[:K])


def pick_window_spread(idx, K, frac=0.10):
    pool = idx[:max(int(len(idx) * frac), 30)]
    return [pool[t] for t in np.unique(
        np.linspace(0, len(pool) - 1, min(K, len(pool))).astype(int))]


def choose_alpha(cal, mode):
    """返回 (α, LOO 或 B 半的验证误差)。mode: 'loo' | 'half'"""
    if mode == "half":
        A_, B_ = cal[::2], cal[1::2]
        f = fit_poly2(A_)
        cb = np.array([f(P[i]) - P[i] for i in B_])
        best = (0.0, 1e9)
        for a in AS:
            e = np.mean([emv(P[i] + a * cb[j], L[i], str(paths[i]))
                         for j, i in enumerate(B_)])
            if e < best[1]:
                best = (a, e)
        return best[0]
    # LOO：每个点用其余点拟合（用全部 K 点做最终拟合，验证样本 = K）
    cb = np.zeros((len(cal), 2))
    for j in range(len(cal)):
        rest = [c for t, c in enumerate(cal) if t != j]
        if len(rest) < 3:
            cb[j] = 0.0
            continue
        f = fit_poly2(rest)
        cb[j] = f(P[cal[j]]) - P[cal[j]]
    best = (0.0, 1e9)
    for a in AS:
        e = np.mean([emv(P[i] + a * cb[j], L[i], str(paths[i]))
                     for j, i in enumerate(cal)])
        if e < best[1]:
            best = (a, e)
    return best[0]


def run(rule, K, mode):
    errs, base, alphas, area, cov, occ = [], [], [], [], [], []
    for s in SESS:
        idx = sess[s]
        if len(idx) < 80:
            continue
        cal = pick_grid(idx, K) if rule == "grid" else pick_window_spread(idx, K)
        if len(cal) < 6:
            continue
        ev = [i for i in idx if i not in set(cal)]
        a = choose_alpha(cal, mode)
        alphas.append(a)
        fF = fit_poly2(cal)
        lp = L[cal]
        area.append((lp[:, 0].max() - lp[:, 0].min()) * (lp[:, 1].max() - lp[:, 1].min()))
        cx = np.clip((lp[:, 0] * 5).astype(int), 0, 4)
        cy = np.clip((lp[:, 1] * 5).astype(int), 0, 4)
        occ.append(len(set(zip(cx.tolist(), cy.tolist()))))
        for i in ev:
            errs.append(emv(P[i] + a * (fF(P[i]) - P[i]), L[i], str(paths[i])))
            base.append(emv(P[i], L[i], str(paths[i])))
            cov.append(np.linalg.norm(lp - L[i][None, :], axis=1).min() < 0.15)
    return (np.array(errs), np.array(base), np.mean(alphas),
            np.mean(area), np.mean(cov), np.mean(occ))


print("=" * 114)
print("B：α 选择 half-split(旧) vs LOO(新)；选点 grid-guided(模拟引导标定) vs window-spread")
print("=" * 114)
print(f"  {'规则':<14}{'α选择':<8}{'K':>4}{'bbox面积占比':>13}{'5x5格数':>9}{'覆盖15%':>9}"
      f"{'不标定':>9}{'poly2+α':>10}{'相对增益':>10}{'ᾱ':>7}")
for rule, rn in (("grid", "grid-guided"), ("window", "window-spread")):
    for mode in ("half", "loo"):
        for K in (9, 13, 25, 50):
            e, b, al, ar, cv, oc = run(rule, K, mode)
            print(f"  {rn:<14}{mode:<8}{K:>4}{ar:>13.2f}{oc:>9.1f}{100*cv:>8.1f}%"
                  f"{b.mean():>9.3f}{e.mean():>10.3f}{100*(e.mean()/b.mean()-1):>9.1f}%{al:>7.2f}")
    print()
