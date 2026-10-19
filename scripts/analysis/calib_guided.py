"""引导式标定的模拟：量化「标定点是否覆盖屏幕」值多少（纯 CPU）。

背景（§6/§7 已证）：标定窗口只取"开头 10%"时，7 个方法全失效 ✗，且 α 安全层救不了
（验证半集与拟合半集同属那个错配 regime ⇒ 看不见"窗口与评估段的分布错配"）
⇒ 瓶颈在**标定协议**，不在估计器。

三种选点规则（方法固定为 §7 的赢家 poly2 + α 内部验证）：
  R1 window-spread ：当前协议——会话开头 frac 内等距取 K 帧（严格因果）
  R2 guided-all    ：**引导式上界（模拟）**——在全时段按「标签覆盖屏幕」选 K 点
                     （等价于"App 指示用户依次看屏幕上这 K 个位置"；用真值选点属模拟，
                       因为真实系统是自己指定这些点 ⇒ 明确标注为上界，不是可部署结果）
  R3 guided-prefix ：引导式但只能在会话开头 frac 内选（"真实引导标定只能发生在开头"）
评估集：该会话**除这 K 个标定帧外**的全部帧；每条规则各自算自己的**不标定**参照，
保证"相对增益"在同一帧集上比较。

另报两个可解释量：
  · 覆盖：评估帧的真值落在任一标定点 r 邻域内的比例（r = 屏对角线 15%）
  · 标定点标签占据的 5×5 网格格子数
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
    """每轴二次多项式：返回 corr(位置) -> 修正坐标。"""
    c = []
    for ax in (0, 1):
        p = P[cal, ax]
        X = np.stack([np.ones_like(p), p, p ** 2], 1)
        c.append(np.linalg.lstsq(X, L[cal, ax], rcond=None)[0])
    return lambda pi: np.array([c[0] @ np.array([1, pi[0], pi[0] ** 2]),
                                c[1] @ np.array([1, pi[1], pi[1] ** 2])])


def pick_rule(rule, idx, K, frac):
    if rule == "R1":                      # 窗口内等距
        pool = idx[:max(int(len(idx) * frac), 30)]
        return [pool[t] for t in np.unique(
            np.linspace(0, len(pool) - 1, min(K, len(pool))).astype(int))]
    pool = idx[:max(int(len(idx) * frac), 30)] if rule == "R3" else idx
    # 覆盖优先：5x5 网格，每格取"标签最靠近格心"的帧，按格心距离升序轮替取满 K
    cells = defaultdict(list)
    for i in pool:
        cx = min(4, max(0, int(L[i, 0] * 5)))
        cy = min(4, max(0, int(L[i, 1] * 5)))
        cells[(cx, cy)].append(i)
    for k in cells:
        ccx, ccy = (k[0] + 0.5) / 5.0, (k[1] + 0.5) / 5.0
        cells[k].sort(key=lambda i: (L[i, 0] - ccx) ** 2 + (L[i, 1] - ccy) ** 2)
    keys = sorted(cells)
    sel, ptr = [], {k: 0 for k in keys}
    while len(sel) < min(K, len(pool)):
        grew = False
        for k in keys:
            if len(sel) >= min(K, len(pool)):
                break
            if ptr[k] < len(cells[k]):
                sel.append(cells[k][ptr[k]]); ptr[k] += 1; grew = True
        if not grew:
            break
    return sorted(sel)


def run(rule, K, frac=0.10):
    errs, base, cov, ngrid = [], [], [], []
    for s in SESS:
        idx = sess[s]
        if len(idx) < 80:
            continue
        cal = pick_rule(rule, idx, K, frac)
        if len(cal) < 6:
            continue
        ev = [i for i in idx if i not in set(cal)]
        A_, B_ = cal[::2], cal[1::2]
        fA = fit_poly2(A_)
        cb = np.array([fA(P[i]) - P[i] for i in B_])
        best_a, best_e = 0.0, 1e9
        for a in AS:
            e = np.mean([emv(P[i] + a * cb[j], L[i], str(paths[i]))
                         for j, i in enumerate(B_)])
            if e < best_e:
                best_e, best_a = e, a
        fF = fit_poly2(cal)
        lp = L[cal]
        for i in ev:
            errs.append(emv(P[i] + best_a * (fF(P[i]) - P[i]), L[i], str(paths[i])))
            base.append(emv(P[i], L[i], str(paths[i])))
            dd = np.linalg.norm(lp - L[i][None, :], axis=1).min()
            cov.append(dd < 0.15)
        cx = np.clip((lp[:, 0] * 5).astype(int), 0, 4)
        cy = np.clip((lp[:, 1] * 5).astype(int), 0, 4)
        ngrid.append(len(set(zip(cx.tolist(), cy.tolist()))))
    return (np.array(errs), np.array(base), np.array(cov), float(np.mean(ngrid)))


print("=" * 104)
print("引导式标定模拟：poly2 + α 内部验证；三种选点规则（评估集 = 该会话除标定帧外全部帧）")
print("=" * 104)
print(f"  {'规则':<16}{'K':>4}{'标定占5x5格数':>14}{'覆盖(≤15%对角)':>16}"
      f"{'不标定':>9}{'poly2+α':>12}{'相对增益':>10}")
for rule, name in (("R1", "window-spread"), ("R3", "guided-prefix10%"), ("R2", "guided-all(上界)")):
    for K in (9, 25, 50):
        e, b, cov, ng = run(rule, K)
        print(f"  {name:<16}{K:>4}{ng:>14.1f}{100 * cov.mean():>15.1f}%"
              f"{b.mean():>9.3f}{e.mean():>12.3f}{100 * (e.mean() / b.mean() - 1):>9.1f}%")
    print()
