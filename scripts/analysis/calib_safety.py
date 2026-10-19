"""标定安全层：让"标定"永远不伤害（宁可不修正，也不要变差）。

背景（刚测出的现实问题）：
  用"开头 K 帧"做一次性标定时，我们的每格场会把误差从 1.47 抬到 1.8~2.65 ✗✗
  （原因：前段注视分布与后段不同 regime，估出的场系统性错位）
⇒ 部署上必须能"自己发现标定不可靠并放弃"。

两个安全机制（都只用标定窗口的标签，零额外标注 ✓）：
  ① 一致性门控：把标定窗口对半切，各自估一套修正；两者差异过大 ⇒ 判为不可靠 ⇒ 不修正
  ② 幅度钳制：把修正量限制在 ±C（避免个别帧被推坏）

对照：B 裸标定 / B+门控 / B+钳制 / B+两者
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/xieliangcai/code/gazeformer-exps")
CSV = str(ROOT / "out/rgbdgaze/analysis/eye224_test.csv")
G, MIN = 8, 8
D2 = G * G

sc = {}
with open(ROOT / "data/RGBDGaze/index/index.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        sc[r["rgb_path"]] = (float(r["screen_w"]) * float(r["cm_px_x"]),
                             float(r["screen_h"]) * float(r["cm_px_y"]))

subj = defaultdict(list)
for r in csv.DictReader(open(CSV, encoding="utf-8")):
    subj[r["subject"]].append(dict(fi=int(r["frame_index"]), path=r["path"],
                                   P=np.array([float(r["pred_x"]), float(r["pred_y"])]),
                                   L=np.array([float(r["label_x"]), float(r["label_y"])])))
for s in subj:
    subj[s].sort(key=lambda r: r["fi"])
SUB = sorted(subj)


def cidx(pt):
    return min(G - 1, max(0, int(pt[0] * G))) * G + min(G - 1, max(0, int(pt[1] * G)))


def em(pred, lab, path):
    w, h = sc[path]
    return float(np.hypot((pred[0] - lab[0]) * w, (pred[1] - lab[1]) * h))


def field_from(frames):
    """由若干帧估一张场（每格平均偏差），返回 (D2,2) 与 计数。"""
    acc = np.zeros((D2, 2)); cnt = np.zeros(D2)
    for r in frames:
        c = cidx(r["P"])
        acc[c] += r["P"] - r["L"]
        cnt[c] += 1
    f = np.zeros((D2, 2))
    m = cnt > 0
    f[m] = acc[m] / cnt[m][:, None]
    return f, cnt


def run(mode, K, gate=False, clamp=np.inf):
    A = B = 0.0; B3 = 0.0; n = 0; nc = 0
    for held in SUB:
        v = subj[held]
        N = len(v)
        if N < 300:
            continue
        cal = (np.arange(min(K, N // 2)) if mode == "head"
               else np.unique(np.linspace(0, N - 1, min(K, N)).astype(int)))
        ev = np.setdiff1d(np.arange(N), cal)
        calr = [v[i] for i in cal]
        # 场（用标定窗口，按预测位置归格）
        f_all, cnt_all = field_from(calr)
        # 一致性门控：对半切，比较两张场的差异（只在两者都有样本的格上比）
        ok = True
        if gate and len(calr) >= 8:
            h = len(calr) // 2
            f1, c1 = field_from(calr[:h]); f2, c2 = field_from(calr[h:])
            both = (c1 > 0) & (c2 > 0)
            ok = (not both.any()) or (np.linalg.norm(f1[both] - f2[both], axis=1).mean()
                                      < 0.06)
        for i, r in enumerate(v):
            P, L = r["P"], r["L"]
            corr = P
            if ok or not gate:
                c = cidx(P); i0, j0 = c // G, c % G
                acc = np.zeros(2); cnt = 0.0
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        ii, jj = i0 + di, j0 + dj
                        if 0 <= ii < G and 0 <= jj < G:
                            cc = ii * G + jj
                            if cnt_all[cc] > 0:
                                acc += f_all[cc] * cnt_all[cc]
                                cnt += cnt_all[cc]
                if cnt >= MIN:
                    d = acc / cnt
                    d = np.clip(d, -clamp, clamp)
                    corr = P - d
                    if i in set(ev.tolist()):
                        nc += 1
            e_a = em(P, L, r["path"]); e_b = em(corr, L, r["path"])
            if i in set(ev.tolist()):
                A += e_a; B += e_b; B3 += e_b > 3; n += 1
    return A / n, B / n, B3 / n, n, nc / max(n, 1)


print("=" * 104)
print("【标定安全层】能否让标定永不伤害（评估帧上的均值 / >3cm 占比）")
print("=" * 104)
print(f"  {'模式':>7}{'K':>6}{'A不修正':>9}{'B裸标定':>9}{'+门控':>9}{'+钳制':>9}{'+两者':>9}"
      f"{'裸>3cm':>9}{'两者>3cm':>10}")
for mode in ("head", "spread"):
    for K in (20, 50, 200):
        r0 = run(mode, K)
        r1 = run(mode, K, gate=True)
        r2 = run(mode, K, clamp=0.05)
        r3 = run(mode, K, gate=True, clamp=0.05)
        print(f"  {mode:>7}{K:>6}{r0[0]:>9.3f}{r0[1]:>9.3f}{r1[1]:>9.3f}{r2[1]:>9.3f}"
              f"{r3[1]:>9.3f}{r0[2] * 100:>8.1f}%{r3[2] * 100:>9.1f}%")
