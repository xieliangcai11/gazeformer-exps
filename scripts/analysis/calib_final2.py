"""最终定型 + 关键对照：把"真值平滑"这个平凡来源分离出去。

背景：λ 越小（记忆越短）越好，说明估计器越来越像"减掉最近几帧的偏置"。
若真值本身在 0.1s 内几乎不变，收益可能只是"复制上一帧"这种平凡效应 ✗。
故必须测持续性对照（全部因果）：
  P1: pred = 上一帧真值（label_{t-1}）
  P2: pred = 上一帧预测（pred_{t-1}）
  P3: pred = 最近 3 帧真值均值
  M : 我们的估计器（模型预测 + 仿射 + 核加权场，遗忘 λ）
⇒ 若 M 显著优于 P1/P3，才说明收益来自"估计器/模型信息"，不是真值平滑。

同时放开门限（固定下限惩罚，不再用 n<MIN 卡死）继续下扫 λ，并测运动项。
"""
import csv
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

ROOT = Path("/data1/xieliangcai/code/gazeformer-exps")
CSV = ROOT / "out/rgbdgaze/analysis/eye224_test.csv"
G, MIN_C = 8, 8
FPS = 30.0

sc = {}
with open(ROOT / "data/RGBDGaze/index/index.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        sc[r["rgb_path"]] = (float(r["screen_w"]) * float(r["cm_px_x"]),
                             float(r["screen_h"]) * float(r["cm_px_y"]))
sess = defaultdict(list)
for r in csv.DictReader(open(CSV, encoding="utf-8")):
    sess[r["path"].rsplit("/", 1)[0]].append(
        dict(fi=int(r["frame_index"]), path=r["path"], subj=r["subject"], act=r["activity"],
             P=np.array([float(r["pred_x"]), float(r["pred_y"])]),
             L=np.array([float(r["label_x"]), float(r["label_y"])])))
for s in sess:
    sess[s].sort(key=lambda r: r["fi"])
SESS = sorted(sess)


def cidx(pt):
    return min(G - 1, max(0, int(pt[0] * G))) * G + min(G - 1, max(0, int(pt[1] * G)))


def emv(pred, lab, path):
    w, h = sc[path]
    return float(np.hypot((pred[0] - lab[0]) * w, (pred[1] - lab[1]) * h))


class Est:
    def __init__(self, lam=0.9, sigma=3.0, rad=2, min_cell=MIN_C, pen_floor=0.05,
                 motion=False):
        self.lam, self.sigma, self.rad, self.min_cell = lam, sigma, rad, min_cell
        self.pen_floor, self.motion = pen_floor, motion
        k = 4 if motion else 3
        self.k = k
        self.n = 0.0
        self.M = np.zeros((k, k)); self.Rx = np.zeros(k); self.Ry = np.zeros(k)
        self.g = defaultdict(lambda: [0.0, 0.0, 0.0])

    def _u(self, P, v):
        return (np.array([1.0, P[0], P[1], v / 10.0]) if self.motion
                else np.array([1.0, P[0], P[1]]))

    def beta(self, P, v):
        if self.n <= 0:
            return None
        pen = max(0.01 * self.n, self.pen_floor)        # 固定下限惩罚 ⇒ 极小样本也稳定
        A = self.M + pen * np.diag([0.0] + [1.0] * (self.k - 1))
        try:
            u = self._u(P, v)
            return np.array([u @ np.linalg.solve(A, self.Rx),
                             u @ np.linalg.solve(A, self.Ry)])
        except np.linalg.LinAlgError:
            return None

    def correct(self, P, v):
        bv = self.beta(P, v)
        if bv is None:
            bv = np.zeros(2)
        p1 = P - bv
        c = cidx(p1); i0, j0 = c // G, c % G
        acc = np.zeros(2); cnt = 0.0
        for di in range(-self.rad, self.rad + 1):
            for dj in range(-self.rad, self.rad + 1):
                ii, jj = i0 + di, j0 + dj
                if not (0 <= ii < G and 0 <= jj < G):
                    continue
                cc = self.g[ii * G + jj]
                if cc[0] <= 0:
                    continue
                w = 1.0 if self.sigma is None else float(
                    np.exp(-(di * di + dj * dj) / (2 * self.sigma ** 2)))
                acc += w * np.array([cc[1], cc[2]])
                cnt += w * cc[0]
        return ((p1 - acc / cnt) if cnt >= self.min_cell else p1), p1, bv

    def update(self, P, L, v, p1, bv):
        if self.lam != 1.0:
            self.n *= self.lam
            self.M *= self.lam
            self.Rx *= self.lam; self.Ry *= self.lam
            for kk in self.g:
                c = self.g[kk]
                c[0] *= self.lam; c[1] *= self.lam; c[2] *= self.lam
        u = self._u(P, v)
        dx, dy = P[0] - L[0], P[1] - L[1]
        self.n += 1.0
        self.M += np.outer(u, u)
        self.Rx += dx * u; self.Ry += dy * u
        c = self.g[cidx(p1)]
        c[0] += 1.0
        c[1] += float(dx) - float(bv[0]); c[2] += float(dy) - float(bv[1])


def run_m(**kw):
    errs, pa, ps = [], defaultdict(list), defaultdict(list)
    for s in SESS:
        fd = Est(**kw)
        prev = None
        for r in sess[s]:
            P = r["P"]
            w, h = sc[r["path"]]
            v = 0.0 if prev is None else float(
                np.hypot((P[0] - prev[0]) * w, (P[1] - prev[1]) * h)) * FPS
            corr, p1, bv = fd.correct(P, v)
            e = emv(corr, r["L"], r["path"])
            errs.append(e); pa[r["act"]].append(e); ps[r["subj"]].append(e)
            fd.update(P, r["L"], v, p1, bv)
            prev = P
    return np.array(errs), pa, ps


def run_controls():
    """因果持续性对照。"""
    p1e, p2e, p3e = [], [], []
    for s in SESS:
        hist = deque(maxlen=3)
        prevP = None
        for r in sess[s]:
            if prevP is not None:
                p2e.append(emv(prevP, r["L"], r["path"]))
            if hist:
                p1e.append(emv(hist[-1], r["L"], r["path"]))
                p3e.append(emv(np.mean(hist, axis=0), r["L"], r["path"]))
            hist.append(r["L"]); prevP = r["P"]
    return np.array(p1e), np.array(p2e), np.array(p3e)


base = np.array([emv(r["P"], r["L"], r["path"]) for s in sess for r in sess[s]])
print("=" * 112)
print(f"基线（模型原始预测）{base.mean():.3f}")
print("=" * 112)
print("\n  【关键对照】因果持续性基线 —— 若不优于它们，收益就只是「真值平滑」")
p1, p2, p3 = run_controls()
print(f"    P1 用上一帧真值         {p1.mean():>8.3f}")
print(f"    P2 用上一帧预测         {p2.mean():>8.3f}")
print(f"    P3 最近3帧真值均值       {p3.mean():>8.3f}")

print("\n  【最终扫描】放开格门限为固定下限惩罚（pen=0.05），λ 下扫；σ=3.0, min_cell=8")
best = (1e9, None)
for lam in (0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3):
    e, _, _ = run_m(lam=lam)
    flag = ""
    if e.mean() < best[0]:
        best = (e.mean(), lam); flag = "  ← 目前最好"
    print(f"    λ={lam:<5}{e.mean():>9.3f}{100 * (e.mean() / base.mean() - 1):>9.1f}%"
          f"   >3cm {100 * (e > 3).mean():>5.2f}%  max {e.max():>6.2f}{flag}")

print(f"\n  最优 λ={best[1]} 的细节：")
for tag, kw in (("基础", dict(lam=best[1])),
                ("+运动项 M1", dict(lam=best[1], motion=True)),
                ("pen=0.2", dict(lam=best[1], pen_floor=0.2)),
                ("σ=2.2", dict(lam=best[1], sigma=2.2)),
                ("min_cell=4", dict(lam=best[1], min_cell=4))):
    e, pa, ps = run_m(**kw)
    print(f"    {tag:<12}{e.mean():>9.3f}{100 * (e.mean() / base.mean() - 1):>9.1f}%"
          f"   >2cm {100 * (e > 2).mean():>5.2f}%  >3cm {100 * (e > 3).mean():>5.2f}%"
          f"  >5cm {100 * (e > 5).mean():>4.2f}%  max {e.max():>6.2f}")
    if tag == "基础":
        print("      按 activity：" + "  ".join(f"{k}={np.mean(v):.3f}" for k, v in sorted(pa.items())))
        print("      按被试：" + "  ".join(f"{k}={np.mean(v):.3f}" for k, v in sorted(ps.items())))
