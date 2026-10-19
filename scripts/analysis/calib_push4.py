"""继续下压 + 机制分解 + 冷启动修复。

已确立：会话内遗忘 λ 单调有效（λ=.95,σ=2.2 ⇒ 0.896 = −39.1%），说明误差有秒级时间自相关；
         运动项 M1/M2 不增益 ✗。
本脚本：
  (a) λ 继续下扫（0.95→0.5）× σ（2.2/3.0/4.0）找下限
  (b) 分解：极短记忆下，收益来自「仿射(全局偏置)」还是「网格场(空间结构)」
  (c) 只继承仿射的跨会话热启动（上轮整场继承全负 ✗，但冷启动确是 λ 救不了的短板）
  (d) 热启动后的冷启动收敛曲线
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/xieliangcai/code/gazeformer-exps")
CSV = ROOT / "out/rgbdgaze/analysis/eye224_test.csv"
G, MIN = 8, 8
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
order_by_subj = defaultdict(list)
for s in sess:
    order_by_subj[s.split("/")[-4]].append(s)        # dict 保序 = 清单顺序
SUBJ = sorted(order_by_subj)


def cidx(pt):
    return min(G - 1, max(0, int(pt[0] * G))) * G + min(G - 1, max(0, int(pt[1] * G)))


def emv(pred, lab, path):
    w, h = sc[path]
    return float(np.hypot((pred[0] - lab[0]) * w, (pred[1] - lab[1]) * h))


class Est:
    def __init__(self, lam=1.0, sigma=2.2, rad=2, use_field=True, use_affine=True):
        self.lam, self.sigma, self.rad = lam, sigma, rad
        self.use_field, self.use_affine = use_field, use_affine
        self.n = 0.0
        self.M = np.zeros((3, 3)); self.Rx = np.zeros(3); self.Ry = np.zeros(3)
        self.g = defaultdict(lambda: [0.0, 0.0, 0.0])

    def _u(self, P):
        return np.array([1.0, P[0], P[1]])

    def beta(self):
        if not self.use_affine or self.n < MIN:
            return None
        A = self.M + 0.01 * self.n * np.diag([0.0, 1.0, 1.0])
        try:
            return np.linalg.solve(A, self.Rx), np.linalg.solve(A, self.Ry)
        except np.linalg.LinAlgError:
            return None

    def correct(self, P):
        b = self.beta()
        bv = np.array([self._u(P) @ b[0], self._u(P) @ b[1]]) if b is not None else np.zeros(2)
        p1 = P - bv
        if not self.use_field:
            return p1, p1, bv
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
        return ((p1 - acc / cnt) if cnt >= MIN else p1), p1, bv

    def update(self, P, L, p1, bv):
        if self.lam != 1.0:
            self.n *= self.lam
            self.M *= self.lam
            self.Rx *= self.lam; self.Ry *= self.lam
            for kk in self.g:
                c = self.g[kk]
                c[0] *= self.lam; c[1] *= self.lam; c[2] *= self.lam
        u = self._u(P)
        dx, dy = P[0] - L[0], P[1] - L[1]
        self.n += 1.0
        self.M += np.outer(u, u)
        self.Rx += dx * u; self.Ry += dy * u
        c = self.g[cidx(p1)]
        c[0] += 1.0
        c[1] += float(dx) - float(bv[0]); c[2] += float(dy) - float(bv[1])

    def carry(self, other, rho):
        """只继承仿射（低维、稳健），按 ρ 折扣；场不继承。"""
        self.n = other.n * rho
        self.M = other.M * rho
        self.Rx = other.Rx * rho
        self.Ry = other.Ry * rho


def run(lam=1.0, sigma=2.2, rad=2, use_field=True, use_affine=True,
        carry_rho=None, by_time=False):
    errs, tpos = [], []
    for sj in (SUBJ if carry_rho is not None else [None]):
        prev_state = None
        for s in (order_by_subj[sj] if sj is not None else sorted(sess)):
            fd = Est(lam, sigma, rad, use_field, use_affine)
            if carry_rho is not None and prev_state is not None:
                fd.carry(prev_state, carry_rho)
            for n, r in enumerate(sess[s]):
                P = r["P"]
                corr, p1, bv = fd.correct(P)
                errs.append(emv(corr, r["L"], r["path"]))
                if by_time:
                    tpos.append(n)
                fd.update(P, r["L"], p1, bv)
            prev_state = fd
    e = np.array(errs)
    return (e, np.array(tpos)) if by_time else e


base = np.array([emv(r["P"], r["L"], r["path"]) for s in sess for r in sess[s]])
print("=" * 112)
print(f"基线 {base.mean():.3f}   上一轮最好 0.896（λ=.95, σ=2.2） 目标：更低")
print("=" * 112)
print("\n  (a) λ 下扫 × σ")
print(f"    {'λ':>7}" + "".join(f"{'σ=' + str(s):>17}" for s in (2.2, 3.0, 4.0)))
for lam in (0.95, 0.9, 0.85, 0.8, 0.7, 0.5, 0.2):
    line = f"    {lam:>7}"
    for sg in (2.2, 3.0, 4.0):
        e = run(lam=lam, sigma=sg)
        line += f"{e.mean():>10.3f}({100 * (e.mean() / base.mean() - 1):>+4.1f}%)"
    print(line)

print("\n  (b) 机制分解（取 λ=0.9/0.95, σ=3.0）")
for lam in (0.95, 0.9):
    for tag, kw in (("仅仿射", dict(use_field=False)),
                    ("仅场", dict(use_affine=False)),
                    ("仿射+场", {})):
        e = run(lam=lam, sigma=3.0, **kw)
        print(f"    λ={lam} {tag:<10}{e.mean():>9.3f}"
              f"{100 * (e.mean() / base.mean() - 1):>9.1f}%"
              f"   >3cm {100 * (e > 3).mean():>5.2f}%  max {e.max():>6.2f}")

print("\n  (c) 只继承仿射的跨会话热启动（ρ=折扣）")
for lam in (0.95, 0.9):
    for rho in (0.0, 0.3, 0.6, 1.0):
        e = run(lam=lam, sigma=3.0, carry_rho=rho)
        print(f"    λ={lam} ρ={rho:<4}{e.mean():>9.3f}"
              f"{100 * (e.mean() / base.mean() - 1):>9.1f}%"
              f"   >3cm {100 * (e > 3).mean():>5.2f}%  max {e.max():>6.2f}")

print("\n  (d) 冷启动收敛（λ=0.9, σ=3.0；热启动 ρ=1 对比）")
e_c, tp = run(lam=0.9, sigma=3.0, by_time=True)
e_h, _ = run(lam=0.9, sigma=3.0, carry_rho=1.0, by_time=True)
e_b, _ = run(lam=1.0, sigma=3.0, by_time=True)
print(f"    {'会话内帧位':>12}{'≈秒':>6}{'无热启动':>11}{'热启动ρ=1':>11}{'累积λ=1':>10}")
for hi in (30, 60, 150, 300, 600, 1200, 2400):
    m = tp < hi
    if m.sum() > 0:
        print(f"    {'<'+str(hi):>12}{hi / FPS:>6.0f}{e_c[m].mean():>11.3f}"
              f"{e_h[m].mean():>11.3f}{e_b[m].mean():>10.3f}")
