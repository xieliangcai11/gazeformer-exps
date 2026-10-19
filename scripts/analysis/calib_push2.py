"""冲击 <1.085：按诊断结论打两块。

诊断（calib_diag2.py）：
  · 全段一次性建场（含未来）反而更差（1.107 > 1.093）⇒ 会话内非平稳 ⇒ 应**带遗忘**（记忆短）
  · 误差质量集中在快速运动：>20cm/s 占 24.5% 帧却占 31.5% 质量；walking 增益最低（−17.4%）
    ⇒ 缺**运动感知**项
本轮候选（全部因果，速度用「预测轨迹」的帧间位移，部署可得）：
  F: 会话内遗忘因子 λ（每帧把统计量乘 λ）
  M1: 仿射设计里加运动项 [1, x, y, v]（v=预测速度 cm/s /10）
  M2: 速度条件化场：慢/快两张网格，按 κ 向池化场收缩
  K: 高斯核加权（上轮胜出 σ≈0.9~1.2, rad=2）
自检：全关（λ=1, M1/M2 off, K硬盒）⇒ 必须回到 1.093
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
SESS = sorted(sess)


def cidx(pt):
    return min(G - 1, max(0, int(pt[0] * G))) * G + min(G - 1, max(0, int(pt[1] * G)))


def emv(pred, lab, path):
    w, h = sc[path]
    return float(np.hypot((pred[0] - lab[0]) * w, (pred[1] - lab[1]) * h))


class Field:
    """仿射(1/4 参数) → 核加权网格场；支持遗忘 λ 与速度条件化。"""

    def __init__(self, use_motion=False, lam=1.0, sigma=None, rad=1, vthr=10.0, kappa=None):
        self.use_motion, self.lam, self.sigma, self.rad = use_motion, lam, sigma, rad
        self.vthr, self.kappa = vthr, kappa
        k = 4 if use_motion else 3
        self.k = k
        self.n = 0.0
        self.M = np.zeros((k, k))
        self.Rx = np.zeros(k); self.Ry = np.zeros(k)
        self.g1 = defaultdict(lambda: [0.0, 0.0, 0.0])          # 池化/慢
        self.g2 = defaultdict(lambda: [0.0, 0.0, 0.0]) if kappa else None   # 快

    def _u(self, P, v):
        return np.array([1.0, P[0], P[1], v / 10.0]) if self.use_motion else np.array([1.0, P[0], P[1]])

    def beta(self):
        if self.n < MIN:
            return None
        A = self.M + 0.01 * self.n * np.diag([0.0] + [1.0] * (self.k - 1))
        try:
            return np.linalg.solve(A, self.Rx), np.linalg.solve(A, self.Ry)
        except np.linalg.LinAlgError:
            return None

    def _cellval(self, grid, key):
        c = cidx(key); i0, j0 = c // G, c % G
        acc = np.zeros(2); cnt = 0.0
        for di in range(-self.rad, self.rad + 1):
            for dj in range(-self.rad, self.rad + 1):
                ii, jj = i0 + di, j0 + dj
                if not (0 <= ii < G and 0 <= jj < G):
                    continue
                cc = grid[ii * G + jj]
                if cc[0] <= 0:
                    continue
                w = 1.0 if self.sigma is None else float(
                    np.exp(-(di * di + dj * dj) / (2 * self.sigma ** 2)))
                acc += w * np.array([cc[1], cc[2]])
                cnt += w * cc[0]
        return (acc / cnt if cnt > 0 else None), cnt

    def correct(self, P, v):
        b = self.beta()
        bv = (np.array([self._u(P, v) @ b[0], self._u(P, v) @ b[1]]) if b is not None
              else np.zeros(2))
        p1 = P - bv
        pool, cntp = self._cellval(self.g1, p1)
        if self.g2 is not None and pool is not None:
            fast = v > self.vthr
            reg, cntr = self._cellval(self.g2 if fast else self.g1, p1)
            if reg is not None:
                k = self.kappa
                val = (cntr * reg + k * pool) / (cntr + k)
            else:
                val = pool
        else:
            val = pool
        return (p1 - val) if val is not None else p1, p1, bv

    def update(self, P, L, v, p1, bv):
        if self.lam != 1.0:                      # 会话内遗忘：统计量整体衰减
            self.n *= self.lam
            self.M *= self.lam
            self.Rx *= self.lam; self.Ry *= self.lam
            for g in ((self.g1,) + ((self.g2,) if self.g2 is not None else ())):
                for kk in g:
                    c = g[kk]
                    c[0] *= self.lam; c[1] *= self.lam; c[2] *= self.lam
        u = self._u(P, v)
        dx, dy = P[0] - L[0], P[1] - L[1]
        self.n += 1.0
        self.M += np.outer(u, u)
        self.Rx += dx * u; self.Ry += dy * u
        c = self.g1[cidx(p1)]
        c[0] += 1.0
        c[1] += float(dx) - float(bv[0]); c[2] += float(dy) - float(bv[1])
        if self.g2 is not None and v > self.vthr:
            c = self.g2[cidx(p1)]
            c[0] += 1.0
            c[1] += float(dx) - float(bv[0]); c[2] += float(dy) - float(bv[1])


def run(**kw):
    errs, per_act, per_sub = [], defaultdict(list), defaultdict(list)
    for s in SESS:
        fd = Field(**kw)
        prev = None
        for r in sess[s]:
            P = r["P"]
            w, h = sc[r["path"]]
            v = 0.0 if prev is None else float(
                np.hypot((P[0] - prev[0]) * w, (P[1] - prev[1]) * h)) * FPS
            corr, p1, bv = fd.correct(P, v)
            e = emv(corr, r["L"], r["path"])
            errs.append(e); per_act[r["act"]].append(e); per_sub[r["subj"]].append(e)
            fd.update(P, r["L"], v, p1, bv)
            prev = P
    return np.array(errs), per_act, per_sub


base = np.array([emv(r["P"], r["L"], r["path"]) for s in sess for r in sess[s]])
print("=" * 112)
print(f"基线 {base.mean():.3f}   目标：低于 1.085（上轮核加权最好）与 1.093（权威）")
print("=" * 112)
print(f"  {'方案':<40}{'EM均值':>9}{'vs基线':>9}{'>2cm':>8}{'>3cm':>8}{'>5cm':>8}{'max':>8}")


def show(tag, e):
    print(f"  {tag:<40}{e.mean():>9.3f}{100 * (e.mean() / base.mean() - 1):>8.1f}%"
          f"{100 * (e > 2).mean():>7.2f}%{100 * (e > 3).mean():>7.2f}%"
          f"{100 * (e > 5).mean():>7.2f}%{e.max():>8.2f}")


show("A 不修正", base)
e0, _, _ = run()
show("自检：硬盒 + 无运动 + 无遗忘（应=1.093）", e0)
eK, _, _ = run(sigma=0.9, rad=2)
show("K 核加权 σ=0.9 rad=2（上轮最好 1.085）", eK)
print()
print("  F 会话内遗忘 λ")
for lam in (0.9999, 0.9995, 0.999, 0.995):
    e, _, _ = run(sigma=0.9, rad=2, lam=lam)
    show(f"  λ={lam}", e)
print()
print("  M1 仿射加运动项 [1,x,y,v]")
eM1, _, _ = run(use_motion=True, sigma=0.9, rad=2)
show("  M1 + 核加权", eM1)
eM1f, _, _ = run(use_motion=True, sigma=0.9, rad=2, lam=0.999)
show("  M1 + 核加权 + λ=0.999", eM1f)
print()
print("  M2 速度条件化场（快/慢两张，向池化场收缩 κ）")
for kap in (5.0, 15.0, 40.0):
    e, _, _ = run(sigma=0.9, rad=2, kappa=kap)
    show(f"  M2 κ={kap}", e)
print()
print("  组合")
best = [
    ("M1+M2 κ=15", dict(use_motion=True, sigma=0.9, rad=2, kappa=15.0)),
    ("M1+M2 κ=15+λ.999", dict(use_motion=True, sigma=0.9, rad=2, kappa=15.0, lam=0.999)),
    ("M1+M2 κ=40+λ.9995", dict(use_motion=True, sigma=0.9, rad=2, kappa=40.0, lam=0.9995)),
]
for tag, kw in best:
    e, pa, ps = run(**kw)
    show(tag, e)
    if tag == "M1+M2 κ=15+λ.999":
        print(f"      按 activity：" + "  ".join(
            f"{k}={np.mean(v):.3f}({100 * (np.mean(v) / np.mean([emv(r['P'], r['L'], r['path']) for s in sess if sess[s][0]['act'] == k for r in sess[s]]) - 1):+.0f}%)"
            for k, v in sorted(pa.items())))
        print(f"      按被试：" + "  ".join(f"{k}={np.mean(v):.3f}" for k, v in sorted(ps.items())))
