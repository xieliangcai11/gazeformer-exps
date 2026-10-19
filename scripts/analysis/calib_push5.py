"""修门限 + 双时间尺度：仿射(快) 与 空间场(慢) 各用独立遗忘因子。

修正的坑：MIN=8 门限 vs 遗忘冲突 ⇒ λ<0.85 时有效样本数饱和在 1/(1-λ)<8 ⇒ 仿射被永久禁用
          ⇒ 结果退化为「完全不修正」(1.472, +0.0%) ✗ —— 上一轮 λ 扫描的下限是门限造成的，不是数据。
双时间尺度假设：全局偏置/尺度（仿射）漂移快 ⇒ 该用短记忆；
                空间结构（格）更稳 ⇒ 该用长记忆。⇒ λ_aff < λ_cell
另含：只继承仿射的跨会话热启动（上轮稳定小胜 ✓，且把冷启动前 1 秒 1.447→1.090 ✓）
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/xieliangcai/code/gazeformer-exps")
CSV = ROOT / "out/rgbdgaze/analysis/eye224_test.csv"
G, MIN = 8, 8
MIN_AFF = 3          # 修正：仿射只需极少有效样本，靠岭正则兜底
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
    order_by_subj[s.split("/")[-4]].append(s)
SUBJ = sorted(order_by_subj)


def cidx(pt):
    return min(G - 1, max(0, int(pt[0] * G))) * G + min(G - 1, max(0, int(pt[1] * G)))


def emv(pred, lab, path):
    w, h = sc[path]
    return float(np.hypot((pred[0] - lab[0]) * w, (pred[1] - lab[1]) * h))


class Est:
    def __init__(self, lam_aff=1.0, lam_cell=1.0, sigma=3.0, rad=2,
                 min_cell=MIN, use_field=True, use_affine=True):
        self.la, self.lc, self.sigma, self.rad = lam_aff, lam_cell, sigma, rad
        self.min_cell, self.use_field, self.use_affine = min_cell, use_field, use_affine
        self.n = 0.0
        self.M = np.zeros((3, 3)); self.Rx = np.zeros(3); self.Ry = np.zeros(3)
        self.g = defaultdict(lambda: [0.0, 0.0, 0.0])

    def beta(self):
        if not self.use_affine or self.n < MIN_AFF:
            return None
        A = self.M + 0.01 * self.n * np.diag([0.0, 1.0, 1.0])
        try:
            return np.linalg.solve(A, self.Rx), np.linalg.solve(A, self.Ry)
        except np.linalg.LinAlgError:
            return None

    def correct(self, P):
        b = self.beta()
        u = np.array([1.0, P[0], P[1]])
        bv = np.array([u @ b[0], u @ b[1]]) if b is not None else np.zeros(2)
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
        return ((p1 - acc / cnt) if cnt >= self.min_cell else p1), p1, bv

    def update(self, P, L, p1, bv):
        if self.la != 1.0:
            self.n *= self.la
            self.M *= self.la
            self.Rx *= self.la; self.Ry *= self.la
        if self.lc != 1.0:
            for kk in self.g:
                c = self.g[kk]
                c[0] *= self.lc; c[1] *= self.lc; c[2] *= self.lc
        u = np.array([1.0, P[0], P[1]])
        dx, dy = P[0] - L[0], P[1] - L[1]
        self.n += 1.0
        self.M += np.outer(u, u)
        self.Rx += dx * u; self.Ry += dy * u
        c = self.g[cidx(p1)]
        c[0] += 1.0
        c[1] += float(dx) - float(bv[0]); c[2] += float(dy) - float(bv[1])

    def carry_affine(self, o, rho):
        self.n, self.M, self.Rx, self.Ry = o.n * rho, o.M * rho, o.Rx * rho, o.Ry * rho


def run(lam_aff=1.0, lam_cell=1.0, sigma=3.0, min_cell=MIN, use_field=True,
        use_affine=True, carry_rho=None, by_time=False, only_session1=False):
    errs, tpos = [], []
    for sj in (SUBJ if carry_rho is not None else [None]):
        prev = None
        for k, s in enumerate(order_by_subj[sj] if sj is not None else sorted(sess)):
            if only_session1 and k > 0:
                continue
            fd = Est(lam_aff, lam_cell, sigma, 2, min_cell, use_field, use_affine)
            if carry_rho is not None and prev is not None:
                fd.carry_affine(prev, carry_rho)
            for n, r in enumerate(sess[s]):
                corr, p1, bv = fd.correct(r["P"])
                errs.append(emv(corr, r["L"], r["path"]))
                if by_time:
                    tpos.append(n)
                fd.update(r["P"], r["L"], p1, bv)
            prev = fd
    e = np.array(errs)
    return (e, np.array(tpos)) if by_time else e


base = np.array([emv(r["P"], r["L"], r["path"]) for s in sess for r in sess[s]])
print("=" * 116)
print(f"基线 {base.mean():.3f}   上一轮最好 0.858（λ=.9 + 仿射热启动）")
print("=" * 116)

print("\n  (a) 修门限后单 λ 重扫（σ=3.0, min_cell=8；无热启动）")
for lam in (0.99, 0.97, 0.95, 0.93, 0.9, 0.85, 0.8, 0.7, 0.5):
    e = run(lam_aff=lam, lam_cell=lam)
    print(f"    λ={lam:<6}{e.mean():>9.3f}{100 * (e.mean() / base.mean() - 1):>9.1f}%"
          f"   >3cm {100 * (e > 3).mean():>5.2f}%  max {e.max():>6.2f}")

print("\n  (b) 双时间尺度：λ_aff(快) × λ_cell(慢)")
print(f"    {'λ_aff':>7}" + "".join(f"{'λ_cell=' + str(c):>17}" for c in (1.0, 0.999, 0.995, 0.99)))
for la in (0.95, 0.9, 0.85, 0.8):
    line = f"    {la:>7}"
    for lc in (1.0, 0.999, 0.995, 0.99):
        e = run(lam_aff=la, lam_cell=lc)
        line += f"{e.mean():>10.3f}({100 * (e.mean() / base.mean() - 1):>+4.1f}%)"
    print(line)

print("\n  (c) 格门限 min_cell 与核宽 σ（取 λ_aff=0.9, λ_cell=0.995）")
for mc in (8, 3, 1):
    for sg in (2.2, 3.0, 5.0):
        e = run(lam_aff=0.9, lam_cell=0.995, sigma=sg, min_cell=mc)
        print(f"    min_cell={mc} σ={sg:<4}{e.mean():>9.3f}"
              f"{100 * (e.mean() / base.mean() - 1):>9.1f}%"
              f"   >3cm {100 * (e > 3).mean():>5.2f}%  max {e.max():>6.2f}")

print("\n  (d) 叠加「只继承仿射」热启动")
for rho in (0.0, 0.6, 1.0):
    e = run(lam_aff=0.9, lam_cell=0.995, sigma=3.0, min_cell=3, carry_rho=rho)
    e1 = run(lam_aff=0.9, lam_cell=0.995, sigma=3.0, min_cell=3, carry_rho=rho,
             only_session1=True)
    print(f"    ρ={rho:<4}全会话 {e.mean():>9.3f}{100 * (e.mean() / base.mean() - 1):>9.1f}%"
          f"   >3cm {100 * (e > 3).mean():>5.2f}%  max {e.max():>6.2f}"
          f"   | 仅首会话(新用户) {e1.mean():>7.3f}")
