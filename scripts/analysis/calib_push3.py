"""收紧：λ（会话内遗忘）与 σ（核宽）联合扫，并测「冷启动收敛速度」。

本轮已确立：
  · 核加权 K：硬盒 1.093 → 1.073 ✓
  · 会话内遗忘 F：λ=0.995 → 0.994（−32.5%）✓✓ 且随记忆变短单调变好 ⇒ 误差存在秒级时间自相关
  · 运动项 M1 不增益（只压 max）✗；速度条件化场 M2 不增益 ✗
待查：λ 与 σ 的联合最优；冷启动多久收敛（若几秒 ⇒ 部署上无需标定阶段）
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
    def __init__(self, lam=1.0, sigma=0.9, rad=2, motion=False):
        self.lam, self.sigma, self.rad, self.motion = lam, sigma, rad, motion
        k = 4 if motion else 3
        self.k = k
        self.n = 0.0
        self.M = np.zeros((k, k))
        self.Rx = np.zeros(k); self.Ry = np.zeros(k)
        self.g = defaultdict(lambda: [0.0, 0.0, 0.0])

    def _u(self, P, v):
        return (np.array([1.0, P[0], P[1], v / 10.0]) if self.motion
                else np.array([1.0, P[0], P[1]]))

    def beta(self):
        if self.n < MIN:
            return None
        A = self.M + 0.01 * self.n * np.diag([0.0] + [1.0] * (self.k - 1))
        try:
            return np.linalg.solve(A, self.Rx), np.linalg.solve(A, self.Ry)
        except np.linalg.LinAlgError:
            return None

    def correct(self, P, v):
        b = self.beta()
        bv = np.array([self._u(P, v) @ b[0], self._u(P, v) @ b[1]]) if b is not None else np.zeros(2)
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
        corr = (p1 - acc / cnt) if cnt >= MIN else p1
        return corr, p1, bv

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


def run(lam=1.0, sigma=0.9, rad=2, motion=False, by_time=False):
    errs, per_act = [], defaultdict(list)
    tpos = []                       # 帧在会话内的位置（用于收敛曲线）
    for s in SESS:
        fd = Field(lam, sigma, rad, motion)
        prev = None
        for n, r in enumerate(sess[s]):
            P = r["P"]
            w, h = sc[r["path"]]
            v = 0.0 if prev is None else float(
                np.hypot((P[0] - prev[0]) * w, (P[1] - prev[1]) * h)) * FPS
            corr, p1, bv = fd.correct(P, v)
            e = emv(corr, r["L"], r["path"])
            errs.append(e); per_act[r["act"]].append(e)
            if by_time:
                tpos.append(n)
            fd.update(P, r["L"], v, p1, bv)
            prev = P
    e = np.array(errs)
    if by_time:
        return e, np.array(tpos), per_act
    return e, None, per_act


base = np.array([emv(r["P"], r["L"], r["path"]) for s in sess for r in sess[s]])
print("=" * 112)
print(f"基线 {base.mean():.3f}   当前最好 0.994（λ=0.995, σ=0.9, rad=2）")
print("=" * 112)
print("\n  λ × σ 联合扫（rad=2；括号=vs 基线）")
print(f"    {'λ':>9}" + "".join(f"{'σ=' + str(s):>17}" for s in (0.9, 1.2, 1.6, 2.2)))
for lam in (1.0, 0.999, 0.997, 0.995, 0.99, 0.98, 0.95):
    line = f"    {lam:>9}"
    for sg in (0.9, 1.2, 1.6, 2.2):
        e, _, _ = run(lam=lam, sigma=sg)
        line += f"{e.mean():>10.3f}({100 * (e.mean() / base.mean() - 1):>+4.1f}%)"
    print(line)

print("\n  最佳组合附近再加运动项（观察 max 与 >5cm）")
for tag, kw in (("λ=.995 σ=.9", dict(lam=0.995, sigma=0.9)),
                ("λ=.995 σ=.9 +M1", dict(lam=0.995, sigma=0.9, motion=True)),
                ("λ=.99 σ=1.2", dict(lam=0.99, sigma=1.2)),
                ("λ=.99 σ=1.2 +M1", dict(lam=0.99, sigma=1.2, motion=True)),
                ("λ=.98 σ=1.6", dict(lam=0.98, sigma=1.6)),
                ("λ=.95 σ=2.2", dict(lam=0.95, sigma=2.2))):
    e, _, pa = run(**kw)
    print(f"    {tag:<18}{e.mean():>9.3f}   >2cm {100 * (e > 2).mean():>5.2f}%"
          f"  >3cm {100 * (e > 3).mean():>5.2f}%  >5cm {100 * (e > 5).mean():>4.2f}%"
          f"  max {e.max():>6.2f}   " +
          " ".join(f"{k}:{np.mean(v):.3f}" for k, v in sorted(pa.items())))

print("\n  冷启动收敛（每会话开头 N 帧内的 EM；若几秒内达标 ⇒ 部署无需标定阶段）")
e, tp, _ = run(lam=0.995, sigma=0.9, by_time=True)
eb, tpb, _ = run(lam=1.0, sigma=0.9, by_time=True)
print(f"    {'会话内帧位':>12}{'≈秒':>6}{'λ=0.995':>10}{'λ=1(累积)':>11}")
for hi in (30, 60, 150, 300, 600, 1200, 2400):
    m = tp < hi
    if m.sum() > 0:
        print(f"    {'<'+str(hi):>12}{hi / FPS:>6.0f}{e[m].mean():>10.3f}{eb[m].mean():>11.3f}")
