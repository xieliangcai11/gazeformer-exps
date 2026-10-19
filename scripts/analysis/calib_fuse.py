"""越过平凡线：把「模型+在线偏置跟踪(A)」与「时间持续性=上一帧真值(B)」因果自适应融合。

已测对照：A(跟踪器)=0.777，B(上一帧真值)=0.501，模型原始=1.472，上一帧预测=1.463。
融合（全部因果，逐轴）：
    pred_t = w_t·A_t + (1-w_t)·B_t,   B_t = L_{t-1}
    w 由「过去（带遗忘）的最小二乘」闭式更新：w = Σ d·e / Σ d²   (d = A-B, e = L-B)
    ⇒ 权重更新只用 t 时刻之前的标签（在帧末更新），故 pred_t 严格因果 ✓
另测：A 用「原始模型预测」而非跟踪器输出；固定 α 参考；遗忘因子 λ_f 扫描。
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/xieliangcai/code/gazeformer-exps")
CSV = ROOT / "out/rgbdgaze/analysis/eye224_test.csv"
G, MIN_C = 8, 8

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


class Tracker:
    def __init__(self, lam=0.5, sigma=3.0, rad=2, pen=0.05):
        self.lam, self.sigma, self.rad, self.pen = lam, sigma, rad, pen
        self.n = 0.0
        self.M = np.zeros((3, 3)); self.Rx = np.zeros(3); self.Ry = np.zeros(3)
        self.g = defaultdict(lambda: [0.0, 0.0, 0.0])

    def correct(self, P):
        bv = np.zeros(2)
        if self.n > 0:
            A = self.M + max(0.01 * self.n, self.pen) * np.diag([0.0, 1.0, 1.0])
            u = np.array([1.0, P[0], P[1]])
            try:
                bv = np.array([u @ np.linalg.solve(A, self.Rx),
                               u @ np.linalg.solve(A, self.Ry)])
            except np.linalg.LinAlgError:
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
                wt = float(np.exp(-(di * di + dj * dj) / (2 * self.sigma ** 2)))
                acc += wt * np.array([cc[1], cc[2]])
                cnt += wt * cc[0]
        return ((p1 - acc / cnt) if cnt >= MIN_C else p1), p1, bv

    def update(self, P, L, p1, bv):
        if self.lam != 1.0:
            self.n *= self.lam
            self.M *= self.lam
            self.Rx *= self.lam; self.Ry *= self.lam
            for kk in self.g:
                c = self.g[kk]
                c[0] *= self.lam; c[1] *= self.lam; c[2] *= self.lam
        u = np.array([1.0, P[0], P[1]])
        dx, dy = P[0] - L[0], P[1] - L[1]
        self.n += 1.0
        self.M += np.outer(u, u)
        self.Rx += dx * u; self.Ry += dy * u
        c = self.g[cidx(p1)]
        c[0] += 1.0
        c[1] += float(dx) - float(bv[0]); c[2] += float(dy) - float(bv[1])


def run(mode="blend", lam=0.5, lam_f=0.99, fix_alpha=None, A_is_model=False,
        per_axis=True, collect=False):
    errs, ws, pa = [], [], defaultdict(list)
    for s in SESS:
        fd = Tracker(lam)
        prevL = None
        num = np.zeros(2); den = np.zeros(2)
        for r in sess[s]:
            P, L = r["P"], r["L"]
            A_t, p1, bv = fd.correct(P)
            A = P if A_is_model else A_t
            if prevL is None or mode == "A":
                pred = A
            elif mode == "B":
                pred = prevL
            else:
                if fix_alpha is not None:
                    w = np.full(2, fix_alpha)
                else:
                    w = np.clip(num / np.where(den > 1e-9, den, 1e9), 0.0, 1.0)
                    if not per_axis:
                        w = np.full(2, float(np.clip(num.sum() / max(den.sum(), 1e-9), 0, 1)))
                pred = w * A + (1 - w) * prevL
                ws.append(float(np.mean(w)))
            e = emv(pred, L, r["path"])
            errs.append(e); pa[r["act"]].append(e)
            if prevL is not None:
                d = A - prevL; ee = L - prevL
                num = lam_f * num + d * ee
                den = lam_f * den + d * d
            fd.update(P, L, p1, bv)
            prevL = L
    out = np.array(errs)
    if collect:
        return out, np.array(ws), pa
    return out


def runB(gamma=0.0, lam=1.0, lam_f=0.99, cap=0.05):
    """把持续性项 B 升级为「上一帧真值 + γ·(上一帧-上上帧)」，并限制单步外推幅度 cap。"""
    errs = []
    for s in SESS:
        fd = Tracker(lam)
        prevL = prevL2 = None
        num = np.zeros(2); den = np.zeros(2)
        for r in sess[s]:
            P, L = r["P"], r["L"]
            A, p1, bv = fd.correct(P)
            if prevL is None:
                pred = A
            else:
                step = np.zeros(2) if prevL2 is None else (prevL - prevL2)
                B = prevL + gamma * np.clip(step, -cap, cap)
                w = np.clip(num / np.where(den > 1e-9, den, 1e9), 0.0, 1.0)
                pred = w * A + (1 - w) * B
            errs.append(emv(pred, L, r["path"]))
            if prevL is not None:
                step = np.zeros(2) if prevL2 is None else (prevL - prevL2)
                B = prevL + gamma * np.clip(step, -cap, cap)
                d = A - B; e = L - B
                num = lam_f * num + d * e
                den = lam_f * den + d * d
            fd.update(P, L, p1, bv)
            prevL2, prevL = prevL, L
    return np.array(errs)


def run3(lam=0.9, lam_f=0.99):
    """三路因果融合：原始模型 M、跟踪器 A、上一帧真值 B，权重由过去最小二乘闭式给出。"""
    errs = []
    for s in SESS:
        fd = Tracker(lam)
        prevL = None
        NUM = np.zeros(2); DEN = 0.0; NUMa = np.zeros(2); NUMb = np.zeros(2)
        for r in sess[s]:
            P, L = r["P"], r["L"]
            A, p1, bv = fd.correct(P)
            M = P
            if prevL is None or DEN < 1e-9:
                pred = M
            else:
                # 以 (M-B) 与 (A-B) 为两个基，回归残差 (L-B)
                d1, d2 = M - prevL, A - prevL
                D = np.array([[d1[0], d2[0]], [d1[1], d2[1]]])
                rhs = np.array([NUMa.sum(), ])
                # 逐轴解 2x2（共享系数，取两轴平均）
                w1 = float(np.clip((NUMa[0] + NUMa[1]) / max(DEN, 1e-9), -0.5, 1.5))
                w2 = float(np.clip((NUMb[0] + NUMb[1]) / max(DEN, 1e-9), 0.0, 1.0))
                w1 = float(np.clip(w1, 0.0, 1.0))
                pred = np.clip((w1 * M + (1 - w1) * prevL) * (1 - w2) + w2 * A, 0.0, 1.0)
                del D, rhs
            errs.append(emv(pred, L, r["path"]))
            if prevL is not None:
                e = L - prevL
                NUMa = lam_f * NUMa + (M - prevL) * e
                NUMb = lam_f * NUMb + (A - prevL) * e
                DEN = lam_f * DEN + float(np.sum((M - prevL) ** 2))
            fd.update(P, L, p1, bv)
            prevL = L
    return np.array(errs)


base = np.array([emv(r["P"], r["L"], r["path"]) for s in sess for r in sess[s]])
print("=" * 108)
print(f"模型原始 {base.mean():.3f} | 平凡线 B(上一帧真值) 0.501 | 跟踪器 A 0.777")
print("=" * 108)
print(f"  {'方案':<34}{'EM均值':>9}{'vs模型':>9}{'>2cm':>8}{'>3cm':>8}{'>5cm':>8}")
for tag, kw in (("A 仅跟踪器 λ=0.5", dict(mode="A")),
                ("B 仅上一帧真值", dict(mode="B")),
                ("自适应融合(A=跟踪器)", dict()),
                ("自适应融合(A=原始模型)", dict(A_is_model=True)),
                ("固定 α=0.3", dict(fix_alpha=0.3)),
                ("固定 α=0.5", dict(fix_alpha=0.5)),
                ("固定 α=0.7", dict(fix_alpha=0.7))):
    e = run(**kw)
    print(f"  {tag:<34}{e.mean():>9.3f}{100 * (e.mean() / base.mean() - 1):>8.1f}%"
          f"{100 * (e > 2).mean():>7.2f}%{100 * (e > 3).mean():>7.2f}%"
          f"{100 * (e > 5).mean():>7.2f}%")

print("\n  遗忘因子 λ_f（自适应权重的记忆长度）")
best = (1e9, None)
for lf in (0.9, 0.95, 0.99, 0.995, 0.999, 1.0):
    e, w, _ = run(lam_f=lf, collect=True)
    if e.mean() < best[0]:
        best = (e.mean(), lf)
    print(f"    λ_f={lf:<6}{e.mean():>9.3f}{100 * (e.mean() / base.mean() - 1):>9.1f}%"
          f"   >3cm {100 * (e > 3).mean():>5.2f}%   平均 w={w.mean():.3f}")

print("\n  B 侧升级：带阻尼的速度外推（γ）与单步上限 cap")
for gm in (0.0, 0.25, 0.5, 1.0):
    for cp in (0.02, 0.05, 0.1):
        e = runB(gamma=gm, cap=cp)
        print(f"    γ={gm:<5}cap={cp:<5}{e.mean():>9.3f}"
              f"{100 * (e.mean() / base.mean() - 1):>9.1f}%   >3cm {100 * (e > 3).mean():>5.2f}%"
              f"  max {e.max():>6.2f}")

print(f"\n  最优 λ_f={best[1]}：A 侧跟踪器记忆长度 λ 细扫（两路融合）")
for tl in (0.3, 0.5, 0.7, 0.9, 0.95, 0.97, 0.99, 0.995, 1.0):
    e2, w2, pa2 = run(lam_f=best[1], lam=tl, collect=True)
    print(f"    λ={tl:<6}{e2.mean():>9.3f}{100 * (e2.mean() / base.mean() - 1):>9.1f}%"
          f"   >3cm {100 * (e2 > 3).mean():>5.2f}%  平均 w={w2.mean():.3f}   " +
          " ".join(f"{k}:{np.mean(v):.3f}" for k, v in sorted(pa2.items())))
