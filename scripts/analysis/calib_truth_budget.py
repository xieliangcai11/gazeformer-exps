"""0.286 到底要多少真值？—— 真值供给率(r)扫描。

估计器（产生 0.286 的那个）：
    pred_t = w_t·A_t + (1-w_t)·B_t
    B_t = L_last + clip(step_last · τ, ±cap),  τ = t - t_last   （越过 r 帧做一步外推）
    A_t = 校准后的模型输出（场+仿射），w_t 由过去（带遗忘）最小二乘给出
r = 真值更新周期（每 r 帧用一次真值）：r=1 为原设定；r 越大表示"隐式标定"越稀疏。
对照线：模型原始 1.472；场单独 1.091；只用真值流外推(逐帧) 0.362；上一帧真值 0.501。
另测："只用前 K 帧真值、之后全程无真值"（一次性标定 + 模型单独跑）。
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/xieliangcai/code/gazeformer-exps")
CSV = ROOT / "out/rgbdgaze/analysis/eye224_test.csv"
G, MIN_C = 8, 8
LAM_F, CAP, SIGMA = 0.99, 0.05, 3.0
TAU_MAX = 12          # 标签陈旧度门控：超过这么多帧没用过真值 ⇒ 放弃 B 项，退回纯模型

sc = {}
with open(ROOT / "data/RGBDGaze/index/index.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        sc[r["rgb_path"]] = (float(r["screen_w"]) * float(r["cm_px_x"]),
                             float(r["screen_h"]) * float(r["cm_px_y"]))
sess = defaultdict(list)
for r in csv.DictReader(open(CSV, encoding="utf-8")):
    sess[r["path"].rsplit("/", 1)[0]].append(
        dict(fi=int(r["frame_index"]), path=r["path"],
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
    def __init__(self):
        self.n = 0.0
        self.M = np.zeros((3, 3)); self.Rx = np.zeros(3); self.Ry = np.zeros(3)
        self.g = defaultdict(lambda: [0.0, 0.0, 0.0])

    def correct(self, P):
        bv = np.zeros(2)
        if self.n > 0:
            A = self.M + max(0.01 * self.n, 0.05) * np.diag([0.0, 1.0, 1.0])
            u = np.array([1.0, P[0], P[1]])
            try:
                bv = np.array([u @ np.linalg.solve(A, self.Rx),
                               u @ np.linalg.solve(A, self.Ry)])
            except np.linalg.LinAlgError:
                bv = np.zeros(2)
        p1 = P - bv
        c = cidx(p1); i0, j0 = c // G, c % G
        acc = np.zeros(2); cnt = 0.0
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                ii, jj = i0 + di, j0 + dj
                if 0 <= ii < G and 0 <= jj < G:
                    cc = self.g[ii * G + jj]
                    if cc[0] <= 0:
                        continue
                    wt = float(np.exp(-(di * di + dj * dj) / (2 * SIGMA ** 2)))
                    acc += wt * np.array([cc[1], cc[2]])
                    cnt += wt * cc[0]
        return ((p1 - acc / cnt) if cnt >= MIN_C else p1), p1, bv

    def update(self, P, L, p1, bv):
        u = np.array([1.0, P[0], P[1]])
        dx, dy = P[0] - L[0], P[1] - L[1]
        self.n += 1.0
        self.M += np.outer(u, u)
        self.Rx += dx * u; self.Ry += dy * u
        c = self.g[cidx(p1)]
        c[0] += 1.0
        c[1] += float(dx) - float(bv[0]); c[2] += float(dy) - float(bv[1])


def run(period=1, warm_frac=None, a_only_model=False):
    """period: 每 period 帧使用一次真值；warm_frac: 只在前 warm_frac 比例用真值（None=全程）"""
    errs, ws, nlab = [], [], 0
    post = []
    for s in SESS:
        v = sess[s]
        N = len(v)
        cut = N if warm_frac is None else int(N * warm_frac)
        fd = Tracker()
        t_last = None
        L_last = None
        step = np.zeros(2)
        num = np.zeros(2); den = np.zeros(2)
        for t, r in enumerate(v):
            P, L = r["P"], r["L"]
            A_t, p1, bv = fd.correct(P)
            A = P if a_only_model else A_t
            if L_last is None or (t - t_last) > TAU_MAX:
                pred = A                      # 真值太久没用过 ⇒ 只用模型（避免过期外推）
            else:
                tau = t - t_last
                B = L_last + np.clip(step * tau, -CAP, CAP)
                w = np.clip(num / np.where(den > 1e-9, den, 1e9), 0.0, 1.0)
                pred = w * A + (1 - w) * B
                ws.append(float(np.mean(w)))
            e_now = emv(pred, L, r["path"])
            errs.append(e_now)
            if warm_frac is not None and t >= cut:
                post.append(e_now)
            # ---- 真值更新（按周期 / 预热期）----
            if t < cut and t % period == 0:
                nlab += 1
                if L_last is not None:
                    tau = t - t_last
                    B_prev = L_last + np.clip(step * tau, -CAP, CAP)
                    d = A - B_prev
                    e = L - B_prev
                    num = LAM_F * num + d * e
                    den = LAM_F * den + d * d
                    step = (L - L_last) / max(tau, 1)
                L_last = L
                t_last = t
                fd.update(P, L, p1, bv)
    return np.array(errs), (np.mean(ws) if ws else 0.0), nlab, np.array(post)


base = np.array([emv(r["P"], r["L"], r["path"]) for s in SESS for r in sess[s]])
tot = len(base)
print("=" * 104)
print(f"模型原始 {base.mean():.3f} | 场单独 1.091 | 逐帧真值外推 0.362 | 上一帧真值 0.501")
print("=" * 104)
print(f"  {'真值更新周期':>14}{'真值帧占比':>11}{'融合EM':>9}{'平均w':>8}{'>3cm':>8}{'相对模型':>10}")
for per in (1, 2, 3, 5, 10, 20, 30, 60, 120):
    e, w, nl, _ = run(period=per)
    print(f"  {'每 '+str(per)+' 帧':>14}{100 * nl / tot:>10.1f}%{e.mean():>9.3f}"
          f"{w:>8.3f}{100 * (e > 3).mean():>7.2f}%"
          f"{100 * (e.mean() / base.mean() - 1):>9.1f}%")

print("\n  只用前 X% 帧的真值，之后全程无真值（一次性标定 + 模型单独跑）")
print(f"  {'预热比例':>10}{'真值帧占比':>11}{'预热后EM':>10}{'平均w':>8}{'预热后>3cm':>11}")
for wf in (0.02, 0.05, 0.1, 0.2, 0.3):
    e, w, nl, post = run(period=1, warm_frac=wf)
    print(f"  {100 * wf:>9.0f}%{100 * nl / tot:>10.1f}%{post.mean():>10.3f}"
          f"{w:>8.3f}{100 * (post > 3).mean():>10.2f}%")
print("  参考：同段落「场单独（=1.091 口径）」约为 1.09~1.15")
