"""0.287 的归因对照：这个数字到底有多少来自模型，多少来自真值流的惯性？

估计器（= 产生 0.287 的那个）：
    pred_t = w_t·A_t + (1-w_t)·B_t
    B_t = L_{t-1} + (L_{t-1}-L_{t-2})     ← 真值流的一步外推（部署时拿不到）
    A_t = 校准后的模型输出（累积误差场+仿射）
    w_t 由过去（带遗忘）最小二乘闭式给出

对照（只换 A，B 与自适应机制完全相同）：
  ① A = 真实模型输出                        （原设定）
  ② A = 同一会话内**错位一半**的模型输出      ⇒ 外观信息与当前帧脱钩，但分布相同
  ③ A = 常数 (0.5,0.5)                     ⇒ 完全无图像信息
  ④ A = 同被试**另一会话**的模型输出          ⇒ 跨会话错位
若 ②③④ 的结果 ≈ ① ⇒ 模型无贡献，0.287 纯来自真值流惯性 ✗
另打印各情形下学到的平均权重 w 与「A 单独」的误差，便于看清来源。
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/xieliangcai/code/gazeformer-exps")
CSV = ROOT / "out/rgbdgaze/analysis/eye224_test.csv"
G, MIN_C = 8, 8
LAM, LAM_F, GAMMA, CAP, SIGMA = 1.0, 0.99, 1.0, 0.05, 3.0

sc = {}
with open(ROOT / "data/RGBDGaze/index/index.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        sc[r["rgb_path"]] = (float(r["screen_w"]) * float(r["cm_px_x"]),
                             float(r["screen_h"]) * float(r["cm_px_y"]))
sess = defaultdict(list)
for r in csv.DictReader(open(CSV, encoding="utf-8")):
    sess[r["path"].rsplit("/", 1)[0]].append(
        dict(fi=int(r["frame_index"]), path=r["path"], subj=r["subject"],
             P=np.array([float(r["pred_x"]), float(r["pred_y"])]),
             L=np.array([float(r["label_x"]), float(r["label_y"])])))
for s in sess:
    sess[s].sort(key=lambda r: r["fi"])
SESS = sorted(sess)
P_ALL = {s: np.array([r["P"] for r in sess[s]]) for s in SESS}   # 便于错位取样


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


def run(a_mode="real"):
    """a_mode: real | shift_half | const | other_session"""
    errs, aerrs, ws = [], [], []
    for k, s in enumerate(SESS):
        v = sess[s]
        N = len(v)
        shift = N // 2
        other = SESS[(k + 1) % len(SESS)]
        fd = Tracker()
        prevL = prevL2 = None
        num = np.zeros(2); den = np.zeros(2)
        for t, r in enumerate(v):
            P, L = r["P"], r["L"]
            A_t, p1, bv = fd.correct(P)
            if a_mode == "shift_half":
                A = P_ALL[s][t - shift] if t >= shift else P_ALL[s][t + shift]
            elif a_mode == "const":
                A = np.array([0.5, 0.5])
            elif a_mode == "other_session":
                j = min(t, len(sess[other]) - 1)
                A = P_ALL[other][j]
            else:
                A = A_t
            if prevL is None:
                pred = A
            else:
                step = np.zeros(2) if prevL2 is None else (prevL - prevL2)
                B = prevL + GAMMA * np.clip(step, -CAP, CAP)
                w = np.clip(num / np.where(den > 1e-9, den, 1e9), 0.0, 1.0)
                pred = w * A + (1 - w) * B
                ws.append(float(np.mean(w)))
            errs.append(emv(pred, L, r["path"]))
            aerrs.append(emv(A, L, r["path"]))
            if prevL is not None:
                step = np.zeros(2) if prevL2 is None else (prevL - prevL2)
                B = prevL + GAMMA * np.clip(step, -CAP, CAP)
                d = A - B; e = L - B
                num = LAM_F * num + d * e
                den = LAM_F * den + d * d
            fd.update(P, L, p1, bv)
            prevL2, prevL = prevL, L
    return np.array(errs), np.array(aerrs), np.array(ws)


base = np.array([emv(r["P"], r["L"], r["path"]) for s in SESS for r in sess[s]])
print("=" * 104)
print(f"模型原始 {base.mean():.3f}；真值流外推单独 ~0.362；上一帧真值 0.501")
print("=" * 104)
print(f"  {'A 的内容':<34}{'融合EM':>9}{'A单独EM':>10}{'平均w':>8}{'>3cm':>8}")
for tag, md in (("① 真实模型（原设定）", "real"),
                ("② 会话内错位一半的模型", "shift_half"),
                ("③ 常数 (0.5,0.5)", "const"),
                ("④ 同被试另一会话的模型", "other_session")):
    e, ae, w = run(md)
    print(f"  {tag:<34}{e.mean():>9.3f}{ae.mean():>10.3f}"
          f"{w.mean() if len(w) else 0:>8.3f}{100 * (e > 3).mean():>7.2f}%")
