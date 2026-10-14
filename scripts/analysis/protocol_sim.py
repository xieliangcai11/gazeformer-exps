"""标定协议仿真：从"每帧都有真值"（1.076 上限）退化到**现实的周期标定**，
找出能逼近上限的最省协议与估计器。

协议模型（可部署）：
  每 N 帧放一个标定目标；该目标贡献 m 帧可用标注（延迟 m 帧后到账）。
  ⇒ 已知点预算 = m/N 帧/帧（例如 N=300,m=15 ⇒ 每 10 秒一次 0.5 秒目标）

估计器（只用"到账时刻之前"的标注，严格因果）：
  none   不修正
  const  最近 windowK 个标注的 (pred-label) 加权均值（2 参数）
  linear 同上，但做局部线性回归 d ≈ a + b·(pred - pred_t)（6 参数）
         权重 = 时间衰减 exp(-Δt/τ) × 注视邻近 exp(-|Δpred|²/2σ²)，带岭 λ

自检：none/const(windowK=30,N=1,m=1,等权) 应复现此前的 1.472 / 1.076。
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

IDX = Path("/data1/xieliangcai/code/gazeformer-exps/data/RGBDGaze/index")
sc = {}
with open(IDX / "index.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        sc[r["rgb_path"]] = (float(r["screen_w"]) * float(r["cm_px_x"]),
                             float(r["screen_h"]) * float(r["cm_px_y"]))

sess = defaultdict(list)
for r in csv.DictReader(open(sys.argv[1], encoding="utf-8")):
    sess[r["path"].rsplit("/", 1)[0]].append(
        dict(path=r["path"], fi=int(r["frame_index"]),
             P=(float(r["pred_x"]), float(r["pred_y"])),
             L=(float(r["label_x"]), float(r["label_y"]))))
for s in sess:
    sess[s].sort(key=lambda r: r["fi"])

# 预算化为 3x3 归一化坐标（用于注视邻近核，与像素无关）
session = {}
for s, v in sess.items():
    P = np.array([r["P"] for r in v])
    L = np.array([r["L"] for r in v])
    S = np.array([sc[r["path"]] for r in v])          # (w_cm, h_cm)
    session[s] = (P, L, S)


def simulate(N=1, m=1, windowK=30, tau=np.inf, sigma=np.inf, ridge=0.0,
             mode="const"):
    """返回 (全部帧误差, 已获标注帧数占比)。"""
    errs, n_lab, n_fr = [], 0, 0
    for s, (P, L, S) in session.items():
        T = len(P)
        revealed = []                                   # (t_idx, P, L)
        for t in range(T):
            n_fr += 1
            # 目标到账：第 t 帧恰是某目标的最后一帧时，整块 m 帧可见
            if N > 0 and m > 0 and t % N == m - 1:
                for k in range(t - m + 1, t + 1):
                    if k >= 0:
                        revealed.append((k, P[k], L[k]))
                        n_lab += 1
            if not revealed or mode == "none":
                corr = P[t]
            else:
                R = np.array([r[1] for r in revealed])
                D = np.array([r[1] - r[2] for r in revealed])
                tt = np.array([r[0] for r in revealed], dtype=float)
                w = np.ones(len(R))
                if np.isfinite(tau):
                    w *= np.exp(-(t - tt) / tau)
                if np.isfinite(sigma):
                    w *= np.exp(-((R - P[t]) ** 2).sum(1) / (2 * sigma ** 2))
                if windowK and len(R) > windowK:      # 只保留最近 windowK 个
                    keep = np.argsort(tt)[-windowK:]
                    R, D, w = R[keep], D[keep], w[keep]
                if w.sum() <= 1e-9:
                    corr = P[t]
                elif mode == "const":
                    d = (D * w[:, None]).sum(0) / w.sum()
                    corr = P[t] - d
                else:                                  # linear：局部线性误差场
                    X = np.hstack([np.ones((len(R), 1)), R - P[t]])
                    Xw = X * np.sqrt(w)[:, None]
                    Dw = D * np.sqrt(w)[:, None]
                    A = Xw.T @ Xw + ridge * np.eye(3)
                    try:
                        coef = np.linalg.solve(A, Xw.T @ Dw)   # (3,2)
                        corr = P[t] - coef[0]
                    except np.linalg.LinAlgError:
                        corr = P[t]
            e = float(np.hypot((corr[0] - L[t][0]) * S[t][0],
                               (corr[1] - L[t][1]) * S[t][1]))
            errs.append(e)
    return np.array(errs), n_lab / max(n_fr, 1)


def line(tag, e, frac, base):
    bm = float(np.asarray(base).mean())
    print(f"  {tag:<34}{e.mean():>9.3f}{100 * (e.mean() / bm - 1):>9.1f}%"
          f"{np.median(e):>8.3f}{100 * (e > 3).mean():>7.2f}%{100 * (e > 5).mean():>7.2f}%"
          f"{e.max():>7.2f}{frac * 100:>9.1f}%")


base, _ = simulate(mode="none")
print("=" * 116)
print(f"【标定协议仿真】{Path(sys.argv[1]).stem}  n={sum(len(v) for v in sess.values())}")
print("=" * 116)
HDR = (f"  {'协议/估计器':<34}{'EM均值':>9}{'vs基线':>10}{'中位':>8}{'>3cm':>8}"
       f"{'>5cm':>8}{'max':>7}{'已知点占比':>10}")
print(HDR)
line("基线（不标定）", base, 0.0, base)
c1, f1 = simulate(N=1, m=1, windowK=30, mode="const")
line("自检: 每帧已知 K=30 常数", c1, f1, base)

print("\n" + "-" * 116)
print("【第一轮：周期目标 × 估计器形态】  τ=600 帧(20s) σ=0.3 ridge=1e-3")
print("-" * 116)
print(HDR)
line("基线（不标定）", base, 0.0, base)
for N, m in ((30, 1), (30, 5), (60, 5), (90, 5), (150, 5), (150, 15), (300, 15),
             (600, 15)):
    for mode, tag in (("const", "常数"), ("linear", "局部线性")):
        e, f = simulate(N=N, m=m, windowK=0, tau=600, sigma=0.3, ridge=1e-3,
                        mode=mode)
        line(f"N={N}({N / 30:.1f}s) m={m} {tag}", e, f, base)

print("\n" + "-" * 116)
print("【第二轮：估计器超参扫描】固定 N=300, m=15（每 10 秒一次 0.5 秒目标）")
print("-" * 116)
print(HDR)
line("基线（不标定）", base, 0.0, base)
for tau in (np.inf, 1800, 600, 150):
    for sigma, tag_s in ((np.inf, "全屏"), (0.3, "σ0.3"), (0.15, "σ0.15")):
        e, f = simulate(N=300, m=15, windowK=0, tau=tau, sigma=sigma,
                        ridge=1e-3, mode="linear")
        line(f"τ={'∞' if tau == np.inf else int(tau)} σ={tag_s} 线性", e, f, base)
for ridge in (0.0, 1e-2, 1e-1):
    e, f = simulate(N=300, m=15, windowK=0, tau=600, sigma=0.3, ridge=ridge,
                    mode="linear")
    line(f"τ=600 σ=0.3 岭={ridge}", e, f, base)
