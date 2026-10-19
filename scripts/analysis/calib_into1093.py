"""把本轮新零件尝试装进 1.093 那条流水线（密标签 / 因果在线 / 键p1 / 仿射）。

1.093 = field_final.py 口径：逐帧因果在线（每帧预测后再用该帧标签更新），
        全局仿射(斜率不钳) + 按 p1 归格的 3x3 池化误差场 + min-count。

本脚本三部分：
  ① 复现 1.093，并检查"偏移钳制"是否真的免费（斜率钳制已知有代价 ✗）
  ② 级联装法（spread 大 K，一次性）：把整条 V1 流水线的基座换成 C2(特征+预测→残差) 的输出
  ③ 因果在线链路（= 部署形态）：C2 每 500 帧用"迄今已见标签(最多1000个,等距抽样)"重拟合，
     其输出再喂给 V1 的场 ⇒ 与 1.093 直接可比
"""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/xieliangcai/code/gazeformer-exps")
NPZ = ROOT / "out/rgbdgaze/analysis/feats_eye224_test.npz"
G, MIN, NZ = 8, 8, 12
ALPHAS = np.logspace(-2, 4, 7)

sc = {}
with open(ROOT / "data/RGBDGaze/index/index.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        sc[r["rgb_path"]] = (float(r["screen_w"]) * float(r["cm_px_x"]),
                             float(r["screen_h"]) * float(r["cm_px_y"]))

d = np.load(NPZ, allow_pickle=True)
F = d["feats"].astype(np.float64)
P = d["pred"].astype(np.float64)
L = d["label"].astype(np.float64)
paths, subj, fi = d["path"], d["subject"], d["frame_index"]

# ---- 投影特征（calibrate_final 口径）：面脸CLS/patch 段经任务训练过的线性投影 ----
nseg = F.shape[1] // 4
import torch  # noqa: E402

st = torch.load(ROOT / "out/rgbdgaze/ablation/checkpoints/eye224_mgpu/best.pt",
                map_location="cpu", weights_only=False)


def _get(n):
    for k in (n, f"module.{n}", f"model.{n}"):
        if k in st:
            return st[k].float().numpy()
    raise KeyError(n)


Wc, bc = _get("proj_rgb_cls.weight"), _get("proj_rgb_cls.bias")
Wp, bp = _get("proj_rgb_patch.weight"), _get("proj_rgb_patch.bias")
CLS = F[:, nseg:2 * nseg] @ Wc.T + bc
PAT = F[:, 2 * nseg:3 * nseg] @ Wp.T + bp
BIAS = np.linalg.norm(P - 0.5, axis=1, keepdims=True)
XC2 = np.concatenate([CLS, PAT, P, BIAS], 1)          # C2 的输入（含预测）

by_subj = defaultdict(list)
for i in range(len(P)):
    by_subj[str(subj[i])].append(i)
for s in by_subj:
    by_subj[s].sort(key=lambda i: fi[i])
SUB = sorted(by_subj)


def cidx(pt):
    return min(G - 1, max(0, int(pt[0] * G))) * G + min(G - 1, max(0, int(pt[1] * G)))


def em(pred, lab, path):
    w, h = sc[path]
    return float(np.hypot((pred[0] - lab[0]) * w, (pred[1] - lab[1]) * h))


def affine_of(glob, slope=np.inf):
    if glob[0] < MIN:
        return None
    M = np.array([[glob[0], glob[1], glob[2]], [glob[1], glob[3], glob[4]],
                  [glob[2], glob[4], glob[5]]]) + 0.01 * glob[0] * np.diag([0, 1, 1])
    try:
        b1 = np.linalg.solve(M, glob[6:9])
        b2 = np.linalg.solve(M, np.array([glob[9], glob[10], glob[11]]))
    except np.linalg.LinAlgError:
        return None
    if np.isfinite(slope):
        b1 = np.array([b1[0], np.clip(b1[1], -slope, slope), np.clip(b1[2], -slope, slope)])
        b2 = np.array([b2[0], np.clip(b2[1], -slope, slope), np.clip(b2[2], -slope, slope)])
    return b1, b2


def grid_lookup(grid, p1):
    c = cidx(p1); i0, j0 = c // G, c % G
    acc = np.zeros(2); cnt = 0.0
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            ii, jj = i0 + di, j0 + dj
            if 0 <= ii < G and 0 <= jj < G:
                cc = grid[ii * G + jj]
                cnt += cc[0]; acc += cc[1:]
    return acc / cnt if cnt >= MIN else None


class Ridge:
    """固定 α 或 LOO 自选 α 的对偶岭回归（n<<d 场景）。"""

    def __init__(self, alpha=None, alphas=ALPHAS):
        self.alpha, self.alphas = alpha, alphas

    def fit(self, X, Y):
        self.mx_, self.sx_ = X.mean(0), X.std(0) + 1e-6
        self.my_ = Y.mean(0)
        Xs, Yc = (X - self.mx_) / self.sx_, Y - self.my_
        K = Xs @ Xs.T
        w, V = np.linalg.eigh(K)
        self.Xs_, self.V_, self.w_ = Xs, V, w
        a = self.alpha
        if a is None:
            V2, best, a = V ** 2, np.inf, self.alphas[0]
            for cand in self.alphas:
                h = V2 @ (w / (w + cand))
                num = V @ ((V.T @ Yc) / (w + cand)[:, None])
                r = (Yc - num) / np.clip(1.0 - h, 1e-6, None)[:, None]
                mse = float((r ** 2).mean())
                if mse < best:
                    best, a = mse, cand
        self.alpha_ = a
        self.coef_ = V @ ((V.T @ Yc) / (w + a)[:, None])
        return self

    def predict(self, X):
        return self.my_ + (((X - self.mx_) / self.sx_) @ self.Xs_.T) @ self.coef_


def stats_factory():
    return np.zeros(NZ), defaultdict(lambda: [0.0, 0.0, 0.0])


def add_stat(glob, grid, base_pt, lab, p1, bv):
    x, y = float(base_pt[0]), float(base_pt[1])
    glob[0] += 1
    glob[1] += x; glob[2] += y
    glob[3] += x * x; glob[4] += x * y; glob[5] += y * y
    glob[6] += base_pt[0] - lab[0]; glob[7] += (base_pt[0] - lab[0]) * x
    glob[8] += (base_pt[0] - lab[0]) * y
    glob[9] += base_pt[1] - lab[1]; glob[10] += (base_pt[1] - lab[1]) * x
    glob[11] += (base_pt[1] - lab[1]) * y
    cg = grid[cidx(p1)]
    cg[0] += 1
    cg[1] += float(base_pt[0] - lab[0]) - float(bv[0])
    cg[2] += float(base_pt[1] - lab[1]) - float(bv[1])


def stage2(base_fn, clip, slope=np.inf, online=True, refit=None):
    """V1 流水线：base_fn(i) 给出该帧的基座预测；返回逐帧修正预测。"""
    acc = []
    for s in SUB:
        idx = by_subj[s]
        glob, grid = stats_factory()
        m2 = None
        for n, i in enumerate(idx):
            if online and refit and n % 500 == 0 and n >= 500:
                seen = idx[:n]
                st_ = np.unique(np.linspace(0, len(seen) - 1, min(1000, len(seen))).astype(int))
                sel = [seen[t] for t in st_]
                m2 = Ridge(alpha=None).fit(XC2[sel], L[sel] - P[sel])
            base = base_fn(i, m2)
            b = affine_of(glob, slope)
            bv = np.array([np.array([1.0, base[0], base[1]]) @ b[0],
                           np.array([1.0, base[0], base[1]]) @ b[1]]) if b else np.zeros(2)
            p1 = base - bv
            dd = grid_lookup(grid, p1)
            corr = p1 - np.clip(dd, -clip, clip) if dd is not None else p1
            acc.append((i, corr))
            add_stat(glob, grid, base, L[i], p1, bv)
    order = [i for i, _ in acc]
    m = {i: c for i, c in acc}
    return order, m


# ---------------- ① 复现 + 钳制是否免费（全标签、因果在线） ----------------
def part1():
    rows = []
    for clip, slope in ((np.inf, np.inf), (0.10, np.inf), (0.05, np.inf), (0.02, np.inf),
                        (np.inf, 0.5), (np.inf, 0.2)):
        order, m = stage2(lambda i, _m: P[i], clip, slope)
        e = np.array([em(m[i], L[i], str(paths[i])) for i in order])
        rows.append((clip, slope, e.mean(), (e > 2).mean(), (e > 3).mean(),
                     (e > 5).mean(), e.max()))
    print("=" * 104)
    print("① 装进 1.093 的那条流水线：偏移钳制免费吗？（全标签 / 因果在线 / 键p1 / 仿射）")
    print("=" * 104)
    print(f"  {'偏移钳制':>9}{'斜率钳制':>9}{'EM均值':>9}{'>2cm':>8}{'>3cm':>8}{'>5cm':>8}{'max':>8}")
    for c, sl, mu, p2, p3, p5, mx in rows:
        print(f"  {('∞' if not np.isfinite(c) else f'{c:.2f}'):>9}"
              f"{('∞' if not np.isfinite(sl) else f'{sl:.2f}'):>9}"
              f"{mu:>9.3f}{p2 * 100:>7.2f}%{p3 * 100:>7.2f}%{p5 * 100:>7.2f}%{mx:>8.2f}")
    return rows


# ---------------- ② 级联装法（spread 大 K，一次性） ----------------
def part2(Ks=(500, 1000, 2000)):
    print()
    print("=" * 104)
    print("② 把 V1 流水线的基座换成 C2(特征+预测→残差) 的输出（交错 K 帧标定，偏移钳制0.05）")
    print("=" * 104)
    print(f"  {'K':>6}{'A不标定':>9}{'V1场(基座=原始)':>19}{'C2':>17}{'C2+V1场(级联)':>19}{'级联 vs V1':>12}")
    for K in Ks:
        res = {}
        eA = []
        for s in SUB:
            idx = by_subj[s]
            N = len(idx)
            if N < 400:
                continue
            cal = [idx[t] for t in np.unique(np.linspace(0, N - 1, min(K, N // 2)).astype(int))]
            calset = set(cal)
            ev = [i for i in idx if i not in calset]
            m2 = Ridge(alpha=None).fit(XC2[cal], L[cal] - P[cal])
            g2 = P + m2.predict(XC2)
            # V1 场（基座=原始）
            glob, grid = stats_factory()
            for i in cal:
                b = affine_of(glob)
                bv = np.array([np.array([1.0, P[i][0], P[i][1]]) @ b[0],
                               np.array([1.0, P[i][0], P[i][1]]) @ b[1]]) if b else np.zeros(2)
                add_stat(glob, grid, P[i], L[i], P[i] - bv, bv)
            b1 = affine_of(glob)
            # 级联场（基座=C2输出）
            glob2, grid2 = stats_factory()
            for i in cal:
                b = affine_of(glob2)
                bv = np.array([np.array([1.0, g2[i][0], g2[i][1]]) @ b[0],
                               np.array([1.0, g2[i][0], g2[i][1]]) @ b[1]]) if b else np.zeros(2)
                add_stat(glob2, grid2, g2[i], L[i], g2[i] - bv, bv)
            b2 = affine_of(glob2)

            def v1(i):
                bv = np.array([np.array([1.0, P[i][0], P[i][1]]) @ b1[0],
                               np.array([1.0, P[i][0], P[i][1]]) @ b1[1]]) if b1 else np.zeros(2)
                p1 = P[i] - bv
                dd = grid_lookup(grid, p1)
                return p1 - np.clip(dd, -0.05, 0.05) if dd is not None else p1

            def casc(i):
                bv = np.array([np.array([1.0, g2[i][0], g2[i][1]]) @ b2[0],
                               np.array([1.0, g2[i][0], g2[i][1]]) @ b2[1]]) if b2 else np.zeros(2)
                p1 = g2[i] - bv
                dd = grid_lookup(grid2, p1)
                return p1 - np.clip(dd, -0.05, 0.05) if dd is not None else p1

            for nm, fn in (("A", lambda i: P[i]), ("V1", v1), ("C2", lambda i: g2[i]),
                           ("C2V1", casc)):
                res.setdefault(nm, []).extend(
                    [em(fn(i), L[i], str(paths[i])) for i in ev])
        base = np.mean(res["A"])
        line = f"  {K:>6}{base:>9.3f}"
        for nm in ("V1", "C2", "C2V1"):
            line += f"{np.mean(res[nm]):>12.3f}({100 * (np.mean(res[nm]) / base - 1):>+4.1f}%)"
        line += f"{100 * (np.mean(res['C2V1']) / np.mean(res['V1']) - 1):>11.1f}%"
        print(line)


# ---------------- ③ 因果在线链路（部署形态） ----------------
def part3():
    print()
    print("=" * 104)
    print("③ 全因果在线：C2 每 500 帧用「迄今已见标签(≤1000等距)」重拟合，输出再喂 V1 场")
    print("=" * 104)
    orderA, mA = stage2(lambda i, _m: P[i], np.inf)          # 1.093 复现
    orderB, mB = stage2(lambda i, m2: P[i] if m2 is None else P[i] + m2.predict(XC2[i:i + 1])[0],
                        np.inf, refit=True)
    eA = np.array([em(mA[i], L[i], str(paths[i])) for i in orderA])
    eB = np.array([em(mB[i], L[i], str(paths[i])) for i in orderB])
    print(f"  {'方案':<26}{'EM均值':>9}{'>2cm':>8}{'>3cm':>8}{'>5cm':>8}{'max':>8}")
    for nm, e in (("A 不标定", None), ("V1 场（=1.093 口径）", eA), ("C2 + V1 场（级联）", eB)):
        if e is None:
            e = np.array([em(P[i], L[i], str(paths[i])) for i in orderA])
        print(f"  {nm:<26}{e.mean():>9.3f}{100 * (e > 2).mean():>7.2f}%"
              f"{100 * (e > 3).mean():>7.2f}%{100 * (e > 5).mean():>7.2f}%{e.max():>8.2f}")
    print(f"\n  级联相对 1.093 口径：{100 * (eB.mean() / eA.mean() - 1):+.1f}%")


part1()
part2()
part3()
