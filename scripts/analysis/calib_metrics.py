"""按训练脚本的 final 终评口径，给出「原始」与「误差场标定后」两套指标。

严格复刻 scripts/train.py 的 summarize()/predict_frames() 定义：
  l2_norm         归一化坐标下的平均欧氏距离 mean||pred-label||_2
  em_dist         平均物理误差 cm；em_x/em_y 为平均绝对分量
  ang_*           几何视线夹角：屏幕点按「到屏心偏移 + 基准视距 30cm」构成 3D 向量后取夹角
  ang_top95_mean  去掉最大 5% 后的角度均值；ang_ratio_le_191 ≤1.91° 占比
标定层与最优配置一致：全局仿射（每帧重解）→ 每格残差偏移（G=8，3x3 池化），严格因果。
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("/data1/xieliangcai/code/gazeformer-exps")
# 视角距离：从 configs 单一来源取，避免与训练/验收漂移（审计：此处曾是硬编码复本 30.0）
try:
    from configs.rgbdgaze_config import VIEWING_DISTANCE_CM as D_CM
except Exception:      # 直接以脚本方式运行且无 sys.path 引导时的兜底
    D_CM = 30.0
G, MIN, NZ = 8, 8, 12
CSV = sys.argv[1] if len(sys.argv) > 1 else str(
    ROOT / "out/rgbdgaze/analysis/eye224_test.csv")

sc = {}
with open(ROOT / "data/RGBDGaze/index/index.csv", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        sc[r["rgb_path"]] = (float(r["screen_w"]) * float(r["cm_px_x"]),
                             float(r["screen_h"]) * float(r["cm_px_y"]))

sess = defaultdict(list)
for r in csv.DictReader(open(CSV, encoding="utf-8")):
    sess[r["path"].rsplit("/", 1)[0]].append(
        dict(path=r["path"], sub=r["subject"], fi=int(r["frame_index"]),
             P=np.array([float(r["pred_x"]), float(r["pred_y"])]),
             L=np.array([float(r["label_x"]), float(r["label_y"])]),
             S=sc[r["path"]]))
for s in sess:
    sess[s].sort(key=lambda r: r["fi"])


def add(st, x, y, dx, dy):
    st[0] += 1
    st[1] += x; st[2] += y
    st[3] += x * x; st[4] += x * y; st[5] += y * y
    st[6] += dx; st[7] += dx * x; st[8] += dx * y
    st[9] += dy; st[10] += dy * x; st[11] += dy * y


preds, labels, scales, subs = [], [], [], []
for s, v in sess.items():
    glob = defaultdict(lambda: np.zeros(NZ))
    grid = defaultdict(lambda: [0.0, 0.0, 0.0])
    for r in v:
        P, L = r["P"], r["L"]
        x, y = float(P[0]), float(P[1])
        a = glob[""]
        b = None
        if a[0] >= MIN:
            A = np.array([[a[0], a[1], a[2]], [a[1], a[3], a[4]], [a[2], a[4], a[5]]]) \
                + 0.01 * a[0] * np.diag([0.0, 1.0, 1.0])
            try:
                b = (np.linalg.solve(A, a[6:9]), np.linalg.solve(A, np.array([a[9], a[10], a[11]])))
            except np.linalg.LinAlgError:
                b = None
        bv = np.array([np.array([1.0, x, y]) @ b[0], np.array([1.0, x, y]) @ b[1]]) \
            if b is not None else np.zeros(2)
        p1 = P - bv
        i0 = min(G - 1, max(0, int(float(p1[0]) * G)))
        j0 = min(G - 1, max(0, int(float(p1[1]) * G)))
        cnt = sx = sy = 0.0
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                i, j = i0 + di, j0 + dj
                if 0 <= i < G and 0 <= j < G:
                    c = grid[(i, j)]
                    cnt += c[0]; sx += c[1]; sy += c[2]
        corr = p1 - np.array([sx / cnt, sy / cnt]) if cnt >= MIN else p1
        preds.append(corr); labels.append(L); scales.append(r["S"]); subs.append(r["sub"])
        add(a, x, y, float(P[0] - L[0]), float(P[1] - L[1]))
        c = grid[(min(G - 1, max(0, int(float(p1[0]) * G))),
                  min(G - 1, max(0, int(float(p1[1]) * G))))]
        c[0] += 1
        c[1] += float(P[0] - L[0]) - float(bv[0])
        c[2] += float(P[1] - L[1]) - float(bv[1])

# 原始预测：重新读一遍（标定流里没保存原始）
rawP = []
for s, v in sess.items():
    rawP += [r["P"] for r in v]
order = [r for s in sess for r in sess[s]]
rawP = np.array([r["P"] for r in order])
PR = np.array(preds)
L = np.array(labels)
S = np.array(scales)
SUB = np.array(subs)


def report(tag, P):
    diff = P - L
    cmx = diff[:, 0] * S[:, 0]
    cmy = diff[:, 1] * S[:, 1]
    px = P[:, 0] * S[:, 0] - S[:, 0] / 2
    py = P[:, 1] * S[:, 1] - S[:, 1] / 2
    tx = L[:, 0] * S[:, 0] - S[:, 0] / 2
    ty = L[:, 1] * S[:, 1] - S[:, 1] / 2
    pv = np.stack([px, py, np.full_like(px, D_CM)], 1)
    tv = np.stack([tx, ty, np.full_like(tx, D_CM)], 1)
    pn = pv / np.linalg.norm(pv, axis=1, keepdims=True)
    tn = tv / np.linalg.norm(tv, axis=1, keepdims=True)
    dot = np.clip((pn * tn).sum(1), -1, 1)
    cross = np.linalg.norm(np.cross(pn, tn), axis=1)
    ang = np.degrees(np.arctan2(cross, dot))
    cnt = len(ang)
    l2 = np.linalg.norm(diff, axis=1).mean()
    n95 = max(1, int(cnt * 0.95))
    print(f"[final] {tag}")
    print(f"[final] test_L2 {l2:.4f}  test_em {np.hypot(cmx, cmy).mean():.2f}cm "
          f"(x:{np.abs(cmx).mean():.2f} y:{np.abs(cmy).mean():.2f})")
    print(f"[final] test_ang 平均 {ang.mean():.2f}° | 中位 {np.median(ang):.2f}° | "
          f"95%分位 {np.percentile(ang, 95):.2f}° | 前95%平均 {np.sort(ang)[:n95].mean():.2f}° | "
          f"≤1.91° {(ang <= 1.91).mean() * 100:.1f}% (n={cnt})")
    return dict(em=np.hypot(cmx, cmy), ang=ang, cmx=cmx, cmy=cmy)


print("=" * 78)
print("【按训练脚本口径：原始 vs 误差场标定后】（测试集 11031 帧 / 5 个未见被试）")
print("=" * 78)
r0 = report("原始（未标定）", rawP)
print()
r1 = report("误差场标定后（因果在线，索引=预测注视点）", PR)
print()
print("分被试（em 均值 cm：原始 → 标定后）：")
for s in sorted(set(SUB)):
    m = SUB == s
    print(f"  {s:<5} n={m.sum():>5}  {r0['em'][m].mean():.3f} → {r1['em'][m].mean():.3f}"
          f"  ({100 * (r1['em'][m].mean() / r0['em'][m].mean() - 1):+.1f}%)"
          f"   ≤1.91°: {100 * (r0['ang'][m] <= 1.91).mean():.1f}% → "
          f"{100 * (r1['ang'][m] <= 1.91).mean():.1f}%")
print(f"  全部  n={len(SUB):>5}  {r0['em'].mean():.3f} → {r1['em'].mean():.3f}"
      f"  ({100 * (r1['em'].mean() / r0['em'].mean() - 1):+.1f}%)")
print()
print("分布细节（cm）：")
for nm, e in (("原始", r0["em"]), ("标定后", r1["em"])):
    q = [np.percentile(e, p) for p in (50, 75, 90, 95, 99)]
    print(f"  {nm:<6} 中位 {q[0]:.3f} | p75 {q[1]:.3f} | p90 {q[2]:.3f} | p95 {q[3]:.3f} | "
          f"p99 {q[4]:.3f} | max {e.max():.2f}")
    print(f"         >1cm {100 * (e > 1).mean():.2f}% | >2cm {100 * (e > 2).mean():.2f}% | "
          f">3cm {100 * (e > 3).mean():.2f}% | >5cm {100 * (e > 5).mean():.2f}% | "
          f">8cm {100 * (e > 8).mean():.3f}%")
