"""因果目标分布匹配（目标日志档）—— 修正版评测。

修正项（2026-10-26 审计）：
1. **真实 cm 尺度**：用索引的 `screen_w * cm_px_x` / `screen_h * cm_px_y`，
   不再用基线误差反推的代理尺度；
2. **排除暖启动**：只评估 `t >= W` 的帧（W 帧之前没有映射）；
3. **TTA 因果段对齐**：`--segment post` 只评估会话后 30%（TTA 适配段之外），
   避免 TTA 在“见过的帧”上评估；
4. **按被试聚类 bootstrap**：只有 5 个被试，按会话重采样会低估方差；
5. **窗口选择**：`--val-preds` 在验证被试上选 W，再到 test 评一次。

用法：
  python -m scripts.analysis.causal_dist_match \
      --preds out/rgbdgaze/analysis/tta_sym_photo_frames.csv \
      --index-csv data/RGBDGaze/index/index.csv \
      --windows 20 30 40 50 60 75 100 \
      --segment post --bootstrap 1000 \
      --val-preds out/rgbdgaze/analysis/val_base_frames.csv
"""
import argparse
import csv
from collections import defaultdict

import numpy as np


def load_preds(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    S = defaultdict(list)
    for r in rows:
        S[r["session"]].append(r)
    for k in S:
        S[k].sort(key=lambda x: int(x["frame_no"]))
    return S


def load_scale(index_csv):
    """session 目录 -> (cm per 归一化 x, cm per 归一化 y)。"""
    sc = {}
    with open(index_csv, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            s = "/".join(r["rgb_path"].split("/")[:-1])
            if s not in sc:
                sc[s] = (float(r["screen_w"]) * float(r["cm_px_x"]),
                         float(r["screen_h"]) * float(r["cm_px_y"]))
    return sc


def arr(rr, k):
    return np.array([float(x[k]) for x in rr])


def em(p, l, sx, sy):
    return np.hypot((p[:, 0] - l[:, 0]) * sx, (p[:, 1] - l[:, 1]) * sy)


def qmap_past(p, tgt, W):
    out = p.copy()
    for t in range(W, len(p)):
        pw, tw = p[t - W:t], tgt[t - W:t]
        for d in range(2):
            q = (pw[:, d] <= p[t, d]).mean()
            out[t, d] = np.quantile(tw[:, d], q)
    return out


def cmm_past(p, tgt, W):
    out = p.copy()
    for t in range(W, len(p)):
        pw, tw = p[t - W:t], tgt[t - W:t]
        s = tw.std(0) / (pw.std(0) + 1e-6)
        out[t] = p[t] * s + (tw.mean(0) - s * pw.mean(0))
    return out


def evaluate(S, scale, W, keys=("base", "tta"), segment="post"):
    """返回 {方法: {会话: 每帧误差数组}}；只保留 t>=W 且（可选）后 30% 的帧。"""
    out = defaultdict(dict)
    for s, rr in S.items():
        if s not in scale:
            continue
        sx, sy = scale[s]
        n = len(rr)
        start = int(0.7 * n) if segment == "post" else 0
        start = max(start, W)
        if n - start < 20:
            continue
        l = np.stack([arr(rr, "label_x"), arr(rr, "label_y")], 1)[start:]
        for key in keys:
            if key + "_px" not in rr[0]:
                continue
            p = np.stack([arr(rr, key + "_px"), arr(rr, key + "_py")], 1)
            out[key + "_raw"][s] = em(p[start:], l, sx, sy)
            out[key + "_cmm"][s] = em(cmm_past(p, np.stack([arr(rr, "label_x"),
                                                             arr(rr, "label_y")], 1), W)[start:], l, sx, sy)
            out[key + "_q"][s] = em(qmap_past(p, np.stack([arr(rr, "label_x"),
                                                           arr(rr, "label_y")], 1), W)[start:], l, sx, sy)
    return out


def subject_of(session):
    return session.split("/")[3]


def summarize(out, boot=0, seed=0):
    """帧加权 EM + 会话等权 + 最差 + 退化 + 按被试聚类 bootstrap CI。"""
    base = out["base_raw"]
    sessions = sorted(base)
    res = {}
    for k, d in out.items():
        if set(d) != set(sessions):
            continue
        e = np.concatenate([d[s] for s in sessions])
        fw = float(e.mean())
        se = float(np.mean([d[s].mean() for s in sessions]))
        r = np.array([d[s].mean() / base[s].mean() - 1 for s in sessions])
        ci = (float("nan"), float("nan"))
        if boot:
            subs = sorted({subject_of(s) for s in sessions})
            by_sub = {u: [s for s in sessions if subject_of(s) == u] for u in subs}
            rng = np.random.RandomState(seed)
            vals = []
            for _ in range(boot):
                pick = rng.choice(subs, len(subs), replace=True)
                sel = [s for u in pick for s in by_sub[u]]
                vals.append(np.concatenate([d[s] for s in sel]).mean())
            ci = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
        res[k] = dict(fw=fw, se=se, worst=100 * r.max(), degrade=int((r > 0.02).sum()),
                      n_sess=len(sessions), ci=ci)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--val-preds", default=None, help="验证集 base 预测（用于选 W）")
    ap.add_argument("--windows", type=int, nargs="+", default=[20, 30, 40, 50, 60, 75, 100])
    ap.add_argument("--segment", choices=["full", "post"], default="post")
    ap.add_argument("--bootstrap", type=int, default=1000)
    a = ap.parse_args()
    scale = load_scale(a.index_csv)
    S = load_preds(a.preds)

    best_w = None
    if a.val_preds:
        Sv = load_preds(a.val_preds)
        rows = []
        for W in a.windows:
            r = summarize(evaluate(Sv, scale, W, keys=("base",), segment=a.segment), boot=0)
            rows.append((r["base_q"]["fw"], W))
        rows.sort()
        best_w = rows[0][1]
        print("[val] W 选择（base_q 帧加权）：")
        for e, W in rows:
            print(f"   W={W:3d}  {e:.4f}")
        print(f"[val] 选定 W={best_w}")

    for W in ([best_w] if best_w else a.windows):
        out = evaluate(S, scale, W, segment=a.segment)
        res = summarize(out, boot=a.bootstrap)
        if not res:
            continue
        B = res["base_raw"]["fw"]
        print(f"\n=== W={W}  segment={a.segment} ===")
        for k in ("base_raw", "base_cmm", "base_q", "tta_raw", "tta_cmm", "tta_q"):
            if k not in res:
                continue
            d = res[k]
            print(f"  {k:9s} {d['fw']:.4f} ({100*(d['fw']/B-1):+.1f}%)  "
                  f"会话等权 {d['se']:.4f}  最差 {d['worst']:+.0f}%  "
                  f"退化 {d['degrade']}/{d['n_sess']}  "
                  f"CI [{d['ci'][0]:.4f},{d['ci'][1]:.4f}]")


if __name__ == "__main__":
    main()
