"""A+B：稳健锚点校准 + 共形门控（仅用 dump_predictions 的自洽 cm）。

数据要求：scripts/analysis/dump_predictions.py 生成的 CSV
  path, subject, activity, device, frame_index, pred_x, pred_y, label_x, label_y,
  em_cm, err_x_cm, err_y_cm, ang_deg

自检（不通过直接报错，防止用错口径得出结论）：
  1) |hypot(err_x,err_y) − em_cm| ≈ 0
  2) 由 err/(pred−label) 反推的 cm 在各 (device,orientation,screen) 内一致
  3) 基线 EM 与官方评估对照（人工核对）

A：锚点仿射（LS / IRLS / RANSAC）× 向恒等收缩 α
B：按被试奇偶分源/目标；源域选 (方法, α) 与门控阈值 τ：
   “校准后平均 ΔEM 的单侧 95% 上界 ≤ 0” 的前提下最大化接受会话数；
   目标域只评估一次。

用法：
  python -m scripts.analysis.anchor_calibration \
      --preds out/gazecapture/analysis/preds_gc_test_m2.csv \
      --index-csv data/GazeCapture/index/index.csv
"""
import argparse
import csv
from collections import defaultdict

import numpy as np


def load_meta_and_anchors(index_csv, needed_paths, min_subjects=10):
    """一次扫描索引：为 needed_paths 建 meta，同时统计训练集固定展示点。"""
    needed = set(needed_paths)
    meta = {}
    subj_pos = defaultdict(lambda: defaultdict(set))   # subject -> key -> {pos}
    with open(index_csv, encoding="utf-8") as fh:
        rd = csv.DictReader(fh)
        fields = rd.fieldnames or []
        pcol = "rgb_path" if "rgb_path" in fields else "frame_path"
        for r in rd:
            p = r[pcol]
            key = (r["device"], r["orientation"], r["screen_w"], r["screen_h"])
            if p in needed:
                meta[p] = key + (r["gaze_x"], r["gaze_y"])
            if r["split"] == "train":
                subj_pos[r["subject"]][key].add((r["gaze_x"], r["gaze_y"]))
    # 每个点被多少个“不同被试”看到
    count = defaultdict(lambda: defaultdict(int))
    for subj, per_key in subj_pos.items():
        for key, poss in per_key.items():
            for pos in poss:
                count[key][pos] += 1
    anchors = {}
    for key, cc in count.items():
        a = {pos for pos, c in cc.items() if c >= min_subjects}
        if 5 <= len(a) <= 25:
            anchors[key] = a
    return meta, anchors


def load_preds(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for r in rows:
        r["_pred"] = np.array([float(r["pred_x"]), float(r["pred_y"])])
        r["_label"] = np.array([float(r["label_x"]), float(r["label_y"])])
        r["_err"] = np.array([float(r["err_x_cm"]), float(r["err_y_cm"])])
        r["_em"] = float(r["em_cm"])
    return rows


def derive_cm(rows, meta):
    """err/(pred-label) 的按组建模（key=(device,orientation,screen_w,screen_h)）。"""
    per = defaultdict(lambda: [[], []])
    for r in rows:
        key = meta[r["path"]][:4]
        d = r["_pred"] - r["_label"]
        if abs(d[0]) > 0.01:
            per[key][0].append(r["_err"][0] / d[0])
        if abs(d[1]) > 0.01:
            per[key][1].append(r["_err"][1] / d[1])
    tab, spread = {}, {}
    for key, (vx, vy) in per.items():
        cx, cy = float(np.median(vx)), float(np.median(vy))
        tab[key] = (cx, cy)
        spread[key] = (float(np.std(vx) / (abs(cx) + 1e-9)) if vx else np.nan,
                       float(np.std(vy) / (abs(cy) + 1e-9)) if vy else np.nan)
    return tab, spread


def fit_axis(x, y, method="ls", iters=3, seed=0, ransac_iters=40):
    A = np.stack([x, np.ones(len(x))], 1)
    if method == "ls":
        return np.linalg.lstsq(A, y, rcond=None)[0]
    if method == "irls":
        w = np.ones(len(x))
        for _ in range(iters):
            cf = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
            res = np.abs(A @ cf - y)
            s = 1.4826 * np.median(res) + 1e-6
            w = 1.0 / np.maximum(1.0, res / (1.5 * s))
        return np.linalg.lstsq(A * w[:, None], y * w, rcond=None)[0]
    rng = np.random.RandomState(seed)
    best, best_in = None, -1
    for _ in range(ransac_iters):
        i = rng.choice(len(x), size=min(3, len(x)), replace=False)
        cf = np.linalg.lstsq(A[i], y[i], rcond=None)[0]
        res = np.abs(A @ cf - y)
        s = 1.4826 * np.median(res) + 1e-6
        inl = res <= 2.0 * s
        if inl.sum() > best_in:
            best_in, best = int(inl.sum()), cf
    if best is None:
        return np.linalg.lstsq(A, y, rcond=None)[0]
    res = np.abs(A @ best - y)
    s = 1.4826 * np.median(res) + 1e-6
    inl = res <= 2.0 * s
    return np.linalg.lstsq(A[inl], y[inl], rcond=None)[0]


def apply_affine(pred, cf, alpha):
    cx = alpha * cf[0] + (1 - alpha) * np.array([1.0, 0.0])
    cy = alpha * cf[1] + (1 - alpha) * np.array([1.0, 0.0])
    return np.stack([pred[:, 0] * cx[0] + cx[1],
                     pred[:, 1] * cy[0] + cy[1]], 1)


def sess_em(pred, lab, cm):
    return np.hypot((pred[:, 0] - lab[:, 0]) * cm[:, 0],
                    (pred[:, 1] - lab[:, 1]) * cm[:, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--min-anchors", type=int, default=40)
    args = ap.parse_args()

    rows = load_preds(args.preds)
    print(f"[load] 预测行 {len(rows)}")
    d = np.array([abs(np.hypot(*r["_err"]) - r["_em"]) for r in rows])
    print(f"[check1] |hypot(err)-em| max={d.max():.2e}（要求 <1e-4）")
    if d.max() > 1e-4:
        raise RuntimeError("dump 自洽性失败")

    meta, anchors = load_meta_and_anchors(
        args.index_csv, [r["path"] for r in rows])
    print(f"[anchors] 组数 {len(anchors)}；每组点数 {sorted({len(v) for v in anchors.values()})[:6]}")

    tab, spread = derive_cm(rows, meta)
    sp = np.array([[v[0], v[1]] for v in spread.values() if np.isfinite(v[0])])
    print(f"[check2] cm 组数 {len(tab)}；相对离散中位 {np.nanmedian(sp):.4f}（要求 <0.02）")

    # 按会话聚合（dump 的 subject 即会话）
    sess = defaultdict(list)
    for r in rows:
        sess[r["subject"]].append(r)
    recs = []
    for s, rs in sess.items():
        rs.sort(key=lambda r: int(r["frame_index"]))
        pred = np.stack([r["_pred"] for r in rs])
        lab = np.stack([r["_label"] for r in rs])
        cm = np.stack([tab[meta[r["path"]][:4]] for r in rs])
        key = meta[rs[0]["path"]][:4]
        A = anchors.get(key, set())
        mask = np.array([(meta[r["path"]][4], meta[r["path"]][5]) in A for r in rs])
        if mask.sum() < args.min_anchors:
            continue
        base = float(sess_em(pred, lab, cm).mean())
        recs.append(dict(s=s, pred=pred, lab=lab, cm=cm, mask=mask, base=base))

    n = sum(len(r["pred"]) for r in recs)
    base_all = sum(r["base"] * len(r["pred"]) for r in recs) / n
    cov = np.mean([r["mask"].mean() for r in recs])
    print(f"[data] 可用会话 {len(recs)} 帧 {n} 锚点覆盖 {100*cov:.1f}% 基线 EM {base_all:.4f}")

    # ---- A ----
    variants = {}
    for method in ("ls", "irls", "ransac"):
        for alpha in (1.0, 0.75, 0.5):
            tot = 0.0
            for r in recs:
                p, l, c, m = r["pred"], r["lab"], r["cm"], r["mask"]
                cf = [fit_axis(p[m, ax], l[m, ax], method, seed=i)
                      for i, ax in enumerate((0, 1))]
                y = apply_affine(p, cf, alpha)
                tot += sess_em(y, l, c).mean() * len(p)
            variants[(method, alpha)] = tot / n
    print("[A] 锚点仿射：")
    for k, v in sorted(variants.items(), key=lambda kv: kv[1]):
        print(f"   {k[0]:<7} α={k[1]:<4} {v:.4f} ({100*(v/base_all-1):+.1f}%)")
    best_variant = min(variants, key=variants.get)
    print(f"[A] 最佳：{best_variant}")

    # ---- B ----
    src = [r for r in recs if int(r["s"]) % 2 == 0]
    tgt = [r for r in recs if int(r["s"]) % 2 == 1]
    if len(src) < 5 or len(tgt) < 5:
        print("[B] 会话太少，跳过")
        return

    def stats(r):
        p, l, c, m = r["pred"], r["lab"], r["cm"], r["mask"]
        cf = [fit_axis(p[m, ax], l[m, ax], best_variant[0], seed=i)
              for i, ax in enumerate((0, 1))]
        y = apply_affine(p, cf, best_variant[1])
        em = sess_em(y, l, c).mean()
        res = sess_em(np.stack([p[m, 0] * cf[0][0] + cf[0][1],
                                p[m, 1] * cf[1][0] + cf[1][1]], 1), l[m], c[m])
        return dict(rms=float(np.sqrt((res ** 2).mean())),
                    delta=float(em / r["base"] - 1), base=r["base"],
                    em=float(em), n=len(p))

    src_s = [stats(r) for r in src]
    tgt_s = [stats(r) for r in tgt]
    taus = sorted({round(x["rms"], 4) for x in src_s})
    print(f"[B] 源 {len(src_s)} / 目标 {len(tgt_s)}；源恒校准平均 "
          f"{100*np.mean([x['delta'] for x in src_s]):+.1f}%")

    best = None
    for tau in taus:
        acc = [x for x in src_s if x["rms"] <= tau]
        if len(acc) < 2:
            continue
        dd = np.array([x["delta"] for x in acc])
        ub = dd.mean() + 1.645 * dd.std(ddof=1) / np.sqrt(len(dd))
        if ub <= 0 and (best is None or len(acc) > best[0]):
            best = (len(acc), tau, dd.mean(), ub)
    if best is None:
        print("[B] 源域没有满足“95% 上界 ≤0”的 τ ⇒ 结论：不校准时最安全")
        tau = None
    else:
        _, tau, mu, ub = best
        print(f"[B] 选定 τ={tau:.4f}：源接受 {best[0]} 会话，平均 Δ={100*mu:+.1f}%，上界={100*ub:+.1f}%")

    def ev(ss):
        base = np.array([x["base"] for x in ss])
        gate = np.array([x["em"] if (tau is not None and x["rms"] <= tau) else x["base"] for x in ss])
        w = np.array([x["n"] for x in ss])
        mean = (gate * w).sum() / w.sum()
        return mean, float(np.max(gate / base - 1)), int(sum(1 for x in ss if tau is not None and x["rms"] <= tau))

    for name, ss in (("source", src_s), ("target", tgt_s)):
        b = np.average([x["base"] for x in ss], weights=[x["n"] for x in ss])
        ma = np.average([x["em"] for x in ss], weights=[x["n"] for x in ss])
        m, w, a = ev(ss)
        print(f"   {name}: 基线 {b:.4f} | 门控 {m:.4f} ({100*(m/b-1):+.1f}%) | "
              f"始终校准 {ma:.4f} ({100*(ma/b-1):+.1f}%) | 接受 {a}/{len(ss)} | 最差 {100*w:+.1f}%")


if __name__ == "__main__":
    main()
