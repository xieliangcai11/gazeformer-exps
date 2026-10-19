"""离线检验：**真误差能否由"无标签可观测特征"预测** ⇒ 决定"学适配策略"是否可行。

为什么需要（本仓库实测依据）：
  · 现有手搓准则与真误差**反向/无关**（Spearman −0.61 / −0.03）⇒ 选步把一个还在改善的
    适配过程扔掉了（p11/sitting：被选步 true 0.0480 vs 求解最优 0.0293，选择损失 +64%）。
  · 但适配本身有效（真误差随步数 −26%~−33%）。
  ⇒ 所以问题变成"**能不能用无标签特征预测真误差**"，若可以，就能**在源域上学一个策略**
    （源域标签只离线用于训练 ✓，测试时零标签 ✓✓）。

输出三张表：
  A. 代理质量：留一被试下，预测误差与真误差的 **Spearman 秩相关**（逐会话 + 合并）
  B. 策略成绩：每条规则所选步的**真误差**（对比 基线/现有准则/oracle）
  C. 特征重要性（岭回归系数绝对值，标准化后）

用法：
  python -m scripts.analysis.tta_policy_check "out/rgbdgaze/analysis/tta_probe_*.log"
"""
from __future__ import annotations

import argparse
import glob
import math
import re
import statistics
from collections import defaultdict

import numpy as np

# 兼容两种探针格式：旧（val/sm/bias/sig）与新（+cons/sigad/dmag ✓）
PAT = re.compile(
    r"\[probe\] step\s+(\d+)\s+val\s+([\d.naN]+)\s+true\(L1,归一化\)\s+([\d.]+)\s+"
    r"sm\s+([\d.]+)\s+(?:cons\s+([\d.]+)\s+)?bias\s+([\d.]+)\s+"
    r"sig\s+\[([\d.]+),([\d.]+)\]\s*(?:sigad\s+([\d.]+)\s+dmag\s+([\d.]+)\s+)?"
    r"(?:bright\s+([\d.]+)\s+sharp\s+([\d.]+)\s+jvar\s+([\d.]+)\s+)?")
# "Δ" 特征（相对步 0 的差值）；输入质量与不确定度类特征用原值 ⇒ 见 design()
DELTA_FEATS = ("val", "sm", "bias", "sig", "cons", "sigad", "dmag")
RAW_FEATS = ("bright", "sharp", "jvar")
FEATS = ["step"] + list(DELTA_FEATS) + list(RAW_FEATS)


def parse(paths):
    """→ {session: [(feat_dict, true)]}（按 step 升序）"""
    out = defaultdict(list)
    for p in paths:
        txt = open(p, encoding="utf-8", errors="ignore").read()
        for blk in re.split(r"\[\d+/\d+\] ", txt)[1:]:
            sess = blk.split("\n")[0].split("  ")[0].strip()
            if not sess:
                continue
            for m in PAT.finditer(blk):
                g = m.groups()
                f = {"step": int(g[0]),
                     "val": float(g[1]) if g[1] not in ("nan", "NaN") else float("nan"),
                     "sm": float(g[3]), "cons": float(g[4]) if g[4] else float("nan"),
                     "bias": float(g[5]),
                     "sig": math.hypot(float(g[6]), float(g[7])),
                     "sigad": float(g[8]) if g[8] else float("nan"),
                     "dmag": float(g[9]) if g[9] else float("nan"),
                     "bright": float(g[10]) if len(g) > 10 and g[10] else float("nan"),
                     "sharp": float(g[11]) if len(g) > 11 and g[11] else float("nan"),
                     "jvar": float(g[12]) if len(g) > 12 and g[12] else float("nan")}
                out[sess].append((f, float(g[2])))
    for s in out:
        out[s].sort(key=lambda x: x[0]["step"])
    return dict(out)


def subject_of(sess):
    for part in sess.split("/"):
        if part.startswith("p") and part[1:].isdigit():
            return part
    return "?"


def design(samples, base):
    """特征向量：相对步 0 的差值 + log 步数（跨会话可比 ✓）"""
    X, y = [], []
    for f, t in samples:
        row = [math.log1p(f["step"])]
        for k in DELTA_FEATS:
            v = f[k]
            b = base.get(k, float("nan"))
            row.append(0.0 if not (v == v) else (v - b if b == b else v))
        for k in RAW_FEATS:                 # 输入质量/不确定度：用原值（绝对值本身有意义 ✓）
            v = f.get(k, float("nan"))
            row.append(0.0 if not (v == v) else v)
        row.append(1.0 if f["step"] == 0 else 0.0)      # 恒等锚点哑变量 ✓
        X.append(row); y.append(t)
    return np.asarray(X, float), np.asarray(y, float)


def ridge_fit(X, y, lam=1.0):
    n, d = X.shape
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(d), Z.T @ y)
    return w, mu, sd


def ridge_pred(model, X):
    w, mu, sd = model
    return ((X - mu) / sd) @ w


def spearman(a, b):
    def rk(v):
        o = np.argsort(v); r = np.empty(len(v)); r[o] = np.arange(len(v)); return r
    ra, rb = rk(a), rk(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    den = math.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


def main():
    ap = argparse.ArgumentParser(description="无标签特征能否预测真误差（学策略的可行性）")
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--dump-policy", default=None,
                    help="把拟合好的岭回归策略（系数+标准化+Δ基准）导出为 json ⇒ 供运行时按策略选步 ✓")
    ap.add_argument("--only-subjects", default=None,
                    help="只保留这些被试（逗号分隔 ✓）。策略**必须只在源域被试上拟合** ✓，"
                         "否则会把目标被试泄漏进策略 ✗")
    ap.add_argument("--relative-target", action="store_true",
                    help="用**相对第 0 步的改善量**（true − true@0）作为回归目标 ✓ ⇒ 抵消源域/目标域"
                         "的绝对误差水平差异（实测：绝对目标会让策略在目标域上永远弃权 ✗）")
    ap.add_argument("--train-subjects", default=None,
                    help="只在这些被试上训练、在所有其余会话上评估（= 真实部署协议 ✓）。"
                         "不给定时退化为留一被试 ✓")
    args = ap.parse_args()
    paths = [p for pat in args.logs for p in glob.glob(pat)] or args.logs
    data = parse(paths)
    data = {s: v for s, v in data.items() if len(v) >= 8}
    if args.only_subjects:
        keep = {x.strip() for x in args.only_subjects.split(",") if x.strip()}
        data = {s: v for s, v in data.items() if subject_of(s) in keep}
        print(f"[filter] 只保留被试 {sorted(keep)} ⇒ {len(data)} 个会话")
    if not data:
        raise SystemExit("没解析到任何会话（≥8 个探针点）✗")
    subs = sorted({subject_of(s) for s in data})
    print(f"解析到 {len(data)} 个会话，被试 {subs}；" +
          "；".join(f"{s} {len(v)} 点" for s, v in list(data.items())[:6]))

    # ---- 相对基准：每会话"恒等"（step 0 或第一个点）的特征值 ----
    base_feat = {}
    for s, v in data.items():
        z = next((f for f, _ in v if f["step"] == 0), v[0][0])
        base_feat[s] = z

    # ---- A. 留一被试：预测真误差 vs 真误差 ----
    print("\n=== A. 代理质量（留一被试 ✓ 训练只用其他被试 ✓）===")
    print(f"{'会话':<34}{'点数':>5}{'ρ(pred,true)':>14}{'  ρ(val,true)':>14}")
    # 相对目标（可选 ✓）：把每会话的真误差减去其第 0 步 ⇒ 域间绝对水平被抵消 ✓
    T0 = {s: v[0][1] for s, v in data.items()}
    def _rel(s, t):
        return (t - T0[s]) if args.relative_target else t
    rows_a = []
    TRAIN_SUB = {x.strip() for x in (args.train_subjects or "").split(",") if x.strip()}
    for s in data:
        s_hold = subject_of(s)
        if TRAIN_SUB:                    # 部署协议：只在源域被试上训练 ✓
            tr_sess = [o for o in data if subject_of(o) in TRAIN_SUB]
            leak = s_hold in TRAIN_SUB
        else:
            tr_sess = [o for o in data if subject_of(o) != s_hold] or [o for o in data if o != s]
            leak = all(subject_of(o) == s_hold for o in tr_sess)
        tr = [(f, _rel(o, t)) for o in tr_sess for f, t in data[o]]
        Xtr, ytr = design(tr, {k: statistics.median([bf[k] for bf in base_feat.values()])
                               for k in ("val", "sm", "bias", "sig", "cons", "sigad", "dmag")})
        mdl = ridge_fit(Xtr, ytr, args.lam)
        Xte, yte = design(data[s], base_feat[s])
        pred = ridge_pred(mdl, Xte)
        vals = np.array([f["val"] for f, _ in data[s]])
        ok = ~np.isnan(vals)
        rows_a.append((spearman(pred, yte), spearman(vals[ok], yte[ok]), len(yte), leak))
        print(f"{s[-32:]:<34}{len(yte):>5}{rows_a[-1][0]:>14.2f}{rows_a[-1][1]:>14.2f}"
              + ("   同被试(乐观界 ✗)" if leak else "   跨被试 ✓"))
    med = lambda i: statistics.median([r[i] for r in rows_a])
    print(f"{'★ 中位':<34}{'':>5}{med(0):>14.2f}{med(1):>14.2f}   "
          f"（>0 且明显大于现状 ⇒ 可学 ✓；≤0 ⇒ 不可学 ✗）"
          + (f"   [注意：本批 {len({subject_of(s) for s in data})} 个被试 ⇒ 部分为同被试乐观界 ✗]" if len({subject_of(s) for s in data}) < 2 else ""))

    # ---- B. 策略成绩：按"预测误差最小的步"选步 ----
    print("\n=== B. 策略成绩（探针小批 L1，越小越好 ✓）===")
    print(f"{'会话':<34}{'恒等':>9}{'现有准则':>10}{'学到的策略':>12}{'oracle':>9}")
    agg = defaultdict(list)
    for s in data:
        s_hold = subject_of(s)
        if TRAIN_SUB:
            tr_sess = [o for o in data if subject_of(o) in TRAIN_SUB]
        else:
            tr_sess = [o for o in data if subject_of(o) != s_hold] or [o for o in data if o != s]
        tr = [(f, _rel(o, t)) for o in tr_sess for f, t in data[o]]
        Xtr, ytr = design(tr, {k: statistics.median([bf[k] for bf in base_feat.values()])
                               for k in ("val", "sm", "bias", "sig", "cons", "sigad", "dmag")})
        mdl = ridge_fit(Xtr, ytr, args.lam)
        pts = data[s]
        Xte, yte = design(pts, base_feat[s])
        pred = ridge_pred(mdl, Xte)
        vals = np.array([f["val"] for f, _ in pts], float)
        i_pol = int(np.nanargmin(pred)); i_val = int(np.nanargmin(vals)); i_orc = int(np.argmin(yte))
        e0 = yte[0]
        agg["base"] += [e0]; agg["val"] += [yte[i_val]]
        agg["pol"] += [yte[i_pol]]; agg["orc"] += [yte[i_orc]]
        print(f"{s[-32:]:<34}{e0:>9.5f}{yte[i_val]:>10.5f}{yte[i_pol]:>12.5f}{yte[i_orc]:>9.5f}"
              f"   （策略选的步 {pts[i_pol][0]['step']}）")
    m = {k: statistics.mean(v) for k, v in agg.items()}
    print(f"{'★ 平均':<34}{m['base']:>9.5f}{m['val']:>10.5f}{m['pol']:>12.5f}{m['orc']:>9.5f}")
    print(f"{'★ 相对恒等':<34}{'—':>9}{100*(m['val']/m['base']-1):>9.1f}%"
          f"{100*(m['pol']/m['base']-1):>11.1f}%{100*(m['orc']/m['base']-1):>8.1f}%")

    # ---- C. 特征重要性 ----
    Xtr, ytr = design([(f, t) for v in data.values() for f, t in v],
                      {k: statistics.median([bf[k] for bf in base_feat.values()])
                       for k in ("val", "sm", "bias", "sig", "cons", "sigad", "dmag")})
    w, mu_c, sd_c = ridge_fit(Xtr, ytr, args.lam)
    names = ["log1p(step)"] + [f"Δ{k}" for k in DELTA_FEATS] + [k for k in RAW_FEATS] + ["is_step0"]
    order = np.argsort(-np.abs(w))
    print("\n=== C. 特征重要性（标准化岭系数，绝对值降序 ✓）===")
    for i in order:
        print(f"   {names[i]:<14}{w[i]:+.4f}")

    if args.dump_policy:
        import json
        base = {k: statistics.median([bf[k] for bf in base_feat.values()])
                for k in DELTA_FEATS}
        pol = {"w": w.tolist(), "mu": mu_c.tolist(), "sd": sd_c.tolist(), "names": names,
               "delta_feats": list(DELTA_FEATS), "raw_feats": list(RAW_FEATS),
               "base": {k: (None if base[k] != base[k] else float(base[k])) for k in base},
               "source_sessions": sorted(data.keys()), "lam": args.lam}
        open(args.dump_policy, "w", encoding="utf-8").write(json.dumps(pol, ensure_ascii=False, indent=1))
        print(f"\n[policy] 已导出策略 → {args.dump_policy}（源域会话 {len(data)} 个 ✓）")


if __name__ == "__main__":
    main()
