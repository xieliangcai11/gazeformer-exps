"""C3 的**规范协议**：源域（out-of-subject 预测）训练/选参，测试集只评一次。

数据：
  · 源域：`cv_fold*.csv`（5 折分组 CV 产出的留出被试预测，即每个被试的预测都
    来自没见过他的模型）——由 `run_group_cv.py` 生成；
  · 目标：`preds_eye224_mgpu_test.csv`（部署模型的测试集预测）。

协议（无测试泄漏）：
  1. 源域 **LOSO**（36 被试）：训练 C3（特征 → 目标矩/尺度），在留出被试上评估；
  2. 用源域 LOSO 选择“与 EMA 的输出混合系数 λ”（判据：最差会话 ≤ +1%，再取均值最好）；
  3. 用**全部源域**重训 C3，对测试集只评一次（λ 已固定）；
  4. 同时报告“常数校正”基线（不用特征），用于判断特征是否真的带来信息。

用法：
    python -m scripts.analysis.c3_source_protocol \
        --cv-glob "out/rgbdgaze/analysis/context/cv_fold*.csv" \
        --test out/rgbdgaze/analysis/preds_eye224_mgpu_test.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np


def _bootstrap() -> Path:
    here = Path(__file__).resolve()
    for d in (here.parent, *here.parents):
        if (d / "src" / "gazelab" / "__init__.py").is_file():
            for p in (str(d), str(d / "src")):
                if p not in sys.path:
                    sys.path.insert(0, p)
            return d
    raise RuntimeError("找不到项目根")


_ROOT = _bootstrap()
from scripts.analysis.context_calib_l2 import (  # noqa: E402
    load_index, load_sessions, session_features,
)


def ema_stream(pred, alpha=0.6):
    acc = pred[0].copy(); out = np.empty_like(pred)
    for i in range(len(pred)):
        acc = alpha * pred[i] + (1 - alpha) * acc
        out[i] = acc
    return out


def build(csv_paths, meta):
    sess = {}
    for p in csv_paths:
        for k, rows in load_sessions(p, meta).items():
            if k in sess:                      # 同一会话不应跨折重复
                raise SystemExit(f"重复会话 {k}（检查 CV 折划分）")
            sess[k] = rows
    F, P = {}, {}
    for k, rows in sess.items():
        F[k], P[k] = session_features(rows, meta)
    return F, P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv-glob", default="out/rgbdgaze/analysis/context/cv_fold*.csv")
    ap.add_argument("--test", default="out/rgbdgaze/analysis/preds_eye224_mgpu_test.csv")
    ap.add_argument("--out-dir", default="out/rgbdgaze/analysis/context")
    ap.add_argument("--lams", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0])
    args = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingRegressor as HGB

    def mk():
        return HGB(max_depth=3, max_iter=200, learning_rate=0.05,
                   l2_regularization=1.0, min_samples_leaf=8, random_state=0)

    meta = load_index()
    cv_paths = sorted(glob.glob(args.cv_glob))
    if not cv_paths:
        raise SystemExit(f"没有找到 CV 预测：{args.cv_glob}（先跑 run_group_cv）")
    Fs, Ps = build(cv_paths, meta)
    Ft, Pt = build([args.test], meta)
    print(f"[protocol] 源域 {len(Fs)} 会话 / {len({s for s,_ in Fs})} 被试 | "
          f"目标 {len(Ft)} 会话")

    keys = sorted(Fs)
    fk = [k for k in Fs[keys[0]] if not k.startswith("target")]
    X = np.array([[Fs[k][x] for x in fk] for k in keys])
    Y = np.array([[np.log(Ps[k]["s"][0]), np.log(Ps[k]["s"][1]),
                   Ps[k]["mu_t"][0], Ps[k]["mu_t"][1]] for k in keys])
    subs = np.array([k[0] for k in keys])

    def params_from(model_x, idx):
        pred = [mk().fit(model_x[0], model_x[1][:, j]).predict(idx) for j in range(4)]
        s = np.exp(np.clip(np.stack([pred[0], pred[1]], 1), np.log(0.5), np.log(2.0)))
        mt = np.clip(np.stack([pred[2], pred[3]], 1), 0.0, 1.0)
        return s, mt

    # ---- 源域 LOSO ----
    lam_src = {lam: [] for lam in args.lams}
    const_src = []
    for hold in sorted(set(subs)):
        tr, te = subs != hold, subs == hold
        te_keys = [keys[i] for i in np.nonzero(te)[0]]
        s_hat, mt_hat = params_from((X[tr], Y[tr]), X[te])
        # 常数基线（训练折参数均值）
        c = Y[tr].mean(0)
        s_c = np.exp(np.clip(c[0:2], np.log(0.5), np.log(2.0)))
        mt_c = np.clip(c[2:4], 0.0, 1.0)
        for j, k in enumerate(te_keys):
            P = Ps[k]
            y_ema = ema_stream(P["pred"])
            y_c3 = s_hat[j] * (P["pred"] - P["mu_p"]) + mt_hat[j]
            y_c = s_c * (P["pred"] - P["mu_p"]) + mt_c
            def em(y):
                return np.hypot((y[:, 0] - P["lab"][:, 0]) * P["cm"][:, 0],
                                (y[:, 1] - P["lab"][:, 1]) * P["cm"][:, 1]).mean()
            e_ema = em(y_ema)
            for lam in args.lams:
                y = (1 - lam) * y_ema + lam * y_c3
                lam_src[lam].append((em(y) / e_ema - 1.0, P["n"], k))
            const_src.append((em(y_c) / e_ema - 1.0, P["n"], k))

    def summarize(rows):
        eq = float(np.mean([r[0] for r in rows]))
        fw = float(np.average([r[0] for r in rows], weights=[r[1] for r in rows]))
        worst = max(rows, key=lambda r: r[0])
        return eq, fw, worst[0], worst[2]

    print(f"\n{'源域 LOSO':>10} {'被试等权':>9} {'帧加权':>8} {'最差会话':>10}")
    tbl = {}
    for lam in args.lams:
        eq, fw, w, k = summarize(lam_src[lam])
        tbl[lam] = dict(eq=eq, fw=fw, worst=w, worst_sess=f"{k[0]}/{k[1]}")
        print(f"{'λ='+format(lam,'.2f'):>10} {100*eq:+8.1f}% {100*fw:+7.1f}% {100*w:+9.1f}% ({k[0]}/{k[1]})")
    eq, fw, w, k = summarize(const_src)
    print(f"{'常数校正':>10} {100*eq:+8.1f}% {100*fw:+7.1f}% {100*w:+9.1f}% ({k[0]}/{k[1]})")

    # ---- 选 λ（源域判据）----
    ok = [x for x in args.lams if tbl[x]["worst"] <= 0.01]
    lam_star = min(ok, key=lambda x: tbl[x]["fw"]) if ok else 0.0
    print(f"\n[select] λ* = {lam_star}（源域：最差 ≤ +1% 中帧加权最好）")
    if not ok:
        print("[select] 无可行 λ ⇒ 回退 EMA（λ=0）")

    # ---- 源域全量训练 → 测试集一次评估 ----
    s_hat, mt_hat = params_from((X, Y), np.array([[Ft[k][x] for x in fk] for k in sorted(Ft)]))
    test_rows = []
    for j, k in enumerate(sorted(Ft)):
        P = Pt[k]
        y_ema = ema_stream(P["pred"])
        y_c3 = s_hat[j] * (P["pred"] - P["mu_p"]) + mt_hat[j]
        y = (1 - lam_star) * y_ema + lam_star * y_c3
        def em(yy):
            return np.hypot((yy[:, 0] - P["lab"][:, 0]) * P["cm"][:, 0],
                            (yy[:, 1] - P["lab"][:, 1]) * P["cm"][:, 1]).mean()
        test_rows.append(dict(subject=k[0], activity=k[1], n=P["n"],
                              em_base=P["em_base"], em_ema=P["em_ema"],
                              em_c3=em(y),          # 融合后预测的真实 EM（不是线性插值）
                              em_oracle=P["em_oracle"]))
    def agg(key):
        eq = float(np.mean([r[key] / r["em_ema"] - 1 for r in test_rows]))
        fw = float(np.average([r[key] / r["em_ema"] - 1 for r in test_rows],
                              weights=[r["n"] for r in test_rows]))
        worst = max(test_rows, key=lambda r: r[key] / r["em_ema"] - 1)
        return eq, fw, worst[key] / worst["em_ema"] - 1, f"{worst['subject']}/{worst['activity']}"

    print(f"\n{'测试集（一次）':>12} {'被试等权':>9} {'帧加权':>8} {'最差会话':>10}")
    for name, key in (("基线", "em_base"), ("EMA", "em_ema"),
                      (f"C3(λ={lam_star})", "em_c3"), ("oracle", "em_oracle")):
        eq, fw, w, k = agg(key)
        print(f"{name:>12} {100*eq:+8.1f}% {100*fw:+7.1f}% {100*w:+9.1f}% ({k})")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "c3_protocol.json").write_text(json.dumps(
        {"lam_star": lam_star, "source_loso": tbl,
         "test": [{k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                   for k, v in r.items()} for r in test_rows]},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[save] {out/'c3_protocol.json'}")


if __name__ == "__main__":
    main()
