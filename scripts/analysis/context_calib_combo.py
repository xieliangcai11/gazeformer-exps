"""C3 的错误排查 + 与现有 TTA 的组合（纯 CPU，用已有逐帧 dump）。

两个诊断：
  A) 常数校正基线：LOSO 时用训练折的**参数均值**（不用特征）⇒ 若 C3 ≈ 常数，
     说明特征没有带来信息（不是 bug，是真没信号）。
  B) C3 + TTA 输出级融合：逐帧混合 y = α·y_tta + (1−α)·y_c3，扫 α，
     并报告均值/最差会话；同时给出 y_ema 作参照。

用法：
    python -m scripts.analysis.context_calib_combo \
        --test out/rgbdgaze/analysis/preds_eye224_mgpu_test.csv \
        --tta  out/rgbdgaze/analysis/ttadump_budget_s100.csv
"""
from __future__ import annotations

import argparse
import csv
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


def load_tta(path):
    """path -> (tta_x, tta_y)。"""
    tta = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            try:
                tta[r["path"]] = (float(r["tta_px"]), float(r["tta_py"]))
            except (KeyError, ValueError):
                continue
    return tta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="out/rgbdgaze/analysis/preds_eye224_mgpu_test.csv")
    ap.add_argument("--tta", default="out/rgbdgaze/analysis/ttadump_budget_s100.csv")
    ap.add_argument("--out", default="out/rgbdgaze/analysis/context/combo.json")
    args = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingRegressor as HGB

    meta = load_index()
    sess = load_sessions(args.test, meta)
    F, P = {}, {}
    for k, rows in sess.items():
        F[k], P[k] = session_features(rows, meta)
    keys = sorted(F)
    fk = [k for k in F[keys[0]] if not k.startswith("target")]
    X = np.array([[F[k][x] for x in fk] for k in keys])
    Y = np.array([[np.log(P[k]["s"][0]), np.log(P[k]["s"][1]),
                   P[k]["mu_t"][0], P[k]["mu_t"][1]] for k in keys])
    subs = np.array([k[0] for k in keys])
    tta = load_tta(args.tta)
    print(f"[combo] 会话 {len(keys)} | TTA 帧 {len(tta)}")

    def mk():
        return HGB(max_depth=3, max_iter=200, learning_rate=0.05,
                   l2_regularization=1.0, min_samples_leaf=8, random_state=0)

    # 逐会话保存：C3 / 常数 / EMA 的预测
    out = {}
    for hold in sorted(set(subs)):
        tr, te = subs != hold, subs == hold
        ks = [keys[i] for i in np.nonzero(te)[0]]
        pred = {}
        for j in range(4):
            pred[j] = mk().fit(X[tr], Y[tr, j]).predict(X[te])
        const = Y[tr].mean(0)      # 常数校正基线（不用特征）
        for j, k in enumerate(ks):
            s_hat = np.exp(np.clip(np.array([pred[0][j], pred[1][j]]), np.log(0.5), np.log(2.0)))
            mt = np.clip(np.array([pred[2][j], pred[3][j]]), 0, 1)
            y_c3 = s_hat * (P[k]["pred"] - P[k]["mu_p"]) + mt
            y_const = np.exp(np.clip(const[0:2], np.log(0.5), np.log(2.0))) * (P[k]["pred"] - P[k]["mu_p"]) + np.clip(const[2:4], 0, 1)
            out[k] = dict(y_c3=y_c3, y_const=y_const, y_ema=ema_stream(P[k]["pred"]),
                          lab=P[k]["lab"], cm=P[k]["cm"], n=P[k]["n"],
                          em_ema=P[k]["em_ema"], em_base=P[k]["em_base"])

    def score(y_of):
        eq, num, den, worst, wn = [], 0.0, 0.0, -9.0, ""
        for k, d in out.items():
            y = y_of(k, d)
            e = np.hypot((y[:, 0] - d["lab"][:, 0]) * d["cm"][:, 0],
                         (y[:, 1] - d["lab"][:, 1]) * d["cm"][:, 1]).mean()
            rel = e / d["em_ema"] - 1
            eq.append(rel); num += e * d["n"]; den += d["em_ema"] * d["n"]
            if rel > worst:
                worst, wn = rel, f"{k[0]}/{k[1]}"
        return float(np.mean(eq)), num / den - 1, worst, wn

    print(f"\n{'方法':>22} {'被试等权':>9} {'帧加权':>8} {'最差会话':>10}")
    for name, fn in (
        ("基线", lambda k, d: d["lab"] * 0 + 0 + P[k]["pred"]),
        ("EMA", lambda k, d: d["y_ema"]),
        ("常数校正", lambda k, d: d["y_const"]),
        ("C3", lambda k, d: d["y_c3"]),
    ):
        eq, fw, w, wn = score(fn)
        print(f"{name:>22} {100*eq:+8.1f}% {100*fw:+7.1f}% {100*w:+9.1f}% ({wn})")

    # ---- C3 + TTA 输出融合（λ=0.3 的安全收缩先套在 C3 上）----
    def c3_safe(k, d, lam=0.3):
        return (1 - lam) * d["y_ema"] + lam * d["y_c3"]

    def y_tta(k, d):
        rows = sess[k]
        arr = np.array([tta.get(r["path"], (np.nan, np.nan)) for r in rows], dtype=np.float64)
        return arr

    def blend_tta_alpha(alpha, base="c3_safe"):
        def fn(k, d):
            yt = y_tta(k, d)
            base_y = c3_safe(k, d) if base == "c3_safe" else d["y_c3"]
            ok = np.isfinite(yt).all(1)
            y = base_y.copy()
            y[ok] = alpha * yt[ok] + (1 - alpha) * base_y[ok]
            return y
        return fn

    print(f"\n{'C3(λ=0.3)+TTA 融合':>22} {'被试等权':>9} {'帧加权':>8} {'最差会话':>10}")
    sweep = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        eq, fw, w, wn = score(blend_tta_alpha(alpha))
        sweep.append(dict(alpha=alpha, eq=eq, fw=fw, worst=w))
        print(f"{'α='+format(alpha,'.2f'):>22} {100*eq:+8.1f}% {100*fw:+7.1f}% {100*w:+9.1f}% ({wn})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"sweep": sweep}, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print(f"\n[save] {args.out}")


if __name__ == "__main__":
    main()
