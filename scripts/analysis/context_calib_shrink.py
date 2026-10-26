"""C3 的**收缩（shrinkage）**安全化：用 λ 在“预测校正”与“恒等/EMA”之间插值。

动机：5 被试 LOSO 上 C3 均值好于 EMA，但最差会话 +22% ⇒ 用收缩控制风险：
    s_eff = 1 + λ(s_hat - 1),  mu_eff = mu_pred + λ(mu_hat - mu_pred)
λ 由**源域 LOSO**决定（只看源域），目标：最差会话不劣于 EMA 1%，同时尽量保留均值收益。

用法：
    python -m scripts.analysis.context_calib_shrink \
        --csv out/rgbdgaze/analysis/preds_eye224_mgpu_test.csv
"""
from __future__ import annotations

import argparse
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
    apply_params, load_index, load_sessions, session_features,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="out/rgbdgaze/analysis/preds_eye224_mgpu_test.csv")
    ap.add_argument("--out", default="out/rgbdgaze/analysis/context/shrink_sweep.json")
    args = ap.parse_args()

    from sklearn.ensemble import HistGradientBoostingRegressor as HGB

    meta = load_index()
    sess = load_sessions(args.csv, meta)
    F, P = {}, {}
    for k, rows in sess.items():
        F[k], P[k] = session_features(rows, meta)
    keys = sorted(F)
    fk = [k for k in F[keys[0]] if not k.startswith("target")]
    X = np.array([[F[k][x] for x in fk] for k in keys])
    Y = np.array([[np.log(P[k]["s"][0]), np.log(P[k]["s"][1]),
                   P[k]["mu_t"][0], P[k]["mu_t"][1]] for k in keys])
    subs = np.array([k[0] for k in keys])

    def mk():
        return HGB(max_depth=3, max_iter=200, learning_rate=0.05,
                   l2_regularization=1.0, min_samples_leaf=8, random_state=0)

    pred = {}
    for hold in sorted(set(subs)):
        tr, te = subs != hold, subs == hold
        pred[hold] = {"keys": [keys[i] for i in np.nonzero(te)[0]]}
        for j in range(4):
            pred[hold][j] = mk().fit(X[tr], Y[tr, j]).predict(X[te])

    # 预计算 EMA 流（与训练/评估同一 α=0.6）
    def ema_stream(pred):
        acc = pred[0].copy(); out = np.empty_like(pred)
        for i in range(len(pred)):
            acc = 0.6 * pred[i] + 0.4 * acc; out[i] = acc
        return out

    rows = []
    print(f"{'lambda':>6} {'被试等权':>8} {'帧加权':>8} {'最差会话':>10}")
    for lam in (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0):
        eq, num, den = [], 0.0, 0.0
        worst, wname = -9.0, ""
        for hold in sorted(set(subs)):
            for j, k in enumerate(pred[hold]["keys"]):
                s_hat = np.exp(np.array([pred[hold][0][j], pred[hold][1][j]]))
                mu_hat = np.array([pred[hold][2][j], pred[hold][3][j]])
                y_c3 = (np.clip(s_hat, 0.5, 2.0) * (P[k]["pred"] - P[k]["mu_p"])
                        + np.clip(mu_hat, 0.0, 1.0))
                y_ema = ema_stream(P[k]["pred"])
                y = (1.0 - lam) * y_ema + lam * y_c3        # λ=0 ⇒ 恰为 EMA ✓
                em = np.hypot((y[:, 0] - P[k]["lab"][:, 0]) * P[k]["cm"][:, 0],
                              (y[:, 1] - P[k]["lab"][:, 1]) * P[k]["cm"][:, 1]).mean()
                rel = em / P[k]["em_ema"] - 1.0
                eq.append(rel); num += em * P[k]["n"]; den += P[k]["em_ema"] * P[k]["n"]
                if rel > worst:
                    worst, wname = rel, f"{k[0]}/{k[1]}"
        eqm = float(np.mean(eq)); fwm = num / den - 1.0
        rows.append(dict(lam=lam, eq=eqm, fw=fwm, worst=worst, worst_sess=wname))
        print(f"{lam:6.2f} {100*eqm:+7.1f}% {100*fwm:+7.1f}% {100*worst:+9.1f}% ({wname})")

    # 选 λ：最差不劣于 EMA 1%，且在满足条件下均值最好
    ok = [r for r in rows if r["worst"] <= 0.01]
    best = min(ok, key=lambda r: r["fw"]) if ok else None
    print(f"\n[选择] 满足“最差 ≤ +1%”的最大收缩 λ = {best['lam'] if best else None}"
          f"（fw {100*best['fw']:+.1f}%）" if best else "\n[选择] 无可行 λ ⇒ 回退 EMA")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"sweep": rows,
                                          "chosen": best}, ensure_ascii=False,
                                         indent=1), encoding="utf-8")
    print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
