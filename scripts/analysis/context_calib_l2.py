"""L2：源域学习的“会话条件”校准（离线评估，不训练大网络）。

流程：
  1. 读逐帧预测（源域 train / 目标 test），按 (subject, activity) 聚合；
  2. 只用**预测序列**（无标签）+ 上下文（屏幕物理尺寸/设备/活动）构造会话特征；
  3. 目标：会话 oracle 参数 (log s_x, log s_y, mu_tx, mu_ty) 与“相对 EMA 的增益”；
  4. 源域 **留一被试（LOSO）**：梯度提升/岭回归；评估校正后 EM；
  5. 门控：用源域学到的“增益预测器”，只在预测增益 < -2% 时采用，否则回退 EMA；
  6. 打印逐被试/汇总（被试等权 + 帧加权），并给出 go/no-go。

用法：
    python -m scripts.analysis.context_calib_l2 \
        --src out/rgbdgaze/analysis/context/src_train_preds.csv \
        --test out/rgbdgaze/analysis/preds_eye224_mgpu_test.csv \
        --out-dir out/rgbdgaze/analysis/context
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
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
from configs.rgbdgaze_config import RGBDGaze_INDEX_DIR  # noqa: E402


def load_index():
    """path -> (subject, activity, device, screen_w_cm, screen_h_cm, w_pt, h_pt)."""
    out = {}
    with open(RGBDGaze_INDEX_DIR / "index.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            try:
                out[r["rgb_path"]] = (
                    r["subject"], r["activity"], r["device"],
                    float(r["screen_w"]) * float(r["cm_px_x"]),
                    float(r["screen_h"]) * float(r["cm_px_y"]),
                    float(r["screen_w"]), float(r["screen_h"]))
            except (KeyError, ValueError):
                continue
    return out


def load_sessions(path, meta):
    sess = defaultdict(list)
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            s, a = r["subject"], r["activity"]
            sess[(s, a)].append(r)
    for k in sess:
        sess[k].sort(key=lambda r: int(float(r["frame_index"])))
    return sess


def _ac(x, lag):
    if len(x) <= lag + 2:
        return 0.0
    a, b = x[:-lag], x[lag:]
    return float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-9 and np.std(b) > 1e-9 else 0.0


def session_features(rows, index_meta):
    pred = np.array([[float(r["pred_x"]), float(r["pred_y"])] for r in rows])
    lab = np.array([[float(r["label_x"]), float(r["label_y"])] for r in rows])
    p0 = rows[0]["path"]
    m = index_meta.get(p0)
    sw_cm, sh_cm, w_pt, h_pt = (m[3], m[4], m[5], m[6]) if m else (7.81, 16.08, 0, 0)
    cm = np.array([(sw_cm, sh_cm)] * len(rows))

    f = {}
    for ax, nm in ((0, "x"), (1, "y")):
        v = pred[:, ax]
        f[f"mu_{nm}"] = float(v.mean())
        f[f"sd_{nm}"] = float(v.std())
        f[f"skew_{nm}"] = float(((v - v.mean()) ** 3).mean() / (v.std() ** 3 + 1e-9))
        for q in (5, 10, 25, 50, 75, 90, 95):
            f[f"q{q}_{nm}"] = float(np.percentile(v, q))
    # 2D 占据
    H, _, _ = np.histogram2d(pred[:, 0], pred[:, 1], bins=5, range=[[0, 1], [0, 1]])
    p = H.ravel() / H.sum()
    f["occ_entropy"] = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    f["occ_nonempty"] = float((p > 0).sum())
    rad = np.linalg.norm(pred - 0.5, axis=1)
    f["rad_mean"] = float(rad.mean()); f["rad_p90"] = float(np.percentile(rad, 90))
    f["edge_frac"] = float(((pred < 0.05) | (pred > 0.95)).any(1).mean())
    # 时序
    d = np.abs(np.diff(pred, axis=0))
    f["diff_mean"] = float(d.mean()) if len(d) else 0.0
    for lag in (1, 2, 5, 10, 20, 50):
        f[f"ac{lag}_x"] = _ac(pred[:, 0], lag)
        f[f"ac{lag}_y"] = _ac(pred[:, 1], lag)
    plateau = (d.max(1) < 0.002) if len(d) else np.zeros(0, bool)
    f["plateau_frac"] = float(plateau.mean()) if len(plateau) else 0.0
    # 上下文
    f["sw_cm"], f["sh_cm"] = sw_cm, sh_cm
    f["aspect"] = min(sw_cm, sh_cm) / max(sw_cm, sh_cm)
    f["log_w"], f["log_h"] = np.log(sw_cm), np.log(sh_cm)
    f["n"] = len(rows)

    # 目标（源域才用）
    mu_t, sd_t = lab.mean(0), lab.std(0)
    mu_p, sd_p = pred.mean(0), pred.std(0)
    s = sd_t / np.maximum(sd_p, 1e-6)
    y = s * (pred - mu_p) + mu_t
    em_oracle = np.hypot((y[:, 0] - lab[:, 0]) * cm[:, 0],
                         (y[:, 1] - lab[:, 1]) * cm[:, 1]).mean()
    em_base = np.hypot((pred[:, 0] - lab[:, 0]) * cm[:, 0],
                       (pred[:, 1] - lab[:, 1]) * cm[:, 1]).mean()
    pred_ema = np.empty_like(pred); acc = pred[0].copy()
    for i in range(len(pred)):
        acc = 0.6 * pred[i] + 0.4 * acc; pred_ema[i] = acc
    em_ema = np.hypot((pred_ema[:, 0] - lab[:, 0]) * cm[:, 0],
                      (pred_ema[:, 1] - lab[:, 1]) * cm[:, 1]).mean()
    return f, dict(mu_p=mu_p, mu_t=mu_t, s=s, em_base=em_base, em_ema=em_ema,
                   em_oracle=em_oracle, pred=pred, lab=lab, cm=cm, n=len(rows))


def apply_params(pred, lab, cm, mu_p, mu_t, s):
    y = s * (pred - mu_p) + mu_t
    return np.hypot((y[:, 0] - lab[:, 0]) * cm[:, 0],
                    (y[:, 1] - lab[:, 1]) * cm[:, 1]).mean()


def main():
    ap = argparse.ArgumentParser(description="L2 源域学习校准（LOSO）")
    ap.add_argument("--src", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out-dir", default="out/rgbdgaze/analysis/context")
    ap.add_argument("--model", choices=["hgb", "ridge"], default="hgb")
    ap.add_argument("--safety", type=float, default=-0.02,
                    help="门控阈值：预测的相对增益低于该值才采用（默认 -2%%）")
    args = ap.parse_args()

    try:
        from sklearn.ensemble import HistGradientBoostingRegressor as HGB
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"需要 scikit-learn: {e}")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    meta = load_index()
    src = load_sessions(args.src, meta)
    tst = load_sessions(args.test, meta)
    print(f"[L2] 会话数：源 {len(src)}（{len({s for s,_ in src})} 被试）| 目标 {len(tst)}")

    def build(sess):
        F, P = {}, {}
        for k, rows in sess.items():
            F[k], P[k] = session_features(rows, meta)
        return F, P

    Fs, Ps = build(src)
    Ft, Pt = build(tst)
    keys = sorted(Fs)
    fkeys = [k for k in Fs[keys[0]] if not k.startswith("target")]
    X = np.array([[Fs[k][fk] for fk in fkeys] for k in keys], dtype=np.float64)
    Y = np.array([[np.log(Ps[k]["s"][0]), np.log(Ps[k]["s"][1]),
                   Ps[k]["mu_t"][0], Ps[k]["mu_t"][1]] for k in keys])
    subs = np.array([k[0] for k in keys])
    def make_model():
        if args.model == "ridge":
            return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        return HGB(max_depth=3, max_iter=200, learning_rate=0.05, l2_regularization=1.0,
                   min_samples_leaf=8, random_state=0)

    # ---- LOSO：参数模型 + 增益模型 ----
    rows_out = []
    gated_em, ema_em, base_em, oracle_em = [], [], [], []
    for hold in sorted(set(subs)):
        tr = subs != hold; te = subs == hold
        preds = {}
        for j in range(4):
            preds[j] = make_model().fit(X[tr], Y[tr, j]).predict(X[te])
        # 增益模型（目标：c3/ema-1，用 LOSO 预测）
        gain_tr = []
        _tr_keys = [keys[i] for i in np.nonzero(tr)[0]]
        for k in _tr_keys:
            # 用 oracle 参数（源域可得）估计“若完美校正”的增益
            em_c3 = Ps[k]["em_oracle"]
            gain_tr.append(em_c3 / Ps[k]["em_ema"] - 1.0)
        gain_tr = np.array(gain_tr)
        gmodel = make_model().fit(X[tr], gain_tr)
        gpred = gmodel.predict(X[te])
        _te_keys = [keys[i] for i in np.nonzero(te)[0]]
        for j, k in enumerate(_te_keys):
            s_hat = np.clip(np.exp(np.array([preds[0][j], preds[1][j]])), 0.5, 2.0)
            mu_t_hat = np.clip(np.array([preds[2][j], preds[3][j]]), 0.0, 1.0)
            P = Ps[k]
            em_c3 = apply_params(P["pred"], P["lab"], P["cm"], P["mu_p"], mu_t_hat, s_hat)
            g = float(gpred[j])
            em_gate = em_c3 if g < args.safety else P["em_ema"]
            rows_out.append(dict(subject=k[0], activity=k[1], base=P["em_base"],
                                 ema=P["em_ema"], c3=em_c3, gate=em_gate,
                                 oracle=P["em_oracle"], gain_pred=g, n=P["n"]))
            gated_em.append(em_gate); ema_em.append(P["em_ema"])
            base_em.append(P["em_base"]); oracle_em.append(P["em_oracle"])

    with open(out / "c3_loso.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0]))
        w.writeheader(); w.writerows(rows_out)

    # ---- 汇总（被试等权 / 帧加权）----
    def agg(key):
        subj_vals = defaultdict(list)
        for r in rows_out:
            subj_vals[r["subject"]].append(r[key])
        eq = np.mean([np.mean(v) for v in subj_vals.values()])
        fw = np.average([r[key] for r in rows_out], weights=[r["n"] for r in rows_out])
        return eq, fw

    print(f"\n{'方法':>12} {'被试等权':>9} {'帧加权':>8}")
    for name, key in (("基线", "base"), ("EMA", "ema"), ("C3(LOSO)", "c3"),
                      ("C3+门控", "gate"), ("oracle", "oracle")):
        eq, fw = agg(key)
        print(f"{name:>12} {eq:9.3f} {fw:8.3f}")

    acc = np.mean([1 if r["gain_pred"] < args.safety else 0 for r in rows_out])
    print(f"\n门控采用率 {100*acc:.0f}%（阈值 {args.safety:+.0%}）")
    worst = max(rows_out, key=lambda r: r["gate"] - r["ema"])
    print(f"最差会话（门控 vs EMA）：{worst['subject']}/{worst['activity']} "
          f"{100*(worst['gate']/worst['ema']-1):+.1f}%")

    ok = agg("gate")[0] < agg("ema")[0] and worst["gate"] <= worst["ema"] * 1.01
    verdict = "go ✓（可进入目标域一次性评估）" if ok else "no-go ✗（收益不足或不安全）"
    print(f"[判定] {verdict}")
    (out / "c3_loso_summary.json").write_text(json.dumps({
        "model": args.model, "safety": args.safety,
        "metrics": {k: agg(k) for k in ("base", "ema", "c3", "gate", "oracle")},
        "accept_rate": float(acc), "worst_gate_vs_ema": worst["gate"] / worst["ema"] - 1,
        "verdict": verdict}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
