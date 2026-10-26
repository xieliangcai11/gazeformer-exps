"""阶段 0：无标签校准的**可学性**分析（不训练、不占 GPU）。

回答两个问题：
  1) 每个会话的“最优校正参数”到底由什么决定？（被试 / 活动·设备 / 会话内残余）
  2) 只用**上下文先验**（同一活动下其他被试的中位参数），能拿回多少收益？
     与基线、EMA、per-session oracle 对比。

输入：逐帧预测 CSV（含 pred_x/pred_y/label_x/label_y）+ RGBDGaze index（屏幕 cm）。
输出：会话特征/参数表、方差分解、留一被试评估、结论。

用法：
    python -m scripts.analysis.context_calib_stage0 \
        --src out/rgbdgaze/analysis/preds_src.csv \
        --test out/rgbdgaze/analysis/preds_eye224_mgpu_test.csv \
        --out-dir out/rgbdgaze/analysis/context
"""
from __future__ import annotations

import argparse
import csv
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


def load_cm():
    """path -> (screen_w_cm, screen_h_cm)。"""
    out = {}
    with open(RGBDGaze_INDEX_DIR / "index.csv", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            try:
                out[r["rgb_path"]] = (float(r["screen_w"]) * float(r["cm_px_x"]),
                                      float(r["screen_h"]) * float(r["cm_px_y"]))
            except (KeyError, ValueError):
                continue
    return out


def load_sessions(paths):
    """按 (subject, activity) 聚合逐帧预测（按 frame_index 排序）。"""
    sess = defaultdict(list)
    for p in paths:
        with open(p, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                sess[(r["subject"], r["activity"])].append(r)
    for k in sess:
        sess[k].sort(key=lambda r: int(float(r["frame_index"])))
    return sess


def oracle_params(rows, cmap):
    """会话 oracle 参数：目标矩 (mu_t, sigma_t)、尺度 s=sigma_l/sigma_p、偏置 b。"""
    pred = np.array([[float(r["pred_x"]), float(r["pred_y"])] for r in rows])
    lab = np.array([[float(r["label_x"]), float(r["label_y"])] for r in rows])
    cm = np.array([cmap.get(r["path"], (7.81, 16.08)) for r in rows])   # 兜底仅用于缺项
    mu_p, mu_t = pred.mean(0), lab.mean(0)
    sd_p, sd_t = pred.std(0), lab.std(0)
    s = sd_t / np.maximum(sd_p, 1e-6)
    b = mu_t - s * mu_p
    y = s * (pred - mu_p) + mu_t
    em_oracle = np.hypot((y[:, 0] - lab[:, 0]) * cm[:, 0],
                         (y[:, 1] - lab[:, 1]) * cm[:, 1]).mean()
    em_base = np.hypot((pred[:, 0] - lab[:, 0]) * cm[:, 0],
                       (pred[:, 1] - lab[:, 1]) * cm[:, 1]).mean()
    # EMA(0.6) 因果
    alpha = 0.6
    ema = np.empty_like(pred); acc = pred[0].copy()
    for i in range(len(pred)):
        acc = alpha * pred[i] + (1 - alpha) * acc
        ema[i] = acc
    em_ema = np.hypot((ema[:, 0] - lab[:, 0]) * cm[:, 0],
                      (ema[:, 1] - lab[:, 1]) * cm[:, 1]).mean()
    feat = {
        "mu_px": mu_p[0], "mu_py": mu_p[1], "sd_px": sd_p[0], "sd_py": sd_p[1],
        "mu_tx": mu_t[0], "mu_ty": mu_t[1], "sd_tx": sd_t[0], "sd_ty": sd_t[1],
        "s_x": s[0], "s_y": s[1], "b_x": b[0], "b_y": b[1],
        "em_base": em_base, "em_ema": em_ema, "em_oracle": em_oracle, "n": len(rows),
    }
    return feat, pred, lab, cm


def apply_params(pred, lab, cm, mu_p, mu_t, s):
    y = s * (pred - mu_p) + mu_t
    return np.hypot((y[:, 0] - lab[:, 0]) * cm[:, 0],
                    (y[:, 1] - lab[:, 1]) * cm[:, 1]).mean()


def anova_two_way(records, key):
    """平衡设计（被试 × 活动）的方差分解：ss_subject / ss_activity / ss_resid。"""
    by_sub = defaultdict(list)
    by_act = defaultdict(list)
    vals = []
    for (sub, act), v in records.items():
        by_sub[sub].append(v)
        by_act[act].append(v)
        vals.append(v)
    vals = np.asarray(vals)
    gm = vals.mean()
    ss_total = ((vals - gm) ** 2).sum()
    ss_sub = sum(len(v) * (np.mean(v) - gm) ** 2 for v in by_sub.values())
    ss_act = sum(len(v) * (np.mean(v) - gm) ** 2 for v in by_act.values())
    ss_res = max(ss_total - ss_sub - ss_act, 0.0)
    return ss_sub / ss_total, ss_act / ss_total, ss_res / ss_total


def main():
    ap = argparse.ArgumentParser(description="阶段0：无标签校准可学性分析")
    ap.add_argument("--src", default=None, help="源域逐帧预测 CSV（可多个，逗号分隔）")
    ap.add_argument("--test", default="out/rgbdgaze/analysis/preds_eye224_mgpu_test.csv")
    ap.add_argument("--out-dir", default="out/rgbdgaze/analysis/context")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    cmap = load_cm()

    src_paths = [p for p in (args.src or "").split(",") if p.strip()]
    frames = {}
    if src_paths:
        frames.update({("src",) + k: v for k, v in load_sessions(src_paths).items()})
    frames.update({("test",) + k: v for k, v in load_sessions([args.test]).items()})

    feats, raws = {}, {}
    for key, rows in frames.items():
        f, pred, lab, cm = oracle_params(rows, cmap)
        feats[key] = f; raws[key] = (pred, lab, cm)

    # 写出会话表
    with open(out / "session_features.csv", "w", newline="", encoding="utf-8") as fh:
        cols = ["split", "subject", "activity"] + list(next(iter(feats.values())).keys())
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for (split, sub, act), f in sorted(feats.items()):
            w.writerow({"split": split, "subject": sub, "activity": act, **f})

    test = {(s, a): f for (sp, s, a), f in feats.items() if sp == "test"}
    print(f"[stage0] 会话数：test={len(test)}  src={sum(1 for k in feats if k[0]=='src')}")

    # ---- 1) 方差分解 ----
    print("\n=== 方差分解（test 20 会话，两因素：被试 / 活动）===")
    print(f"{'参数':>8} {'被试':>8} {'活动':>8} {'残余':>8}")
    for k in ("mu_tx", "mu_ty", "sd_tx", "sd_ty", "s_x", "s_y", "b_x", "b_y"):
        rs, ra, rr = anova_two_way({(s, a): f[k] for (s, a), f in test.items()}, k)
        print(f"{k:>8} {100*rs:7.1f}% {100*ra:7.1f}% {100*rr:7.1f}%")

    # ---- 2) 留一被试 + 活动先验 ----
    subjects = sorted({s for s, _ in test})
    acts = sorted({a for _, a in test})
    print("\n=== 留一被试：用“同活动、其他被试”的中位参数 ===")
    print(f"{'被试':>6} {'基线':>6} {'EMA':>6} {'活动先验':>8} {'oracle':>7} {'Δ vs EMA':>9}")
    tot = {"b": [], "e": [], "p": [], "o": []}
    for hold in subjects:
        bsum = esum = psum = osum = 0.0
        nsum = 0
        for a in acts:
            key = (hold, a)
            if key not in test:
                continue
            f = test[key]; pred, lab, cm = raws[("test",) + key]
            peers = [ff for (s2, a2), ff in test.items()
                     if s2 != hold and a2 == a]
            if not peers:
                continue
            mu_p = np.array([f["mu_px"], f["mu_py"]])
            mu_t = np.median([[p["mu_tx"], p["mu_ty"]] for p in peers], axis=0)
            s = np.median([[p["s_x"], p["s_y"]] for p in peers], axis=0)
            em_p = apply_params(pred, lab, cm, mu_p, mu_t, s)
            bsum += f["em_base"] * f["n"]; esum += f["em_ema"] * f["n"]
            psum += em_p * f["n"]; osum += f["em_oracle"] * f["n"]; nsum += f["n"]
        b, e, p, o = bsum / nsum, esum / nsum, psum / nsum, osum / nsum
        tot["b"].append(b); tot["e"].append(e); tot["p"].append(p); tot["o"].append(o)
        print(f"{hold:>6} {b:6.3f} {e:6.3f} {p:8.3f} {o:7.3f} {100*(p/e-1):+8.1f}%")

    print("\n=== 汇总（被试等权）===")
    b, e, p, o = (np.mean(tot[k]) for k in "bepo")
    print(f"基线 {b:.3f} | EMA {e:.3f} | 活动先验 {p:.3f} ({100*(p/e-1):+.1f}% vs EMA) "
          f"| oracle {o:.3f}")
    verdict = "有信号 ✓" if p < e else "无信号 ✗（活动先验不如 EMA）"
    print(f"[判定] {verdict}")


if __name__ == "__main__":
    main()
