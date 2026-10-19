"""复现 / 核验「EMA + 全局分位数映射」主结果（无用户真值协议）。

背景：`out/rgbdgaze/analysis/HANDOFF.md` 与 `docs/方法章节草稿_2026-10-19.md`
记录了 0.8533 cm（−42.0%）这一主结果，但仓库里**既没有实现脚本，也没有逐帧产物**。
本脚本按文档协议重建，并用既有的逐帧基线预测（`ttadump_*.csv` 的 base_px/base_py）
复算，回答"这个数字能不能复现"。

协议（文档 §3.3）：
  s_t = α·y_t + (1-α)·s_{t-1}          # 对**预测**做因果 EMA（不看标签）
  D̂   = 计划目标分布的分位数           # App 设计的目标点分布（非测试标签）
  f   = 单调映射 sort(s) → D̂ 的分位    # 闭式、无训练
  ŷ_t = f(s_t)

本脚本提供三种目标分布，用来区分"可复现"与"靠泄漏"：
  · uniform : 计划网格在 [0,1]² 上均匀 ⇒ 映射 = 对 EMA 流做秩变换（无任何标签）
  · gridK   : K×K 网格格心边缘分布（无标签）
  · labels  : 直接用测试标签的分位数（**泄漏对照**，只用于诊断，不得写进论文）

用法：
    python -m scripts.analysis.reproduce_ema_quantile \
        --dump out/rgbdgaze/analysis/ttadump_20sess_noes.csv \
        --index-csv data/RGBDGaze/index/index.csv
"""
from __future__ import annotations

import argparse
import csv
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


def load_dump(path: Path):
    """返回 {session: dict(frames, base[N,2], label[N,2])}（按 frame_no 排序）。"""
    per = {}
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            s = r["session"]
            d = per.setdefault(s, {"frame": [], "base": [], "label": [], "path": []})
            d["frame"].append(int(float(r["frame_no"])))
            d["base"].append([float(r["base_px"]), float(r["base_py"])])
            d["label"].append([float(r["label_x"]), float(r["label_y"])])
            d["path"].append(r["path"])
    for s, d in per.items():
        order = np.argsort(d["frame"])
        d["base"] = np.asarray(d["base"])[order]
        d["label"] = np.asarray(d["label"])[order]
        d["path"] = [d["path"][i] for i in order]
    return per


def load_cm(index_csv: Path):
    """path -> (screen_w, screen_h, cm_px_x, cm_px_y)，仅用于把归一化误差换成 cm。"""
    need = {}
    with open(index_csv, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                need[r["rgb_path"]] = (float(r["screen_w"]), float(r["screen_h"]),
                                       float(r["cm_px_x"]), float(r["cm_px_y"]))
            except (KeyError, ValueError):
                continue
    return need


def ema(seq: np.ndarray, alpha: float) -> np.ndarray:
    """因果 EMA（逐轴）。首步用第一帧预测初始化。"""
    out = np.empty_like(seq)
    s = seq[0].copy()
    for i, y in enumerate(seq):
        s = alpha * y + (1.0 - alpha) * s
        out[i] = s
    return out


def target_quantiles(kind: str, n: int, labels: np.ndarray | None = None,
                     grid: int = 20) -> np.ndarray:
    """返回 n 个分位点（0..1 等分）对应的目标值（逐轴相同分布）。"""
    q = (np.arange(1, n + 1) - 0.5) / n
    if kind == "uniform":
        return q
    if kind == "grid":
        centers = (np.arange(grid) + 0.5) / grid
        # 经验分位：把网格格心当作等权样本
        return np.quantile(centers, q)
    if kind == "labels":
        if labels is None:
            raise ValueError("labels 目标需要 labels")
        return np.quantile(labels.ravel(), q)
    raise ValueError(kind)


def rank_map(s: np.ndarray, target: np.ndarray) -> np.ndarray:
    """把 s 单调映射到 target 的分位：排序后按分位插值（保留 s 的秩序）。"""
    order = np.argsort(s)
    out = np.empty_like(s)
    out[order] = target
    return out


def score_cm(pred: np.ndarray, label: np.ndarray, cm: np.ndarray) -> np.ndarray:
    sw, sh, cpx, cpy = cm[:, 0], cm[:, 1], cm[:, 2], cm[:, 3]
    dx = (pred[:, 0] - label[:, 0]) * sw * cpx
    dy = (pred[:, 1] - label[:, 1]) * sh * cpy
    return np.sqrt(dx ** 2 + dy ** 2)


def evaluate(per, cmap, alpha: float, target: str, grid: int):
    """全部会话合并后做**全局**分位映射（文档口径：全局分位数匹配）。"""
    all_s, all_lab, all_cm, owner = [], [], [], []
    for si, (sess, d) in enumerate(sorted(per.items())):
        s = ema(d["base"], alpha)
        all_s.append(s)
        all_lab.append(d["label"])
        cms = []
        for p in d["path"]:
            cms.append(cmap.get(p, (np.nan,) * 4))
        all_cm.append(np.asarray(cms, dtype=np.float64))
        owner.append(np.full(len(s), si))
    S = np.vstack(all_s)
    L = np.vstack(all_lab)
    C = np.vstack(all_cm)
    ok = np.isfinite(C).all(1)
    S, L, C = S[ok], L[ok], C[ok]
    if target == "labels":
        tq = target_quantiles("labels", len(S), labels=L)
    else:
        tq = target_quantiles(target, len(S), grid=grid)
    P = np.stack([rank_map(S[:, 0], tq), rank_map(S[:, 1], tq)], axis=1)
    e_ema = score_cm(S, L, C)
    e_map = score_cm(P, L, C)
    return e_ema, e_map, len(S)


def main():
    ap = argparse.ArgumentParser(description="复现 EMA + 全局分位数映射")
    ap.add_argument("--dump", default="out/rgbdgaze/analysis/ttadump_20sess_noes.csv")
    ap.add_argument("--index-csv", default="data/RGBDGaze/index/index.csv")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
    ap.add_argument("--targets", nargs="+", default=["uniform", "grid", "labels"])
    ap.add_argument("--grid", type=int, default=20)
    args = ap.parse_args()

    per = load_dump(Path(args.dump))
    cmap = load_cm(Path(args.index_csv))
    n = sum(len(d["base"]) for d in per.values())
    print(f"[repro] dump={args.dump} 会话={len(per)} 帧={n}", flush=True)

    base = np.vstack([d["base"] for _, d in sorted(per.items())])
    lab = np.vstack([d["label"] for _, d in sorted(per.items())])
    cms = np.vstack([np.asarray([cmap.get(p, (np.nan,) * 4) for p in d["path"]],
                                dtype=np.float64) for _, d in sorted(per.items())])
    ok = np.isfinite(cms).all(1)
    e_base = score_cm(base[ok], lab[ok], cms[ok])
    print(f"[repro] 基线(帧加权) = {e_base.mean():.4f} cm", flush=True)

    best = {}
    for target in args.targets:
        row = []
        for a in args.alphas:
            e_ema, e_map, m = evaluate(per, cmap, a, target, args.grid)
            row.append((a, e_ema.mean(), e_map.mean()))
            if target != "labels":
                prev = best.get(target)
                if prev is None or e_map.mean() < prev[1]:
                    best[target] = (a, e_map.mean())
        line = "  ".join(f"α={a:.1f}: EMA {x:.3f} / 映射 {y:.3f}" for a, x, y in row)
        print(f"[repro] 目标={target:7s} {line}", flush=True)
    for t, (a, v) in best.items():
        print(f"[repro] 最佳(无标签 {t}) α={a} -> {v:.4f} cm "
              f"（文档主结果 0.8533；差 {v - 0.8533:+.4f}）", flush=True)
    print("[repro] 注：labels 目标为泄漏对照，仅用于诊断，不可作为论文口径", flush=True)


if __name__ == "__main__":
    main()
