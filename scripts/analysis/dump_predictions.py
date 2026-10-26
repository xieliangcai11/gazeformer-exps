"""把一个 ckpt 在某划分上的**逐帧预测**导出为 CSV，并直接给出结构分析。

为什么需要：现有分析脚本（error_budget / bias_decomposition）各自从 ckpt 出发、
口径写死且偏重 GPU 计算；而"先导出全部预测、再任意分析"更透明、可本地复看、
也便于把最差样本挑出来看原图。

**方向必须与训练一致**（关键坑）：历史 ckpt（如 test 1.64cm 的 g6_multi）是在
np.rot90(img,1) 的躺倒朝向下训练的 ⇒ 评估要加 --orient rot90；当前代码默认 raw
（正立）。用错朝向会让数字完全失真。本脚本会打印汇总并与预期核对（自检）。

输出 CSV 列：
  path, subject, activity, device, frame_index,
  pred_x, pred_y, label_x, label_y, em_cm, err_x_cm, err_y_cm, ang_deg
其中 pred/label 为归一化屏幕坐标（0~1），em_cm 为屏幕物理厘米误差，ang_deg 为
"到屏心偏移 + 基准视距"口径的几何角度误差（与训练期验收口径一致）。

用法：
    # 复现 1.64cm 的权重（躺倒朝向，448，无 geom）
    python -m scripts.analysis.dump_predictions \
        --ckpt out/rgbdgaze/ablation/checkpoints/g6_multi/best.pt \
        --resolution 448 --no-geom --orient rot90 \
        --out out/rgbdgaze/analysis/g6_multi_test.csv
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
    raise RuntimeError(f"找不到项目根：从 {here} 向上未发现 src/gazelab/__init__.py")


_PROJECT_ROOT = _bootstrap()

from configs.paths import experiment_dirs  # noqa: E402
from gazelab.datasets.table_index import TableIndex  # noqa: E402
from gazelab.models.dinov2_gaze import RGBDGazeDINOv2, load_state_dict_any_res  # noqa: E402
from scripts.train import DATASET_SPECS, build_dataloader, predict_frames, summarize  # noqa: E402


def _frame_no(path: str) -> int:
    stem = Path(path).stem
    return int(stem) if stem.isdigit() else -1


def _group_report(name, keys, em, ang, thr=1.91):
    """按分组键给出 中位/均值/前95%平均/≤1.91°占比，按样本数降序。"""
    buckets = defaultdict(list)
    for k, e, a in zip(keys, em, ang):
        buckets[k].append((e, a))
    rows = []
    for k, v in buckets.items():
        e = np.array([x[0] for x in v])
        a = np.array([x[1] for x in v])
        rows.append((k, len(v), float(np.median(e)), float(e.mean()),
                     float(np.percentile(e, 95)), float((a <= thr).mean() * 100)))
    rows.sort(key=lambda r: -r[1])
    print(f"\n  {name:<14}{'n':>7}{'EM中位':>9}{'EM均值':>9}{'EM p95':>9}{'≤1.91°':>9}")
    for k, n, med, mean, p95, hit in rows:
        print(f"  {str(k):<14}{n:>7}{med:>9.3f}{mean:>9.3f}{p95:>9.3f}{hit:>8.1f}%")


def main():
    ap = argparse.ArgumentParser(description="导出逐帧预测 CSV + 结构分析")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", choices=sorted(DATASET_SPECS), default="rgbdgaze")
    ap.add_argument("--index-dir", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--resolution", type=int, default=448)
    ap.add_argument("--orient", choices=["raw", "rot90"], default="raw",
                    help="raw=正立原图（新实验）；rot90=历史躺倒朝向（评估 g6_multi 等旧 ckpt）")
    ap.add_argument("--no-geom", dest="use_geom", action="store_false", default=True)
    ap.add_argument("--use-eye", dest="use_eye", action="store_true")
    ap.add_argument("--unfreeze", type=int, default=12,
                    help="只影响 requires_grad/前向分支，对评估无影响；默认与训练一致")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--limit", type=int, default=0, help="总共最多评估多少帧（0=全部）")
    ap.add_argument("--per-session", type=int, default=0,
                    help="每个 (subject,activity) 会话最多取多少帧（0=全部；用于快速生成源域表）")
    ap.add_argument("--eye-scale", type=float, default=0.0,
                    help="眼框收紧倍数：**必须与训练时一致**，否则眼支流输入与训练不符"
                         "（此前评估脚本无法传该参数 ⇒ 用 eye_scale>0 训的权重会被按 0 评估）")
    ap.add_argument("--eye-shuffle", dest="eye_shuffle", action="store_true",
                    help="对照实验：打乱眼图与当前帧的对应。评估 eyeshuf* 权重时必须开启，"
                         "否则用对齐眼图评一个没见过对应关系的模型 ⇒ 数字被抬高")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # eye_shuffle 在 scripts/train.py 里是模块级全局（predict_frames 读它）⇒ 显式设置
    import scripts.train as _T
    _T.EYE_SHUFFLE = bool(args.eye_shuffle)

    # ckpt 旁边的配置侧车（training 时写入）：打印出来，避免"靠日志猜配置"
    import json
    _sc = Path(args.ckpt).parent / "train_args.json"
    args.geom_orient = False
    if _sc.is_file():
        try:
            _cfg = json.loads(_sc.read_text(encoding="utf-8"))
            print(f"[ckpt] 训练配置侧车 {_sc}")
            print("       " + json.dumps(_cfg, ensure_ascii=False)[:500])
            # 以训练时配置为准：给无几何权重塞随机 proj_geom / 给带朝向权重
            # 少喂 4 维都会让指标失真
            args.use_geom = bool(_cfg.get("use_geom", args.use_geom))
            args.geom_orient = bool(_cfg.get("geom_orient", False)) and args.use_geom
        except Exception as e:  # 侧车损坏不应影响评估
            print(f"[ckpt] 侧车读取失败（忽略）：{e}")

    import torch
    spec = DATASET_SPECS[args.dataset]
    index_dir = Path(args.index_dir) if args.index_dir else Path(spec["default_index"])
    out_csv = Path(args.out) if args.out else (
        experiment_dirs(args.dataset, "analysis")["root"] /
        f"preds_{Path(args.ckpt).parent.name}_{args.split}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    dev = args.device if args.device != "auto" else (
        "cuda:0" if torch.cuda.is_available() else "cpu")

    # ---- 选取帧（可选：每会话截断 / 总数截断）----
    paths = [ln.strip() for ln in (index_dir / f"{args.split}.txt").read_text(
        encoding="utf-8").splitlines() if ln.strip()]
    data_dir = index_dir
    if args.per_session or args.limit:
        _tbl = TableIndex(index_dir / "index.csv")
        _p2i = _tbl.lookup(spec["path_col"])
        if args.per_session:
            # **均匀抽样**（不能取前 N 帧：会话开头往往更容易，会把统计带偏）
            by_key = {}
            for i_, p_ in enumerate(paths):
                r = _tbl[_p2i[p_]]
                k = (r.get("subject", ""), r.get("activity", ""))
                by_key.setdefault(k, []).append(i_)
            sel = []
            for k, idxs in by_key.items():
                n = args.per_session
                if len(idxs) <= n:
                    sel.extend(idxs)
                else:
                    take = np.linspace(0, len(idxs) - 1, n).round().astype(int)
                    sel.extend([idxs[t] for t in np.unique(take)])
            paths = [paths[i] for i in sorted(set(sel))]
        if args.limit:
            paths = paths[: args.limit]
        # 用临时目录承载"截断后的清单 + 同一份 index.csv"
        tmp = Path(args.out).parent / "_split_tmp" if args.out else \
            (experiment_dirs(args.dataset, "analysis")["root"] / "_split_tmp")
        tmp.mkdir(parents=True, exist_ok=True)
        idx_link = tmp / "index.csv"
        if not idx_link.exists():
            try:
                idx_link.symlink_to((index_dir / "index.csv").resolve())
            except OSError:
                idx_link.write_bytes((index_dir / "index.csv").read_bytes())
        # 眼框表也需就位（数据集按 index 同目录加载）
        for _extra in ("eye_boxes.npz",):
            _src, _dst = index_dir / _extra, tmp / _extra
            if _src.is_file() and not _dst.exists():
                try:
                    _dst.symlink_to(_src.resolve())
                except OSError:
                    import shutil as _sh
                    _sh.copy2(_src, _dst)
        (tmp / f"{args.split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
        data_dir = tmp
        print(f"[split] 截断后 {len(paths)} 帧（per-session={args.per_session} limit={args.limit}）")

    # ---- 数据（朝向/眼支流必须与训练一致）----
    dl = build_dataloader(args.dataset, args.split, data_dir, args.batch_size, False,
                          args.num_workers, augment=False, resolution=args.resolution,
                          eye_crops=args.use_eye, orient=args.orient,
                          eye_scale=args.eye_scale,
                          geom_orient=getattr(args, "geom_orient", False))

    # ---- 模型 ----
    model = RGBDGazeDINOv2(unfreeze_last=args.unfreeze, use_geom=args.use_geom,
                           use_geom_orient=getattr(args, "geom_orient", False),
                           use_eye=args.use_eye, img_size=args.resolution).to(dev)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    miss, unexp, resampled = load_state_dict_any_res(model, state)
    print(f"[load] {args.ckpt}")
    if resampled is not None:
        print(f"[load] pos_embed {resampled} -> 目标网格，已插值")
    print(f"[load] 缺失键 {list(miss) or '无'} | 多余键 {list(unexp) or '无'}")
    if miss or unexp:
        print("  ⚠ 存在缺失/多余键：确认模型配置（--resolution/--no-geom/--use-eye）是否与训练一致")

    print(f"[eval] 划分={args.split} 帧数={len(paths)} 朝向={args.orient} "
          f"分辨率={args.resolution} geom={args.use_geom} eye={args.use_eye} "
          f"eye_scale={args.eye_scale} eye_shuffle={bool(args.eye_shuffle)} 设备={dev}")
    fr = predict_frames(model, dl, dev)
    # 加载器可能已剔除 filled 眼框行（build_dataloader 的行为）⇒ 用其 idces 对齐 paths
    _idces = getattr(dl.dataset, "idces", None)
    if _idces is not None:
        # Subset.idces 是 index.csv 的**行号**（不是清单下标）
        _tbl_all = TableIndex(index_dir / "index.csv")
        _pcol = spec["path_col"]
        paths = [_tbl_all[int(i)][_pcol] for i in _idces]
    n = len(fr["pred"])
    if n != len(paths):
        raise SystemExit(f"预测数 {n} != 清单帧数 {len(paths)}（顺序假设不成立，不能贴标签）")

    # ---- 元数据（按清单顺序对齐）----
    table = TableIndex(index_dir / "index.csv")
    p2i = table.lookup(spec["path_col"])
    rows = [table[p2i[p]] for p in paths]

    pred, label = fr["pred"], fr["label"]
    # predict_frames 只给 cm_x/cm_y（cm_dist 由 summarize 现算），故在此合成距离
    em = np.hypot(fr["cm_x"], fr["cm_y"])
    ang = fr["ang_deg"]
    subj = [r.get("subject", "?") for r in rows]
    act = [r.get("activity", r.get("orientation", "?")) for r in rows]
    devl = [r.get("device", "?") for r in rows]

    # ---- 写 CSV ----
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "subject", "activity", "device", "frame_index",
                    "pred_x", "pred_y", "label_x", "label_y",
                    "em_cm", "err_x_cm", "err_y_cm", "ang_deg"])
        for i in range(n):
            w.writerow([paths[i], subj[i], act[i], devl[i], _frame_no(paths[i]),
                        f"{pred[i, 0]:.6f}", f"{pred[i, 1]:.6f}",
                        f"{label[i, 0]:.6f}", f"{label[i, 1]:.6f}",
                        f"{em[i]:.6f}", f"{fr['cm_x'][i]:.6f}", f"{fr['cm_y'][i]:.6f}",
                        f"{ang[i]:.4f}"])
    print(f"[dump] 已写出 {out_csv}（{n} 帧）")

    # ---- 自检 + 结构分析 ----
    s = summarize(fr)
    print(f"\n[自检] test_em {s['em_dist']:.3f}cm | 角 均 {s['ang_mean']:.2f}° "
          f"中位 {s['ang_median']:.2f}° | ≤1.91° {s['ang_ratio_le_191']:.1f}% | n={s['count']}")
    print("        （评估 g6_multi/best.pt 时应≈1.64cm；偏差大说明朝向/口径不对）")
    print(f"\n  总体 EM: 中位 {np.median(em):.3f} | 均值 {em.mean():.3f} | "
          f"p90 {np.percentile(em, 90):.3f} | p95 {np.percentile(em, 95):.3f} | "
          f"最大 {em.max():.3f} cm")
    for t in (1.0, 2.0, 3.0, 5.0):
        print(f"    误差 >{t:.0f}cm 的比例: {(em > t).mean() * 100:5.1f}%")
    _group_report("按被试", subj, em, ang)
    _group_report("按活动", act, em, ang)
    _group_report("按设备", devl, em, ang)

    # 屏幕 3x3 分区（标签位置）——看是否存在系统性空间偏差
    gx = np.clip((label[:, 0] * 3).astype(int), 0, 2)
    gy = np.clip((label[:, 1] * 3).astype(int), 0, 2)
    _group_report("按屏幕分区", [f"x{gx[i] + 1}y{gy[i] + 1}" for i in range(n)], em, ang)

    # 最差帧（供看原图）
    worst = np.argsort(-em)[:15]
    print("\n  最差 15 帧（路径 / 被试 / EM cm）：")
    for i in worst:
        print(f"    {em[i]:6.2f}  {subj[i]:>6}  {paths[i]}")
    print(f"\n  CSV: {out_csv}")


if __name__ == "__main__":
    main()
