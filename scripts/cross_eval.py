"""跨设备 / 跨数据集交叉评估（P1：建立度量，否则改进无法证伪）。

回答的问题：
    ① 域内        RGBDGaze train -> RGBDGaze test          （当前 1.67cm）
    ② 留一设备    RGBDGaze 去掉某机型训练 -> 该机型测试      （尺寸泛化，内插）
    ③ 跨数据集    GazeCapture 训练 -> RGBDGaze 测试          （宽高比+光学外推，零样本）
    ④ 留一被试    去掉某人训练 -> 该人测试                   （个人泛化）

为什么必须做：P0（几何 token）、P2（几何域随机化）都只是"让模型有能力泛化"，
是否真的泛化只能靠 ②③ 的数字判断。①变好但②③不变 = 只是记住了 RGBDGaze。

指标口径复用 scripts.train.evaluate_split（cm 误差、几何角度、
≤1.91° 占比等），保证与训练期验收完全一致，不另起一套。

用法：
    # 域内：比较两个权重
    python -m scripts.cross_eval --dataset rgbdgaze --protocol domain \
        --ckpt out/.../g5_nodepth_448_aug/best.pt out/.../g6_nodepth_448_aug/best.pt

    # 留一设备（不训练，只看现有权重在该机型上的表现分布）
    python -m scripts.cross_eval --dataset rgbdgaze --protocol leave-device \
        --holdout "iPhone 11"

    # 生成"留一设备"的 train/val/test 清单，供 ablation 训练用
    python -m scripts.cross_eval --dataset rgbdgaze --protocol leave-device \
        --holdout "iPhone 11" --write-split out/rgbdgaze/cross_eval/leaveout_iPhone11

    # 跨数据集：GazeCapture 上预训练的权重，零样本测 RGBDGaze
    python -m scripts.cross_eval --dataset rgbdgaze --protocol domain \
        --ckpt out/gazecapture/pretrain/best.pt --all-splits --limit 5000
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from configs.paths import DEVICE as DEFAULT_DEVICE, experiment_dirs  # noqa: E402
from configs.rgbdgaze_config import RGBDGaze_INDEX_DIR  # noqa: E402
from configs.gazecapture_config import GAZECAPTURE_INDEX_DIR  # noqa: E402
from gazelab.datasets import (  # noqa: E402
    GazeCaptureDataset, RGBDGazeDataset, rgb_preprocess,
)
from gazelab.datasets.common import randomize_screen_geometry  # noqa: E402
from gazelab.models.rgbdgaze import RGBDGazeDINOv2  # noqa: E402
from scripts.train import Subset, predict_frames, summarize  # noqa: E402

# 各数据集用来分组的列（都存在于对应 index.csv 中）
GROUP_COLUMNS = {
    "rgbdgaze": [("device", "设备"), ("subject", "被试"), ("activity", "活动")],
    "gazecapture": [("device", "设备"), ("subject", "被试"), ("orientation", "朝向")],
}


def build_dataset(dataset: str, index_dir: Path, image_size: int):
    """按数据集名构造基础数据集（评估用：不做任何增强）。"""
    index_csv = index_dir / "index.csv"
    if dataset == "rgbdgaze":
        return RGBDGazeDataset(index_csv, color_transform=rgb_preprocess,
                               image_size=image_size)
    return GazeCaptureDataset(index_csv=index_csv, image_size=image_size)


def protocol_indices(dataset: str, rows, index_dir: Path, protocol: str, holdout):
    """返回 (train_idx, val_idx, test_idx)。

    domain        : 用索引目录里已有的 train/val/test 清单
    leave-device  : test = 指定机型的全部帧；train/val = 其余按被试 8:2 划分
    leave-subject : test = 指定被试；train/val = 其余被试 8:2 划分
    """
    path_col = "rgb_path" if dataset == "rgbdgaze" else "frame_path"
    if protocol == "domain":
        out = []
        for name in ("train", "val", "test"):
            f = index_dir / f"{name}.txt"
            if not f.is_file():
                raise FileNotFoundError(f"缺少划分清单 {f}（先跑对应 preprocess）")
            want = {ln.strip() for ln in f.read_text(encoding="utf-8").splitlines()
                    if ln.strip()}
            out.append([i for i, r in enumerate(rows) if r[path_col] in want])
        return tuple(out)

    if not holdout:
        raise ValueError(f"--protocol {protocol} 需要 --holdout")
    if protocol == "leave-device":
        is_test = [r["device"] == holdout for r in rows]
        what = "机型"
    elif protocol == "leave-subject":
        is_test = [r["subject"] == holdout for r in rows]
        what = "被试"
    else:
        raise ValueError(f"未知 protocol: {protocol}")

    test_idx = [i for i, t in enumerate(is_test) if t]
    rest = [i for i, t in enumerate(is_test) if not t]
    if not test_idx:
        raise ValueError(f"没有帧属于{what} {holdout!r}")

    # 其余按被试 8:2 划分（同一被试不跨 train/val）
    subjects = sorted({rows[i]["subject"] for i in rest})
    rng = np.random.RandomState(0)
    rng.shuffle(subjects)
    n_val = max(1, int(len(subjects) * 0.2))
    val_sub = set(subjects[:n_val])
    val_idx = [i for i in rest if rows[i]["subject"] in val_sub]
    train_idx = [i for i in rest if rows[i]["subject"] not in val_sub]
    return train_idx, val_idx, test_idx


def write_split(index_dir_out: Path, rows, dataset: str, splits, log):
    """把 train/val/test 清单写到指定目录（每行一个路径）。"""
    path_col = "rgb_path" if dataset == "rgbdgaze" else "frame_path"
    index_dir_out.mkdir(parents=True, exist_ok=True)
    for name, idx in zip(("train", "val", "test"), splits):
        with open(index_dir_out / f"{name}.txt", "w", encoding="utf-8") as f:
            for i in idx:
                f.write(rows[i][path_col] + "\n")
        log(f"[split] {name}: {len(idx)} 帧 -> {index_dir_out / (name + '.txt')}")


class GeomShiftDataset(Dataset):
    """把 (标签, 几何 token) 重投影到一个固定合成几何，用于测量模型的几何外推。

    这是判断 P0（几何 token）+ P2（几何域随机化）是否真的有效的**决定性实验**：
      · 带几何 token 且训练时做过几何随机化的模型 -> 换几何后误差应基本不变；
      · 不带几何 token 的模型 -> 无法感知几何变化，误差随几何偏移单调变大。

    目标屏幕取"两个方向都不小于真实屏幕"的尺寸，保证所有注视点仍落在屏内
    （不丢样本）。物理模型与 P2 的几何域随机化同源
    （gazelab.datasets.common.randomize_screen_geometry）。

    ⚠ 读法：不同 aspect 之间的**绝对值不可直接横向比较** —— 合成屏尺寸不同会让
    标签整体缩向屏幕中心（u'=0.5+(u-0.5)·W/W'），从而改变归一化误差的尺度。
    正确用法是在**同一个 aspect** 下比较"带几何 token 的模型"与"不带几何 token
    的模型"：前者应基本不变，后者应随几何偏移单调变差。
    """

    def __init__(self, base, indices, aspect: float):
        self.base = base
        self.idces = list(indices)
        self.aspect = float(aspect)
        w_real = max(float(r["screen_w"]) * float(r["cm_px_x"]) for r in base.rows)
        h_real = max(float(r["screen_h"]) * float(r["cm_px_y"]) for r in base.rows)
        # 取"不更小"的合成屏：W' >= w_real 且 H' = W'/a >= h_real
        self.width_cm = max(w_real, h_real * self.aspect) * 1.02
        self.height_cm = self.width_cm / self.aspect
        self._rng = np.random.RandomState(0)

    def __len__(self):
        return len(self.idces)

    def __getitem__(self, i):
        j = self.idces[i]
        inp, label = self.base[j]
        row = self.base.rows[j]
        w_cm = float(row["screen_w"]) * float(row["cm_px_x"])
        h_cm = float(row["screen_h"]) * float(row["cm_px_y"])
        eps = 1e-6
        out = randomize_screen_geometry(
            float(label[0]), float(label[1]), w_cm, h_cm, self._rng,
            (self.aspect, self.aspect), (self.width_cm, self.width_cm),
            (self.height_cm - eps, self.height_cm + eps))
        if out is None:
            raise RuntimeError("合成屏不小于真实屏，不应出现越界；请检查尺寸选取")
        u, v, geom = out
        inp.geom = geom
        return inp, torch.tensor([u, v], dtype=torch.float32)


def load_model(ckpt: Path, device, image_size: int, use_geom: bool = True):
    """加载 RGBDGazeDINOv2 权重。

    use_geom=False 用于评估"加入几何 token 之前"训练的老权重：老权重没有
    proj_geom 参数，若强行开启几何 token 它会是随机初始化的，结果无意义。

    strict=False：允许被评估权重缺少新增参数，但会把缺失/多余的键打印出来，
    避免"其实没加载上"这种静默错误。
    """
    model = RGBDGazeDINOv2(img_size=image_size, use_geom=use_geom).to(device)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [load] {ckpt.name}: 缺失键 {list(missing)}")
    if unexpected:
        print(f"  [load] {ckpt.name}: 多余键 {list(unexpected)}")
    model.eval()
    return model


def fmt(res):
    """把 evaluate_split 的返回格式化成一行关键指标。"""
    return (f"EM {res['em_dist']:.3f}cm | 角度 均 {res['ang_mean']:.2f}° 中位 "
            f"{res['ang_median']:.2f}° | ≤1.91° {res['ang_ratio_le_191']:.1f}% | n={res['count']}")


def main():
    ap = argparse.ArgumentParser(
        description="跨设备 / 跨数据集交叉评估",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--dataset", choices=["rgbdgaze", "gazecapture"],
                    default="rgbdgaze", help="用哪个数据集做测试")
    ap.add_argument("--index-dir", default=None,
                    help="索引目录；默认按 --dataset 从 config 取")
    ap.add_argument("--protocol", choices=["domain", "leave-device", "leave-subject"],
                    default="domain")
    ap.add_argument("--holdout", default=None,
                    help="留一设备时为机型名（如 'iPhone 11'）；留一被试时为被试 ID")
    ap.add_argument("--ckpt", nargs="+", required=False, default=None,
                    help="一个或多个权重路径")
    ap.add_argument("--image-size", type=int, default=448, help="模型输入分辨率")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0,
                    help="每组最多评估多少帧（0=全部，调试用）")
    ap.add_argument("--all-splits", action="store_true",
                    help="domain 协议下同时评估 train/val/test（跨数据集零样本用）")
    ap.add_argument("--write-split", default=None,
                    help="把 train/val/test 清单写到该目录（供训练用）")
    ap.add_argument("--no-geom", dest="use_geom", action="store_false", default=True,
                    help="关闭屏幕几何 token（评估 P0 之前训练的老权重时必须加）")
    ap.add_argument("--geom-shift", nargs="*", type=float, default=None,
                    help="几何外推实验：把标签+几何重投影到给定宽高比(可多个)，"
                         "比较各 aspect 下的误差（需配合带几何 token 的权重）")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = args.device if args.device != "auto" else DEFAULT_DEVICE
    index_dir = Path(args.index_dir) if args.index_dir else (
        RGBDGaze_INDEX_DIR if args.dataset == "rgbdgaze" else GAZECAPTURE_INDEX_DIR)

    log_dir = experiment_dirs("cross_eval", args.dataset)["log"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{datetime.now():%Y%m%d_%H%M%S}_cross_eval_{args.protocol}.txt"

    def log(msg=""):
        print(msg, flush=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log("=" * 78)
    log(f"交叉评估 | 数据集={args.dataset} 协议={args.protocol} "
        f"holdout={args.holdout}")
    log(f"索引={index_dir}  设备={device}  日志={log_file}")
    log("=" * 78)

    ds = build_dataset(args.dataset, index_dir, args.image_size)
    rows = ds.rows
    log(f"数据集 {len(rows)} 帧 / {len({r['subject'] for r in rows})} 被试 / "
        f"{len({r['device'] for r in rows})} 机型")

    splits = protocol_indices(args.dataset, rows, index_dir, args.protocol, args.holdout)
    names = ["train", "val", "test"]
    for n, idx in zip(names, splits):
        log(f"  划分 {n}: {len(idx)} 帧")
    if args.write_split:
        write_split(Path(args.write_split), rows, args.dataset, splits, log)

    if not args.ckpt:
        log("未提供 --ckpt，仅完成划分/清单生成。")
        return

    # 评估范围
    if args.protocol == "domain" and args.all_splits:
        eval_names = ["train", "val", "test"]
    else:
        eval_names = ["test"]
    eval_sets = [(n, splits[names.index(n)]) for n in eval_names]

    log("")
    for ckpt in args.ckpt:
        log("-" * 78)
        log(f"权重: {ckpt}  (use_geom={args.use_geom})")
        model = load_model(Path(ckpt), device, args.image_size,
                           use_geom=args.use_geom)

        for split_name, idx in eval_sets:
            used = idx if not args.limit else idx[:args.limit]
            log(f"\n  === {split_name} ({len(used)} 帧) ===")
            # 每个权重只前向一遍，所有分组/总体指标都由逐帧结果聚合得到
            dl = DataLoader(Subset(ds, used), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
            fr = predict_frames(model, dl, device)
            log(f"    总体            : {fmt(summarize(fr))}")

            if args.geom_shift:
                log("    -- 几何外推（标签+几何一起重投影到合成宽高比）--")
                for aspect in args.geom_shift:
                    gs = GeomShiftDataset(ds, used, aspect)
                    gdl = DataLoader(gs, batch_size=args.batch_size, shuffle=False,
                                     num_workers=args.num_workers)
                    log(f"    aspect={aspect:<6.3f}    : "
                        f"{fmt(summarize(predict_frames(model, gdl, device)))}")

            for col, label in GROUP_COLUMNS[args.dataset]:
                vals = np.array([rows[i][col] for i in used])
                for value in sorted(set(vals.tolist())):
                    sel = vals == value
                    if int(sel.sum()) < 50:
                        continue
                    log(f"    {f'{label}={value}':22s}: {fmt(summarize(fr, sel))}")

    log("")
    log(f"完成。日志: {log_file}")


if __name__ == "__main__":
    main()
