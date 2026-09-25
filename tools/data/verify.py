"""
verify_gaze360.py — GazeHub 格式 Gaze360 数据校验

对 preprocess_gaze360.py 生成的 GazeHub 标准格式数据做端到端校验：
  1. 目录结构 + 表头
  2. 标签逐行解析（模拟真实解析器 line.split(" ")）
  3. Face 图像尺寸 (224, 224)
  4. 集合间无样本泄露（Face 路径 + Origin 帧标识）
  5. DatasetGaze360ByGazeHub 实例化（与 train.py 完全一致的构造方式）
  6. DataLoader 批量遍历：shape / dtype / 有限性 / 值域 / 单位向量标签
  7. 统计与结论（全部通过打印 "数据校验通过"，exit 0）

不导入 train.py（其 model_zhao_test 导入在本仓库是坏导入）；
ZhaoDataset 只是向内透传 DatasetGaze360ByGazeHub 的薄封装，直接校验内层类等价。
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import numpy as np
from collections import Counter
from PIL import Image
from torch.utils.data import DataLoader

from configs.config import CLIP_PREPROCESS, CNN_PREPROCESS, BATCH_SIZE, NUM_WORKERS
from gazelab.datasets import DatasetGaze360ByGazeHub

SPLITS = ("train", "val", "test")
HEADER = "Face Left Right Origin 3DGaze 2DGaze"
FACE_SIZE = (224, 224)


def fail(msg: str) -> None:
    print(f"[校验失败] {msg}")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GazeHub 格式 Gaze360 数据校验")
    p.add_argument("--data-dir", default="./data/Gaze360",
                   help="数据目录（默认 ./data/Gaze360，与 config.DATASETS_PATH/TRAIN_DATASET_NAME 对齐）")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                   help=f"DataLoader batch size（默认 config.BATCH_SIZE={BATCH_SIZE}）")
    p.add_argument("--max-batches-per-split", type=int, default=0,
                   help="每集合最多遍历的 batch 数；0=全量遍历（默认 0）")
    return p.parse_args()


def norm_check(g: np.ndarray) -> bool:
    n = float(np.linalg.norm(g))
    return math.isfinite(n) and abs(n - 1.0) <= 1e-3


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    gaze_hub = data_dir / "GazeHub"

    # ---------- 1. 结构 ----------
    print("[1/7] 检查目录结构...")
    label_files = {}
    for s in SPLITS:
        fp = gaze_hub / "Label" / f"{s}.label"
        if not fp.is_file():
            fail(f"缺少 label 文件: {fp}")
        label_files[s] = fp
        img_root = gaze_hub / "Image" / s
        for sub in ("Face", "Left", "Right"):
            d = img_root / sub
            if not d.is_dir():
                fail(f"缺少图像目录: {d}")
    # 表头
    for s in SPLITS:
        with label_files[s].open(encoding="utf-8") as f:
            header = f.readline().rstrip("\n")
        if header != HEADER:
            fail(f"{label_files[s]} 表头错误: {header!r}（期望 {HEADER!r}）")
    print(f"      结构 OK: Label/{{{','.join(SPLITS)}}}.label, Image/{{split}}/{{Face,Left,Right}} 齐备")

    # ---------- 2. 标签逐行解析 ----------
    print("[2/7] 解析标签（模拟真实解析器 line.split(' ')）...")
    face_paths = {}   # split -> list[str]
    origins = {}      # split -> set[str]
    n_label_lines = {}
    for s in SPLITS:
        lines = label_files[s].read_text(encoding="utf-8").splitlines()[1:]
        fp_list = []
        origin_set = set()
        for ln, line in enumerate(lines, start=2):  # 行号含表头
            if line.strip() == "":
                continue
            fields = line.split(" ")
            if len(fields) != 6 or any(fld == "" for fld in fields):
                fail(f"{label_files[s]}:{ln} 字段数错误（期望 6 个非空字段）: {line!r}")
            face, left, right, origin, d3, d2 = fields
            # Face 文件存在
            if not (gaze_hub / "Image" / s / face).is_file():
                fail(f"{label_files[s]}:{ln} Face 文件不存在: {face}")
            if not (gaze_hub / "Image" / s / left).is_file():
                fail(f"{label_files[s]}:{ln} Left 文件不存在: {left}")
            if not (gaze_hub / "Image" / s / right).is_file():
                fail(f"{label_files[s]}:{ln} Right 文件不存在: {right}")
            # 3DGaze: 3 个有限浮点、单位向量、分量范围
            try:
                g = np.array([float(x) for x in d3.split(",")])
            except ValueError:
                fail(f"{label_files[s]}:{ln} 3DGaze 解析失败: {d3!r}")
            if g.shape != (3,) or not np.all(np.isfinite(g)):
                fail(f"{label_files[s]}:{ln} 3DGaze 非 3 个有限浮点: {d3!r}")
            if not np.all((g >= -1 - 1e-6) & (g <= 1 + 1e-6)):
                fail(f"{label_files[s]}:{ln} 3DGaze 分量超出 [-1,1]: {d3!r}")
            if not norm_check(g):
                fail(f"{label_files[s]}:{ln} 3DGaze 非单位向量（norm={float(np.linalg.norm(g))}）: {d3!r}")
            # 2DGaze: 2 个有限浮点
            try:
                e = np.array([float(x) for x in d2.split(",")])
            except ValueError:
                fail(f"{label_files[s]}:{ln} 2DGaze 解析失败: {d2!r}")
            if e.shape != (2,) or not np.all(np.isfinite(e)):
                fail(f"{label_files[s]}:{ln} 2DGaze 非 2 个有限浮点: {d2!r}")
            if origin in origin_set:
                fail(f"{label_files[s]}:{ln} Origin 重复: {origin}")
            origin_set.add(origin)
            fp_list.append(face)
        n_label_lines[s] = len(lines)
        face_paths[s] = fp_list
        origins[s] = origin_set
        print(f"      {s}: {n_label_lines[s]} 行，全部 6 字段合法，3D 单位向量 OK")

    # ---------- 3. 图像尺寸 ----------
    print("[3/7] 检查 Face 图像尺寸 (224, 224)...")
    for s in SPLITS:
        for k, face in enumerate(face_paths[s]):
            with Image.open(gaze_hub / "Image" / s / face) as im:
                if im.size != FACE_SIZE:
                    fail(f"Image/{s}/{face} 尺寸 {im.size} != {FACE_SIZE}")
        print(f"      {s}: {len(face_paths[s])} 张 Face 均为 224x224")

    # ---------- 4. 无样本泄露 ----------
    # 注：官方 split 的被试（person）级重叠是 Gaze360 官方划分的固有属性，
    #     样本级（图像/帧）天然互斥；此处按样本级校验。
    print("[4/7] 检查集合间无样本泄露...")
    sets_face = {s: set(face_paths[s]) for s in SPLITS}
    sets_origin = {s: set(origins[s]) for s in SPLITS}
    for i, s1 in enumerate(SPLITS):
        for s2 in SPLITS[i + 1:]:
            inter_face = sets_face[s1] & sets_face[s2]
            if inter_face:
                fail(f"{s1} 与 {s2} Face 路径交集非空，示例: {sorted(inter_face)[:3]}")
            inter_origin = sets_origin[s1] & sets_origin[s2]
            if inter_origin:
                fail(f"{s1} 与 {s2} Origin 交集非空，示例: {sorted(inter_origin)[:3]}")
    # 内部 Face 路径无重复
    for s in SPLITS:
        dup = {f: c for f, c in Counter(face_paths[s]).items() if c > 1}
        if dup:
            fail(f"{s} 集合内 Face 路径重复: {list(dup.items())[:3]}")
    # 全局 Origin 无重复
    all_origins = [o for s in SPLITS for o in origins[s]]
    if len(all_origins) != len(set(all_origins)):
        fail("全局 Origin（rec/pid/frame）存在重复")
    print(f"      两两交集为空；各集合内部 Face 无重复；全局 {len(all_origins)} 个 Origin 无重复")

    # ---------- 5. Dataset 实例化 ----------
    print("[5/7] 实例化 DatasetGaze360ByGazeHub（与 train.py 一致）...")
    datasets = {}
    for s in SPLITS:
        ds = DatasetGaze360ByGazeHub(
            gaze_hub / "Image" / s,
            [label_files[s]],
            CLIP_PREPROCESS,
            CNN_PREPROCESS,
        )
        if len(ds) != n_label_lines[s]:
            fail(f"{s}: len(ds)={len(ds)} != label 行数 {n_label_lines[s]}")
        datasets[s] = ds
        print(f"      {s}: len(ds) = {len(ds)} == label 行数 OK")

    # ---------- 6. 批量遍历 ----------
    full = args.max_batches_per_split == 0
    print(f"[6/7] DataLoader 批量遍历（batch_size={args.batch_size}"
          f"{'，全量' if full else f'，每集合前 {args.max_batches_per_split} batch'}）...")
    traversed = {}
    for s in SPLITS:
        ds = datasets[s]
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=NUM_WORKERS)
        seen = 0
        for bi, (_input, label) in enumerate(loader):
            if not full and bi >= args.max_batches_per_split:
                break
            face = _input["face"]
            other = _input["other_face"]
            B = face.shape[0]
            if face.shape != (B, 3, 224, 224) or other.shape != (B, 3, 224, 224):
                fail(f"{s} batch {bi}: face/other_face shape {face.shape}/{other.shape} 异常")
            for name, t in (("face", face), ("other_face", other)):
                if t.dtype != torch.float32:
                    fail(f"{s} batch {bi}: {name} dtype {t.dtype} != float32")
                if not torch.isfinite(t).all():
                    fail(f"{s} batch {bi}: {name} 含非有限值")
                if float(t.abs().max()) > 3.0:
                    fail(f"{s} batch {bi}: {name} 值域超出 [-3,3]，max|v|={float(t.abs().max())}")
            if label.shape != (B, 3):
                fail(f"{s} batch {bi}: label shape {label.shape} != [B,3]")
            if not torch.isfinite(label).all():
                fail(f"{s} batch {bi}: label 含非有限值")
            norms = label.norm(dim=-1)
            if float((norms - 1.0).abs().max()) > 1e-3:
                fail(f"{s} batch {bi}: label 非单位向量，max|norm-1|={float((norms - 1.0).abs().max())}")
            seen += B
        traversed[s] = seen
        expect = len(ds) if full else min(seen, len(ds))
        if full and seen != len(ds):
            fail(f"{s}: 遍历 {seen} 样本 != len(ds) {len(ds)}")
        mode = "全量" if full else f"前 {args.max_batches_per_split} batch"
        print(f"      {s}: {mode}遍历 {seen} 样本，shape/dtype/有限性/值域/单位向量 OK")

    # ---------- 7. 统计与结论 ----------
    print("[7/7] 统计汇总")
    print("-" * 52)
    total = 0
    for s in SPLITS:
        ok = n_label_lines[s] == len(datasets[s])
        if full:
            ok = ok and traversed[s] == len(datasets[s])
        print(f"  {s:<6}: label 行数 {n_label_lines[s]:>7d} | "
              f"Dataset 长度 {len(datasets[s]):>7d} | 遍历 {traversed[s]:>7d}"
              f" | {'一致' if ok else '不一致'}")
        total += len(datasets[s])
    print(f"  {'total':<6}: {total:>7d}")
    print("-" * 52)
    print("数据校验通过")
    sys.exit(0)


if __name__ == "__main__":
    main()
