"""
preprocess_gaze360.py — Gaze360 原始数据预处理

将 ./gaze360/ 下的原始数据（metadata.mat + imgs/ 头部裁剪图）处理为
GazeHub 标准格式，可被 gazehub_datasets.py 的 DatasetGaze360ByGazeHub
（即 DatasetEyeDiapByGazeHub）与 train.py / train_test.py 直接加载。

输出结构（与 train.py 的 images_path 接线一致，Face 路径相对 Image/{split}/）：
    {output-dir}/GazeHub/Label/{train,val,test}.label
    {output-dir}/GazeHub/Image/{train,val,test}/{Face,Left,Right}/*.jpg

label 文件格式（表头 + 单空格分隔 6 字段）：
    Face Left Right Origin 3DGaze 2DGaze
    Face/000001.jpg Left/000001.jpg Right/000001.jpg rec_000/head/000001/000002.jpg x,y,z yaw,pitch

裁剪算法照抄 GazeHub 官方参考实现 data_processing_gaze360.py：
- Face:  face bbox 重投影到 head 裁剪坐标系，取正方形中心裁剪，resize 224x224
- Left/Right: 同式重投影，裁剪后 resize 60x36
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio
from tqdm import tqdm

SPLITS = ("train", "val", "test")
HEADER = "Face Left Right Origin 3DGaze 2DGaze\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gaze360 -> GazeHub 标准格式预处理")
    p.add_argument("--input-dir", default="./gaze360", help="原始数据目录（默认 ./gaze360）")
    p.add_argument("--output-dir", default="./data/Gaze360",
                   help="输出目录（默认 ./data/Gaze360，与 config.DATASETS_PATH/TRAIN_DATASET_NAME 对齐）")
    p.add_argument("--split-mode", choices=["official", "ratio"], default="official",
                   help="official=按官方 split 字段；ratio=按比例随机划分（默认 official）")
    p.add_argument("--train-ratio", type=float, default=0.8, help="ratio 模式 train 比例（默认 0.8）")
    p.add_argument("--val-ratio", type=float, default=0.1, help="ratio 模式 val 比例（默认 0.1，test=1-train-val）")
    p.add_argument("--seed", type=int, default=0, help="ratio 模式打乱随机种子（默认 0）")
    p.add_argument("--force", action="store_true",
                   help="输出目录 GazeHub 已存在且非空时，删除其下 Image/ 与 Label/ 后重建")
    return p.parse_args()


def crop_face(img: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Face 裁剪（照抄 GazeHub 参考 CropFaceImg）：
    bbox 为归一化 [x,y,w,h]（相对 head 裁剪图），输出 224x224。
    参考实现特性：center 只钳制下界（左上），右下越界由 numpy 切片自然截断。
    """
    h, w = img.shape[:2]
    bbox_px = np.concatenate([bbox[:2] * [w, h], bbox[2:] * [w, h]]).astype(int)
    center = [bbox_px[0] + bbox_px[2] // 2, bbox_px[1] + bbox_px[3] // 2]
    length = int(max(bbox_px[2], bbox_px[3]) / 2)
    for i in range(2):
        center[i] = max(center[i], length)
    crop = img[center[1] - length:center[1] + length, center[0] - length:center[0] + length]
    if crop.size == 0:  # 极端越界防护：回退整幅 head 图
        crop = img
    return cv2.resize(crop, (224, 224))


def crop_eye(img: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Left/Right 眼裁剪（照抄 GazeHub 参考 CropEyeImg），输出 60x36。"""
    h, w = img.shape[:2]
    bbox_px = np.concatenate([bbox[:2] * [w, h], bbox[2:] * [w, h]]).astype(int)
    center = [bbox_px[0] + bbox_px[2] // 2, bbox_px[1] + bbox_px[3] // 2]
    height = bbox_px[3] / 36
    width = bbox_px[2] / 60
    ratio = max(height, width)
    size = [int(ratio * 30), int(ratio * 18)]
    for i in range(2):
        center[i] = max(center[i], size[i])
    crop = img[center[1] - size[1]:center[1] + size[1], center[0] - size[0]:center[0] + size[0]]
    if crop.size == 0:  # 极端越界防护
        crop = np.zeros((36, 60, 3), dtype=np.uint8)
    return cv2.resize(crop, (60, 36))


def main() -> None:
    args = parse_args()
    t0 = time.time()

    # ---- 1. 校验输入 ----
    input_dir = Path(args.input_dir)
    meta_path = input_dir / "metadata.mat"
    imgs_dir = input_dir / "imgs"
    if not meta_path.is_file():
        sys.exit(f"[错误] 找不到 metadata 文件: {meta_path}")
    if not imgs_dir.is_dir():
        sys.exit(f"[错误] 找不到图像目录: {imgs_dir}")

    # ---- 磁盘空间前置检查（输出约 3GB，要求剩余 >= 7GB）----
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(out_root).free / (1024 ** 3)
    if free_gb < 7:
        sys.exit(f"[错误] 磁盘剩余空间不足: {free_gb:.1f}GB < 7GB（{out_root} 所在卷）")

    # ---- 输出目录冲突处理 ----
    gaze_hub_dir = out_root / "GazeHub"
    if gaze_hub_dir.exists() and any(gaze_hub_dir.iterdir()):
        if not args.force:
            sys.exit(f"[错误] 输出目录已存在且非空: {gaze_hub_dir}\n"
                     f"       确认后可加 --force 删除其下 Image/ 与 Label/ 后重建")
        for sub in ("Image", "Label"):
            p = gaze_hub_dir / sub
            if p.exists():
                shutil.rmtree(p)
        print(f"[提示] --force: 已删除 {gaze_hub_dir / 'Image'} 与 {gaze_hub_dir / 'Label'}")

    # ---- 2. 读取 metadata ----
    print(f"[步骤] 读取 {meta_path} ...")
    m = sio.loadmat(str(meta_path))
    def col(key: str) -> np.ndarray:
        v = m[key]
        if v.ndim == 2 and v.shape[0] == 1:  # MATLAB cell 数组 (1, N)
            v = v[0]
        elif v.ndim == 2 and v.shape[1] == 1:  # (N, 1)
            v = v[:, 0]
        return v
    recordings = [np.asarray(r).item() for r in col("recordings")]
    rec_idx = col("recording").astype(np.int64)  # 每行对应的 recordings 索引
    frame = col("frame").astype(np.int64)
    pid = col("person_identity").astype(np.int64)
    gaze_dir = np.asarray(col("gaze_dir"), dtype=np.float64)
    head_bbox = np.asarray(col("person_head_bbox"), dtype=np.float64)
    face_bbox = np.asarray(col("person_face_bbox"), dtype=np.float64)
    split = col("split").astype(np.int64)
    n_total = len(frame)
    print(f"[步骤] metadata 共 {n_total} 行")

    # ---- 3. 有效行（跳过 face bbox = [-1,-1,-1,-1]，无 face 标注）----
    valid_mask = ~np.all(face_bbox == -1, axis=1)
    n_no_face = int((~valid_mask).sum())
    valid_idx = np.where(valid_mask)[0]  # 全局行号（0 基），用作唯一图像文件名
    print(f"[步骤] 有效行（face 标注存在）: {len(valid_idx)}，跳过无 face 标注: {n_no_face}")

    # ---- 4. 划分 ----
    n_unused_valid = 0
    if args.split_mode == "official":
        # 官方 split: 0=train, 1=val, 2=test, 3=unused（先剔除 unused）
        split_of = {0: "train", 1: "val", 2: "test"}
        n_unused_valid = int(((split[valid_idx] == 3)).sum())
        valid_idx = valid_idx[split[valid_idx] != 3]
        split_of_idx = [split_of[int(split[i])] for i in valid_idx]
        print(f"[步骤] official 划分（跳过 unused 有效行 {n_unused_valid}）: "
              + " / ".join(f"{s}={split_of_idx.count(s)}" for s in SPLITS))
    else:
        if not (args.train_ratio > 0 and args.val_ratio > 0
                and args.train_ratio + args.val_ratio < 1):
            sys.exit(f"[错误] 比例非法: train={args.train_ratio}, val={args.val_ratio}，"
                     f"需满足 0 < train_ratio、0 < val_ratio、train_ratio + val_ratio < 1")
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(valid_idx))
        n = len(valid_idx)
        n_train = int(n * args.train_ratio)
        n_val = int(n * args.val_ratio)
        # test = 剩余
        split_of_idx = (["train"] * n_train + ["val"] * n_val
                        + ["test"] * (n - n_train - n_val))
        split_of_idx = [split_of_idx[k] for k in perm]
        print(f"[步骤] ratio 划分（seed={args.seed}）: "
              + " / ".join(f"{s}={split_of_idx.count(s)}" for s in SPLITS))

    # ---- 5. 创建目录 ----
    for s in SPLITS:
        for sub in ("Face", "Left", "Right"):
            (gaze_hub_dir / "Image" / s / sub).mkdir(parents=True, exist_ok=True)
    label_dir = gaze_hub_dir / "Label"
    label_dir.mkdir(parents=True, exist_ok=True)
    label_files = {}
    for s in SPLITS:
        fp = label_dir / f"{s}.label"
        label_files[s] = fp.open("w", encoding="utf-8")
        label_files[s].write(HEADER)

    # ---- 6. 逐行处理 ----
    print(f"[步骤] 逐行裁剪 {len(valid_idx)} 幅头部图（Face 224x224 + Left/Right 60x36）...")
    counts = {s: 0 for s in SPLITS}
    for pos, i in enumerate(tqdm(valid_idx, desc="裁剪", unit="row")):
        i = int(i)
        rec = recordings[rec_idx[i]]
        s = split_of_idx[pos]
        name = f"{i + 1:06d}.jpg"  # 全局行号+1，全局唯一

        img = cv2.imread(str(imgs_dir / rec / "head" / f"{int(pid[i]):06d}" / f"{int(frame[i]):06d}.jpg"))
        if img is None:
            label_files[s].close()
            sys.exit(f"[错误] 图像读取失败（cv2.imread 返回 None），metadata 行号 {i}，路径 "
                     f"{imgs_dir / rec / 'head' / f'{int(pid[i]):06d}' / f'{int(frame[i]):06d}.jpg'}")

        hb = head_bbox[i]
        fb = face_bbox[i]
        # face bbox 重投影到 head 裁剪图归一化坐标 [x,y,w,h]
        bbox = np.array([(fb[0] - hb[0]) / hb[2], (fb[1] - hb[1]) / hb[3],
                         fb[2] / hb[2], fb[3] / hb[3]])

        cv2.imwrite(str(gaze_hub_dir / "Image" / s / "Face" / name), crop_face(img, bbox))
        left = crop_eye(img, bbox)
        cv2.imwrite(str(gaze_hub_dir / "Image" / s / "Left" / name), left)
        cv2.imwrite(str(gaze_hub_dir / "Image" / s / "Right" / name), left)

        g = gaze_dir[i]
        gaze_str = ",".join(repr(float(v)) for v in g)
        yaw = np.arctan2(g[0], -g[2])
        pitch = np.arcsin(g[1])
        eye_str = f"{repr(float(yaw))},{repr(float(pitch))}"
        origin = f"{rec}/head/{int(pid[i]):06d}/{int(frame[i]):06d}.jpg"
        # 字段间恰好单个空格，路径一律正斜杠；Face 相对 Image/{split}/
        label_files[s].write(
            f"Face/{name} Left/{name} Right/{name} {origin} {gaze_str} {eye_str}\n")
        counts[s] += 1

    for s in SPLITS:
        label_files[s].close()

    # ---- 7. 统计 ----
    elapsed = time.time() - t0
    print("=" * 60)
    for s in SPLITS:
        print(f"  {s:<6}: {counts[s]:>7d}")
    total = sum(counts.values())
    print(f"  total : {total:>7d}")
    print(f"  跳过无 face 标注: {n_no_face} 行"
          + (f"；official 跳过 unused 有效行: {n_unused_valid}" if args.split_mode == "official" else ""))
    print(f"  输出目录: {out_root}")
    print(f"  耗时: {elapsed:.1f}s")
    print("[完成] 预处理结束")


if __name__ == "__main__":
    main()
