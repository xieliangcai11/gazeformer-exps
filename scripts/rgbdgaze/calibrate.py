"""RGBDGaze 一次校准实验（per-user one-time calibration）。

流程：
    1. 加载骨干模型（g5_nodepth_448_aug, test 1.68cm）。
    2. 对每个测试被试：按时间顺序取前 calib_ratio% 帧作为校准集，
       剩余帧作为测试集。
    3. 用校准集的 (骨干特征, 真实注视点) 训练 Calibrator MLP。
    4. 用校准后的 MLP 在该被试剩余帧上评估。
    5. 对所有测试被试取平均 → 校准后整体性能。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from configs.gaze360_config import DEVICE as DEFAULT_DEVICE
from configs.rgbdgaze_config import RGBDGaze_INDEX_DIR
from gazelab.datasets.rgbdgaze import (
    RGBDGazeDataset, rgb_preprocess, inverse_depth_preprocess,
)
from gazelab.models.rgbdgaze.rgbdgaze_dinov2 import RGBDGazeDINOv2
from gazelab.models.rgbdgaze.calibrator import Calibrator, extract_features


class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i], self.labels[i]


def frame_sort_key(path_str):
    return int(path_str.replace("\\", "/").split("/")[-1].split(".")[0])


def build_subject_split(indices, rows, calib_ratio):
    """对每个测试被试，按时间顺序取前 calib_ratio% 做校准，剩余做测试。"""
    groups = defaultdict(list)
    for idx in indices:
        r = rows[idx]
        groups[r["subject"]].append((frame_sort_key(r["rgb_path"]), idx))
    result = {}
    for subj, frames in groups.items():
        frames.sort()
        n = len(frames)
        n_calib = max(1, int(n * calib_ratio))
        result[subj] = {
            "calib": [idx for _, idx in frames[:n_calib]],
            "test": [idx for _, idx in frames[n_calib:]],
        }
    return result


@torch.no_grad()
def extract_all_features(model, indices, dataset, device, batch_size=32):
    model.eval()
    all_feats, all_labels = [], []
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        faces, depths, labels = [], [], []
        for idx in batch_idx:
            inp, label = dataset[idx]
            faces.append(inp.rgb)
            depths.append(inp.depth)
            labels.append(label)
        f = torch.stack(faces).to(device).float()
        d = torch.stack(depths).to(device).float()
        feat = extract_features(model, f, d)
        all_feats.append(feat.cpu())
        all_labels.append(torch.stack(labels))
    return torch.cat(all_feats), torch.cat(all_labels)


def train_calibrator(feats, labels, device, epochs=200, lr=1e-3, hidden=128):
    calibrator = Calibrator(feature_dim=feats.shape[1], hidden=hidden).to(device)
    optimizer = optim.Adam(calibrator.parameters(), lr=lr)
    criterion = nn.MSELoss()
    feats, labels = feats.to(device), labels.to(device)
    dataset = FeatureDataset(feats, labels)
    loader = DataLoader(dataset, batch_size=min(64, len(dataset)), shuffle=True)
    calibrator.train()
    for epoch in range(epochs):
        for f_batch, l_batch in loader:
            optimizer.zero_grad()
            pred = calibrator(f_batch)
            loss = criterion(pred, l_batch)
            loss.backward()
            optimizer.step()
    calibrator.eval()
    return calibrator


def main():
    p = argparse.ArgumentParser(description="RGBDGaze 一次校准实验")
    p.add_argument("--backbone-ckpt", required=True)
    p.add_argument("--resolution", type=int, default=448)
    p.add_argument("--calib-ratios", type=float, nargs="+", default=[0.05, 0.10, 0.15])
    p.add_argument("--calib-epochs", type=int, default=200)
    p.add_argument("--calib-lr", type=float, default=1e-3)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = args.device if args.device != "auto" else DEFAULT_DEVICE
    run_dirs = Path("out/rgbdgaze/calibration")
    run_dirs.mkdir(parents=True, exist_ok=True)
    log_file = run_dirs / f"{datetime.now():%Y%m%d_%H%M%S}_calibration_log.txt"

    def emit(msg):
        print(msg)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    emit("=" * 60)
    emit(" 一次校准实验")
    emit(f" 骨干: {args.backbone_ckpt}")
    emit(f" 分辨率: {args.resolution}  校准比例: {args.calib_ratios}")
    emit("=" * 60)

    # 1) 加载骨干
    model = RGBDGazeDINOv2(use_depth=False, unfreeze_last=12, img_size=args.resolution)
    model.load_state_dict(torch.load(args.backbone_ckpt, map_location="cpu", weights_only=False))
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    emit("[calib] 骨干模型已加载并冻结")

    # 2) 读取测试集
    rows = list(csv.DictReader(open(RGBDGaze_INDEX_DIR / "index.csv", encoding="utf-8")))
    path_to_idx = {r["rgb_path"]: i for i, r in enumerate(rows)}
    test_indices = [path_to_idx[l.strip()] for l in
                    (RGBDGaze_INDEX_DIR / "test.txt").read_text(encoding="utf-8").splitlines()
                    if l.strip()]
    dataset = RGBDGazeDataset(RGBDGaze_INDEX_DIR / "index.csv",
                              color_transform=rgb_preprocess,
                              depth_transform=inverse_depth_preprocess,
                              image_size=args.resolution)
    emit(f"[calib] 测试集: {len(test_indices)} 样本")

    # 3) 提取特征（一次性）
    emit("[calib] 提取骨干特征...")
    t0 = time.time()
    all_feats, all_labels = extract_all_features(model, test_indices, dataset, device)
    emit(f"[calib] 特征提取完成: {all_feats.shape}, 耗时 {time.time()-t0:.0f}s")

    idx_to_pos = {idx: pos for pos, idx in enumerate(test_indices)}

    # 4) 对每个 calib_ratio 执行
    for ratio in args.calib_ratios:
        emit(f"\n{'='*60}")
        emit(f"[calib] 校准比例: {ratio*100:.0f}%")
        subject_splits = build_subject_split(test_indices, rows, ratio)
        total_err, total_count = 0.0, 0
        per_subject = []

        for subj, split in sorted(subject_splits.items()):
            calib_idx = split["calib"]
            test_idx = split["test"]
            calib_feats = all_feats[[idx_to_pos[i] for i in calib_idx]]
            calib_labels = all_labels[[idx_to_pos[i] for i in calib_idx]]
            test_feats = all_feats[[idx_to_pos[i] for i in test_idx]]
            test_labels = all_labels[[idx_to_pos[i] for i in test_idx]]

            calibrator = train_calibrator(calib_feats, calib_labels, device,
                                          epochs=args.calib_epochs, lr=args.calib_lr)
            with torch.no_grad():
                calibrator.eval()
                pred = calibrator(test_feats.to(device)).cpu()
            diff = pred - test_labels

            subj_rows = [rows[i] for i in test_idx]
            sw = np.array([float(r["screen_w"]) * float(r["cm_px_x"]) for r in subj_rows])
            sh = np.array([float(r["screen_h"]) * float(r["cm_px_y"]) for r in subj_rows])
            cm_err = np.sqrt((diff[:, 0].numpy() * sw) ** 2 + (diff[:, 1].numpy() * sh) ** 2)
            mean_cm = float(cm_err.mean())
            per_subject.append((subj, len(test_idx), mean_cm))
            total_err += cm_err.sum()
            total_count += len(test_idx)
            emit(f"  {subj}: test {len(test_idx)} 帧, 校准 {len(calib_idx)} 帧, 误差 {mean_cm:.2f}cm")

        overall = total_err / total_count
        emit(f"[calib] 校准后整体误差: {overall:.2f}cm (n={total_count})")
        emit(f"[calib] 无校准基准: 1.68cm → 校准后 {overall:.2f}cm "
             f"| {'有效' if overall < 1.68 else '无效'} | 提升 {(1-overall/1.68)*100:.1f}%")

        result_file = run_dirs / f"calibration_ratio{int(ratio*100)}.json"
        result_file.write_text(json.dumps({
            "ratio": ratio, "overall_cm": overall,
            "per_subject": {s: cm for s, _, cm in per_subject},
        }, ensure_ascii=False, indent=2))
        emit(f"[calib] 结果保存: {result_file}")


if __name__ == "__main__":
    main()
