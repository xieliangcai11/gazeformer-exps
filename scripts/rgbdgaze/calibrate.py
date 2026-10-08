"""RGBDGaze 一次校准实验 — 方案：学习"被试的系统性偏移修正"。

核心思想（区别于之前的失败版本）：
    之前的校准器试图从骨干特征重新映射到注视点（数据太少→退化为均值预测→崩溃）。
    现在改为：模型已有的预测已经很好，校准器只需要学习一个 **2 维修正偏移量**。
        corrected_gaze = model_prediction + offset(feature)
    校准器从一个 2 维回归问题（远比 768→2 的全映射简单）中学习。

流程：
    1. 加载骨干模型，对每个测试被试取前 calib_ratio% 帧做校准。
    2. 计算校准帧的 模型预测 与 GT 之间的偏移 → 训练一个轻量偏移预测器。
    3. 测试时：最终预测 = 模型预测 + 校准器预测的偏移。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
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

from configs.gaze360_config import DEVICE as DEFAULT_DEVICE
from configs.rgbdgaze_config import RGBDGaze_INDEX_DIR
from gazelab.datasets.rgbdgaze import (
    RGBDGazeDataset, rgb_preprocess, inverse_depth_preprocess,
)
from gazelab.models.rgbdgaze.rgbdgaze_dinov2 import RGBDGazeDINOv2
from gazelab.models.rgbdgaze.calibrator import extract_features


class OffsetCalibrator(nn.Module):
    """学习"骨干特征 → 2D偏移修正量"的轻量模块。"""

    def __init__(self, feature_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


def frame_sort_key(path_str):
    return int(path_str.replace("\\", "/").split("/")[-1].split(".")[0])


@torch.no_grad()
def predict_batch(model, indices, dataset, device, batch_size=32):
    """批量获取模型预测和 GT。"""
    preds, labels = {}, {}
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start+batch_size]
        faces, depths = [], []
        for idx in batch:
            inp, label = dataset[idx]
            faces.append(inp.rgb)
            depths.append(inp.depth)
            labels[idx] = label.numpy()
        f = torch.stack(faces).to(device).float()
        d = torch.stack(depths).to(device).float()
        out = model(f, d)
        for i, idx in enumerate(batch):
            preds[idx] = out[i].cpu().numpy()
    return preds, labels


@torch.no_grad()
def feature_batch(model, indices, dataset, device, batch_size=32):
    """批量提取骨干特征。"""
    feats = {}
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start+batch_size]
        faces, depths = [], []
        for idx in batch:
            inp, _ = dataset[idx]
            faces.append(inp.rgb)
            depths.append(inp.depth)
        f = torch.stack(faces).to(device).float()
        d = torch.stack(depths).to(device).float()
        feat = extract_features(model, f, d)
        for i, idx in enumerate(batch):
            feats[idx] = feat[i].cpu()
    return feats


def main():
    p = argparse.ArgumentParser(description="RGBDGaze 一次校准实验（偏移修正方案）")
    p.add_argument("--backbone-ckpt", required=True)
    p.add_argument("--resolution", type=int, default=448)
    p.add_argument("--calib-ratios", type=float, nargs="+", default=[0.05, 0.10, 0.15])
    p.add_argument("--calib-epochs", type=int, default=50)
    p.add_argument("--calib-lr", type=float, default=1e-4)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = args.device if args.device != "auto" else DEFAULT_DEVICE
    run_dirs = Path("out/rgbdgaze/calibration")
    run_dirs.mkdir(parents=True, exist_ok=True)
    log_file = run_dirs / f"{datetime.now():%Y%m%d_%H%M%S}_calibration_offset_log.txt"

    def emit(msg):
        print(msg)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    emit("=" * 60)
    emit(" 一次校准实验（偏移修正方案）")
    emit(f" 骨干: {args.backbone_ckpt}")
    emit(f" 分辨率: {args.resolution}")
    emit(f" 校准比例: {args.calib_ratios}")
    emit("=" * 60)

    # 加载模型
    model = RGBDGazeDINOv2(use_depth=False, unfreeze_last=12, img_size=args.resolution)
    model.load_state_dict(torch.load(args.backbone_ckpt, map_location="cpu", weights_only=False))
    model.to(device).eval()
    for param in model.parameters():
        param.requires_grad = False
    emit(f"[calib] 骨干模型已加载并冻结")

    # 读取测试集
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

    # 按被试分组
    groups = defaultdict(list)
    for idx in test_indices:
        r = rows[idx]
        groups[r["subject"]].append((frame_sort_key(r["rgb_path"]), idx))
    subject_frames = {sub: sorted(frames) for sub, frames in groups.items()}

    # ---- 无校准基准 ----
    emit("\n[baseline] 无校准基准:")
    preds_all, labels_all = predict_batch(model, test_indices, dataset, device)
    base_errors = []
    base_by_sub = defaultdict(list)
    for idx in test_indices:
        r = rows[idx]
        gx = float(r["gaze_x"]) / float(r["screen_w"])
        gy = float(r["gaze_y"]) / float(r["screen_h"])
        sw_cm = float(r["screen_w"]) * float(r["cm_px_x"])
        sh_cm = float(r["screen_h"]) * float(r["cm_px_y"])
        e = math.hypot((preds_all[idx][0]-gx)*sw_cm, (preds_all[idx][1]-gy)*sh_cm)
        base_errors.append(e)
        base_by_sub[r["subject"]].append(e)
    baseline_cm = np.mean(base_errors)
    emit(f"  test 平均: {baseline_cm:.2f}cm")
    for sub in sorted(base_by_sub.keys()):
        emit(f"  {sub}: {np.mean(base_by_sub[sub]):.2f}cm")

    # ---- 校准实验 ----
    for ratio in args.calib_ratios:
        emit(f"\n{'='*60}")
        emit(f"[calib] 校准比例: {ratio*100:.0f}%")
        all_corrected_errors = []

        for sub in sorted(subject_frames.keys()):
            frames = subject_frames[sub]
            n_calib = max(1, int(len(frames) * ratio))
            calib_ids = [idx for _, idx in frames[:n_calib]]
            test_ids = [idx for _, idx in frames[n_calib:]]

            if len(test_ids) == 0:
                continue

            # 校准帧的特征 + 模型预测 + GT
            calib_feats = feature_batch(model, calib_ids, dataset, device)
            offsets_target = []
            for idx in calib_ids:
                r = rows[idx]
                gx = float(r["gaze_x"]) / float(r["screen_w"])
                gy = float(r["gaze_y"]) / float(r["screen_h"])
                mp = preds_all[idx]
                offsets_target.append([gx - mp[0], gy - mp[1]])
            offsets_target = np.array(offsets_target, dtype=np.float32)
            calib_feat_tensor = torch.stack([calib_feats[i] for i in calib_ids])

            # 训练偏移校准器
            calibrator = OffsetCalibrator(calib_feat_tensor.shape[1], hidden=32).to(device)
            optimizer = optim.Adam(calibrator.parameters(), lr=args.calib_lr,
                                   weight_decay=0.01)
            criterion = nn.MSELoss()
            target_t = torch.tensor(offsets_target, device=device)
            feat_t = calib_feat_tensor.to(device)
            for epoch in range(args.calib_epochs):
                calibrator.train()
                optimizer.zero_grad()
                loss = criterion(calibrator(feat_t), target_t)
                loss.backward()
                optimizer.step()

            # 在测试帧上评估
            test_feats = feature_batch(model, test_ids, dataset, device)
            calibrator.eval()
            mean_offset = np.mean(offsets_target, axis=0)
            sub_errors = []
            for idx in test_ids:
                r = rows[idx]
                gx = float(r["gaze_x"]) / float(r["screen_w"])
                gy = float(r["gaze_y"]) / float(r["screen_h"])
                sw_cm = float(r["screen_w"]) * float(r["cm_px_x"])
                sh_cm = float(r["screen_h"]) * float(r["cm_px_y"])
                # 校准: 模型预测 + 特征级偏移修正
                feat = test_feats[idx].unsqueeze(0).to(device)
                with torch.no_grad():
                    offset = calibrator(feat).cpu().numpy()[0]
                mp = preds_all[idx]
                corrected_x = mp[0] + offset[0]
                corrected_y = mp[1] + offset[1]
                e = math.hypot((corrected_x-gx)*sw_cm, (corrected_y-gy)*sh_cm)
                sub_errors.append(e)
                all_corrected_errors.append(e)
            emit(f"  {sub}: calib={n_calib} test={len(test_ids)} "
                 f"平均={np.mean(sub_errors):.2f}cm")

        overall = np.mean(all_corrected_errors) if all_corrected_errors else 0
        emit(f"[calib] 校准后整体: {overall:.2f}cm (n={len(all_corrected_errors)})")
        emit(f"[calib] 无校准: {baseline_cm:.2f}cm → 校准后 {overall:.2f}cm "
             f"| 提升 {(1-overall/baseline_cm)*100:.1f}%")

        result_file = run_dirs / f"calibration_offset_r{int(ratio*100)}.json"
        result_file.write_text(json.dumps({
            "ratio": ratio, "baseline_cm": baseline_cm,
            "calibrated_cm": overall,
            "improvement_pct": (1 - overall / baseline_cm) * 100,
        }, indent=2))


if __name__ == "__main__":
    main()