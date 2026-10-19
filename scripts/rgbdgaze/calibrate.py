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

# ---- 路径引导：把项目根与 src/ 加入 sys.path ----
# 已改为「向上搜索项目根」而不是数 __file__ 的目录层级：后者在文件被移动到
# 不同深度后会静默失效（曾因 scripts/rgbdgaze/ablation.py -> scripts/train.py
# 的移动导致 `No module named 'gazelab'`）。
def _bootstrap() -> Path:
    """从本文件向上找到含 src/gazelab/__init__.py 的目录，并加入 sys.path。"""
    here = Path(__file__).resolve()
    for d in (here.parent, *here.parents):
        if (d / "src" / "gazelab" / "__init__.py").is_file():
            for p in (str(d), str(d / "src")):
                if p not in sys.path:
                    sys.path.insert(0, p)
            return d
    raise RuntimeError(
        f"找不到项目根：从 {here} 向上未发现 src/gazelab/__init__.py")


_PROJECT_ROOT = _bootstrap()

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from configs.paths import DEVICE as DEFAULT_DEVICE
from configs.rgbdgaze_config import RGBDGaze_INDEX_DIR
from gazelab.datasets.rgbdgaze import RGBDGazeDataset, rgb_preprocess
from gazelab.models.dinov2_gaze.rgbdgaze_dinov2 import RGBDGazeDINOv2
from gazelab.models.dinov2_gaze.calibrator import extract_features


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
def predict_batch(model, indices, dataset, device, batch_size=32,
                  use_imu=False, use_eye=False):
    """批量获取模型预测和 GT。use_imu/use_eye 必须与 ckpt 结构一致。"""
    preds, labels = {}, {}
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start+batch_size]
        faces, geoms, imus, el, er = [], [], [], [], []
        for idx in batch:
            inp, label = dataset[idx]
            faces.append(inp.rgb)
            geoms.append(inp.geom)
            if use_imu:
                imus.append(inp.imu)
            if use_eye:
                el.append(inp.eye_l)
                er.append(inp.eye_r)
            labels[idx] = label.numpy()
        f = torch.stack(faces).to(device).float()
        g = torch.stack(geoms).to(device).float()
        kw = {}
        if use_imu:
            kw["imu"] = torch.stack(imus).to(device).float()
        if use_eye:
            kw["eye_l"] = torch.stack(el).to(device).float()
            kw["eye_r"] = torch.stack(er).to(device).float()
        out = model(f, geom=g, **kw)
        for i, idx in enumerate(batch):
            preds[idx] = out[i].cpu().numpy()
    return preds, labels


@torch.no_grad()
def feature_batch(model, indices, dataset, device, batch_size=32,
                  use_imu=False, use_eye=False):
    """批量提取骨干特征。眼/IMU 支流开启时，把它们的特征一并拼接。"""
    feats = {}
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start+batch_size]
        faces, el, er, imus = [], [], [], []
        for idx in batch:
            inp, _ = dataset[idx]
            faces.append(inp.rgb)
            if use_eye:
                el.append(inp.eye_l)
                er.append(inp.eye_r)
            if use_imu:
                imus.append(inp.imu)
        f = torch.stack(faces).to(device).float()
        parts = [extract_features(model, f)]
        if use_eye:
            parts.append(extract_features(model, torch.stack(el).to(device).float()))
            parts.append(extract_features(model, torch.stack(er).to(device).float()))
        if use_imu:
            parts.append(torch.stack(imus).to(device).float().cpu())
        feat = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
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
    p.add_argument("--calib-split", choices=["head", "stride"], default="head",
                   help="校准帧划分：head=被试前 N%% 帧（历史做法；动画使注视点"
                        "时间非平稳 ⇒ 校准窗口分布≠测试分布）；stride=全 session "
                        "交错抽样（注视分布与测试集一致）")
    p.add_argument("--use-geom", action="store_true", default=False,
                   help="骨干含屏幕几何 token。必须与 ckpt 一致：模型类默认 "
                        "use_geom=True，而 nodepth/no-geom 的 ckpt 无 proj_geom，"
                        "不显式关闭会直接 load_state_dict 失败")
    p.add_argument("--no-geom", dest="use_geom", action="store_false",
                   help="兼容 dump_predictions 的写法（本脚本默认已是关闭 geom）")
    p.add_argument("--use-imu", action="store_true", help="骨干含 IMU token（须与 ckpt 一致）")
    p.add_argument("--use-eye", action="store_true", help="骨干含眼部支流（须与 ckpt 一致）")
    p.add_argument("--orient", choices=["raw", "rot90"], default="raw",
                   help="裁剪朝向，必须与 ckpt 训练时一致：raw=正立（当前代码默认）；"
                        "rot90=躺倒朝向（历史 ckpt：g5/g6/base224/g6_multi 等）。"
                        "用错朝向会让基线 EM 涨 2.5~3 倍且不会报错，务必核对打印的基线值")
    p.add_argument("--offset-mode", choices=["feat", "const", "affine"], default="feat",
                   help="偏移来源：feat=特征相关逐帧偏移（可记忆校准窗口的注视"
                        "位置）；const=校准集平均偏移（仅 2 参数，鲁棒）；"
                        "affine=校准集最小二乘仿射（6 参数，可修正幅度/收缩误差）")
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
    model = RGBDGazeDINOv2(unfreeze_last=12, img_size=args.resolution,
                           use_geom=args.use_geom, use_imu=args.use_imu,
                           use_eye=args.use_eye)
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
                              image_size=args.resolution,
                              orient=args.orient,
                              eye_crops=args.use_eye)
    emit(f"[calib] 测试集: {len(test_indices)} 样本")

    # 按被试分组
    groups = defaultdict(list)
    for idx in test_indices:
        r = rows[idx]
        groups[r["subject"]].append((frame_sort_key(r["rgb_path"]), idx))
    subject_frames = {sub: sorted(frames) for sub, frames in groups.items()}

    # ---- 无校准基准 ----
    emit("\n[baseline] 无校准基准:")
    preds_all, labels_all = predict_batch(model, test_indices, dataset, device,
                                          use_imu=args.use_imu, use_eye=args.use_eye)
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
            if args.calib_split == "head":
                calib_ids = [idx for _, idx in frames[:n_calib]]
                test_ids = [idx for _, idx in frames[n_calib:]]
            else:
                # 交错抽样：校准帧摊满整个 session ⇒ 注视分布与测试集一致
                stride = max(1, int(round(1.0 / ratio)))
                calib_ids = [idx for k, (_, idx) in enumerate(frames)
                             if k % stride == 0]
                test_ids = [idx for k, (_, idx) in enumerate(frames)
                            if k % stride != 0]
                n_calib = len(calib_ids)

            if len(test_ids) == 0:
                continue

            # 校准帧的特征 + 模型预测 + GT
            calib_feats = (feature_batch(model, calib_ids, dataset, device,
                                         use_imu=args.use_imu, use_eye=args.use_eye)
                           if args.offset_mode == "feat" else None)
            offsets_target = []
            for idx in calib_ids:
                r = rows[idx]
                gx = float(r["gaze_x"]) / float(r["screen_w"])
                gy = float(r["gaze_y"]) / float(r["screen_h"])
                mp = preds_all[idx]
                offsets_target.append([gx - mp[0], gy - mp[1]])
            offsets_target = np.array(offsets_target, dtype=np.float32)

            # 仿射模式：只用校准帧做最小二乘 (pred -> gt)，6 参数（2x2 + 偏置）
            affine_A = None
            if args.offset_mode == "affine":
                Pc = np.array([preds_all[i] for i in calib_ids], dtype=np.float64)
                Lc = np.array([[float(rows[i]["gaze_x"]) / float(rows[i]["screen_w"]),
                                float(rows[i]["gaze_y"]) / float(rows[i]["screen_h"])]
                               for i in calib_ids], dtype=np.float64)
                affine_A, *_ = np.linalg.lstsq(
                    np.hstack([Pc, np.ones((len(Pc), 1))]), Lc, rcond=None)
            calib_feat_tensor = (torch.stack([calib_feats[i] for i in calib_ids])
                                 if args.offset_mode == "feat" else None)

            # 训练偏移校准器（仅 feat 模式需要）
            calibrator = None
            if args.offset_mode == "feat":
                calibrator = OffsetCalibrator(calib_feat_tensor.shape[1],
                                              hidden=32).to(device)
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
            test_feats = (feature_batch(model, test_ids, dataset, device,
                                        use_imu=args.use_imu, use_eye=args.use_eye)
                          if args.offset_mode == "feat" else None)
            if calibrator is not None:
                calibrator.eval()
            mean_offset = np.mean(offsets_target, axis=0)
            sub_errors = []
            for idx in test_ids:
                r = rows[idx]
                gx = float(r["gaze_x"]) / float(r["screen_w"])
                gy = float(r["gaze_y"]) / float(r["screen_h"])
                sw_cm = float(r["screen_w"]) * float(r["cm_px_x"])
                sh_cm = float(r["screen_h"]) * float(r["cm_px_y"])
                mp = preds_all[idx]
                if args.offset_mode == "const":
                    offset = mean_offset          # 2 参数：校准集平均偏移
                    corrected_x, corrected_y = mp[0] + offset[0], mp[1] + offset[1]
                elif args.offset_mode == "affine":
                    cx, cy = np.append(mp, 1.0) @ affine_A
                    corrected_x, corrected_y = float(cx), float(cy)
                else:
                    with torch.no_grad():
                        feat = test_feats[idx].unsqueeze(0).to(device)
                        offset = calibrator(feat).cpu().numpy()[0]
                    corrected_x, corrected_y = mp[0] + offset[0], mp[1] + offset[1]
                e = math.hypot((corrected_x-gx)*sw_cm, (corrected_y-gy)*sh_cm)
                sub_errors.append(e)
                all_corrected_errors.append(e)
            emit(f"  {sub}: calib={n_calib} test={len(test_ids)} "
                 f"平均={np.mean(sub_errors):.2f}cm")

        overall = np.mean(all_corrected_errors) if all_corrected_errors else 0
        arr = np.array(all_corrected_errors)
        base_arr = np.array(base_errors)

        def tail(a):
            return (f"中位 {np.median(a):.3f} | >2cm {100 * (a > 2).mean():.2f}% | "
                    f">3cm {100 * (a > 3).mean():.2f}% | >5cm {100 * (a > 5).mean():.2f}%"
                    f" | max {a.max():.2f}")

        emit(f"[calib] 校准后整体: {overall:.2f}cm (n={len(all_corrected_errors)})")
        emit(f"[calib]   基线  : {base_arr.mean():.3f} | {tail(base_arr)}")
        emit(f"[calib]   校准后: {arr.mean():.3f} | {tail(arr)}")
        emit(f"[calib] 无校准: {baseline_cm:.2f}cm → 校准后 {overall:.2f}cm "
             f"| 提升 {(1-overall/baseline_cm)*100:.1f}%")

        result_file = run_dirs / (f"calibration_offset_{args.calib_split}_"
                                  f"{args.offset_mode}_r{int(ratio*100)}.json")
        result_file.write_text(json.dumps({
            "ratio": ratio, "split": args.calib_split, "mode": args.offset_mode,
            "baseline_cm": baseline_cm,
            "calibrated_cm": overall,
            "improvement_pct": (1 - overall / baseline_cm) * 100,
        }, indent=2))


if __name__ == "__main__":
    main()