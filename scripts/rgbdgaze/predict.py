#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_rgbdgaze.py — RGBDGaze 单张样本推理：预测屏幕注视点并可视化

输入：一张 RGB 人脸图 + 一张 Depth 人脸图（同一时刻、同一人脸）。
流程：
  1. mediapipe 检测人脸框（在 RGB 图上）
  2. 按 bbox 从 RGB 与 Depth 各裁人脸，双流预处理
  3. RGBDGazeModel 预测屏幕注视点（2D 归一化坐标）
  4. 在手机屏幕示意上标注预测点，并打印坐标 / 物理cm / 角度

用法：
    python scripts/rgbdgaze/predict.py \
        --rgb 图片路径 --depth 深度图路径 \
        [--checkpoint 权重] [--out 输出png] [--screen-w 16 --screen-h 8]

说明：
  - 屏幕默认按 16x8cm 绘制（任务规格）；实际误差换算用每设备的 cm/px。
  - 若无 --depth 会报错（RGBDGaze 双流必需 depth）。
"""

import argparse
import sys
from pathlib import Path

# ---- 路径引导 ----
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (str(_PROJECT_ROOT), str(_PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2
import numpy as np
import torch

from configs.gaze360_config import DEVICE, experiment_dirs
from configs.rgbdgaze_config import IMAGE_SIZE, VIEWING_DISTANCE_CM
from gazelab.datasets.rgbdgaze import rgb_preprocess, depth_preprocess
from gazelab.models.rgbdgaze import RGBDGazeModel

CHECKPOINT_DEFAULT = str(
    experiment_dirs("rgbdgaze", "train")["checkpoint"] / "best.pt"
)


# ---------------------------------------------------------------------------
# 人脸检测（复用 mediapipe，提取 bbox）
# ---------------------------------------------------------------------------
def detect_face_box(image):
    """用 mediapipe 检测最大人脸，返回 (x,y,w,h) 或 None。"""
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from mediapipe import Image, ImageFormat
    base = python.BaseOptions(
        model_asset_path=str(_PROJECT_ROOT / "assets" / "face_landmarker.task"))
    landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(base_options=base, num_faces=1))
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    res = landmarker.detect(Image(image_format=ImageFormat.SRGB, data=rgb))
    landmarker.close()
    if not res.face_landmarks:
        return None
    lm = res.face_landmarks[0]
    h, w = image.shape[:2]
    xs = [lm[i].x * w for i in range(468)]
    ys = [lm[i].y * h for i in range(468)]
    x0, x1, y0, y1 = int(min(xs)), int(max(xs)), int(min(ys)), int(max(ys))
    return (x0, y0, x1 - x0, y1 - y0)


def crop_face(img, bbox_xywh, margin=1.2, size=IMAGE_SIZE):
    """按 bbox 裁人脸（中心+边长*margin），resize 到 size。"""
    x, y, w, h = bbox_xywh
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * margin
    half = side / 2
    X0 = max(0, int(cx - half)); X1 = min(img.shape[1], int(cx + half))
    Y0 = max(0, int(cy - half)); Y1 = min(img.shape[0], int(cy + half))
    c = img[Y0:Y1, X0:X1]
    if c.size == 0:
        c = np.zeros((2, 2, 3), dtype=np.uint8)
    return cv2.resize(c, (size, size))


# ---------------------------------------------------------------------------
# 可视化：在手机屏幕示意上标注注视点
# ---------------------------------------------------------------------------
def draw_screen(pred_xy, gt_xy=None, screen_cm=(16.0, 8.0), out_path="screen_pred.png"):
    """在手机屏幕示意(16x8cm比例)上画预测点/真值点。"""
    # 画布 480x240 (16:8 = 2:1)
    W, H = 480, 240
    canvas = np.full((H, W, 3), 255, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (W - 1, H - 1), (0, 0, 0), 2)

    def to_px(xy):
        return int(xy[0] * W), int(xy[1] * H)

    # 真值点（绿）
    if gt_xy is not None:
        gx, gy = to_px(gt_xy)
        cv2.circle(canvas, (gx, gy), 8, (0, 180, 0), -1)
        cv2.putText(canvas, "GT", (gx + 10, gy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 180, 0), 1)
    # 预测点（红）
    px, py = to_px(pred_xy)
    cv2.circle(canvas, (px, py), 8, (0, 0, 255), -1)
    cv2.putText(canvas, "Pred", (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 1)
    cv2.imwrite(out_path, canvas)
    print(f"[输出] 屏幕示意已保存 -> {out_path}")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def predict_pair(rgb_path, depth_path, ckpt_path, out_path, screen_cm,
                 gt_xy=None, device=None):
    device = device or DEVICE
    rgb = cv2.imread(str(rgb_path))
    depth = cv2.imread(str(depth_path))
    if rgb is None or depth is None:
        raise FileNotFoundError(f"读取失败: {rgb_path} / {depth_path}")

    # 1. 人脸检测
    box = detect_face_box(rgb)
    if box is None:
        raise RuntimeError("未检测到人脸，请换一张图或检查 mediapipe 模型")
    print(f"[检测] 人脸框: {box}")

    # 2. 双流裁剪（depth 按比例缩放 bbox）
    rgb_face = crop_face(rgb, box)
    sx = depth.shape[1] / rgb.shape[1]
    sy = depth.shape[0] / rgb.shape[0]
    depth_face = crop_face(depth, (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy))

    # 3. 预处理
    face_t = rgb_preprocess(rgb_face).unsqueeze(0).to(device)
    depth_t = depth_preprocess(depth_face).unsqueeze(0).to(device)

    # 4. 模型
    model = RGBDGazeModel().to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(sd)
    model.eval()

    # 5. 预测
    with torch.no_grad():
        pred = model(face_t, depth_t)[0].cpu().numpy()  # [2] 归一化坐标
    pred = np.clip(pred, 0, 1)

    # 6. 换算物理 cm 与角度
    px_cm = (pred[0] - 0.5) * screen_cm[0]
    py_cm = (pred[1] - 0.5) * screen_cm[1]
    ang = np.degrees(np.arctan2(np.hypot(px_cm, py_cm), VIEWING_DISTANCE_CM))

    print("=" * 52)
    print(f"预测屏幕坐标(归一化): ({pred[0]:.4f}, {pred[1]:.4f})")
    print(f"相对屏幕中心(cm)   : ({px_cm:.2f}, {py_cm:.2f})")
    print(f"注视角度(距30cm)    : {ang:.2f}°")
    if gt_xy is not None:
        err = np.hypot(pred[0] - gt_xy[0], pred[1] - gt_xy[1])
        print(f"真值坐标            : ({gt_xy[0]:.4f}, {gt_xy[1]:.4f})")
        print(f"归一化误差(L2)      : {err:.4f}")
    print("=" * 52)

    draw_screen(pred, gt_xy, screen_cm, out_path)


def main():
    p = argparse.ArgumentParser(description="RGBDGaze 单样本推理")
    p.add_argument("--rgb", required=True, help="RGB 人脸图路径")
    p.add_argument("--depth", required=True, help="Depth 人脸图路径")
    p.add_argument("--checkpoint", default=CHECKPOINT_DEFAULT, help="权重路径")
    p.add_argument("--out", default="screen_pred.png", help="输出屏幕示意 png")
    p.add_argument("--screen-w", type=float, default=16.0, help="屏幕物理宽(cm)")
    p.add_argument("--screen-h", type=float, default=8.0, help="屏幕物理高(cm)")
    p.add_argument("--gt", default=None,
                   help="可选真值归一化坐标，格式 x,y（用于对比标注）")
    args = p.parse_args()

    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"权重不存在: {args.checkpoint}")
    gt_xy = tuple(map(float, args.gt.split(","))) if args.gt else None
    predict_pair(args.rgb, args.depth, args.checkpoint, args.out,
                 (args.screen_w, args.screen_h), gt_xy)


if __name__ == "__main__":
    main()