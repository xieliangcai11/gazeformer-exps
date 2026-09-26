"""RGBDGaze 数据集类。

遵循 src/gazelab/datasets.py 的返回协议：
    __getitem__ 返回 (edict(face=..., other_face=...), label)

但 RGBDGaze 是双流（RGB + depth）+ 2D 屏幕坐标目标：
    face        = RGB 人脸裁剪图（走 CLIP/主视觉流）
    other_face  = depth 人脸裁剪图（走 CNN/几何流）
    label       = 归一化后的 2D 屏幕注视坐标 (x, y)

依赖 configs/rgbdgaze_config.py 中的路径与超参。
数据索引由 tools/data/rgbdgaze_preprocess.py 生成（index csv）。
"""

from pathlib import Path
import csv
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from easydict import EasyDict as edict

from configs.rgbdgaze_config import (
    RGBDGaze_INDEX_DIR,
    IMAGE_SIZE,
    CROP_MARGIN,
    NORMALIZE_GAZE,
)


def _to_tensor_label(label):
    """把 2D 标签转成 float32 tensor。命名函数便于多进程 pickle。"""
    return torch.tensor(label, dtype=torch.float32)


# ---- 图像预处理（numpy BGR -> torch tensor） ----
# RGB 用 ImageNet 归一化（与 CLIP 编码器输入约定不同，CLIP 用自身 preprocess；
# 这里给出两套：clip 归一化（CLIP_PREPROCESS 在 configs.config 中，需要 PIL）与
# CNN 风格。为避免依赖 PIL，这里用 numpy 版实现（ToTensor + Normalize）。

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _to_tensor_np(img_bgr: np.ndarray) -> torch.Tensor:
    """numpy BGR(H,W,3) uint8 -> tensor(C,H,W) float32 [0,1]，通道转 RGB。"""
    rgb = img_bgr[:, :, ::-1]  # BGR -> RGB
    t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0
    return t


def rgb_preprocess(img_bgr: np.ndarray) -> torch.Tensor:
    """RGB 人脸：转 RGB + ImageNet 归一化（适配 CLIP 体）。"""
    t = _to_tensor_np(img_bgr)
    mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)
    return (t - mean) / std


def depth_preprocess(img_bgr: np.ndarray) -> torch.Tensor:
    """Depth 人脸：转 RGB + ImageNet 归一化（适配 ResNet）。"""
    t = _to_tensor_np(img_bgr)
    mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)
    return (t - mean) / std


class RGBDGazeDataset(Dataset):
    """RGBDGaze 数据集：从索引文件加载 (RGB 路径, depth 路径, 设备, 注视点)。

    索引文件格式（由 preprocess 生成，用人脸 bbox 裁出统一的人脸图）：
        header: rgb_path, depth_path, device, screen_w, screen_h, gaze_x, gaze_y
    其中 gaze_x/gaze_y 是屏幕像素注视点，screen_w/screen_h 是该 iPhone 屏幕宽高。
    """

    def __init__(
        self,
        index_csv: Path,
        color_transform: Optional[Callable] = None,
        depth_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.index_csv = Path(index_csv)
        with open(self.index_csv, encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

        # RGB 与 depth 各有独立预处理
        self.color_transform = color_transform
        self.depth_transform = depth_transform

        self.target_transform = (
            target_transform if target_transform is not None else _to_tensor_label
        )

    def __len__(self):
        return len(self.rows)

    def _make_face_pair(self, rgb_path, depth_path, bbox_xywh):
        """读 RGB + depth，用人脸 bbox 各裁一帧返回 dict

        注：RGB 与 depth 都是原图，bbox 坐标是在 RGB 分辨率上的。
        depth 图分辨率不同，按比例缩放 bbox。
        返回 (rgb_224, depth_224) 均 resize 成 IMAGE_SIZE。
        """
        rgb = cv2.imread(str(rgb_path))
        depth = cv2.imread(str(depth_path))

        # 本轮先把裁剪逻辑放这里，供 preprocess 与推理复用同一份
        rgb_face = self._crop(rgb, bbox_xywh, rgb.shape[1], rgb.shape[0])
        # depth 分辨率缩放
        dw, dh = depth.shape[1], depth.shape[0]
        sc = (dw / rgb.shape[1], dh / rgb.shape[0])
        bx, by, bw, bh = bbox_xywh
        depth_bbox = (bx * sc[0], by * sc[1], bw * sc[0], bh * sc[1])
        depth_face = self._crop(depth, depth_bbox, dw, dh)

        if self.color_transform is not None:
            rgb_face = self.color_transform(rgb_face)
        if self.depth_transform is not None:
            depth_face = self.depth_transform(depth_face)
        return rgb_face, depth_face

    @staticmethod
    def _crop(img, bbox_xywh, W, H):
        """按 bbox 裁人脸 (中心 + 边长*CROP_MARGIN)，resize 到 IMAGE_SIZE。"""
        import math
        x, y, w, h = bbox_xywh
        cx, cy = x + w / 2, y + h / 2
        side = max(w, h) * CROP_MARGIN
        half = side / 2
        x0 = max(0, int(cx - half)); x1 = min(W, int(cx + half))
        y0 = max(0, int(cy - half)); y1 = min(H, int(cy + half))
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            crop = np.zeros((2, 2, 3), dtype=np.uint8)
        return cv2.resize(crop, (IMAGE_SIZE, IMAGE_SIZE))

    def __getitem__(self, idx):
        row = self.rows[idx]
        rgb_path = Path(row["rgb_path"])
        depth_path = Path(row["depth_path"])
        # bbox 放在 index 的附加列（preprocess 写入）
        bbox = (float(row["bbox_x"]), float(row["bbox_y"]),
                float(row["bbox_w"]), float(row["bbox_h"]))
        rgb_face, depth_face = self._make_face_pair(rgb_path, depth_path, bbox)

        if NORMALIZE_GAZE:
            # gaze 归一化到 [0,1]（除以设备屏幕像素尺寸）
            gx = float(row["gaze_x"]) / float(row["screen_w"])
            gy = float(row["gaze_y"]) / float(row["screen_h"])
            label = np.array([gx, gy], dtype=np.float32)
        else:
            label = np.array([float(row["gaze_x"]), float(row["gaze_y"])],
                             dtype=np.float32)

        return edict(face=rgb_face, other_face=depth_face), self.target_transform(label)