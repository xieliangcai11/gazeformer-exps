"""RGBDGaze 数据集类。

遵循统一返回协议，`__getitem__` 返回 `(edict(rgb=..., depth=...), gaze)`：

但 RGBDGaze 是双流（RGB + depth）+ 2D 屏幕坐标目标：
    rgb   = RGB 人脸裁剪图（走视觉流，ImageNet 归一化）
    depth = 深度人脸几何图（逆深度单通道 [1,H,W] 或灰度）
    gaze  = 归一化后的 2D 屏幕注视坐标 (x, y)

依赖 configs/rgbdgaze_config.py 中的路径与超参。
数据索引由 tools/data/rgbdgaze/preprocess.py 生成（index csv）。
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
    PROJECT_ROOT,
    IMAGE_SIZE,
    CROP_MARGIN,
    NORMALIZE_GAZE,
)


def _resolve_data_path(p: str) -> Path:
    """把 index 里的相对路径（相对项目根）解析成绝对路径；绝对路径则原样返回。

    兼容性：index.csv 存的是"相对项目根 + 前向斜杠"的路径，在任意机器上
    都能用当前项目的 PROJECT_ROOT 拼回正确位置。
    """
    p = Path(p)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _to_tensor_label(label):
    """把 2D 标签转成 float32 tensor。命名函数便于多进程 pickle。"""
    return torch.tensor(label, dtype=torch.float32)


# ---- 图像预处理（numpy BGR -> torch tensor） ----
# RGB 用 ImageNet 归一化（与 CLIP 编码器输入约定不同，CLIP 用自身 preprocess；
# 这里给出两套：clip 归一化（CLIP_PREPROCESS 在 configs.gaze360_config 中，需要 PIL）与
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


def inverse_depth_preprocess(img_bgr: np.ndarray) -> torch.Tensor:
    """Depth 人脸 -> 逆深度单通道 [1, H, W]（适配 DINOv2 新模型）。

    思路（对齐"深度图是灰度单值"这一实测事实）：
      1. 深度图三通道 R==G==B，取第 0 通道即灰度深度值。
      2. 转浮点并归一化到 [0,1]（相对距离，大=近）。
      3. 取逆深度 1/(z+eps)，放大近处几何信息（逆深度与视差成正比，是立体几何的标准范式）。
      4. 返回 [1, H, W] float32。
    """
    z = img_bgr[:, :, 0].astype(np.float32)          # [H,W] 深度灰度
    z = np.clip(z, 0.0, 255.0) / 255.0                # -> [0,1]
    inv = 1.0 / (z + 1e-3)                            # 逆深度，近处大
    # 归一化到 [0,1] 附近，避免量纲过大
    inv = np.clip(inv, 0.0, 255.0) / 255.0
    t = torch.from_numpy(inv).unsqueeze(0).float()    # [1,H,W]
    return t


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
        augment: bool = False,
        crop_jitter: float = 0.0,
        image_size: int = None,
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

        # ---- 数据增强（只应训练集开启） ----
        # augment     : 水平翻转(注视点x镜像) + 光度扰动
        # crop_jitter : 裁剪框随机抖动比例（相对bbox边长，如0.05=±5%）
        self.augment = augment
        self.crop_jitter = float(crop_jitter) if augment else 0.0
        self._rng = np.random.RandomState()
        # 输入分辨率（None = 使用配置的 IMAGE_SIZE）
        self.image_size = int(image_size) if image_size else IMAGE_SIZE

        # 每样本的"像素 -> 厘米"换算系数（来自 index，用于把预测误差转成物理 cm）
        try:
            self.cm_px = np.array(
                [[float(r["cm_px_x"]), float(r["cm_px_y"])] for r in self.rows],
                dtype=np.float32,
            )
            self.screen_size = np.array(
                [[float(r["screen_w"]), float(r["screen_h"])] for r in self.rows],
                dtype=np.int32,
            )
        except KeyError:
            # 旧版 index 无 cm 列时退化为近似（角标）
            self.cm_px = np.full((len(self.rows), 2), 0.0182, dtype=np.float32)
            self.screen_size = np.ones((len(self.rows), 2), dtype=np.int32)

    def __len__(self):
        return len(self.rows)

    def _make_face_pair(self, rgb_path, depth_path, bbox_xywh,
                        jitter=0.0, rng=None):
        """读 RGB + depth，用人脸 bbox 各裁一帧返回 dict

        重要：原始 RGB/depth 图是横向存储的，但作者 bbox 坐标基于
        逆时针旋转 90° 后的竖图。故先对两图都 np.rot90(img, 1)，再按
        作者 bbox 裁剪（bbox 即竖图坐标），depth 再按旋转后尺寸比例缩放。
        返回 (rgb_224, depth_224) 均 resize 成 IMAGE_SIZE。
        """
        rgb = cv2.imread(str(rgb_path))
        if rgb is None:
            raise FileNotFoundError(
                f"无法读取 RGB 图像: {rgb_path}（文件缺失或损坏？）")
        depth = cv2.imread(str(depth_path))
        if depth is None:
            raise FileNotFoundError(
                f"无法读取 depth 图像: {depth_path}（文件缺失或损坏？）")

        # 图片需逆时针旋转 90°（作者 bbox 基于竖图坐标系）
        rgb = np.rot90(rgb, 1)
        depth = np.rot90(depth, 1)

        rgb_face = self._crop(rgb, bbox_xywh, rgb.shape[1], rgb.shape[0],
                              jitter=jitter, rng=rng)
        # depth 分辨率缩放（相对旋转后的尺寸）
        dw, dh = depth.shape[1], depth.shape[0]
        sc = (dw / rgb.shape[1], dh / rgb.shape[0])
        bx, by, bw, bh = bbox_xywh
        depth_bbox = (bx * sc[0], by * sc[1], bw * sc[0], bh * sc[1])
        depth_face = self._crop(depth, depth_bbox, dw, dh,
                                jitter=jitter, rng=rng)

        if self.color_transform is not None:
            rgb_face = self.color_transform(rgb_face)
        if self.depth_transform is not None:
            depth_face = self.depth_transform(depth_face)
        return rgb_face, depth_face

    def _augment_pair(self, rgb_t, depth_t, gx):
        """对已裁剪归一化的 (rgb, depth, gaze_x) 应用增强，返回增强后三元组。

        水平翻转：人脸左右镜像时，注视点 x 必须镜像 gx -> 1-gx（正确性关键）。
        光度扰动：只作用于 rgb（亮度/对比度/通道级微扰），depth 不动。
        """
        if self._rng.random() < 0.5:
            rgb_t = torch.flip(rgb_t, dims=[2])
            depth_t = torch.flip(depth_t, dims=[2])
            gx = 1.0 - gx
        # 光度扰动：亮度 ±20%，对比度 ±20%，以 1.0 为中性
        rgb_t = rgb_t * self._rng.uniform(0.8, 1.2) + self._rng.uniform(-0.05, 0.05)
        rgb_t = torch.clamp(rgb_t, -2.5, 2.5)  # ImageNet 归一化后合理范围
        return rgb_t, depth_t, gx

    def _crop(self, img, bbox_xywh, W, H, jitter=0.0, rng=None):
        """按 bbox 裁人脸 (中心 + 边长*CROP_MARGIN)，resize 到 IMAGE_SIZE。

        jitter>0 时对裁剪中心与边长做随机抖动（数据增强用）：
            中心偏移 ±jitter*side，边长缩放 [1-jitter, 1+jitter]。
        """
        import math
        x, y, w, h = bbox_xywh
        cx, cy = x + w / 2, y + h / 2
        side = max(w, h) * CROP_MARGIN
        if jitter > 0 and rng is not None:
            cx += rng.uniform(-1, 1) * jitter * side
            cy += rng.uniform(-1, 1) * jitter * side
            side *= rng.uniform(1 - jitter, 1 + jitter)
        half = side / 2
        x0 = max(0, int(cx - half)); x1 = min(W, int(cx + half))
        y0 = max(0, int(cy - half)); y1 = min(H, int(cy + half))
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            crop = np.zeros((2, 2, 3), dtype=np.uint8)
        return cv2.resize(crop, (self.image_size, self.image_size))

    def __getitem__(self, idx):
        row = self.rows[idx]
        rgb_path = _resolve_data_path(row["rgb_path"])
        depth_path = _resolve_data_path(row["depth_path"])
        # bbox 放在 index 的附加列（preprocess 写入）
        bbox = (float(row["bbox_x"]), float(row["bbox_y"]),
                float(row["bbox_w"]), float(row["bbox_h"]))
        rgb_face, depth_face = self._make_face_pair(
            rgb_path, depth_path, bbox,
            jitter=self.crop_jitter, rng=self._rng)

        if NORMALIZE_GAZE:
            # gaze 归一化到 [0,1]（除以设备屏幕像素尺寸）
            gx = float(row["gaze_x"]) / float(row["screen_w"])
            gy = float(row["gaze_y"]) / float(row["screen_h"])
        else:
            gx = float(row["gaze_x"])
            gy = float(row["gaze_y"])

        # 数据增强（训练集）：翻转时注视点 x 必须同步镜像
        if self.augment:
            rgb_face, depth_face, gx = self._augment_pair(rgb_face, depth_face, gx)

        label = np.array([gx, gy], dtype=np.float32)

        # 每样本屏幕物理尺寸(cm)，供"训练时按验收几何算角度"使用（消融/可选）
        # 向后兼容：旧消费者只取 rgb/depth/label，忽略该额外字段
        screen_cm = torch.tensor(
            [float(row["screen_w"]) * float(row["cm_px_x"]),
             float(row["screen_h"]) * float(row["cm_px_y"])],
            dtype=torch.float32,
        )

        # IMU 重力向量（若 index 含该列则返回，供 use_imu 模型使用；否则零向量占位）
        try:
            imu = torch.tensor([float(row["imu_x"]), float(row["imu_y"]),
                                float(row["imu_z"])], dtype=torch.float32)
        except (KeyError, ValueError):
            imu = torch.zeros(3, dtype=torch.float32)

        return edict(rgb=rgb_face, depth=depth_face,
                     screen_cm=screen_cm, imu=imu), self.target_transform(label)