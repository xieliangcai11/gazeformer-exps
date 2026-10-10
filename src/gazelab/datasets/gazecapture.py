"""GazeCapture 数据集类。

用途：作为大规模"注视先验"预训练语料（1474 名被试），在 RGBDGaze 微调之前
先让模型学会"跨用户的通用注视映射"，从而解决 RGBDGaze 仅 36 名训练被试导致
的对新用户泛化不足问题。

返回协议与 RGBDGazeDataset 保持一致（便于复用同一套训练脚本/模型）：
    __getitem__ -> (edict(rgb=..., screen_cm=..., imu=..., geom=..., face_grid=...), gaze)

其中：
    rgb       [3, S, S]  RGB 人脸裁剪图（ImageNet 归一化）
    screen_cm [2]        该帧屏幕的物理宽高（厘米），供误差换算
    imu       [3]        重力向量（设备坐标系），来自 motion.json
    geom      [3]        屏幕几何 [宽高比, log(物理宽cm), log(物理高cm)]，喂给模型的几何 token
    face_grid [4]        人脸在 25x25 网格中的 (x, y, w, h)（iTracker 风格输入）
    gaze      [2]        归一化 2D 屏幕注视点 (x, y) ∈ [0,1]

edict 内全部为 Tensor，可直接走 torch 的 default_collate，无需自定义 collate_fn。

索引由 tools/data/gazecapture/preprocess.py 生成。
"""

import csv
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
from torch.utils.data import Dataset

from configs.gazecapture_config import (
    GAZECAPTURE_INDEX_DIR,
    PROJECT_ROOT,
    IMAGE_SIZE,
    CROP_MARGIN,
    NORMALIZE_GAZE,
    AUG_SHIFT,
    AUG_SCALE,
    AUG_ROTATE,
)
from gazelab.datasets.common import (
    crop_face_square,
    rgb_preprocess,  # noqa: F401  (re-export，供既有调用方使用)
    screen_geometry,
)


def _resolve_data_path(p: str) -> Path:
    """把索引里的相对路径（相对项目根）解析成绝对路径；绝对路径原样返回。"""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _to_tensor_label(label):
    """把 2D 标签转成 float32 tensor。命名函数便于多进程 pickle。"""
    return torch.tensor(label, dtype=torch.float32)


class GazeCaptureDataset(Dataset):
    """GazeCapture：从索引 CSV 读取 (整帧路径, 人脸框, 屏幕注视点, 设备几何, IMU)。

    索引列（tools/data/gazecapture/preprocess.py 写出）：
        frame_path, bbox_x, bbox_y, bbox_w, bbox_h, device, screen_w, screen_h,
        gaze_x, gaze_y, cm_px_x, cm_px_y, imu_x, imu_y, imu_z, subject,
        orientation, split, face_grid_x, face_grid_y, face_grid_w, face_grid_h,
        dot_num, frame_index
    其中 gaze_x/gaze_y 是屏幕点，screen_w/screen_h 是该帧屏幕点数。

    坐标系：图像与标签在所有 orientation 下天然对齐，**不做旋转**，直接按
    bbox 裁剪（结论与实测依据见 preprocess.py 头部说明）。
    """

    def __init__(
        self,
        index_csv: Path = None,
        color_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        augment: bool = False,
        aug_shift: float = AUG_SHIFT,
        aug_scale: float = AUG_SCALE,
        aug_rotate: float = AUG_ROTATE,
        image_size: int = None,
    ):
        super().__init__()
        self.index_csv = Path(index_csv) if index_csv else GAZECAPTURE_INDEX_DIR / "index.csv"
        with open(self.index_csv, encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

        # 默认用通用的 ImageNet 归一化（与 RGBDGaze 一致，保证预训练/微调同分布）
        self.color_transform = color_transform if color_transform is not None else rgb_preprocess
        self.target_transform = (
            target_transform if target_transform is not None else _to_tensor_label
        )

        # ---- 数据增强（只应训练集开启） ----
        # augment    : 水平翻转(注视点 x 镜像) + 光度扰动
        # aug_shift  : 裁剪中心随机偏移幅度（相对裁剪边长）
        # aug_scale  : 裁剪边长随机缩放幅度
        # aug_rotate : 随机旋转角度（度）
        # 这三个几何抖动用于抹平两数据集的人脸框分布差异（见 configs 注释）。
        self.augment = augment
        self.aug_shift = float(aug_shift) if augment else 0.0
        self.aug_scale = float(aug_scale) if augment else 0.0
        self.aug_rotate = float(aug_rotate) if augment else 0.0
        self._rng = np.random.RandomState()
        self.image_size = int(image_size) if image_size else IMAGE_SIZE

    def __len__(self):
        return len(self.rows)

    def _augment(self, rgb_t, gx):
        """水平翻转（注视点 x 必须同步镜像）+ 光度扰动（只作用于 RGB）。"""
        if self._rng.random() < 0.5:
            rgb_t = torch.flip(rgb_t, dims=[2])
            gx = 1.0 - gx
        rgb_t = rgb_t * self._rng.uniform(0.8, 1.2) + self._rng.uniform(-0.05, 0.05)
        rgb_t = torch.clamp(rgb_t, -2.5, 2.5)   # ImageNet 归一化后的合理范围
        return rgb_t, gx

    def __getitem__(self, idx):
        row = self.rows[idx]

        # ---- 读整帧 + 按人脸框裁正方形人脸图 ----
        frame_path = _resolve_data_path(row["frame_path"])
        img = cv2.imread(str(frame_path))
        if img is None:
            raise FileNotFoundError(
                f"无法读取 GazeCapture 帧: {frame_path}（文件缺失或损坏？）")
        bbox = (float(row["bbox_x"]), float(row["bbox_y"]),
                float(row["bbox_w"]), float(row["bbox_h"]))
        face = crop_face_square(img, bbox, CROP_MARGIN, self.image_size,
                                shift=self.aug_shift, scale=self.aug_scale,
                                rotate=self.aug_rotate, rng=self._rng)
        if self.color_transform is not None:
            face = self.color_transform(face)

        # ---- 标签：屏幕点归一化到 [0,1] ----
        if NORMALIZE_GAZE:
            gx = float(row["gaze_x"]) / float(row["screen_w"])
            gy = float(row["gaze_y"]) / float(row["screen_h"])
        else:
            gx = float(row["gaze_x"])
            gy = float(row["gaze_y"])

        if self.augment:
            face, gx = self._augment(face, gx)

        label = np.array([gx, gy], dtype=np.float32)

        # ---- 辅助信息 ----
        # 屏幕物理尺寸(cm)：由"每点厘米"乘屏幕点数得到（设备不在规格表时为 NaN）
        cm_px_x = float(row["cm_px_x"])
        cm_px_y = float(row["cm_px_y"])
        screen_cm = torch.tensor(
            [float(row["screen_w"]) * cm_px_x,
             float(row["screen_h"]) * cm_px_y],
            dtype=torch.float32,
        )
        # IMU 重力向量（preprocess 已按 DotNum+Time 从 motion.json 对齐；缺失为 0）
        imu = torch.tensor(
            [float(row["imu_x"]), float(row["imu_y"]), float(row["imu_z"])],
            dtype=torch.float32,
        )
        # 屏幕几何 token 输入（跨设备泛化的关键，见模型 docstring）
        geom = screen_geometry(float(row["screen_w"]), float(row["screen_h"]),
                               cm_px_x, cm_px_y)
        # 人脸在 25x25 网格中的位置（iTracker 风格输入特征）
        face_grid = torch.tensor(
            [float(row["face_grid_x"]), float(row["face_grid_y"]),
             float(row["face_grid_w"]), float(row["face_grid_h"])],
            dtype=torch.float32,
        )

        return edict(rgb=face, screen_cm=screen_cm, imu=imu, geom=geom,
                     face_grid=face_grid), self.target_transform(label)
