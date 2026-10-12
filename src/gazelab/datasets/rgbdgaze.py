"""RGBDGaze 数据集类。

遵循统一返回协议，`__getitem__` 返回
`(edict(rgb=..., screen_cm=..., imu=..., geom=...), gaze)`：

    rgb       = RGB 人脸裁剪图（走视觉流，ImageNet 归一化）
    screen_cm = 该帧屏幕的物理宽高（厘米），供误差换算
    imu       = 重力向量（设备坐标系）
    geom      = 屏幕几何 [宽高比, log(物理宽cm), log(物理高cm)]，喂给模型的几何 token
    gaze      = 归一化后的 2D 屏幕注视坐标 (x, y)

深度模态已移除：实验证明其在 RGBDGaze 上无正收益。

依赖 configs/rgbdgaze_config.py 中的路径与超参。
数据索引由 tools/data/rgbdgaze/preprocess.py 生成（index csv）。
"""

from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from easydict import EasyDict as edict

from configs.rgbdgaze_config import (
    PROJECT_ROOT,
    IMAGE_SIZE,
    CROP_MARGIN,
    NORMALIZE_GAZE,
    AUG_SHIFT,
    AUG_SCALE,
    AUG_ROTATE,
    GEOM_RAND_PROB,
    GEOM_RAND_ASPECT,
    GEOM_RAND_WIDTH_CM,
    GEOM_RAND_HEIGHT_CM,
)
from gazelab.datasets.table_index import TableIndex
from gazelab.datasets.common import (
    crop_face_square,
    randomize_screen_geometry,
    rgb_preprocess,  # noqa: F401  (re-export，供既有调用方使用)
    screen_geometry,
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


# ---- 图像预处理 ----
# 通用实现（BGR->tensor、ImageNet 归一化、人脸裁剪、屏幕几何）已上提到
# gazelab.datasets.common，与 GazeCapture 共用同一输入约定。


class RGBDGazeDataset(Dataset):
    """RGBDGaze 数据集：从索引文件加载 (RGB 路径, 设备, 注视点)。

    索引文件格式（由 tools/data/rgbdgaze/preprocess.py 生成）：
        header: rgb_path, bbox_x, bbox_y, bbox_w, bbox_h, device, screen_w,
                screen_h, gaze_x, gaze_y, cm_px_x, cm_px_y, imu_x/y/z,
                subject, activity
    其中 gaze_x/gaze_y 是屏幕像素注视点，screen_w/screen_h 是该设备屏幕点数。
    """

    def __init__(
        self,
        index_csv: Path,
        color_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        augment: bool = False,
        aug_shift: float = AUG_SHIFT,
        aug_scale: float = AUG_SCALE,
        aug_rotate: float = AUG_ROTATE,
        geom_randomize: float = None,
        image_size: int = None,
    ):
        super().__init__()
        self.index_csv = Path(index_csv)
        # 列式索引：list(csv.DictReader) 会按 1.9KB/行 生成 dict，
        # 149 万行时约 2.4GB/进程，fork 给 DataLoader worker 后触发写时复制，
        # 是多卡训练卡顿的主因。TableIndex 取值语义与 DictReader 完全一致。
        self.rows = TableIndex(self.index_csv)

        self.color_transform = color_transform

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
        # geom_randomize: 几何域随机化触发概率（None=用 config 默认；仅在
        # augment 开启时生效，显式传 0 可关闭）
        self.geom_randomize = (float(GEOM_RAND_PROB) if geom_randomize is None
                               else float(geom_randomize))
        if not augment:
            self.geom_randomize = 0.0
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

    def _make_face(self, rgb_path, bbox_xywh, rng=None):
        """读 RGB 整帧，按人脸 bbox 裁出正方形人脸图并做归一化。

        重要：原始 RGB 图是横向存储的，但作者 bbox 坐标基于逆时针旋转 90°
        后的竖图。故先 np.rot90(img, 1)，再按作者 bbox 裁剪（bbox 即竖图坐标）。
        """
        rgb = cv2.imread(str(rgb_path))
        if rgb is None:
            raise FileNotFoundError(
                f"无法读取 RGB 图像: {rgb_path}（文件缺失或损坏？）")

        # 图片需逆时针旋转 90°（作者 bbox 基于竖图坐标系）
        rgb = np.rot90(rgb, 1)

        rgb_face = crop_face_square(rgb, bbox_xywh, CROP_MARGIN,
                                    self.image_size, shift=self.aug_shift,
                                    scale=self.aug_scale,
                                    rotate=self.aug_rotate, rng=rng)
        if self.color_transform is not None:
            rgb_face = self.color_transform(rgb_face)
        return rgb_face

    def _augment(self, rgb_t, gx):
        """对已裁剪归一化的 (rgb, gaze_x) 应用增强，返回增强后二元组。

        水平翻转：人脸左右镜像时，注视点 x 必须镜像 gx -> 1-gx（正确性关键）。
        光度扰动：亮度/对比度微扰。
        """
        if self._rng.random() < 0.5:
            rgb_t = torch.flip(rgb_t, dims=[2])
            gx = 1.0 - gx
        # 光度扰动：亮度 ±20%，对比度 ±20%，以 1.0 为中性
        rgb_t = rgb_t * self._rng.uniform(0.8, 1.2) + self._rng.uniform(-0.05, 0.05)
        rgb_t = torch.clamp(rgb_t, -2.5, 2.5)  # ImageNet 归一化后合理范围
        return rgb_t, gx

    def __getitem__(self, idx):
        row = self.rows[idx]
        rgb_path = _resolve_data_path(row["rgb_path"])
        # bbox 放在 index 的附加列（preprocess 写入）
        bbox = (float(row["bbox_x"]), float(row["bbox_y"]),
                float(row["bbox_w"]), float(row["bbox_h"]))
        rgb_face = self._make_face(rgb_path, bbox, rng=self._rng)

        if NORMALIZE_GAZE:
            # gaze 归一化到 [0,1]（除以设备屏幕点数）
            gx = float(row["gaze_x"]) / float(row["screen_w"])
            gy = float(row["gaze_y"]) / float(row["screen_h"])
        else:
            gx = float(row["gaze_x"])
            gy = float(row["gaze_y"])

        # 数据增强（训练集）：翻转时注视点 x 必须同步镜像
        if self.augment:
            rgb_face, gx = self._augment(rgb_face, gx)

        # 屏幕物理尺寸(cm) 与几何 token 输入
        cm_px_x = float(row["cm_px_x"])
        cm_px_y = float(row["cm_px_y"])
        w_cm = float(row["screen_w"]) * cm_px_x
        h_cm = float(row["screen_h"]) * cm_px_y
        screen_cm = torch.tensor([w_cm, h_cm], dtype=torch.float32)
        geom = screen_geometry(float(row["screen_w"]), float(row["screen_h"]),
                               cm_px_x, cm_px_y)
        # 几何域随机化（与 GazeCapture 同源，见 common.randomize_screen_geometry）
        if self.geom_randomize > 0.0 and self._rng.random() < self.geom_randomize:
            out = randomize_screen_geometry(
                gx, gy, w_cm, h_cm, self._rng,
                aspect_range=GEOM_RAND_ASPECT,
                width_range=GEOM_RAND_WIDTH_CM,
                height_range=GEOM_RAND_HEIGHT_CM)
            if out is not None:
                gx, gy, geom = out

        label = np.array([gx, gy], dtype=np.float32)

        # IMU 重力向量（若 index 含该列则返回，供 use_imu 模型使用；否则零向量占位）
        try:
            imu = torch.tensor([float(row["imu_x"]), float(row["imu_y"]),
                                float(row["imu_z"])], dtype=torch.float32)
        except (KeyError, ValueError):
            imu = torch.zeros(3, dtype=torch.float32)

        return edict(rgb=rgb_face, screen_cm=screen_cm, imu=imu, geom=geom), \
            self.target_transform(label)