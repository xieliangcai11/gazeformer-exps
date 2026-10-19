"""gazelab.datasets 包导出层。

命名空间说明：
- rgbdgaze.py    : RGBDGaze 数据集（域内：36 名被试、2D 屏幕坐标）
- gazecapture.py : GazeCapture 数据集（大规模注视先验预训练语料，~149 万帧）
- common.py      : 通用图像工具（裁剪/归一化/屏幕几何），各数据集共用
- table_index.py : 索引 CSV 的惰性取值工具（避免逐行建 dict 的启动开销）

Gaze360 系列（3D gaze、GazeHub 格式）已按要求移除；如需恢复请从 git 历史取回。
"""

from gazelab.datasets.common import (
    crop_face_square,
    rgb_preprocess,
    screen_geometry,
)
from gazelab.datasets.gazecapture import GazeCaptureDataset
from gazelab.datasets.rgbdgaze import RGBDGazeDataset, _to_tensor_label

__all__ = [
    "RGBDGazeDataset",
    "GazeCaptureDataset",
    "rgb_preprocess",
    "crop_face_square",
    "screen_geometry",
    "_to_tensor_label",
]
