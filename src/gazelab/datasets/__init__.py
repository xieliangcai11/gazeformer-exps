"""gazelab.datasets 包导出层。

历史演进：
- 原单文件 gazelab/datasets.py（GazeHub 格式通用数据集）已迁移为本目录，
  其内容现位于 gaze360_dataset.py（面向 Gaze360/多数据集的 GazeHub 格式）。
- RGBDGaze 数据集位于 rgbdgaze.py。
- GazeCapture 数据集位于 gazecapture.py（大规模注视先验预训练语料）。
- 通用图像工具（裁剪/归一化/屏幕几何）统一在 common.py，供各数据集共用。

深度模态已移除（实验证明无正收益），故不再导出 depth_preprocess。

为了保持既有调用（`from gazelab.datasets import DatasetGaze360ByGazeHub`）不变，
这里把各数据集模块的公开符号统一 re-export。
"""

from gazelab.datasets.common import (
    crop_face_square,
    rgb_preprocess,
    screen_geometry,
)
from gazelab.datasets.gaze360_dataset import (
    DatasetMPIIFaceGazeByGazeHub,
    DatasetEyeDiapByGazeHub,
    DatasetGaze360ByGazeHub,
    DatasetETHXGazeByGazeHub,
)
from gazelab.datasets.gazecapture import GazeCaptureDataset
from gazelab.datasets.rgbdgaze import (
    RGBDGazeDataset,
    _to_tensor_label,
)

__all__ = [
    "DatasetMPIIFaceGazeByGazeHub",
    "DatasetEyeDiapByGazeHub",
    "DatasetGaze360ByGazeHub",
    "DatasetETHXGazeByGazeHub",
    "RGBDGazeDataset",
    "GazeCaptureDataset",
    "rgb_preprocess",
    "crop_face_square",
    "screen_geometry",
    "_to_tensor_label",
]