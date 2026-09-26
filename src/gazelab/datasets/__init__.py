"""gazelab.datasets 包导出层。

历史演进：
- 原单文件 gazelab/datasets.py（GazeHub 格式通用数据集）已迁移为本目录，
  其内容现位于 gaze360_dataset.py（面向 Gaze360/多数据集的 GazeHub 格式）。
- 新增 RGBDGaze 数据集位于 rgbdgaze.py。

为了保持既有调用（`from gazelab.datasets import DatasetGaze360ByGazeHub`）不变，
这里把 gaze360_dataset 与 rgbdgaze 的公开符号统一 re-export。
"""

from gazelab.datasets.gaze360_dataset import (
    DatasetMPIIFaceGazeByGazeHub,
    DatasetEyeDiapByGazeHub,
    DatasetGaze360ByGazeHub,
    DatasetETHXGazeByGazeHub,
)
from gazelab.datasets.rgbdgaze import (
    RGBDGazeDataset,
    rgb_preprocess,
    depth_preprocess,
    _to_tensor_label,
)

__all__ = [
    "DatasetMPIIFaceGazeByGazeHub",
    "DatasetEyeDiapByGazeHub",
    "DatasetGaze360ByGazeHub",
    "DatasetETHXGazeByGazeHub",
    "RGBDGazeDataset",
    "rgb_preprocess",
    "depth_preprocess",
    "_to_tensor_label",
]