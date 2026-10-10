"""gazelab.datasets 包导出层。

命名空间说明：
- gaze360_dataset.py : GazeHub 格式数据集（Gaze360/MPIIFaceGaze/EyeDiap/ETH-XGaze）
- rgbdgaze.py        : RGBDGaze 数据集
- gazecapture.py     : GazeCapture 数据集（大规模注视先验预训练语料）
- common.py          : 通用图像工具（裁剪/归一化/屏幕几何），各数据集共用

为什么 gaze360_dataset 用惰性导入（PEP 562 `__getattr__`）：
    gaze360_dataset → `from configs.gaze360_config import *` →
    该模块在**导入期**执行 `clip.load("ViT-B/32", device="cuda")`。
    若在本 __init__ 里直接 import，则任何 `from gazelab.datasets import
    RGBDGazeDataset` 都会顺带把 CLIP 加载到 **cuda:0**（模型尚未
    torch.cuda.set_device(local_rank)，DEVICE="cuda" 恒指 0 号卡）——
    torchrun 多卡下每个 rank 各加载一份，GPU0 白占 N x ~300MB。
    惰性化后，只有真正使用 Gaze360 系列数据集时才会触发 CLIP 加载。

公开 API 不变：`from gazelab.datasets import DatasetGaze360ByGazeHub` 仍可用。
"""

from gazelab.datasets.common import (
    crop_face_square,
    rgb_preprocess,
    screen_geometry,
)
from gazelab.datasets.gazecapture import GazeCaptureDataset
from gazelab.datasets.rgbdgaze import RGBDGazeDataset, _to_tensor_label

#: 需要惰性导入的名字 -> 所属子模块（这些子模块会触发 CLIP 加载）
_LAZY = {
    "DatasetMPIIFaceGazeByGazeHub": "gaze360_dataset",
    "DatasetEyeDiapByGazeHub": "gaze360_dataset",
    "DatasetGaze360ByGazeHub": "gaze360_dataset",
    "DatasetETHXGazeByGazeHub": "gaze360_dataset",
}

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


def __getattr__(name):
    """PEP 562：首次访问 Gaze360 系列数据集时才导入对应子模块。

    （这些子模块会 `from configs.gaze360_config import *`，从而加载 CLIP；
    见文件头说明。）
    """
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(f"gazelab.datasets.{mod}"), name)


def __dir__():
    return sorted(list(globals()) + list(_LAZY))
