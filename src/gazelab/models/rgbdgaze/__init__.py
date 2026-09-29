"""gazelab.models.rgbdgaze 子包：RGBDGaze 模型家族。

- rgbdgaze.py           : 旧版基线（CLIP 语义 + ResNet 深度 + MLP 2D 回归）
- rgbdgaze_dinov2.py    : 新版（DINOv2 冻结 RGB + 逆深度 token + BlockMoba 融合）

为保持既有调用 `from gazelab.models.rgbdgaze import RGBDGazeModel` 不变，在此统一导出。
注意：导入本包会同时触发旧基线的 CLIP 依赖加载；若只要新模型可直连子模块。
"""

from gazelab.models.rgbdgaze.rgbdgaze import RGBDGazeModel
from gazelab.models.rgbdgaze.rgbdgaze_dinov2 import (
    RGBDGazeDINOv2,
    DINOV2_CKPT,
    DINOV2_MODEL,
)

__all__ = ["RGBDGazeModel", "RGBDGazeDINOv2", "DINOV2_CKPT", "DINOV2_MODEL"]