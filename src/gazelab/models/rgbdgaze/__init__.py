"""gazelab.models.rgbdgaze 子包：RGBDGaze 模型家族。

- rgbdgaze_dinov2.py : 主模型（DINOv2 冻结 RGB + 屏幕几何 token + BlockMoba 融合）
- vpt_model.py       : VPT 适配（冻结骨干 + 少量可学习 prompt token）
- calibrator.py      : 校准器（在骨干特征上学偏移修正）

深度模态已移除（实验证明无正收益），故旧版 CLIP+ResNet 双流基线
（rgbdgaze.py）及其训练/推理脚本（train_baseline.py / predict.py）一并删除。
"""

from gazelab.models.rgbdgaze.rgbdgaze_dinov2 import (
    RGBDGazeDINOv2,
    DINOV2_CKPT,
    DINOV2_MODEL,
)

__all__ = ["RGBDGazeDINOv2", "DINOV2_CKPT", "DINOV2_MODEL"]