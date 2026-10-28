"""gazelab.models.dinov2_gaze 子包：DINOv2 注视回归模型家族。

**数据集无关**：同一套模型同时用于 RGBDGaze（域内）与 GazeCapture（大规模预训练），
差异只在数据集构造与训练超参（见 scripts/train.py 的 --dataset）。

- rgbdgaze_dinov2.py : 主模型（DINOv2 冻结/部分微调 + 屏幕几何 token + BlockMoba 融合）

深度模态已移除（实验证明无正收益）。个性化/校准在输出层完成，见
scripts/calibration/。
"""

from gazelab.models.dinov2_gaze.rgbdgaze_dinov2 import (
    RGBDGazeDINOv2,
    DINOV2_CKPT,
    DINOV2_MODEL,
    load_state_dict_any_res,
)

__all__ = ["RGBDGazeDINOv2", "DINOV2_CKPT", "DINOV2_MODEL",
           "load_state_dict_any_res"]