"""gazelab.models.dinov2_gaze 子包：DINOv2 注视回归模型家族。

**数据集无关**：同一套模型同时用于 RGBDGaze（域内）与 GazeCapture（大规模预训练），
差异只在数据集构造与训练超参（见 scripts/train.py 的 --dataset）。

- rgbdgaze_dinov2.py : 主模型（DINOv2 冻结/部分微调 + 屏幕几何 token + BlockMoba 融合）
- vpt_model.py       : VPT 适配（冻结骨干 + 少量可学习 prompt token）
- calibrator.py      : 校准器（在骨干特征上学偏移修正）
- tta.py             : 测试时自适应

深度模态已移除（实验证明无正收益），故旧版 CLIP+ResNet 双流基线
（rgbdgaze.py）及其训练/推理脚本（train_baseline.py / predict.py）一并删除。
"""

from gazelab.models.dinov2_gaze.rgbdgaze_dinov2 import (
    RGBDGazeDINOv2,
    DINOV2_CKPT,
    DINOV2_MODEL,
    load_state_dict_any_res,
)

__all__ = ["RGBDGazeDINOv2", "DINOV2_CKPT", "DINOV2_MODEL",
           "load_state_dict_any_res"]