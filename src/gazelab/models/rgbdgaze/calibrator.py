"""RGBDGaze 一次校准模块（per-user calibration）。

设计参考 MAC-Gaze (2025)：
    骨干模型提取特征 → 轻量 MLP 校准器 → 修正后的注视点
    骨干冻结，只训练校准器（数万参数）
    校准数据 = 被试前 5/10/15% 帧；测试 = 剩余 85-95%

特征来源：融合 Transformer 的 CLS 输出（d_model=384 维），
    该向量已整合全脸视觉信息，是模型对当前帧的"最终内部表示"。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class Calibrator(nn.Module):
    """轻量个人校准器：学习"骨干特征 → 该用户修正后注视点"的映射。

    极小参数量（~5万），用被试少量校准数据训练，
    修正骨干模型的个人系统性偏差。
    """

    def __init__(self, feature_dim: int = 384, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: [B, feature_dim] → 校正后 2D 注视点 [B, 2]。"""
        return self.net(features)


def extract_features(model, face, depth, imu=None):
    """提取骨干模型的 CLS 特征（校准器的输入）。

    复用 RGBDGazeDINOv2 的 forward 逻辑，但返回 CLS 特征而非注视点。
    """
    tokens = model.rgb(face)
    cls = tokens[:, 0]
    patches = tokens[:, 1:]
    # 校准器特征来源：DINOv2 骨干的原始 CLS token（而非融合 Transformer 后的 CLS）。
    # 原因：融合后的 CLS 已被压缩为"全局摘要"，帧间差异极小（实测 norm 几乎恒定），
    # 导致校准器退化为输出恒定值。骨干原始 CLS 保留了逐帧细节变化，适合校准。
    # 将骨干 384 维投影到模型的 d_model 空间以保持维度一致。
    raw_cls_feat = model.proj_rgb_cls(cls)  # [B, d_model]

    # 同时拼接 patch tokens 的平均池化（提供空间聚合信息，增加帧间差异性）
    patch_mean = model.proj_rgb_patch(patches.mean(dim=1))  # [B, d_model]

    return torch.cat([raw_cls_feat, patch_mean], dim=-1)  # [B, 2*d_model]


def calibrator_feature_dim(model) -> int:
    """返回校准器输入的特征维度。"""
    return model.proj_rgb_cls.out_features * 2