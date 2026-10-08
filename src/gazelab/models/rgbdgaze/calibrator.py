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
    rgb_cls = model.proj_rgb_cls(cls).unsqueeze(1)
    rgb_patch = model.proj_rgb_patch(patches)

    if model.use_depth and model.depth_mode == "attn":
        d_patch = model.depth_attn(depth, patches)
        seq = torch.cat([rgb_cls, d_patch, rgb_patch], dim=1)
    elif model.use_depth and model.depth_mode == "token":
        seq = torch.cat([rgb_cls, model.depth_patch(depth), rgb_patch], dim=1)
    else:
        seq = torch.cat([rgb_cls, rgb_patch], dim=1)

    if model.use_imu:
        if imu is None:
            imu = face.new_zeros(face.size(0), 3)
        seq = torch.cat([model.proj_imu(imu).unsqueeze(1), seq], dim=1)

    B = face.size(0)
    x = torch.cat([model.cls_token.expand(B, -1, -1), seq], dim=1)
    for layer in model.layers:
        x = layer(x, cross_input=None, mask=None)
    return model.norm(x[:, 0, :])  # [B, d_model] — 校准器输入