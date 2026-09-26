"""RGBDGaze 专用模型：双流（RGB + Depth）2D 屏幕注视回归。

设计目标：最大化复用 gazelab 的 CLIP 语义先验思想，同时适配
RGBDGaze 的数据特性（RGB 人脸 + Depth 人脸 + 2D 屏幕坐标回归）。

流设计：
    RGB 流（face）   → 冻结 CLIP 编码器 + 文本属性检索（复用现有语义先验）
    Depth 流（depth）→ ResNet-50 编码几何特征
    融合            → 拼接 → MLP 回归 2D 屏幕坐标 (x, y)

训练目标：2D 屏幕坐标（MSE），输入为 RGB + Depth 人脸裁剪图。
"""

import copy
import torch
import torch.nn as nn
from torchvision import models
import timm
from torchvision.models.feature_extraction import create_feature_extractor

from configs.config import CLIP_MODEL, CNN_MODEL, DEVICE
import clip


class RGBDGazeModel(nn.Module):
    """RGBD 双流 2D 注视回归模型。

    forward(face, depth) -> 2D 注视点 (B, 2)
      face  : [B, 3, H, W] RGB 人脸（CLIP 预处理）
      depth : [B, 3, H, W] depth 人脸（CNN 预处理，3 通道伪彩色或灰度复制）
    """

    def __init__(
        self,
        gaze_dim: int = 2,
        frozen_clip: bool = True,
    ):
        super().__init__()

        # ---------- CLIP 语义流（复用现有 text 属性检索） ----------
        # 与 GEWithCLIPModel 相同的文本提示
        self.illumination_texts = ["a face with bright light",
                                   "a face with low light", "a face with shadows"]
        self.headpose_texts = ["a frontal face", "a profile face"]
        self.background_texts = ["a face on bright background",
                                 "a face on dark background"]
        self.label_texts = [
            "A photo of a face looking left",
            "A photo of a face looking upper left",
            "A photo of a face looking up",
            "A photo of a face looking upper right",
            "A photo of a face looking right",
            "A photo of a face looking lower right",
            "A photo of a face looking down",
            "A photo of a face looking lower left",
        ]

        # 文本 tokenize + 预计算属性特征（不可学习，构造期一次性算好）
        illum_tokens = clip.tokenize(self.illumination_texts).to(DEVICE)
        headpose_tokens = clip.tokenize(self.headpose_texts).to(DEVICE)
        bg_tokens = clip.tokenize(self.background_texts).to(DEVICE)
        self.label_tokens = clip.tokenize(self.label_texts).to(DEVICE)

        clip_tmp = copy.deepcopy(CLIP_MODEL)
        clip_tmp.eval()
        with torch.no_grad():
            self.illum_feats = clip_tmp.encode_text(illum_tokens)      # [3,512]
            self.head_feats = clip_tmp.encode_text(headpose_tokens)    # [2,512]
            self.bg_feats = clip_tmp.encode_text(bg_tokens)            # [2,512]
            self.illum_norm = nn.functional.normalize(self.illum_feats, dim=-1)
            self.head_norm = nn.functional.normalize(self.head_feats, dim=-1)
            self.bg_norm = nn.functional.normalize(self.bg_feats, dim=-1)
        del clip_tmp

        self.model = CLIP_MODEL
        self.encoder_i = CLIP_MODEL.encode_image
        self.encoder_t2 = CLIP_MODEL.encode_text
        self.logit_scale = CLIP_MODEL.logit_scale

        if frozen_clip:
            # 冻结 CLIP 主体（含文本/图像编码器）
            for p in self.model.parameters():
                p.requires_grad = False
            # 但让 logit_scale 仍可学习（可选）

        # ---------- Depth 流：ResNet-50（复用 CNN_MODEL） ----------
        if CNN_MODEL == "ResNet-50":
            dnn = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            return_nodes = {"layer4": "features"}
            self.depth_fe = create_feature_extractor(dnn, return_nodes=return_nodes)
            depth_feats_dim = 2048
        elif CNN_MODEL == "ResNet-18":
            dnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            return_nodes = {"layer4": "features"}
            self.depth_fe = create_feature_extractor(dnn, return_nodes=return_nodes)
            depth_feats_dim = 512
        else:
            dnn = timm.create_model("edgenext_small", pretrained=True)
            depth_feats_dim = dnn.head.fc.in_features
            dnn.head.fc = nn.Identity()
            self.depth_fe = dnn

        self.depth_feats_dim = depth_feats_dim  # ResNet50 layer4: 2048

        # ---------- 融合 regressor：feature_1 + feature_2 + depth → 2D ----------
        # feature_1 / feature_2 各 512 维，depth 特征图先全局池化成 1 维向量
        #（在 forward 里做自适应平均池化）
        fused_dim = 512 + 512 + depth_feats_dim
        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, gaze_dim),   # 输出 2D 屏幕坐标
        )

    def forward_rgb_features(self, face):
        """RGB 流的语义特征（feature_1 + feature_2），复用 CLIP 属性检索。"""
        label_feats = self.encoder_t2(self.label_tokens)       # [8,512]
        img_feats = self.encoder_i(face)                       # [B,512]
        img_norm = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-2)
        label_norm = label_feats / (label_feats.norm(dim=-1, keepdim=True) + 1e-2)

        sim_illum = self.logit_scale.exp() * img_norm @ self.illum_norm.T
        sim_head = self.logit_scale.exp() * img_norm @ self.head_norm.T
        sim_bg = self.logit_scale.exp() * img_norm @ self.bg_norm.T
        sim_label = self.logit_scale.exp() * img_norm @ label_norm.T

        idx_illum = sim_illum.argmax(-1)
        idx_head = sim_head.argmax(-1)
        idx_bg = sim_bg.argmax(-1)
        idx_label = sim_label.argmax(-1)

        selected_illum = self.illum_feats[idx_illum]
        selected_head = self.head_feats[idx_head]
        selected_bg = self.bg_feats[idx_bg]
        selected_label = label_feats[idx_label]

        feature_1 = img_feats + selected_illum + selected_head + selected_bg
        feature_1 = nn.functional.normalize(feature_1, dim=-1)
        feature_2 = img_feats + selected_label
        feature_2 = nn.functional.normalize(feature_2, dim=-1)
        return feature_1, feature_2

    def forward_depth_features(self, depth):
        """Depth 流的几何特征：ResNet layer4 全局池化为向量 [B, C]"""
        feat_map = self.depth_fe(depth)["features"]   # [B, C, H, W]
        # 自适应平均池化 → [B, C]
        pooled = nn.functional.adaptive_avg_pool2d(feat_map, 1).flatten(1)
        return pooled

    def forward(self, face, depth):
        f1, f2 = self.forward_rgb_features(face)
        dfeat = self.forward_depth_features(depth)
        fused = torch.cat([f1, f2, dfeat], dim=-1)
        gaze = self.regressor(fused)   # [B,2]
        return gaze