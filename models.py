"""
models.py
=========
本文件定义了基于 CLIP 的 gaze（视线/注视方向）估计模型。

包含两个类：
    - GEWithCLIPModel      : 使用 CNN 提取的全局特征（fc 前的展平向量）作为 feature_3。
    - GEWithCLIPModel_zhao : 使用 CNN 的中间特征图（layer4 输出，形如 [B,C,H,W]）作为 feature_3，
                             以便后续把特征图切成 patch token 送入 Transformer。

核心思想（CLIP 语义先验 + 多特征融合）：
    1. 冻结的 CLIP 视觉编码器 encode_image(face) 得到图像语义向量 img_feats。
    2. 用 CLIP 文本编码器把若干"属性文本提示"编码成向量，并按余弦相似度选出与当前图像最匹配的属性向量：
        - illumination（光照）：亮光 / 暗光 / 阴影
        - headpose（头部姿态）：正面 / 侧面
        - background（背景）：亮背景 / 暗背景
        - label（视线方向）：8 个方向描述
    3. 把选出的"无关属性"（光照/头姿/背景）与图像特征相加 -> feature_1（上下文补偿特征）。
    4. 把选出的"视线方向标签"特征与图像特征相加 -> feature_2（任务对齐特征）。
    5. CNN 提取视觉特征 -> feature_3。
    6. 拼接 feature_1、feature_2、feature_3，送入融合 MLP 回归 3 维 gaze 向量。

注：在本项目 train.py 中，实际训练时使用 GEWithCLIPModel_zhao 来生成中间特征，
    而最终的 3D gaze 回归由 transformer_models.py 中的 TransformerDeepSeek_gaze 完成。
"""

import clip
import torch
import torch.nn as nn
from torchvision import models
import copy
import timm
from config import *
from torchvision.models.feature_extraction import create_feature_extractor


class GEWithCLIPModel(nn.Module):
    """
    基于 CLIP + CNN 的 gaze 估计模型（feature_3 使用 CNN 的全局向量特征）。

    Args:
        irrelevant_feats_dim: 无关特征（feature_1）维度，默认 512（CLIP 嵌入维度）。
        relevant_feats_dim  : 相关特征（feature_2）维度，默认 512。
    """

    def __init__(
        self,
        irrelevant_feats_dim=512,
        relevant_feats_dim=512,
    ):
        super().__init__()

        # ---------- 定义各类文本提示（prompt） ----------
        # 光照属性文本（3 类）
        self.illumination_texts = [
            "a face with bright light",   # 亮光
            "a face with low light",      # 弱光
            "a face with shadows",        # 阴影
        ]
        # 头部姿态文本（2 类）
        self.headpose_texts = [
            "a frontal face",             # 正面
            "a profile face",             # 侧面
        ]
        # 背景文本（2 类）
        self.background_texts = [
            "a face on bright background",  # 亮背景
            "a face on dark background",    # 暗背景
        ]
        # 视线方向标签文本（8 个方向）
        self.label_texts = [
            "A photo of a face looking left",          # 向左看
            "A photo of a face looking upper left",    # 向左上看
            "A photo of a face looking up",            # 向上看
            "A photo of a face looking upper right",   # 向右上看
            "A photo of a face looking right",         # 向右看
            "A photo of a face looking lower right",   # 向右下看
            "A photo of a face looking down",          # 向下看
            "A photo of a face looking lower left",    # 向左下看
        ]

        # ---------- 用 CLIP 的 tokenizer 把文本转为 token id ----------
        illum_tokens = clip.tokenize(self.illumination_texts).to(DEVICE)
        headpose_tokens = clip.tokenize(self.headpose_texts).to(DEVICE)
        bg_tokens = clip.tokenize(self.background_texts).to(DEVICE)
        self.label_tokens = clip.tokenize(self.label_texts).to(DEVICE)

        # 深拷贝一份 CLIP 模型，仅用于预计算文本属性特征（避免影响主模型）
        clip_model_1 = copy.deepcopy(CLIP_MODEL)
        clip_model_1.eval()
        with torch.no_grad():
            # 用文本编码器（encoder_t1）预计算三类无关属性的特征向量
            self.illum_feats = clip_model_1.encode_text(illum_tokens)      # [3, 512]
            self.head_feats = clip_model_1.encode_text(headpose_tokens)    # [2, 512]
            self.bg_feats = clip_model_1.encode_text(bg_tokens)            # [2, 512]

            # 对上述特征做 L2 归一化，便于后续计算余弦相似度
            self.illum_norm = self.illum_feats / self.illum_feats.norm(
                dim=-1, keepdim=True
            )
            self.head_norm = self.head_feats / self.head_feats.norm(
                dim=-1, keepdim=True
            )
            self.bg_norm = self.bg_feats / self.bg_feats.norm(dim=-1, keepdim=True)
        # 释放临时 CLIP 模型，清空显存
        del clip_model_1
        torch.cuda.empty_cache()

        # ---------- 主 CLIP 模型及其编码器 ----------
        self.model = CLIP_MODEL
        self.encoder_i = CLIP_MODEL.encode_image   # 图像编码器
        self.encoder_t2 = CLIP_MODEL.encode_text   # 文本编码器

        # ---------- CNN 主干网络：用于提取视觉几何特征 ----------
        if CNN_MODEL == "ResNet-50":
            # ResNet50：默认期望输入形状 (batch_size, 3, 224, 224)
            main_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            main_model_feats_dim = main_model.fc.in_features  # 2048
            main_model.fc = nn.Identity()                     # 去掉分类头，保留特征
        elif CNN_MODEL == "ResNet-18":
            main_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            main_model_feats_dim = main_model.fc.in_features  # 512
            main_model.fc = nn.Identity()
        elif CNN_MODEL == "EdgeNeXt-Small":
            main_model = timm.create_model("edgenext_small", pretrained=True)
            main_model_feats_dim = main_model.head.fc.in_features
            main_model.head.fc = nn.Identity()
        self.main_model = main_model

        # ---------- 融合层 ----------
        # 融合维度 = 无关特征 + 相关特征 + CNN 特征
        fused_dim = irrelevant_feats_dim + relevant_feats_dim + main_model_feats_dim
        # 简单 MLP：fused_dim -> 256 -> ReLU -> 3（输出 3D gaze 向量）
        self.fuse_model = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.ReLU(), nn.Linear(256, 3)  # 3D gaze output
        )
        # 复用 CLIP 的可学习温度系数
        self.logit_scale = CLIP_MODEL.logit_scale

    def forward(
        self,
        face,
        other_face,
    ):
        """
        Args:
            face       : 经过 CLIP 预处理的原始人脸图像。
            other_face : 经过 CNN 预处理的人脸图像。

        Returns:
            gaze_pred  : [B, 3] 预测的 3D gaze 向量。
            sim_label  : [B, 8] 图像与 8 个方向文本的相似度 logits。
            feature_1  : 上下文补偿特征（图像 + 光照/头姿/背景）。
            feature_2  : 任务对齐特征（图像 + 视线方向标签）。
        """
        # ---------- 1. 图像语义特征 + 方向标签特征 ----------
        img_feats = self.encoder_i(face)                      # [B, 512]
        label_feats = self.encoder_t2(self.label_tokens)      # [8, 512]

        # L2 归一化后计算余弦相似度
        img_norm = img_feats / img_feats.norm(dim=-1, keepdim=True)    # [B, 512]
        label_norm = label_feats / label_feats.norm(dim=-1, keepdim=True)  # [8, 512]

        # 计算图像与各类属性文本的相似度 logits（乘以温度系数）
        sim_illum = self.logit_scale.exp() * img_norm @ self.illum_norm.T  # [B, 3]
        sim_head = self.logit_scale.exp() * img_norm @ self.head_norm.T    # [B, 2]
        sim_bg = self.logit_scale.exp() * img_norm @ self.bg_norm.T        # [B, 2]

        # 方向标签的相似度额外 clamp 温度，避免 logits 过大
        scale = self.logit_scale.exp().clamp(max=10)
        sim_label = scale * img_norm @ label_norm.T                        # [B, 8]

        # ---------- 2. 选出相似度最高的属性向量（argmax） ----------
        idx_illum = sim_illum.argmax(dim=-1)  # [B]
        idx_head = sim_head.argmax(dim=-1)    # [B]
        idx_bg = sim_bg.argmax(dim=-1)        # [B]
        idx_label = sim_label.argmax(dim=-1)  # [B]

        # 用索引取出每个样本最匹配的属性特征
        selected_illum = self.illum_feats[idx_illum]  # [B, 512]
        selected_head = self.head_feats[idx_head]      # [B, 512]
        selected_bg = self.bg_feats[idx_bg]            # [B, 512]
        selected_label = label_feats[idx_label]        # [B, 512]

        # ---------- 3. 构造两类融合特征 ----------
        # feature_1：图像 + 无关属性（光照/头姿/背景），用于上下文补偿
        feature_1 = img_feats + selected_illum + selected_head + selected_bg
        feature_1 = feature_1 / feature_1.norm(dim=-1, keepdim=True)  # L2 归一化

        # feature_2：图像 + 视线方向标签，用于任务对齐
        feature_2 = img_feats + selected_label
        feature_2 = feature_2 / feature_2.norm(dim=-1, keepdim=True)

        # ---------- 4. CNN 视觉特征 ----------
        # feature_3 = self.main_model(other_face).view(face.size(0), -1)
        feature_3 = self.main_model(other_face).reshape(face.size(0), -1)  # [B, C']

        # ---------- 5. 融合并回归 gaze ----------
        fused = torch.cat([feature_1, feature_2, feature_3], dim=-1)
        gaze_pred = self.fuse_model(fused)  # [B, 3]

        return gaze_pred, sim_label, feature_1, feature_2


class GEWithCLIPModel_zhao(nn.Module):
    """
    与 GEWithCLIPModel 相同的 CLIP 语义融合模型，
    唯一区别在于 CNN 分支：
        - 使用 create_feature_extractor 提取 CNN 中间层（layer4）的特征图，
          而不是全局向量。
        - feature_3 的形状为 [B, C, H, W]，便于后续切成 patch token 送入 Transformer。

    （类内注释同 GEWithCLIPModel，仅在 CNN 分支处有差异。）
    """

    def __init__(
        self,
        irrelevant_feats_dim=512,
        relevant_feats_dim=512,
    ):
        super().__init__()

        # ---------- 文本提示定义（同 GEWithCLIPModel） ----------
        self.illumination_texts = [
            "a face with bright light",
            "a face with low light",
            "a face with shadows",
        ]
        self.headpose_texts = [
            "a frontal face",
            "a profile face",
        ]
        self.background_texts = [
            "a face on bright background",
            "a face on dark background",
        ]
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

        # ---------- 文本 tokenize ----------
        illum_tokens = clip.tokenize(self.illumination_texts).to(DEVICE)
        headpose_tokens = clip.tokenize(self.headpose_texts).to(DEVICE)
        bg_tokens = clip.tokenize(self.background_texts).to(DEVICE)
        self.label_tokens = clip.tokenize(self.label_texts).to(DEVICE)

        # 用临时 CLIP 预计算文本属性特征
        clip_model_1 = copy.deepcopy(CLIP_MODEL)
        clip_model_1.eval()
        with torch.no_grad():
            # encoder_t1
            self.illum_feats = clip_model_1.encode_text(illum_tokens)
            self.head_feats = clip_model_1.encode_text(headpose_tokens)
            self.bg_feats = clip_model_1.encode_text(bg_tokens)

            # L2 归一化
            self.illum_norm = self.illum_feats / self.illum_feats.norm(
                dim=-1, keepdim=True
            )
            self.head_norm = self.head_feats / self.head_feats.norm(
                dim=-1, keepdim=True
            )
            self.bg_norm = self.bg_feats / self.bg_feats.norm(dim=-1, keepdim=True)
        del clip_model_1
        torch.cuda.empty_cache()

        self.model = CLIP_MODEL
        self.encoder_i = CLIP_MODEL.encode_image
        self.encoder_t2 = CLIP_MODEL.encode_text

        # ---------- CNN 主干网络（使用中间特征图） ----------
        if CNN_MODEL == "ResNet-50":
            base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            # 使用 create_feature_extractor 提取 layer4 输出的特征图
            return_nodes = {"layer4": "features"}  # layer4 输出特征图
            main_model = create_feature_extractor(base_model, return_nodes=return_nodes)
            # main_model.forward(x) 返回一个字典，key 是 "features"，值形状为 [B, 2048, H, W]
            main_model_feats_dim = 2048  # ResNet50 layer4 的输出通道数

        elif CNN_MODEL == "ResNet-18":
            base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            return_nodes = {"layer4": "features"}
            main_model = create_feature_extractor(base_model, return_nodes=return_nodes)
            main_model_feats_dim = 512  # ResNet18 layer4 的输出通道数
        elif CNN_MODEL == "EdgeNeXt-Small":
            main_model = timm.create_model("edgenext_small", pretrained=True)
            main_model_feats_dim = main_model.head.fc.in_features
            main_model.head.fc = nn.Identity()
        self.main_model = main_model

        # ---------- 融合层 ----------
        fused_dim = irrelevant_feats_dim + relevant_feats_dim + main_model_feats_dim
        self.fuse_model = nn.Sequential(
            nn.Linear(fused_dim, 256), nn.ReLU(), nn.Linear(256, 3)  # 3D gaze output
        )
        self.logit_scale = CLIP_MODEL.logit_scale

    def forward(
        self,
        face,
        other_face,
    ):
        # 图像语义特征 + 方向标签特征
        img_feats = self.encoder_i(face)
        label_feats = self.encoder_t2(self.label_tokens)

        # 归一化后计算相似度，选出最高索引
        img_norm = img_feats / img_feats.norm(dim=-1, keepdim=True)
        label_norm = label_feats / label_feats.norm(dim=-1, keepdim=True)

        sim_illum = self.logit_scale.exp() * img_norm @ self.illum_norm.T
        sim_head = self.logit_scale.exp() * img_norm @ self.head_norm.T
        sim_bg = self.logit_scale.exp() * img_norm @ self.bg_norm.T

        scale = self.logit_scale.exp().clamp(max=10)
        sim_label = scale * img_norm @ label_norm.T

        idx_illum = sim_illum.argmax(dim=-1)
        idx_head = sim_head.argmax(dim=-1)
        idx_bg = sim_bg.argmax(dim=-1)
        idx_label = sim_label.argmax(dim=-1)
        selected_illum = self.illum_feats[idx_illum]
        selected_head = self.head_feats[idx_head]
        selected_bg = self.bg_feats[idx_bg]
        selected_label = label_feats[idx_label]

        feature_1 = img_feats + selected_illum + selected_head + selected_bg
        feature_1 = feature_1 / feature_1.norm(dim=-1, keepdim=True)

        feature_2 = img_feats + selected_label
        feature_2 = feature_2 / feature_2.norm(dim=-1, keepdim=True)

        # 注意：此处 CNN 输出是 create_feature_extractor 返回的字典，
        # 但在此处直接 reshape 会因返回 dict 而出错（保留原代码，实际训练在 train.py 中另行处理）
        # feature_3 = self.main_model(other_face).view(face.size(0), -1)
        feature_3 = self.main_model(other_face).reshape(face.size(0), -1)

        fused = torch.cat([feature_1, feature_2, feature_3], dim=-1)
        gaze_pred = self.fuse_model(fused)

        return gaze_pred, sim_label, feature_1, feature_2
