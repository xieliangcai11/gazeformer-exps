"""RGBDGaze VPT (Visual Prompt Tuning) 适配模型。

VPT 原理：
    冻结整个预训练模型，在 transformer 输入端加入少量可学习的 "prompt tokens"。
    这些 tokens 与图像 patch tokens 一起通过 attention 层，
    从而在不改变模型权重的情况下，适配新用户/新数据分布。

适配流程：
    1. 加载已训练好的骨干模型权重（全部冻结）。
    2. 初始化 N 个可学习的 prompt tokens（每个 d_model 维）。
    3. 用新用户的校准数据，只训练 prompt tokens（参数量极少）。
    4. 推理时，prompt tokens 与 patch tokens 一起通过 attention，
       使模型对该用户的预测被"校准"。

数据流：
    face -> DINOv2(patch_embed + pos_embed) -> [prompt_tokens, cls, patches] -> blocks -> 输出
                                                 ↑
                                    可学习参数（每个用户独立）
"""

from __future__ import annotations

import torch
import torch.nn as nn

from gazelab.models.dinov2_gaze.rgbdgaze_dinov2 import (
    DINOv2Backbone,
    RMSNorm, BlockMoba, DINOV2_CKPT, DINOV2_MODEL,
)


class VPTModel(nn.Module):
    """VPT 适配模型：冻结骨干 + 可学习 prompt tokens + 融合回归头。

    与 RGBDGazeDINOv2 的区别：
        - 骨干完全冻结（不做任何微调）
        - 在 transformer 的 patch tokens 输入端加入 n_prompt 个可学习的
          "用户 prompt tokens"，它们通过 attention 影响所有 patch tokens 的表征
        - 适配新用户时，只需训练这 n_prompt 个 tokens

    forward(face, imu=None, geom=None) -> gaze [B, 2]
    """

    def __init__(
        self,
        gaze_dim: int = 2,
        d_model: int = 384,
        num_heads: int = 8,
        num_layers: int = 6,
        inter_dim: int = 1536,
        use_imu: bool = False,
        use_geom: bool = False,
        img_size: int = 448,
        n_prompt: int = 4,
        dino_model: str = DINOV2_MODEL,
        dino_ckpt: str = DINOV2_CKPT,
    ):
        super().__init__()
        # ---- 冻结的 DINOv2 骨干 ----
        self.rgb = DINOv2Backbone(dino_model, dino_ckpt, freeze=True,
                                  unfreeze_last=0, img_size=img_size)
        dino_dim = self.rgb.embed_dim

        # ---- VPT prompt tokens（可学习，每个用户一组） ----
        self.n_prompt = n_prompt
        self.prompt_tokens = nn.Parameter(
            torch.randn(1, n_prompt, dino_dim) * 0.02  # 标准初始化
        )

        # ---- 投影层 ----
        self.proj_rgb_cls = nn.Linear(dino_dim, d_model)
        self.proj_rgb_patch = nn.Linear(dino_dim, d_model)

        # ---- IMU / 屏幕几何 token（可选，与主模型保持一致） ----
        self.use_imu = use_imu
        if use_imu:
            self.proj_imu = nn.Linear(3, d_model)
        self.use_geom = use_geom
        if use_geom:
            self.proj_geom = nn.Linear(3, d_model)

        # ---- 融合 Transformer ----
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.layers = nn.ModuleList([
            BlockMoba(d_model, num_heads,
                      n_routed_experts=4, n_activated_experts=2,
                      n_shared_experts=1, moe_inter_dim=inter_dim // 2,
                      inter_dim=inter_dim, layer_id=i)
            for i in range(num_layers)
        ])
        self.norm = RMSNorm(d_model, eps=1e-5)
        self.head = nn.Linear(d_model, gaze_dim)

        # 默认全部冻结
        self.freeze_all()

    def freeze_all(self):
        """冻结所有参数（除 prompt_tokens 和融合层）。"""
        # 冻结骨干
        for p in self.rgb.parameters():
            p.requires_grad = False
        self.rgb.backbone.eval()
        self.rgb._no_grad = True
        # 冻结投影层
        for p in self.proj_rgb_cls.parameters():
            p.requires_grad = False
        for p in self.proj_rgb_patch.parameters():
            p.requires_grad = False
        # 冻结融合层和回归头（这些已经在训练集上训练好了）
        for layer in self.layers:
            for p in layer.parameters():
                p.requires_grad = False
        for p in self.head.parameters():
            p.requires_grad = False
        for p in self.norm.parameters():
            p.requires_grad = False
        # 只保留 prompt tokens 可训练
        self.prompt_tokens.requires_grad = True

    def unfreeze_all(self):
        """解冻所有参数（完整训练模式，如初始骨干训练时）。"""
        for p in self.parameters():
            p.requires_grad = True
        self.rgb._no_grad = False

    def forward(self, face, imu=None, geom=None):
        # 1) DINOv2 骨干（冻结，只提特征）
        # 关键：prompt tokens 在 DINOv2 的 patch_embed + pos_embed 之后拼接，
        #       然后一起通过 transformer blocks。
        # 由于我们无法直接修改 DINOv2 的 forward（timm 内部实现），
        # 这里采用另一种方式：在 hook 捕获的 tokens 输出端拼接 prompt tokens，
        # 让融合 transformer 中的 attention 起到类似作用。
        tokens = self.rgb(face)                      # [B, 1+N, dino_dim]
        cls = tokens[:, 0]                           # [B, dino_dim]
        patches = tokens[:, 1:]                      # [B, N, dino_dim]

        # 2) 投影到统一维度
        rgb_cls = self.proj_rgb_cls(cls).unsqueeze(1)      # [B, 1, d_model]
        rgb_patch = self.proj_rgb_patch(patches)           # [B, N, d_model]

        # 3) 可选 token：IMU 姿态 / 屏幕几何
        seq = torch.cat([rgb_cls, rgb_patch], dim=1)
        if self.use_imu:
            if imu is None:
                raise ValueError("use_imu=True 但 forward 未提供 imu")
            seq = torch.cat([self.proj_imu(imu).unsqueeze(1), seq], dim=1)
        if self.use_geom:
            if geom is None:
                raise ValueError("use_geom=True 但 forward 未提供 geom")
            seq = torch.cat([self.proj_geom(geom).unsqueeze(1), seq], dim=1)

        # 4) VPT：展开 prompt tokens 并拼接到序列前面
        #    这些 tokens 通过融合 transformer 的 attention 影响所有其他 tokens
        B = face.size(0)
        prompt = self.prompt_tokens.expand(B, -1, -1)    # [B, n_prompt, d_model]
        x = torch.cat([self.cls_token.expand(B, -1, -1), prompt, seq], dim=1)

        # 5) 融合 transformer（VPT tokens 在 attention 中影响所有其他 tokens）
        for layer in self.layers:
            x = layer(x, cross_input=None, mask=None)

        # 6) CLS -> 2D（VPT tokens 的信息已被 attention 融合到 CLS 中）
        return self.head(self.norm(x[:, 0, :]))


def load_backbone_weights(model: VPTModel, ckpt_path: str, device: str = "cpu"):
    """从已训练的 RGBDGazeDINOv2 checkpoint 加载所有非 VPT 权重。

    VPT prompt tokens 是新增的，保持随机初始化。
    """
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    # 移除 VPT 特有的参数（如果 checkpoint 中没有）
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print(f"[VPT] checkpoint 中的多余键: {unexpected}")
    # 确保 prompt_tokens 可训练
    model.prompt_tokens.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[VPT] 可训练: {trainable:,} / 总计: {total:,} ({trainable/total*100:.2f}%)")
    print(f"[VPT] 其中 prompt_tokens: {model.prompt_tokens.numel()} 参数")