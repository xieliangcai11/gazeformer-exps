"""RGBDGaze 新模型：DINOv2 冻结 RGB + 逆深度几何 + Transformer(BlockMoba) 融合。

设计（替代旧的 CLIP+ResNet 双流，输入一致：RGB 人脸 + 深度人脸）：
    RGB 流   → DINOv2（ViT-S/14，冻结）提取 cls + patch tokens。
    Depth 流 → 逆深度单通道，grid 池化 + 线性嵌入为少量 depth tokens（几何补充，不喧宾夺主）。
    融合     → 复用 gazelab.models.transformers 的 BlockMoba（标准自注意力 + MoE/FFN），
               把 [汇总CLS, rgb_cls, depth_tokens, rgb_patch_tokens] 一起自注意力。
    回归     → 取汇总 CLS 经线性头输出 2D 屏幕坐标。

训练目标（在训练脚本中实现）：2D 坐标 MSE + 几何角度损失（配合 VIEWING_DISTANCE_CM）。

输入：
    face  : [B, 3, 224, 224] RGB 人脸（ImageNet 归一化，适配 DINOv2 预处理）。
    depth : [B, 1, 224, 224] 逆深度单通道（由 inverse_depth_preprocess 产生）。

设计取舍（重要，避免逻辑/资源纰漏）：
    - 不直接复用 TransformerDeepSeek_gaze：其 proj_f1/f2/f3 把输入维度写死为
      512/512/2048（面向 CLIP+ResNet）。RGBD 无 CLIP 时若强行喂入，会因
      2048->768 投影作用在 256 个 token 上而产生约 4 亿参数，浪费且错误。
      此处复用其底层 BlockMoba 块，构造 RGBD 专用、维度正确的投影。
    - DINOv2 冻结当现成"眼睛"，只训融合与回归头。
"""

import torch
import torch.nn as nn
import timm
from pathlib import Path

from gazelab.models.transformers import RMSNorm, BlockMoba

from timm.layers.pos_embed import resample_abs_pos_embed

# 项目根 = 本文件 src/gazelab/models/ 向上 4 级；使权重路径不依赖运行 CWD
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DINOV2_CKPT = str(_PROJECT_ROOT / "model" / "dinov2_vits14_pretrain.pth")
DINOV2_MODEL = "vit_small_patch14_dinov2"
# 数据统一 224；DINOv2 原生 518，用插值 pos_embed 让模型接受 224 输入
DINOV2_IMG_SIZE = 224
NUM_PREFIX_TOKENS = 1  # cls token 数


class DINOv2Backbone(nn.Module):
    """封装 timm DINOv2：patch_embed -> pos embed -> blocks(hook)。

    注意：这里不能直接调 self.backbone(x) 的返回值来取 patch tokens，
    因为不同 timm 版本会把 cluster token 塞进 pos_embed；用 hook 取 blocks 输出最稳。
    """

    def __init__(self, model_name=DINOV2_MODEL, ckpt=DINOV2_CKPT, freeze=True):
        super().__init__()
        backbone = timm.create_model(model_name, pretrained=False,
                                     img_size=DINOV2_IMG_SIZE)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        # DINOv2 官方权重额外带 'mask_token'（预训练专用），加载时忽略
        state.pop("mask_token", None)
        # 权重是 518 训练的，输入 224 时需把 pos_embed 插值到 224 对应格数
        if "pos_embed" in state:
            # 权重是 518 训练的，输入 224 时把 pos_embed 从 37x37 网格插值到 16x16。
            # new_size 传 patch 网格 (16,16)；其余 prefix(cls) 由库内 num_prefix_tokens 保留。
            grid = DINOV2_IMG_SIZE // backbone.patch_embed.patch_size[0]
            state["pos_embed"] = resample_abs_pos_embed(
                state["pos_embed"],
                (grid, grid),
                num_prefix_tokens=NUM_PREFIX_TOKENS,
            )
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        if unexpected:
            raise ValueError(f"[DINOv2] 意外键: {unexpected}")
        if missing:
            raise ValueError(f"[DINOv2] 缺失键: {missing}")
        self.backbone = backbone
        self._frozen = freeze
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()  # 冻结时锁死内部 dropout/训练态，保证确定性
        self._tokens = None
        # 抓最后一个 Block 的输出（在 final norm / head 之前），含 cls + patches
        self._hook = self.backbone.blocks[-1].register_forward_hook(self._capture)

    def _capture(self, module, args, output):
        self._tokens = output

    def forward(self, x):
        # 冻结时：无条件 eval + no_grad，避免外层 model.train() 把 DINOv2 内部
        # dropout/随机深度重新打开（那会令冻结模型前向不确定）。
        # no_grad 下产出的张量仍会流入后续可训练层并回传梯度到它们，功能正确。
        if self._frozen:
            self.backbone.eval()
            with torch.no_grad():
                self.backbone(x)
        else:
            self.backbone(x)
        t = self._tokens
        self._tokens = None
        if t is None:
            raise RuntimeError("[DINOv2] hook 未捕获 block 输出")
        return t

    @property
    def embed_dim(self):
        return self.backbone.embed_dim

    def __del__(self):
        try:
            self._hook.remove()
        except Exception:  # noqa
            pass


class DepthPatchEmbed(nn.Module):
    """把逆深度单通道图嵌入为 depth tokens（少量，几何补充）。

    流程：逆深度 [B,1,H,W] -> AdaptiveAvgPool 到 (grid, grid) -> 展平 [B,N,1]
          -> Linear(1, d_model) -> [B, N, d_model]。
    N = grid*grid（默认 8*8=64），远小于 RGB 的 256 个 patch，保持 RGB 主导。
    """

    def __init__(self, in_channels=1, grid=8, d_model=384):
        super().__init__()
        self.grid = grid
        self.pool = nn.AdaptiveAvgPool2d((grid, grid))
        self.proj = nn.Linear(in_channels, d_model)

    def forward(self, depth):
        """depth: [B, in_channels, H, W]。返回 [B, grid*grid, d_model]."""
        B = depth.size(0)
        x = self.pool(depth)                      # [B, C, grid, grid]
        x = x.reshape(B, self.grid * self.grid, -1)  # [B, N, C]
        return self.proj(x)                       # [B, N, d_model]


class RGBDGazeDINOv2(nn.Module):
    """DINOv2(RGB,冻结) + 逆深度 depth tokens + BlockMoba 融合 -> 2D 屏幕坐标。

    forward(face, depth) -> [B, 2]
      face  : [B, 3, 224, 224] RGB（ImageNet 归一化，DINOv2 预处理）
      depth : [B, 1, 224, 224] 逆深度单通道
    """

    def __init__(
        self,
        gaze_dim=2,
        d_model=384,
        num_heads=8,
        num_layers=6,
        inter_dim=1536,
        depth_grid=8,
        freeze_dino=True,
        dino_model=DINOV2_MODEL,
        dino_ckpt=DINOV2_CKPT,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model({d_model}) 必须能被 num_heads({num_heads}) 整除")

        # ---- RGB 流：冻结 DINOv2 ----
        self.rgb = DINOv2Backbone(dino_model, dino_ckpt, freeze=freeze_dino)
        dino_dim = self.rgb.embed_dim
        self.proj_rgb_cls = nn.Linear(dino_dim, d_model)
        self.proj_rgb_patch = nn.Linear(dino_dim, d_model)

        # ---- Depth 流 ----（投影在本模块内完成，无对外的 Linear）
        self.depth_patch = DepthPatchEmbed(in_channels=1, grid=depth_grid, d_model=d_model)
        # proj 输入为 1 通道 grid（网格内特征本质上是逆深度值），Linear(1, d_model)

        # ---- 融合：BlockMoba 栈 + 汇总 token ----
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.layers = nn.ModuleList([
            BlockMoba(
                d_model, num_heads,
                n_routed_experts=4, n_activated_experts=2,
                n_shared_experts=1, moe_inter_dim=inter_dim // 2,
                inter_dim=inter_dim, layer_id=i,  # 浅层(i<9)用 dense FFN，省资源
            )
            for i in range(num_layers)
        ])
        self.norm = RMSNorm(d_model, eps=1e-5)
        self.head = nn.Linear(d_model, gaze_dim)

    def forward(self, face, depth):
        # 1) RGB：DINOv2（冻结）-> cls + patches
        tokens = self.rgb(face)                      # [B, 1+N, C]
        cls = tokens[:, 0]                           # [B, C]
        patches = tokens[:, 1:]                      # [B, N, C]
        rgb_cls = self.proj_rgb_cls(cls).unsqueeze(1)    # [B, 1, d_model]
        rgb_patch = self.proj_rgb_patch(patches)         # [B, N, d_model]

        # 2) Depth：逆深度单通道 -> grid depth tokens
        depth_tokens = self.depth_patch(depth)        # [B, N_d, d_model]

        # 3) 拼接全部 token：汇总CLS + rgb_cls + depth + rgb_patches
        B = depth.size(0)
        seq = torch.cat([rgb_cls, depth_tokens, rgb_patch], dim=1)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, seq], dim=1)       # [B, 1+1+N_d+N, d_model]

        # 4) BlockMoba 栈（自注意力融合）
        for layer in self.layers:
            x = layer(x, cross_input=None, mask=None)

        # 5) 汇总 CLS -> 2D
        out = self.head(self.norm(x[:, 0, :]))
        return out