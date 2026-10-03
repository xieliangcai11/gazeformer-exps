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

# 项目根由 configs.rgbdgaze_config.PROJECT_ROOT 提供（configs 深度固定，最稳），
# 不在这里数 __file__ 层级（避免目录移动后出错）。使权重路径不依赖运行 CWD。
from configs.rgbdgaze_config import PROJECT_ROOT

DINOV2_CKPT = str(PROJECT_ROOT / "model" / "dinov2_vits14_pretrain.pth")
DINOV2_MODEL = "vit_small_patch14_dinov2"
# 数据统一 224；DINOv2 原生 518，用插值 pos_embed 让模型接受 224 输入
DINOV2_IMG_SIZE = 224
NUM_PREFIX_TOKENS = 1  # cls token 数


class DINOv2Backbone(nn.Module):
    """封装 timm DINOv2：patch_embed -> pos embed -> blocks(hook)。

    注意：这里不能直接调 self.backbone(x) 的返回值来取 patch tokens，
    因为不同 timm 版本会把 cluster token 塞进 pos_embed；用 hook 取 blocks 输出最稳。
    """

    def __init__(self, model_name=DINOV2_MODEL, ckpt=DINOV2_CKPT,
                 freeze=True, unfreeze_last=0):
        """freeze: 是否冻结 DINOv2；unfreeze_last: 解冻最后 N 个 block 参与微调。
        unfreeze_last>0 时允许任务相关微调（代价是显存/耗时增加）。"""
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
        self._unfreeze_last = unfreeze_last
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            if unfreeze_last > 0:
                # 解冻最后 N 个 block 用于任务微调
                blocks = list(self.backbone.blocks)
                for blk in blocks[-unfreeze_last:]:
                    for p in blk.parameters():
                        p.requires_grad = True
        # 只有"全部冻结"才用 no_grad 包前向；一旦有任何可训练参数就不能包
        self._no_grad = bool(freeze and unfreeze_last == 0)
        if self._no_grad:
            self.backbone.eval()  # 冻结时锁死内部 dropout/训练态，保证确定性
        self._tokens = None
        # 抓最后一个 Block 的输出（在 final norm / head 之前），含 cls + patches
        self._hook = self.backbone.blocks[-1].register_forward_hook(self._capture)

    def _capture(self, module, args, output):
        self._tokens = output

    def forward(self, x):
        # 全部冻结时：eval + no_grad（外层 model.train() 不会干扰冻结部分）。
        # 若有可训练参数（未冻结/解冻），正常走训练态并保留梯度。
        if self._no_grad:
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


class DepthAttentionFusion(nn.Module):
    """深度特征图 -> 空间注意力权重图 -> 与 RGB 特征图逐像素相乘（论文同款机制）。

    结构（对应 RGBDGaze 论文 Figure 3 的 CNN Spatial Attention Unit）：
        depth [B,1,H,W]
          -> 轻量 CNN（几层 3x3）得 depth 特征 [B, C_d, H', W']
          -> 1x1 conv 出单通道空间权重图 [B, 1, H', W']（sigmoid）
          -> RGB 特征图与权重图逐元素相乘后重新投影到 d_model
    保留深度空间结构（相比旧版压成 64 个 token，信息不再被池化丢弃）。

    为什么加 RGB 特征图：RGB tokens 是 DINOv2 输出（[B,N,C]，N=256, 网格16x16），
    深度注意力图下采样到 16x16 与之对齐，乘法融合后得到"被深度引导"的 RGB 表征，
    再与原 RGB 表征并联，保证不丢原信息。
    """

    def __init__(self, dino_dim: int, d_model: int, hidden: int = 32, grid: int = 16):
        super().__init__()
        self.grid = grid
        # 轻量 CNN：1通道逆深度 -> hidden 特征
        self.cnn = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(inplace=True),
        )
        # 1x1 conv 出空间注意力图（单通道）
        self.attn = nn.Conv2d(hidden, 1, 1)
        # 融合后投影回 d_model（输入 = RGB特征(被门控) + 原RGB特征 拼接? 简化为乘法后直接投影）
        self.proj = nn.Linear(dino_dim, d_model)
        self.gate = nn.Parameter(torch.zeros(1))  # 融合强度门控（可学习，初始0=不干扰）

    def forward(self, depth: torch.Tensor, rgb_tokens: torch.Tensor):
        """depth: [B,1,H,W] 逆深度; rgb_tokens: [B, N, dino_dim] (无cls)。
        返回 [B, N, d_model] 的深度门控 RGB 表征。
        """
        B = depth.size(0)
        # 1) 深度特征 -> 注意力图，池化到 patch 网格
        feat = self.cnn(depth)                                # [B, hidden, H, W]
        attn = torch.sigmoid(self.attn(feat))                 # [B, 1, H, W]
        attn = nn.functional.adaptive_avg_pool2d(attn, (self.grid, self.grid))
        attn = attn.reshape(B, 1, self.grid * self.grid)      # [B, 1, N]
        # 2) 门控乘法：深度引导的 RGB 表征 = 原 tokens * (1 + gate * attn)
        gated = rgb_tokens * (1.0 + self.gate * attn.permute(0, 2, 1))  # [B,N,C]
        return self.proj(gated)                               # [B, N, d_model]


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
        use_depth=True,
        unfreeze_last=0,
        depth_mode="token",
        use_imu=False,
        dino_model=DINOV2_MODEL,
        dino_ckpt=DINOV2_CKPT,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model({d_model}) 必须能被 num_heads({num_heads}) 整除")

        # ---- RGB 流：冻结 DINOv2（可选解冻最后 N 层微调） ----
        self.rgb = DINOv2Backbone(dino_model, dino_ckpt, freeze=freeze_dino,
                                  unfreeze_last=unfreeze_last)
        dino_dim = self.rgb.embed_dim
        self.proj_rgb_cls = nn.Linear(dino_dim, d_model)
        self.proj_rgb_patch = nn.Linear(dino_dim, d_model)

        # ---- Depth 流（可关；两种用法可选，做消融） ----
        # depth_mode:
        #   "none"      : 关闭深度
        #   "attn"      : 轻量 CNN 特征图 + 与 RGB 特征图空间注意力相乘（论文同款机制）
        #   "token"     : 旧版逆深度 grid token（消融已证弱，保留作对照）
        self.use_depth = use_depth
        self.depth_mode = depth_mode
        if use_depth:
            if depth_mode == "attn":
                self.depth_attn = DepthAttentionFusion(dino_dim, d_model)
                self.depth_patch = None
            elif depth_mode == "token":
                self.depth_patch = DepthPatchEmbed(in_channels=1, grid=depth_grid,
                                                   d_model=d_model)
                self.depth_attn = None
            else:
                raise ValueError(f"未知 depth_mode: {depth_mode}")
        else:
            self.depth_patch = None
            self.depth_attn = None

        # ---- IMU 姿态 token（可选） ----
        self.use_imu = use_imu
        if use_imu:
            self.proj_imu = nn.Linear(3, d_model)

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

    def forward(self, face, depth, imu=None):
        # 1) RGB：DINOv2（冻结/部分解冻）-> cls + patches
        tokens = self.rgb(face)                      # [B, 1+N, C]
        cls = tokens[:, 0]                           # [B, C]
        patches = tokens[:, 1:]                      # [B, N, C]
        rgb_cls = self.proj_rgb_cls(cls).unsqueeze(1)    # [B, 1, d_model]
        rgb_patch = self.proj_rgb_patch(patches)         # [B, N, d_model]

        # 2) 拼接：汇总CLS + rgb_cls [+ depth] + rgb_patches
        if self.use_depth and self.depth_mode == "attn":
            # 深度空间注意力融合：门控增强后的 RGB patch 表征
            d_patch = self.depth_attn(depth, patches)    # [B, N, d_model]
            seq = torch.cat([rgb_cls, d_patch, rgb_patch], dim=1)
        elif self.use_depth and self.depth_mode == "token":
            depth_tokens = self.depth_patch(depth)       # [B, N_d, d_model]
            seq = torch.cat([rgb_cls, depth_tokens, rgb_patch], dim=1)
        else:
            seq = torch.cat([rgb_cls, rgb_patch], dim=1)

        # 2b) IMU 姿态 token（可选，1 个 token）
        if self.use_imu:
            if imu is None:
                raise ValueError("use_imu=True 但 forward 未提供 imu")
            imu_tok = self.proj_imu(imu).unsqueeze(1)    # [B, 1, d_model]
            seq = torch.cat([imu_tok, seq], dim=1)

        B = depth.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, seq], dim=1)       # [B, 1+1+[imu]+[N_d]+N, d_model]

        # 3) BlockMoba 栈（自注意力融合）
        for layer in self.layers:
            x = layer(x, cross_input=None, mask=None)

        # 4) 汇总 CLS -> 2D
        out = self.head(self.norm(x[:, 0, :]))
        return out