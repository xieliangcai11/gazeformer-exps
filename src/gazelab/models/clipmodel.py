"""
clipmodel.py
============
该文件实现了 OpenAI CLIP 模型的核心网络结构（视觉编码器 + 文本编码器）。

在本项目（GazeFormer）中的作用：
    - 通过 `clip.load("ViT-B/32")` 加载的 CLIP 模型就是由本文件中的类定义的。
    - 提供 `encode_image`（图像 -> 512 维语义向量）和 `encode_text`（文本 -> 512 维语义向量）
      两个方法，用于 gaze 估计任务中"图像特征"与"文本提示特征"的相似度计算。
    - `build_model(state_dict)` 用于从官方预训练权重中还原模型结构并加载参数。

包含的主要组件：
    - Bottleneck             : ResNet 瓶颈残差块
    - AttentionPool2d        : 注意力池化（替代 ResNet 的全局平均池化）
    - ModifiedResNet         : 修改过的 ResNet 视觉编码器（RN50 / RN101 等）
    - LayerNorm              : 支持 fp16 的 LayerNorm
    - QuickGELU              : GELU 激活函数的快速近似
    - ResidualAttentionBlock : Transformer 残差注意力块
    - Transformer            : 由残差注意力块堆叠而成的 Transformer
    - VisionTransformer      : ViT 视觉编码器（ViT-B/32 等）
    - CLIP                   : 完整的 CLIP 双塔模型（视觉 + 文本）
    - convert_weights        : 将模型参数转换为 fp16
    - build_model            : 根据 state_dict 推断超参并构建模型

术语对照（中英对照，全文通用）：
    - logits    : 未归一化打分（logits），经 softmax 后才变成概率分布。
    - mask      : 掩码（mask），用于屏蔽注意力中不应关注的位置，通常被屏蔽处为 -inf。
    - token     : 词元/图块（token），序列的基本单元：一个词、一个图像块或一个特征向量。
    - CLS token : 额外拼接在序列开头的全局词元，用于聚合整个序列的信息。
    - broadcast : 广播（broadcasting），形状不同的张量从最右维向左自动补齐后逐元素运算。
    - residual  : 残差连接（residual connection），输出 = 输入 + 变换(输入)，利于深层网络训练。
    - buffer    : 缓冲张量（buffer），随模型保存/随设备移动，但不是可学习参数。
"""

from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.nn.init as init
import math


class Bottleneck(nn.Module):
    """
    ResNet 的瓶颈（Bottleneck）残差块。

    结构：1x1 降维卷积 -> 3x3 卷积 -> 1x1 升维卷积，配合残差连接。
    与标准 torchvision 实现的不同之处：
        - 下采样不使用 stride=2 的卷积，而是用 avgpool 完成（抗混叠，anti-aliasing）。
        - 所有卷积层 stride 均为 1。

    关于 BatchNorm（批归一化）：
        - 训练时用当前 batch 的均值/方差做归一化，并同步维护滑动平均统计量（running stats）。
        - 推理时固定使用滑动平均统计量，因此推理前必须调用 model.eval()。
        - 注意: batch 很小时（如 1~2）训练统计量噪声大，BN 表现会不稳定。
    """

    # 每个 Bottleneck 会把通道数扩展为 planes 的 4 倍
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        """
        Args:
            inplanes (int): 输入通道数。
            planes (int): 瓶颈中间通道数（输出通道为 planes*4）。
            stride (int, 默认 1): 空间下采样倍数，>1 时启用 avgpool 下采样与捷径分支。
        """
        super().__init__()

        # 第一个 1x1 卷积：把输入通道 inplanes 压缩到 planes（降维）
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        # 第二个 3x3 卷积：在 planes 通道内做空间特征提取（padding=1 保持尺寸）
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        # 当 stride > 1 时用平均池化实现下采样（而不是 stride=2 卷积，避免混叠）
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        # 第三个 1x1 卷积：把通道数扩展到 planes * 4（升维）
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        # stride: 普通成员（int，不可学习），记录下采样倍数，仅构造期使用
        self.stride = stride

        # 当需要改变空间尺寸（stride>1）或通道数不匹配时，构造"捷径分支"使残差相加维度一致
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # 下采样分支：先 avgpool，再用 1x1 卷积对齐通道数
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: 输入特征图，[B, inplanes, H, W]，float32/fp16。
        Returns:
            输出特征图，[B, planes*4, H', W']（有下采样时 H'=H/stride，否则同输入），
            dtype 与输入一致。
        """
        # x: [B, inplanes, H, W]（B 为 batch size，C 通道，H/W 空间高宽）
        # 保存输入作为残差连接的 identity
        identity = x

        # 主路径：conv1 -> bn1 -> relu -> conv2 -> bn2 -> relu -> avgpool -> conv3 -> bn3
        # ->（conv1 降维）-> [B, planes, H, W]
        out = self.relu1(self.bn1(self.conv1(x)))
        # ->（conv2 + avgpool）-> [B, planes, H', W']（有下采样时 H'=H/stride）
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        # ->（conv3 升维）-> [B, planes*4, H', W']
        out = self.bn3(self.conv3(out))

        # 若存在下采样分支，则对 identity 做同样的下采样/通道对齐 -> [B, planes*4, H', W']
        if self.downsample is not None:
            identity = self.downsample(x)

        # 残差相加，再经过 ReLU
        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    """
    注意力池化层（Attention Pooling）。

    替代 ResNet 末尾的全局平均池化：把空间特征图 (HxW) 压平成一组 token，
    并额外附加一个全局查询 token（CLS），通过多头注意力聚合整张图的信息，
    最终输出该 CLS token 对应的特征向量。
    """

    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        """
        Args:
            spacial_dim (int): 特征图边长（pool 前的空间尺寸，如 7）。
            embed_dim (int): 输入特征通道数。
            num_heads (int): 注意力头数。
            output_dim (int, 可为 None): 输出特征维度；None 时等于 embed_dim。
        """
        super().__init__()
        # 位置编码：spacial_dim^2 个空间位置 + 1 个 CLS token
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        # QKV 投影层
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        # 输出投影层
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        # positional_embedding 形状 [HW+1, embed_dim]，第一行对应 CLS token 的位置编码。
        # 全部成员均为可学习参数，float32，随模型 .to(device) 移动；生命周期与模型相同。

    def forward(self, x):
        """
        Args:
            x: 输入特征图，[B, C, H, W]，C=embed_dim，H=W=spacial_dim，float32/fp16。
        Returns:
            聚合后的全局特征，[B, output_dim]，dtype 同输入。
        """
        # x: [B, C, H, W]（H=W=spacial_dim，C=embed_dim）
        # 输入 x 形状: NCHW，flatten 把 H/W 合并成一维，permute 把空间维移到最前 -> (HW)NC
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        # 在所有 token 前插入一个全局平均 token 作为查询（CLS token）
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        # 加上位置编码
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        # 注意力原理：Q（query，查询）与每个 K（key，键）算相似度，在 key 维（dim=-1）
        # 做 softmax 得到权重，再对 V（value，值）加权求和。
        # 此处 query 只有 1 个（CLS token），key/value 是全部 (HW+1) 个 token，
        # 因此输出可理解为"全图信息的加权聚合"。
        # 使用 PyTorch 的多头注意力前向函数：
        # query 只取 CLS token（x[:1]），key/value 使用全部 token，从而聚合全局信息
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        # 去掉最前面的维度，返回 CLS token 的特征 -> [B, output_dim]
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    修改过的 ResNet 视觉编码器（对应 CLIP 的 RN50 / RN101 / RN50x4 等变体）。

    与 torchvision ResNet 的主要差异：
        1. "stem" 由 3 层卷积组成（而非 1 层），并使用 avgpool 而非 maxpool。
        2. 使用抗混叠的下采样方式：stride>1 的卷积前会先做 avgpool。
        3. 最后的池化层是 QKV 注意力池化（AttentionPool2d）而非平均池化。
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        """
        Args:
            layers (tuple[int, int, int, int]): 4 个残差阶段各自的 Bottleneck 数量，
                如 (3, 4, 6, 3) 对应 RN50。
            output_dim (int): 最终输出特征维度（= CLIP 的 embed_dim，通常 512）。
            heads (int): 末尾注意力池化的头数。
            input_resolution (int, 默认 224): 期望输入图像边长。
            width (int, 默认 64): stem 的基础通道数。
        """
        super().__init__()
        # output_dim: 普通成员（int，不可学习），最终输出特征维度（= embed_dim）
        self.output_dim = output_dim
        # input_resolution: 普通成员（int，不可学习），期望输入分辨率（如 224）
        self.input_resolution = input_resolution

        # ---- 3 层 stem（茎部）----
        # 第一层 stride=2 将 224x224 降采样为 112x112
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        # 最后 avgpool 2x2 将 112x112 降采样为 56x56
        self.avgpool = nn.AvgPool2d(2)

        # ---- 4 个残差阶段 ----
        # _inplanes 是一个在构造过程中被不断更新的"可变"变量，记录当前通道数
        self._inplanes = width
        # 各层输出通道分别为 width*4, width*8, width*16, width*32（Bottleneck.expansion=4）
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        # ResNet 最终输出的特征维度：width * 32
        embed_dim = width * 32
        # 用注意力池化把 7x7（input_resolution//32）的特征图聚合为 output_dim 维向量
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        """
        构造一个残差阶段：第一个 Bottleneck 可能带 stride（用于下采样），
        其余 blocks-1 个 Bottleneck 保持尺寸不变。

        Args:
            planes (int): 该阶段瓶颈中间通道数（输出为 planes*4）。
            blocks (int): 该阶段 Bottleneck 数量。
            stride (int, 默认 1): 第一个 Bottleneck 的下采样倍数。
        Returns:
            nn.Sequential: 由 blocks 个 Bottleneck 组成的残差阶段。
        """
        layers = [Bottleneck(self._inplanes, planes, stride)]

        # 经过第一个 Bottleneck 后，通道数变为 planes * expansion
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: 输入图像，[B, 3, input_resolution, input_resolution]，float32/fp16。
        Returns:
            全局图像特征，[B, output_dim]，dtype 同模型权重（fp16 模型返回 fp16）。
        """
        # 内部函数：执行 3 层 stem
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        # 将输入转换为与权重一致的 dtype（便于 fp16 混合精度训练）
        x = x.type(self.conv1.weight.dtype)
        # stem: [B, 3, 224, 224] -> [B, width, 56, 56]
        x = stem(x)
        # 依次通过 4 个残差阶段，空间尺寸逐次减半、通道逐次翻倍：
        # layer1 -> [B, 256, 56, 56]；layer2 -> [B, 512, 28, 28]；
        # layer3 -> [B, 1024, 14, 14]；layer4 -> [B, 2048, 7, 7]（width=64 时）
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 注意力池化得到最终特征 -> [B, output_dim]
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """
    继承 torch 的 LayerNorm，目的是兼容 fp16：
    前向时先把输入转成 float32 计算，再把结果转回原来的 dtype，
    避免 fp16 下数值溢出。
    """

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: 任意形状/dtype 的输入张量。
        Returns:
            LayerNorm 后的张量，形状与输入相同，dtype 与输入相同。
        """
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    """
    GELU 激活函数的快速近似实现：
        QuickGELU(x) = x * sigmoid(1.702 * x)
    相比标准 GELU 计算更快，是 CLIP 官方实现采用的做法。
    """

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: 任意形状的输入张量（逐元素作用）。
        Returns:
            QuickGELU 激活后的张量，形状与 dtype 同输入。
        """
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    """
    Transformer 的残差注意力块（Pre-LN 结构）。

    结构：
        x = x + MultiheadAttention(LayerNorm(x))   # 自注意力 + 残差
        x = x + MLP(LayerNorm(x))                  # 前馈网络 + 残差
    MLP 由 Linear(d_model -> 4*d_model) -> QuickGELU -> Linear(4*d_model -> d_model) 组成。
    """

    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        """
        Args:
            d_model (int): token 特征维度。
            n_head (int): 注意力头数（d_model 需能被 n_head 整除）。
            attn_mask (torch.Tensor, 可为 None): 加性注意力掩码，
                形状 [L, L]，被屏蔽处为 -inf；文本编码器传入因果掩码。
        """
        super().__init__()

        # 多头自注意力
        self.attn = nn.MultiheadAttention(d_model, n_head)
        # 注意力前的 LayerNorm
        self.ln_1 = LayerNorm(d_model)
        # 前馈网络（MLP）
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        # MLP 前的 LayerNorm
        self.ln_2 = LayerNorm(d_model)
        # 可选的注意力掩码（文本编码器中用于因果掩码）
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        """
        Args:
            x: 输入序列，[L, B, d_model]（LND 布局），float32/fp16。
        Returns:
            自注意力输出，[L, B, d_model]，dtype 同输入。
        """
        # 将掩码转换到与输入相同的 dtype 和 device
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        # mask（掩码）为"加性掩码"：被屏蔽位置填 -inf，注意力分数加上 -inf 后
        # 经 softmax（在 key 维 dim=-1 上）权重变为 0。
        # 文本编码器使用因果掩码（causal mask，下三角形状）：每个 token 只能看到自己之前的 token，
        # 这是自回归语言建模的标准做法。
        # 自注意力：query=key=value=x，只取输出（不需要注意力权重）
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: 输入序列，[L, B, d_model]。
        Returns:
            经过 自注意力+MLP 两个残差子层后的序列，[L, B, d_model]，dtype 同输入。
        """
        # 第一残差：自注意力
        x = x + self.attention(self.ln_1(x))
        # 第二残差：MLP
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    """
    由多个 ResidualAttentionBlock 堆叠而成的 Transformer 编码器。
    """

    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        """
        Args:
            width (int): token 特征维度（d_model）。
            layers (int): 残差注意力块数量。
            heads (int): 每块注意力头数。
            attn_mask (torch.Tensor, 可为 None): 加性掩码 [L, L]，None 表示不屏蔽。
        """
        super().__init__()
        self.width = width
        self.layers = layers
        # 堆叠 layers 个残差注意力块
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: 输入序列，[L, B, width]，LND 布局。
        Returns:
            逐块编码后的序列，[L, B, width]，dtype 同输入。
        """
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) 视觉编码器（CLIP 的 ViT-B/32、ViT-L/14 等）。

    流程：
        1. 用 patch 卷积把图像切成 patch 并映射到 width 维（embedding）。
        2. 展平为 token 序列，拼接 CLS token，加上位置编码。
        3. LayerNorm -> Transformer -> 取 CLS token -> LayerNorm -> 线性投影。
    """

    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        """
        Args:
            input_resolution (int): 输入图像边长（如 224）。
            patch_size (int): 每个图像块（patch）的边长（ViT-B/32 为 32）。
            width (int): token 特征维度（ViT-B/32 为 768）。
            layers (int): Transformer 层数。
            heads (int): 每层注意力头数。
            output_dim (int): 输出嵌入维度（统一嵌入空间，通常 512）。
        """
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        # patch 卷积：kernel_size=patch_size, stride=patch_size，实现"分块 + 线性映射"
        # 对 ViT-B/32：输入 224x224x3 -> 7x7x768
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        # Kaiming 均匀初始化卷积权重
        init.kaiming_uniform_(self.conv1.weight, a=math.sqrt(5))

        # 注册 hook 打印 conv1 权重梯度信息（用于调试梯度是否正常传播）
        self.conv1.weight.register_hook(lambda grad: print("conv1 weight grad norm:", grad.norm().item()))

        # 缩放因子（与 GPT 初始化类似，保证初始方差合理）
        scale = width ** -0.5
        # CLS token（可学习）
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        # 位置编码：patch 数量 + 1 个 CLS token
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        # Transformer 前的 LayerNorm
        self.ln_pre = LayerNorm(width)
        # Transformer 编码器
        self.transformer = Transformer(width, layers, heads)
        # Transformer 后的 LayerNorm
        self.ln_post = LayerNorm(width)
        # 最终投影层：把 width 维投影到 output_dim（CLIP 中为 512 的统一嵌入空间）
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        # 说明：class_embedding / positional_embedding / proj 都是 nn.Parameter（可学习成员参数，
        # dtype 为 float32，随 model.to(device) 一起移动；加载官方权重后随 convert_weights 变为 fp16）。

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: 输入图像，[B, 3, input_resolution, input_resolution]，float32/fp16。
        Returns:
            图像语义向量，[B, output_dim]，dtype 同模型权重（fp16 模型返回 fp16）。
        """
        # 打印卷积前输入的均值和标准差（调试用）
        print("Before conv1: ", x.mean().item(), x.std().item())
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        print("After conv1: ", x.mean().item(), x.std().item())  # shape = [*, width, grid, grid]
        # 展平空间维：把 grid*grid 合并为一维 -> [*, width, grid**2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        # 转置为序列形式 -> [*, grid**2, width]
        x = x.permute(0, 2, 1)
        # 在序列最前面拼接 CLS token -> [*, grid**2+1, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)
        # 加上位置编码
        x = x + self.positional_embedding.to(x.dtype)
        # 进入 Transformer 前的 LayerNorm
        x = self.ln_pre(x)
        # 转置成 LND（序列长度, batch, 维度），满足 nn.MultiheadAttention 的输入要求
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        # 转回 NLD
        x = x.permute(1, 0, 2)  # LND -> NLD
        # 只取 CLS token（第 0 个 token），再 LayerNorm
        x = self.ln_post(x[:, 0, :])
        # 投影到统一嵌入空间（output_dim 维）
        # 矩阵乘 x @ self.proj：最后一维 width 与 proj 的第 0 维做乘法，输出 [*, output_dim]；
        # 其余维（如 batch）当作 batch 维保留。
        if self.proj is not None:
            x = x @ self.proj
        return x


class CLIP(nn.Module):
    """
    完整的 CLIP 双塔模型：视觉编码器 + 文本编码器。

    - encode_image(image)：把图像映射到 embed_dim 维语义向量。
    - encode_text(text)：把 token 序列映射到 embed_dim 维语义向量。
    - forward(image, text)：返回图像-文本的余弦相似度 logits（用于对比学习训练）。
    """

    def __init__(self,
                 embed_dim: int,
                 # vision 相关超参
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text 相关超参
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        """
        Args:
            embed_dim (int): 图像/文本统一嵌入维度（通常 512）。
            image_resolution (int): 视觉编码器期望的输入图像边长（如 224）。
            vision_layers (tuple[int,...] 或 int): ResNet 变体为 4 元组（各 stage 块数），
                ViT 变体为 int（层数）。
            vision_width (int): 视觉编码器基础宽度（ResNet stem 宽度 / ViT token 维度）。
            vision_patch_size (int): ViT 的 patch 边长；ResNet 变体传 None。
            context_length (int): 文本最大 token 数（通常 77）。
            vocab_size (int): 词表大小。
            transformer_width (int): 文本 Transformer 的 token 维度。
            transformer_heads (int): 文本 Transformer 每层注意力头数。
            transformer_layers (int): 文本 Transformer 层数。
        """
        super().__init__()

        # context_length: 普通成员（int，不可学习），文本最大 token 数（如 77），构造期使用
        self.context_length = context_length

        # ---- 选择视觉编码器 ----
        # 若 vision_layers 是元组/列表 -> 使用 ModifiedResNet（RN 系列）
        if isinstance(vision_layers, (tuple, list)):
            # ResNet 注意力池化的 head 数 = 特征维度 / 64
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        # 否则是 int -> 使用 VisionTransformer（ViT 系列）
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        # ---- 文本编码器（Transformer + 因果注意力掩码）----
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        # vocab_size: 普通成员（int，不可学习），词表大小，构造期使用
        self.vocab_size = vocab_size
        # 词嵌入表：可学习模块，权重形状 [vocab_size, transformer_width]，float32；
        # 把 token id（int64）映射为 transformer_width 维稠密向量
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        # 文本位置编码
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        # 文本 Transformer 后的 LayerNorm
        self.ln_final = LayerNorm(transformer_width)

        # 文本特征投影到统一嵌入空间（embed_dim 维）
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        # 可学习的温度系数（logit scale），初始化为 log(1/0.07)，即温度约 0.07
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # 初始化所有参数
        self.initialize_parameters()

    def initialize_parameters(self):
        """
        初始化所有可学习参数（词嵌入、位置编码、注意力投影、MLP、bn3 的 gamma 等）。
        无参数、无返回值；仅在构造期调用一次。
        """
        # 词嵌入与位置编码初始化
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        # ResNet 分支：初始化注意力池化的 QKV/输出投影
        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            # 将每个 Bottleneck 最后一个 BN（bn3）的 gamma 初始化为 0，
            # 使残差块初始近似为恒等映射，利于深层网络训练
            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        # 文本 Transformer 各层的初始化（参考 GPT 系列的做法）
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        # 文本投影矩阵初始化
        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        """
        构造因果注意力掩码（下三角掩码）。
        PyTorch 使用"加性"掩码，因此用 -inf 填充上三角（被屏蔽的位置）。

        Returns:
            mask: [context_length, context_length] 的 float32 张量，
                  下三角为 0（可见），上三角为 -inf（屏蔽）；不参与学习（非 Parameter）。
        """
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # 保留对角线及以下（下三角），上三角置为 -inf
        return mask

    @property
    def dtype(self):
        # 以视觉编码器卷积权重的 dtype 作为模型的 dtype
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        """
        Args:
            image: 预处理后的图像，[B, 3, H, W]，float32 或 fp16（内部自动对齐模型 dtype）。
        Returns:
            图像语义向量，[B, embed_dim]，dtype 同模型权重。
        """
        # 图像编码：先把输入转成模型 dtype，再送入视觉编码器
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        """
        Args:
            text: token id 序列，[batch_size, n_ctx]，int64（由 clip.tokenize 生成，
                  已 padding 到固定长度 n_ctx=context_length）。
        Returns:
            文本语义向量，[batch_size, embed_dim]，dtype 同模型权重。
        """
        # text: [batch_size, n_ctx] 的 token id 序列
        # 词嵌入 -> [batch_size, n_ctx, d_model]
        x = self.token_embedding(text).type(self.dtype)

        # 加上位置编码
        x = x + self.positional_embedding.type(self.dtype)
        # NLD -> LND（适配 MultiheadAttention）
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        # LND -> NLD
        x = x.permute(1, 0, 2)
        # 最终 LayerNorm
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # 取每个序列中 EOT（end-of-text）token 位置的特征：
        # text.argmax(dim=-1) 返回每行最大值（EOT token id 最大）的索引
        # 矩阵乘：[batch_size, width] @ [width, embed_dim]，把宽度维投影到统一嵌入空间
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, image, text):
        """
        Args:
            image: 图像张量，[B, 3, H, W]，float32/fp16。
            text: token id 序列，[B, n_ctx]，int64。
        Returns:
            logits_per_image: [B, B] float，行 i 是第 i 张图与所有文本的相似度打分。
            logits_per_text: [B, B] float，前一者的转置。
            （对角线位置为配对图文对，对比学习训练时以此为监督信号。）
        """
        # image: [B, 3, H, W]；text: [B, n_ctx]（token id，int64）
        # 分别编码图像和文本 -> 各 [B, embed_dim]
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # L2 归一化，使特征位于单位超球面上
        # 广播：norm 结果 [B, 1] 与 [B, embed_dim] 从右往左对齐后逐元素相除
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # logits（未归一化打分）：对角线位置是"配对"图文对的相似度，非对角是"错配"的。
        # logit_scale 是可学习标量参数（成员、float32），exp 后作为温度系数的倒数乘在相似度上，
        # 数值越大，softmax 后的分布越尖锐（越自信）。
        # 矩阵乘 img @ text^T：[B, embed] 与 [embed, B] 相乘，embed 维做内积，得到 [B, B] 相似度矩阵。
        # 余弦相似度作为 logits（乘以温度系数）
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """
    将模型中适用的参数转换为 fp16（半精度），用于混合精度训练/推理。
    """

    def _convert_weights_to_fp16(l):
        # 卷积与线性层权重/偏置 -> half
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        # 多头注意力层：所有投影权重/偏置 -> half
        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        # 文本/视觉投影参数 -> half
        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict):
    """
    根据一个 state_dict 推断 CLIP 模型结构（超参）并构建、加载权重。

    通过检查 state_dict 中是否存在 "visual.proj" 判断是 ViT 还是 ResNet，
    再从各层权重形状反推 image_resolution、层数、宽度等超参。
    """
    # 判断是否为 ViT：ViT 有 "visual.proj" 参数，而 ResNet 没有
    vit = "visual.proj" in state_dict

    if vit:
        # ---- ViT 分支 ----
        vision_width = state_dict["visual.conv1.weight"].shape[0]  # patch 卷积输出通道数
        # 通过统计 transformer 中 attention 层数得到 ViT 的层数
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]  # patch 尺寸
        # 由位置编码数量反推 grid 尺寸：位置编码数 - 1（去掉 CLS）后开方
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        # ---- ResNet 分支 ----
        # 统计每个 stage 中 Bottleneck 的数量（用 layer 内部 conv1 的独有索引数表示）
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]  # 基础宽度
        # 由注意力池化位置编码反推空间尺寸
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32  # 输入分辨率 = 空间尺寸 * 32（总下采样倍数）

    # ---- 文本编码器超参 ----
    embed_dim = state_dict["text_projection"].shape[1]          # 统一嵌入维度（通常 512）
    context_length = state_dict["positional_embedding"].shape[0]  # 上下文长度
    vocab_size = state_dict["token_embedding.weight"].shape[0]  # 词表大小
    transformer_width = state_dict["ln_final.weight"].shape[0]  # 文本 Transformer 宽度
    transformer_heads = transformer_width // 64                 # 注意力头数（每头 64 维）
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))  # 文本层数

    # 构建 CLIP 模型
    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    # 这些键不是模型参数，需要从 state_dict 中删除
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # 转为 fp16 后加载权重
    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()
