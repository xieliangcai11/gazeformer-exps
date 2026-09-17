"""
transformer_models.py
=====================
本文件实现了一系列 Transformer / 混合专家（MoE）相关的模块，
最终组合成用于 3D gaze 回归的 TransformerDeepSeek_gaze 模型。

整体结构（自底向上）：
    基础组件：
        - RMSNorm           : 均方根归一化（LLaMA 风格）
        - Attention         : 基于 nn.MultiheadAttention 的标准多头注意力封装
        - FlashAttention    : 基于 flash_attn 的高效注意力封装
        - FeedForward       : 标准前馈网络（SiLU 激活）
        - Router            : 简单 MoE 路由器（线性 + softmax）
        - MixtureOfExperts  : 简单 MoE 层（top-k 专家加权求和，注意其中有未定义变量，属历史遗留代码）
        - Gate              : DeepSeek 风格 MoE 门控机制
        - Expert            : DeepSeek 风格专家层（SwiGLU）
        - MoE               : DeepSeek 风格 MoE 模块（gate + 路由专家 + 共享专家）
        - PositionalEncoding: 正弦位置编码

    组合组件：
        - TransformerBlock         : 自注意力 + 交叉注意力 + MoE 的 Transformer 块
        - Transformer              : 用 TransformerBlock 堆叠的简单 Transformer
        - Block                    : DeepSeek 风格 Transformer 块（注意力 + MoE/FF）
        - BlockMoba                : 用"标准注意力"替代 MoBA 注意力的 Block 变体

    顶层模型：
        - TransformerDeepSeek_gaze : 将 CLIP 特征（feature_1/feature_2）、CNN 特征图 token
                                     （feature_3）、以及 ViT patch token 投影到统一维度后，
                                     拼接 CLS token 送入多层 BlockMoba，最后回归 3D gaze 向量。

说明：本文件中 MoE/BlockMoba 等命名参考了 DeepSeek-V3 / MoBA 架构，
      但 BlockMoba 内实际使用的是标准缩放点积注意力（并非真正的 MoBA 稀疏注意力）。
"""

import random
import os
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from flash_attn.modules.mha import MHA

# from torch.nn import TransformerEncoder, TransformerEncoderLayer

import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

# from nn_common import check_nan, check_nan_is_all
# 设置 cuBLAS 工作区配置，保证某些 op 的确定性
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# 分布式训练相关全局变量（当前按单卡配置）
world_size = 1
rank = 0


@dataclass
class ModelArgs:
    """
    用于定义模型参数与超参数的数据类（参考 DeepSeek-V3 的配置项）。

    Attributes:
        max_batch_size (int): 最大 batch size。
        max_seq_len (int): 最大序列长度。
        dtype (Literal["bf16", "fp8"]): 计算所用数据类型。
        vocab_size (int): 词表大小。
        dim (int): 模型维度。
        inter_dim (int): MLP 层的中间维度。
        moe_inter_dim (int): MoE 层的中间维度。
        n_layers (int): Transformer 层数。
        n_dense_layers (int): 模型中密集（非 MoE）层数。
        n_heads (int): 注意力头数。
        n_routed_experts (int): MoE 中被路由的专家数量。
        n_shared_experts (int): MoE 中共享专家数量。
        n_activated_experts (int): MoE 中每个输入激活的专家数量。
        n_expert_groups (int): 专家分组数量。
        n_limited_groups (int): MoE 路由中限制的组数量。
        score_func (Literal["softmax", "sigmoid"]): MoE 路由打分函数。
        route_scale (float): 路由打分的缩放因子。
        q_lora_rank (int): query 投影的 LoRA 秩。
        kv_lora_rank (int): key-value 投影的 LoRA 秩。
        qk_nope_head_dim (int): 不含位置编码的 query-key 投影维度。
        qk_rope_head_dim (int): 含旋转位置编码的 query-key 投影维度。
        v_head_dim (int): value 投影维度。
        original_seq_len (int): 原始序列长度。
        rope_theta (float): 旋转位置编码的基。
        rope_factor (float): 扩展序列长度的缩放因子。
        beta_fast (int): 快速 beta 校正因子。
        beta_slow (int): 慢速 beta 校正因子。
        mscale (float): 扩展注意力的缩放因子。
    """
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    # moe 相关
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.
    # mla 相关
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn 相关
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.


class RMSNorm(torch.nn.Module):
    """
    RMSNorm（Root Mean Square Layer Normalization）。

    与 LayerNorm 不同，RMSNorm 不减去均值，只用均方根做归一化，计算更快，
    是 LLaMA / DeepSeek 等大模型常用的归一化方式。
    公式：RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 可学习的缩放参数 gamma
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # 用均方根归一化（rsqrt 即 1/sqrt）
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 先在 float32 下计算，再转回原 dtype，最后乘以可学习权重
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Attention(nn.Module):
    """
    标准多头注意力封装（batch_first=True）。

    注意：这里没有显式的 QKV 投影，nn.MultiheadAttention 内部自带 in_proj（QKV 投影）和 out_proj。
    """

    def __init__(self, embed_dim, num_heads, batch_first=True):
        super(Attention, self).__init__()
        # embed_dim 必须能被 num_heads 整除（由 nn.MultiheadAttention 校验）
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=batch_first)

    def forward(self, query, key, value, mask=None):
        # 返回注意力输出（忽略注意力权重）
        attn_output, _ = self.attention(query, key, value, attn_mask=mask)
        return attn_output


class FlashAttention(nn.Module):
    """
    基于 flash_attn 库的高效注意力封装。

    Args:
        embed_dim      : 嵌入维度。
        num_heads      : 注意力头数。
        cross_attn     : 是否为交叉注意力。
        use_flash_attn : 是否使用 flash attention。
        return_residual: 是否返回残差（输入本身）。
    """

    def __init__(self, embed_dim, num_heads, cross_attn = False, use_flash_attn = True, return_residual = False):
        super().__init__()
        self.cross_attn = cross_attn
        self.return_residual = return_residual
        self.attention = MHA(embed_dim, num_heads, cross_attn = cross_attn, use_flash_attn = use_flash_attn, return_residual=return_residual)

    def forward(self, x, x_kv=None, mask=None):
        if not self.cross_attn:
            # 自注意力：query=key=value=x
            attn_output = self.attention(x, x_kv=None, key_padding_mask=mask)
        else:
            # 交叉注意力：query=x, key/value=x_kv
            assert x_kv is not None
            attn_output = self.attention(x, x_kv=x_kv, key_padding_mask=mask)
        # 返回输出；若开启 return_residual，则额外返回输入作为残差
        return attn_output if not self.return_residual else (attn_output, x)


class FeedForward(nn.Module):
    """
    标准前馈网络（FFN）：
        x -> Linear(embed_dim -> inter_dim) -> SiLU -> Linear(inter_dim -> embed_dim)
    """

    def __init__(self, embed_dim, inter_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, inter_dim)  # 升维
        self.fc2 = nn.Linear(inter_dim, embed_dim)  # 降维回原维度
        # self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = F.silu(self.fc1(x))  # Swish/SiLU 激活
        # x = self.dropout(x)
        # x = self.dropout(x) # remove dropout
        x = self.fc2(x)
        return x


class Router(nn.Module):
    """
    简单 MoE 路由器：线性层 + softmax，输出每个专家被选择的概率。
    """

    def __init__(self, input_dim, num_experts):
        super(Router, self).__init__()
        # 门控线性层：input_dim -> num_experts
        self.gating = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        # 计算专家 logits
        expert_logits = self.gating(x)  # Shape: [batch_size, num_experts]
        # softmax 得到专家选择概率
        expert_probs = F.softmax(expert_logits, dim=-1)
        return expert_probs


class MixtureOfExperts(nn.Module):
    """
    简单 Mixture-of-Experts 层（top-k 专家加权求和）。

    注意：本类的 forward 中存在未定义的变量（如 j、sample_indices），
    属于历史遗留/未完成的实现，实际项目中未使用此类（实际使用下面的 MoE 类）。
    """

    def __init__(self, input_dim, output_dim, num_experts, capacity):
        """
        初始化 MoE 层。

        :param input_dim  : 输入特征维度。
        :param output_dim : 输出特征维度。
        :param num_experts: 专家数量。
        :param capacity   : 每个输入激活的 top-k 专家数量。
        """
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.capacity = capacity
        # 每个专家就是一个线性层
        self.experts = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(num_experts)])
        self.router = Router(input_dim, num_experts)

    def forward(self, x):
        """
        前向过程。

        :param x: 输入张量，形状 (batch_size, seq_length, input_dim) 或 (batch_size, input_dim)。
        :return: 输出张量，形状 (batch_size, seq_length, output_dim) 或 (batch_size, output_dim)。
        """
        if x.dim() == 2:  # 情况：(batch_size, input_dim)
            batch_size, input_dim = x.size()
            seq_length = 1
            x = x.unsqueeze(1)  # 增加一个虚拟的序列长度维度
        elif x.dim() == 3:  # 情况：(batch_size, seq_length, input_dim)
            batch_size, seq_length, input_dim = x.size()
        else:
            raise ValueError(f"Unsupported input dimensions: {x.shape}")

        # 展平 batch 和序列维度，便于路由
        x_flat = x.view(batch_size * seq_length, input_dim)

        # 计算专家选择概率
        expert_probs = self.router(x_flat)  # Shape: [batch_size * seq_length, num_experts]
        #debug
        # print("MoE.forward: x_flat shape:", x_flat.shape)
        # print("MoE.forward: expert_probs stats: min={:.4f}, max={:.4f}, mean={:.4f}".format(
        #     expert_probs.min().item(), expert_probs.max().item(), expert_probs.mean().item()))

        # 取 top-k 专家及其概率
        top_k = torch.topk(expert_probs, self.capacity, dim=-1)
        selected_experts = top_k.indices  # Shape: [batch_size * seq_length, capacity]
        selected_probs = top_k.values  # Shape: [batch_size * seq_length, capacity]
        # #debug
        # print("MoE.forward: selected_experts shape:", selected_experts.shape)
        # print("MoE.forward: selected_probs stats: min={:.4f}, max={:.4f}, mean={:.4f}".format(
        #     selected_probs.min().item(), selected_probs.max().item(), selected_probs.mean().item()))

        # 准备输出张量
        output_dim = self.experts[0].out_features
        outputs = torch.zeros(batch_size * seq_length, output_dim, device=x.device)

        # 处理 top-k 专家
        for i in range(self.capacity):
            expert_index = selected_experts[:, i]  # 每个样本选中的第 i 个专家索引
            expert_weight = selected_probs[:, i]  # 对应的专家权重

            # 注意：以下代码引用了未定义的 j 与 sample_indices，为历史遗留问题，保留原文
            print(f"MoE.forward: Expert {j} processes {sample_indices.numel()} samples")

            # 将输入路由到被选中的专家
            expert_outputs = torch.cat([
                self.experts[j](x_flat[expert_index == j])  # 专家 j 处理对应输入
                if (expert_index == j).sum() > 0 else torch.zeros(0, output_dim, device=x.device)
                for j in range(self.num_experts)
            ], dim=0)

            # 加权累加输出
            outputs += expert_outputs * expert_weight.unsqueeze(-1)

        # 恢复原始形状
        outputs = outputs.view(batch_size, seq_length, output_dim)
        if seq_length == 1:  # 如果之前添加了虚拟序列维，则去掉
            outputs = outputs.squeeze(1)

        return outputs


class Gate(nn.Module):
    """
    DeepSeek 风格 MoE 的门控机制。

    功能：对每个 token 计算各路由专家的打分，选 top-k 专家并返回对应的（归一化）权重和索引。

    Attributes:
        dim (int): 输入特征维度。
        topk (int): 每个输入激活的专家数量。
        n_groups (int): 分组数量。
        topk_groups (int): 每组激活的组数量。
        score_func (str): 打分函数（'softmax' 或 'sigmoid'）。
        route_scale (float): 路由权重缩放因子。
        weight (torch.nn.Parameter): 门控可学习权重。
        bias (Optional[torch.nn.Parameter]): 可选偏置项。
    """

    def __init__(self, embed_dim, n_routed_experts, n_activated_experts, n_expert_groups, n_limited_groups, score_func="softmax", route_scale=1.0):
        """
        初始化 Gate 模块。

        Args:
            embed_dim           : 嵌入维度。
            n_routed_experts    : 路由专家数量。
            n_activated_experts : 激活专家数量（topk）。
            n_expert_groups     : 专家分组数。
            n_limited_groups    : 限制的组数量。
            score_func          : 打分函数。
            route_scale         : 路由缩放因子。
        """
        super().__init__()
        self.dim = embed_dim
        self.topk = n_activated_experts
        self.n_groups = n_expert_groups
        self.topk_groups = n_limited_groups
        self.score_func = score_func
        self.route_scale = route_scale
        # 门控权重矩阵：[n_routed_experts, embed_dim]
        self.weight = nn.Parameter(torch.empty(n_routed_experts, embed_dim))

        # Xavier/Glorot 初始化
        torch.nn.init.xavier_uniform_(self.weight)

        # 当维度为 7168 时启用偏置（历史特定配置，一般 dim 下为 None）
        self.bias = nn.Parameter(torch.empty(n_routed_experts)) if self.dim == 7168 else None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        门控前向。

        Args:
            x (torch.Tensor): 输入张量 [N, dim]。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (路由权重, 选中的专家索引)。
        """
        # 计算打分 logits：x @ weight^T
        scores = F.linear(x, self.weight)

        if self.score_func == "softmax":
            # 减去最大值做数值稳定，再 softmax
            scores = scores - scores.max(dim=-1, keepdim=True)[0]
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            # sigmoid 打分
            scores = scores.sigmoid()

        # 保存原始打分（用于最终取权重，避免组掩码/偏置影响权重取值）
        original_scores = scores
        # 加偏置
        if self.bias is not None:
            scores = scores + self.bias
        # 若分组数 > 1，则先按组选 topk_groups，再在组内选专家
        if self.n_groups > 1:
            scores = scores.view(x.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = torch.zeros_like(scores[..., 0]).scatter_(1, indices, True)
            scores = (scores * mask.unsqueeze(-1)).flatten(1)
        # 取 top-k 专家的索引
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        # 从原始打分中取出对应权重
        weights = original_scores.gather(1, indices)

        # 避免过小/零权重
        weights = torch.clamp(weights, min=1e-7)
        if self.score_func == "sigmoid":
            # sigmoid 打分需要归一化
            weights /= weights.sum(dim=-1, keepdim=True)
        # 路由权重缩放
        weights *= self.route_scale

        return weights.type_as(x), indices


class Expert(nn.Module):
    """
    DeepSeek 风格专家层（SwiGLU 结构）：
        output = w2( SiLU(w1(x)) * w3(x) )
    """

    def __init__(self, dim: int, inter_dim: int):
        """
        Args:
            dim       : 输入/输出维度。
            inter_dim : 隐藏层维度。
        """
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim)  # 门控线性层（激活）
        self.w2 = nn.Linear(inter_dim, dim)  # 输出线性层
        self.w3 = nn.Linear(dim, inter_dim)  # 另一路线性层

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU：SiLU(w1(x)) 与 w3(x) 逐元素相乘后，经 w2 输出
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoE(nn.Module):
    """
    DeepSeek 风格 Mixture-of-Experts 模块。

    组成：
        - gate          : 门控（选 top-k 专家并给出权重）。
        - experts       : 路由专家列表（ModuleList）。
        - shared_experts: 对所有输入都生效的共享专家（FFN）。

    前向：对每个 token 用 gate 选专家，加权聚合专家输出，再加上共享专家输出。
    """

    def __init__(self, embed_dim, n_routed_experts, n_activated_experts, n_shared_experts, moe_inter_dim):
        """
        Args:
            embed_dim           : 嵌入维度。
            n_routed_experts    : 路由专家总数。
            n_activated_experts : 每个输入激活的专家数。
            n_shared_experts    : 共享专家数量。
            moe_inter_dim       : 每个专家的中间维度。
        """
        super().__init__()
        self.dim = embed_dim
        self.n_routed_experts = n_routed_experts
        # 本地（当前卡）专家数量（单卡时等于总数）
        self.n_local_experts = n_routed_experts // world_size
        self.n_activated_experts = n_activated_experts
        # 当前 rank 负责的专家索引区间
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        # 门控
        self.gate = Gate(embed_dim, n_routed_experts, n_activated_experts, n_expert_groups=1, n_limited_groups=1, score_func="softmax", route_scale=1.0)
        # 专家列表：仅当前 rank 负责的区间内构造真正的 Expert，其余为 None（分布式下由其他卡持有）
        self.experts = nn.ModuleList([Expert(embed_dim, moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
                                      for i in range(self.n_routed_experts)])
        # 共享专家（本质是一个 FFN，中间维度 = 共享专家数 * 单个专家中间维度）
        self.shared_experts = FeedForward(embed_dim, n_shared_experts * moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向过程：
          1. 将输入展平为 (N, dim)。
          2. 通过 gate 得到每个样本的 topk 专家索引和对应权重（形状均为 [N, topk]）。
          3. 对每个路由专家，用 mask 找出选择它的样本，并用 index_add_ 聚合加权输出。
          4. 加上共享专家输出后恢复原始形状。
        """
        shape = x.size()
        # 展平：[N, dim]
        x_flat = x.view(-1, self.dim)
        # gate 输出：weights, indices 均为 [N, topk]
        weights, indices = self.gate(x_flat)
        N, topk = weights.shape

        # 结果缓冲，初始为 0
        y = torch.zeros_like(x_flat)
        for expert_idx in range(self.n_routed_experts):
            # 找出哪些样本选择了 expert_idx（布尔 mask）
            mask = (indices == expert_idx)  # [N, topk] boolean
            if mask.sum() == 0:
                continue  # 没有样本选择该专家则跳过
            # nonzero 返回形状 [M, 2]：第一列为样本索引，第二列为该样本在 topk 中的位置
            sel = mask.nonzero(as_tuple=False)
            sample_indices = sel[:, 0]  # [M]
            # 对应权重（若某样本多次选中同一专家则自动累加）
            weight_values = weights[mask].unsqueeze(-1)  # [M, 1]
            # 计算专家输出
            expert = self.experts[expert_idx]
            expert_output = expert(x_flat[sample_indices])  # [M, dim]
            # 用 index_add_ 把加权输出累加到 y（同一样本多次命中会累加）
            y.index_add_(0, sample_indices, expert_output * weight_values)
        # 共享专家输出（对所有样本）
        z = self.shared_experts(x_flat)
        out_flat = y + z
        # 恢复原始形状
        return out_flat.view(shape)


class TransformerBlock(nn.Module):
    """
    一个包含 自注意力 + 交叉注意力 + MoE 的 Transformer 块。

    结构：
        x = x + SelfAttention(x)
        x = x + CrossAttention(x, cross_input)
        x = x + MoE(x)
    每步后接 RMSNorm（Pre-Norm 写法）。
    """

    def __init__(self, embed_dim, num_heads, ff_dim, num_experts, capacity):
        super(TransformerBlock, self).__init__()
        self.self_attn = Attention(embed_dim, num_heads)   # 自注意力
        self.cross_attn = Attention(embed_dim, num_heads)  # 交叉注意力
        # self.ff = FeedForward(embed_dim, ff_dim)
        self.moe = MixtureOfExperts(embed_dim, embed_dim, num_experts, capacity)  # 简单 MoE
        # self.moe = MOE()
        self.norm1 = RMSNorm(embed_dim, eps=1e-5)
        self.norm2 = RMSNorm(embed_dim, eps=1e-5)
        self.norm3 = RMSNorm(embed_dim, eps=1e-5)

    def forward(self, x, cross_input, mask=None):
        # 自注意力 + 残差 + 归一化
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + attn_output)

        # 交叉注意力 + 残差 + 归一化
        cross_attn_output = self.cross_attn(x, cross_input, cross_input, mask)
        x = self.norm2(x + cross_attn_output)

        # MoE + 残差 + 归一化
        # ff_output = self.ff(x)
        moe_output = self.moe(x)
        # x = self.norm3(x + ff_output + moe_output)
        x = self.norm3(x + moe_output)

        return x


class PositionalEncoding(nn.Module):
    """
    正弦位置编码（Sinusoidal Positional Encoding）。

    用 sin/cos 函数构造与位置相关的编码矩阵，无需训练。
    """

    def __init__(self, embed_dim, max_seq_len=8192):
        """
        Args:
            embed_dim   : 嵌入维度。
            max_seq_len : 最大序列长度。
        """
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim

        # 位置索引 (max_seq_len, 1)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        # 频率因子：exp(-log(10000)/embed_dim * 2i)，对应不同频率
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))

        # 构造编码矩阵
        encoding = torch.zeros(max_seq_len, embed_dim)
        encoding[:, 0::2] = torch.sin(position * div_term)  # 偶数索引用 sin
        encoding[:, 1::2] = torch.cos(position * div_term)  # 奇数索引用 cos

        # 增加 batch 维度 (1, max_seq_len, embed_dim)
        encoding = encoding.unsqueeze(0)
        # 注册为 buffer（不参与反向传播）
        self.register_buffer('positional_encoding', encoding)

    def forward(self, x):
        """
        返回与输入序列长度对应的位置编码。

        Args:
            x: 输入张量，形状 (batch_size, seq_len, embed_dim)。
        Returns:
            位置编码张量，形状 (1, seq_len, embed_dim)。
        """
        seq_len = x.size(1)
        # return x + self.positional_encoding[:, :seq_len]
        return self.positional_encoding[:, :seq_len]


class Transformer(nn.Module):
    """
    用上面的 TransformerBlock 堆叠而成的简单 Transformer。

    注意：此处 embedding 用的是 nn.Embedding(seq_length, embed_dim)，
    其中 seq_length 被当作"词表大小"使用，属于该旧实现的特殊设计。
    """

    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, num_experts, capacity, seq_length):
        super(Transformer, self).__init__()
        # 嵌入层（把 token 索引映射为 embed_dim 向量）
        self.embedding = nn.Embedding(seq_length, embed_dim)
        print(self.embedding)
        # 位置编码
        self.pos_embedding = PositionalEncoding(embed_dim, max_seq_len=seq_length)
        # 堆叠 num_layers 个 TransformerBlock
        self.layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, num_experts, capacity) for _ in range(num_layers)
        ])

    def forward(self, x, cross_input, mask=None):
        # 词嵌入 + 位置编码
        x = self.embedding(x) + self.pos_embedding(x)

        # 逐层前向
        for layer in self.layers:
            x = layer(x, cross_input, mask)
        return x


class Block(nn.Module):
    """
    DeepSeek 风格 Transformer 块：注意力 + （MoE 或 前馈网络）。

    - 前 n_dense_layers 层使用普通 FFN（dense）。
    - 其余层使用 MoE。
    支持自注意力（cross_input=None）和交叉注意力。
    """

    def __init__(self, embed_dim, num_heads, n_routed_experts, n_activated_experts, n_shared_experts, moe_inter_dim, inter_dim = 10944, layer_id=None):
        super(Block, self).__init__()
        self.self_attn = Attention(embed_dim, num_heads)
        self.n_dense_layers = 3  # 前 3 层为 dense FFN
        n_dense_layers = self.n_dense_layers

        # 根据层 id 决定用 FFN 还是 MoE
        if layer_id is not None:
            self.moe = FeedForward(embed_dim, inter_dim) if layer_id < n_dense_layers else MoE(embed_dim,
                                                                                               n_routed_experts,
                                                                                               n_activated_experts,
                                                                                               n_shared_experts,
                                                                                               moe_inter_dim)
        else:
            self.moe = MoE(embed_dim, n_routed_experts, n_activated_experts,
                           n_shared_experts,
                           moe_inter_dim)

        self.norm1 = RMSNorm(embed_dim, eps=1e-5)
        self.norm3 = RMSNorm(embed_dim, eps=1e-5)

    def forward(self, x, cross_input=None, mask=None):
        # Pre-Norm 后做注意力
        x_norm = self.norm1(x)
        if cross_input is None:
            # 自注意力
            attn_output = self.self_attn(x_norm, x_norm, x_norm, mask)
            attn_output = x + attn_output
        else:
            # 交叉注意力
            cross_input_norm = self.norm1(cross_input)
            attn_output = self.self_attn(x_norm, cross_input_norm, cross_input_norm, mask)
            attn_output = x + attn_output

        # MoE / FFN + 残差
        moe_output = self.moe(self.norm3(attn_output))
        moe_output = attn_output + moe_output

        return moe_output


class BlockMoba(nn.Module):
    """
    替换原 Block 的标准注意力为"moba 注意力"（实际实现为标准缩放点积注意力），
    保留 MoE 逻辑与相同的 init 参数。

    命名源自 MoBA（Mixture of Block Attention），但本实现中 _moba_attention
    使用的是标准多头注意力，未实现真正的稀疏/块注意力。
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        n_routed_experts,
        n_activated_experts,
        n_shared_experts,
        moe_inter_dim,
        inter_dim=10944,
        layer_id=None,
        moba_chunk_size=5,
        moba_topk=2
    ):
        super(BlockMoba, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # MoBA 相关参数（当前未真正使用）
        self.moba_chunk_size = moba_chunk_size
        self.moba_topk = moba_topk

        # 前 9 层为 dense FFN，其余层为 MoE
        self.n_dense_layers = 9
        if layer_id is not None and layer_id < self.n_dense_layers:
            self.moe = FeedForward(embed_dim, inter_dim)
        else:
            self.moe = MoE(embed_dim, n_routed_experts, n_activated_experts, n_shared_experts, moe_inter_dim)
        self.norm1 = RMSNorm(embed_dim, eps=1e-5)
        self.norm3 = RMSNorm(embed_dim, eps=1e-5)

    def forward(self, x, cross_input=None, mask=None):
        """
        Args:
            x           : [batch, seq_len, embed_dim]。
            cross_input : 若需要交叉注意力，可传入另一序列。
            mask        : [batch, seq_len, seq_len] 可选注意力掩码。
        """
        x_norm = self.norm1(x)

        if cross_input is None:
            # 自注意力
            attn_output = self._moba_attention(x_norm, x_norm, x_norm, mask)
        else:
            # 交叉注意力（query=x_norm, key/value=cross_input_norm）
            cross_input_norm = self.norm1(cross_input)
            attn_output = self._moba_attention(x_norm, cross_input_norm, cross_input_norm, mask)

        # 残差
        out = x + attn_output

        # MoE / FF 处理 + 残差
        moe_output = self.moe(self.norm3(out))
        out = out + moe_output

        return out

    def _moba_attention(self, q, k, v, mask=None):
        """
        使用标准缩放点积多头注意力替代真正的 moba_attn_varlen。

        Args:
            q, k, v: 形状 [bsz, seqlen, d_model]。
            mask   : 可选掩码。
        Returns:
            注意力输出，形状 [bsz, seqlen, d_model]。
        """
        bsz, seqlen, d_model = q.shape
        head_dim = d_model // self.num_heads

        # 拆分为多头：q/k/v -> [bsz, num_heads, seqlen, head_dim]
        q = q.view(bsz, seqlen, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.num_heads, head_dim).transpose(1, 2)

        # 缩放点积注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        if mask is not None:
            # 被掩码位置置为很小的数（-1e9），softmax 后接近 0
            scores = scores.masked_fill(mask == 0, -1e9)

        # softmax 归一化
        attn_weights = torch.softmax(scores, dim=-1)
        # 加权求和 value
        attn_output = torch.matmul(attn_weights, v)

        # 合并多头并恢复原始形状 [bsz, seqlen, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        return attn_output


class TransformerDeepSeek_gaze(nn.Module):
    """
    顶层 gaze 回归 Transformer（DeepSeek 风格 + MoE）。

    输入（raw_inputs 字典）：
        - feature_1      : [B, 512]  CLIP 上下文补偿特征（图像 + 光照/头姿/背景）。
        - feature_2      : [B, 512]  CLIP 任务对齐特征（图像 + 视线方向标签）。
        - feature_3      : [B, C, H*W] 或 [B, N, C]  CNN 特征图展平后的 patch token。
        - token_img_patch: [B, num_patches, d_model]（可选）ViT 的 patch token。

    处理流程：
        1. 分别用 proj_f1/proj_f2/proj_f3/proj_patch 把所有特征投影到统一的 d_model 维。
        2. 每个特征展开为 token（feature_1/2 各 1 个 token，feature_3/patch 多个 token）。
        3. 拼接 CLS token + 所有 token，送入 num_layers 层 BlockMoba。
        4. 取 CLS token 经线性头回归输出 3 维 gaze 向量。
    """

    def __init__(
        self,
        num_layers,
        embed_dim,
        inter_dim,
        num_heads,
        n_routed_experts,
        n_activated_experts,
        n_shared_experts,
        moe_inter_dim,
        d_model=768,
        out_dim=3,
        dropout_rate=0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout_rate)  # Dropout

        # 各特征到统一维度 d_model 的投影层
        self.proj_f1 = nn.Linear(512, d_model)     # feature_1（CLIP 512 维）
        self.proj_f2 = nn.Linear(512, d_model)     # feature_2（CLIP 512 维）
        self.proj_f3 = nn.Linear(2048, d_model)    # feature_3（CNN layer4 通道 2048）
        self.proj_patch = nn.Linear(d_model, d_model)  # ViT patch token（已是 d_model 维）
        # 可学习 CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # 堆叠 num_layers 个 BlockMoba（前 9 层 FFN，其余 MoE）
        self.layers = nn.ModuleList([
            BlockMoba(
                d_model,
                num_heads,
                n_routed_experts,
                n_activated_experts,
                n_shared_experts,
                moe_inter_dim,
                inter_dim=inter_dim,
                layer_id=i,
                moba_chunk_size=5,
                moba_topk=2
            )
            for i in range(num_layers)
        ])
        # 输出头：CLS token -> 3D gaze
        self.linear_head = nn.Linear(d_model, out_dim)

    def forward(self, raw_inputs, mask=None):
        # ---------- 1. 各特征投影并构造成 token ----------
        # feature_1/feature_2：各投影为 1 个 token [B, 1, d_model]
        token_f1 = self.dropout(self.proj_f1(raw_inputs["feature_1"]).unsqueeze(1))
        token_f2 = self.dropout(self.proj_f2(raw_inputs["feature_2"]).unsqueeze(1))
        # feature_3：CNN 特征图展平后的 token 序列 [B, N, d_model]
        token_f3 = self.dropout(self.proj_f3(raw_inputs["feature_3"]))
        token_list = [token_f1, token_f2, token_f3]

        # 可选：ViT patch token
        if "token_img_patch" in raw_inputs and raw_inputs["token_img_patch"] is not None:
            token_patch = self.dropout(self.proj_patch(raw_inputs["token_img_patch"]))
            token_list.append(token_patch)

        # 沿序列维拼接所有 token -> [B, L, d_model]
        transformer_input = torch.cat(token_list, dim=1)

        # ---------- 2. 拼接 CLS token ----------
        batch_size = transformer_input.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        transformer_input = torch.cat([cls_tokens, transformer_input], dim=1)

        # ---------- 3. 逐层前向 ----------
        x = transformer_input
        for layer in self.layers:
            x = self.dropout(layer(x, cross_input=None, mask=mask))  # 每层后加 Dropout

        # ---------- 4. 取 CLS token 回归 gaze ----------
        global_repr = x[:, 0, :]  # [B, d_model]
        out = self.linear_head(global_repr)  # [B, 3]
        return out
