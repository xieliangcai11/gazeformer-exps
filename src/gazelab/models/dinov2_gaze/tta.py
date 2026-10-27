"""测试时自适应（TTA）：**不使用任何真值**的逐用户个性化。

设置来源（本仓库 docs/ 内）：
  · MetaPrompt (AAAI'24, "Test-Time Personalization with Meta Prompt for Gaze Estimation")：
    "personalization … **without labels**"——在测试时只更新极少量参数（prompt）。
  · Alfa (AAAI'26, "Low-Rank Personalized Gaze")：只用**少量无标签样本**做低秩适配。
  ⇒ 本模块沿用它们的**设置**（冻结骨干 + 测试时极少量参数 + 无标签目标域数据），
    但**不是**复现它们的损失函数（那需要原论文实现与超参）⇒ 目标函数是我们自己的设计，
    代码与文档都如此标注，避免误引。

为什么要这么做（本项目的实测动机，见 docs/experiments_2026-10-15.md）：
  · 我们的误差主要来自**逐用户的系统性偏置/收缩**（D1/D1b：真值/预测 std ≈1.13~1.15；
    跨被试不可迁移 ⇒ 必须逐用户修）；
  · 但"逐用户"以前需要该用户的**真值**（标定点）⇒ 真值消耗大；
  · ⇒ 若无标签也能把这些个人偏置修掉，就不需要任何真值标定 ✓。

无监督目标（三项，权重可配）：
  1) **输出矩匹配** L_moment：让目标域预测的**均值/标准差**匹配"训练集标签的分布先验"
     （先验由训练划分的标签离线统计得到，测试时不需要真值）。这一项直接针对上述偏置/收缩 ✓。
  2) **注视保持增强一致性** L_cons：光度抖动（亮度/对比度，**不做**水平翻转——翻转会改变注视）
     前后预测应一致 ⇒ 抑制对成像条件的敏感。
  3) **参数正则** L_reg：把适配器拉向"恒等"，防止无标签漂移（结构上"最多只赚不亏"的近似）。

架构：**非侵入式**——骨干与回归头全部冻结，只在 `head` 前注入低秩增量：
    head_in' = head_in + W2 @ (W1 @ h) + b      （h = 融合 CLS；r 很小，如 8）
  ⇒ 与含眼支流的现有模型兼容（`VPTModel` 是独立架构且不含眼支流，故不能直接用于旗舰 ✗）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class TTAConfig:
    """TTA 超参（默认值保守：小步数、小学习率、强正则 ⇒ 先保证不伤害）。"""
    steps: int = 300
    lr: float = 1e-3
    rank: int = 8                 # 低秩增量秩；0 = 只学偏置
    batch_size: int = 128
    w_moment: float = 1.0         # 输出矩匹配（我们的）权重
    w_cons: float = 1.0           # 光度一致性权重（亮度/对比度）
    #: 几何一致性权重：小仿射（缩放/平移）后注视不变 ⇒ 等价"增强一致性"。
    #: 与文献的"对称/旋转一致性"同族。2D 屏幕版本：镜像输入的预测应为
    #: (1 - x, y)（已在本项目模型上实测：翻转图 vs 镜像标签误差 0.1077→0.1079 ✓）。
    w_geo: float = 0.0
    #: 左右翻转对称（2D 屏幕版）：f(flip(x)) 的 x 应等于 1 - f(x) 的 x，y 不变。
    #: 与 Alfa 的 3D 对称损失同源；2D 需要镜像标签（已在数据增强中验证 ✓）。
    w_sym: float = 0.0
    geo_scale_amp: float = 0.05
    geo_shift_amp: float = 0.02
    w_reg: float = 1e-3           # 参数正则（拉向恒等）
    cons_jitter: float = 0.05     # 光度抖动幅度（增亮/对比度）
    grad_clip: float = 1.0
    seed: int = 0
    #: 无标签安全层：把目标帧按时间切成"适配段/留出段"，用**无监督损失**在留出段上
    #: 选最佳步；若没有任何一步优于第 0 步（恒等），则还原恒等 ⇒ 结构上不会更差。
    holdout_frac: float = 0.3
    eval_every: int = 10
    early_stop: bool = True
    #: 内建护栏：适配集小于此值就直接不做 TTA。
    #: 门限从 200 放宽到 80，原因：batch 已改为"池内自适应"（bs = min(batch, max(8, 池//4))）
    #: ⇒ 当初 109 帧失败的真因（batch 128 > 池子 ⇒ 矩统计退化为拟合那几帧）已被根除 ✓；
    #: 且因果版（--adapt-frac 0.3 再乘 0.7 留出）的有效池只有会话的 ~21%，
    #: 200 的门限会让它几乎全被跳过（实测 ③ 20/20 会话几乎全跳 ✗）。
    min_frames: int = 80
    #: 只对"输出偏置/尺度"更敏感的轴加权（x 与 y 分开；None = 等权）
    axis_weight: Optional[tuple] = None


class GazeTTAAdapter(nn.Module):
    """低秩**特征空间**增量：h' = h + W2 @ (W1 @ h) + b（零初始化 ⇒ 起点与原模型完全一致）。

    为什么放在特征空间而不是输出空间：放在 `head` **之前**才能同时对"偏置"（b）与
    "逐用户映射"（W2@W1@h）建模，且冻结的 head 会把特征增量正确映射到注视输出；
    放到输出空间只能得到 2 维增量、无法表达逐用户映射。
    """

    def __init__(self, feature_dim: int = 384, rank: int = 8):
        super().__init__()
        self.rank = int(rank)
        self.bias = nn.Parameter(torch.zeros(feature_dim))
        if self.rank > 0:
            self.down = nn.Linear(feature_dim, self.rank, bias=False)
            self.up = nn.Linear(self.rank, feature_dim, bias=False)
            nn.init.zeros_(self.up.weight)          # ⇒ delta 初值为 0（不改变原模型）
            nn.init.normal_(self.down.weight, std=0.02)
        else:
            self.down = self.up = None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        d = self.bias.expand_as(h)
        if self.rank > 0:
            d = d + self.up(self.down(h))
        return h + d

    def reg_loss(self) -> torch.Tensor:
        """拉向恒等（参数越小越接近原始模型）。"""
        s = (self.bias ** 2).sum()
        if self.rank > 0:
            s = s + (self.up.weight ** 2).sum()
        return s


def attach_adapter(model: nn.Module, cfg: TTAConfig,
                   feature_dim: Optional[int] = None) -> GazeTTAAdapter:
    """把适配器挂到**已训练模型**的 `head` 前（非侵入：不改结构、不动权重）。

    注入点：`head_in' = adapter(head_in)`，其中 head_in = `model.norm` 的输出（融合 CLS）。
    适配器自身是"恒等 + 低秩增量"，零初始化 ⇒ 未训练时与原模型逐位一致 ✓。
    特征维度从 `model.head.in_features` 自动读取（无需硬编码 ✓）。
    对含眼支流的模型同样有效（眼支流只影响 head_in 的计算，不涉及注入逻辑）。
    """
    if feature_dim is None:
        feature_dim = int(getattr(model.head, "in_features", 384))
    _p = next(model.parameters())
    # 适配器必须与模型**同设备同 dtype**：否则前向 cut 在 matmul 上
    # （CPU 参数 × CUDA 输入 ⇒ "Expected all tensors to be on the same device"）。
    # 这个坑只在 GPU 上才暴露 ⇒ --smoke 现在也优先用 GPU（见 tta_personalize.py）。
    adapter = GazeTTAAdapter(feature_dim, cfg.rank).to(device=_p.device, dtype=_p.dtype)
    state = {"h": None}

    def _norm_hook(_m, _i, out):
        state["h"] = out if out.requires_grad else out.detach()

    def _head_pre_hook(_m, args):
        h = state["h"]
        if h is None or h.shape[-1] != feature_dim:
            return None
        inp = args[0]
        return (adapter(h.to(inp.dtype)),) + tuple(args[1:])

    model._tta_norm_hook = model.norm.register_forward_hook(_norm_hook)
    model._tta_head_hook = model.head.register_forward_pre_hook(_head_pre_hook)
    return adapter


def detach_adapter(model: nn.Module) -> None:
    """移除注入 hook（评估"原始模型"时用）。"""
    for name in ("_tta_norm_hook", "_tta_head_hook"):
        h = getattr(model, name, None)
        if h is not None:
            h.remove()
            setattr(model, name, None)


def build_prior(labels_norm: np.ndarray) -> dict:
    """由**训练划分**的标签统计输出分布先验（归一化屏幕坐标口径）。

    返回 {'mean': [2], 'std': [2], 'count': n}；测试时不需要任何真值。
    """
    a = np.asarray(labels_norm, dtype=np.float64).reshape(-1, 2)
    return {"mean": a.mean(0).tolist(), "std": (a.std(0) + 1e-6).tolist(),
            "count": int(len(a))}


def _moment_loss(pred: torch.Tensor, prior: dict, axis_w=None) -> torch.Tensor:
    mu = pred.new_tensor(prior["mean"])
    sd = pred.new_tensor(prior["std"])
    w = pred.new_tensor(axis_w) if axis_w is not None else pred.new_tensor([1.0, 1.0])
    lm = ((pred.mean(0) - mu) * w).pow(2).mean()
    ls = ((pred.std(0) - sd) * w).pow(2).mean()
    return lm + ls


def _photometric_4d(x: torch.Tensor, gain: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """对任一 4D 张量施加同一组光度抖动（face 与两眼共用同一组 ⇒ 输入自洽）。"""
    return x * gain.to(x.device) + bias.to(x.device)


def _flip_kw(kw: dict) -> dict:
    """水平镜像输入：face/眼图翻转，左右眼互换；非图像字段原样保留。"""
    out = {}
    for k, v in kw.items():
        if v.dim() == 4:
            out[k] = torch.flip(v, dims=[3])
        else:
            out[k] = v
    if "eye_l" in out and "eye_r" in out:
        out["eye_l"], out["eye_r"] = out["eye_r"], out["eye_l"]
    return out


def _sym_loss(pred: torch.Tensor, pred_flip: torch.Tensor) -> torch.Tensor:
    """2D 屏幕翻转对称：镜像输入的预测 = (1 - x, y)。"""
    target = torch.stack([1.0 - pred[:, 0], pred[:, 1]], dim=1)
    return ((pred_flip - target) ** 2).mean()


def adapt(model: nn.Module, frames, prior: dict, cfg: TTAConfig,
          extra_forward: Optional[Callable] = None, log: Callable[[str], None] = print,
          probe: Optional[Callable] = None):
    """在**无标签**目标域数据上做测试时自适应，返回挂好适配器的模型（就地注入）。

    frames: 目标域输入，两种形式皆可
        · [N, 3, H, W] 张量（仅整脸输入的模型）
        · **list[dict]**：每帧一个 kwargs 字典（如 {"face":…, "eye_l":…, "eye_r":…}），
          供含眼支流/imu/geom 的模型使用 ⇒ 采样时整帧配对，绝不错配。
    只需图像，**不需要真值** ⇒ 满足"无标签测试时自适应"的设置。
    """
    torch.manual_seed(cfg.seed)
    opt_adapter = attach_adapter(model, cfg)
    if cfg.init_adapter:
        from pathlib import Path as _P
        if _P(cfg.init_adapter).is_file():
            opt_adapter.load_state_dict(torch.load(cfg.init_adapter, map_location="cpu"))
            log(f"    [init] adapter 初始化 <- {cfg.init_adapter}")
        else:
            log(f"    [init] 未找到 {cfg.init_adapter}，从零开始")
    for p in model.parameters():
        p.requires_grad_(False)
    params = list(opt_adapter.parameters())
    opt = torch.optim.Adam(params, lr=cfg.lr)
    gen = torch.Generator().manual_seed(cfg.seed)
    dev = next(model.parameters()).device
    is_dict_list = isinstance(frames, (list, tuple)) and len(frames) > 0 and isinstance(frames[0], dict)
    # ---- 无标签安全层：按时间切"适配段 / 留出段"（留出段只用于选步，不用真值 ✓）----
    def _sub(pool_idx):
        if is_dict_list:
            return [frames[int(i)] for i in pool_idx]
        return frames[pool_idx]

    n_all = len(frames)
    if cfg.holdout_frac and cfg.holdout_frac > 0 and n_all >= 20:
        n_val = max(4, int(n_all * float(cfg.holdout_frac)))
        adapt_ids = list(range(0, n_all - n_val))
        val_ids = list(range(n_all - n_val, n_all))
        pool_a, pool_v = _sub(adapt_ids), _sub(val_ids)
    else:
        pool_a, pool_v = frames, None
    n = len(pool_a)
    if n == 0:
        raise ValueError("adapt: 目标域数据为空")
    # ---- 内建护栏（无标签 ✓，让方法自带"不该用就别用"的判断）----
    # ① 适配集太小 ⇒ 矩统计就是在拟合那几帧（实测：109 帧的会话 EM 反而变差 ✗）；
    # ② batch 必须显著小于池子，否则"batch 矩"≈"全池矩"，优化目标退化为常量匹配。
    if n < cfg.min_frames:
        log(f"    [tta] 跳过：目标域可用帧 {n} < min_frames({cfg.min_frames}) ⇒ 不做 TTA")
        with torch.no_grad():
            for p in opt_adapter.parameters():
                p.zero_()
        for p in params:
            p.requires_grad_(False)
        return model
    bs = int(min(cfg.batch_size, max(8, n // 4)))

    def _batch(pool, idx):
        if is_dict_list:
            keys = list(pool[0].keys())
            return {k: torch.stack([pool[int(i)][k] for i in idx]).to(dev) for k in keys}
        return {"face": pool[idx]}

    def _val_loss(frames_pool):
        """无监督验证损失（同适配目标，仅前向、不含正则）。"""
        if frames_pool is None:
            return float("nan")
        m = min(bs, len(frames_pool))
        idx = torch.arange(m)
        kw = {k: v.float() for k, v in _batch(frames_pool, idx).items()}
        with torch.no_grad():
            p1 = extra_forward(model, kw) if extra_forward else model(**kw)
            ls = cfg.w_moment * _moment_loss(p1, prior, cfg.axis_weight)
            if cfg.w_cons > 0:
                g = 1.0 + cfg.cons_jitter * (torch.rand(m, 1, 1, 1, generator=gen) * 2 - 1)
                b = cfg.cons_jitter * (torch.rand(m, 1, 1, 1, generator=gen) * 2 - 1)
                kw2 = {k: (_photometric_4d(v, g, b) if v.dim() == 4 else v) for k, v in kw.items()}
                p2 = extra_forward(model, kw2) if extra_forward else model(**kw2)
                ls = ls + cfg.w_cons * ((p1 - p2) ** 2).mean()
            if cfg.w_sym > 0:
                kwf = _flip_kw(kw)
                pf = extra_forward(model, kwf) if extra_forward else model(**kwf)
                ls = ls + cfg.w_sym * _sym_loss(p1, pf)
        return float(ls)

    model.eval()
    v0 = _val_loss(pool_v)                       # 第 0 步 = 恒等（未训练）参考
    best_v, best_state, best_step = (v0, None, 0) if (cfg.early_stop and pool_v is not None) else (float("inf"), None, -1)
    for step in range(cfg.steps):
        idx = torch.randint(0, n, (bs,), generator=gen)
        kw = {k: v.float() for k, v in _batch(pool_a, idx).items()}
        pred = extra_forward(model, kw) if extra_forward else model(**kw)
        loss = cfg.w_moment * _moment_loss(pred, prior, cfg.axis_weight)
        if cfg.w_cons > 0:
            g = 1.0 + cfg.cons_jitter * (torch.rand(len(idx), 1, 1, 1, generator=gen) * 2 - 1)
            b = cfg.cons_jitter * (torch.rand(len(idx), 1, 1, 1, generator=gen) * 2 - 1)
            kw2 = {k: (_photometric_4d(v, g, b) if v.dim() == 4 else v) for k, v in kw.items()}
            pred2 = extra_forward(model, kw2) if extra_forward else model(**kw2)
            loss = loss + cfg.w_cons * ((pred - pred2) ** 2).mean()
        if cfg.w_geo > 0:
            # 注视不变的几何抖动（小仿射）⇒ 预测应不变（同族于文献的增强一致性 ✓）
            sc = 1.0 + cfg.geo_scale_amp * (torch.rand(len(idx), generator=gen) * 2 - 1)
            tx = cfg.geo_shift_amp * (torch.rand(len(idx), generator=gen) * 2 - 1)
            ty = cfg.geo_shift_amp * (torch.rand(len(idx), generator=gen) * 2 - 1)
            kw3 = {}
            for k, v in kw.items():
                if v.dim() == 4:
                    th = torch.zeros(v.shape[0], 2, 3, device=v.device, dtype=v.dtype)
                    th[:, 0, 0] = sc.to(v.device); th[:, 1, 1] = sc.to(v.device)
                    th[:, 0, 2] = tx.to(v.device); th[:, 1, 2] = ty.to(v.device)
                    gr = torch.nn.functional.affine_grid(th, list(v.shape), align_corners=False)
                    kw3[k] = torch.nn.functional.grid_sample(v, gr, align_corners=False,
                                                             padding_mode="border")
                else:
                    kw3[k] = v
            pred3 = extra_forward(model, kw3) if extra_forward else model(**kw3)
            loss = loss + cfg.w_geo * ((pred - pred3) ** 2).mean()
        if cfg.w_sym > 0:
            kwf = _flip_kw(kw)
            predf = extra_forward(model, kwf) if extra_forward else model(**kwf)
            loss = loss + cfg.w_sym * _sym_loss(pred, predf)
        if cfg.w_reg > 0:
            loss = loss + cfg.w_reg * opt_adapter.reg_loss()
        opt.zero_grad()
        loss.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()
        if (step + 1) % max(1, cfg.eval_every) == 0:
            # 探针与早停解耦：--no-early-stop（= 采用最终状态）时也必须能采到完整曲线 ✓
            if probe is not None:
                probe(step + 1, _val_loss(pool_v), best_step, opt_adapter)
            if cfg.early_stop and pool_v is not None:
                v = _val_loss(pool_v)
                if v < best_v:
                    best_v, best_step = v, step + 1
                    best_state = {k: t.detach().clone() for k, t in opt_adapter.state_dict().items()}
        if (step + 1) % max(1, cfg.steps // 5) == 0:
            log(f"    [tta] step {step + 1}/{cfg.steps}  loss {float(loss.detach()):.5f}"
                f"  |bias| {float(opt_adapter.bias.abs().mean()):.4f}"
                f"  val {(_val_loss(pool_v) if pool_v is not None else float('nan')):.5f}"
                f"  best@{best_step}")
    if cfg.early_stop:
        if best_state is None:                   # 没有任何一步优于恒等 ⇒ 回退（不做 TTA）
            log(f"    [tta] 留出段未优于恒等（v0={v0:.5f}）⇒ 回退为不做 TTA")
            with torch.no_grad():
                for p in opt_adapter.parameters():
                    p.zero_()
        else:
            opt_adapter.load_state_dict(best_state)
            log(f"    [tta] 采用最佳步 {best_step}（val {best_v:.5f} vs 恒等 {v0:.5f}）")
    # ---- 无标签安全门控：留出损失变差则回退；变好则只应用 gate_alpha ----
    if cfg.gate and pool_v is not None:
        v_final = _val_loss(pool_v)
        accept = v_final <= v0 * (1.0 + cfg.gate_margin)
        scale = cfg.gate_alpha if accept else 0.0
        with torch.no_grad():
            for p_ in opt_adapter.parameters():
                p_.mul_(scale)
        log(f"    [gate] v0={v0:.5f} v_final={v_final:.5f} "
            f"accept={accept} scale={scale:.2f}")
    for p in params:
        p.requires_grad_(False)
    return model
