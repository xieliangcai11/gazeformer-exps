"""RGBDGaze 一次校准模块（per-user calibration）。

设计参考 MAC-Gaze (2025)：
    骨干模型提取特征 → 轻量 MLP 校准器 → 修正后的注视点
    骨干冻结，只训练校准器（数万参数）
    校准数据 = 被试前 5/10/15% 帧；测试 = 剩余 85-95%

特征来源：融合 Transformer 的 CLS 输出（d_model=384 维），
    该向量已整合全脸视觉信息，是模型对当前帧的"最终内部表示"。
"""

from __future__ import annotations

import numpy as np
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


def extract_features(model, face):
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


class PolyCalibrator:
    """一次性标定（少量真值）：每轴多项式 + 留一(LOO) 选收缩力度 α + 覆盖(span)检查。

    与上面的 MLP `Calibrator` 的区别：那是"在无监督/特征空间上训练"，本类只做**闭式标定**
    ——给定 K 个带真值的标定点，拟合"预测位置 → 真值位置"的小多项式，**不需要训练、不需要特征**。

    实测依据（docs/experiments_2026-10-15.md §7/§8 + B 组实验，同一模型 eye224_mgpu、评估期零真值）：
      · 二次多项式（每轴 3 参，共 6 参）在 7 种方法里最好或并列最好；
      · **覆盖(span) 是成败开关**：标定点铺满屏幕时稳定 −11%~−19%（K=9/13/25/50），
        而只取"会话开头 N 帧"（bbox 面积占比 ≈0.31）时二次曲面会**反噬 +33%~+80%** ✗
        ⇒ 因此本类内置 **span 检查，不达标就把 α 置 0（= 回退为不标定）**；
      · α 的选择用 **留一(LOO)**：K 小时比"标定集对半切"显著更好（K=13：−11.3% → −14.5%）。

    典型用法（部署时 `targets` = App 指示用户看的位置；离线评估时可传真值）::

        cal = PolyCalibrator().fit(pred_cal, label_cal, targets=commanded_targets,
                                   screen_cm=(w_cm, h_cm))
        if cal.ok:                       # span 检查未过则为 False，α=0
            pred_corrected = cal.apply(pred_stream)
    """

    def __init__(self, degree: int = 2, alphas=(0.0, 0.25, 0.5, 0.75, 1.0),
                 span_min_area: float = 0.40, min_points: int = 8):
        self.degree = int(degree)
        self.alphas = tuple(float(a) for a in alphas)
        self.span_min_area = float(span_min_area)
        self.min_points = int(min_points)
        self.alpha = 0.0
        self.coef_ = None
        self.info: dict = {}

    # ---------------- 内部：多项式拟合 ----------------
    def _design(self, p: np.ndarray, ax: int) -> np.ndarray:
        x = p[:, ax]
        return np.stack([np.ones_like(x)] + [x ** k for k in range(1, self.degree + 1)], 1)

    def _fit_coef(self, pred: np.ndarray, label: np.ndarray):
        return [np.linalg.lstsq(self._design(pred, ax), label[:, ax], rcond=None)[0]
                for ax in (0, 1)]

    def _apply_coef(self, coef, p: np.ndarray) -> np.ndarray:
        return np.stack([self._design(p, ax) @ coef[ax] for ax in (0, 1)], 1)

    @staticmethod
    def _span(points: np.ndarray) -> dict:
        """覆盖度量：标签(或指令目标)的 bbox 面积占比 + 5x5 网格占用格数。"""
        area = float((points[:, 0].max() - points[:, 0].min())
                     * (points[:, 1].max() - points[:, 1].min()))
        cx = np.clip((points[:, 0] * 5).astype(int), 0, 4)
        cy = np.clip((points[:, 1] * 5).astype(int), 0, 4)
        return dict(area=area, cells=len(set(zip(cx.tolist(), cy.tolist()))))

    # ---------------- 对外 API ----------------
    def fit(self, pred, label, targets=None, screen_cm=None):
        """pred/label/targets: (n,2) 归一化屏幕坐标；screen_cm=(w,h) 用于按厘米挑 α。"""
        pred = np.asarray(pred, dtype=np.float64).reshape(-1, 2)
        label = np.asarray(label, dtype=np.float64).reshape(-1, 2)
        span_pts = np.asarray(targets if targets is not None
                              else label, dtype=np.float64).reshape(-1, 2)
        self.info = dict(n_points=int(len(pred)), span=self._span(span_pts))
        self.info["span_ok"] = bool(self.info["span"]["area"] >= self.span_min_area)
        # 点太少 ⇒ 不拟合，直接回退（α=0）
        if len(pred) < self.min_points:
            self.alpha, self.coef_ = 0.0, None
            self.info["reason"] = f"标定点不足（{len(pred)} < {self.min_points}）"
            return self
        if not self.info["span_ok"]:
            self.alpha, self.coef_ = 0.0, None
            self.info["reason"] = (f"覆盖不足（bbox 面积占比 {self.info['span']['area']:.2f} "
                                   f"< {self.span_min_area}）⇒ 回退为不标定")
            return self
        # ---- LOO 选 α（用全部点做最终拟合；验证样本 = 全部 K 个点）----
        sc_ = np.asarray(screen_cm, dtype=np.float64) if screen_cm is not None else np.ones(2)
        loo_corr = np.zeros_like(pred)
        for j in range(len(pred)):
            rest = np.arange(len(pred)) != j
            if rest.sum() < self.degree + 2:
                continue
            c = self._fit_coef(pred[rest], label[rest])
            loo_corr[j] = self._apply_coef(c, pred[j:j + 1])[0] - pred[j]
        best_a, best_e = 0.0, np.inf
        for a in self.alphas:
            q = pred + a * loo_corr
            e = float(np.mean(np.hypot((q[:, 0] - label[:, 0]) * sc_[0],
                                       (q[:, 1] - label[:, 1]) * sc_[1])))
            if e < best_e:
                best_e, best_a = e, a
        self.alpha = float(best_a)
        self.coef_ = self._fit_coef(pred, label)
        self.info["loo_em_norm"] = best_e
        return self

    def apply(self, pred) -> np.ndarray:
        """把 α 后的修正作用到任意帧（评估期不需要真值）。"""
        pred = np.asarray(pred, dtype=np.float64).reshape(-1, 2)
        if self.coef_ is None or self.alpha == 0.0:
            return pred.copy()
        corr = self._apply_coef(self.coef_, pred)
        return pred + self.alpha * (corr - pred)

    @property
    def ok(self) -> bool:
        """是否真的在做标定（False = 已按安全规则回退为不标定）。"""
        return self.coef_ is not None and self.alpha > 0.0