"""RGBDGaze 模型：DINOv2 冻结 RGB + 屏幕几何 token + Transformer(BlockMoba) 融合。

设计：
    RGB 流     → DINOv2（ViT-S/14，冻结）提取 cls + patch tokens。
    几何 token → 屏幕物理几何（宽高比 + 物理宽高），1 个 token。
    姿态 token → IMU 重力向量（可选），1 个 token。
    融合       → 复用 gazelab.models.transformers 的 BlockMoba（标准自注意力 + MoE/FFN），
                 把 [汇总CLS, geom, imu, rgb_cls, rgb_patch_tokens] 一起自注意力。
    回归       → 取汇总 CLS 经线性头输出 2D 屏幕坐标。

为什么需要"屏幕几何 token"（跨设备泛化的关键）：
    归一化屏幕坐标 (x,y) 的物理含义随设备变化。实测（GazeCapture vs RGBDGaze）：
      - 宽高比 0.562~0.750（GazeCapture 手机/iPad）vs 0.462（RGBDGaze 现代手机）；
      - 屏幕物理对角线 11.94~25.47 cm，相差 2.1 倍。
    同一个归一化坐标 (0.1, 0.1) 在不同设备上对应的物理方向完全不同
    （与水平夹角 52.9° / 60.7° / 64.1°）。而屏幕在正面摄像头画面里是**不可见**的，
    模型无法从人脸图推断屏幕尺寸 —— 必须显式喂入，否则只能学到"平均设备"的映射。

注意：深度模态已移除 —— 实验证明其在 RGBDGaze 上无正收益
（见 out/rgbdgaze/ablation 消融结果），故不再保留相关分支。

输入：
    face : [B, 3, S, S] RGB 人脸（ImageNet 归一化，适配 DINOv2 预处理）。
    imu  : [B, 3] 重力向量（use_imu=True 时必需）。
    geom : [B, 3] 屏幕几何 [宽高比, log(物理宽cm), log(物理高cm)]（use_geom=True 时必需）。

设计取舍（重要，避免逻辑/资源纰漏）：
    - 不直接复用 TransformerDeepSeek_gaze：其 proj_f1/f2/f3 把输入维度写死为
      512/512/2048（面向 CLIP+ResNet）。RGBD 无 CLIP 时若强行喂入，会因
      2048->768 投影作用在 256 个 token 上而产生约 4 亿参数，浪费且错误。
      此处复用其底层 BlockMoba 块，构造维度正确的投影。
    - DINOv2 冻结当现成"眼睛"，只训融合与回归头。
"""

import torch
import torch.nn as nn
import timm

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
                 freeze=True, unfreeze_last=0, img_size=DINOV2_IMG_SIZE):
        """freeze: 是否冻结 DINOv2；unfreeze_last: 解冻最后 N 个 block 参与微调。
        unfreeze_last>0 时允许任务相关微调（代价是显存/耗时增加）。
        img_size: 输入分辨率（pos_embed 会插值到对应网格）。"""
        super().__init__()
        backbone = timm.create_model(model_name, pretrained=False,
                                     img_size=img_size)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        # DINOv2 官方权重额外带 'mask_token'（预训练专用），加载时忽略
        state.pop("mask_token", None)
        # 权重是 518 训练的，输入分辨率不同时需把 pos_embed 插值到对应网格
        if "pos_embed" in state:
            grid = img_size // backbone.patch_embed.patch_size[0]
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


class RGBDGazeDINOv2(nn.Module):
    """DINOv2(RGB,冻结) + 屏幕几何 token + BlockMoba 融合 -> 2D 屏幕坐标。

    forward(face, imu=None, geom=None, eye_l=None, eye_r=None) -> [B, 2]
      face  : [B, 3, S, S] RGB（ImageNet 归一化，DINOv2 预处理）
      imu   : [B, 3] 重力向量（use_imu=True 时必需）
      geom  : [B, 3] 屏幕几何 [宽高比, log(物理宽cm), log(物理高cm)]（use_geom=True 时必需）
      eye_l : [B, 3, S, S] 左眼放大裁剪（use_eye=True 时必需）
      eye_r : [B, 3, S, S] 右眼放大裁剪（use_eye=True 时必需）

    眼部支流的动机（用实测数字，不是猜测）：480×640/1080×1440 原图中虹膜直径约 22~36 px，
    整脸裁剪后（CROP_MARGIN=1.2）resize 到 224 时虹膜只剩约 14 px ≈ DINOv2 一个 patch(14)，
    即"一个视觉单元 ≈ 一整个虹膜"；而把眼区（含 1.6 倍边距约 134 px）放大到 224 可得
    虹膜约 60 px，恰好跨过"1°/px"门槛（约 58 px）。两条支流共用同一个 DINOv2 骨干，
    眼部只贡献各自的 cls token（融合序列 259 -> 261，几乎不增加融合开销）。
    """

    #: 几何 token 的输入维度：[宽高比, log(w_cm), log(h_cm)]
    GEOM_DIM = 3
    #: 可选朝向 one-hot 维数（1=竖屏, 2=倒竖, 3/4=横屏；GazeCapture 专用）
    GEOM_ORIENT_DIM = 4

    def __init__(
        self,
        gaze_dim=2,
        d_model=384,
        num_heads=8,
        num_layers=6,
        inter_dim=1536,
        freeze_dino=True,
        unfreeze_last=0,
        use_imu=False,
        use_geom=True,
        use_geom_orient=False,
        use_eye=False,
        img_size=DINOV2_IMG_SIZE,
        dino_model=DINOV2_MODEL,
        dino_ckpt=DINOV2_CKPT,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model({d_model}) 必须能被 num_heads({num_heads}) 整除")

        # ---- RGB 流：冻结 DINOv2（可选解冻最后 N 层微调） ----
        self.rgb = DINOv2Backbone(dino_model, dino_ckpt, freeze=freeze_dino,
                                  unfreeze_last=unfreeze_last,
                                  img_size=img_size)
        dino_dim = self.rgb.embed_dim
        self.proj_rgb_cls = nn.Linear(dino_dim, d_model)
        self.proj_rgb_patch = nn.Linear(dino_dim, d_model)

        # ---- IMU 姿态 token（可选） ----
        self.use_imu = use_imu
        if use_imu:
            self.proj_imu = nn.Linear(3, d_model)

        # ---- 屏幕几何 token（可选；跨设备泛化的关键，见模块 docstring） ----
        self.use_geom = use_geom
        self.use_geom_orient = bool(use_geom_orient) and bool(use_geom)
        self.geom_dim = self.GEOM_DIM + (self.GEOM_ORIENT_DIM if self.use_geom_orient else 0)
        if use_geom:
            self.proj_geom = nn.Linear(self.geom_dim, d_model)

        # ---- 眼部放大支流（可选）：与整脸共用骨干，只取各自的 cls token ----
        self.use_eye = use_eye
        if use_eye:
            self.proj_eye = nn.Linear(dino_dim, d_model)

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

    def forward(self, face, imu=None, geom=None, eye_l=None, eye_r=None):
        # 1) RGB：DINOv2（冻结/部分解冻）-> cls + patches
        tokens = self.rgb(face)                      # [B, 1+N, C]
        cls = tokens[:, 0]                           # [B, C]
        patches = tokens[:, 1:]                      # [B, N, C]
        rgb_cls = self.proj_rgb_cls(cls).unsqueeze(1)    # [B, 1, d_model]
        rgb_patch = self.proj_rgb_patch(patches)         # [B, N, d_model]

        # 2) 拼接：汇总CLS + [imu] + [geom] + rgb_cls + rgb_patches
        seq = torch.cat([rgb_cls, rgb_patch], dim=1)

        # 2a) IMU 姿态 token（可选，1 个 token）
        if self.use_imu:
            if imu is None:
                raise ValueError("use_imu=True 但 forward 未提供 imu")
            imu_tok = self.proj_imu(imu).unsqueeze(1)    # [B, 1, d_model]
            seq = torch.cat([imu_tok, seq], dim=1)

        # 2b) 屏幕几何 token（可选，1 个 token）——跨设备泛化的关键信息，
        #     屏幕在正面摄像头画面里不可见，必须显式提供。
        if self.use_geom:
            if geom is None:
                raise ValueError("use_geom=True 但 forward 未提供 geom")
            geom_tok = self.proj_geom(geom).unsqueeze(1)  # [B, 1, d_model]
            seq = torch.cat([geom_tok, seq], dim=1)

        # 2c) 眼部放大支流（可选，2 个 token）：与整脸共用同一个 DINOv2 骨干，
        #     每只眼只贡献自己的 cls token，因此融合序列只增加 2 个位置。
        if self.use_eye:
            if eye_l is None or eye_r is None:
                raise ValueError("use_eye=True 但 forward 未提供 eye_l/eye_r")
            eye_tok = torch.cat([
                self.proj_eye(self.rgb(eye_l)[:, 0]).unsqueeze(1),
                self.proj_eye(self.rgb(eye_r)[:, 0]).unsqueeze(1),
            ], dim=1)                                    # [B, 2, d_model]
            seq = torch.cat([eye_tok, seq], dim=1)

        B = face.size(0)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, seq], dim=1)       # [B, 1+[imu]+[geom]+[2眼]+1+N, d_model]

        # 3) BlockMoba 栈（自注意力融合）
        for layer in self.layers:
            x = layer(x, cross_input=None, mask=None)

        # 4) 汇总 CLS -> 2D
        out = self.head(self.norm(x[:, 0, :]))
        return out


def load_state_dict_any_res(model: nn.Module, state: dict):
    """加载权重，并把 pos_embed 自动插值到当前模型的分辨率。

    为什么需要：预训练为省时常用 224（16x16 网格），微调为对齐基线常用 448
    （32x32 网格）。两者 pos_embed 形状不同（[1,257,C] vs [1,1025,C]），而
    `load_state_dict(strict=False)` 只豁免"键缺失/多余"，对形状不匹配仍会抛
    RuntimeError —— 直接把 224 权重喂给 448 模型会崩，且崩在训练启动阶段。

    这里复用 DINOv2Backbone 加载官方 DINOv2 权重时的同一套做法
    （resample_abs_pos_embed + 同样的 num_prefix_tokens），因此两条路径的
    插值口径一致。

    返回 (missing, unexpected, resampled_from)：
      - missing / unexpected 语义同 nn.Module.load_state_dict；
      - resampled_from 为 None 表示未发生插值，否则是原 pos_embed 的形状元组，
        供调用方写进日志（避免"悄悄插值了"这种不可追溯的情况）。
    """
    key = "rgb.backbone.pos_embed"
    state = dict(state)                       # 不改调用方持有的对象
    resampled_from = None
    src = state.get(key)
    dst = model.state_dict().get(key)
    if src is not None and dst is not None and tuple(src.shape) != tuple(dst.shape):
        n = int(dst.shape[1]) - NUM_PREFIX_TOKENS
        grid = int(round(n ** 0.5))
        if grid * grid != n:
            raise ValueError(
                f"[DINOv2] 目标 pos_embed 第二维 {dst.shape[1]} 不等于 "
                f"网格点数+{NUM_PREFIX_TOKENS}，无法确定插值网格")
        state[key] = resample_abs_pos_embed(src, (grid, grid),
                                            num_prefix_tokens=NUM_PREFIX_TOKENS)
        resampled_from = tuple(src.shape)
    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected, resampled_from