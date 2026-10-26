"""RGBDGaze 数据集类。

遵循统一返回协议，`__getitem__` 返回
`(edict(rgb=..., screen_cm=..., imu=..., geom=...), gaze)`：

    rgb       = RGB 人脸裁剪图（走视觉流，ImageNet 归一化）
    screen_cm = 该帧屏幕的物理宽高（厘米），供误差换算
    imu       = 重力向量（设备坐标系）
    geom      = 屏幕几何 [宽高比, log(物理宽cm), log(物理高cm)]，喂给模型的几何 token
    gaze      = 归一化后的 2D 屏幕注视坐标 (x, y)

深度模态已移除：实验证明其在 RGBDGaze 上无正收益。

依赖 configs/rgbdgaze_config.py 中的路径与超参。
数据索引由 tools/data/rgbdgaze/preprocess.py 生成（index csv）。
"""

from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from easydict import EasyDict as edict

from configs.rgbdgaze_config import (
    PROJECT_ROOT,
    IMAGE_SIZE,
    CROP_MARGIN,
    NORMALIZE_GAZE,
    AUG_SHIFT,
    AUG_SCALE,
    AUG_ROTATE,
    GEOM_RAND_PROB,
    GEOM_RAND_ASPECT,
    GEOM_RAND_WIDTH_CM,
    GEOM_RAND_HEIGHT_CM,
)
from gazelab.datasets.table_index import TableIndex
from gazelab.datasets.common import (
    crop_face_square,
    randomize_screen_geometry_frame,
    rgb_preprocess,  # noqa: F401  (re-export，供既有调用方使用)
    screen_geometry,
)


def _resolve_data_path(p: str) -> Path:
    """把 index 里的相对路径（相对项目根）解析成绝对路径；绝对路径则原样返回。

    兼容性：index.csv 存的是"相对项目根 + 前向斜杠"的路径，在任意机器上
    都能用当前项目的 PROJECT_ROOT 拼回正确位置。
    """
    p = Path(p)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _to_tensor_label(label):
    """把 2D 标签转成 float32 tensor。命名函数便于多进程 pickle。"""
    return torch.tensor(label, dtype=torch.float32)


# ---- 图像预处理 ----
# 通用实现（BGR->tensor、ImageNet 归一化、人脸裁剪、屏幕几何）已上提到
# gazelab.datasets.common，与 GazeCapture 共用同一输入约定。


class RGBDGazeDataset(Dataset):
    """RGBDGaze 数据集：从索引文件加载 (RGB 路径, 设备, 注视点)。

    索引文件格式（由 tools/data/rgbdgaze/preprocess.py 生成）：
        header: rgb_path, bbox_x, bbox_y, bbox_w, bbox_h, device, screen_w,
                screen_h, gaze_x, gaze_y, cm_px_x, cm_px_y, imu_x/y/z,
                subject, activity
    其中 gaze_x/gaze_y 是屏幕像素注视点，screen_w/screen_h 是该设备屏幕点数。
    """

    def __init__(
        self,
        index_csv: Path,
        color_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        augment: bool = False,
        aug_shift: float = AUG_SHIFT,
        aug_scale: float = AUG_SCALE,
        aug_rotate: float = AUG_ROTATE,
        geom_randomize: float = None,
        image_size: int = None,
        eye_crops: bool = False,
        orient: str = "raw",
        eye_scale: float = 0.0,
    ):
        """裁剪朝向 orient（**只用于评估历史 ckpt**；所有训练与新实验用默认 raw）

          - "raw"（默认）：直接用原图 + 把索引 bbox 反映射回来。原图人脸正立
            （MediaPipe 眼框 137x40 = 宽>高），与预训练数据（GazeCapture 亦正立）一致。
            一致性验证：反解后框中心与 MediaPipe 人脸框中心相差 4.68% 脸边长。
          - "rot90"：历史行为（先 np.rot90(img,1) 再按索引 bbox 裁）。实测量化该朝向
            使**人脸躺倒**（rot90 后两眼中心连线 Δx=1.0 px / Δy=208 px，即竖直）。
            **仅**用于复现/评估在躺倒朝向下训练的旧 ckpt（例如 test 1.64cm 的
            g6_multi）——用错朝向评估会让数字完全失真。

        注意 rot90 模式不支持 eye_crops（眼框表存的是原图坐标，没有旋转后坐标）。
        """
        super().__init__()
        self.index_csv = Path(index_csv)
        # 列式索引：list(csv.DictReader) 会按 1.9KB/行 生成 dict，
        # 149 万行时约 2.4GB/进程，fork 给 DataLoader worker 后触发写时复制，
        # 是多卡训练卡顿的主因。TableIndex 取值语义与 DictReader 完全一致。
        self.rows = TableIndex(self.index_csv)
        # 单设备数据集：所有样本同属一个组（均衡采样退化为普通 shuffle）
        self.sample_groups = np.zeros(len(self.rows), dtype=np.int64)

        # 默认与 GazeCapture 一致：ImageNet 归一化（显式传 None 会得到 [0,1] 未归一化
        # 输入，与 DINOv2 预训练/微调约定不符，且 augment 下 torch.flip 会 TypeError）。
        self.color_transform = color_transform if color_transform is not None else rgb_preprocess

        self.target_transform = (
            target_transform if target_transform is not None else _to_tensor_label
        )

        # ---- 数据增强（只应训练集开启） ----
        # augment    : 水平翻转(注视点 x 镜像) + 光度扰动
        # aug_shift  : 裁剪中心随机偏移幅度（相对裁剪边长）
        # aug_scale  : 裁剪边长随机缩放幅度
        # aug_rotate : 随机旋转角度（度）
        # 这三个几何抖动用于抹平两数据集的人脸框分布差异（见 configs 注释）。
        self.augment = augment
        self.aug_shift = float(aug_shift) if augment else 0.0
        self.aug_scale = float(aug_scale) if augment else 0.0
        self.aug_rotate = float(aug_rotate) if augment else 0.0
        # geom_randomize: 几何域随机化触发概率（None=用 config 默认；仅在
        # augment 开启时生效，显式传 0 可关闭）
        self.geom_randomize = (float(GEOM_RAND_PROB) if geom_randomize is None
                               else float(geom_randomize))
        if not augment:
            self.geom_randomize = 0.0
        self._rng = np.random.RandomState()
        # 逐样本确定性增强种子（由 scripts/train.py 的 build_dataloader 注入）：
        # None = 旧行为（RandomState() 用 OS 熵 ⇒ 同 --seed 两次训练增强不同 ✗）
        self._seed = None
        self.epoch = 0
        # 输入分辨率（None = 使用配置的 IMAGE_SIZE）
        self.image_size = int(image_size) if image_size else IMAGE_SIZE

        # ---- 裁剪朝向（见 docstring）：raw=正立原图（默认）；rot90=历史躺倒朝向 ----
        if orient not in ("raw", "rot90"):
            raise ValueError(f"orient 只能是 'raw' 或 'rot90'，收到 {orient!r}")
        self.orient = orient
        self.eye_scale = eye_scale
        if orient == "rot90" and eye_crops:
            raise ValueError(
                "orient='rot90' 不支持 eye_crops：眼框表只存了原图坐标。"
                "评估带眼支流的历史 ckpt 需要另建旋转后坐标的框表。")

        # ---- 眼部放大支流（可选）----
        # 动机（实测）：眼区框（含 1.6 倍边距）中位 128 px，放大到 224 时虹膜约 62 px，
        # 越过"1°/px"门槛（约 58 px）；而整脸裁剪下虹膜只剩约 14 px ≈ DINOv2 一个 patch。
        # 框表由 tools/data/rgbdgaze/build_eye_boxes.py 生成（**原图坐标**，与裁剪同帧；
        # 8 帧检测失败用同被试中位数兜底，filled 掩码备查）。
        self.eye_crops = bool(eye_crops)
        if self.eye_crops:
            # 必须校验 idx == arange(N)：只比长度会在"索引重建但行数恰好相同"时
            # 让眼框与帧逐行错位（训练照跑、指标静默失真）。
            with np.load(Path(self.index_csv).parent / "eye_boxes.npz") as _z:
                self.eye_boxes = _z["boxes"].astype(np.float32)
                self.eye_filled = _z["filled"]
                _idx = _z["idx"]
            n = len(self.rows)
            if len(self.eye_boxes) != n or not np.array_equal(_idx, np.arange(n)):
                raise ValueError(
                    f"eye_boxes.npz 与索引不对齐（npz {len(self.eye_boxes)} 行 / "
                    f"索引 {n} 行 / idx 是否 0..N-1: "
                    f"{len(_idx) == n and np.array_equal(_idx, np.arange(n))}）⇒ "
                    f"请重跑 tools/data/rgbdgaze/build_eye_boxes.py")

        # 每样本的"像素 -> 厘米"换算系数（来自 index，用于把预测误差转成物理 cm）
        # 整列取出（float_column）而不是逐行建 dict：14 万行下后者是启动期主要开销，
        # 且多卡时每个 rank 都要付一次。取值语义与逐行 float(r[col]) 完全一致。
        try:
            self.cm_px = np.stack(
                [self.rows.float_column("cm_px_x", np.float32),
                 self.rows.float_column("cm_px_y", np.float32)], axis=1)
            self.screen_size = np.stack(
                [self.rows.float_column("screen_w", np.int32),
                 self.rows.float_column("screen_h", np.int32)], axis=1)
        except KeyError:
            # 旧版 index 无 cm 列时退化为近似（角标）
            self.cm_px = np.full((len(self.rows), 2), 0.0182, dtype=np.float32)
            self.screen_size = np.ones((len(self.rows), 2), dtype=np.int32)

    def __len__(self):
        return len(self.rows)

    def set_aug_seed(self, seed):
        """设定增强的逐样本确定性种子（None = 恢复旧的不播种行为）。

        动机（2026-10-15 审计 A2）：此前 self._rng = RandomState() 用 OS 熵，
        实测同一 --seed、同一配置两次构造时 10/10 样本增强不同（含翻转方向不同
        ⇒ 标签被镜像）⇒ 训练不可复现、run 间差异无法归因。
        """
        self._seed = None if seed is None else int(seed)

    def set_epoch(self, epoch):
        """训练循环每轮调用：让同一 idx 在不同 epoch 拿到不同增强（保持多样性）。"""
        self.epoch = int(epoch)

    def aug_summary(self):
        """**实际生效**的增强/几何参数（供日志如实记录）。

        以前日志打印的是命令行参数值，而 build_dataloader 并未把它传下来 ⇒
        日志记录的 shift/scale/rotate 与实际生效的 config 默认值不一致 ✗。
        """
        return dict(augment=bool(self.augment), shift=self.aug_shift,
                    scale=self.aug_scale, rotate=self.aug_rotate,
                    geom_randomize=self.geom_randomize,
                    eye_crops=self.eye_crops, eye_scale=self.eye_scale,
                    orient=self.orient, image_size=self.image_size,
                    aug_seed=self._seed)

    def _make_face(self, rgb_path, bbox_xywh, idx, rngs):
        """读 RGB 整帧，裁出人脸（以及可选的左右眼放大裁剪）。

        图像**不旋转**：实测原图本身人脸正立（MediaPipe 眼框 137x40，宽>高），
        而历史的 np.rot90(img, 1) 会让脸躺倒、白费 DINOv2 的正立人脸先验。
        索引 bbox 位于作者坐标系（= rot90 后），故先反映射回原图（见
        _index_bbox_to_raw），再按它裁正方形人脸。

        eye_crops=True 时返回 (face, eye_l, eye_r)，三张裁剪都取自同一张原图
        （眼部因此拿到放大后的虹膜细节）。几何抖动按各自裁剪边长施加（同种子的
        RandomState ⇒ 抽取序列相同）。
        """
        rgb = cv2.imread(str(rgb_path))
        if rgb is None:
            raise FileNotFoundError(
                f"无法读取 RGB 图像: {rgb_path}（文件缺失或损坏？）")

        if self.orient == "rot90":
            # 历史朝向：仅用于评估在躺倒朝向下训练的旧 ckpt（其 bbox 就在该坐标系）
            img = np.rot90(rgb, 1)
            bbox_use = bbox_xywh
        else:
            # 默认（正立）：用原图；索引 bbox 在作者（rot90 后）坐标系，需反映射回来
            img = rgb
            bbox_use = self._index_bbox_to_raw(bbox_xywh, rgb.shape[1])

        rgb_face = crop_face_square(img, bbox_use, CROP_MARGIN,
                                    self.image_size, shift=self.aug_shift,
                                    scale=self.aug_scale,
                                    rotate=self.aug_rotate, rng=rngs[0])
        if self.color_transform is not None:
            rgb_face = self.color_transform(rgb_face)
        if not self.eye_crops:
            return rgb_face, None, None
        return (rgb_face,
                self._crop_eye(img, idx, 0, rngs[1]),
                self._crop_eye(img, idx, 4, rngs[2]))

    @staticmethod
    def _index_bbox_to_raw(bbox, W0):
        """把索引 bbox（作者＝rot90 后坐标系）反映射回原图坐标。

        正映射 raw→rot90: (x', y') = (y, W0-1-x) ⇒ 反解 x = W0-1-y', y = x'；
        对框 (x',y',w',h') ⇒ (W0-1-(y'+h'), x', h', w')，W0 = 原图宽。
        """
        x, y, w, h = bbox
        return (W0 - 1 - (y + h), x, h, w)

    def _crop_eye(self, raw, idx, off, rng):
        """从**原图**（未旋转）上裁一只眼的放大正方形（框来自 eye_boxes.npz）。

        复用 crop_face_square：margin=1.0 时它以 bbox 中心取 max(w,h) 的正方形
        （眼框中位 128 px，放大到 224 ⇒ 1.75x ⇒ 虹膜约 62 px，越过"1°/px"门槛）。

        已知限制（实测 0.03% 的眼框会触发，可忽略但需知情）：
        crop_face_square 对越界是**按边界截断**、随后 resize ⇒ 这类样本会被**拉伸**
        （没有黑边，但放大倍数与其它样本不一致）。出界比例：左眼 0.02% / 右眼 0.01%。

        另一个注意点：rotate>0 时人脸与眼部**各自绕自己裁剪中心**旋转，两支流的
        相对几何因此不完全一致；本项目实验均用 --aug-rotate 0，故不触发。
        """
        box = tuple(float(v) for v in self.eye_boxes[idx, off:off + 4])
        if self.eye_scale > 0:
            # 收紧眼框：以**眼框高度**为基准取正方形 ⇒ 虹膜在裁剪中的占比大幅提高
            # （默认 0 = 历史行为，以 max(w,h) 为边长，虹膜仅占约 28%）。
            _ex, _ey, _ew, _eh = box
            _s = max(self.eye_scale * _eh, 1.0)
            _cx, _cy = _ex + _ew / 2.0, _ey + _eh / 2.0
            box = (_cx - _s / 2.0, _cy - _s / 2.0, _s, _s)
        eye = crop_face_square(raw, box, 1.0, self.image_size,
                               shift=self.aug_shift, scale=self.aug_scale,
                               rotate=self.aug_rotate, rng=rng)
        return self.color_transform(eye) if self.color_transform is not None else eye

    def _augment(self, rgb_t, gx, flip, gain, bias):
        """按**调用方抽好的**随机量增强 (rgb, gaze_x)。

        参数外置的原因：人脸与眼部两支流必须看到**完全相同**的成像条件
        （是否镜像、亮度/对比度扰动一致），否则双流输入自相矛盾。
        """
        if flip:
            rgb_t = torch.flip(rgb_t, dims=[2])
            gx = 1.0 - gx
        rgb_t = torch.clamp(rgb_t * gain + bias, -2.5, 2.5)
        return rgb_t, gx

    @staticmethod
    def _augment_eyes(eye_l, eye_r, flip, gain, bias):
        """对眼部裁剪施加与整脸相同的翻转/光度；水平翻转时**左右眼互换**。

        几何含义：整脸被镜像后，画面左侧眼睛的内容来自原来的右眼；不交换的话
        "左眼 token"会指向镜像后位于右侧的眼睛，与整脸几何自相矛盾。
        """
        if flip:
            eye_l, eye_r = eye_r, eye_l

        def post(e):
            if flip:
                e = torch.flip(e, dims=[2])
            return torch.clamp(e * gain + bias, -2.5, 2.5)

        return post(eye_l), post(eye_r)

    def __getitem__(self, idx):
        if self._seed is not None:
            # 逐样本确定性增强：同一 (seed, epoch, idx) ⇒ 完全相同的增强，
            # 与 worker 数/分片顺序无关（2026-10-15 审计 A2 的修复）。
            self._rng = np.random.RandomState(
                (self._seed * 100003 + self.epoch * 7919 + int(idx)) % (2 ** 31 - 1))
        row = self.rows[idx]
        rgb_path = _resolve_data_path(row["rgb_path"])
        # bbox 放在 index 的附加列（preprocess 写入）
        bbox = (float(row["bbox_x"]), float(row["bbox_y"]),
                float(row["bbox_w"]), float(row["bbox_h"]))
        # 三张裁剪（人脸 + 可选左右眼）使用**同一组几何抖动参数**：三个同种子的
        # RandomState 抽取序列一致 ⇒ 人脸被平移/缩放/旋转时眼部同步变换。
        if self.eye_crops and self.augment:
            _seed = int(self._rng.randint(0, 2 ** 31 - 1))
            rngs = tuple(np.random.RandomState(_seed) for _ in range(3))
        else:
            rngs = (self._rng, None, None)
        rgb_face, eye_l, eye_r = self._make_face(rgb_path, bbox, idx, rngs)

        if NORMALIZE_GAZE:
            # gaze 归一化到 [0,1]（除以设备屏幕点数）
            gx = float(row["gaze_x"]) / float(row["screen_w"])
            gy = float(row["gaze_y"]) / float(row["screen_h"])
        else:
            gx = float(row["gaze_x"])
            gy = float(row["gaze_y"])

        # 数据增强（训练集）：翻转时注视点 x 必须同步镜像；眼部同步翻转并互换
        if self.augment:
            flip = self._rng.random() < 0.5
            gain = self._rng.uniform(0.8, 1.2)
            bias = self._rng.uniform(-0.05, 0.05)
            rgb_face, gx = self._augment(rgb_face, gx, flip, gain, bias)
            if self.eye_crops:
                eye_l, eye_r = self._augment_eyes(eye_l, eye_r, flip, gain, bias)

        # 屏幕物理尺寸(cm) 与几何 token 输入（用构造期缓存的整列数组，缺列兜底才生效）
        cm_px_x, cm_px_y = (float(v) for v in self.cm_px[idx])
        sw_pt = float(self.screen_size[idx, 0])
        sh_pt = float(self.screen_size[idx, 1])
        w_cm = sw_pt * cm_px_x
        h_cm = sh_pt * cm_px_y
        screen_cm = torch.tensor([w_cm, h_cm], dtype=torch.float32)
        geom = screen_geometry(sw_pt, sh_pt, cm_px_x, cm_px_y)
        # 几何域随机化（与 GazeCapture 同源，见 common.randomize_screen_geometry）
        if self.geom_randomize > 0.0 and self._rng.random() < self.geom_randomize:
            out = randomize_screen_geometry_frame(
                gx, gy, w_cm, h_cm, self._rng,
                aspect_range=GEOM_RAND_ASPECT,
                width_range=GEOM_RAND_WIDTH_CM,
                height_range=GEOM_RAND_HEIGHT_CM)
            if out is not None:
                # 标签已重投影到合成屏 ⇒ screen_cm 必须同步替换，
                # 否则角度损失/ cm 指标用真实屏尺寸算，口径错配。
                gx, gy, geom, w_cm, h_cm = out
                screen_cm = torch.tensor([w_cm, h_cm], dtype=torch.float32)

        label = np.array([gx, gy], dtype=np.float32)

        # IMU 重力向量（若 index 含该列则返回，供 use_imu 模型使用；否则零向量占位）
        try:
            imu = torch.tensor([float(row["imu_x"]), float(row["imu_y"]),
                                float(row["imu_z"])], dtype=torch.float32)
        except (KeyError, ValueError):
            imu = torch.zeros(3, dtype=torch.float32)

        out = edict(rgb=rgb_face, screen_cm=screen_cm, imu=imu, geom=geom)
        if self.eye_crops:
            # 眼部放大支流（use_eye=True 的模型消费）：默认 collate 会堆成 [B,3,S,S]
            out.eye_l = eye_l
            out.eye_r = eye_r
        return out, self.target_transform(label)