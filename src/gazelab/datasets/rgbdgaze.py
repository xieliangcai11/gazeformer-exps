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
    randomize_screen_geometry,
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
    ):
        """裁剪朝向说明（2026-10-14 定案，勿再改回）

        历史实现先 np.rot90(img, 1) 再按索引 bbox 裁。实测该朝向让**人脸躺倒**
        （rot90 后两眼中心连线 Δx=1.0 px / Δy=208 px，即竖直），而原图本身人脸
        是正立的（MediaPipe 眼框 137x40 = 宽>高）⇒ 躺倒输入对 DINOv2 属分布外，
        白费正立人脸先验，伤的是跨被试泛化。

        现改为**直接用原图**，并把索引 bbox（作者坐标系＝rot90 后）反映射回来。
        验证：反解后框中心与 MediaPipe 人脸框中心相差 14 px（脸边长 2.7%）。
        眼框表随之简化为 jsonl 的原图坐标，不再需要任何坐标映射。
        """
        super().__init__()
        self.index_csv = Path(index_csv)
        # 列式索引：list(csv.DictReader) 会按 1.9KB/行 生成 dict，
        # 149 万行时约 2.4GB/进程，fork 给 DataLoader worker 后触发写时复制，
        # 是多卡训练卡顿的主因。TableIndex 取值语义与 DictReader 完全一致。
        self.rows = TableIndex(self.index_csv)

        self.color_transform = color_transform

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
        # 输入分辨率（None = 使用配置的 IMAGE_SIZE）
        self.image_size = int(image_size) if image_size else IMAGE_SIZE

        # ---- 眼部放大支流（可选）----
        # 动机（实测）：眼区框（含 1.6 倍边距）中位 128 px，放大到 224 时虹膜约 62 px，
        # 越过"1°/px"门槛（约 58 px）；而整脸裁剪下虹膜只剩约 14 px ≈ DINOv2 一个 patch。
        # 框表由 tools/data/rgbdgaze/build_eye_boxes.py 生成（**原图坐标**，与裁剪同帧；
        # 8 帧检测失败用同被试中位数兜底，filled 掩码备查）。
        self.eye_crops = bool(eye_crops)
        if self.eye_crops:
            _npz = np.load(Path(self.index_csv).parent / "eye_boxes.npz")
            # 直接是 jsonl（MediaPipe）的原图坐标：与裁剪所用帧一致，无需映射
            self.eye_boxes = _npz["boxes"].astype(np.float32)
            self.eye_filled = _npz["filled"]
            if len(self.eye_boxes) != len(self.rows):
                raise ValueError(
                    f"eye_boxes.npz 行数 {len(self.eye_boxes)} 与索引 {len(self.rows)} "
                    f"不一致，请重跑 tools/data/rgbdgaze/build_eye_boxes.py")

        # 每样本的"像素 -> 厘米"换算系数（来自 index，用于把预测误差转成物理 cm）
        try:
            self.cm_px = np.array(
                [[float(r["cm_px_x"]), float(r["cm_px_y"])] for r in self.rows],
                dtype=np.float32,
            )
            self.screen_size = np.array(
                [[float(r["screen_w"]), float(r["screen_h"])] for r in self.rows],
                dtype=np.int32,
            )
        except KeyError:
            # 旧版 index 无 cm 列时退化为近似（角标）
            self.cm_px = np.full((len(self.rows), 2), 0.0182, dtype=np.float32)
            self.screen_size = np.ones((len(self.rows), 2), dtype=np.int32)

    def __len__(self):
        return len(self.rows)

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

        # 原图本身人脸正立（见 __init__ docstring）：不旋转，直接裁；
        # 索引 bbox 在作者（rot90 后）坐标系，需反映射回原图。
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

    def _crop_eye(self, oriented, idx, off, rng):
        """从已旋转的原分辨率整帧上裁一只眼（框来自 eye_boxes.npz）。

        复用 crop_face_square：margin=1.0 时它以 bbox 中心取 max(w,h) 的正方形，
        正好把整只眼含边距框进去；越界由该函数内部补齐，不会产生黑边。
        """
        box = tuple(float(v) for v in self.eye_boxes[idx, off:off + 4])
        eye = crop_face_square(oriented, box, 1.0, self.image_size,
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

        # 屏幕物理尺寸(cm) 与几何 token 输入
        cm_px_x = float(row["cm_px_x"])
        cm_px_y = float(row["cm_px_y"])
        w_cm = float(row["screen_w"]) * cm_px_x
        h_cm = float(row["screen_h"]) * cm_px_y
        screen_cm = torch.tensor([w_cm, h_cm], dtype=torch.float32)
        geom = screen_geometry(float(row["screen_w"]), float(row["screen_h"]),
                               cm_px_x, cm_px_y)
        # 几何域随机化（与 GazeCapture 同源，见 common.randomize_screen_geometry）
        if self.geom_randomize > 0.0 and self._rng.random() < self.geom_randomize:
            out = randomize_screen_geometry(
                gx, gy, w_cm, h_cm, self._rng,
                aspect_range=GEOM_RAND_ASPECT,
                width_range=GEOM_RAND_WIDTH_CM,
                height_range=GEOM_RAND_HEIGHT_CM)
            if out is not None:
                gx, gy, geom = out

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