"""GazeCapture 数据集类。

用途：作为大规模"注视先验"预训练语料（1474 名被试），在 RGBDGaze 微调之前
先让模型学会"跨用户的通用注视映射"，从而解决 RGBDGaze 仅 36 名训练被试导致
的对新用户泛化不足问题。

返回协议与 RGBDGazeDataset 保持一致（便于复用同一套训练脚本/模型）：
    __getitem__ -> (edict(rgb=..., screen_cm=..., imu=..., geom=..., face_grid=...), gaze)

其中：
    rgb       [3, S, S]  RGB 人脸裁剪图（ImageNet 归一化）
    screen_cm [2]        该帧屏幕的物理宽高（厘米），供误差换算
    imu       [3]        重力向量（设备坐标系），来自 motion.json
    geom      [3]        屏幕几何 [宽高比, log(物理宽cm), log(物理高cm)]，喂给模型的几何 token
    face_grid [4]        人脸在 25x25 网格中的 (x, y, w, h)（iTracker 风格输入）
    gaze      [2]        归一化 2D 屏幕注视点 (x, y) ∈ [0,1]

edict 内全部为 Tensor，可直接走 torch 的 default_collate，无需自定义 collate_fn。

索引由 tools/data/gazecapture/preprocess.py 生成。
"""

from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import torch
from easydict import EasyDict as edict
from torch.utils.data import Dataset

from configs.gazecapture_config import (
    GAZECAPTURE_INDEX_DIR,
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
    """把索引里的相对路径（相对项目根）解析成绝对路径；绝对路径原样返回。"""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _to_tensor_label(label):
    """把 2D 标签转成 float32 tensor。命名函数便于多进程 pickle。"""
    return torch.tensor(label, dtype=torch.float32)


class GazeCaptureDataset(Dataset):
    """GazeCapture：从索引 CSV 读取 (整帧路径, 人脸框, 屏幕注视点, 设备几何, IMU)。

    索引列（tools/data/gazecapture/preprocess.py 写出）：
        frame_path, bbox_x, bbox_y, bbox_w, bbox_h, device, screen_w, screen_h,
        gaze_x, gaze_y, cm_px_x, cm_px_y, imu_x, imu_y, imu_z, subject,
        orientation, split, face_grid_x, face_grid_y, face_grid_w, face_grid_h,
        dot_num, frame_index
    其中 gaze_x/gaze_y 是屏幕点，screen_w/screen_h 是该帧屏幕点数。

    坐标系：图像与标签在所有 orientation 下天然对齐，**不做旋转**，直接按
    bbox 裁剪（结论与实测依据见 preprocess.py 头部说明）。
    """

    def __init__(
        self,
        index_csv: Path = None,
        color_transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        augment: bool = False,
        aug_shift: float = AUG_SHIFT,
        aug_scale: float = AUG_SCALE,
        aug_rotate: float = AUG_ROTATE,
        geom_randomize: float = None,
        image_size: int = None,
        eye_crops: bool = False,
        eye_scale: float = 0.0,
        geom_orient: bool = False,
        target_space: str = "norm",
    ):
        super().__init__()
        # 本数据集的所有增强/翻转/几何重投影都建立在归一化坐标上；像素口径下
        # 翻转会得到负值、重投影的 [0,1] 检查恒失败（静默 no-op）⇒ 直接约束。
        if not NORMALIZE_GAZE:
            raise ValueError("GazeCaptureDataset 要求 NORMALIZE_GAZE=True（见 configs）")
        self.index_csv = Path(index_csv) if index_csv else GAZECAPTURE_INDEX_DIR / "index.csv"
        # 列式索引：list(csv.DictReader) 会按 1.9KB/行 生成 dict，
        # 149 万行时约 2.4GB/进程，fork 给 DataLoader worker 后触发写时复制，
        # 是多卡训练卡顿的主因。TableIndex 取值语义与 DictReader 完全一致。
        self.rows = TableIndex(self.index_csv)

        # 默认用通用的 ImageNet 归一化（与 RGBDGaze 一致，保证预训练/微调同分布）
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
        # 逐样本确定性增强种子（与 RGBDGazeDataset 同一设计；2026-10-16 移植）：
        # None = 旧行为（无播种 ⇒ 同一 --seed 两次训练增强不同 ✗ 不可复现）
        self._seed = None
        self.epoch = 0
        self.image_size = int(image_size) if image_size else IMAGE_SIZE

        # 与 RGBDGazeDataset 保持 API 一致：评估脚本需要按样本取
        # 「每点厘米」与「屏幕点数」把归一化误差换算成物理厘米/角度。
        # 整列取出而非逐行建 dict：GazeCapture train 划分 125 万行时，逐行 build
        # 是本数据集启动期的主要开销（多卡时每个 rank 各付一次）。
        self.cm_px = np.stack(
            [self.rows.float_column("cm_px_x", np.float32),
             self.rows.float_column("cm_px_y", np.float32)], axis=1)
        self.screen_size = np.stack(
            [self.rows.float_column("screen_w", np.float32),
             self.rows.float_column("screen_h", np.float32)], axis=1)

        # ---- 眼部放大支流（可选）----
        # 框表由 tools/data/gazecapture/build_eye_boxes_mp.py 生成
        # （boxes = 左眼(x,y,w,h)+右眼(x,y,w,h)，原图坐标；filled 标记兜底/退化）。
        self.eye_crops = bool(eye_crops)
        self.eye_scale = float(eye_scale)
        if target_space not in ("norm", "cm"):
            raise ValueError("target_space 只能是 'norm' 或 'cm'")
        self.target_space = target_space
        # ---- 朝向 token（GazeCapture）：设备/朝向是最大误差来源 ----
        # 1=竖屏、2=倒竖（iPad）、3/4=横屏。作为 4 维 one-hot 追加到几何 token，
        # 让模型显式知道"屏幕坐标轴是哪个方向"，避免从图像里反推。
        self.geom_orient = bool(geom_orient)
        # (设备族, 朝向) 分组标签：供 GroupBalancedSampler 均衡采样（iPad 只占 14%）
        _codes, _uniq = self.rows.code_column("device")
        _fam = np.array([1 if u.startswith("iPad") else 0 for u in _uniq], dtype=np.int64)
        _fam = _fam[_codes]
        _ori = np.clip(self.rows.float_column("orientation", np.int64) - 1, 0, 3)
        self.sample_groups = (_fam * 4 + _ori).astype(np.int64)
        if self.eye_crops:
            npz_path = self.index_csv.parent / "eye_boxes.npz"
            if not npz_path.is_file():
                raise FileNotFoundError(
                    f"缺少眼框表 {npz_path} ⇒ 先跑 "
                    f"tools/data/gazecapture/build_eye_boxes_mp.py")
            with np.load(npz_path) as _z:
                self.eye_boxes = _z["boxes"].astype(np.float32)
                self.eye_filled = _z["filled"]
                _idx = _z["idx"]
            n = len(self.rows)
            if len(self.eye_boxes) != n or not np.array_equal(_idx, np.arange(n)):
                raise ValueError(
                    f"eye_boxes.npz 与索引不对齐（npz {len(self.eye_boxes)} 行 / "
                    f"索引 {n} 行）⇒ 先合并分片："
                    f"build_eye_boxes_mp.py --merge ... --out {npz_path}")

    def __len__(self):
        return len(self.rows)

    def set_aug_seed(self, seed):
        """设定增强的逐样本确定性种子（None = 恢复旧的不播种行为）。

        与 RGBDGazeDataset 同设计：修复"同一 --seed 两次训练增强不同"（GazeCapture
        此前也是 `RandomState()` 无播种 ⇒ 不可复现 ✗）。
        """
        self._seed = None if seed is None else int(seed)

    def set_epoch(self, epoch):
        """训练循环每轮调用：让同一 idx 在不同 epoch 拿到不同增强。"""
        self.epoch = int(epoch)

    def aug_summary(self):
        """**实际生效**的增强/几何参数（供日志与侧车如实记录）。

        此前该数据集缺此方法 ⇒ train.py 的 hasattr 判断静默回退 ⇒ 日志把
        aug(shift/scale/rot) 打成 0.0/0.0/0.0 ✗（而实际是在增强的）。
        """
        return dict(augment=bool(self.augment), shift=self.aug_shift,
                    scale=self.aug_scale, rotate=self.aug_rotate,
                    geom_randomize=self.geom_randomize,
                    eye_crops=self.eye_crops, eye_scale=self.eye_scale,
                    geom_orient=self.geom_orient,
                    orient="raw", image_size=self.image_size, aug_seed=self._seed)

    @staticmethod
    def _augment(rgb_t, gx, flip, gain, bias):
        """按调用方抽好的随机量增强（人脸与眼部必须看到同一组成像条件）。"""
        if flip:
            rgb_t = torch.flip(rgb_t, dims=[2])
            gx = 1.0 - gx
        rgb_t = torch.clamp(rgb_t * gain + bias, -2.5, 2.5)   # ImageNet 归一化范围
        return rgb_t, gx

    @staticmethod
    def _post_eye(e, flip, gain, bias):
        if flip:
            e = torch.flip(e, dims=[2])
        return torch.clamp(e * gain + bias, -2.5, 2.5)

    def _crop_eye(self, raw, idx, off, rng):
        """从原图裁一只眼的放大正方形（框来自 eye_boxes.npz 的 (x,y,w,h)）。"""
        box = tuple(float(v) for v in self.eye_boxes[idx, off:off + 4])
        if self.eye_scale > 0:
            # 以眼框高度为基准取正方形 ⇒ 虹膜在裁剪中占比更高
            _ex, _ey, _ew, _eh = box
            _s = max(self.eye_scale * _eh, 1.0)
            _cx, _cy = _ex + _ew / 2.0, _ey + _eh / 2.0
            box = (_cx - _s / 2.0, _cy - _s / 2.0, _s, _s)
        eye = crop_face_square(raw, box, 1.0, self.image_size,
                               shift=self.aug_shift, scale=self.aug_scale,
                               rotate=self.aug_rotate, rng=rng)
        return self.color_transform(eye) if self.color_transform is not None else eye

    def __getitem__(self, idx):
        if self._seed is not None:
            # 逐样本确定性增强：同一 (seed, epoch, idx) ⇒ 完全相同的增强，
            # 与 worker 数/分片顺序无关（与 RGBDGazeDataset 同一设计）
            self._rng = np.random.RandomState(
                (self._seed * 100003 + self.epoch * 7919 + int(idx)) % (2 ** 31 - 1))
        row = self.rows[idx]

        # ---- 读整帧 + 按人脸框裁正方形人脸图 ----
        frame_path = _resolve_data_path(row["frame_path"])
        img = cv2.imread(str(frame_path))
        if img is None:
            raise FileNotFoundError(
                f"无法读取 GazeCapture 帧: {frame_path}（文件缺失或损坏？）")
        bbox = (float(row["bbox_x"]), float(row["bbox_y"]),
                float(row["bbox_w"]), float(row["bbox_h"]))
        # 人脸与左右眼使用**同一组**几何抖动参数（同种子 RandomState ⇒ 抽样序列一致）
        if self.eye_crops and self.augment:
            _seed = int(self._rng.randint(0, 2 ** 31 - 1))
            rngs = tuple(np.random.RandomState(_seed) for _ in range(3))
        else:
            rngs = (self._rng, None, None)
        face = crop_face_square(img, bbox, CROP_MARGIN, self.image_size,
                                shift=self.aug_shift, scale=self.aug_scale,
                                rotate=self.aug_rotate, rng=rngs[0])
        if self.color_transform is not None:
            face = self.color_transform(face)
        eye_l = eye_r = None
        if self.eye_crops:
            eye_l = self._crop_eye(img, idx, 0, rngs[1])
            eye_r = self._crop_eye(img, idx, 4, rngs[2])

        # ---- 标签：屏幕点归一化到 [0,1] ----
        if NORMALIZE_GAZE:
            gx = float(row["gaze_x"]) / float(row["screen_w"])
            gy = float(row["gaze_y"]) / float(row["screen_h"])
        else:
            gx = float(row["gaze_x"])
            gy = float(row["gaze_y"])

        if self.augment:
            flip = self._rng.random() < 0.5
            gain = self._rng.uniform(0.8, 1.2)
            bias = self._rng.uniform(-0.05, 0.05)
            face, gx = self._augment(face, gx, flip, gain, bias)
            if self.eye_crops:
                if flip:                      # 镜像后左右眼互换
                    eye_l, eye_r = eye_r, eye_l
                eye_l = self._post_eye(eye_l, flip, gain, bias)
                eye_r = self._post_eye(eye_r, flip, gain, bias)

        # ---- 辅助信息 ----
        # 屏幕物理尺寸(cm)：由"每点厘米"乘屏幕点数得到（设备不在规格表时为 NaN）
        cm_px_x, cm_px_y = (float(v) for v in self.cm_px[idx])
        sw_pt = float(self.screen_size[idx, 0])
        sh_pt = float(self.screen_size[idx, 1])
        w_cm = sw_pt * cm_px_x
        h_cm = sh_pt * cm_px_y
        screen_cm = torch.tensor([w_cm, h_cm], dtype=torch.float32)
        # IMU 重力向量（preprocess 已按 DotNum+Time 从 motion.json 对齐；缺失为 0）
        imu = torch.tensor(
            [float(row["imu_x"]), float(row["imu_y"]), float(row["imu_z"])],
            dtype=torch.float32,
        )
        # 屏幕几何 token 输入（跨设备泛化的关键，见模型 docstring）
        geom = screen_geometry(sw_pt, sh_pt, cm_px_x, cm_px_y)
        # 几何域随机化：把 (标签, 几何) 一起重投影到合成设备，把"宽高比外推"
        # 变成"内插"（推导与实测验证见 common.randomize_screen_geometry）。
        if self.geom_randomize > 0.0 and self._rng.random() < self.geom_randomize:
            out = randomize_screen_geometry_frame(
                gx, gy, w_cm, h_cm, self._rng,
                aspect_range=GEOM_RAND_ASPECT,
                width_range=GEOM_RAND_WIDTH_CM,
                height_range=GEOM_RAND_HEIGHT_CM)
            if out is not None:
                # 标签已重投影到合成屏 ⇒ screen_cm 必须同步替换（否则角度/ cm 口径错配）
                gx, gy, geom, w_cm, h_cm = out
                screen_cm = torch.tensor([w_cm, h_cm], dtype=torch.float32)

        if self.geom_orient:
            o = int(float(row["orientation"])) if row.get("orientation") else 1
            oh = torch.zeros(4, dtype=torch.float32)
            if 1 <= o <= 4:
                oh[o - 1] = 1.0
            geom = torch.cat([geom, oh])

        # 归一化屏幕坐标（评测/兼容旧口径）
        label = np.array([gx, gy], dtype=np.float32)
        # 标准物理厘米：屏幕中心为原点，旋转到竖屏标准方向
        phi = float(np.arctan2(float(imu[0]), -float(imu[1])))
        xc = (gx - 0.5) * w_cm
        yc = (gy - 0.5) * h_cm
        c, s_ = np.cos(phi), np.sin(phi)
        target_cm = torch.tensor([c * xc + s_ * yc,
                                  -s_ * xc + c * yc], dtype=torch.float32)
        # 人脸在 25x25 网格中的位置（iTracker 风格输入特征）
        face_grid = torch.tensor(
            [float(row["face_grid_x"]), float(row["face_grid_y"]),
             float(row["face_grid_w"]), float(row["face_grid_h"])],
            dtype=torch.float32,
        )

        out = edict(rgb=face, screen_cm=screen_cm, imu=imu, geom=geom,
                     face_grid=face_grid, phi=torch.tensor(phi),
                     orientation=torch.tensor(
                         int(float(row.get("orientation", 0) or 0))),
                     label_norm=torch.from_numpy(label.copy()),
                     target_cm=target_cm)
        if self.eye_crops:
            out.eye_l = eye_l
            out.eye_r = eye_r
        y = target_cm if self.target_space == "cm" else self.target_transform(label)
        return out, y
