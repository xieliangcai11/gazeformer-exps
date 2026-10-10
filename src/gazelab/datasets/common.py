"""gazelab.datasets 通用图像工具（各数据集共用，单一来源）。

提供：
  - `to_tensor_np`   : numpy BGR uint8 -> torch tensor（RGB 通道、[0,1]）
  - `rgb_preprocess` : 上述 + ImageNet 归一化（DINOv2/ResNet 等通用输入约定）
  - `crop_face_square`: 以人脸 bbox 为中心、按边距倍数裁正方形并 resize

原先这些实现散落在 rgbdgaze.py 内，RGBDGaze 与 GazeCapture 都要用同一套
裁剪/归一化约定（否则预训练与微调的输入分布不一致），故上提到此处统一维护。
"""

import math

import cv2
import numpy as np
import torch

# ImageNet 统计量（与 torchvision 默认一致）
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

#: 公开别名，供需要自行归一化的调用方复用同一数值
IMAGENET_MEAN = _IMAGENET_MEAN
IMAGENET_STD = _IMAGENET_STD


def to_tensor_np(img_bgr: np.ndarray) -> torch.Tensor:
    """numpy BGR(H,W,3) uint8 -> tensor(C,H,W) float32 [0,1]，通道转 RGB。"""
    rgb = img_bgr[:, :, ::-1]  # BGR -> RGB
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 255.0


def rgb_preprocess(img_bgr: np.ndarray) -> torch.Tensor:
    """RGB 人脸：转 RGB + ImageNet 归一化。"""
    t = to_tensor_np(img_bgr)
    mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(3, 1, 1)
    return (t - mean) / std


def crop_face_square(img: np.ndarray, bbox_xywh, margin: float, size: int,
                     shift: float = 0.0, scale: float = 0.0, rotate: float = 0.0,
                     rng=None) -> np.ndarray:
    """按 bbox 裁人脸（中心 + 边长*margin），resize 到 (size, size)。

    可选的几何增强（需给出 rng；只应在训练集开启）：
        shift  : 裁剪中心随机偏移 ±shift*边长（如 0.10 = ±10%）
        scale  : 边长随机缩放 ±scale（如 0.25 = 边长 ×[0.75, 1.25]）
        rotate : 随机旋转 ±rotate 度（旋转在人脸块上做，用边缘复制填充）

    为什么要这些增强：GazeCapture（Apple 检测框）与 RGBDGaze（自有检测框）
    的人脸框松紧分布差异很大 —— "bbox 边长/帧短边" 中位数 0.467 vs 0.427，
    5~95% 区间 0.352~0.789 vs 0.312~0.591，GazeCapture 的方差是 2.2 倍。
    若不做几何抖动，模型容易依赖"人脸在裁剪图里的构图"这种域特有线索。

    裁剪框越界时按图像边界截断（不补边）；退化到空裁剪时返回全零小块，
    保证下游 resize 不会因空数组报错。rotate=0 时行为与旧实现完全一致。

    img      : HWC 图像（BGR 或单通道均可，通道数原样保留）
    bbox_xywh: (x, y, w, h)，与 img 同一像素坐标系
    """
    x, y, w, h = bbox_xywh
    H, W = img.shape[:2]
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * margin
    ang = 0.0
    if rng is not None:
        if shift > 0:
            cx += rng.uniform(-1, 1) * shift * side
            cy += rng.uniform(-1, 1) * shift * side
        if scale > 0:
            side *= 1.0 + rng.uniform(-scale, scale)
        if rotate > 0:
            ang = rng.uniform(-rotate, rotate)

    if ang:
        # 旋转后要不裁掉内容，取区域需放大到内接正方形能容纳 side
        rad = math.radians(abs(ang))
        grab = side * (math.cos(rad) + math.sin(rad))
        half = grab / 2
        x0 = max(0, int(cx - half)); x1 = min(W, int(cx + half))
        y0 = max(0, int(cy - half)); y1 = min(H, int(cy + half))
        patch = img[y0:y1, x0:x1]
        if patch.size == 0:
            patch = np.zeros((2, 2) + img.shape[2:], dtype=img.dtype)
        ph, pw = patch.shape[:2]
        m = cv2.getRotationMatrix2D((pw / 2.0, ph / 2.0), ang, 1.0)
        # 边缘复制填充（与官方 GazeCapture cropRepeatingEdge.m 的做法一致，
        # 比黑边更接近真实图像统计）
        patch = cv2.warpAffine(patch, m, (pw, ph), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
        # 中心裁回 side 见方
        ch = min(ph, max(1, int(side)))
        cw = min(pw, max(1, int(side)))
        top, left = (ph - ch) // 2, (pw - cw) // 2
        crop = patch[top:top + ch, left:left + cw]
    else:
        half = side / 2
        x0 = max(0, int(cx - half))
        x1 = min(W, int(cx + half))
        y0 = max(0, int(cy - half))
        y1 = min(H, int(cy + half))
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            crop = np.zeros((2, 2) + img.shape[2:], dtype=img.dtype)

    return cv2.resize(crop, (size, size))


def geom_vector(w_cm: float, h_cm: float) -> torch.Tensor:
    """由物理屏幕尺寸(cm)构造几何 token 向量 [宽高比, log(宽), log(h)]。"""
    return torch.tensor(
        [w_cm / h_cm, math.log(w_cm), math.log(h_cm)], dtype=torch.float32)


def screen_geometry(screen_w: float, screen_h: float,
                    cm_px_x: float, cm_px_y: float) -> torch.Tensor:
    """屏幕几何向量 [宽高比, log(物理宽cm), log(物理高cm)]，供模型的几何 token。

    为什么用物理厘米而不是点数：点数只反映"逻辑分辨率"（同一块屏可开显示缩放，
    点数会变），而物理尺寸才决定"屏幕在用户视野里张多大"。跨设备时
    (1) 宽高比决定归一化 x/y 的物理含义比例（GazeCapture 0.562~0.750 vs
        RGBDGaze 0.462）；
    (2) 物理尺寸决定同一归一化偏移对应多大的视角（实测 11.94~25.47 cm）。
    两者都无法从人脸图推断（屏幕在正面摄像头画面里不可见），必须显式提供。

    log 是为了让 5~20cm 的跨度在数值上更均匀。物理尺寸未知时（NaN）用 1cm 占位。
    """
    w_cm = screen_w * cm_px_x
    h_cm = screen_h * cm_px_y
    if not math.isfinite(w_cm) or w_cm <= 0:
        w_cm = 1.0
    if not math.isfinite(h_cm) or h_cm <= 0:
        h_cm = 1.0
    return geom_vector(w_cm, h_cm)


def randomize_screen_geometry(u: float, v: float, w_cm: float, h_cm: float, rng,
                              aspect_range, width_range, height_range,
                              max_tries: int = 8):
    """几何域随机化：把 (标签, 屏幕几何) 重投影到一个"合成设备"上。

    物理模型（已在 GazeCapture 真实数据上逐位验证）：
      · 相机在屏幕上的归一化位置 u_c 对同一机型是**严格常数**
        （实测 iPad Air 2 / iPhone 5 / 5S 的 u_c = 0.5000 ± 0.0000，
          iPhone 6 = 0.3181 ± 0.0000）；
      · 横向尺度严格等于屏幕物理宽：XCam = W_cm · (u - u_c)。
    因此，在"相机相对屏幕中心的物理位置不变、只更换屏幕尺寸"的假设下，
    同一个物理注视点在合成屏幕上的归一化坐标是**纯比例变换**：

        u' = 0.5 + (u - 0.5) · (W_cm / W')
        v' = 0.5 + (v - 0.5) · (H_cm / H')

    为什么需要它：GazeCapture 只有 0.562 / 0.563 / 0.750 三种宽高比，而
    RGBDGaze（现代 19.5:9 手机）是 0.462 —— 比预训练见过的最窄还要窄 22%，
    属于**外推**。用真实物理标签重投影到合成几何，可把这一维从"外推"变成"内插"；
    物理标注是真实的，只有"假设的屏幕形状"是合成的，不属于编造数据。

    参数范围来自 configs（aspect_range / width_range / height_range）。
    返回 (u', v', geom)；若合成点落到屏幕外、或多次抽样都没抽到合理的
    (宽, 高) 组合，则返回 None，调用方应放弃本次随机化（用真实标签）。
    """
    for _ in range(max_tries):
        aspect = rng.uniform(*aspect_range)
        w = rng.uniform(*width_range)
        h = w / aspect
        if not (height_range[0] <= h <= height_range[1]):
            continue
        up = 0.5 + (u - 0.5) * (w_cm / w)
        vp = 0.5 + (v - 0.5) * (h_cm / h)
        if 0.0 <= up <= 1.0 and 0.0 <= vp <= 1.0:
            return up, vp, geom_vector(w, h)
    return None
