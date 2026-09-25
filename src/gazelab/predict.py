#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_gaze.py — 单张图片视线预测 + 在原图画箭头

用法：
    python predict_gaze.py --image 图片路径 [--checkpoint 权重路径] [--out 输出路径]

功能：
    1. 检测图片中的人脸和左右眼位置
       - 优先用 mediapipe FaceLandmarker（468 点，双眼精确），模型文件 assets/face_landmarker.task
       - 若无 mediapipe 或模型文件，降级用 OpenCV Haar cascade（assets/haarcascade_*.xml）
       - 两者都不可用则按整图作为人脸区域、经验眼位
    2. 裁出人脸区域，resize 到 224x224
    3. 复刻 train.py 的 process_batch 提取特征（CLIP 语义 + CNN 局部 + hook 抓 token）
    4. 用训练好的 TransformerDeepSeek_gaze 预测 3D 视线方向
    5. 把 3D 视线投影到图片平面，从双眼中心画一个箭头

环境：
    在项目的 conda 环境运行（含 torch / clip / config / model 等）。
    Windows 上建议用 dl 环境（已装 mediapipe）。

输出：
    原图 + 箭头保存为 --out 指定的路径；默认与输入同目录，文件名加 _gaze 后缀。
    终端打印 3D 视线向量、yaw/pitch、双眼位置。

坐标说明（Gaze360 约定，与训练一致）：
    x 向右、y 向下、z 向前；箭头水平偏移 ∝ x，垂直偏移 ∝ y。
"""

import argparse
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from gazelab.config import DEVICE, CNN_PREPROCESS, ABLA_CONFIG
from gazelab.models.gazeformer import GEWithCLIPModel_zhao as GEWithCLIPModel
from gazelab.models.transformers import TransformerDeepSeek_gaze

CHECKPOINT_DEFAULT = "checkpoints/best—separate-added_Gaze360.pt"
ARROW_LEN_DEFAULT = 120


# ---------------------------------------------------------------------------
# 人脸 / 眼睛检测
# ---------------------------------------------------------------------------
class FaceEyeDetector:
    """封装两种检测方式：mediapipe（首选）和 Haar（后备）。"""

    def __init__(self):
        self.mp_landmarker = None
        self.face_cas = None
        self.eye_cas = None
        self._init_mediapipe()
        if self.mp_landmarker is None:
            self._init_haar()

    # ---- mediapipe ----
    def _init_mediapipe(self):
        mp_model = "assets/face_landmarker.task"
        if not os.path.exists(mp_model):
            print("[提示] mediapipe 模型文件缺失，回退 Haar 检测：", mp_model)
            return
        try:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            base = python.BaseOptions(model_asset_path=mp_model)
            opts = vision.FaceLandmarkerOptions(base_options=base, num_faces=1)
            self._mp_vision = vision
            self._mp_python = python
            self._mp_img_cls = None
            from mediapipe import Image as MPImage, ImageFormat as MPImageFormat
            self._MPImage = MPImage
            self._MPImageFormat = MPImageFormat
            self.mp_landmarker = vision.FaceLandmarker.create_from_options(opts)
            print("[检测器] mediapipe FaceLandmarker 就绪")
        except Exception as e:
            print(f"[提示] mediapipe 初始化失败，回退 Haar：{e}")
            self.mp_landmarker = None

    # ---- Haar ----
    def _init_haar(self):
        face_names = [
            "assets/haarcascade_frontalface_default.xml",
            os.path.join(os.path.dirname(cv2.__file__), "data",
                         "haarcascade_frontalface_default.xml"),
        ]
        eye_names = [
            "assets/haarcascade_eye.xml",
            os.path.join(os.path.dirname(cv2.__file__), "data",
                         "haarcascade_eye.xml"),
        ]
        face_xml = next((p for p in face_names if os.path.exists(p)), None)
        eye_xml = next((p for p in eye_names if os.path.exists(p)), None)
        if not face_xml or not eye_xml:
            print("[提示] 未找到 Haar cascade xml，将按整图作为人脸区域")
            return
        self.face_cas = cv2.CascadeClassifier(face_xml)
        self.eye_cas = cv2.CascadeClassifier(eye_xml)
        if self.face_cas.empty() or self.eye_cas.empty():
            self.face_cas = self.eye_cas = None
            print("[提示] Haar cascade 加载失败，将按整图作为人脸区域")
        else:
            print(f"[检测器] Haar cascade 就绪 (face={os.path.basename(face_xml)}, "
                  f"eye={os.path.basename(eye_xml)})")

    # ---- 主检测 ----
    def detect(self, image):
        """
        返回 (face_box, eye_centers)。
        face_box=(x,y,w,h) 原图像素；eye_centers=[(cx,cy),...] 原图像素，可能为空。
        """
        if self.mp_landmarker is not None:
            box, eyes = self._detect_mediapipe(image)
            if box is not None:
                return box, eyes
        if self.face_cas is not None:
            return self._detect_haar(image)
        return None, []

    def _detect_mediapipe(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_img = self._MPImage(image_format=self._MPImageFormat.SRGB, data=rgb)
        res = self.mp_landmarker.detect(mp_img)
        if not res.face_landmarks:
            return None, []
        lm = res.face_landmarks[0]
        h, w = image.shape[:2]
        # FaceMesh 索引：左眼外33 内133；右眼外263 内362
        left = ((lm[33].x + lm[133].x) / 2, (lm[33].y + lm[133].y) / 2)
        right = ((lm[263].x + lm[362].x) / 2, (lm[263].y + lm[362].y) / 2)
        # 人脸框：取全部关键点的外接矩形
        xs = [lm[i].x * w for i in range(468)]
        ys = [lm[i].y * h for i in range(468)]
        x0, x1, y0, y1 = int(min(xs)), int(max(xs)), int(min(ys)), int(max(ys))
        box = (x0, y0, x1 - x0, y1 - y0)
        eyes = [(int(left[0] * w), int(left[1] * h)),
                (int(right[0] * w), int(right[1] * h))]
        return box, eyes

    def _detect_haar(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cas.detectMultiScale(gray, scaleFactor=1.1,
                                               minNeighbors=5, minSize=(50, 50))
        if len(faces) == 0:
            return None, []
        faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        fx, fy, fw, fh = faces[0]
        roi = gray[fy:fy + fh, fx:fx + fw]
        eyes = self.eye_cas.detectMultiScale(roi, scaleFactor=1.1,
                                             minNeighbors=5, minSize=(20, 20))
        eye_centers = [(fx + ex + ew // 2, fy + ey + eh // 2)
                       for (ex, ey, ew, eh) in eyes]
        eye_centers.sort(key=lambda p: p[0])
        return (fx, fy, fw, fh), eye_centers


def arrow_start(face_box, eye_centers):
    """箭头起点 = 双眼中心；无眼睛时退化为脸框上 1/3 处（经验眼位）。"""
    if len(eye_centers) >= 2:
        xs = [p[0] for p in eye_centers]
        ys = [p[1] for p in eye_centers]
        return int(np.mean(xs)), int(np.mean(ys))
    if face_box is None:
        return None
    fx, fy, fw, fh = face_box
    return (fx + fw // 2, fy + int(fh * 0.35))


# ---------------------------------------------------------------------------
# 特征提取（复刻 train.py 的 process_batch）
# ---------------------------------------------------------------------------
def extract_features(batch, model):
    _input, label = batch
    _input.face = _input.face.to(DEVICE)
    _input.other_face = _input.other_face.to(DEVICE)
    label = label.to(DEVICE).float()

    hook_outputs = {}

    def vt_hook(module, inp, output):
        hook_outputs["full_tokens"] = inp[0].permute(1, 0, 2)

    hook_handle = model.model.visual.transformer.register_forward_hook(vt_hook)
    img_feats = model.encoder_i(_input.face).float()   # CLIP 输出 fp16 -> 转 float32
    hook_handle.remove()

    token_img_patch = None
    if "full_tokens" in hook_outputs:
        full_tokens = hook_outputs["full_tokens"]
        if full_tokens.dim() == 3:
            token_img_patch = full_tokens[:, 1:, :].float()

    label_feats = model.encoder_t2(model.label_tokens).float()
    img_norm = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-2)
    label_norm = label_feats / (label_feats.norm(dim=-1, keepdim=True) + 1e-2)

    sim_illum = model.logit_scale.exp().float() * img_norm @ model.illum_norm.T.float()
    sim_head = model.logit_scale.exp().float() * img_norm @ model.head_norm.T.float()
    sim_bg = model.logit_scale.exp().float() * img_norm @ model.bg_norm.T.float()
    sim_label = model.logit_scale.exp().float() * img_norm @ label_norm.T.float()

    idx_illum = sim_illum.argmax(dim=-1)
    idx_head = sim_head.argmax(dim=-1)
    idx_bg = sim_bg.argmax(dim=-1)
    idx_label = sim_label.argmax(dim=-1)

    selected_illum = model.illum_feats[idx_illum].float()
    selected_head = model.head_feats[idx_head].float()
    selected_bg = model.bg_feats[idx_bg].float()
    selected_label = label_feats[idx_label].float()

    feature_1 = img_feats + selected_illum + selected_head + selected_bg
    feature_1 = feature_1 / (feature_1.norm(dim=-1, keepdim=True) + 1e-3)
    feature_2 = img_feats + selected_label
    feature_2 = feature_2 / (feature_2.norm(dim=-1, keepdim=True) + 1e-3)

    feature_map = model.main_model(_input.other_face)["features"]
    B, C, H, W = feature_map.shape
    tokens_feature3 = feature_map.view(B, C, H * W).transpose(1, 2).float()

    features = {}
    features["label"] = label
    features["feature_1"] = feature_1 if ABLA_CONFIG.get('use_feature_1', True) \
        else torch.zeros_like(feature_1)
    features["feature_2"] = feature_2 if ABLA_CONFIG.get('use_feature_2', True) \
        else torch.zeros_like(feature_2)
    features["feature_3"] = tokens_feature3 if ABLA_CONFIG.get('use_feature_3', True) \
        else torch.zeros_like(tokens_feature3)
    if ABLA_CONFIG.get('use_feature_4', True):
        features["token_img_patch"] = token_img_patch
    else:
        features["token_img_patch"] = torch.zeros_like(token_img_patch) \
            if token_img_patch is not None else None
    return features


# ---------------------------------------------------------------------------
# 模型构建 + 加载
# ---------------------------------------------------------------------------
def build_models(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = GEWithCLIPModel().to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    # Transformer 超参与 train.py 的 TransformerConfigBase 一致
    cfg = dict(num_layers=12, embed_dim=512, inter_dim=2048, num_heads=8,
               n_routed_experts=4, n_activated_experts=2,
               n_shared_experts=2, moe_inter_dim=1024)
    transformer_model = TransformerDeepSeek_gaze(
        cfg["num_layers"], cfg["embed_dim"], cfg["inter_dim"], cfg["num_heads"],
        cfg["n_routed_experts"], cfg["n_activated_experts"], cfg["n_shared_experts"],
        cfg["moe_inter_dim"], d_model=768, out_dim=3,
    ).to(DEVICE)
    transformer_model.load_state_dict(ckpt["transformer_state_dict"])
    model.eval()
    transformer_model.eval()
    return model, transformer_model


# ---------------------------------------------------------------------------
# 3D 视线 -> 2D 箭头
# ---------------------------------------------------------------------------
def gaze_to_pixel_direction(gaze_3d, arrow_len=ARROW_LEN_DEFAULT):
    """3D 视线向量 -> 图片平面位移向量 (dx,dy)，长度固定为 arrow_len。

    方向约定：Gaze360 相机与被摄者面对面，+x 指向被摄者右侧，而图像里
    被摄者右侧显示在图像左侧（镜像），故水平方向 dx 需取反；
    垂直方向 +y 向下看 = 图像下方，dy 保持 y 不变。
    """
    x, y, _ = gaze_3d
    norm = np.hypot(x, y)
    if norm == 0:
        return 0.0, 0.0
    return -x / norm * arrow_len, y / norm * arrow_len


def gaze_to_yaw_pitch(gaze_3d):
    x, y, z = gaze_3d
    yaw = np.arctan2(x, -z)
    pitch = np.arcsin(np.clip(y, -1.0, 1.0))
    return yaw, pitch


def draw_gaze_arrow(image, start, gaze_3d, arrow_len=ARROW_LEN_DEFAULT):
    out = image.copy()
    dx, dy = gaze_to_pixel_direction(gaze_3d, arrow_len)
    end = (int(start[0] + dx), int(start[1] + dy))
    cv2.arrowedLine(out, start, end, (0, 0, 255), thickness=3,
                    line_type=cv2.LINE_AA, tipLength=0.3)
    return out


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def predict_single_image(image_path, ckpt_path, out_path, arrow_len):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"无法读取图片: {image_path}")

    # 1. 检测人脸 + 眼睛
    detector = FaceEyeDetector()
    face_box, eye_centers = detector.detect(image)
    if face_box is None:
        print("[警告] 未检测到人脸，按整图作为人脸区域。")
        face_box = (0, 0, image.shape[1], image.shape[0])

    fx, fy, fw, fh = face_box
    pad = int(min(fw, fh) * 0.1)
    x0, y0 = max(0, fx - pad), max(0, fy - pad)
    x1 = min(image.shape[1], fx + fw + pad)
    y1 = min(image.shape[0], fy + fh + pad)
    face_crop = image[y0:y1, x0:x1]
    if face_crop.size == 0:
        raise RuntimeError("人脸区域裁剪为空")

    # 2. 预处理
    from PIL import Image
    face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
    from gazelab.config import CLIP_PREPROCESS
    face_t = CLIP_PREPROCESS(face_pil).unsqueeze(0).to(DEVICE)
    other_face_t = CNN_PREPROCESS(face_pil).unsqueeze(0).to(DEVICE)

    # 3. 构建模型
    model, transformer_model = build_models(ckpt_path)

    # 4. 预测
    from easydict import EasyDict as edict
    batch = (edict(face=face_t, other_face=other_face_t), torch.zeros(1, 3))
    with torch.no_grad():
        raw_inputs = extract_features(batch, model)
        gaze_pred = transformer_model(raw_inputs)
        gaze_3d = gaze_pred[0].cpu().numpy()
    # 模型输出的是未归一化的视线向量（训练用 angular_loss 内部归一化），
    # 画箭头/算角度前归一化成单位向量
    gnorm = np.linalg.norm(gaze_3d)
    if gnorm > 0:
        gaze_3d = gaze_3d / gnorm

    # 5. 眼睛起点（face_box 已含 pad，直接映射回整图）
    start = arrow_start(face_box, eye_centers)
    if start is None:
        start = (x0 + (x1 - x0) // 2, y0 + int((y1 - y0) * 0.35))
    # 注意：face_box 来自整图检测，mediapipe 关键点坐标也是整图像素，无需再映射。

    # 6. 画箭头
    result = draw_gaze_arrow(image, start, gaze_3d, arrow_len)

    # 7. 输出
    cv2.imwrite(str(out_path), result)
    yaw, pitch = gaze_to_yaw_pitch(gaze_3d)
    print("=" * 52)
    print(f"输入图片    : {image_path}")
    print(f"人脸框      : (x={fx}, y={fy}, w={fw}, h={fh})")
    print(f"箭头起点    : {start}")
    print(f"3D 视线向量 : {gaze_3d[0]:.4f}, {gaze_3d[1]:.4f}, {gaze_3d[2]:.4f}")
    print(f"yaw(水平)   : {yaw:.4f} rad = {np.degrees(yaw):.2f}°")
    print(f"pitch(垂直) : {pitch:.4f} rad = {np.degrees(pitch):.2f}°")
    print(f"输出图片    : {out_path}")
    print("=" * 52)


def main():
    p = argparse.ArgumentParser(description="单张图片视线预测 + 画箭头")
    p.add_argument("--image", required=True, help="输入图片路径")
    p.add_argument("--checkpoint", default=CHECKPOINT_DEFAULT, help="权重文件路径")
    p.add_argument("--out", default=None, help="输出图片路径（默认同目录 _gaze 后缀）")
    p.add_argument("--arrow-len", type=int, default=ARROW_LEN_DEFAULT,
                   help="箭头像素长度")
    args = p.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"图片不存在: {img_path}")
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"权重不存在: {args.checkpoint}")
    out_path = args.out or str(img_path.parent / f"{img_path.stem}_gaze{img_path.suffix}")

    predict_single_image(img_path, args.checkpoint, out_path, args.arrow_len)


if __name__ == "__main__":
    main()