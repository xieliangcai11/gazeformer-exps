"""眼/人脸检测的**共享工具**（只依赖 cv2 ✓ 零新依赖）。

被 `tools/data/gazecapture/*` 使用。RGBDGaze 的眼框仍走
`tools/data/rgbdgaze/build_eye_boxes.py`（MediaPipe + rot90 映射）✓ **本文件不改它** ✓。

设计要点：
  · 默认**不读索引**（GazeCapture 索引 149 万行 ⇒ 逐行扫很慢 ✗）；人脸框直接用 Haar ✓。
  · 需要索引人脸框做对照时，用 `lookup_rows` ✓（流式读取、命中即覆盖 ✓ 不缓存全表 ✓）。
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np


def load_cascades(root: Path):
    face = cv2.CascadeClassifier(str(root / "assets" / "haarcascade_frontalface_default.xml"))
    eye = cv2.CascadeClassifier(str(root / "assets" / "haarcascade_eye.xml"))
    if face.empty() or eye.empty():
        raise RuntimeError("级联文件加载失败 ✗（检查 assets/）")
    return face, eye


def detect(bgr, face_cas, eye_cas):
    """返回 (face_box, left, right, partial)。坐标均为**原图**坐标 ✓。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cas.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(40, 40))
    if len(faces) == 0:
        return None, None, None, True
    fx, fy, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    upper = gray[fy:fy + fh, fx:fx + fw][: int(fh * 0.62)]
    eyes = eye_cas.detectMultiScale(upper, scaleFactor=1.08, minNeighbors=4,
                                    minSize=(max(8, fw // 16), max(8, fw // 16)))
    boxes = sorted([(int(x) + fx, int(y) + fy, int(ew), int(eh)) for x, y, ew, eh in eyes],
                   key=lambda b: -b[2] * b[3])[:2]
    if len(boxes) == 2:
        boxes.sort(key=lambda b: b[0])
        return (fx, fy, fw, fh), boxes[0], boxes[1], False
    if len(boxes) == 1:
        x, y, ew, eh = boxes[0]
        mx = int(fx + (fw - ((x - fx) + ew / 2)) - ew / 2)     # 关于人脸中线镜像 ✓
        return (fx, fy, fw, fh), boxes[0], (mx, y, ew, eh), True
    # 全漏检 ⇒ 几何先验（人脸上半区左右两侧 ✓）并标 partial ✗
    ew, eh = max(10, fw // 4), max(10, fh // 5)
    ey = fy + int(fh * 0.25)
    return (fx, fy, fw, fh), (fx + int(fw * 0.18), ey, ew, eh), (fx + int(fw * 0.57), ey, ew, eh), True


def draw(bgr, face_box, left, right):
    vis = bgr.copy()
    if face_box:
        fx, fy, fw, fh = face_box
        cv2.rectangle(vis, (fx, fy), (fx + fw, fy + fh), (255, 0, 0), 2)      # 蓝=脸
    for b in (left, right):
        if b:
            x, y, w, h = b
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)        # 绿=眼
    return vis


def to_tile(img, cell):
    s = cell / max(img.shape[:2])
    return cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))


def montage(tiles, cols, cell):
    if not tiles:
        raise RuntimeError("没有可显示帧 ✗")
    th = max(t.shape[0] for t in tiles); tw = max(t.shape[1] for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    canvas = np.full((rows * th, cols * tw, 3), 30, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas[r * th:r * th + t.shape[0], c * tw:c * tw + t.shape[1]] = t
    return canvas


def lookup_rows(index_csv, want_paths, key="frame_path"):
    """流式查表 ✓：只保留想要的路径，命中数够了就提前结束 ✓（不构造全表 ✗）。"""
    want = set(want_paths); out = {}
    if not want:
        return out
    with open(index_csv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            k = row.get(key)
            if k in want:
                out[k] = row
                if len(out) == len(want):
                    break
    return out


# ---------------- MediaPipe 后端（对倒置/侧脸更稳 ✓） ----------------
#: 眼周轮廓索引（FaceMesh 468 点）
L_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
R_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]


def make_landmarker(root: Path):
    """创建 MediaPipe FaceLandmarker（IMAGE 模式 ✓）。mediapipe 缺失时调用方需自行回退 ✓。"""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_py
    from mediapipe.tasks.python import vision
    base = mp_py.BaseOptions(model_asset_path=str(root / "assets" / "face_landmarker.task"))
    opts = vision.FaceLandmarkerOptions(base_options=base, num_faces=1,
                                        running_mode=vision.RunningMode.IMAGE)
    return vision.FaceLandmarker.create_from_options(opts)


def detect_mp(bgr, landmarker):
    """MediaPipe 版：返回 (face_box, left, right, partial)。无网格 ⇒ 全 None ✓。"""
    import mediapipe as mp
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not res.face_landmarks:
        return None, None, None, True
    pts = res.face_landmarks[0]

    def box(idx):
        xy = np.array([[pts[i].x * w, pts[i].y * h] for i in idx])
        x0, y0 = xy.min(0).astype(int); x1, y1 = xy.max(0).astype(int)
        return int(x0), int(y0), int(x1 - x0), int(y1 - y0)

    le, re = box(L_EYE), box(R_EYE)
    xs = [p.x * w for p in pts]; ys = [p.y * h for p in pts]
    face = (int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys)))
    return face, le, re, False
