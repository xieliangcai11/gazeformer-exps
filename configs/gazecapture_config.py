"""GazeCapture 数据集配置。

GazeCapture（Krafka et al., CVPR 2016, "Eye Tracking for Everyone"）：
  1474 名被试、约 145 万帧的手机/平板前置摄像头注视数据。用途（本项目）：
  作为大规模"注视先验"预训练语料，之后再在 RGBDGaze 上微调 + 校准，
  以解决"仅用 36 名被试训练导致对新用户泛化不足"的瓶颈。

官方目录结构（每名被试一个 tar.gz，或已解压的同名目录）：
    NNNNN/
    ├── frames/00000.jpg ...      # 原始整帧（未裁剪，人脸框在图像像素坐标系）
    ├── appleFace.json            # 人脸框 X,Y,W,H,IsValid（相对整帧左上角）
    ├── appleLeftEye.json         # 左眼框（X,Y 相对"人脸框"左上角，不是整帧！）
    ├── appleRightEye.json        # 右眼框（同上）
    ├── faceGrid.json             # 人脸在 25x25 网格中的位置（iTracker 的输入之一）
    ├── dotInfo.json              # DotNum,XPts,YPts,XCam,YCam,Time
    ├── motion.json               # 60Hz 运动数据（重力/姿态），带 DotNum/Time
    ├── screen.json               # H,W,Orientation
    ├── frames.json               # 帧文件名列表（顺序即帧序号）
    └── info.json                 # TotalFrames,Dataset(train/val/test),DeviceName

坐标系结论（已在本仓库用样本数据实测确认，详见
tools/data/gazecapture/preprocess.py 顶部"坐标系说明"）：
  1. 图像帧在全部 4 种 Orientation 下都是"人脸正立"（左右眼向量滚转角中位数
     分别为 +0.0°/+0.0°/+3.8°/-2.3°，100% 落在 ±30° 内）。
  2. dotInfo 的 XPts/YPts 与相机/世界坐标系始终对齐：对同一屏幕尺寸，
     Orientation=1/2/3/4 拟合出的 (XPts,YPts)->(XCam,YCam) 线性变换完全一致
     （M≈diag(+s,-s)，列角度 0°/-90°）。
  => 图像坐标系与标签坐标系天然一致，**无需任何旋转**，直接按 bbox 裁剪即可。
     这与官方 PyTorch 预处理（pytorch/prepareDataset.py 不旋转）一致。
"""

from pathlib import Path

# 项目根目录 = 本文件(configs/gazecapture_config.py)的上一级；不依赖运行 CWD
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = _PROJECT_ROOT

# 归档（*.tar.gz）所在目录；若已解压，解压结果默认放 GAZECAPTURE_EXTRACT_DIR
GAZECAPTURE_ARCHIVE_DIR = _PROJECT_ROOT / "data" / "GazeCapture"

# 解压目标（tar.gz 里的顶层目录 NNNNN/ 会被原样释放到这里）
GAZECAPTURE_EXTRACT_DIR = _PROJECT_ROOT / "data" / "GazeCapture_extracted"

# 索引输出（构建后的统一标签/清单）
GAZECAPTURE_INDEX_DIR = _PROJECT_ROOT / "data" / "GazeCapture" / "index"

# 设备物理屏幕规格表（官方 code/apple_device_data.csv，已随仓库提供一份）
GAZECAPTURE_DEVICE_CSV = Path(__file__).resolve().parent.parent / "tools" / "data" / "gazecapture" / "apple_device_data.csv"

# 设备名别名 -> 官方表中的同名/同面板机型。
# 官方 CSV 未覆盖下列机型，但它们与所列机型是**同一块屏幕面板**（物理尺寸相同），
# 故直接复用其屏幕尺寸，避免为 cm 误差换算凭空编造数值。
# 命中别名时 preprocess 会打印日志，便于核对。
DEVICE_ALIASES = {
    "iphone4": "iPhone 4s",      # 3.5" 320x480，与 4s 同面板
    "iphonese": "iPhone 5s",     # 4.0" 320x568，与 5s 同面板
    "ipadmini4": "iPad Mini",    # 7.9" 768x1024，与 iPad Mini 同面板
}

# ---- 官方划分（info.json 的 Dataset 字段） ----
SPLITS = ("train", "val", "test")

# 图像目标尺寸（与 RGBDGaze 当前最优模型 g5/g6 一致，保证预训练/微调输入同尺度）
IMAGE_SIZE = 448

# 人脸裁剪边距倍数（相对人脸 bbox 最大边；与 RGBDGaze 保持一致）
CROP_MARGIN = 1.2

# 标签归一化：2D 屏幕坐标除以该帧 screen.json 的 W/H，落到 [0,1]
NORMALIZE_GAZE = True

# 用户眼睛到屏幕中心的基准距离（正视），用于把屏幕坐标误差换算成视线角度。
# GazeCapture 自带 XCam/YCam（厘米，相对相机）可做更精确换算，此处仅作缺省近似。
VIEWING_DISTANCE_CM = 30.0

# 训练/验证/测试可用性过滤阈值（被试帧数过少时跳过，避免噪声样本）
MIN_VALID_FRAMES_PER_SUBJECT = 10

# ---- 训练期几何增强（与 RGBDGaze 用同一套，保证两域增强分布一致）----
AUG_SHIFT = 0.10
AUG_SCALE = 0.25
AUG_ROTATE = 10.0

# ---- 几何域随机化（P2）：把标签重投影到"合成设备" ----
# 目的：GazeCapture 只有 0.562/0.563/0.750 三种宽高比，而 RGBDGaze(19.5:9) 是
# 0.462 —— 比预训练见过的最窄还窄 22%，属于外推。用真实物理标签重投影到合成
# 几何，把这一维变成内插（推导与验证见 gazelab.datasets.common.randomize_screen_geometry）。
# 期间相机相对屏幕中心的物理位置保持不变。
GEOM_RAND_PROB = 0.5            # 每个样本被随机化的概率
GEOM_RAND_ASPECT = (0.40, 0.80)  # 合成屏幕宽高比(短/长)，覆盖 19.5:9(0.462)~4:3(0.75)
GEOM_RAND_WIDTH_CM = (5.5, 8.5)  # 合成屏幕物理宽(cm)
GEOM_RAND_HEIGHT_CM = (10.0, 17.5)  # 合成屏幕物理高(cm)，用于过滤不合理组合
