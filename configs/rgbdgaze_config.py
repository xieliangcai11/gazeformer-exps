"""RGBDGaze 数据集配置。

此文件是 RGBDGaze 适配的配置中心。所有路径、超参、数据/标签定义集中在
这里定义，供本次适配的所有模块引用，保证单一数据来源 (DRY)。
"""

from pathlib import Path

# 项目根目录 = 本文件(configs/rgbdgaze_config.py)的上一级；使路径不依赖运行 CWD
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# RGBDGaze 目录结构（官方）
# data/RGBDGaze_dataset/
# ├── p{1..45}/
# │   └── decoded/
# │       ├── sitting/  {rgb/, depth/, label.csv}
# │       ├── standing/ {rgb/, depth/, label.csv}
# │       ├── walking/  {rgb/, depth/, label.csv}
# │       └── lying/    {rgb/, depth/, label.csv}
# ├── README.txt
# ├── iphone_spec.csv
# └── (p*/intrinsic.json)

# 原始数据根目录（官方目录，只读）
RGBDGaze_RAW_ROOT = _PROJECT_ROOT / "data" / "RGBDGaze" / "RGBDGaze_dataset"

# 活动中包含的姿态
ACTIVITIES = ("sitting", "standing", "walking", "lying")

# 索引输出（构建后的统一标签/清单）
RGBDGaze_INDEX_DIR = _PROJECT_ROOT / "data" / "RGBDGaze" / "index"

# 屏幕规格表（device -> 屏幕像素宽高 (w_pt, h_pt)，来自 iphone_spec.csv）
IPHONE_SPEC_CSV = RGBDGaze_RAW_ROOT / "iphone_spec.csv"

# 相机内参（每被试 intrinsic.json）
# 若无则用默认近似（data 中大多是同一矩阵）

# 图像目标尺寸（双流：RGB 与 depth 都 resize 到该尺寸）
IMAGE_SIZE = 224

# 人脸裁剪边距倍数（相对人脸 bbox 最大边）
CROP_MARGIN = 1.2

# 标签归一化：2D 屏幕坐标。true 则除以屏幕尺寸归一化到 [0,1]
NORMALIZE_GAZE = True

# 归一化后 gaze 目标范围（用于回归，配合 sigmoid/tanh 或直接回归）
# 若 NORMALIZE_GAZE: 目标在 [0,1] 左右