import torch
import clip
from torchvision.transforms import transforms

# 路径与输出目录定义已抽到 configs/paths.py（那里不加载 CLIP，训练脚本用它可以
# 避免"仅为拿 DEVICE/experiment_dirs 就把 CLIP 加载到 cuda:0"）。此处 re-export
# 这些名字，保证既有 `from configs.gaze360_config import *` 的调用不受影响。
from configs.paths import (
    PROJECT_ROOT as _PROJECT_ROOT,  # noqa: F401  (re-export，供 import * 使用)
    DATASETS_PATH,                  # noqa: F401
    OUT_PATH,                       # noqa: F401
    CHECKPOINTS_PATH,               # noqa: F401
    LOGS_PATH,                      # noqa: F401
    RUNS_PATH,                      # noqa: F401
    experiment_dirs,                # noqa: F401
)

# MPIIFaceGaze
# EyeDiap
# Gaze360
# ETH-XGaze
TRAIN_RUN_NAME = "ResNet-50_gaze360_train"
TEST_RUN_NAME = "ResNet-50_egaze360_test"
CNN_MODEL = "ResNet-50"
# TRAIN_DATASET_NAME = "ETH-XGaze"
TRAIN_DATASET_NAME = "Gaze360"
# TRAIN_DATASET_NAME = "EyeDiap"
# TRAIN_DATASET_NAME = "MPIIFaceGaze"
# TEST_DATASET_NAME = "ETH-XGaze"
# TEST_DATASET_NAME = "MPIIFaceGaze"
# TEST_DATASET_NAME = "EyeDiap"
TEST_DATASET_NAME = "Gaze360"

IS_TRAIN = False
TEST_EPOCH = 50
TEST_CHECKPOINT = f"epoch_{TEST_EPOCH}.pth"
# （OUT_PATH / CHECKPOINTS_PATH / LOGS_PATH / RUNS_PATH / experiment_dirs
#   的定义已移到 configs/paths.py，见文件开头的 re-export）
SEED = 0
is_ablation = True
ABLA_CONFIG = {
    'use_feature_1': True,
    'use_feature_2': True,
    'use_feature_3': True,
    'use_feature_4': True,  
}
NUM_EPOCHS = 50
BATCH_SIZE = 64
TEST_STEP = 10
SAVE_STEP = 10
NUM_WORKERS = 4
LEARNING_RATE = 1e-5
MOMENTUM = 0.9
WEIGHT_DECAY = 0.01
BETAS=(0.9, 0.999)
ETA_MIN = 1e-7

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL, CLIP_PREPROCESS = clip.load("ViT-B/32", device=DEVICE)
CNN_PREPROCESS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

