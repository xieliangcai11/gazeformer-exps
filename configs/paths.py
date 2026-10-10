"""项目路径与输出目录约定（不依赖 CLIP / torch 之外的任何东西）。

为什么单独成文件：原先这些定义在 configs/gaze360_config.py 里，而该模块在
**导入时**就执行 `clip.load("ViT-B/32", device="cuda")`。于是任何只想拿
`DEVICE` / `experiment_dirs` 的脚本（如 RGBDGaze / GazeCapture 的训练脚本）
都会顺带：
  1) 把 CLIP 权重加载到 **cuda:0**（此时还没执行 torch.cuda.set_device(local_rank)，
     所以 DEVICE="cuda" 恒指 0 号卡）；
  2) 在 torchrun 多卡下每个 rank 各加载一份 -> GPU0 上白占 N x ~300MB，
     且 N 个进程并发读同一个 checkpoint 文件。
把路径类定义搬到这里后，训练脚本不再触碰 CLIP。

configs/gaze360_config.py 会 re-export 本模块的名字，保持既有调用
（`from configs.gaze360_config import *`）不变。
"""

from pathlib import Path

import torch

# 项目根目录 = 本文件(configs/paths.py)的上一级；使路径不依赖运行 CWD
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: 公开的项目根
PROJECT_ROOT = _PROJECT_ROOT

#: 数据根目录（GazeHub 结构：<data>/<数据集名>/GazeHub/{Image,Label}）
DATASETS_PATH = _PROJECT_ROOT / "data"

#: 输出根目录
OUT_PATH = _PROJECT_ROOT / "out"
CHECKPOINTS_PATH = OUT_PATH / "checkpoints"
LOGS_PATH = OUT_PATH / "logs"
RUNS_PATH = OUT_PATH / "runs"

#: 默认设备（与 configs/gaze360_config.py 中同名变量保持一致）
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def experiment_dirs(dataset: str, task: str) -> dict:
    """返回按数据集+任务分层的输出目录。

    规范：out/<dataset>/<task>/{logs, checkpoints, runs}
      - out/gaze360/train/{logs, checkpoints, runs}
      - out/rgbdgaze/train/{logs, checkpoints, runs}
      - out/gazecapture/ablation/{logs, checkpoints, runs}

    各脚本请用此函数生成统一的输出路径，勿再硬编码相对 CWD 的 'log/' 等。
    """
    base = OUT_PATH / dataset / task
    return {
        "root": base,
        "log": base / "logs",
        "checkpoint": base / "checkpoints",
        "run": base / "runs",
    }
