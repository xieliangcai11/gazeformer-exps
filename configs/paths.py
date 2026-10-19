"""项目路径与输出目录约定（只依赖 torch，不触碰任何模型/数据集）。

只放"路径 + 设备 + 输出目录"这类零副作用定义，供所有脚本共用；
数据集超参放在 configs/<dataset>_config.py，模型权重路径放在各模型模块内。
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

#: 默认设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def experiment_dirs(dataset: str, task: str) -> dict:
    """返回按数据集+任务分层的输出目录。

    规范：out/<dataset>/<task>/{logs, checkpoints, runs}
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
