# Gazelab

> 3D 视线估计（Gaze Estimation）研究框架：融合 CLIP 语义先验、CNN 视觉特征与 MoE Transformer。

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 项目简介

Gazelab 是一个**模块化、可扩展**的视线估计研究框架。核心思想是用 CLIP 的图文语义能力，
为一张人脸补充"光照 / 头姿 / 背景 / 视线方向"等属性先验，再与 CNN 提取的视觉特征融合，
经 MoE Transformer 回归 3D 视线方向。

设计目标是让"换模型、换数据集、加新功能"都尽可能低成本：
- 模型放在 `src/gazelab/models/`，新增一个文件即可接入
- 数据集适配在 `src/gazelab/datasets/`（`gaze360_dataset.py` / `rgbdgaze.py`）
- 训练 / 测试 / 推理 / 预处理各自独立入口在 `scripts/`
- 支持多实验目录 `experiments/`

---

## 目录结构

```
gazelab/
├── configs/                 # 配置（按数据集分）
│   ├── gaze360_config.py    # Gaze360 配置（参考/遗留）
│   └── rgbdgaze_config.py   # RGBDGaze 配置（主推）
├── src/gazelab/             # 核心包
│   ├── datasets/            # 数据集
│   │   ├── gaze360_dataset.py
│   │   └── rgbdgaze.py      # RGBDGaze：RGB+depth 双流，返回 (edict(rgb,depth), gaze)
│   ├── models/              # 模型（按工作流分子包）
│   │   ├── gaze360/         # Gaze360 主模型（参考/遗留）
│   │   │   ├── gaze360_gazeformer.py
│   │   │   └── clipmodel.py
│   │   ├── rgbdgaze/        # RGBDGaze 模型家族（主推）
│   │   │   ├── rgbdgaze.py            # 旧版基线（CLIP+ResNet）
│   │   │   └── rgbdgaze_dinov2.py     # 新版（DINOv2+逆深度+BlockMoba）
│   │   └── transformers.py  # 共享融合块（BlockMoba / MoE / Transformer）
│   └── utils/
│       ├── common.py
│       └── loggers.py
├── tools/
│   └── data/
│       ├── gaze360/         # preprocess.py / verify.py（参考）
│       └── rgbdgaze/        # preprocess.py（建索引+划分）
├── scripts/                 # 训练/推理入口（按工作流分子包）
│   ├── gaze360/             # （参考/遗留）train.py / train_test.py / predict.py
│   └── rgbdgaze/            # （主推）
│       ├── train.py             # DINOv2 新模型训练（主入口）
│       ├── train_baseline.py    # 旧基线训练
│       └── predict.py           # 单样本推理+可视化
├── out/                     # 训练产物（日志/权重，gitignore 忽略）
│   └── {dataset}/{task}/{logs,checkpoints,runs}
├── assets/                 # 推理用检测模型（mediapipe / Haar）
├── docs/                   # 文档
├── model/                  # 预训练权重（如 DINOv2，gitignore 忽略）
├── pyproject.toml          # 包定义
└── requirements.txt        # 依赖
```

---

## 安装

### 1. 环境

- Python 3.10+
- PyTorch（CUDA 版）
- 建议使用 conda

### 2. 安装依赖

```bash
pip install -e .                    # 安装 gazelab 包（editable）
pip install -r requirements.txt     # 依赖（torch/clip/scipy/timm 等）
pip install mediapipe               # 推理可选：人脸关键点检测
```

---

## 快速开始

> `configs/gaze360_config.py` 中的数据路径已基于项目根目录自动推导，
> 训练/推理脚本也带路径引导，因此**可以从任意目录运行**以下命令。
> 若用 `python -m gazelab.predict`，需先 `pip install -e .` 或设 `PYTHONPATH=src`。

### 单图推理（预测视线 + 画箭头）

```bash
python -m gazelab.predict --image 你的图片.jpg
# 或
conda run -n dl python -m gazelab.predict --image 你的图片.jpg \
    --checkpoint out/gaze360/train/checkpoints/best—separate-added_Gaze360.pt \
    --out result.png
```

输出：原图 + 从双眼中心指向视线方向的红色箭头，另打印 3D 视线向量、yaw/pitch。

### 数据预处理（Gaze360）

```bash
python tools/data/preprocess.py --input-dir ./gaze360 --output-dir ./data/Gaze360
```

### 训练 Gaze360

```bash
python scripts/train_gaze360.py
```

### 训练 RGBDGaze

```bash
# 1. 先建索引（sample=随机 / subject=按人 / activity=按活动）
conda run -n dl python -m tools.data.rgbdgaze.preprocess --split sample
# 2a. 新模型（DINOv2 + 逆深度 + BlockMoba，推荐）
conda run -n dl python scripts/rgbdgaze/train.py --epochs 30 --batch-size 48
# 2b. 旧基线（CLIP + ResNet）
conda run -n dl python scripts/rgbdgaze/train_baseline.py --epochs 30 --batch-size 64 --lr 1e-4
```

### 测试

```bash
python scripts/train_test.py
```

---

## 模型架构

Gazelab 的 gaze 估计分两个阶段：

1. **特征提取（`models/gaze360_gazeformer.py`）**
   - 冻结的 CLIP 编码人脸图像与一组文本提示（光照 / 头姿 / 背景 / 视线方向）
   - 用余弦相似度选出最匹配的属性向量，与图像特征融合
   - 得到 `feature_1`（环境：光照+头姿+背景）与 `feature_2`（视线方向）
   - CNN（如 ResNet-50）提取局部特征图 `feature_3`

2. **回归（`models/transformers.py`）**
   - 各特征投影成 token 序列，前置 CLS token
   - 多层 DeepSeek 风格 Block（自注意力 + 交叉注意力 + MoE）
   - 取 CLS token 经线性头回归 3D 视线方向

训练用 `angular_loss`（角度差）为主，可选 `feature_separation_loss` 辅助。

---

## 训练产物目录规范

所有日志、权重、tensorboard 统一输出到 `out/{dataset}/{task}/` 下，由
`configs/gaze360_config.py` 的 `experiment_dirs()` 生成各数据集一致结构：

```
out/{dataset}/{task}/
├── logs/           # 训练/测试日志（含时间戳）
├── checkpoints/    # 模型权重
└── runs/           # tensorboard events
```

例如：
- Gaze360 训练：日志与权重在 `out/gaze360/train/{logs,checkpoints}`
- RGBDGaze 训练：在 `out/rgbdgaze/train/{logs,checkpoints}`

各训练脚本通过 `experiment_dirs(dataset, task)` 获取这三类路径，避免硬编码相对 CWD。

---

## 扩展指南

### 新增模型

在 `src/gazelab/models/` 新建一个文件，实现 `nn.Module`，并在入口脚本中替换 import 即可。

### 新增数据集

在 `src/gazelab/datasets/` 新建一个数据集文件（参考 `gaze360_dataset.py` / `rgbdgaze.py`），
在 `src/gazelab/datasets/__init__.py` 的 re-export 层暴露，
并在对应 `configs/*_config.py` 中设置路径与超参。

### 消融实验

通过 `configs/gaze360_config.py` 的 `ABLA_CONFIG` 开关各特征流（`use_feature_1` ~ `use_feature_4`）。

---

## 目录约定

| 目录 | 用途 |
|---|---|
| `data/` | 数据集（原始 + GazeHub 格式，gitignore 忽略） |
| `out/{dataset}/{task}/logs` | 训练/测试日志 |
| `out/{dataset}/{task}/checkpoints` | 模型权重 |
| `out/{dataset}/{task}/runs` | tensorboard events |
| `configs/` | 各数据集的配置 |
| `assets/` | 推理检测模型（需提交） |
| `experiments/` | 多实验 / 多模型结果 |
| `docs/` | 项目文档 |

---

## License

MIT