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
- 数据集适配在 `src/gazelab/datasets.py`
- 训练 / 测试 / 推理 / 预处理各自独立入口在 `scripts/`
- 支持多实验目录 `experiments/`

---

## 目录结构

```
gazelab/
├── configs/
│   └── config.py           # 全局配置（数据集/模型/超参）
├── src/gazelab/            # 核心包
│   ├── datasets.py         # 数据集适配层（Gaze360 / ETH-XGaze / MPIIFaceGaze / EyeDiap）
│   ├── predict.py          # 单图推理 + 箭头可视化（CLI）
│   ├── models/
│   │   ├── gazeformer.py   # 主模型：CLIP 语义融合（GEWithCLIPModel / _zhao）
│   │   ├── transformers.py # MoE Transformer：特征 -> 3D 视线
│   │   └── clipmodel.py    # CLIP 相关
│   └── utils/
│       ├── common.py       # 通用工具（leave_one_out 等）
│       └── loggers.py      # 日志（TensorBoard / wandb）
├── tools/
│   └── data/               # 数据处理脚本
│       ├── preprocess.py   # 数据预处理（Gaze360 -> GazeHub 标准格式）
│       └── verify.py       # 数据校验
├── scripts/
│   ├── train.py            # 训练入口
│   └── train_test.py       # 测试入口
├── assets/                 # 推理用检测模型（mediapipe / Haar）
├── experiments/            # （预留）多模型 / 多实验
├── tests/                  # （预留）测试
├── docs/                   # 文档
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

> `configs/config.py` 中的数据路径已基于项目根目录自动推导，
> 训练/推理脚本也带路径引导，因此**可以从任意目录运行**以下命令。
> 若用 `python -m gazelab.predict`，需先 `pip install -e .` 或设 `PYTHONPATH=src`。

### 单图推理（预测视线 + 画箭头）

```bash
python -m gazelab.predict --image 你的图片.jpg
# 或
conda run -n dl python -m gazelab.predict --image 你的图片.jpg \
    --checkpoint checkpoints/best—separate-added_Gaze360.pt \
    --out result.png
```

输出：原图 + 从双眼中心指向视线方向的红色箭头，另打印 3D 视线向量、yaw/pitch。

### 数据预处理

```bash
python tools/data/preprocess.py --input-dir ./gaze360 --output-dir ./data/Gaze360
```

### 训练

```bash
python scripts/train.py
```

### 测试

```bash
python scripts/train_test.py
```

---

## 模型架构

Gazelab 的 gaze 估计分两个阶段：

1. **特征提取（`models/gazeformer.py`）**
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

## 扩展指南

### 新增模型

在 `src/gazelab/models/` 新建一个文件，实现 `nn.Module`，并在入口脚本中替换 import 即可。

### 新增数据集

在 `src/gazelab/datasets.py` 新增一个 `Dataset` 子类（参考已有实现），
并在 `configs/config.py` 中设置 `TRAIN_DATASET_NAME` / `TEST_DATASET_NAME`。

### 消融实验

通过 `configs/config.py` 的 `ABLA_CONFIG` 开关各特征流（`use_feature_1` ~ `use_feature_4`）。

---

## 目录约定

| 目录 | 用途 |
|---|---|
| `data/` | 数据集（GazeHub 格式，gitignore 忽略） |
| `checkpoints/` | 模型权重（gitignore 忽略） |
| `log/` | 训练日志 |
| `assets/` | 推理检测模型（需提交） |
| `experiments/` | 多实验 / 多模型结果 |
| `docs/` | 项目文档 |

---

## License

MIT