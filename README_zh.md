# GazeFormer：基于 CLIP 与 MoE Transformer 的上下文感知视线估计

> 一个用于 3D 视线估计的 PyTorch 框架，通过混合专家（MoE）Transformer 将 CLIP 的语义先验与视觉特征相融合。

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

本仓库包含 **GazeFormer** 的官方实现。

## 核心特性

- **文本条件特征融合：** 利用针对光照、头部姿态、视线方向等属性的文本提示，动态引导与视线相关的特征提取。
- **多源 Token 聚合：** 采用混合专家（MoE）Transformer 架构，融合来自 CLIP 的语义嵌入、CNN 骨干网络的空间特征以及原始图像块（patch）Token。
- **跨数据集泛化能力：** 针对多个标准视线数据集（Gaze360、ETH-XGaze、MPIIFaceGaze、EyeDiap）设计并验证了鲁棒性，只需极少的改动即可迁移。
- **便于消融实验：** 通过集中的 `ABLA_CONFIG` 配置即可轻松开启或关闭不同的特征流（`feature_1` 至 `feature_4`），便于进行可控实验。

## 架构概览

GazeFormer 通过三条并行流处理人脸图像，随后将其 Token 化并送入 Transformer，完成最终的 3D 视线回归。

1.  **CLIP 语义流：** 使用冻结的 CLIP 模型对输入图像和一组文本提示进行编码。通过余弦相似度选出最相关的文本属性嵌入（例如 "一张光线明亮的脸"、"一张望向左侧的脸"），并与图像嵌入相融合，从而生成与任务对齐且经过上下文补偿的特征。
2.  **CNN 视觉流：** 使用标准 CNN 骨干网络（如 ResNet-50）提取丰富的空间特征图，提供强大的视觉几何先验。
3.  **融合 Transformer：** 将上述各流的输出投影到统一的 Token 空间。由混合专家（MoE）层增强的 Transformer 对这些 Token 进行聚合，预测最终的 3D 视线向量。

## 环境搭建

### 1. 环境要求

- Python 3.10+
- PyTorch 1.2.1+
- CUDA 11.3+

### 2. 安装

克隆本仓库并安装所需依赖：

```bash
git clone https://github.com/your-username/Gazeformer_submission.git
cd Gazeformer_submission
pip install -r requirements.txt
```

`requirements.txt` 应包含以下内容：

```
torch
torchvision
timm
easydict
ftfy
regex
opencv-python
numpy
tqdm
wandb
git+https://github.com/openai/CLIP.git
```

### 3. 数据集

下载所需数据集，并按照 GazeHub 的约定进行组织：

```
datasets/
├── Gaze360/
│   └── GazeHub/
│       ├── Image/
│       └── Label/
├── ETH-XGaze/
│   └── GazeHub/
│       ├── Image/
│       └── Label/
...
```

如果你的目录结构不同，请相应修改 `config.py` 中的路径。

## 使用方法

### 训练

主训练脚本 `train.py` 负责数据集加载、模型初始化以及训练循环。

运行以下命令开始训练：

```bash
python train.py
```

- **配置：** 修改 `config.py` 以设置超参数、选择数据集（`TRAIN_DATASET_NAME`、`TEST_DATASET_NAME`）以及选择 CNN 骨干网络（`CNN_MODEL`）。
- **消融实验：** 通过编辑 `config.py` 中的 `ABLA_CONFIG` 字典来启用或禁用各特征流。
- **日志记录：** 训练过程和验证结果会记录到 `log/` 目录，并可通过 TensorBoard 进行监控。

### 评估

训练过程中会定期在验证集上对模型进行评估。如需单独进行评估，通常是加载一个检查点（checkpoint）并运行测试循环。
