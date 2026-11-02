# Gazelab: 2D Screen Gaze Estimation

面向手机前置摄像头的 2D 屏幕注视点估计。核心链路只有一条：

```text
数据索引 → 人脸/眼部裁剪 → DINOv2 + BlockMoba → cm 坐标
```

支持 GazeCapture（GC）预训练和 RGBDGaze（RGD）微调。模型输出**物理厘米**坐标，
屏幕几何在输入侧显式提供；校准方法作为可选的轻量后处理保留。

---

## 1. 方法

### 输入

- 整脸裁剪：224×224
- 左眼裁剪：224×224
- 右眼裁剪：224×224
- 屏幕几何：7 维
  ```text
  [短边/长边, log(短边cm), log(长边cm), orientation one-hot ×4]
  ```

### 骨干

- DINOv2 ViT-S/14；patch embedding、position embedding、final norm 冻结；
- Transformer block 可全部或部分解冻；
- 人脸 CLS + patch tokens、双眼 CLS、几何 token 拼接；
- 6 层 BlockMoba（d_model=384，MoE/FFN）；
- RMSNorm + 线性头 → 2D cm。

### 输出

```text
(X, Y)_cm
```

- 原点：屏幕中心；
- 坐标轴：竖屏标准方向；
- 归一化标签通过 IMU 角度旋转到 cm。

### 训练

- 损失：归一化 MSE + 0.01 × cm Huber；
- 阶段 A：冻结 DINOv2，训练融合层和输出头；
- 阶段 B：解冻 Transformer，backbone 使用较低学习率；
- 支持 DDP、断点续训、验证集 best checkpoint、早停。

---

## 2. 目录

```text
configs/                      数据集路径与超参
src/gazelab/datasets/         GazeCapture / RGBDGaze 数据读取
src/gazelab/models/           DINOv2 + BlockMoba 模型
src/gazelab/utils/            通用工具
scripts/train.py              GC / RGD 主训练入口
scripts/calibration/
  cm_infer.py                 模型评估（norm / cm / 分设备）
  cm_train_epoch.py           cm 空间续训
  rgd_cm_ft_v2.py             GC → RGD 微调
  dump_feats.py               导出特征与预测
  pose_features.py            位姿代理特征
  seq_bayes_calib.py          最终校准方法
  robust_calib.py             Huber-B2 对照
  cc_eb.py                    覆盖选择 + ridge prior
tools/data/                   数据预处理与索引构建
tests/                        核心单元测试
docs/papers/                  参考论文
```

---

## 3. 安装

```bash
pip install -e .
```

可选：

```bash
pip install mediapipe
```

---

## 4. 数据

### GazeCapture

```bash
python tools/data/gazecapture/preprocess.py \
  --archive-dir data/GazeCapture \
  --extract-dir data/GazeCapture_extracted \
  --index-dir data/GazeCapture/index

python tools/data/gazecapture/build_eye_boxes_mp.py \
  --index-dir data/GazeCapture/index
```

### RGBDGaze

```bash
python tools/data/rgbdgaze/preprocess.py \
  --index-dir data/RGBDGaze/index

python tools/data/rgbdgaze/build_eye_boxes.py \
  --index-dir data/RGBDGaze/index

python tools/data/rgbdgaze/add_imu_to_index.py \
  --index-dir data/RGBDGaze/index
```

---

## 5. 训练

完整命令见 `TRAINING_COMMANDS.md`。最小示例：

### GazeCapture

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 -m scripts.train \
  --dataset gazecapture \
  --index-dir data/GazeCapture/index \
  --resolution 224 \
  --use-eye \
  --geom --geom-orient --geom-randomize 0 \
  --augment --aug-shift 0.05 --aug-scale 0.05 --aug-rotate 0 \
  --unfreeze 12 --diff-lr --lr 1.4e-4 \
  --batch-size 32 --num-workers 4 \
  --epochs 6 --seed 0 \
  --tag gc_cm
```

### RGBDGaze 微调

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
torchrun --standalone --nproc_per_node=4 -m scripts.calibration.rgd_cm_ft_v2 \
  --ckpt out/gazecapture/.../best.pt \
  --index-csv data/RGBDGaze/index/index_imu.csv \
  --split-dir data/RGBDGaze/index/cv/fold3 \
  --epochs 8 --head-epochs 1 \
  --batch 16 --accum 2 \
  --head-lr 2e-4 --lr 5e-5 \
  --backbone-lr-ratio 0.1 \
  --augment --val-frac 0.15 \
  --patience 3 --distributed
```

---

## 6. 评估

```bash
python -m scripts.calibration.cm_infer \
  --ckpt out/.../best.pt \
  --index-dir data/GazeCapture/index \
  --split test --stride 10 \
  --out out/eval/pred.npz
```

输出：

- 归一化 L2；
- cm 误差；
- 分朝向/设备误差。

---

## 7. 校准（可选）

最终校准方法是按上下文拟合仿射修正：

```text
u = sx * x + bx
v = sy * y + by
```

流程：

1. `dump_feats.py` 导出基模型预测/特征；
2. `pose_features.py` 导出位姿代理特征；
3. `seq_bayes_calib.py` 做覆盖选择、ridge prior、Huber IRLS。

示例：

```bash
python -m scripts.calibration.seq_bayes_calib \
  --train-npz out/.../feats_train.npz \
  --val-npz out/.../feats_val.npz \
  --test-npz out/.../feats_test.npz \
  --index-csv data/GazeCapture/index/index.csv \
  --pose-npz out/.../pose_features.npz \
  --context-key orientation \
  --select-mode coverage_b2 \
  --prior-mode ridge --prior-lam 1.0 \
  --huber-delta 2.0 \
  --kmin 3 --kmax 9
```

校准只在少量目标域标签上拟合；评估标签不参与拟合。

---

## 8. 测试

```bash
python -m unittest tests.test_calibration -v
python -m compileall -q src scripts tools
```

---

## 9. 说明

- 数据、`out/`、`experiments/` 为本地生成物，不进入 git；
- 训练资源不超过服务器 75%，不使用 GPU 4；
- 归一化/cm 两种口径必须在同一 split 上比较，不能混用。
