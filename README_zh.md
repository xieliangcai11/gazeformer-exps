# Gazelab

注视（gaze）估计研究仓库：以 **DINOv2 视觉骨干 + 屏幕几何 token** 为核心，做 **2D 屏幕注视点回归**。

两条使用线：

| 数据集 | 用途 | 规模 |
|---|---|---|
| **GazeCapture** | 大规模"注视先验"预训练 | ~149 万帧 / 1474 名被试 |
| **RGBDGaze** | 域内训练、校准、测试时自适应 | ~14 万帧 / 45 名被试 |

两者共用同一条训练循环、同一套模型与评估口径（`cm` 与角度），差异只在数据集构造与索引路径。

---

## 目录结构

```
gazeformer-exps/
├── configs/                     # 只放配置（零副作用）
│   ├── paths.py                 #   项目根 / DEVICE / out 目录约定
│   ├── gazecapture_config.py    #   GazeCapture 路径、几何域随机化范围等
│   └── rgbdgaze_config.py       #   RGBDGaze 路径、裁剪/增强/视距等
├── src/gazelab/
│   ├── datasets/
│   │   ├── common.py            #   裁剪 / 归一化 / 屏幕几何 / 几何随机化
│   │   ├── table_index.py       #   列式 CSV 索引（避免 2.4GB/进程的 dict 列表）
│   │   ├── gazecapture.py       #   GazeCapture 数据集（可选眼支流）
│   │   └── rgbdgaze.py          #   RGBDGaze 数据集（可选眼支流 / 朝向兼容）
│   ├── models/
│   │   ├── dinov2_gaze/         #   **数据集无关**的注视模型家族
│   │   │   ├── rgbdgaze_dinov2.py  # 主模型：DINOv2 + 几何/IMU/眼 token + BlockMoba
│   │   │   ├── calibrator.py       # 校准器（特征空间偏移 / 多项式）
│   │   │   ├── tta.py              # 测试时自适应
│   │   │   └── vpt_model.py        # Visual Prompt Tuning 适配
│   │   └── transformers.py      #   BlockMoba / RMSNorm 等基础块
│   └── utils/                   #   通用工具
├── scripts/
│   ├── train.py                 # **统一训练入口**：--dataset rgbdgaze|gazecapture
│   ├── cross_eval.py            # 跨数据集/跨几何评估、划分导出
│   ├── analysis/                # 一次性分析脚本（校准、误差预算、可视化等）
│   ├── rgbdgaze/                # RGBDGaze 专用：校准 / TTA / 坏例筛查
│   └── gazecapture/             # GazeCapture 专用：训练队列脚本
├── tools/data/
│   ├── common/eye_detect.py     # 人脸/眼检测（Haar + MediaPipe，共享）
│   ├── rgbdgaze/                # preprocess / build_eye_boxes / build_blink_scores
│   └── gazecapture/             # preprocess / 眼框生成·体检·修复 / 数据验证
├── assets/                      # 检测模型（face_landmarker、Haar）
├── docs/                        # 论文与实验记录
├── tests/ experiments/          # 预留
├── data/ out/ model/ temp/      # 数据与产物（gitignore）
├── pyproject.toml requirements.txt
```

---

## 安装

```bash
conda create -n dl python=3.11 -y && conda activate dl
pip install -r requirements.txt
pip install mediapipe     # 眼框生成/数据质检可选
```

DINOv2 权重放在 `model/dinov2_vits14_pretrain.pth`（不入库）。

---

## 数据准备

```bash
# GazeCapture：解压 → 扫描元数据 → index.csv + train/val/test.txt
python -m tools.data.gazecapture.preprocess

# RGBDGaze：构建索引与划分
python -m tools.data.rgbdgaze.preprocess --split subject

# 眼框表（训练眼支流时才需要）
python -m tools.data.gazecapture.build_eye_boxes_mp --shard 0 --nshards 8   # 分片并行
python -m tools.data.gazecapture.build_eye_boxes_mp --merge \
    data/GazeCapture/index/eye_boxes_shard*.npz --out data/GazeCapture/index/eye_boxes.npz
python -m tools.data.gazecapture.check_shards --nshards 8        # 完整性体检
python -m tools.data.gazecapture.fix_eye_boxes --dry-run         # 退化框统计
python -m tools.data.gazecapture.fix_eye_boxes                    # 同被试中位兜底 + filled 标记
```

### 数据验证

```bash
python -m tools.data.gazecapture.verify                 # 结构/标签/抽样图像/眼框对齐
python -m tools.data.gazecapture.verify --recheck 50    # 额外用 MediaPipe 现场重检并比 IoU
```

本地若只有完整数据集的**子集**，缺失清单/图像默认记为 `WARN`（用 `--strict-data` 切到严格模式）。

---

## 训练

统一入口 `scripts/train.py`，`--dataset` 选择数据集：

```bash
# 单卡
python -m scripts.train --dataset rgbdgaze --resolution 448 --unfreeze 12 --diff-lr \
    --lr 1e-4 --batch-size 24 --epochs 8 --tag g6_nodepth

# 多卡（GazeCapture 预训练示例）
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 torchrun --standalone --nproc_per_node=6 \
  -m scripts.train --dataset gazecapture --resolution 224 --no-geom --geom-randomize 0 \
  --augment --aug-shift 0.05 --aug-scale 0.05 --aug-rotate 0 --unfreeze 12 --diff-lr \
  --lr 2e-4 --batch-size 32 --num-workers 8 --epochs 5 --seed 0 --tag gc_run1

# 带眼支流（需先准备 eye_boxes.npz）
python -m scripts.train --dataset gazecapture --use-eye --eye-scale 2.0 ...
```

产物：`out/<dataset>/ablation/checkpoints/<tag>/{best.pt,train_args.json}` 与 `logs/`。
`train_args.json` 记录完整参数，可直接复现。

### 评估

```bash
python -m scripts.cross_eval --ckpt out/gazecapture/ablation/checkpoints/gc_run1/best.pt \
    --dataset gazecapture --image-size 224 --use-geom 0
```

---

## 关键约定

- **坐标**：标签为屏幕像素点，数据集内归一化到 `[0,1]`；眼框为 `(x,y,w,h)` 原图坐标。
- **几何 token**：`[短边/长边, log(短边cm), log(长边cm)]`（横竖屏同一语义）；`--geom-randomize` 把标签与几何一起重投影到合成屏，并把 `screen_cm` 同步替换；尺寸按标签可行性反推，生效率≈100%。
- **增强可复现**：逐样本种子 `(aug_seed, epoch, idx)`，与 worker 数无关；DataLoader 每 epoch 重建 worker（`persistent_workers=False`），保证 `set_epoch` 生效。
- **眼框质量**：MediaPipe 在侧脸/遮挡下会产生退化框（<8px，实测约 1.8%）。`fix_eye_boxes` 用**同被试中位框**替换并置 `filled=True`；**不做镜像**（镜像会把瞳孔水平翻转）。

---

## 已知限制

- `--loss mse_angle` 的角度项以"度"为单位，与归一化 MSE（~1e-4）不同量级；如需两者同量级请设 `--angle-weight 0.01~0.05`。
- 部分分析脚本（`scripts/analysis/`）是一次性实验代码，接口未做稳定化承诺。

## License

MIT
