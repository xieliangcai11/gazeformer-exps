# Gaze360 训练管线说明

本文基于对仓库代码的只读分析，说明 `train.py`、`train_test.py` 两个脚本的职责、数据集与 label 的含义，以及训练日志的读法。涉及行号均对应当前工作副本。

---

## 一、两个脚本的定位

结论：**Gaze360 训练只需要跑 `train.py`，`train_test.py` 不必运行。**

| | `train.py` | `train_test.py` |
|---|---|---|
| 角色 | Gaze360 实际训练脚本 | 独立实验/冒烟脚本 |
| 数据 | 真实 `train/val/test.label` | `random_loader` 假数据（`torch.rand`） |
| 模型 | `GEWithCLIPModel_zhao`（部分冻结） + `TransformerDeepSeek_gaze` | `GEWithCLIPModel` 单模型 |
| gaze 损失 | 角度损失 `angular_loss` | `L1Loss` 回归 3D 向量 |
| 混合精度 | `autocast` + `GradScaler` | 无 |
| LR 调度 | `CosineAnnealingLR` | 无 |
| 日志 | `./log/{ts}_Gaze360-Gaze360_log.txt` | `./logs/ResNet-50_egaze360_test/test.log` |
| checkpoint | `./checkpoints/best—separate-added_Gaze360.pt` | `./checkpoints/ResNet-50_gaze360_train/` |

两套产物互相不兼容：checkpoint 命名、状态字典结构（前者只存 `transformer_model.state_dict()`）均不同，且入口模型类不同。

---

## 二、`train.py`

### 1. 两阶段模型

**阶段一：`GEWithCLIPModel_zhao` —— 特征工厂（部分冻结）**

`train.py:240-243` 只冻结 CLIP 的两部分：

```python
for param in model.model.visual.parameters():        # CLIP ViT
    param.requires_grad = False
for param in model.model.transformer.parameters():   # CLIP Transformer
    param.requires_grad = False
```

`model.main_model`（ResNet-50 CNN）与 `model.fuse_model` **未冻结，参与训练**，因此优化器包含它们：

```python
optimizer = torch.optim.AdamW(
    list(transformer_model.parameters()) +
    list(model.fuse_model.parameters()) +
    list(model.main_model.parameters()),
    lr=LEARNING_RATE,
)
```

**阶段二：`TransformerDeepSeek_gaze` —— gaze 回归头**

每步训练先跑 `process_batch(batch, model=model)`（`train.py:66-142`）产出四个特征：

| 特征 | 形状 | 来源 |
|---|---|---|
| `feature_1` | `[B, 512]` | `img_feats + 选中的光照/头姿/背景属性向量`，上下文补偿 |
| `feature_2` | `[B, 512]` | `img_feats + 选中的视线方向标签向量`，任务对齐 |
| `feature_3` | `[B, N, 2048]` | CNN `main_model` 特征图铺平成 token 序列 |
| `token_img_patch` | `[B, P, d]` | ViT 中间层 token（forward hook 抓取，去掉 CLS） |

四路特征经 `TransformerDeepSeek_gaze.forward`（`transformer_models.py:1059`）：

```
feature_1/2  → proj_f1/proj_f2 → unsqueeze(1) → [B,1,d_model]
feature_3    → proj_f3                      → [B,N,d_model]
token_patch  → proj_patch                   → [B,P,d_model]
→ cat 拼序列 → 前置 CLS token → N 层 BlockMoba → 取 x[:,0,:] → linear_head → [B,3]
```

`BlockMoba` = 自注意力 + 交叉注意力 + DeepSeek 风格 MoE（gate + routed expert + shared expert）。

### 2. 损失

```python
loss = loss_gaze + lambda_sep * loss_sep     # lambda_sep = 1.0 硬编码（train.py:295）
```

- `angular_loss`：预测与 GT 归一化后 `acos(cos_sim).mean()`，**弧度单位**
- `feature_separation_loss`：推动 `feature_1` 与 `feature_2` 相似，与 gaze 精度无直接因果

总 loss 中角度项往往只占约 1/4，优化器主要梯度信号花在辅助项上。

### 3. 每轮循环（`train.py:285-353`）

```
for epoch in range(NUM_EPOCHS):
    1. for i, batch in enumerate(train_dl):            # 287  训练
       process_batch → transformer_model → loss → scaler.step
       if i % 50 == 0: print + write_log                # 304
    2. transformer_model.eval(); model.eval()
       for batch in test_dl:                            # 323  用测试集评估
       → mean_test_angle
    3. if not is_ablation and mean_test_angle < best_angle:   # 346
           torch.save({...}, "checkpoints/best—separate-added_Gaze360.pt")
    4. scheduler.step()                                 # 353
```

### 4. 产物

| 产物 | 路径 |
|---|---|
| 训练日志（追加写） | `./log/{时间戳}_Gaze360-Gaze360_log.txt` |
| 最佳模型 | `./checkpoints/best—separate-added_Gaze360.pt`，dict 含 `epoch` / `model_state_dict`（`GEWithCLIPModel_zhao` 全套）/ `transformer_state_dict` / `optimizer_state_dict` / `mean_test_angle` |
| ETH-XGaze 专用 dump | 仅在 `TRAIN_DATASET_NAME == "ETH-XGaze"` 时生成 pitch 预测 txt |

checkpoint 目录是 `os.makedirs("checkpoints")`（相对 CWD），**不是** `config.py` 的 `CHECKPOINTS_PATH`。

---

## 三、`train_test.py`

`IS_TRAIN = False`（`config.py:23`）时走 `train_test.py:374-378`：加载 `epoch_50.pth` 后只跑 `test()`；置 `True` 则进入训练分支。

三个不必运行的理由：

1. 数据源为 `random_loader`（`train_test.py:51-58`），`torch.rand` 假张量，不读磁盘。
2. 要求的 checkpoint `./checkpoints/ResNet-50_gaze360_train/epoch_50.pth` 由它自己训练时产出，`train.py` 产不出这个文件名。
3. 模型类不同（`GEWithCLIPModel` vs `_zhao`），状态字典结构不兼容。

已知小缺陷：`test()` 只写 `log_file`、不打印 stdout，终端无输出。

---

## 四、数据集与 label

### 原始数据

`gaze360/`：80 个录像段（`rec_000`~`rec_079`）+ `metadata.mat`，共 197588 行元数据。每行 = 一帧画面中一个人的标注：头部 bbox、脸部 bbox、左右眼 bbox、3D gaze 单位向量、person/frame/recording id、`split` 字段（0=train, 1=val, 2=test, 3=unused）。

### 预处理产物

由 `preprocess_gaze360.py` 生成，目录 `data/Gaze360/GazeHub/`：

```
Label/{train,val,test,unused}.label
Image/{train,val,test,unused}/{Face,Left,Right}/*.jpg
```

label 文件 = 表头 + 每行一样本，单空格分隔 6 字段：

```
Face                Left                  Right                 Origin                 3DGaze                      2DGaze
train/Face/7371.jpg train/Left/7371.jpg train/Right/7371.jpg rec_001/head/000000/001620.jpg 0.6258,-0.3859,-0.6778 0.7455,-0.3962
```

| 字段 | 含义 |
|---|---|
| `Face` | 人脸裁剪图路径（相对 `Image/`），224×224 |
| `Left` / `Right` | 眼部区域裁剪图，60×36；**两者内容相同**（同源 face bbox 裁剪，见下） |
| `Origin` | 原始 head 裁剪图路径，溯源用 |
| `3DGaze` | 单位方向向量，**训练目标** |
| `2DGaze` | 由 3D 转出的 yaw/pitch 弧度 |

> `Left` 与 `Right` 同源：预处理中 `left = crop_eye(...)` 后同时写入 Left 与 Right 两处。`person_eye_left_bbox` / `person_eye_right_bbox` 未被使用。

### 四个 split

| label | 样本数 | 磁盘图片数 | 含义 |
|---|---|---|---|
| `train.label` | 84902 | 84902 | 训练集（官方 split=0） |
| `val.label` | 11318 | 11318 | 验证集（官方 split=1） |
| `test.label` | 16031 | 16031 | 测试集（官方 split=2） |
| `unused.label` | 12580 | 12580 | 官方标为 unused 的样本（split=3），不参与训练/验证/测试 |

与 197588 的差额：无 face 标注的行 + 缺失/损坏的图片（预处理会跳过并计数）。

### 代码读取方式

```python
# train.py:184-186
train_label_file = TRAIN_LABELS_PATH / "train.label"
val_label_file   = TRAIN_LABELS_PATH / "val.label"
test_label_file  = TEST_LABELS_PATH   / "test.label"
```

三文件均存在 → `train.py:190-194` 各建一个 `ZhaoDataset`。`images_path` 是 `Image` 根目录（不含 split 子目录），因 label 首列自带 `train/`、`val/`、`test/` 前缀，直接拼接。

`ZhaoDataset` 内部把 `ds_name` 映射到 `gazehub_datasets.py` 的类；Gaze360 直接复用 `DatasetEyeDiapByGazeHub`（`gazehub_datasets.py:163` 为别名）。该类只取 label 第 0 列（Face 路径）与第 4 列（3DGaze）：

```python
edict(Face=label[0],
      _3DGaze=np.array(label[4].split(",")).astype("float") * __coefficients)
```

`__coefficients` 仅在跨数据集测试时翻转符号（如 Gaze360→MPIIFaceGaze），同数据集训练时为 `[1,1,1]`。

每个样本的 `face` 与 `other_face` 是**同一张图的两套预处理**（`gazehub_datasets.py:147-158`：`other_face_img = copy.deepcopy(face_img)`），不是左右眼。

### DataLoader（`BATCH_SIZE=64`）

```python
train_dl = DataLoader(train_ds, shuffle=True,  generator=gen)   # train.py:207
val_dl   = DataLoader(val_ds,   shuffle=False)                  # train.py:217
test_dl  = DataLoader(test_ds,  shuffle=False)                  # train.py:223
```

`gen = torch.Generator().manual_seed(SEED)`（`SEED=0`），训练顺序可复现；评估顺序固定。

| split | 样本 | batch 数 |
|---|---|---|
| train | 84902 | 1327（末批不足 64） |
| val | 11318 | 177 |
| test | 16031 | 251 |

---

## 五、训练日志解读

示例：

```
2026-09-22 21:33:55——Gaze360-Gaze360 Epoch 0 Step 2650, Loss: 1.182098, Angular Loss: 0.278315, Feature Loss: 0.903784, Mean Angular Error: 15.95°
2026-09-22 21:35:34——Gaze360 Epoch 0 Test Mean Angular Error: 20.54°
```

**第一条 = 训练期单 batch 快照**（`train.py:304` `if i % 50 == 0`），仅代表那 64 张图：

- `Loss` = `loss_gaze + 1.0 * loss_sep`；`0.278315 + 0.903784 = 1.182099` ✓
- `Angular Loss` 为**弧度**；`0.278 rad ≈ 15.95°`，与 `Mean Angular Error` 一致 ✓
- `Feature Loss` 无单位

**第二条 = 整个 epoch 结束后在 `test_dl` 上的平均角度误差**（`train.py:321-342`），覆盖 test.label 全部 16031 个样本。

15.95° vs 20.54° 的差异属正常：前者是单 batch 高方差快照，后者是 251 个 batch 的统计平均；Epoch 0 训练远未拟合。

---

## 六、已知问题

1. **验证集未使用，早停指标来自测试集。**
   `val_dl`（`train.py:210-221`）构建后全文再无引用；每轮评估与最优模型选择都用 `test_dl`（`train.py:323`、`346`）。测试集信息泄漏进模型选择，报告的 best 角度偏乐观。
   修法：`train.py:323` `for batch in test_dl` → `val_dl`。

2. **checkpoint 已含全部模型（修复后）。**
   当前 `train.py:343-349` 同时保存 `model_state_dict`（`GEWithCLIPModel_zhao` 全套：`main_model` + `fuse_model` + CLIP）与 `transformer_state_dict`，加载时按 key 取即可完整恢复两个模型。注意旧版 checkpoint（只有 `transformer_model`）无法直接加载到新结构。

3. **`Left`/`Right` 图片相同**，两路输入未提供独立信息。

4. **`lambda_sep = 1.0` 硬编码**（`train.py:295`），特征分离损失主导梯度；可通过调小该值对比若干 epoch 的 `Test Mean Angular Error` 验证影响。

5. **`Step` 序号与 batch 数可能不符。** 若日志出现 `Step` 大于 train batch 数（如 1326），说明数据集走了 `train.py:205-215` 的 `random_split` 回退分支，或日志跨 epoch 累计。启动时打印的 `Using explicit train/val/test labels: ...` / `Fallback split from train.label: ...` 可确认实际划分。

6. **`train_test.py` 的 `test()` 不向终端打印结果**，只写日志文件。

---

## 七、常用路径速查

| 内容 | 路径 |
|---|---|
| 原始数据 | `./gaze360/`（`imgs/` + `metadata.mat`） |
| 标准格式数据 | `./data/Gaze360/GazeHub/{Label,Image}` |
| 训练日志 | `./log/{时间戳}_Gaze360-Gaze360_log.txt` |
| 最佳模型 | `./checkpoints/best—separate-added_Gaze360.pt` |
| 模型定义 | `models.py`（`GEWithCLIPModel_zhao`）、`transformer_models.py`（`TransformerDeepSeek_gaze`） |
| 超参 | `config.py`（`BATCH_SIZE=64`、`NUM_EPOCHS=50`、`SEED=0`） |
