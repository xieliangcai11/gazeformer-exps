# 项目状态与路线图（交接文档）

> 目的：记录当前项目的事实、结论、代码位置、实验口径和下一步计划，避免上下文丢失或重复踩坑。
> 最后更新：2026-10-31

---

## 1. 项目目标

用**一个模型 + 一套校准方法**，在手机前置摄像头场景下预测屏幕注视点，并在多个数据集/设备/朝向上保持一致的方法。

当前数据集：

- **GazeCapture（GC）**：约 125 万训练帧，1263 个训练被试，150 个测试被试；手机 + iPad。
- **RGBDGaze（RGD）**：约 16 万帧，45 个被试；坐/站/走/躺四种活动。

核心约束：

- 不是只追求单数据集精度，而是**方法统一、可泛化**；
- 校准尽量少标签；
- 训练支持多卡、断点续训、完整日志；
- 服务器最多使用 75% 资源；**不使用 GPU 4**。

---

## 2. 当前模型

模型类：`RGBDGazeDINOv2`  
文件：`src/gazelab/models/dinov2_gaze/rgbdgaze_dinov2.py`

### 2.1 输入

- 整脸裁剪：224×224；
- 左眼裁剪：224×224；
- 右眼裁剪：224×224；
- 几何 token：7 维
  ```text
  [aspect, log(物理宽cm), log(物理高cm),
   orientation one-hot ×4]
  ```
- 不使用 IMU token（当前版本）。

### 2.2 骨干

- DINOv2 ViT-S/14；
- `freeze_dino=True, unfreeze_last=12`；
- 12 个 Transformer block 全部可训练；
- patch embedding、position embedding、final norm 冻结；
- 参数：总 29.17M，全解冻训练时 trainable 28.84M；
- 阶段 A（冻结骨干）时 trainable 7.54M。

### 2.3 融合

- 人脸 CLS + patch tokens；
- 左眼 CLS、右眼 CLS；
- geom token；
- 汇总 CLS；
- 6 层 BlockMoba（d_model=384，MoE/FFN）；
- RMSNorm + 线性输出头 → 2D。

### 2.4 输出空间

当前最优版本输出**物理厘米**：

```text
(X, Y)_cm
```

定义：

- 原点：屏幕中心；
- 坐标轴：竖屏标准方向；
- 数据集里的归一化标签通过 IMU 角度旋转得到。

模型输入仍保留 orientation one-hot，因为：

> 目标坐标不变 ≠ 模型不需要视角信息。

---

## 3. 当前最优结果

### 3.1 旧归一化模型（可靠基线）

测试集 stride=10：

| orientation | cm | 归一化 |
|---|---|---|
1 | 1.691 | 0.187 |
2（iPad） | 2.195 | 0.126 |
3 | 1.473 | 0.181 |
4 | 1.466 | 0.182 |
总体 | ~1.66 | **0.1806** |

### 3.2 当前厘米模型（分阶段训练）

训练方式：

- 阶段 A：从旧归一化 checkpoint 初始化，重置输出头，冻结骨干，只训头/融合；
- 阶段 B：解冻 12 个 block，`lr=1e-4`，backbone 0.1×；
- 损失：归一化 MSE + 0.01 × cm Huber；
- geom 7 维，含 orientation；
- `geom_randomize=0`；
- 增强：shift/scale，不旋转。

验证最好：

```text
epoch 2，step 72000
val_norm = 0.1750
val_cm   = 1.5988
```

测试集（best.pt，stride=10）：

```text
normalized = 0.1856
cm         = 1.6219
```

分朝向：

| orientation | cm |
|---|---|
1 | 1.714 |
2（iPad） | 1.948 |
3 | 1.522 |
4 | 1.541 |

结论：

- 厘米指标优于旧模型；
- 归一化指标略差（0.1856 vs 0.1806）；
- iPad 明显改善；
- 手机三朝向误差接近。

### 3.3 校准迁移

物理空间，只在朝向 1 校准：

| 模型 | K | 迁移误差 |
|---|---|---|
旧归一化 | 52 | 1.7569 |
新厘米 | 52 | 1.7513 |
旧归一化 | 104 | 1.7356 |
新厘米 | 104 | 1.7427 |

结论：

> 模型精度改善，但“只在一个朝向校准”仍然不够；  
> 校准参数跨上下文不通用的问题没有自动解决。

---

## 4. 关键实验结论

### 4.1 旧模型原始朝向误差已经接近

旧归一化模型三个手机朝向原始 cm：1.69 / 1.47 / 1.47。  
所以“模型不会横屏”不是主要问题。

真正的问题是：

> **校准参数与用户/设备/姿态相关，不能跨上下文共享。**

### 4.2 偏差来源诊断（当前 cm 模型，GC 测试）

残差方差解释率：

| 上下文 | η² |
|---|---|
device | 1.44% |
orientation | 0.41% |
**subject（用户）** | **23.68%** |

结论：

> **用户是主要偏差来源；设备次之；朝向很小。**

### 4.3 face grid 诊断

- 与残差线性相关很弱（0.01–0.08）；
- 线性回归 R²：0.013 / 0.002；
- 但按人脸面积分箱：
  - 最小 25%：2.007 cm；
  - 最大 25%：1.347 cm。

结论：

> face grid 有非线性作用，主要影响小脸样本；  
> 整体提升预计有限，**暂不加入**，可作为后续消融。

### 4.4 相机位置

- 设备 η² 仅 1.44%；
- 相机偏移主要是固定平移；
- 依赖设备规格表，泛化性差。

结论：

> 低优先级，暂不加入。

### 4.5 几何随机化默认值（已修复）

`GazeCaptureDataset(augment=True)` 未显式传 `geom_randomize` 时默认：

```python
GEOM_RAND_PROB = 0.5
```

会导致一半训练样本的屏幕几何/标签被替换成合成设备。

已修复：训练显式 `geom_randomize=0`。

### 4.6 增强种子（已修复）

必须调用：

```python
tr.set_aug_seed(seed)
tr.set_epoch(epoch)
```

否则多 worker 增强重复、不可复现。

### 4.7 训练续训方式

- 正式训练使用 **epoch 级续训**；
- checkpoint 保存 `epoch`，从 `epoch+1` 开始；
- 不再按 batch 跳过数据；
- 每 epoch 结束保存 `epoch_i.pt` / `last.pt` / `best.pt`；
- mid-epoch 验证改善也会更新 `best.pt`。


### 4.8 按用户校准诊断（当前 cm 模型）

测试集 stride=10，标准厘米空间，每用户仿射/偏置校准：

| 方案 | K | cm | 归一化 |
|---|---|---|---|
无校准 | — | 1.6143 | 0.1846 |
用户 oracle（全拟合/全评估，乐观） | — | 1.4731 | 0.1675 |
每用户仿射 | 5 | 2.5489 | 0.2852 |
每用户仿射 | 10 | 1.8153 | 0.2039 |
每用户仿射 | 20 | 1.6353 | 0.1846 |
每用户仿射 | 50 | 1.5399 | 0.1730 |
每用户仿射 | 100 | 1.5299 | 0.1691 |
每用户偏置 | 5 | 1.7443 | 0.1970 |
每用户偏置 | 10 | 1.6383 | 0.1861 |
每用户偏置 | 20 | 1.5875 | 0.1804 |
每用户偏置 | 50 | 1.5623 | 0.1772 |

结论：

- 每用户校准上限约 **9%**（1.614 → 1.473）；
- K<20 时仿射严重过拟合，偏置模型更稳；
- K≈50 时仿射开始优于偏置；
- 需要层次先验/收缩，在偏置和仿射之间自适应。


### 4.9 当前 cm 模型上的上下文对照（完整测试集）

测试集 179,496 帧，`new_input_calib`，robust，landmark/pose 特征：

| 上下文 | K | base | 校准后 |
|---|---|---|---|
orientation | 100 | 1.2145 | 1.1202 |
orientation | 200 | 1.1925 | **1.0684** |
subject | 100 | 1.3755 | 1.2435 |
subject | 200 | 1.3673 | 1.1983 |

结论：

- 当前 cm 模型上，**orientation 上下文仍优于 subject 单独分组**；
- subject 单独分组会把不同朝向混在一起，校准反而变差；
- 每用户校准应和朝向/姿态结合，而不是替代朝向。

---

## 5. 代码地图

### 模型

- `src/gazelab/models/dinov2_gaze/rgbdgaze_dinov2.py`
  - DINOv2 backbone；
  - geom/orientation token；
  - 眼裁剪；
  - BlockMoba 融合；
  - 输出头。

### 数据集

- `src/gazelab/datasets/gazecapture.py`
  - 人脸/眼裁剪；
  - geom 7 维；
  - `target_space="norm" | "cm"`；
  - `imu_valid`；
  - `set_aug_seed` / `set_epoch`。

### 训练

- `scripts/calibration/cm_train_epoch.py`（当前主训练脚本）
  - epoch 级续训；
  - DDP；
  - 梯度累积；
  - 冻结/解冻；
  - 归一化+cm 损失；
  - 完整日志。

- `scripts/calibration/cm_train.py`（旧 batch 续训版，已不建议用于正式训练）

### 推理/评测

- `scripts/calibration/cm_infer.py`
  - 厘米模型推理，转回归一化坐标，输出 npz。
- `scripts/calibration/canon_infer.py`
  - 归一化模型推理/TTA。
- `scripts/calibration/physical_space_audit.py`
  - 坐标一致性审计。
- `scripts/calibration/facegrid_diag.py`
  - face grid / 设备 / 用户解释率诊断。
- `scripts/calibration/phys_calib.py`
  - 物理空间校准/迁移。
- `scripts/calibration/new_input_calib.py`
  - 特征校准（landmark/pose/IMU/context）。

### 配对/等变实验（历史）

- `scripts/calibration/pair_feasibility.py`
- `scripts/calibration/build_pairs.py`
- `scripts/calibration/pair_train.py`
- `scripts/calibration/equivariance_diag.py`
- `scripts/calibration/equiv_probe.py`

这些是之前“厘米/等变”探索；目前主训练不使用。

---

## 6. 关键 checkpoint

旧归一化基线：

```text
out/gazecapture/ablation/checkpoints/gc_geom_go224_g7/last.pt
```

当前厘米模型：

```text
out/gazecapture/analysis/cm_ctx_ft/best.pt
out/gazecapture/analysis/cm_ctx_ft/epoch_1.pt
out/gazecapture/analysis/cm_ctx_ft/epoch_2.pt
out/gazecapture/analysis/cm_ctx_ft/last.pt
```

当前 best：

```text
epoch 2 step 72000
val_norm 0.1750
val_cm   1.5988
```

---

## 7. 当前数据事实

GC 训练集：

```text
总帧数 1,251,935
训练被试 1,263
测试被试 150（与训练无重叠）
orientation 分布：1 / 4 / 3 / 2
IMU 无效帧：2,125（|g_xy| < 0.05）
NaN cm/gaze：0
```

设备：

- GC 15 种设备，全部在官方设备表；
- RGD 13 种设备，官方旧表没有，相机偏移不可用。

---

## 8. 日志与监控

训练日志包含：

- git commit；
- 完整参数；
- train/val 数量；
- split / orientation / device 分布；
- IMU 无效帧数；
- NaN 数；
- 参数量；
- 每 step：loss / lnorm / lcm / grad norm / lr / samples/s / ETA；
- 每 epoch：train_loss / val_cm / val_norm / 分朝向 n / cm / bias；
- checkpoint 保存/恢复。

日志位置：

```text
out/<dataset>/analysis/<run>.log
out/<dataset>/analysis/<run>/train_log.csv
out/<dataset>/analysis/<run>/val_log.csv
```

---

## 9. 下一步：模型架构改进前期工作

### 优先级 1：用户个性化

依据：subject η² = 23.68%。

候选：

- 用户 embedding；
- 轻量 adapter；
- 测试时少量标定点学习个性化参数；
- 主体共享，个性化只改小部分。

### 优先级 2：不确定性预测

候选：

- 输出均值 + 方差；
- Gaussian NLL；
- 用于校准加权、剔除不可靠帧、主动选点。

### 优先级 3：时序融合

GC 每个注视点多帧：

- 多帧预测平均/加权；
- 校准更稳；
- 不改模型结构也可先做。

### 优先级 4：face grid（可选消融）

- 诊断显示只对小脸有帮助；
- 如果做，作为 token，先训头再微调；
- 不保证整体提升。

### 不做/低优先级

- 相机偏移；
- 全部解冻 patch/pos；
- 更大模型/更高分辨率；
- 继续无目的训练更多 epoch。

---

## 10. 校准下一步

### 1. 按用户/会话校准

$$
\hat y = A_s p + b_s + W f
$$

- \(s\)：用户/会话；
- \(f\)：landmark / pose / IMU / 屏幕尺寸；
- K 个标签。

### 2. 层次先验

- 全局先验；
- 用户/设备先验；
- 少量标签 MAP 更新。

### 3. 标签选择

- D-optimal；
- 按不确定性；
- 按屏幕区域分层。

### 4. 不确定性加权

- 低置信度帧降权；
- 校准更稳。

### 5. 时序平均

- 同一 dot 多帧平均。

---

## 11. 工程规则

- 提交信息用中文；
- 先写代码，再自查，再预检，最后跑正式训练；
- 日志必须完整，方便定位；
- 正式训练用 epoch 级续训；
- 多卡 DDP，断点续训；
- GPU 使用 1,2,3,5,6,7，避免 4；
- 资源不超过 75%；
- 不重复跑已知失败的路径。

---

## 12. 已知失败/不要重复

- 归一化坐标 2D 旋转迁移；
- 同图合成配对等变训练；
- 纯厘米 MSE 从头训练；
- 相机偏移优先；
- 简单池化不同上下文；
- 逐 batch 续训跳过数据；
- 默认 `geom_randomize=0.5`；
- 未设置增强种子。
