# 两个数据集的训练命令存档

> 目的：保存实际使用过的训练/微调命令，避免后续凭印象判断。
> 说明：命令来自训练日志中的“复现/命令”行和 `train_args.json`；没有记录的命令会明确标注“重建”。


---

## 0. 关键口径：1.47 是固定 split，不是 fold3

来源：`out/rgbdgaze/eye224_mgpu.log`

```text
[final] test_L2 0.1345  test_em 1.47cm (x:0.74 y:1.11)
```

完整原始命令：

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 \
torchrun --standalone --nproc_per_node=6 -m scripts.train \
  --dataset rgbdgaze \
  --resolution 224 \
  --no-geom \
  --geom-randomize 0 \
  --augment \
  --aug-shift 0.05 \
  --aug-scale 0.05 \
  --aug-rotate 0 \
  --unfreeze 12 \
  --diff-lr \
  --batch-size 4 \
  --lr 1e-4 \
  --num-workers 6 \
  --epochs 8 \
  --use-eye \
  --tag eye224_mgpu
```

配置：

```text
数据集：RGBDGaze
索引：data/RGBDGaze/index        # 固定 split，不是 cv/fold3
划分：train 127257 / val 4381 / test 11031
模型：旧归一化模型，输出归一化坐标，无 geom
loss：mse
unfreeze：12
eye：True
diff_lr：True
batch/卡：4
卡数：6
有效批：24
lr：1e-4
epochs：8
seed：0
```

checkpoint：

```text
out/rgbdgaze/ablation/checkpoints/eye224_mgpu/best.pt
```

注意：

```text
1.47 = 固定 split 的结果
1.48 = CV fold3 的结果
```

两者不能混用。

---

## 0b. GC 最近一次续训（cm_ctx_ft3）

来源：`out/gazecapture/analysis/cm_ctx_ft3.log` / `args.json`。
这是 `cm_ctx_ft` 的第 4 个 epoch 续训，使用 6 卡、batch 8、accum 4。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,5,6 \
torchrun --standalone --nproc_per_node=6 -m scripts.calibration.cm_train_epoch \
  --ckpt out/gazecapture/analysis/cm_ctx_ft/last.pt \
  --geom-orient \
  --resume out/gazecapture/analysis/cm_ctx_ft/last.pt \
  --index-dir data/GazeCapture/index \
  --out out/gazecapture/analysis/cm_ctx_ft \
  --epochs 4 \
  --batch 8 \
  --accum 4 \
  --lr 5e-05 \
  --backbone-lr-ratio 0.05 \
  --weight-decay 0.01 \
  --warmup 200 \
  --loss norm_plus_cm \
  --cm-weight 0.01 \
  --huber-beta 0.05 \
  --augment \
  --aug-shift 0.05 \
  --aug-scale 0.05 \
  --aug-rotate 0 \
  --geom-randomize 0 \
  --val-stride 300 \
  --val-limit 512 \
  --log-every 200 \
  --val-every 2000 \
  --max-norm-rise 0.01 \
  --num-workers 3 \
  --distributed
```

说明：

```text
日志里的 args 不含 CUDA_VISIBLE_DEVICES；
GPU 列表按项目约定“跳过 GPU4”记录为 0,1,2,3,5,6。
如果实际卡号不同，以日志中的 rank/device 为准。
```

记录结果：

```text
resume start_epoch=3 step=78246 best_norm=0.1750
```

checkpoint：

```text
out/gazecapture/analysis/cm_ctx_ft/best.pt
```



---

## 0c. 旧 recipe 复跑 fold3（1.51 cm）

用途：验证“当前代码 + 旧 1.47 recipe”能否复现旧基线。

```bash
CUDA_VISIBLE_DEVICES=0,1,5,7 \
torchrun --standalone --nproc_per_node=4 -m scripts.train \
  --dataset rgbdgaze \
  --index-dir data/RGBDGaze/index/cv/fold3 \
  --resolution 224 \
  --use-eye \
  --no-geom \
  --geom-randomize 0 \
  --augment \
  --aug-shift 0.05 \
  --aug-scale 0.05 \
  --aug-rotate 0 \
  --unfreeze 12 \
  --diff-lr \
  --batch-size 6 \
  --lr 1e-4 \
  --num-workers 4 \
  --epochs 8 \
  --seed 0 \
  --resume off \
  --tag rgd_fold3_oldrecipe_b6
```

说明：

```text
有效批 = 6 × 4 = 24
旧 1.47：batch 4 × 6 卡 = 24
```

结果：

```text
test_L2  = 0.1359
test_em  = 1.51 cm
test_ang = 2.83°
n        = 25703
```

训练损失：

```text
epoch 0  0.0220
epoch 1  0.0096
epoch 2  0.0074
epoch 3  0.0062
epoch 7  0.0041
```

对照旧 1.47：

```text
epoch 0  0.0217
epoch 1  0.0089
epoch 2  0.0072
epoch 3  0.0061
epoch 7  0.0042
```

结论：

```text
当前代码 + 旧 recipe 可复现旧基线（1.51 vs 1.48）。
当前归一化实验差主要来自 training recipe，不是归一化输出本身。
```

checkpoint：

```text
out/rgbdgaze/ablation/checkpoints/rgd_fold3_oldrecipe_b6/best.pt
```



---

## 0d. 旧 recipe 续训到 16 epoch（过拟合）

续训命令（从 8 epoch 的 last.pt 接上）：

```bash
CUDA_VISIBLE_DEVICES=0,1,5,7 \
torchrun --standalone --nproc_per_node=4 -m scripts.train \
  --dataset rgbdgaze \
  --index-dir data/RGBDGaze/index/cv/fold3 \
  --resolution 224 \
  --use-eye \
  --no-geom \
  --geom-randomize 0 \
  --augment \
  --aug-shift 0.05 --aug-scale 0.05 --aug-rotate 0 \
  --unfreeze 12 --diff-lr \
  --batch-size 6 --lr 1e-4 \
  --num-workers 4 --epochs 16 --seed 0 \
  --resume out/rgbdgaze/ablation/checkpoints/rgd_fold3_oldrecipe_b6/last.pt \
  --tag rgd_fold3_oldrecipe_b6_e16
```

训练损失：

```text
epoch 7  0.0041
epoch 8  0.0038
epoch 10 0.0033
epoch 12 0.0030
epoch 15 0.0027
```

test：

```text
8 epoch  : 1.51 cm / L2 0.1359 / ang 2.83°
16 epoch : 1.61 cm / L2 0.1432 / ang 3.02°
```

结论：

```text
训练损失继续下降，但 test 变差 ⇒ 16 epoch 已过拟合。
fold3 旧 recipe 的最佳轮数在 8 epoch 左右。
```


---

## 1. GazeCapture（GC）最佳基线

来源：`out/gazecapture/gc_geom_go224_g7.log` 的“复现”行。

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7 \
torchrun --standalone --nproc_per_node=7 \
  -m scripts.train \
  --dataset gazecapture \
  --index-dir data/GazeCapture/index \
  --resolution 224 \
  --geom \
  --geom-orient \
  --geom-randomize 0 \
  --augment \
  --aug-shift 0.05 \
  --aug-scale 0.05 \
  --aug-rotate 0 \
  --unfreeze 12 \
  --diff-lr \
  --lr 1.4e-4 \
  --batch-size 36 \
  --num-workers 4 \
  --epochs 6 \
  --seed 0 \
  --use-eye \
  --tag gc_geom_go224_g7 \
  --resume auto
```

记录结果：

```text
验证：约 1.65 cm / 3.12°
测试归一化：0.1806
测试 cm：约 1.66
```

checkpoint：

```text
out/gazecapture/ablation/checkpoints/gc_geom_go224_g7/best.pt
out/gazecapture/ablation/checkpoints/gc_geom_go224_g7/last.pt
```

---

## 2. RGBDGaze（RGD）CV 训练命令

来源：`out/rgbdgaze/ablation/logs/*cv_fold*.txt` 的“命令”行。

### fold0

```bash
python scripts/train.py \
  --dataset rgbdgaze \
  --index-dir data/RGBDGaze/index/cv/fold0 \
  --resolution 224 \
  --use-eye \
  --no-geom \
  --augment \
  --aug-shift 0.05 \
  --aug-scale 0.05 \
  --aug-rotate 0 \
  --geom-randomize 0 \
  --unfreeze 12 \
  --diff-lr \
  --lr 0.0001 \
  --batch-size 24 \
  --num-workers 4 \
  --epochs 8 \
  --seed 0 \
  --resume off \
  --tag cv_fold0
```

### fold1

```bash
python scripts/train.py \
  --dataset rgbdgaze \
  --index-dir data/RGBDGaze/index/cv/fold1 \
  --resolution 224 \
  --use-eye \
  --no-geom \
  --augment \
  --aug-shift 0.05 \
  --aug-scale 0.05 \
  --aug-rotate 0 \
  --geom-randomize 0 \
  --unfreeze 12 \
  --diff-lr \
  --lr 0.0001 \
  --batch-size 24 \
  --num-workers 4 \
  --epochs 8 \
  --seed 0 \
  --resume off \
  --tag cv_fold1
```

### fold2

```bash
python scripts/train.py \
  --dataset rgbdgaze \
  --index-dir data/RGBDGaze/index/cv/fold2 \
  --resolution 224 \
  --use-eye \
  --no-geom \
  --augment \
  --aug-shift 0.05 \
  --aug-scale 0.05 \
  --aug-rotate 0 \
  --geom-randomize 0 \
  --unfreeze 12 \
  --diff-lr \
  --lr 0.0001 \
  --batch-size 24 \
  --num-workers 4 \
  --epochs 8 \
  --seed 0 \
  --resume off \
  --tag cv_fold2
```

### fold3

```bash
python scripts/train.py \
  --dataset rgbdgaze \
  --index-dir data/RGBDGaze/index/cv/fold3 \
  --resolution 224 \
  --use-eye \
  --no-geom \
  --augment \
  --aug-shift 0.05 \
  --aug-scale 0.05 \
  --aug-rotate 0 \
  --geom-randomize 0 \
  --unfreeze 12 \
  --diff-lr \
  --lr 0.0001 \
  --batch-size 24 \
  --num-workers 4 \
  --epochs 8 \
  --seed 0 \
  --resume off \
  --tag cv_fold3
```

### fold4

```bash
python scripts/train.py \
  --dataset rgbdgaze \
  --index-dir data/RGBDGaze/index/cv/fold4 \
  --resolution 224 \
  --use-eye \
  --no-geom \
  --augment \
  --aug-shift 0.05 \
  --aug-scale 0.05 \
  --aug-rotate 0 \
  --geom-randomize 0 \
  --unfreeze 12 \
  --diff-lr \
  --lr 0.0001 \
  --batch-size 24 \
  --num-workers 4 \
  --epochs 8 \
  --seed 0 \
  --resume off \
  --tag cv_fold4
```

记录结果（测试 cm）：

```text
fold3: 1.48 cm
fold4: 2.02 cm
```

checkpoint：

```text
out/rgbdgaze/ablation/checkpoints/cv_fold3/best.pt
```

---

## 3. RGBDGaze champion 配置（日志无完整命令）

来源：`out/rgbdgaze/ablation/logs/20261005_005527_ablation_champion_final_test.txt` 头部。

```text
tag: champion_final_test
loss: mse
depth: True
unfreeze: 12
train: 127257
val: 4381
test: 11031
batch: 64
lr: 1e-4
epochs: 20
```

没有找到完整“命令”行，下面是按头部信息重建的近似命令，**不能视为原始命令**：

```bash
python scripts/rgbdgaze/ablation.py \
  --tag champion_final_test \
  --loss mse \
  --depth \
  --unfreeze 12 \
  --batch-size 64 \
  --lr 1e-4 \
  --epochs 20
```

记录测试结果：

```text
test_em = 1.96 cm
test_ang = 3.67°
```

checkpoint：

```text
out/rgbdgaze/ablation/checkpoints/champion_final_test/best.pt
```

---

## 4. 当前厘米模型（GC）训练命令

### 4.1 阶段 A：冻结骨干，训输出头

```bash
CUDA_VISIBLE_DEVICES=1,2,3,5,6,7 \
torchrun --standalone --nproc_per_node=6 \
  -m scripts.calibration.cm_train_epoch \
  --ckpt out/gazecapture/ablation/checkpoints/gc_geom_go224_g7/last.pt \
  --reset-head \
  --freeze-backbone \
  --epochs 1 \
  --batch 8 --accum 4 \
  --lr 1e-3 --warmup 200 \
  --loss norm_plus_cm --cm-weight 0.01 \
  --geom-orient \
  --augment --aug-shift 0.05 --aug-scale 0.05 --aug-rotate 0 \
  --geom-randomize 0 \
  --val-stride 300 --val-limit 512 \
  --log-every 200 --val-every 2000 \
  --num-workers 3 --distributed \
  --out out/gazecapture/analysis/cm_ctx_head
```

结果：

```text
best val_norm = 0.1871
test norm = 0.1924
test cm = 1.7014
```

### 4.2 阶段 B：解冻 12 block 微调

```bash
CUDA_VISIBLE_DEVICES=1,2,3,5,6,7 \
torchrun --standalone --nproc_per_node=6 \
  -m scripts.calibration.cm_train_epoch \
  --ckpt out/gazecapture/analysis/cm_ctx_head/last.pt \
  --resume out/gazecapture/analysis/cm_ctx_head/last.pt \
  --epochs 2 \
  --batch 8 --accum 4 \
  --lr 1e-4 --backbone-lr-ratio 0.1 --warmup 500 \
  --loss norm_plus_cm --cm-weight 0.01 \
  --geom-orient \
  --augment --aug-shift 0.05 --aug-scale 0.05 --aug-rotate 0 \
  --geom-randomize 0 \
  --val-stride 300 --val-limit 512 \
  --log-every 200 --val-every 2000 \
  --num-workers 3 --distributed \
  --out out/gazecapture/analysis/cm_ctx_ft
```

结果：

```text
epoch 1 最好：val_norm 0.1754 / val_cm 1.6118
epoch 2 最好：val_norm 0.1750 / val_cm 1.5988
测试（best）：norm 0.1856 / cm 1.6219
```

### 4.3 第 2 个 epoch

```bash
CUDA_VISIBLE_DEVICES=1,2,3,5,6,7 \
torchrun --standalone --nproc_per_node=6 \
  -m scripts.calibration.cm_train_epoch \
  --ckpt out/gazecapture/analysis/cm_ctx_ft/last.pt \
  --resume out/gazecapture/analysis/cm_ctx_ft/last.pt \
  --epochs 3 \
  --batch 8 --accum 4 \
  --lr 1e-4 --backbone-lr-ratio 0.1 --warmup 500 \
  --loss norm_plus_cm --cm-weight 0.01 \
  --geom-orient \
  --augment --aug-shift 0.05 --aug-scale 0.05 --aug-rotate 0 \
  --geom-randomize 0 \
  --val-stride 300 --val-limit 512 \
  --log-every 200 --val-every 2000 \
  --num-workers 3 --distributed \
  --out out/gazecapture/analysis/cm_ctx_ft
```

---

## 5. 当前厘米模型在 RGBDGaze 上微调

```bash
CUDA_VISIBLE_DEVICES=1,2 \
torchrun --standalone --nproc_per_node=2 \
  -m scripts.calibration.rgd_cm_ft \
  --ckpt out/gazecapture/analysis/cm_ctx_ft/best.pt \
  --epochs 2 \
  --batch 16 --accum 2 \
  --lr 1e-4 --backbone-lr-ratio 0.1 --warmup 200 \
  --augment \
  --val-every 500 --log-every 100 \
  --num-workers 3 --distributed \
  --out out/rgbdgaze/analysis/cm_ft_full
```

结果：

```text
零样本：val 3.88 cm / test 4.32 cm
epoch 0：val 1.6964 / test 1.9257
epoch 1：val 约 1.75（test 未完成）
```

结论：

> 没有超过历史最好 1.47–1.48 cm；仅略好于旧 champion 1.96 cm。

---

## 6. 关键评估/标定命令

### GC 完整测试预测

```bash
python -m scripts.calibration.cm_infer \
  --ckpt out/gazecapture/analysis/cm_ctx_ft/best.pt \
  --index-dir data/GazeCapture/index \
  --out out/gazecapture/analysis/pred_cm_full.npz \
  --split test --stride 1 --batch 16 --gpu 0
```

### 校准

```bash
python -m scripts.calibration.new_input_calib \
  --train-npz out/gazecapture/analysis/feats_gc_train300.npz \
  --val-npz out/gazecapture/analysis/feats_gc_val_m2_full.npz \
  --test-npz out/gazecapture/analysis/pred_cm_full.npz \
  --index-csv data/GazeCapture/index/index.csv \
  --pose-npz out/gazecapture/analysis/pose_features.npz \
  --landmark-npz out/gazecapture/analysis/lm_gc_test.npz \
  --context-key orientation \
  --K 50 --lam 0.1 --delta 1.0 --seeds 2 \
  --split test --groups all --method ridge --robust --robust-iters 3 \
  --extra-npz out/gazecapture/analysis/dino_pca16.npz \
  --out out/gazecapture/analysis/ncal_dino_dino_k50.csv
```

---

## 7. 重要提醒

- 正式训练必须显式 `--geom-randomize 0`；
- 必须调用 `tr.set_aug_seed(seed)` / `set_epoch(epoch)`；
- 续训按 epoch，不按 batch 跳过数据；
- GPU 不使用 4；
- 资源不超过 75%；
- 结果必须注明是 val 还是 test、哪个 split。
