# 两个数据集的训练命令存档

> 目的：保存实际使用过的训练/微调命令，避免后续凭印象判断。
> 说明：命令来自训练日志中的“复现/命令”行和 `train_args.json`；没有记录的命令会明确标注“重建”。

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
