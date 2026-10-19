#!/usr/bin/env bash
# 等"3 种子基线队列"跑完 ⇒ 自动接上「低学习率 + 更多 epoch」变体
#   （用户的判断：GazeCapture 效果不好可能也是 lr 太大 ⇒ 记下来实测）
#
# 变体：lr 2e-4 → 1e-4；epochs 5 → 12；其余**完全相同**（同一套修好的可复现增强 ✓）
# 只跑 1 个种子先看方向（每条约 20~26 小时）；若明显更好再补 3 种子 ✓
#
# 注意：train.py 目前**没有学习率调度器**（审计已记录 ✗）⇒ 本变体只改 lr 与 epochs；
#      "加余弦衰减"需要改代码，是下一步的候选（比单纯调 lr 更可能有收益 ✓）。
#
# 用法：nohup bash scripts/rgbdgaze/gc_lr_queue.sh > out/gazecapture/gc_lr_driver.log 2>&1 &
set -u
cd /data1/xieliangcai/code/gazeformer-exps
source /data1/xieliangcai/conda/etc/profile.d/conda.sh
conda activate dl

LOG=out/gazecapture/gc_lr_queue.log
echo "[gc-lr] 等 3 种子基线队列完成（gc_queue.log 出现『全部完成』）..." | tee -a "$LOG"
for _ in $(seq 1 3000); do
  grep -q "全部完成" out/gazecapture/gc_queue.log 2>/dev/null && break
  sleep 120
done
echo "[gc-lr] 基线队列已完成 ⇒ 启动低 lr 变体 $(date '+%F %T')" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --standalone --nproc_per_node=6 \
  -m scripts.train \
  --dataset gazecapture --resolution 224 --no-geom --geom-randomize 0 --augment \
  --aug-shift 0.05 --aug-scale 0.05 --aug-rotate 0 --unfreeze 12 --diff-lr \
  --batch-size 32 --lr 1.0e-4 --num-workers 8 --epochs 12 --seed 0 --tag gc_lr1e4_e12 \
  >> "$LOG" 2>&1

echo "[gc-lr] 变体完成 $(date '+%F %T')" | tee -a "$LOG"
grep -E "^\[epoch [0-9]+\]  train_loss" out/gazecapture/ablation/logs/*gc_lr1e4_e12*.txt 2>/dev/null | tail -3 | tee -a "$LOG"
