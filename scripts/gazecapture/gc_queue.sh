#!/usr/bin/env bash
# GazeCapture 3 种子排队跑（用**修复后**的 harness：可复现增强 + 如实日志 + 配置侧车）
#
# 为什么这么跑：
#   · 预览 run `gc_seed0` 的增强随机源**未播种** ✗ ⇒ 不可复现 ⇒ 只能看趋势，不能当结论；
#   · 对外宣称需要 **≥3 seeds 的均值±方差**，且每条 run 必须**同样的卡数/有效批**（6 卡 × 32 = 192）
#     —— 否则与其它 run 不可比（这是审计里踩过的坑 ✗）。
#
# 用法（detached，建议过夜跑）：约 3 × 9~11 小时 ≈ 27~33 小时
#   nohup bash scripts/rgbdgaze/gc_queue.sh > out/gazecapture/gc_queue_driver.log 2>&1 &
set -u
cd /data1/xieliangcai/code/gazeformer-exps
source /data1/xieliangcai/conda/etc/profile.d/conda.sh
conda activate dl

QUEUE_LOG=out/gazecapture/gc_queue.log
mkdir -p out/gazecapture/ablation/checkpoints

echo "[queue] 等预览 run (gc_seed0) 结束 ..." | tee -a "$QUEUE_LOG"
while pgrep -f "\-\-tag gc_seed0" >/dev/null; do
  sleep 120
done
echo "[queue] 预览 run 已结束，开始 3 种子：$(date '+%F %T')" | tee -a "$QUEUE_LOG"

for S in 0 1 2; do
  TAG="gc_fix_s${S}"
  echo "[queue] === seed $S -> $TAG  $(date '+%F %T') ===" | tee -a "$QUEUE_LOG"
  CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --standalone --nproc_per_node=6 \
    -m scripts.train \
    --dataset gazecapture --resolution 224 --no-geom --geom-randomize 0 --augment \
    --aug-shift 0.05 --aug-scale 0.05 --aug-rotate 0 --unfreeze 12 --diff-lr \
    --batch-size 32 --lr 2.0e-4 --num-workers 8 --epochs 5 --seed "$S" --tag "$TAG" \
    >> "$QUEUE_LOG" 2>&1
  echo "[queue] === seed $S 结束 $(date '+%F %T') ===" | tee -a "$QUEUE_LOG"
done

echo "[queue] 全部完成；汇总各 seed 的 test：" | tee -a "$QUEUE_LOG"
for S in 0 1 2; do
  echo -n "  gc_fix_s${S}: " | tee -a "$QUEUE_LOG"
  grep -E "^\[epoch [0-9]+\]  train_loss" "out/gazecapture/ablation/logs/"*"gc_fix_s${S}"*.txt 2>/dev/null \
    | tail -1 | tee -a "$QUEUE_LOG"
done
