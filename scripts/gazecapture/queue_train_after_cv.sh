#!/usr/bin/env bash
# 等待 RGBDGaze 5 折 CV 完成，然后自动用 3 卡续训 GazeCapture（--resume auto）。
# 用法（tmux）：
#   tmux new -s train_after_cv "bash scripts/gazecapture/queue_train_after_cv.sh \
#       > out/gazecapture/queue_train_after_cv.log 2>&1"
set -u
cd /data1/xieliangcai/code/gazeformer-exps
source /data1/xieliangcai/conda/etc/profile.d/conda.sh
conda activate dl

D=out/rgbdgaze/analysis/context
for i in $(seq 1 240); do          # 最长等 ~12h（每 3 分钟查一次）
  n=$(ls "$D"/cv_fold*.csv 2>/dev/null | wc -l)
  if [ "$n" -ge 5 ] && ! pgrep -f "[r]un_group_cv" >/dev/null; then
    echo "[watcher] CV 完成（$n 折）"
    break
  fi
  sleep 180
done

echo "[watcher] 启动 3 卡续训（resume auto）"
tmux new-session -d -s train_gc3 "cd /data1/xieliangcai/code/gazeformer-exps && \
  source /data1/xieliangcai/conda/etc/profile.d/conda.sh && conda activate dl && \
  CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --standalone --nproc_per_node=4 \
  -m scripts.train --dataset gazecapture --resolution 224 --no-geom \
  --geom-randomize 0 --augment --aug-shift 0.05 --aug-scale 0.05 --aug-rotate 0 \
  --unfreeze 12 --diff-lr --lr 1e-4 --batch-size 32 --num-workers 4 --epochs 5 \
  --seed 0 --use-eye --tag gc_eye_lr1e4 > out/gazecapture/gc_eye_lr1e4_v7.log 2>&1"
echo "[watcher] train_gc3 已启动"
