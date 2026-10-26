#!/usr/bin/env bash
# 启动两个完全脱离会话的 CV worker（A: fold 0/2/4 用 GPU1；B: fold 1/3 用 GPU2）。
# 用法：bash scripts/analysis/launch_cv_workers.sh
set -u
cd /data1/xieliangcai/code/gazeformer-exps
source /data1/xieliangcai/conda/etc/profile.d/conda.sh
conda activate dl
D=out/rgbdgaze/analysis/context
mkdir -p "$D"

setsid nohup bash -c 'for k in 0 3; do python -u -m scripts.analysis.run_group_cv --gpu 1 --only-fold $k --epochs 8 --num-workers 4; done' \
  > "$D/cv_worker_a.log" 2>&1 < /dev/null &
setsid nohup bash -c 'for k in 1 4; do python -u -m scripts.analysis.run_group_cv --gpu 2 --only-fold $k --epochs 8 --num-workers 4; done' \
  > "$D/cv_worker_b.log" 2>&1 < /dev/null &
setsid nohup bash -c 'python -u -m scripts.analysis.run_group_cv --gpu 3 --only-fold 2 --epochs 8 --num-workers 4' \
  > "$D/cv_worker_c.log" 2>&1 < /dev/null &

sleep 2
echo "[launch] worker A pid=$! （GPU1: folds 0/2/4）"
echo "[launch] worker B （GPU2: folds 1/3）"
