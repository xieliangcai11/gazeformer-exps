#!/usr/bin/env bash
# 等待至少 4 张非 GPU4 的卡各自有 >= MIN_FREE_MIB 空闲，然后启动 GC cm+IMU 阶段 B。
# 只启动一次；训练进程结束后脚本结束。
set -u
cd /data1/xieliangcai/code/gazeformer-exps || exit 1
source /data1/xieliangcai/conda/etc/profile.d/conda.sh
conda activate dl || exit 1
MIN_FREE_MIB=${MIN_FREE_MIB:-14000}
MAX_GPUS=${MAX_GPUS:-6}
LOG=out/gazecapture/analysis/cm_imu_stageB_watch.log
mkdir -p out/gazecapture/analysis
echo "[watchB] start $(date) min_free=${MIN_FREE_MIB}MiB max_gpus=${MAX_GPUS}" >> "$LOG"
while true; do
  mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F',' -v m="$MIN_FREE_MIB" '$1+0 != 4 && $2+0 >= m {gsub(/ /,"",$1); print $1}')
  N=${#GPUS[@]}
  echo "[watchB] $(date +%H:%M:%S) free_gpus=$N ${GPUS[*]:-}" >> "$LOG"
  if [ "$N" -ge 4 ]; then
    if [ "$N" -gt "$MAX_GPUS" ]; then
      GPUS=("${GPUS[@]:0:$MAX_GPUS}")
      N=$MAX_GPUS
    fi
    GPU_CSV=$(IFS=,; echo "${GPUS[*]}")
    echo "[watchB] launching stageB on ${GPU_CSV} ($N gpus) $(date)" >> "$LOG"
    CUDA_VISIBLE_DEVICES="$GPU_CSV" torchrun --standalone --nproc_per_node="$N" \
      -m scripts.calibration.cm_train_epoch \
      --ckpt out/gazecapture/analysis/cm_imu_stageA/last.pt \
      --resume out/gazecapture/analysis/cm_imu_stageA/last.pt \
      --use-imu --val-split val \
      --epochs 6 \
      --batch 24 --accum 1 \
      --lr 5e-5 --backbone-lr-ratio 0.05 \
      --weight-decay 0.01 --warmup 200 \
      --loss norm_plus_cm --cm-weight 0.01 \
      --augment --aug-shift 0.05 --aug-scale 0.05 --aug-rotate 0 \
      --geom-randomize 0 \
      --val-stride 10 --val-limit 0 \
      --log-every 200 --val-every 0 --max-norm-rise 0.01 \
      --num-workers 4 --distributed \
      --out out/gazecapture/analysis/cm_imu_stageB \
      >> out/gazecapture/analysis/cm_imu_stageB.log 2>&1
    RC=$?
    echo "[watchB] stageB exited rc=$RC $(date)" >> "$LOG"
    tail -30 out/gazecapture/analysis/cm_imu_stageB.log >> "$LOG"
    exit "$RC"
  fi
  sleep 300
done
