#!/usr/bin/env bash
# 启动前资源检查：内存/显存不足直接拒绝，避免把机器打爆。
# 用法：CUDA_VISIBLE_DEVICES=5 MIN_FREE_RAM_GB=30 MIN_FREE_GPU_MB=8000 \
#        scripts/rgbdgaze/run_guarded.sh python -m ...
set -euo pipefail
MIN_RAM_GB="${MIN_FREE_RAM_GB:-30}"
MIN_GPU_MB="${MIN_FREE_GPU_MB:-8000}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
FREE_RAM=$(free -g | awk '/^Mem:/{print $7}')
if [ "$FREE_RAM" -lt "$MIN_RAM_GB" ]; then
  echo "[guard] 可用内存 ${FREE_RAM}GiB < ${MIN_RAM_GB}GiB，拒绝启动" >&2
  exit 2
fi
FREE_GPU=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | head -1 | tr -d ' ')
if [ -z "$FREE_GPU" ] || [ "$FREE_GPU" -lt "$MIN_GPU_MB" ]; then
  echo "[guard] GPU${GPU} 空闲显存 ${FREE_GPU:-?}MiB < ${MIN_GPU_MB}MiB，拒绝启动" >&2
  exit 3
fi
echo "[guard] ok ram=${FREE_RAM}GiB gpu${GPU}=${FREE_GPU}MiB :: $*"
exec "$@"
