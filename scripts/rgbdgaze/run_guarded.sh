#!/usr/bin/env bash
# 启动前资源检查：默认要求"至少 25% 空闲"（即总占用不超过 75%）。
# 用法：CUDA_VISIBLE_DEVICES=5 scripts/rgbdgaze/run_guarded.sh python -m ...
# 可覆盖：MIN_FREE_FRAC=0.30 MIN_FREE_RAM_GB=.. MIN_FREE_GPU_MB=..
set -euo pipefail
FRAC="${MIN_FREE_FRAC:-0.25}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
FREE_RAM=$(free -g | awk '/^Mem:/{print $7}')
MIN_RAM=$(awk -v t="$TOTAL_RAM" -v f="$FRAC" 'BEGIN{printf "%d", t*f}')
if [ -n "${MIN_FREE_RAM_GB:-}" ]; then MIN_RAM="$MIN_FREE_RAM_GB"; fi
if [ "$FREE_RAM" -lt "$MIN_RAM" ]; then
  echo "[guard] 可用内存 ${FREE_RAM}GiB < 要求 ${MIN_RAM}GiB（总 ${TOTAL_RAM}GiB，空闲比例<${FRAC}），拒绝启动" >&2
  exit 2
fi
TOTAL_GPU=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$GPU" | head -1 | tr -d ' ')
FREE_GPU=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | head -1 | tr -d ' ')
MIN_GPU=$(awk -v t="${TOTAL_GPU:-0}" -v f="$FRAC" 'BEGIN{printf "%d", t*f}')
if [ -n "${MIN_FREE_GPU_MB:-}" ]; then MIN_GPU="$MIN_FREE_GPU_MB"; fi
if [ -z "$FREE_GPU" ] || [ "$FREE_GPU" -lt "$MIN_GPU" ]; then
  echo "[guard] GPU${GPU} 空闲 ${FREE_GPU:-?}MiB < 要求 ${MIN_GPU}MiB（总 ${TOTAL_GPU}MiB），拒绝启动" >&2
  exit 3
fi
echo "[guard] ok ram=${FREE_RAM}/${TOTAL_RAM}GiB gpu${GPU}=${FREE_GPU}/${TOTAL_GPU}MiB :: $*"
exec "$@"
