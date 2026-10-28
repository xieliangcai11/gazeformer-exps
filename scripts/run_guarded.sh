#!/usr/bin/env bash
# Launch a job only if at least 25% of RAM and target GPU memory are free.
set -euo pipefail
FRAC="${MIN_FREE_FRAC:-0.25}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
FREE_RAM=$(free -g | awk '/^Mem:/{print $7}')
MIN_RAM=$(awk -v t="$TOTAL_RAM" -v f="$FRAC" 'BEGIN{printf "%d", t*f}')
TOTAL_GPU=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$GPU" | head -1 | tr -d ' ')
FREE_GPU=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU" | head -1 | tr -d ' ')
MIN_GPU=$(awk -v t="${TOTAL_GPU:-0}" -v f="$FRAC" 'BEGIN{printf "%d", t*f}')
if [ "$FREE_RAM" -lt "$MIN_RAM" ]; then
  echo "[guard] free RAM ${FREE_RAM}GiB < ${MIN_RAM}GiB" >&2; exit 2
fi
if [ -z "$FREE_GPU" ] || [ "$FREE_GPU" -lt "$MIN_GPU" ]; then
  echo "[guard] free GPU${GPU} ${FREE_GPU:-?}MiB < ${MIN_GPU}MiB" >&2; exit 3
fi
echo "[guard] ok ram=${FREE_RAM}/${TOTAL_RAM}GiB gpu${GPU}=${FREE_GPU}/${TOTAL_GPU}MiB :: $*"
exec "$@"
