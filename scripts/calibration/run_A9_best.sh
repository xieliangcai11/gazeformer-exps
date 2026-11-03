#!/usr/bin/env bash
# A9：cm 模型 full-test 校准主结果（2026-11-03）
#
# 结果：raw 1.5918 / base 1.2064 / calib 1.1461 cm，431/443 会话变好。
# 配置：DINO16 + landmark21 + 3x3 region，target standardization，
#       LOO lambda，per-axis Huber，fixed clip 0.02 norm，K=50，seeds=3。
#
# 资源：单进程、纯 CPU、2 个 BLAS 线程、nice 10；实测约 45 秒，maxrss 1.22GB。
# 用法：bash scripts/calibration/run_A9_best.sh
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"

CONDA_SH=${CONDA_SH:-/data1/xieliangcai/conda/etc/profile.d/conda.sh}
if [ -f "$CONDA_SH" ]; then
  # shellcheck disable=SC1090
  source "$CONDA_SH"
  conda activate dl
else
  echo "[warn] 未找到 $CONDA_SH；假设当前 python 已在 dl 环境" >&2
fi

TRAIN=out/gazecapture/analysis/feats_cm_val.npz
TEST=out/gazecapture/analysis/feats_cm_test_full.npz
INDEX=data/GazeCapture/index/index.csv
DINO=out/gazecapture/analysis/dino_pca16_cm_full.npz
LANDMARK=out/gazecapture/analysis/lm_gc_test.npz
OUT=out/gazecapture/analysis/A9_best.csv
DETAIL=out/gazecapture/analysis/A9_best_detail.csv
LOG=out/gazecapture/analysis/A9_best.log

for f in "$TRAIN" "$TEST" "$INDEX" "$DINO" "$LANDMARK"; do
  if [ ! -f "$f" ]; then
    echo "[error] 缺少输入：$f" >&2
    exit 1
  fi
done

mkdir -p "$(dirname "$OUT")"

echo "[A9] start $(date '+%F %T')"
echo "[A9] repo=$ROOT"
echo "[A9] out=$OUT"
echo "[A9] log=$LOG"
echo "[A9] cmd=cm_residual_calib features=dino,landmark regions=3x3 "
echo "     K=50 seeds=3 target_standardize=1 loo_lam=1 huber_scale=axis clip_norm=0.02"

set +e
nice -n 10 env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -u -m scripts.calibration.cm_residual_calib \
    --train-npz "$TRAIN" \
    --test-npz "$TEST" \
    --index-csv "$INDEX" \
    --dino-npz "$DINO" \
    --landmark-npz "$LANDMARK" \
    --features dino,landmark \
    --regions 3x3 \
    --K 50 --seeds 3 \
    --target-standardize --loo-lam \
    --huber-scale axis \
    --clip-mode fixed --clip-norm 0.02 \
    --out "$OUT" \
    --detail "$DETAIL" \
    > "$LOG" 2>&1

rc=$?
set -e
echo "[A9] exit=$rc $(date '+%F %T')"
echo "[A9] last lines:"
tail -n 12 "$LOG"
exit "$rc"
