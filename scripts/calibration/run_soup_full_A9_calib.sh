#!/usr/bin/env bash
# 复现 soup 模型 full-test A9 校准（2026-11-04）
#
# 结果：raw 1.5817 / base 1.1975 / calib 1.1383 cm；434/443 会话变好。
# 资源：单进程、纯 CPU、2 个 BLAS 线程、nice 10；约 50 秒，maxrss 1.22GB。
#
# 前置：特征文件已在服务器上（531MB / 176MB，未入 git）：
#   out/gazecapture/analysis/feats_soup_val.npz
#   out/gazecapture/analysis/feats_soup_test_full.npz
#   out/gazecapture/analysis/lm_gc_test.npz
# 若 dino npz 不存在，本脚本会自动生成。
#
# 用法：bash scripts/calibration/run_soup_full_A9_calib.sh
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

VAL=out/gazecapture/analysis/feats_soup_val.npz
TEST=out/gazecapture/analysis/feats_soup_test_full.npz
INDEX=data/GazeCapture/index/index.csv
LANDMARK=out/gazecapture/analysis/lm_gc_test.npz
DINO=out/gazecapture/analysis/dino_pca16_soup_full.npz
OUT=out/gazecapture/analysis/soup_full_A9_repro.csv
DETAIL=out/gazecapture/analysis/soup_full_A9_repro_detail.csv
LOG=out/gazecapture/analysis/soup_full_A9_repro.log

for f in "$VAL" "$TEST" "$INDEX" "$LANDMARK"; do
  if [ ! -f "$f" ]; then
    echo "[error] 缺少输入：$f" >&2
    echo "[hint] soup val/test 特征需先在服务器生成（见 docs/实验_soup_full_test_A9_2026-11-04.md）" >&2
    exit 1
  fi
done

if [ ! -f "$DINO" ]; then
  echo "[dino] 生成 $DINO"
  nice -n 10 env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m \
    scripts.calibration.dino_extra_features \
    --train-npz "$VAL" --test-npz "$TEST" --index-csv "$INDEX" \
    --components 16 --out "$DINO" \
    > out/gazecapture/analysis/dino_pca16_soup_full.log 2>&1
fi

echo "[calib] start $(date '+%F %T')"
echo "[calib] out=$OUT"
echo "[calib] log=$LOG"

set +e
nice -n 10 env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -u -m scripts.calibration.cm_residual_calib \
    --train-npz "$VAL" \
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

echo "[calib] exit=$rc $(date '+%F %T')"
tail -n 15 "$LOG"
exit "$rc"
