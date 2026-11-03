#!/usr/bin/env bash
# 主动校准第一版：dot-level + feature-diversity/hybrid（2026-11-03）
#
# 结果（seeds=3，464 会话，DINO-only，dot-level 残差）：
#   frame@50（24.5 dot） 1.2045 cm，worst +41.0%
#   hybrid@20（19.1 dot）1.1986 cm，worst +13.3%
#   hybrid@16（15.5 dot）1.2102 cm，worst +14.7%
#   hybrid@12（11.8 dot）1.2274 cm，worst +14.2%
#
# 资源：单进程、纯 CPU、2 个 BLAS 线程、nice 10；约 2.5 分钟，maxrss 1.22GB。
# 用法：bash scripts/calibration/run_active_calib_best.sh
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
OUTDIR=out/gazecapture/analysis/active_sim_dino_s3_repro
LOG=out/gazecapture/analysis/active_sim_dino_s3_repro.log

for f in "$TRAIN" "$TEST" "$INDEX" "$DINO"; do
  if [ ! -f "$f" ]; then
    echo "[error] 缺少输入：$f" >&2
    exit 1
  fi
done

mkdir -p "$OUTDIR"

echo "[active] start $(date '+%F %T')"
echo "[active] repo=$ROOT"
echo "[active] out=$OUTDIR"
echo "[active] log=$LOG"
echo "[active] methods=random,dotcov,featdiv,hybrid,frame"
echo "[active] K_dots=12,16,20 K_frames=50 seeds=3 features=dino regions=none"

set +e
nice -n 10 env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python -u -m scripts.calibration.active_calib_sim \
    --train-npz "$TRAIN" \
    --test-npz "$TEST" \
    --index-csv "$INDEX" \
    --dino-npz "$DINO" \
    --features dino \
    --regions none \
    --residual-level dot \
    --methods random,dotcov,featdiv,hybrid,frame \
    --K-dots 12 16 20 \
    --K-frames 50 \
    --seeds 3 \
    --out-dir "$OUTDIR" \
    > "$LOG" 2>&1

rc=$?
set -e
echo "[active] exit=$rc $(date '+%F %T')"
echo "[active] last lines:"
tail -n 20 "$LOG"
exit "$rc"
