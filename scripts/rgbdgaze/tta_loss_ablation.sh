#!/usr/bin/env bash
# 损失对照（只换无监督损失，其余完全相同）—— 判定"我们的输出矩匹配"是否优于
# 文献同族的增强一致性（光度 / 几何）。
#
# 五个臂（都带安全层与护栏 ✓、100 步、lr 1e-3、rank 8、batch 128、同一模型同一批测试帧）：
#   ours     : 仅 输出矩匹配（我们的）
#   photo    : 仅 光度一致性
#   geo      : 仅 几何一致性（注视不变的小仿射）
#   oursgeo  : 矩匹配 + 几何
#   all      : 三者全开
# 每臂各自输出逐会话明细 CSV；"不标定"参照由每次运行的日志逐会话打印（同帧集可比 ✓）。
#
# 用法（detached，约 2 小时，用空闲的 GPU 1）：
#   nohup bash scripts/rgbdgaze/tta_loss_ablation.sh > out/rgbdgaze/tta_loss_driver.log 2>&1 &
set -u
cd /data1/xieliangcai/code/gazeformer-exps
source /data1/xieliangcai/conda/etc/profile.d/conda.sh
conda activate dl

LOG=out/rgbdgaze/analysis/tta_loss_ablation.log
CKPT=out/rgbdgaze/ablation/checkpoints/eye224_mgpu/best.pt
COMMON="--ckpt $CKPT --split test --resolution 224 --use-eye --no-geom --steps 100 --lr 1e-3 --rank 8 --batch-size 128"

run() {  # name w_moment w_photo w_geo
  echo "" | tee -a "$LOG"
  echo "=== $1   w_moment=$2 w_photo=$3 w_geo=$4   $(date '+%F %T') ===" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=1 python -u -m scripts.rgbdgaze.tta_personalize $COMMON \
    --w-moment "$2" --w-photo "$3" --w-geo "$4" \
    --out "out/rgbdgaze/analysis/tta_loss_$1.csv" >> "$LOG" 2>&1
  echo "--- $1 结束 $(date '+%F %T')" | tee -a "$LOG"
}

run ours     1 0 0
run photo    0 1 0
run geo      0 0 1
run oursgeo  1 0 1
run all      1 1 1

echo "[ablation done] $(date '+%F %T')" | tee -a "$LOG"
