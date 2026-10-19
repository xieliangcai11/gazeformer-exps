#!/usr/bin/env bash
# 学习率扫描：**等损失对照跑完**后自动接上（用户判断 lr=1e-3 太大 ⇒ 换小再跑 ✓）
#
# 证据（本仓库日志）：多次出现 best@0（第一步就把无监督损失弄差 ✗）、best@20 之后 val 一路升 ✗、
# 大量会话直接回退 ✗ ⇒ 典型"步子太大"。故在**同一配置**（矩匹配 + 光度一致性，
# 即产出 −12.1% 的那套）下只换学习率与步数。
#
# 三个臂（GPU 1；每臂约 45 分钟，共 ~2.2 小时）：
#   lr=3e-4 steps=300
#   lr=1e-4 steps=300
#   lr=3e-5 steps=300
# 观察点：① EM 增益 ② **回退会话数**（学习率合适时应显著减少 ✓）③ 被选中的最佳步分布
#
# 用法（detached）：
#   nohup bash scripts/rgbdgaze/tta_lr_sweep.sh > out/rgbdgaze/tta_lr_driver.log 2>&1 &
set -u
cd /data1/xieliangcai/code/gazeformer-exps
source /data1/xieliangcai/conda/etc/profile.d/conda.sh
conda activate dl

LOG=out/rgbdgaze/analysis/tta_lr_sweep.log
CKPT=out/rgbdgaze/ablation/checkpoints/eye224_mgpu/best.pt

echo "[lr-sweep] 等损失对照结束（等 tta_loss_ablation.log 出现 [ablation done]）..." | tee -a "$LOG"
for _ in $(seq 1 480); do
  grep -q "\[ablation done\]" out/rgbdgaze/analysis/tta_loss_ablation.log 2>/dev/null && break
  sleep 60
done
echo "[lr-sweep] 开始 $(date '+%F %T')" | tee -a "$LOG"

run() {  # tag lr steps
  echo "" | tee -a "$LOG"
  echo "=== lr=$2 steps=$3  $(date '+%F %T') ===" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=1 python -u -m scripts.rgbdgaze.tta_personalize \
    --ckpt "$CKPT" --split test --resolution 224 --use-eye --no-geom \
    --rank 8 --batch-size 128 --w-moment 1 --w-photo 1 --w-geo 0 \
    --lr "$2" --steps "$3" --out "out/rgbdgaze/analysis/tta_lr_$1.csv" >> "$LOG" 2>&1
  echo "--- lr=$2 结束 $(date '+%F %T')" | tee -a "$LOG"
}

run 3e-4 3e-4 300
run 1e-4 1e-4 300
run 3e-5 3e-5 300

echo "[lr sweep done] $(date '+%F %T')" | tee -a "$LOG"
