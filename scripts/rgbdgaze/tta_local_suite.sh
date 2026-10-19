#!/usr/bin/env bash
# 本机（RTX 2080 Ti / Windows msys）TTA 证据套件：损失消融 5 臂 + 学习率扫描 3 臂。
#
# 为什么需要本机版：服务器版 scripts/rgbdgaze/tta_{loss_ablation,lr_sweep}.sh 硬编码了
#   /data1/... 路径与 conda activate，且 CUDA 目前整节点故障 ⇒ 无法执行。
#
# 与该服务器历史 run 的可比性（已验证 ✓）：本机在完全相同的 ckpt/划分/口径下，
#   逐会话 em_base 与服务器 CSV 的最大绝对差 = 3.16e-07（20/20 会话一致）。
#
# 关键修复（本套件能跑起来的前提）：tta_personalize.py 之前漏定义 --num-workers
#   ⇒ 服务器上 5 臂损失消融与 3 臂 lr 扫描全部在开跑前 AttributeError 崩溃。
#   Windows 上 num_workers>0 会因 spawn 无法 pickle 局部 _worker_init ⇒ 这里固定传 0。
#
# 用法（在项目根，或直接 bash scripts/rgbdgaze/tta_local_suite.sh）：
#   bash scripts/rgbdgaze/tta_local_suite.sh
# 产物：out/rgbdgaze/analysis/tta_local_<arm>.csv      （逐会话）
#       out/rgbdgaze/analysis/ttadump_<arm>.csv       （逐帧：base/TTA 预测、误差、帧号）
set -u
cd "$(dirname "$0")/../.." || exit 1
# ⚠️ 脚本在真 bash 里只认 /d/...，工具调用里认 D:/... ⇒ 自动择一（曾误用致 rc=127 ✗）
PY=/d/conda/envs/dl/python.exe
[ -x "$PY" ] || PY="D:/conda/envs/dl/python.exe"
P="$PY"
export CUDA_VISIBLE_DEVICES=0
LOG=out/rgbdgaze/analysis/tta_local_suite.log
CKPT=out/rgbdgaze/ablation/checkpoints/eye224_mgpu/best.pt
COMMON="--ckpt $CKPT --split test --resolution 224 --use-eye --no-geom --num-workers 0 --rank 8 --batch-size 128 --adapt-frac 1.0"

echo "[local suite] 开始 $(date '+%F %T')  ckpt=$CKPT" | tee -a "$LOG"

run() {  # run <tag> <extra args...>
  tag=$1; shift
  echo "" | tee -a "$LOG"
  echo "=== $tag   $(date '+%F %T') ===" | tee -a "$LOG"
  "$P" -u -m scripts.rgbdgaze.tta_personalize $COMMON "$@" \
      --out "out/rgbdgaze/analysis/tta_local_$tag.csv" \
      --dump-preds "out/rgbdgaze/analysis/ttadump_$tag.csv" >> "$LOG" 2>&1
  echo "--- $tag 结束 rc=$? $(date '+%F %T')" | tee -a "$LOG"
}

# ---- A. 损失消融：只换无监督损失，其余完全相同（100 步）----
# 目的：判定"输出矩匹配"（我们的）相对文献同族的增强一致性（光度/几何）是否有优势。
run loss_ours    --w-moment 1 --w-photo 0 --w-geo 0 --steps 100
run loss_photo   --w-moment 0 --w-photo 1 --w-geo 0 --steps 100
run loss_geo     --w-moment 0 --w-photo 0 --w-geo 1 --steps 100
run loss_oursgeo --w-moment 1 --w-photo 0 --w-geo 1 --steps 100
run loss_all     --w-moment 1 --w-photo 1 --w-geo 0 --steps 100

# ---- B. 学习率扫描：主配置（矩匹配+光度）下只换 lr 与步数（300 步）----
# 观察点：EM 增益、**回退会话数**（lr 合适应显著减少）、被选中的最佳步分布。
run lr3e4 --w-moment 1 --w-photo 1 --w-geo 0 --steps 300 --lr 3e-4
run lr1e4 --w-moment 1 --w-photo 1 --w-geo 0 --steps 300 --lr 1e-4
run lr3e5 --w-moment 1 --w-photo 1 --w-geo 0 --steps 300 --lr 3e-5

echo "[local suite done] $(date '+%F %T')" | tee -a "$LOG"
