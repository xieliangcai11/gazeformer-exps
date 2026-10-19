#!/usr/bin/env bash
# 探针 v2 队列：为新特征（cons / sigad / dmag / bright / sharp / jvar）采集**源域 + 目标域**数据
# 为什么：策略必须**在源域被试上学**、在**目标域被试上评**（测试时零标签 ✓），所以两边都要采 ✓。
#   源域被试（--split train）：p1 / p10
#   目标域被试（--split test） ：p11 / p33
# 每被试 2 个活动（sitting / walking）⇒ 共 8 会话 ×（300 步 + 每 10 步探针）。
set -u
# 用相对定位 ✓：脚本在 WSL 里 cd 到 "E:/..." 会失败 ✗，必须走 dirname ✓
cd "$(dirname "$0")/../.." || exit 1
# ⚠️ 路径/进程可见性依上下文而不同（实测）：
#   · 工具调用里的 bash 认 D:/...（我的 python 启动都靠它 ✓）
#   · 脚本里的 bash 是 WSL ⇒ 只认 /mnt/d/...（/d/... 与 D:/... 均 rc=127 ✗）
#   · WSL 的 ps 看不到 Windows 进程 ⇒ 不能用进程探测等前序作业，改用**产物文件**等待 ✓
for cand in /mnt/d/conda/envs/dl/python.exe /d/conda/envs/dl/python.exe D:/conda/envs/dl/python.exe; do
  [ -x "$cand" ] && P="$cand" && break
done
: "${P:?找不到 python 解释器 ✗}"
CK=out/rgbdgaze/ablation/checkpoints/eye224_mgpu/best.pt
LOG=out/rgbdgaze/analysis/tta_probe_v2.log
WAIT_FOR=out/rgbdgaze/analysis/tta_probe_noes_8sess.csv      # 前序作业的产物（出现即结束 ✓）
echo "[driver v2] 解释器=$P" | tee -a "$LOG"
echo "[driver v2] 等前序产物 $WAIT_FOR（最多 6h）$(date '+%F %T')" | tee -a "$LOG"
for _ in $(seq 1 360); do [ -f "$WAIT_FOR" ] && break; sleep 60; done
echo "[driver v2] 开始 $(date '+%F %T')" | tee -a "$LOG"

run() {  # tag split sessions
  tag=$1; split=$2; sess=$3
  echo "" | tee -a "$LOG"; echo "=== $tag (split=$split) $(date '+%F %T') ===" | tee -a "$LOG"
  "$P" -u -m scripts.rgbdgaze.tta_personalize \
      --corr grad --no-early-stop --probe-true-em --num-workers 0 \
      --ckpt "$CK" --split "$split" --resolution 224 --use-eye --no-geom \
      --steps 300 --lr 1e-3 --rank 8 --adapt-frac 1.0 --sessions "$sess" \
      --out "out/rgbdgaze/analysis/tta_probe_v2_$tag.csv" >> "$LOG" 2>&1
  echo "--- $tag 结束 rc=$? $(date '+%F %T')" | tee -a "$LOG"
}

# 源域（策略训练集 ✓ —— 真值仅离线使用 ✓）
run src_p1  train "p1/decoded/sitting,p1/decoded/walking"
run src_p10 train "p10/decoded/sitting,p10/decoded/walking"
# 目标域（策略评估 ✓ —— 测试时零标签 ✓）
run tgt_p11 test  "p11/decoded/sitting,p11/decoded/walking"
run tgt_p33 test  "p33/decoded/sitting,p33/decoded/walking"

echo "[driver v2] 全部完成 $(date '+%F %T')" | tee -a "$LOG"
