#!/usr/bin/env bash
# 【源域被试】探针队列：为"学适配策略"采集训练数据（等前序作业结束后自动接上 ✓）
#
# 为什么要源域被试：策略必须**在源域上学**（源域真值仅离线用于训练 ✓），再在**目标被试**上
# 验证（测试时零真值零标签 ✓✓）。当前探针只跑了测试被试（p11/p33）⇒ 无法做留一被试 ✗。
#
# 用 --split train 取训练划分的会话（p1/p10/p5 等 ✓），每会话 300 步 + 每 10 步探针 ✓。
set -u
cd E:/code/gazeformer-exps
P="D:/conda/envs/dl/python.exe"
CK=out/rgbdgaze/ablation/checkpoints/eye224_mgpu/best.pt
LOG=out/rgbdgaze/analysis/tta_probe_source.log
running() {
  local n
  n=$(powershell -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*$1*' }).Count" 2>/dev/null | tr -d ' \r')
  [ "${n:-0}" != "0" ]
}
echo "[driver] 等待前序 tta_probe_noes 结束（最多 6 小时）$(date '+%F %T')" | tee -a "$LOG"
for _ in $(seq 1 360); do running "tta_probe_noes" || break; sleep 60; done
echo "[driver] 前序结束，开始源域探针 $(date '+%F %T')" | tee -a "$LOG"

run() {  # tag sessions-子串
  tag=$1; sess=$2
  echo "" | tee -a "$LOG"; echo "=== $tag  $(date '+%F %T') ===" | tee -a "$LOG"
  "$P" -u -m scripts.rgbdgaze.tta_personalize \
      --corr grad --no-early-stop --probe-true-em --num-workers 0 \
      --ckpt "$CK" --split train --resolution 224 --use-eye --no-geom \
      --steps 300 --lr 1e-3 --rank 8 --adapt-frac 1.0 --sessions "$sess" \
      --out "out/rgbdgaze/analysis/tta_probe_src_$tag.csv" >> "$LOG" 2>&1
  echo "--- $tag 结束 rc=$? $(date '+%F %T')" | tee -a "$LOG"
}

# 两个源域被试 × 两个活动（= 4 会话，每个 ~10–15 分钟 ✓）
run p1  "p1/decoded/sitting,p1/decoded/walking"
run p10 "p10/decoded/sitting,p10/decoded/walking"

echo "[driver] 源域探针全部完成 $(date '+%F %T')" | tee -a "$LOG"
