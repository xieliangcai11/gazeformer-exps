#!/usr/bin/env bash
# 跨模型泛化探针队列（等前序作业结束后自动接上 ✓）
#
# 为什么：C1（无监督准则与真误差反向/无关）目前只有 1 个模型（eye224_mgpu）的证据 ✗。
# 已有泛化臂只证明了"旧梯度法可换人/换模型"（tta_val.csv −4.3% / tta_noeye_test.csv −7.9%），
# 并未把**新结论**放到别的模型上验证 ⇒ 本队列补这一条轴（普适性最便宜的一步 ✓）。
#
# 每个模型跑 8 个会话（p11/p33 各 4 个活动 ✓）的 --no-early-stop 探针：
#   · 给出该模型上的"候选准则 vs 真误差"秩相关 ⇒ 检验 C1 是否跨模型成立
#   · 给出该模型上"关掉选择"的 EM ⇒ 检验修法是否跨模型有效
set -u
cd E:/code/gazeformer-exps
P="D:/conda/envs/dl/python.exe"
LOG=out/rgbdgaze/analysis/tta_probe_crossmodel.log
running() {  # 是否存在名为 python.exe 且命令行含给定串的进程（避免匹配自身 ✗）
  local n
  n=$(powershell -NoProfile -Command "(Get-CimInstance Win32_Process | Where-Object { \$_.Name -eq 'python.exe' -and \$_.CommandLine -like '*$1*' }).Count" 2>/dev/null | tr -d ' \r')
  [ "${n:-0}" != "0" ]
}
echo "[driver] 启动 $(date '+%F %T')，等待前序 tta_probe_noes 结束（最多 6 小时）..." | tee -a "$LOG"
for _ in $(seq 1 360); do running "tta_probe_noes" || break; sleep 60; done
echo "[driver] 前序结束 $(date '+%F %T')" | tee -a "$LOG"

run() {  # tag ckpt extra...
  tag=$1; ck=$2; shift 2
  echo "" | tee -a "$LOG"; echo "=== $tag  $(date '+%F %T') ===" | tee -a "$LOG"
  "$P" -u -m scripts.rgbdgaze.tta_personalize \
      --corr grad --no-early-stop --probe-true-em --num-workers 0 \
      --ckpt "$ck" --split test --use-eye --no-geom --steps 300 --lr 1e-3 --rank 8 \
      --adapt-frac 1.0 --sessions "p11/decoded,p33/decoded" "$@" \
      --out "out/rgbdgaze/analysis/tta_probe_$tag.csv" >> "$LOG" 2>&1
  echo "--- $tag 结束 rc=$? $(date '+%F %T')" | tee -a "$LOG"
}

# ② 换模型：无眼支流（结构不同 ✓ ⇒ 换架构轴）
run xmodel_noeye out/rgbdgaze/ablation/checkpoints/noeye224_ctl/best.pt \
    --resolution 224 --no-eye
# ②′ 换分辨率（输入几何/感受野改变 ✓ ⇒ 换尺度轴）
run xmodel_448 out/rgbdgaze/ablation/checkpoints/eye448_v1/best.pt --resolution 448

echo "[driver] 全部完成 $(date '+%F %T')" | tee -a "$LOG"
