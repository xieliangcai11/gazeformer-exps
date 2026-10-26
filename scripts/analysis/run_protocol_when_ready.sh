#!/usr/bin/env bash
# 等 5 折 CV 产出齐全且 worker 退出后，自动跑“源域选参 + 测试集一次评估”。
# 用法：setsid nohup bash scripts/analysis/run_protocol_when_ready.sh >/dev/null 2>&1 &
set -u
cd /data1/xieliangcai/code/gazeformer-exps
source /data1/xieliangcai/conda/etc/profile.d/conda.sh
conda activate dl

D=out/rgbdgaze/analysis/context
LOG=$D/protocol_watch.log
echo "[protocol-watch] $(date +%F_%T) 启动" >> "$LOG"

for i in $(seq 1 120); do
  n=$(ls "$D"/cv_fold*.csv 2>/dev/null | wc -l)
  if [ "$n" -ge 5 ] && ! pgrep -f "[r]un_group_cv" >/dev/null; then
    echo "[protocol-watch] CV 完成（$n 折），开始协议" >> "$LOG"
    python -u -m scripts.analysis.c3_source_protocol \
      --cv-glob "$D/cv_fold*.csv" \
      --test out/rgbdgaze/analysis/preds_eye224_mgpu_test.csv >> "$LOG" 2>&1
    echo "[protocol-watch] 协议结束 $(date +%F_%T)" >> "$LOG"
    exit 0
  fi
  sleep 180
done
echo "[protocol-watch] 超时退出" >> "$LOG"
