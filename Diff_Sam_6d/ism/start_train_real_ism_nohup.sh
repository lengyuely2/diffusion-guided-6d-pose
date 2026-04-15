#!/usr/bin/env bash
# 后台跑 train_real ISM，日志写入 diff/output。
set -euo pipefail
DIFF_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="${ISM_NOHUP_LOG:-$DIFF_ROOT/diff/output/ism_train_real_ism.log}"
PID_FILE="${ISM_NOHUP_PID:-${LOG%.log}.pid}"
mkdir -p "$(dirname "$LOG")"
cd "$DIFF_ROOT"
nohup bash ism/run_ycbv_train_real_ism.sh "$@" >>"$LOG" 2>&1 &
echo $! >"$PID_FILE"
echo "已启动 ISM 后台任务 PID=$(cat "$PID_FILE")"
echo "日志: $LOG"
echo "看进度: tail -f $LOG"
