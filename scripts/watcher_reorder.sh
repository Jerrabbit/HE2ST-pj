#!/bin/bash
# 在 st_net 结束后自动重排：杀掉主脚本 run_bench_unf.sh，启动 v2（短方法先跑）。
# 触发条件：主日志出现 "== hist2st official train =="（= st_net 已训练+测试完，
# 主脚本正要开始慢的 hist2st）—— 此时 kill 掉主脚本与刚启动的 hist2st，改跑 v2。
# 兜底：主脚本意外退出时也启动 v2。
# 用法：nohup bash scripts/watcher_reorder.sh > logs/watcher_reorder.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
LOG=logs/bench_unf_all.log
say() { echo "[$(date '+%F %T')] $*" | tee -a logs/watcher_reorder.log; }

while true; do
  if grep -q "== hist2st official train ==" "$LOG" 2>/dev/null; then
    say "检测到 st_net 结束（hist2st 将开始），杀掉主脚本与 hist2st，启动 v2 ..."
    pkill -9 -f "[r]un_bench_unf.sh"
    pkill -9 -f "[m]ethod hist2st"
    pkill -9 -f "[s]cripts/train.py --method hist2st"
    sleep 3
    if ! pgrep -f "[r]un_bench_unf_v2.sh" >/dev/null; then
      nohup bash scripts/run_bench_unf_v2.sh > logs/bench_unf_v2.log 2>&1 &
      say "v2 已启动 PID=$!"
    fi
    break
  fi
  if ! pgrep -f "[r]un_bench_unf.sh" >/dev/null 2>&1; then
    say "主脚本已退出（未到 hist2st），直接启动 v2"
    if ! pgrep -f "[r]un_bench_unf_v2.sh" >/dev/null; then
      nohup bash scripts/run_bench_unf_v2.sh > logs/bench_unf_v2.log 2>&1 &
      say "v2 已启动 PID=$!"
    fi
    break
  fi
  sleep 30
done
say "watcher 结束"
