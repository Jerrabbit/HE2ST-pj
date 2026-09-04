#!/bin/bash
# Phoenix 280 就地微调训练完成后：① rep2 全量 test → ② run_pending_unf.sh（缺口方法）。
# 用法（远程）：cd 项目根 && nohup bash scripts/run_after_phoenix.sh > logs/phoenix_after_train.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
mkdir -p outputs logs
say() { echo "[$(date -u '+%F %T')] $*"; }

say "== Phoenix 280 全量 test（rep2 111k，ODE 慢）开始 =="
python3 scripts/phoenix_official280_finetune.py test \
  --test_dir data/rep2 --ckpt outputs/bench_phoenix_official280_ft/best.pt \
  --output_dir outputs/bench_phoenix_official280_ft \
  > logs/full_phoenix280_ft_test.log 2>&1
code=$?
say "== Phoenix 280 全量 test 结束 exit=$code =="

say "== 启动 pending-unf 链（SQUALL-decoder/GHIST/BLEEP）=="
bash scripts/run_pending_unf.sh > logs/pending_unf.log 2>&1
say "== pending-unf 链结束 =="
