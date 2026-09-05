#!/bin/bash
# GHIST unf 重试：小 batch（8）+ 可扩展显存分配，规避 80GB 单卡 OOM。
# 背景：run_pending_unf.sh 里 GHIST 用了通用 train.py 的大默认 batch(2048) → 单步 2048 个
# 256×256 patch 爆显存；GHIST train_function 用 getattr(args,'batch_size',8)，显式给 8 即可。
# 用法：cd 项目根 && nohup bash scripts/run_ghist_unf_retry.sh > logs/ghist_unf_retry.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
mkdir -p outputs logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=logs/ghist_unf_retry.log
say() { echo "[$(date -u '+%F %T')] $*" | tee -a "$LOG"; }
D=outputs/bench_ghist_unf

if [ -f "$D/test_results.json" ]; then
  say "== GHIST unf 已有 test_results.json，跳过 =="
  exit 0
fi

for attempt in 1 2 3; do
  say "== GHIST unf train（attempt $attempt/3，batch=8）=="
  python3 scripts/train.py --method ghist \
    --train_dir data/ghist_rep1 --valid_dir data/ghist_rep2 \
    --epochs 50 --patience 10 --lr 1e-3 --batch_size 8 \
    --output_dir "$D" --device cuda >> logs/bench_ghist_unf_train.log 2>&1
  if [ -f "$D/best.pt" ]; then
    say "== GHIST unf test =="
    python3 scripts/test_ghist.py --ckpt "$D/best.pt" --test_dir data/ghist_rep2 \
      --output_dir "$D" --device cuda >> logs/bench_ghist_unf_train.log 2>&1
    say "== GHIST unf done =="
    exit 0
  else
    say "!! GHIST unf 训练失败（attempt $attempt），5 分钟后重试"
    sleep 300
  fi
done
say "== GHIST unf 3 次尝试均失败 =="
