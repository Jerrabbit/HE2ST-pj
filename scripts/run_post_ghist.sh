#!/bin/bash
# GHIST+segCE 完成后（用户确认方案）：① Phoenix 313 微调复用 best.pt 只补"全指标评测"；
# ② BLEEP/Hist2ST 固定 seed 0/1/2 复跑（观察能否回 ~0.21）。顺序串行。
# 用法：nohup bash scripts/run_post_ghist.sh > logs/post_ghist.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
mkdir -p outputs logs
LOG=logs/post_ghist.log
say() { echo "[$(date -u '+%F %T')] $*" | tee -a "$LOG"; }

# ---------- ① Phoenix 313 微调：全指标评测（复用 bench_phoenix_finetune/best.pt） ----------
CKPT=outputs/bench_phoenix_finetune/best.pt
if [ -f "$CKPT" ]; then
  D=outputs/bench_phoenix_finetune_full
  if [ -f "$D/test_results.json" ]; then say "== phoenix313 full 已有结果，跳过 =="; else
    say "== phoenix313 full-eval 测试（ODE，较慢）=="
    TOKARG=""
    if [ -d /tmp/dino_tokens/rep2 ]; then TOKARG="--tokens_dir /tmp/dino_tokens"; fi
    python3 scripts/test_phoenix_finetune.py --ckpt "$CKPT" --test_dir data/rep2 \
      --output_dir "$D" $TOKARG --device cuda > logs/phoenix313_full_test.log 2>&1
    say "== phoenix313 full done =="
  fi
else
  say "!! 缺 $CKPT，跳过 Phoenix 313 全评测"
fi

# ---------- ② BLEEP/Hist2ST 固定 seed 0/1/2 ----------
say "== BLEEP/Hist2ST seed(0 1 2) 复跑 =="
SEEDS="0 1 2" bash scripts/run_seed_rerun.sh
say "==== post-ghist 全部结束 ===="
