#!/bin/bash
# GHIST OOM 恢复 + 已完成方法测试重跑（SSIM 统一到 log1p 空间，补 CSV）。
# 背景（2026-08-29）：
#   1) GHIST 训练 17:45 因 GPU 被外部进程(PID 37353, 77GB)抢占 CUDA OOM，无 best.pt；
#   2) spatialex 测试在 json 打印处因 gene_ssims(ndarray) 崩溃，CSV 未生成；
#      —— 已修 test_*.py 用 scalar_results 打印，本脚本重跑 spatialex test 补 CSV；
#   3) SSIM 从"各方法自身归一化空间"改为"统一 log1p(counts) 空间"（zscore 会虚高，
#      实测弱预测时 zscore=1.0 vs log1p=0.43）→ 需重跑已完成的 5 方法测试刷新 SSIM。
# 本脚本等主 benchmark(run_bench_unf.sh) 全部完成后再运行，避免 GPU/IO 争抢。
# 用法：nohup bash scripts/recover_ghist_spatialex.sh > logs/recover_ghist_spatialex.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
mkdir -p logs
LOG=logs/recover_ghist_spatialex.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "等待主 benchmark (run_bench_unf.sh) 完成..."
while pgrep -f "run_bench_unf.sh" >/dev/null 2>&1; do sleep 300; done
say "主 benchmark 完成，开始恢复..."

# ---------- 0. 已完成方法测试重跑（刷新 SSIM→log1p 空间 + 补齐 CSV） ----------
retest() { # $1=方法名  $2..=额外参数
  local m=$1; shift
  local D=outputs/bench_${m}_unf
  if [ ! -f "$D/best.pt" ]; then say "!! $m 无 best.pt，跳过"; return; fi
  say "== $m test 重跑（SSIM 刷新）=="
  python3 scripts/test.py --method $m --ckpt "$D/best.pt" --test_dir data/rep2 \
    --gene_norm log1p_zscore --output_dir "$D" --device cuda "$@" >> "$LOG" 2>&1
}
say "--- 重跑已完成的特征类方法测试 ---"
retest uni2_mlp
retest pixel2gene --variant cell --gene_norm log1p
retest path2space
retest deeppt --feature_file X_resnet50.npy --feat_dim 2048

# ---------- 1. spatialex 测试重跑（补 CSV + SSIM 刷新） ----------
D=outputs/bench_spatialex_unf
if [ -f "$D/best.pt" ]; then
  say "== spatialex test 重跑 =="
  python3 scripts/test_spatialex.py --test_dir data/rep2 --checkpoint "$D/best.pt" \
    --gene_norm log1p_zscore --output_dir "$D" --device cuda >> "$LOG" 2>&1
  ls "$D"/eval_metrics*.csv "$D"/topk_curve.csv 2>/dev/null && say "spatialex CSV 已补齐"
else
  say "!! spatialex best.pt 不存在，跳过"
fi

# ---------- 2. GHIST 训练+测试（OOM 恢复，最多 3 次重试） ----------
m=ghist; D=outputs/bench_${m}_unf
for attempt in 1 2 3; do
  if [ -f "$D/best.pt" ]; then say "GHIST best.pt 已存在，跳过"; break; fi
  say "== GHIST train（attempt $attempt/3）=="
  python3 scripts/train.py --method $m --train_dir data/ghist_rep1 --valid_dir data/ghist_rep2 \
    --epochs 50 --patience 10 --lr 1e-3 --output_dir "$D" --device cuda >> "$LOG" 2>&1
  if [ -f "$D/best.pt" ]; then
    say "== GHIST test =="
    python3 scripts/test_ghist.py --ckpt "$D/best.pt" --test_dir data/ghist_rep2 \
      --output_dir "$D" --device cuda >> "$LOG" 2>&1
    say "GHIST 完成：$(ls $D/test_results.json 2>/dev/null)"
    break
  else
    say "!! GHIST 训练失败（可能 GPU 被抢占），10 分钟后重试"
    sleep 600
  fi
done
say "恢复完成"
