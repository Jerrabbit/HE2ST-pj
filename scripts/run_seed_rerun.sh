#!/bin/bash
# 固定种子复跑 BLEEP/Hist2ST（unfiltered rep1→rep2），核查旧 0.2131/0.2139 的可复现区间。
# 顺序：BLEEP(冻结,统一协议, 用 /tmp 快盘副本) → Hist2ST(官方配置)。SEEDS 默认 "0 1"。
# 用法：nohup bash scripts/run_seed_rerun.sh > logs/seed_rerun.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
mkdir -p outputs logs
LOG=logs/seed_rerun.log
say() { echo "[$(date -u '+%F %T')] $*" | tee -a "$LOG"; }
SEEDS="${SEEDS:-0 1}"

say "==== seed 复跑开始（SEEDS=$SEEDS）===="

# ---------- BLEEP(冻结) unf，统一协议，/tmp 快盘 ----------
for S in $SEEDS; do
  D=outputs/bench_bleep_unf_s$S
  if [ -f "$D/test_results.json" ]; then say "== bleep unf s$S 已有结果，跳过 =="; continue; fi
  say "== bleep unf seed=$S train（统一协议，/tmp）=="
  python3 scripts/train.py --method bleep --no_finetune --pretrained_weights weights/resnet50_imagenet.pth \
    --train_dir /tmp/bl_rep1 --valid_dir /tmp/bl_rep2 \
    --epochs 50 --patience 10 --batch_size 2048 --lr 1e-3 --gene_norm log1p_zscore \
    --seed $S --output_dir "$D" --device cuda > logs/bench_bleep_unf_s${S}_train.log 2>&1
  if [ -f "$D/best.pt" ]; then
    say "== bleep unf seed=$S test =="
    python3 scripts/test.py --method bleep --img_size 224 --pretrained_weights weights/resnet50_imagenet.pth \
      --ckpt "$D/best.pt" --test_dir /tmp/bl_rep2 --gene_norm log1p_zscore \
      --output_dir "$D" --device cuda >> logs/bench_bleep_unf_s${S}_train.log 2>&1
  else
    say "!! bleep unf seed=$S 训练失败（无 best.pt）"
  fi
done

# ---------- Hist2ST(官方配置) unf ----------
for S in $SEEDS; do
  D=outputs/bench_hist2st_unf_s$S
  if [ -f "$D/test_results.json" ]; then say "== hist2st unf s$S 已有结果，跳过 =="; continue; fi
  say "== hist2st unf seed=$S train（官方配置）=="
  python3 scripts/train.py --method hist2st \
    --train_dir data/rep1 --valid_dir data/rep2 \
    --epochs 100 --lr 1e-5 --zinb 0.25 --zinb_coef 0.25 --bake 5 --lamb 0.5 \
    --seed $S --output_dir "$D" --device cuda > logs/bench_hist2st_unf_s${S}_train.log 2>&1
  if [ -f "$D/best.pt" ]; then
    say "== hist2st unf seed=$S test =="
    python3 scripts/test_hist2st.py --ckpt "$D/best.pt" --test_dir data/rep2 \
      --output_dir "$D" --device cuda >> logs/bench_hist2st_unf_s${S}_train.log 2>&1
  else
    say "!! hist2st unf seed=$S 训练失败（无 best.pt）"
  fi
done

say "==== seed 复跑全部结束 ===="
