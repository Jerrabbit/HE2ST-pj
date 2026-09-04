#!/bin/bash
# "未过滤"rep1→rep2 全评测轮次真正缺失的 3 个方法：GHIST / BLEEP(冻结) / SQUALL(解码器头)。
# （Hist2ST 官方配置 unf 已完成 bench_hist2st_unf=0.1818；Phoenix 微调已有 0.2055 测试。）
# 严格串行：每个 训练→测试 完成后才下一个；GHIST 失败重试 1 次。
# 跳过任何已产出 test_results.json 的方法（幂等）。
# 用法：nohup bash scripts/run_pending_unf.sh > logs/pending_unf.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
mkdir -p outputs logs
DEV="cuda"
LOG=logs/pending_unf.log
GN=log1p_zscore
EPOCHS=50; PATIENCE=10; LR=1e-3; BS=2048
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "==== pending-unf 开始（SQUALL-decoder / GHIST / BLEEP，串行）===="

# ---------- 1) SQUALL 解码器头（unfiltered token） ----------
m=squall_decoder; D=outputs/bench_${m}_unf
if [ -f "$D/test_results.json" ]; then
  say "== squall(decoder) unf 已有结果，跳过"
else
  say "== squall(decoder) unf train =="
  python3 scripts/train.py --method squall --variant decoder \
    --feature_file X_squall_tokens.npy \
    --train_dir data/rep1 --valid_dir data/rep2 --epochs $EPOCHS --patience $PATIENCE \
    --batch_size 256 --lr $LR --gene_norm $GN \
    --output_dir "$D" --device $DEV > logs/bench_squall_decoder_unf_train.log 2>&1
  if [ -f "$D/best.pt" ]; then
    say "== squall(decoder) unf test =="
    python3 scripts/test.py --method squall --variant decoder \
      --feature_file X_squall_tokens.npy \
      --ckpt "$D/best.pt" --test_dir data/rep2 \
      --gene_norm $GN --batch_size 256 --output_dir "$D" --device $DEV >> logs/bench_squall_decoder_unf_train.log 2>&1
  else
    say "!! squall(decoder) unf 训练失败（无 best.pt）"
  fi
fi

# ---------- 2) GHIST（unfiltered，失败重试 1 次） ----------
m=ghist; D=outputs/bench_${m}_unf
if [ -f "$D/test_results.json" ]; then
  say "== ghist unf 已有结果，跳过"
else
  for attempt in 1 2; do
    say "== ghist unf train（attempt $attempt/2）=="
    python3 scripts/train.py --method $m --train_dir data/ghist_rep1 --valid_dir data/ghist_rep2 \
      --epochs $EPOCHS --patience $PATIENCE --lr $LR --output_dir "$D" --device $DEV >> logs/bench_${m}_unf_train.log 2>&1
    if [ -f "$D/best.pt" ]; then
      say "== ghist unf test =="
      python3 scripts/test_ghist.py --ckpt "$D/best.pt" --test_dir data/ghist_rep2 \
        --output_dir "$D" --device $DEV >> logs/bench_${m}_unf_train.log 2>&1
      break
    else
      say "!! ghist unf 训练失败（attempt $attempt），5 分钟后重试"
      sleep 300
    fi
  done
fi

# ---------- 3) BLEEP（冻结 resnet50，unfiltered） ----------
m=bleep; D=outputs/bench_${m}_unf
if [ -f "$D/test_results.json" ]; then
  say "== bleep unf 已有结果，跳过"
else
  say "== bleep(冻结) unf train =="
  python3 scripts/train.py --method $m --no_finetune --pretrained_weights weights/resnet50_imagenet.pth \
    --train_dir data/rep1 --valid_dir data/rep2 --epochs $EPOCHS --patience $PATIENCE \
    --batch_size $BS --lr $LR --gene_norm $GN \
    --output_dir "$D" --device $DEV > logs/bench_${m}_unf_train.log 2>&1
  if [ -f "$D/best.pt" ]; then
    say "== bleep(冻结) unf test =="
    python3 scripts/test.py --method $m --img_size 224 --pretrained_weights weights/resnet50_imagenet.pth \
      --ckpt "$D/best.pt" --test_dir data/rep2 --gene_norm $GN \
      --output_dir "$D" --device $DEV >> logs/bench_${m}_unf_train.log 2>&1
  else
    say "!! bleep(冻结) unf 训练失败（无 best.pt）"
  fi
fi

say "==== pending-unf 全部结束 ===="
