#!/bin/bash
# SQUALL 官方解码器头单独重跑：读 /tmp 缓存的 squall tokens（避开 cpfs 随机读 D-state 卡死）。
# 用法：nohup bash scripts/run_squall_retry.sh > logs/squall_retry.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
TR=/tmp/squall_tokens/rep1_f/X_squall_tokens.npy
VA=/tmp/squall_tokens/rep2_f/X_squall_tokens.npy
D=outputs/bench_squall_decoder_f

if [ ! -f "$TR" ] || [ ! -f "$VA" ]; then
  echo "!! 缺少 /tmp squall tokens，先复制：cp data/rep{1,2}_f/X_squall_tokens.npy /tmp/squall_tokens/rep{1,2}_f/"
  exit 1
fi

echo "[$(date '+%F %T')] == squall decoder train (读 /tmp) =="
python scripts/train.py --method squall --variant decoder \
  --feature_file "$TR" \
  --train_dir data/rep1_f --valid_dir data/rep2_f \
  --epochs 50 --patience 10 --batch_size 256 --lr 1e-3 --gene_norm log1p_zscore \
  --output_dir "$D" --device cuda
echo "[$(date '+%F %T')] == squall decoder test =="
python scripts/test.py --method squall --variant decoder \
  --feature_file "$VA" \
  --ckpt "$D/best.pt" --test_dir data/rep2_f \
  --gene_norm log1p_zscore --batch_size 256 --output_dir "$D" --device cuda
echo "[$(date '+%F %T')] SQUALL 重跑完成"
