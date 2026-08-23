#!/bin/bash
# 严格串行跑 4 个缺失方法（squall/ghist/st_net/bleep），每个完成后再下一个，
# 避免 bench 多 pass 的孤儿进程/OOM 问题。
set -u
cd "$(dirname "$0")/.."
DEV="cuda"
LOG=logs/missing_seq.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# 1. SQUALL 解码器（cpfs tokens；之前孤儿已能训到 ep4）
say "== squall =="
python scripts/train.py --method squall --variant decoder \
  --feature_file data/rep1_f/X_squall_tokens.npy \
  --train_dir data/rep1_f --valid_dir data/rep2_f --epochs 50 --patience 10 \
  --batch_size 256 --lr 1e-3 --gene_norm log1p_zscore \
  --output_dir outputs/bench_squall_decoder_f --device $DEV >> "$LOG" 2>&1
python scripts/test.py --method squall --variant decoder \
  --feature_file data/rep2_f/X_squall_tokens.npy \
  --ckpt outputs/bench_squall_decoder_f/best.pt --test_dir data/rep2_f \
  --gene_norm log1p_zscore --batch_size 256 --output_dir outputs/bench_squall_decoder_f --device $DEV >> "$LOG" 2>&1

# 2. GHIST（KeyError 已修；GPU 干净不再 OOM）
say "== ghist =="
python scripts/train.py --method ghist --train_dir data/ghist_rep1_f --valid_dir data/ghist_rep2_f \
  --epochs 50 --patience 10 --lr 1e-3 --output_dir outputs/bench_ghist_f --device $DEV >> "$LOG" 2>&1
python scripts/test_ghist.py --ckpt outputs/bench_ghist_f/best.pt --test_dir data/ghist_rep2_f \
  --output_dir outputs/bench_ghist_f --device $DEV >> "$LOG" 2>&1

# 3. ST-Net（/tmp patches）
say "== st_net =="
python scripts/train.py --method st_net --no_finetune \
  --train_dir data/rep1_f --valid_dir data/rep2_f --epochs 50 --patience 10 \
  --batch_size 2048 --lr 1e-3 --gene_norm log1p_zscore \
  --output_dir outputs/bench_st_net_f --device $DEV >> "$LOG" 2>&1
python scripts/test.py --method st_net --ckpt outputs/bench_st_net_f/best.pt --test_dir data/rep2_f \
  --gene_norm log1p_zscore --output_dir outputs/bench_st_net_f --device $DEV >> "$LOG" 2>&1

# 4. BLEEP（/tmp patches）
say "== bleep =="
python scripts/train.py --method bleep --no_finetune --pretrained_weights weights/resnet50_imagenet.pth \
  --train_dir data/rep1_f --valid_dir data/rep2_f --epochs 50 --patience 10 \
  --batch_size 2048 --lr 1e-3 --gene_norm log1p_zscore \
  --output_dir outputs/bench_bleep_f --device $DEV >> "$LOG" 2>&1
python scripts/test.py --method bleep --img_size 224 --pretrained_weights weights/resnet50_imagenet.pth \
  --ckpt outputs/bench_bleep_f/best.pt --test_dir data/rep2_f \
  --gene_norm log1p_zscore --output_dir outputs/bench_bleep_f --device $DEV >> "$LOG" 2>&1

say "== 4 方法全部完成 =="
