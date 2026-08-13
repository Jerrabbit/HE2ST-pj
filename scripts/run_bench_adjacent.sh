#!/bin/bash
# 相邻切片 benchmark：rep1 训练 → rep2 测试（Xenium 乳腺癌相邻切片）
# 统一协议：50 epochs, lr 1e-3, log1p_zscore, 训练集归一化统计量复用（防泄漏）
# 用法：nohup bash scripts/run_bench_adjacent.sh > logs/bench_all.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
mkdir -p outputs logs

EPOCHS=${EPOCHS:-50}
PATIENCE=${PATIENCE:-10}
LR=${LR:-1e-3}
GN=${GN:-log1p_zscore}
BS=${BS:-2048}
DEV="cuda"

echo "[$(date '+%F %T')] bench start: $EPOCHS epochs (patience=$PATIENCE early stop), lr=$LR, gene_norm=$GN" | tee -a logs/bench_all.log

# --- 特征/标准 harness 方法（fit 训练 + evaluate 测试） ---
for m in uni2_mlp phoenix; do
  echo "[$(date '+%F %T')] ==== $m train ====" | tee -a logs/bench_all.log
  python scripts/train.py --method "$m" --train_dir data/rep1 --valid_dir data/rep2 \
    --epochs "$EPOCHS" --patience "$PATIENCE" --batch_size "$BS" --lr "$LR" --gene_norm "$GN" \
    --output_dir "outputs/bench_$m" --device "$DEV" > "logs/bench_${m}_train.log" 2>&1
  echo "[$(date '+%F %T')] ==== $m test ====" | tee -a logs/bench_all.log
  python scripts/test.py --method "$m" --ckpt "outputs/bench_$m/best.pt" \
    --test_dir data/rep2 --gene_norm "$GN" --output_dir "outputs/bench_$m" \
    --device "$DEV" >> "logs/bench_${m}_train.log" 2>&1
done

# --- spatialex（整片超图，自定义训练 + evaluate_slide 评估） ---
echo "[$(date '+%F %T')] ==== spatialex train ====" | tee -a logs/bench_all.log
python scripts/train.py --method spatialex --train_dir data/rep1 --valid_dir data/rep2 \
  --epochs "$EPOCHS" --patience "$PATIENCE" --lr "$LR" --gene_norm "$GN" \
  --output_dir "outputs/bench_spatialex" --device "$DEV" > "logs/bench_spatialex_train.log" 2>&1
echo "[$(date '+%F %T')] ==== spatialex test ====" | tee -a logs/bench_all.log
python scripts/test_spatialex.py --test_dir data/rep2 --checkpoint "outputs/bench_spatialex/best.pt" \
  --gene_norm "$GN" --output_dir "outputs/bench_spatialex" --device "$DEV" >> "logs/bench_spatialex_train.log" 2>&1

# --- hist2st（ROI 图，自定义训练 + evaluate_slide 评估） ---
echo "[$(date '+%F %T')] ==== hist2st train ====" | tee -a logs/bench_all.log
python scripts/train.py --method hist2st --train_dir data/rep1 --valid_dir data/rep2 \
  --epochs "$EPOCHS" --patience "$PATIENCE" --lr "$LR" --gene_norm "$GN" \
  --output_dir "outputs/bench_hist2st" --device "$DEV" > "logs/bench_hist2st_train.log" 2>&1
echo "[$(date '+%F %T')] ==== hist2st test ====" | tee -a logs/bench_all.log
python scripts/test_hist2st.py --ckpt "outputs/bench_hist2st/best.pt" --test_dir data/rep2 \
  --output_dir "outputs/bench_hist2st" --device "$DEV" >> "logs/bench_hist2st_train.log" 2>&1

echo "[$(date '+%F %T')] bench ALL_DONE" | tee -a logs/bench_all.log
