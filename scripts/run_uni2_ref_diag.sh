#!/bin/bash
# UNI2+MLP 改进诊断（对照实验，隔离各改动效果）：
#   实验1：基线 MLPHead + LN 特征 + log1p_zscore（隔离 LN 提取效果，目标空间与基线一致）
#   实验2：ref MLP(无 Softplus) + LN 特征 + log1p_zscore（隔离 LN+MLP 架构效果）
# 对比基线 0.3240（原始 X_uni2 + MLPHead + log1p_zscore）。
# 用法：nohup bash scripts/run_uni2_ref_diag.sh > logs/uni2_ref_diag.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
mkdir -p outputs logs
LOG=logs/uni2_ref_diag.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "==== UNI2+MLP 诊断开始 ===="

# 实验1：基线 MLPHead + LN 特征
D1=outputs/bench_uni2_mlp_ln_unf
if [ ! -f "$D1/best.pt" ]; then
  say "== 实验1: 基线MLP + LN特征 + log1p_zscore 训练 =="
  python3 scripts/train.py --method uni2_mlp --feature_file X_uni2_ln.npy \
    --train_dir data/rep1 --valid_dir data/rep2 --gene_norm log1p_zscore \
    --epochs 50 --patience 10 --lr 1e-3 --batch_size 2048 \
    --output_dir "$D1" --device cuda >> "$LOG" 2>&1
fi
say "== 实验1 测试 =="
python3 scripts/test.py --method uni2_mlp --feature_file X_uni2_ln.npy \
  --ckpt "$D1/best.pt" --test_dir data/rep2 --gene_norm log1p_zscore \
  --output_dir "$D1" --device cuda >> "$LOG" 2>&1

# 实验2：ref MLP(无 Softplus) + LN 特征
D2=outputs/bench_uni2_mlp_refnosoft_unf
if [ ! -f "$D2/best.pt" ]; then
  say "== 实验2: ref MLP(无Softplus) + LN特征 + log1p_zscore 训练 =="
  python3 scripts/train.py --method uni2_mlp --variant ref --no_softplus \
    --feature_file X_uni2_ln.npy \
    --train_dir data/rep1 --valid_dir data/rep2 --gene_norm log1p_zscore \
    --epochs 50 --patience 10 --lr 1e-3 --batch_size 2048 \
    --output_dir "$D2" --device cuda >> "$LOG" 2>&1
fi
say "== 实验2 测试 =="
python3 scripts/test.py --method uni2_mlp --variant ref --no_softplus \
  --feature_file X_uni2_ln.npy \
  --ckpt "$D2/best.pt" --test_dir data/rep2 --gene_norm log1p_zscore \
  --output_dir "$D2" --device cuda >> "$LOG" 2>&1

say "==== 诊断完成 ===="
