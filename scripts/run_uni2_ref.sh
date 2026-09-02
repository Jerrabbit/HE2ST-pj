#!/bin/bash
# UNI2+MLP 改进验证（2026-09-02）：
#   ① 提取 LN 版 CLS 特征 X_uni2_ln.npy（l1=256 与基线一致，--layernorm）
#   ② 训练 ref 基线（variant=ref, RefMLPHead: LN→512→GELU→Dropout→Softplus, gene_norm=log1p）
#   ③ 测试（含 SSIM/全Top-k/逐基因CSV）
# 原 X_uni2.npy 保持不变（供 SpatialEx/STFlow 等，公平性决策后再统一）。
# 用法：nohup bash scripts/run_uni2_ref.sh > logs/uni2_ref.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
mkdir -p outputs logs
LOG=logs/uni2_ref.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "==== UNI2+MLP ref 验证开始 ===="

# ① LN 版 CLS 特征（l1=256，与基线 X_uni2.npy 提取方式一致，只是加 LayerNorm）
for rep in 1 2; do
  OUT=data/rep${rep}/X_uni2_ln.npy
  if [ ! -f "$OUT" ]; then
    say "== 提取 LN CLS rep$rep =="
    python3 scripts/extract_local_global.py --rep $rep --data_dir data/rep$rep \
      --stage global --l1 256 --layernorm --output "$OUT" --device cuda >> "$LOG" 2>&1
  else
    say "X_uni2_ln.npy rep$rep 已存在，跳过"
  fi
done

# ② 训练 ref 基线
D=outputs/bench_uni2_mlp_ref_unf
if [ ! -f "$D/best.pt" ]; then
  say "== ref 训练 =="
  python3 scripts/train.py --method uni2_mlp --variant ref --feature_file X_uni2_ln.npy \
    --train_dir data/rep1 --valid_dir data/rep2 --gene_norm log1p \
    --epochs 50 --patience 10 --lr 1e-3 --batch_size 2048 \
    --output_dir "$D" --device cuda >> "$LOG" 2>&1
fi

# ③ 测试
say "== ref 测试 =="
python3 scripts/test.py --method uni2_mlp --variant ref --feature_file X_uni2_ln.npy \
  --ckpt "$D/best.pt" --test_dir data/rep2 --gene_norm log1p \
  --output_dir "$D" --device cuda >> "$LOG" 2>&1

say "==== UNI2+MLP ref 验证完成 ===="
