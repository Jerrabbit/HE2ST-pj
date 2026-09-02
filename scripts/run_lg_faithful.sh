#!/bin/bash
# LG 忠实复刻验证（2026-09-02）：extract_faithful 用 intermediates[-1] 提 Local，
# 训练 local_global_ref（RefMLPHead + Softplus + log1p），对比旧 LG 标准版 0.3712。
# 基线已确认：忠实 CLS == 原 CLS（probe diff=0），ref MLP 对基线不提升（0.3160）。
# 用法：nohup bash scripts/run_lg_faithful.sh > logs/lg_faithful.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
mkdir -p outputs logs
LOG=logs/lg_faithful.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "==== LG 忠实复刻验证开始 ===="

# ① 忠实提取 LG 特征（rep1/rep2, l1=112, 全部 l2 一次 forward 免费复用）
for rep in 1 2; do
  G=data/rep${rep}/X_uni2_g112_ref.npy
  if [ ! -f "$G" ]; then
    say "== 忠实提取 rep$rep (l1=112) =="
    python3 scripts/extract_faithful.py --rep $rep --data_dir data/rep$rep \
      --l1 112 --l2_list 28 42 56 70 84 98 112 --device cuda >> "$LOG" 2>&1
  else
    say "rep$rep 忠实特征已存在，跳过"
  fi
done

# ② 训练 local_global_ref（l1=112/l2=56, gene_norm=log1p, Softplus）
D=outputs/bench_uni2_mlp_lg_ref_unf
if [ ! -f "$D/best.pt" ]; then
  say "== LG ref(忠实) 训练 =="
  python3 scripts/train.py --method uni2_mlp --variant local_global_ref \
    --feature_file "X_uni2_g112_ref.npy,X_uni2_l56_ref.npy" \
    --train_dir data/rep1 --valid_dir data/rep2 --gene_norm log1p \
    --epochs 50 --patience 10 --lr 1e-3 --batch_size 2048 \
    --output_dir "$D" --device cuda >> "$LOG" 2>&1
fi

# ③ 测试
say "== LG ref(忠实) 测试 =="
python3 scripts/test.py --method uni2_mlp --variant local_global_ref \
  --feature_file "X_uni2_g112_ref.npy,X_uni2_l56_ref.npy" \
  --ckpt "$D/best.pt" --test_dir data/rep2 --gene_norm log1p \
  --output_dir "$D" --device cuda >> "$LOG" 2>&1

say "==== LG 忠实复刻验证完成 ===="
