#!/bin/bash
# ref MLP + 忠实特征 LG l2 sweep @ l1=112（RefMLPHead Softplus → log1p）。
# 验证：ref MLP 配忠实特征在调优 l2 后能否超标准 MLP+忠实(0.3682) 或 原LG(0.3712)。
set -u
cd "$(dirname "$0")/.."
mkdir -p outputs logs
LOG=logs/lg_ref_op2.log
L1=112
L2S=(28 42 56 70 84 98)
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "==== ref MLP + 忠实特征 LG l2 sweep @ l1=$L1 开始 ===="
# 特征已存在（op2 忠实提取过 g112_ref/l{l2}_ref），直接训
OUT=outputs/lg_ref_op2
mkdir -p $OUT
for l2 in "${L2S[@]}"; do
  D=$OUT/l${l2}
  say "== 训练 local_global_ref l1=$L1 l2=$l2 (log1p) =="
  python3 scripts/train.py --method uni2_mlp --variant local_global_ref \
    --feature_file "X_uni2_g${L1}_ref.npy,X_uni2_l${l2}_ref.npy" \
    --train_dir data/rep1 --valid_dir data/rep2 --gene_norm log1p \
    --epochs 30 --patience 10 --lr 1e-3 --batch_size 2048 \
    --output_dir "$D" --device cuda >> "$LOG" 2>&1
  v=$(python3 -c "
import json
h=json.load(open('$D/history.json'))
p=[x['PCC'] for x in h if 'PCC' in x and x['PCC']==x['PCC']]
print(f'{max(p):.4f}' if p else 'nan')")
  say "l2=$l2 best val_PCC=$v"
done

echo "l1=$L1 ref MLP + 忠实 结果:" >> $LOG
for l2 in "${L2S[@]}"; do
  v=$(python3 -c "
import json
h=json.load(open('$OUT/l${l2}/history.json'))
p=[x['PCC'] for x in h if 'PCC' in x and x['PCC']==x['PCC']]
print(f'{max(p):.4f}' if p else 'nan')")
  echo "  l2=$l2: $v" | tee -a "$LOG"
done
say "==== 完成 ===="
