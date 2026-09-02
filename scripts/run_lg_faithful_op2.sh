#!/bin/bash
# op2 @ l1=112：忠实特征 Local 调参（标准 MLPHead + log1p_zscore）。
# l1 不扫（CLS 与原始一致，best l1≈112），只调 l2。
set -u
cd "$(dirname "$0")/.."
mkdir -p outputs logs
LOG=logs/lg_faithful_op2.log
L1=112
L2S=(28 42 56 70 84 98)
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "==== 忠实特征 op2 @ l1=$L1 开始 ===="

# 提取 l1=112 的忠实 Global + 全部 l2 Local（单 forward 复用）
for rep in 1 2; do
  if [ ! -f "data/rep${rep}/X_uni2_g${L1}_ref.npy" ] || [ ! -f "data/rep${rep}/X_uni2_l56_ref.npy" ]; then
    say "== 忠实提取 rep$rep (l1=$L1, 全部 l2) =="
    python3 scripts/extract_faithful.py --rep $rep --data_dir data/rep$rep \
      --l1 $L1 --l2_list "${L2S[@]}" --device cuda >> "$LOG" 2>&1
  else
    say "rep$rep 特征已存在"
  fi
done

# 每 l2 训练 local_global 30ep
OUT=outputs/lg_faithful_op2
mkdir -p $OUT
for l2 in "${L2S[@]}"; do
  D=$OUT/l${l2}
  say "== 训练 l1=$L1 l2=$l2 =="
  python3 scripts/train.py --method uni2_mlp --variant local_global \
    --feature_file "X_uni2_g${L1}_ref.npy,X_uni2_l${l2}_ref.npy" \
    --train_dir data/rep1 --valid_dir data/rep2 --gene_norm log1p_zscore \
    --epochs 30 --patience 10 --lr 1e-3 --batch_size 2048 \
    --output_dir "$D" --device cuda >> "$LOG" 2>&1
  v=$(python3 -c "
import json
h=json.load(open('$D/history.json'))
p=[x['PCC'] for x in h if 'PCC' in x and x['PCC']==x['PCC']]
print(f'{max(p):.4f}' if p else 'nan')")
  say "l2=$l2 best val_PCC=$v"
done

# 汇总
echo "l1=$L1, results:" >> $LOG
for l2 in "${L2S[@]}"; do
  v=$(python3 -c "
import json
h=json.load(open('$OUT/l${l2}/history.json'))
p=[x['PCC'] for x in h if 'PCC' in x and x['PCC']==x['PCC']]
print(f'{max(p):.4f}' if p else 'nan')")
  echo "  l2=$l2: $v" | tee -a "$LOG"
done
say "==== op2@l1=$L1 完成 ===="
