#!/bin/bash
# 过滤后 Local+Global 正式实验：op1 sweep → op2 sweep → 最终 50ep → 消融。
# 特征复用 filter_cells.py 已切片的 X_uni2_g{l1}.npy / X_uni2_l{l2}.npy（sweep 自动跳过提取）。
# 用法：nohup bash scripts/run_lg_filtered.sh > logs/lg_filtered.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
mkdir -p outputs logs
DEV="cuda"
GN=${GN:-log1p_zscore}
BS=${BS:-2048}

say() { echo "[$(date '+%F %T')] $*" | tee -a logs/lg_filtered.log; }

best_val() { # best_val <csv> → 打印 pcc 最大的 param
  python -c "
import csv, sys
rows=list(csv.DictReader(open('$1')))
b=max(rows, key=lambda r: float(r['best_val_pcc']))
print(int(b['param']))
"
}

# ---------- 1. op1 sweep（只测 Global，l1 448→28） ----------
S1=outputs/sweep_op1_f
if [ ! -f "$S1/op1_results.csv" ]; then
  say "== op1 sweep（过滤后）=="
  python scripts/sweep.py --stage op1 --train_rep 1 --valid_rep 2 \
    --train_dir data/rep1_f --valid_dir data/rep2_f \
    --l1_values 448 420 392 364 336 308 280 252 224 196 168 140 112 84 56 28 \
    --out_dir $S1 > logs/sweep_op1_f.log 2>&1
fi
L1=$(best_val $S1/op1_results.csv)
say "== op1 best l1 = $L1 =="

# ---------- 2. op2 sweep（固定 best l1，l2 28..112） ----------
S2=outputs/sweep_op2_f
if [ ! -f "$S2/op2_results.csv" ]; then
  say "== op2 sweep（过滤后，l1=$L1）=="
  python scripts/sweep.py --stage op2 --l1 "$L1" --train_rep 1 --valid_rep 2 \
    --train_dir data/rep1_f --valid_dir data/rep2_f \
    --l2_values 28 42 56 70 84 98 112 \
    --out_dir $S2 > logs/sweep_op2_f.log 2>&1
fi
L2=$(best_val $S2/op2_results.csv)
say "== op2 best l2 = $L2 =="

# ---------- 3. 最终训练 + 消融（50ep） ----------
F=outputs/bench_uni2_lg_final_f
run_cfg() { # run_cfg <out_dir> <variant> <feature_file>
  local D=$1 VAR=$2 FF=$3
  if [ ! -f "$D/best.pt" ]; then
    say "== $VAR train（l1=$L1 l2=$L2）=="
    python scripts/train.py --method uni2_mlp --variant "$VAR" --feature_file "$FF" \
      --train_dir data/rep1_f --valid_dir data/rep2_f \
      --epochs 50 --patience 10 --batch_size $BS --lr 1e-3 --gene_norm $GN \
      --output_dir "$D" --device $DEV > "logs/${VAR}_f_train.log" 2>&1
  fi
  say "== $VAR test =="
  python scripts/test.py --method uni2_mlp --variant "$VAR" --feature_file "$FF" \
    --ckpt "$D/best.pt" --test_dir data/rep2_f --gene_norm $GN \
    --output_dir "$D" --device $DEV >> "logs/${VAR}_f_train.log" 2>&1
}

run_cfg outputs/bench_uni2_lg_final_f   local_global "X_uni2_g${L1}.npy,X_uni2_l${L2}.npy"
run_cfg outputs/bench_uni2_global_f     global_only  "X_uni2_g${L1}.npy"
run_cfg outputs/bench_uni2_local_f      local_only   "X_uni2_l${L2}.npy"
run_cfg outputs/bench_uni2_lg_ln_f      local_global_ln "X_uni2_g${L1}.npy,X_uni2_l${L2}.npy"

say "==== 过滤后 Local+Global 实验完成（l1=$L1 l2=$L2）===="
