#!/bin/bash
# rep1→rep2 benchmark 续段（v2，用户 2026-08-29 要求：耗时短先跑）。
# 由 watcher_reorder.sh 在 st_net 训练完成后启动。
# 顺序：
#   ① st_net 测试（多 worker 并行读图加速，patch 方法，~40 min）
#   ② 重跑已完成的 5 方法测试（uni2_mlp/pixel2gene/path2space/deeppt/spatialex）
#      刷新 SSIM → 统一 log1p(counts) 空间（快，~5-10 min）
#   ③ stflow 训练+测试（快，~30 min）
#   ④ hist2st 训练+测试（慢，patch 方法）
#   ⑤ bleep 冻结训练+测试（慢，patch 方法）
#   ⑥ LG 完整调参（op1 sweep → op2 sweep → 最终 50ep）——最后
#   ⑦ GHIST 训练+测试（OOM 恢复，失败重试）
# 用法：ulimit -n 65535 && nohup bash scripts/run_bench_unf_v2.sh > logs/bench_unf_v2.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
mkdir -p outputs logs

EPOCHS=${EPOCHS:-50}
PATIENCE=${PATIENCE:-10}
LR=${LR:-1e-3}
BS=${BS:-2048}
GN=${GN:-log1p_zscore}
DEV="cuda"
LOG=logs/bench_unf_v2.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "==== rep1→rep2 v2 续段开始（重排：短方法先跑）===="

# ---------- ① st_net 测试（多 worker 并行读图） ----------
m=st_net; D=outputs/bench_${m}_unf
if [ ! -f "$D/test_results.json" ] || [ ! -f "$D/gene_pcc.csv" ]; then
  say "== $m test（多 worker）=="
  python3 scripts/test.py --method $m --ckpt "$D/best.pt" --test_dir data/rep2 \
    --gene_norm $GN --output_dir "$D" --device cuda --workers 8 >> logs/bench_${m}_unf_train.log 2>&1
fi

# ---------- ② 重跑已完成的 5 方法测试（SSIM→log1p 空间） ----------
retest() { # $1=方法名  $2..=额外参数
  local m=$1; shift
  local D=outputs/bench_${m}_unf
  if [ ! -f "$D/best.pt" ]; then say "!! $m 无 best.pt，跳过"; return; fi
  say "== $m test 重跑（SSIM→log1p 空间）=="
  python3 scripts/test.py --method $m --ckpt "$D/best.pt" --test_dir data/rep2 \
    --gene_norm log1p_zscore --output_dir "$D" --device cuda "$@" >> "$LOG" 2>&1
}
say "--- ① 重跑已完成方法 ---"
retest uni2_mlp
retest pixel2gene --variant cell --gene_norm log1p
retest path2space
retest deeppt --feature_file X_resnet50.npy --feat_dim 2048
D=outputs/bench_spatialex_unf
if [ -f "$D/best.pt" ]; then
  say "== spatialex test 重跑 =="
  python3 scripts/test_spatialex.py --test_dir data/rep2 --checkpoint "$D/best.pt" \
    --gene_norm log1p_zscore --output_dir "$D" --device cuda >> "$LOG" 2>&1
fi

# ---------- ② stflow（官方协议，快） ----------
m=stflow; D=outputs/bench_${m}_unf
if [ ! -f "$D/best.pt" ]; then
  say "== $m train =="
  python3 scripts/train.py --method $m --gene_norm log1p --prior zinb --n_sample_steps 50 \
    --train_dir data/rep1 --valid_dir data/rep2 \
    --epochs 100 --patience 20 --lr 5e-4 \
    --output_dir "$D" --device cuda > logs/bench_${m}_unf_train.log 2>&1
fi
say "== $m test =="
python3 scripts/test_stflow.py --ckpt "$D/best.pt" --test_dir data/rep2 \
  --output_dir "$D" --device cuda >> logs/bench_${m}_unf_train.log 2>&1

# ---------- ③ hist2st（官方配置，慢） ----------
m=hist2st; D=outputs/bench_${m}_unf
if [ ! -f "$D/best.pt" ]; then
  say "== $m official train =="
  python3 scripts/train.py --method $m --epochs 100 --lr 1e-5 --zinb 0.25 --zinb_coef 0.25 \
    --bake 5 --lamb 0.5 --train_dir data/rep1 --valid_dir data/rep2 \
    --output_dir "$D" --device cuda > logs/bench_${m}_unf_train.log 2>&1
fi
say "== $m official test =="
python3 scripts/test_hist2st.py --ckpt "$D/best.pt" --test_dir data/rep2 \
  --output_dir "$D" --device cuda >> logs/bench_${m}_unf_train.log 2>&1

# ---------- ④ bleep（冻结 resnet50，慢） ----------
m=bleep; D=outputs/bench_${m}_unf
if [ ! -f "$D/best.pt" ]; then
  say "== $m frozen train =="
  python3 scripts/train.py --method $m --no_finetune \
    --pretrained_weights weights/resnet50_imagenet.pth \
    --train_dir data/rep1 --valid_dir data/rep2 \
    --epochs $EPOCHS --patience $PATIENCE --batch_size $BS --lr $LR --gene_norm $GN \
    --output_dir "$D" --device cuda > logs/bench_${m}_unf_train.log 2>&1
fi
say "== $m frozen test =="
python3 scripts/test.py --method $m --img_size 224 \
  --pretrained_weights weights/resnet50_imagenet.pth \
  --ckpt "$D/best.pt" --test_dir data/rep2 \
  --gene_norm $GN --output_dir "$D" --device cuda >> logs/bench_${m}_unf_train.log 2>&1

# ---------- ⑤ LG 完整调参（最后） ----------
OP1=outputs/sweep_op1_unf
OP2=outputs/sweep_op2_unf
say "== LG op1 sweep（Global 视野 l1 448→28）=="
python3 scripts/sweep.py --stage op1 --train_rep 1 --valid_rep 2 \
  --train_dir data/rep1 --valid_dir data/rep2 \
  --l1_values 448 420 392 364 336 308 280 252 224 196 168 140 112 84 56 28 \
  --out_dir $OP1 >> logs/lg_op1_unf.log 2>&1 || { say "!! LG op1 失败"; exit 1; }
best_l1=$(python3 -c "
import csv
rows=list(csv.reader(open('$OP1/op1_results.csv')))[1:]
b=max(rows, key=lambda r: float(r[1])); print(b[0])")
say "== best l1 = $best_l1，LG op2 sweep（Local l2 28..112）=="
python3 scripts/sweep.py --stage op2 --l1 "$best_l1" --train_rep 1 --valid_rep 2 \
  --train_dir data/rep1 --valid_dir data/rep2 \
  --l2_values 28 42 56 70 84 98 112 \
  --out_dir $OP2 >> logs/lg_op2_unf.log 2>&1 || { say "!! LG op2 失败"; exit 1; }
best_l2=$(python3 -c "
import csv
rows=list(csv.reader(open('$OP2/op2_results.csv')))[1:]
b=max(rows, key=lambda r: float(r[1])); print(b[0])")
say "== best l2 = $best_l2，LG 最终 50ep 训练+测试 =="
m=uni2_mlp; D=outputs/bench_${m}_lg_unf
python3 scripts/train.py --method $m --variant local_global \
  --feature_file "X_uni2_g${best_l1}.npy,X_uni2_l${best_l2}.npy" \
  --train_dir data/rep1 --valid_dir data/rep2 \
  --epochs 50 --patience $PATIENCE --batch_size $BS --lr $LR --gene_norm $GN \
  --output_dir "$D" --device cuda > logs/bench_${m}_lg_unf_train.log 2>&1
python3 scripts/test.py --method $m --variant local_global \
  --feature_file "X_uni2_g${best_l1}.npy,X_uni2_l${best_l2}.npy" \
  --ckpt "$D/best.pt" --test_dir data/rep2 --gene_norm $GN \
  --output_dir "$D" --device cuda >> logs/bench_${m}_lg_unf_train.log 2>&1

# ---------- ⑥ GHIST（OOM 恢复，失败重试 3 次） ----------
m=ghist; D=outputs/bench_${m}_unf
for attempt in 1 2 3; do
  if [ -f "$D/best.pt" ]; then say "GHIST best.pt 已存在，跳过"; break; fi
  say "== GHIST train（attempt $attempt/3）=="
  python3 scripts/train.py --method $m --train_dir data/ghist_rep1 --valid_dir data/ghist_rep2 \
    --epochs 50 --patience 10 --lr 1e-3 --output_dir "$D" --device cuda >> "$LOG" 2>&1
  if [ -f "$D/best.pt" ]; then
    say "== GHIST test =="
    python3 scripts/test_ghist.py --ckpt "$D/best.pt" --test_dir data/ghist_rep2 \
      --output_dir "$D" --device cuda >> "$LOG" 2>&1
    break
  else
    say "!! GHIST 训练失败（可能 GPU 被抢占），10 分钟后重试"
    sleep 600
  fi
done

say "==== v2 全部完成 ===="
