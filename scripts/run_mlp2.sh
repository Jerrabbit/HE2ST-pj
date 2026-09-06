#!/bin/bash
# "2层256+BN+LeakyReLU，无LN/无Softplus" MLP(bn2) × 3 组特征对比（可与 GHIST+segCE 并行）。
# 用法：nohup bash scripts/run_mlp2.sh > logs/mlp2.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
mkdir -p outputs logs
LOG=logs/mlp2.log
say() { echo "[$(date -u '+%F %T')] $*" | tee -a "$LOG"; }

run() { # name train_feat test_feat
  local name=$1 tf=$2 vf=$3 D=outputs/bench_mlp2_$name
  if [ -f "$D/test_results.json" ]; then say "== $name 已有，跳过 =="; return; fi
  say "== $name (bn2) train+test =="
  python3 scripts/train_classmate_feat.py --arch bn2 \
    --train_dir data/rep1 --test_dir data/rep2 \
    --train_feat "$tf" --test_feat "$vf" --output_dir "$D" --device cuda \
    > logs/bench_mlp2_${name}.log 2>&1
  say "== $name done =="
}

say "==== mlp2(bn2) 3 组开始 ===="
# ① 我们 UNI2 CLS(1536)
if [ -f data/rep1/X_uni2.npy ] && [ -f data/rep2/X_uni2.npy ]; then
  run uni2cls data/rep1/X_uni2.npy data/rep2/X_uni2.npy
else
  say "!! 缺 data/rep1/X_uni2.npy，跳过 uni2cls"
fi
# ② 同学 BLIP2(768)
run blip2 outputs/classmate_feat/rep1/DAVID_BLIP2_features.npy outputs/classmate_feat/rep2/DAVID_BLIP2_features.npy
# ③ 同学 UNI(3072)
run uni outputs/classmate_feat/rep1/UNI_features.npy outputs/classmate_feat/rep2/UNI_features.npy
say "==== mlp2 3 组全部结束 ===="
