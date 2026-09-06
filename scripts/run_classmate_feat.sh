#!/bin/bash
# 同学特征 + 我们两种 MLP 头：2 特征 × 2 头 = 4 组，串行跑（轻量，GPU 计算可空跑时并行）。
# 用法：nohup bash scripts/run_classmate_feat.sh > logs/classmate_feat.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
mkdir -p outputs logs
LOG=logs/classmate_feat.log
say() { echo "[$(date -u '+%F %T')] $*" | tee -a "$LOG"; }
ROOT=outputs/classmate_feat
GN=""

run() { # key arch
  local key=$1 arch=$2
  local tok=${key//_/-}
  local D=outputs/bench_cc_${tok}_${arch}
  if [ -f "$D/test_results.json" ]; then say "== $key/$arch 已有，跳过 =="; return; fi
  say "== $key ($arch) train+test =="
  python3 scripts/train_classmate_feat.py --arch $arch \
    --train_dir data/rep1 --test_dir data/rep2 \
    --train_feat $ROOT/rep1/$key.npy --test_feat $ROOT/rep2/$key.npy \
    --output_dir "$D" --device cuda > logs/bench_cc_${tok}_${arch}.log 2>&1
  say "== $key/$arch done =="
}

say "==== 同学特征 4 组开始 ===="
for k in DAVID_BLIP2_features UNI_features; do
  run $k mlp
  run $k ref
done
say "==== 同学特征 4 组全部结束 ===="
