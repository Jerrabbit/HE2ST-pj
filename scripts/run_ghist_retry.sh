#!/bin/bash
# GHIST 单独重跑（过滤后数据，genes.txt 已补 + KeyError 已修）。
set -u
cd "$(dirname "$0")/.."
D=outputs/bench_ghist_f
echo "[$(date '+%F %T')] == ghist train =="
python scripts/train.py --method ghist --train_dir data/ghist_rep1_f --valid_dir data/ghist_rep2_f \
  --epochs 50 --patience 10 --lr 1e-3 --output_dir "$D" --device cuda
echo "[$(date '+%F %T')] == ghist test =="
python scripts/test_ghist.py --ckpt "$D/best.pt" --test_dir data/ghist_rep2_f \
  --output_dir "$D" --device cuda
echo "[$(date '+%F %T')] GHIST 重跑完成"
