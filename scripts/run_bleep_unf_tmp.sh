#!/bin/bash
# BLEEP(冻结) unf：把 patches 复制到 /tmp NVMe 快盘再训（cpfs 读图 ~2h/epoch 瓶颈）。
# 做法：准备 /tmp/bl_rep{1,2}（metadata 的 patch_path 改写为 /tmp 绝对路径 + patches + expr + 基因名），
#      再以 --train_dir/valid_dir=/tmp/... 训练→测试。
# 用法：cd 项目根 && nohup bash scripts/run_bleep_unf_tmp.sh > logs/bleep_tmp.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
mkdir -p outputs logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=logs/bleep_tmp.log
say() { echo "[$(date -u '+%F %T')] $*" | tee -a "$LOG"; }
RE=$(pwd)

# ---------- 准备 /tmp 快盘副本（幂等，已存在则跳过） ----------
prep() { # $1=repN
  local src=$RE/data/$1 dst=/tmp/bl_$1
  if [ -f "$dst/patches/.done" ]; then say "== $dst 已备好，跳过 =="; return; fi
  say "== 准备 $dst =="
  mkdir -p "$dst/patches"
  cp -f "$src/gene_expression.npy" "$src/gene_names.txt" "$dst/"
  sed "s#data/$1/patches/#$dst/patches/#g" "$src/metadata.csv" > "$dst/metadata.csv"
  cp -a "$src/patches/." "$dst/patches/"
  touch "$dst/patches/.done"
  say "== $dst 就绪（patch 数 $(ls "$dst/patches" | wc -l)）=="
}
prep rep1
prep rep2

# ---------- BLEEP(冻结) 训练 + 测试（/tmp 快盘） ----------
D=outputs/bench_bleep_unf
if [ -f "$D/test_results.json" ]; then
  say "== bleep unf 已有结果，跳过 =="
  exit 0
fi
say "== bleep(冻结) unf train（/tmp 快盘）=="
python3 scripts/train.py --method bleep --no_finetune --pretrained_weights weights/resnet50_imagenet.pth \
  --train_dir /tmp/bl_rep1 --valid_dir /tmp/bl_rep2 --epochs 50 --patience 10 \
  --batch_size 2048 --lr 1e-3 --gene_norm log1p_zscore \
  --output_dir "$D" --device cuda > logs/bench_bleep_unf_train.log 2>&1
if [ -f "$D/best.pt" ]; then
  say "== bleep(冻结) unf test =="
  python3 scripts/test.py --method bleep --img_size 224 --pretrained_weights weights/resnet50_imagenet.pth \
    --ckpt "$D/best.pt" --test_dir /tmp/bl_rep2 --gene_norm log1p_zscore \
    --output_dir "$D" --device cuda >> logs/bench_bleep_unf_train.log 2>&1
  say "== bleep unf done =="
else
  say "!! bleep unf 训练失败（无 best.pt）"
  exit 1
fi
