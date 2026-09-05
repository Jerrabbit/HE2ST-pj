#!/bin/bash
# 官方对齐改进版顺序重跑（2026-09-06）：跑完当前 GHIST(3-MSE) 后启动，**跳过 seed 复跑**。
# 顺序：SpatialEx(超图 k 修复) → DeepPT(AE 冻结) → Path2Space(bias_init) → BLEEP(raw 检索加权)
#        → GHIST(+分割 CE)。各输出 *_unf_fix / *_unf_seg 目录（保留旧 *_unf 作对照），幂等。
# 用法：cd 项目根 && nohup bash scripts/run_improved.sh > logs/run_improved.log 2>&1 < /dev/null &
set -u
cd "$(dirname "$0")/.."
mkdir -p outputs logs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=logs/run_improved.log
say() { echo "[$(date -u '+%F %T')] $*" | tee -a "$LOG"; }
GN=log1p_zscore; EPOCHS=50; PATIENCE=10; LR=1e-3; BS=2048; DEV=cuda

say "==== improved 队列开始 ===="

# ---------- 1) SpatialEx（超图 k=7 修复版） ----------
m=spatialex; D=outputs/bench_${m}_unf_fix
if [ -f "$D/test_results.json" ]; then say "== $m unf_fix 已有结果，跳过 =="; else
  say "== $m unf_fix train =="
  python3 scripts/train.py --method $m --train_dir data/rep1 --valid_dir data/rep2 \
    --epochs $EPOCHS --patience $PATIENCE --lr $LR --gene_norm $GN \
    --output_dir "$D" --device $DEV > logs/bench_${m}_unf_fix_train.log 2>&1
  if [ -f "$D/best.pt" ]; then
    say "== $m unf_fix test =="
    python3 scripts/test_spatialex.py --test_dir data/rep2 --checkpoint "$D/best.pt" \
      --gene_norm $GN --output_dir "$D" --device $DEV >> logs/bench_${m}_unf_fix_train.log 2>&1
  fi
fi

# ---------- 2) DeepPT ResNet50 忠实版（AE 训后冻结） ----------
m=deeppt; D=outputs/bench_${m}_resnet50_unf_fix
if [ -f "$D/test_results.json" ]; then say "== $m unf_fix 已有结果，跳过 =="; else
  say "== $m(R50) unf_fix train =="
  python3 scripts/train.py --method $m --feature_file X_resnet50.npy --feat_dim 2048 \
    --train_dir data/rep1 --valid_dir data/rep2 \
    --epochs $EPOCHS --patience $PATIENCE --lr $LR --gene_norm $GN \
    --output_dir "$D" --device $DEV > logs/bench_${m}_resnet50_unf_fix_train.log 2>&1
  if [ -f "$D/best.pt" ]; then
    say "== $m(R50) unf_fix test =="
    python3 scripts/test.py --method $m --feature_file X_resnet50.npy --feat_dim 2048 \
      --ckpt "$D/best.pt" --test_dir data/rep2 --gene_norm $GN \
      --output_dir "$D" --device $DEV >> logs/bench_${m}_resnet50_unf_fix_train.log 2>&1
  fi
fi

# ---------- 3) Path2Space（官方 bias_init 训练头） ----------
m=path2space; D=outputs/bench_${m}_unf_fix
if [ -f "$D/test_results.json" ]; then say "== $m unf_fix 已有结果，跳过 =="; else
  say "== $m unf_fix train =="
  python3 scripts/train.py --method $m --feature_file X_ctranspath_ctx512.npy \
    --train_dir data/rep1 --valid_dir data/rep2 \
    --epochs $EPOCHS --patience $PATIENCE --lr $LR --gene_norm $GN \
    --output_dir "$D" --device $DEV > logs/bench_${m}_unf_fix_train.log 2>&1
  if [ -f "$D/best.pt" ]; then
    say "== $m unf_fix test =="
    python3 scripts/test.py --method $m --feature_file X_ctranspath_ctx512.npy \
      --ckpt "$D/best.pt" --test_dir data/rep2 --gene_norm $GN \
      --output_dir "$D" --device $DEV >> logs/bench_${m}_unf_fix_train.log 2>&1
  fi
fi

# ---------- 4) BLEEP(冻结) unf（检索改官方 raw 加权；/tmp 快盘） ----------
m=bleep; D=outputs/bench_${m}_unf_fix
if [ -f "$D/test_results.json" ]; then say "== $m unf_fix 已有结果，跳过 =="; else
  say "== $m unf_fix train =="
  python3 scripts/train.py --method $m --no_finetune --pretrained_weights weights/resnet50_imagenet.pth \
    --train_dir /tmp/bl_rep1 --valid_dir /tmp/bl_rep2 \
    --epochs $EPOCHS --patience $PATIENCE --batch_size $BS --lr $LR --gene_norm $GN \
    --output_dir "$D" --device $DEV > logs/bench_${m}_unf_fix_train.log 2>&1
  if [ -f "$D/best.pt" ]; then
    say "== $m unf_fix test =="
    python3 scripts/test.py --method $m --img_size 224 --pretrained_weights weights/resnet50_imagenet.pth \
      --ckpt "$D/best.pt" --test_dir /tmp/bl_rep2 --gene_norm $GN \
      --output_dir "$D" --device $DEV >> logs/bench_${m}_unf_fix_train.log 2>&1
  fi
fi

# ---------- 5) GHIST + 分割 CE（对照 3-MSE 版 bench_ghist_unf） ----------
m=ghist; D=outputs/bench_${m}_unf_seg
if [ -f "$D/test_results.json" ]; then say "== $m unf_seg 已有结果，跳过 =="; else
  say "== $m unf_seg train（+分割 CE）=="
  python3 scripts/train.py --method $m --train_dir data/ghist_rep1 --valid_dir data/ghist_rep2 \
    --epochs $EPOCHS --patience $PATIENCE --lr $LR --batch_size 8 \
    --output_dir "$D" --device $DEV > logs/bench_${m}_unf_seg_train.log 2>&1
  if [ -f "$D/best.pt" ]; then
    say "== $m unf_seg test =="
    python3 scripts/test_ghist.py --ckpt "$D/best.pt" --test_dir data/ghist_rep2 \
      --output_dir "$D" --device $DEV >> logs/bench_${m}_unf_seg_train.log 2>&1
  fi
fi

say "==== improved 队列全部结束 ===="
