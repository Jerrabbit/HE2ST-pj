#!/bin/bash
# 过滤后 benchmark：rep1_f 训练 → rep2_f 测试（11 种方法，Phoenix 权重待下载后单独补）。
# 统一协议：50ep + 早停 + 训练集统计量复用；评估用扩展后的统一模块
# （含 cell-level PCC、Top-k 全 k 值，CSV 导出到各 outputs/bench_*_f/eval_metrics.csv）。
# 幂等：已有 best.pt 的步骤自动跳过。
# 用法：nohup bash scripts/run_bench_filtered.sh > logs/bench_filtered_all.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
mkdir -p outputs logs

EPOCHS=${EPOCHS:-50}
PATIENCE=${PATIENCE:-10}
LR=${LR:-1e-3}
BS=${BS:-2048}
GN=${GN:-log1p_zscore}
MIN_GENES=${MIN_GENES:-200}
MIN_UMIS=${MIN_UMIS:-500}
DEV="cuda"

say() { echo "[$(date '+%F %T')] $*" | tee -a logs/bench_filtered_all.log; }

say "==== 过滤后 benchmark 开始: EPOCHS=$EPOCHS GN=$GN min_genes=$MIN_GENES min_umis=$MIN_UMIS ===="

# ---------- 0. 细胞过滤（幂等） ----------
for rep in rep1 rep2; do
  if [ ! -d "data/${rep}_f" ]; then
    say "[filter] $rep → ${rep}_f ..."
    python scripts/filter_cells.py --src_dir "data/$rep" --out_dir "data/${rep}_f" \
      --min_genes "$MIN_GENES" --min_umis "$MIN_UMIS" \
      --ghist_src "data/ghist_$rep" --ghist_out "data/ghist_${rep}_f" \
      >> "logs/filter_${rep}.log" 2>&1 || { say "!! filter $rep 失败"; exit 1; }
  fi
done
say "[filter] 完成"

# ---------- 1. UNI2+MLP（基线） ----------
m=uni2_mlp; D=outputs/bench_${m}_f
if [ ! -f "$D/best.pt" ]; then
  say "== $m train =="
  python scripts/train.py --method $m --train_dir data/rep1_f --valid_dir data/rep2_f \
    --epochs $EPOCHS --patience $PATIENCE --batch_size $BS --lr $LR --gene_norm $GN \
    --output_dir $D --device $DEV > logs/bench_${m}_train.log 2>&1
fi
say "== $m test =="
python scripts/test.py --method $m --ckpt $D/best.pt --test_dir data/rep2_f \
  --gene_norm $GN --output_dir $D --device $DEV >> logs/bench_${m}_train.log 2>&1

# ---------- 2. SQUALL 官方解码器头 ----------
m=squall; D=outputs/bench_${m}_decoder_f
if [ ! -f "$D/best.pt" ]; then
  say "== $m decoder train =="
  python scripts/train.py --method $m --variant decoder --feature_file X_squall_tokens.npy \
    --train_dir data/rep1_f --valid_dir data/rep2_f \
    --epochs $EPOCHS --patience $PATIENCE --batch_size $BS --lr $LR --gene_norm $GN \
    --output_dir $D --device $DEV > logs/bench_${m}_decoder_train.log 2>&1
fi
say "== $m decoder test =="
python scripts/test.py --method $m --variant decoder --feature_file X_squall_tokens.npy \
  --ckpt $D/best.pt --test_dir data/rep2_f --gene_norm $GN \
  --output_dir $D --device $DEV >> logs/bench_${m}_decoder_train.log 2>&1

# ---------- 3. GHIST（从头，ghist_data 格式） ----------
m=ghist; D=outputs/bench_${m}_f
if [ ! -f "$D/best.pt" ]; then
  say "== $m train =="
  python scripts/train.py --method $m --train_dir data/ghist_rep1_f --valid_dir data/ghist_rep2_f \
    --epochs $EPOCHS --patience $PATIENCE --lr $LR \
    --output_dir $D --device $DEV > logs/bench_${m}_train.log 2>&1
fi
say "== $m test =="
python scripts/test_ghist.py --ckpt $D/best.pt --test_dir data/ghist_rep2_f \
  --output_dir $D --device $DEV >> logs/bench_${m}_train.log 2>&1

# ---------- 4. SpatialEx（整片超图） ----------
m=spatialex; D=outputs/bench_${m}_f
if [ ! -f "$D/best.pt" ]; then
  say "== $m train =="
  python scripts/train.py --method $m --train_dir data/rep1_f --valid_dir data/rep2_f \
    --epochs $EPOCHS --patience $PATIENCE --lr $LR --gene_norm $GN \
    --output_dir $D --device $DEV > logs/bench_${m}_train.log 2>&1
fi
say "== $m test =="
python scripts/test_spatialex.py --test_dir data/rep2_f --checkpoint $D/best.pt \
  --gene_norm $GN --output_dir $D --device $DEV >> logs/bench_${m}_train.log 2>&1

# ---------- 5. Pixel2Gene cell（官方 ForwardSum 头, log1p 空间） ----------
m=pixel2gene; D=outputs/bench_${m}_cell_f
if [ ! -f "$D/best.pt" ]; then
  say "== $m cell train =="
  python scripts/train.py --method $m --variant cell --gene_norm log1p \
    --train_dir data/rep1_f --valid_dir data/rep2_f \
    --epochs $EPOCHS --patience $PATIENCE --batch_size $BS --lr $LR \
    --output_dir $D --device $DEV > logs/bench_${m}_cell_train.log 2>&1
fi
say "== $m cell test =="
python scripts/test.py --method $m --variant cell --gene_norm log1p \
  --ckpt $D/best.pt --test_dir data/rep2_f \
  --output_dir $D --device $DEV >> logs/bench_${m}_cell_train.log 2>&1

# ---------- 6. Path2Space（重训官方训练头） ----------
m=path2space; D=outputs/bench_${m}_f
if [ ! -f "$D/best.pt" ]; then
  say "== $m train =="
  python scripts/train.py --method $m --train_dir data/rep1_f --valid_dir data/rep2_f \
    --epochs $EPOCHS --patience $PATIENCE --batch_size $BS --lr $LR --gene_norm $GN \
    --output_dir $D --device $DEV > logs/bench_${m}_train.log 2>&1
fi
say "== $m test =="
python scripts/test.py --method $m --ckpt $D/best.pt --test_dir data/rep2_f \
  --gene_norm $GN --output_dir $D --device $DEV >> logs/bench_${m}_train.log 2>&1

# ---------- 7. DeepPT（ResNet50 官方实现） ----------
m=deeppt; D=outputs/bench_${m}_resnet50_f
if [ ! -f "$D/best.pt" ]; then
  say "== $m ResNet50 train =="
  python scripts/train.py --method $m --feature_file X_resnet50.npy --feat_dim 2048 \
    --train_dir data/rep1_f --valid_dir data/rep2_f \
    --epochs $EPOCHS --patience $PATIENCE --batch_size $BS --lr $LR --gene_norm $GN \
    --output_dir $D --device $DEV > logs/bench_${m}_r50_train.log 2>&1
fi
say "== $m ResNet50 test =="
python scripts/test.py --method $m --feature_file X_resnet50.npy --feat_dim 2048 \
  --ckpt $D/best.pt --test_dir data/rep2_f --gene_norm $GN \
  --output_dir $D --device $DEV >> logs/bench_${m}_r50_train.log 2>&1

# ---------- 8. ST-Net（冻结 DenseNet） ----------
m=st_net; D=outputs/bench_${m}_f
if [ ! -f "$D/best.pt" ]; then
  say "== $m frozen train =="
  python scripts/train.py --method $m --no_finetune \
    --train_dir data/rep1_f --valid_dir data/rep2_f \
    --epochs $EPOCHS --patience $PATIENCE --batch_size $BS --lr $LR --gene_norm $GN \
    --output_dir $D --device $DEV > logs/bench_${m}_train.log 2>&1
fi
say "== $m frozen test =="
python scripts/test.py --method $m --ckpt $D/best.pt --test_dir data/rep2_f \
  --gene_norm $GN --output_dir $D --device $DEV >> logs/bench_${m}_train.log 2>&1

# ---------- 9. Hist2ST（官方配置从头） ----------
m=hist2st; D=outputs/bench_${m}_f
if [ ! -f "$D/best.pt" ]; then
  say "== $m official train =="
  python scripts/train.py --method $m --epochs 100 --lr 1e-5 --zinb 0.25 --zinb_coef 0.25 \
    --bake 5 --lamb 0.5 --train_dir data/rep1_f --valid_dir data/rep2_f \
    --output_dir $D --device $DEV > logs/bench_${m}_train.log 2>&1
fi
say "== $m official test =="
python scripts/test_hist2st.py --ckpt $D/best.pt --test_dir data/rep2_f \
  --output_dir $D --device $DEV >> logs/bench_${m}_train.log 2>&1

# ---------- 10. BLEEP（冻结 resnet50） ----------
m=bleep; D=outputs/bench_${m}_f
if [ ! -f "$D/best.pt" ]; then
  say "== $m frozen train =="
  python scripts/train.py --method $m --no_finetune \
    --pretrained_weights weights/resnet50_imagenet.pth \
    --train_dir data/rep1_f --valid_dir data/rep2_f \
    --epochs $EPOCHS --patience $PATIENCE --batch_size $BS --lr $LR --gene_norm $GN \
    --output_dir $D --device $DEV > logs/bench_${m}_train.log 2>&1
fi
say "== $m frozen test =="
python scripts/test.py --method $m --img_size 224 \
  --pretrained_weights weights/resnet50_imagenet.pth \
  --ckpt $D/best.pt --test_dir data/rep2_f \
  --gene_norm $GN --output_dir $D --device $DEV >> logs/bench_${m}_train.log 2>&1

# ---------- 11. STFlow（log1p + zinb 官方默认） ----------
m=stflow; D=outputs/bench_${m}_f
if [ ! -f "$D/best.pt" ]; then
  say "== $m train =="
  python scripts/train.py --method $m --gene_norm log1p --prior zinb --n_sample_steps 50 \
    --train_dir data/rep1_f --valid_dir data/rep2_f \
    --epochs $EPOCHS --patience $PATIENCE --lr $LR \
    --output_dir $D --device $DEV > logs/bench_${m}_train.log 2>&1
fi
say "== $m test =="
python scripts/test_stflow.py --ckpt $D/best.pt --test_dir data/rep2_f \
  --output_dir $D --device $DEV >> logs/bench_${m}_train.log 2>&1

say "==== 11 方法过滤后 benchmark 全部完成 ===="
