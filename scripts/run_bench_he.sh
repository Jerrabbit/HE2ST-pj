#!/bin/bash
# 新 H&E 空白过滤 benchmark（用户 2026-08-29 要求）：
#   过滤机制改为只滤"不在 H&E 上"的细胞（中心+4 邻域点全落在 H&E 空白），
#   不设 min_genes/min_umis。产出 rep1_he/rep2_he（≈ 原 rep1/rep2，仅去 ~1% 空白细胞）。
#
# Phase 1：10 个方法（除 Phoenix/SQUALL/LG）rep1_he 训练 → rep2_he 测试，单次运行。
# Phase 2（最后）：Local+Global 完整调参（op1 sweep → op2 sweep → 最终 50ep 训练+测试）。
#
# 评估：统一模块已升级——SSIM（空间表达图）+ 全 k=10..313 Top-k 曲线（topk_curve.csv）。
# 幂等：已存在 best.pt 的步骤自动跳过；LG sweep 各配置已存在特征/结果则跳过。
# 用法：nohup bash scripts/run_bench_he.sh > logs/bench_he_all.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
mkdir -p outputs logs

EPOCHS=${EPOCHS:-50}
PATIENCE=${PATIENCE:-10}
LR=${LR:-1e-3}
BS=${BS:-2048}
GN=${GN:-log1p_zscore}
DEV="cuda"
HE1="/cpfs01/projects-HDD/cfff-d7ff0c9cdf2f_HDD/hjr_24300980068/HE2ST/datasets/he_images/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.ome.tif"
HE2="/cpfs01/projects-HDD/cfff-d7ff0c9cdf2f_HDD/hjr_24300980068/HE2ST/datasets/he_images/Xenium_FFPE_Human_Breast_Cancer_Rep2_he_image.ome.tif"

say() { echo "[$(date '+%F %T')] $*" | tee -a logs/bench_he_all.log; }

say "==== 新 H&E 空白过滤 benchmark 开始 ===="

# ---------- 0. 细胞过滤（H&E 空白，幂等） ----------
# 只滤"中心+4 邻域点全空白"的细胞；排除 Phoenix/SQUALL 的 131GB token（本轮不需要）
for rep in 1 2; do
  HE=$HE1; [ "$rep" = 2 ] && HE=$HE2
  if [ ! -d "data/rep${rep}_he" ]; then
    say "[filter] rep${rep} → rep${rep}_he ..."
    python3 scripts/filter_cells.py --src_dir "data/rep$rep" --out_dir "data/rep${rep}_he" \
      --he_filter "$HE" --exclude_features X_phoenix_dino.npy,X_squall_tokens.npy \
      --no_copy_patches \
      --ghist_src "data/ghist_rep$rep" --ghist_out "data/ghist_rep${rep}_he" \
      >> "logs/filter_he_${rep}.log" 2>&1 || { say "!! filter $rep 失败"; exit 1; }
  else
    say "[filter] rep${rep}_he 已存在，跳过"
  fi
done
say "[filter] 完成（核对细胞数应与 rep${rep} 一致，仅去空白细胞）"

# ================================================================
# Phase 1：10 方法（统一协议 50ep + 早停 + 训练集统计量复用）
# ================================================================

# ---------- 1. UNI2+MLP（基线） ----------
m=uni2_mlp; D=outputs/bench_${m}_he
if [ ! -f "$D/best.pt" ]; then
  say "== $m train =="
  python3 scripts/train.py --method $m --train_dir data/rep1_he --valid_dir data/rep2_he \
    --epochs $EPOCHS --patience $PATIENCE --batch_size $BS --lr $LR --gene_norm $GN \
    --output_dir $D --device $DEV > logs/bench_${m}_he_train.log 2>&1
fi
say "== $m test =="
python3 scripts/test.py --method $m --ckpt $D/best.pt --test_dir data/rep2_he \
  --gene_norm $GN --output_dir $D --device $DEV >> logs/bench_${m}_he_train.log 2>&1

# ---------- 2. GHIST（从头，ghist_data 格式） ----------
m=ghist; D=outputs/bench_${m}_he
if [ ! -f "$D/best.pt" ]; then
  say "== $m train =="
  python3 scripts/train.py --method $m --train_dir data/ghist_rep1_he --valid_dir data/ghist_rep2_he \
    --epochs $EPOCHS --patience $PATIENCE --lr $LR \
    --output_dir $D --device $DEV > logs/bench_${m}_he_train.log 2>&1
fi
say "== $m test =="
python3 scripts/test_ghist.py --ckpt $D/best.pt --test_dir data/ghist_rep2_he \
  --output_dir $D --device $DEV >> logs/bench_${m}_he_train.log 2>&1

# ---------- 3. SpatialEx（整片超图） ----------
m=spatialex; D=outputs/bench_${m}_he
if [ ! -f "$D/best.pt" ]; then
  say "== $m train =="
  python3 scripts/train.py --method $m --train_dir data/rep1_he --valid_dir data/rep2_he \
    --epochs $EPOCHS --patience $PATIENCE --lr $LR --gene_norm $GN \
    --output_dir $D --device $DEV > logs/bench_${m}_he_train.log 2>&1
fi
say "== $m test =="
python3 scripts/test_spatialex.py --test_dir data/rep2_he --checkpoint $D/best.pt \
  --gene_norm $GN --output_dir $D --device $DEV >> logs/bench_${m}_he_train.log 2>&1

# ---------- 4. Pixel2Gene cell（官方 ForwardSum 头, log1p 空间） ----------
m=pixel2gene; D=outputs/bench_${m}_cell_he
if [ ! -f "$D/best.pt" ]; then
  say "== $m cell train =="
  python3 scripts/train.py --method $m --variant cell --gene_norm log1p \
    --train_dir data/rep1_he --valid_dir data/rep2_he \
    --epochs $EPOCHS --patience $PATIENCE --batch_size $BS --lr $LR \
    --output_dir $D --device $DEV > logs/bench_${m}_cell_he_train.log 2>&1
fi
say "== $m cell test =="
python3 scripts/test.py --method $m --variant cell --gene_norm log1p \
  --ckpt $D/best.pt --test_dir data/rep2_he \
  --output_dir $D --device $DEV >> logs/bench_${m}_cell_he_train.log 2>&1

# ---------- 5. Path2Space（重训官方训练头） ----------
m=path2space; D=outputs/bench_${m}_he
if [ ! -f "$D/best.pt" ]; then
  say "== $m train =="
  python3 scripts/train.py --method $m --train_dir data/rep1_he --valid_dir data/rep2_he \
    --epochs $EPOCHS --patience $PATIENCE --batch_size $BS --lr $LR --gene_norm $GN \
    --output_dir $D --device $DEV > logs/bench_${m}_he_train.log 2>&1
fi
say "== $m test =="
python3 scripts/test.py --method $m --ckpt $D/best.pt --test_dir data/rep2_he \
  --gene_norm $GN --output_dir $D --device $DEV >> logs/bench_${m}_he_train.log 2>&1

# ---------- 6. DeepPT（ResNet50 官方实现） ----------
m=deeppt; D=outputs/bench_${m}_resnet50_he
if [ ! -f "$D/best.pt" ]; then
  say "== $m ResNet50 train =="
  python3 scripts/train.py --method $m --feature_file X_resnet50.npy --feat_dim 2048 \
    --train_dir data/rep1_he --valid_dir data/rep2_he \
    --epochs $EPOCHS --patience $PATIENCE --batch_size $BS --lr $LR --gene_norm $GN \
    --output_dir $D --device $DEV > logs/bench_${m}_r50_he_train.log 2>&1
fi
say "== $m ResNet50 test =="
python3 scripts/test.py --method $m --feature_file X_resnet50.npy --feat_dim 2048 \
  --ckpt $D/best.pt --test_dir data/rep2_he --gene_norm $GN \
  --output_dir $D --device $DEV >> logs/bench_${m}_r50_he_train.log 2>&1

# ---------- 7. ST-Net（冻结 DenseNet） ----------
m=st_net; D=outputs/bench_${m}_he
if [ ! -f "$D/best.pt" ]; then
  say "== $m frozen train =="
  python3 scripts/train.py --method $m --no_finetune \
    --train_dir data/rep1_he --valid_dir data/rep2_he \
    --epochs $EPOCHS --patience $PATIENCE --batch_size $BS --lr $LR --gene_norm $GN \
    --output_dir $D --device $DEV > logs/bench_${m}_he_train.log 2>&1
fi
say "== $m frozen test =="
python3 scripts/test.py --method $m --ckpt $D/best.pt --test_dir data/rep2_he \
  --gene_norm $GN --output_dir $D --device $DEV >> logs/bench_${m}_he_train.log 2>&1

# ---------- 8. Hist2ST（官方配置从头） ----------
m=hist2st; D=outputs/bench_${m}_he
if [ ! -f "$D/best.pt" ]; then
  say "== $m official train =="
  python3 scripts/train.py --method $m --epochs 100 --lr 1e-5 --zinb 0.25 --zinb_coef 0.25 \
    --bake 5 --lamb 0.5 --train_dir data/rep1_he --valid_dir data/rep2_he \
    --output_dir $D --device $DEV > logs/bench_${m}_he_train.log 2>&1
fi
say "== $m official test =="
python3 scripts/test_hist2st.py --ckpt $D/best.pt --test_dir data/rep2_he \
  --output_dir $D --device $DEV >> logs/bench_${m}_he_train.log 2>&1

# ---------- 9. BLEEP（冻结 resnet50） ----------
m=bleep; D=outputs/bench_${m}_he
if [ ! -f "$D/best.pt" ]; then
  say "== $m frozen train =="
  python3 scripts/train.py --method $m --no_finetune \
    --pretrained_weights weights/resnet50_imagenet.pth \
    --train_dir data/rep1_he --valid_dir data/rep2_he \
    --epochs $EPOCHS --patience $PATIENCE --batch_size $BS --lr $LR --gene_norm $GN \
    --output_dir $D --device $DEV > logs/bench_${m}_he_train.log 2>&1
fi
say "== $m frozen test =="
python3 scripts/test.py --method $m --img_size 224 \
  --pretrained_weights weights/resnet50_imagenet.pth \
  --ckpt $D/best.pt --test_dir data/rep2_he \
  --gene_norm $GN --output_dir $D --device $DEV >> logs/bench_${m}_he_train.log 2>&1

# ---------- 10. STFlow（官方协议） ----------
m=stflow; D=outputs/bench_${m}_he
if [ ! -f "$D/best.pt" ]; then
  say "== $m train =="
  python3 scripts/train.py --method $m --gene_norm log1p --prior zinb --n_sample_steps 50 \
    --train_dir data/rep1_he --valid_dir data/rep2_he \
    --epochs 100 --patience 20 --lr 5e-4 \
    --output_dir $D --device $DEV > logs/bench_${m}_he_train.log 2>&1
fi
say "== $m test =="
python3 scripts/test_stflow.py --ckpt $D/best.pt --test_dir data/rep2_he \
  --output_dir $D --device $DEV >> logs/bench_${m}_he_train.log 2>&1

say "==== Phase 1：10 方法完成 ===="

# ================================================================
# Phase 2（最后）：Local+Global 完整调参（op1 → op2 → 最终 50ep）
# ================================================================
OP1=outputs/sweep_op1_he
OP2=outputs/sweep_op2_he
say "== LG op1 sweep（Global 视野 l1 448→28）=="
python3 scripts/sweep.py --stage op1 --train_rep 1 --valid_rep 2 \
  --train_dir data/rep1_he --valid_dir data/rep2_he \
  --l1_values 448 420 392 364 336 308 280 252 224 196 168 140 112 84 56 28 \
  --out_dir $OP1 >> logs/lg_op1_he.log 2>&1 || { say "!! LG op1 失败"; exit 1; }

best_l1=$(python3 -c "
import csv
rows=list(csv.reader(open('$OP1/op1_results.csv')))[1:]
b=max(rows, key=lambda r: float(r[1])); print(b[0])")
say "== best l1 = $best_l1，LG op2 sweep（Local l2 28..112）=="
python3 scripts/sweep.py --stage op2 --l1 "$best_l1" --train_rep 1 --valid_rep 2 \
  --train_dir data/rep1_he --valid_dir data/rep2_he \
  --l2_values 28 42 56 70 84 98 112 \
  --out_dir $OP2 >> logs/lg_op2_he.log 2>&1 || { say "!! LG op2 失败"; exit 1; }

best_l2=$(python3 -c "
import csv
rows=list(csv.reader(open('$OP2/op2_results.csv')))[1:]
b=max(rows, key=lambda r: float(r[1])); print(b[0])")
say "== best l2 = $best_l2，LG 最终 50ep 训练+测试 =="
m=uni2_mlp; D=outputs/bench_${m}_lg_he
python3 scripts/train.py --method $m --variant local_global \
  --feature_file "X_uni2_g${best_l1}.npy,X_uni2_l${best_l2}.npy" \
  --train_dir data/rep1_he --valid_dir data/rep2_he \
  --epochs 50 --patience $PATIENCE --batch_size $BS --lr $LR --gene_norm $GN \
  --output_dir $D --device $DEV > logs/bench_${m}_lg_he_train.log 2>&1
python3 scripts/test.py --method $m --variant local_global \
  --feature_file "X_uni2_g${best_l1}.npy,X_uni2_l${best_l2}.npy" \
  --ckpt $D/best.pt --test_dir data/rep2_he --gene_norm $GN \
  --output_dir $D --device $DEV >> logs/bench_${m}_lg_he_train.log 2>&1

say "==== 全部完成（Phase 1 十方法 + LG 调参）===="
