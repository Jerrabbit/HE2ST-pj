#!/bin/bash
# 从原始数据重建 rep1/rep2 数据目录 + 特征（全部存 cpfs，经 ~/HE2ST-pj 符号链接）。
# 前提：~/HE2ST-pj -> /cpfs01/.../HE2ST-pj 符号链接已建；原始 h5ad/HE/outs 在 HE2ST/datasets。
# 缺失权重（HIPT/SQUALL）的步骤自动跳过，权重到位后重跑本脚本补齐。
# 用法：nohup bash scripts/run_rebuild_data.sh > logs/rebuild_data.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
mkdir -p logs data
LOG=logs/rebuild_data.log
say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

say "==== 数据重建开始 ===="

# ---------- 1. patches + 表达（rep1/rep2，无需权重） ----------
for rep in 1 2; do
  OUT=data/rep$rep
  if [ ! -f "$OUT/metadata.csv" ]; then
    say "[patches] rep$rep ..."
    python scripts/preprocess_he.py --rep "$rep" --stage patches --output_dir "$OUT" >> "$LOG" 2>&1 || { say "!! rep$rep patches 失败"; exit 1; }
  else
    say "[patches] rep$rep 已存在，跳过"
  fi
done

# ---------- 2. UNI2 特征（X_uni2.npy，UNI2 权重在 cpfs） ----------
for rep in 1 2; do
  OUT=data/rep$rep/X_uni2.npy
  if [ ! -f "$OUT" ]; then
    say "[uni2] rep$rep ..."
    python scripts/preprocess_he.py --rep "$rep" --stage features --data_dir "data/rep$rep" \
      --output "$OUT" >> "$LOG" 2>&1 || { say "!! rep$rep uni2 失败"; exit 1; }
  else
    say "[uni2] rep$rep 已存在，跳过"
  fi
done

# ---------- 3. Local+Global Global 特征（X_uni2_g{l1}，op1 sweep 需要全集） ----------
# 注意：X_uni2_l{l2}（Local）依赖裁剪源 l1，必须等过滤后 op1 sweep 确定 best l1
# 再用该 l1 提取（sweep 的 _extract_local 会自动做），这里不预提取。
L1S="448 420 392 364 336 308 280 252 224 196 168 140 112 84 56 28"
for rep in 1 2; do
  for l1 in $L1S; do
    F=data/rep$rep/X_uni2_g$l1.npy
    if [ ! -f "$F" ]; then
      say "[lg-global] rep$rep l1=$l1 ..."
      python scripts/extract_local_global.py --rep "$rep" --stage global --l1 "$l1" \
        --output "$F" --device cuda >> "$LOG" 2>&1 || { say "!! rep$rep l1=$l1 失败"; exit 1; }
    fi
  done
done

# ---------- 4. CTransPath ctx512（Path2Space，权重在 cpfs） ----------
for rep in 1 2; do
  OUT=data/rep$rep/X_ctranspath_ctx512.npy
  if [ ! -f "$OUT" ]; then
    say "[ctranspath] rep$rep ..."
    python scripts/extract_ctranspath_context.py --rep "$rep" --data_dir "data/rep$rep" \
      --output "$OUT" --device cuda >> "$LOG" 2>&1 || { say "!! rep$rep ctranspath 失败"; exit 1; }
  else
    say "[ctranspath] rep$rep 已存在，跳过"
  fi
done

# ---------- 5. ResNet50 特征（DeepPT） ----------
# extract_resnet50.py 无 --data_dir/--output；从 ~/HE2ST-pj/data/rep{N} 读 patches，
# 默认存 X_resnet50.npy，默认用 torchvision IMAGENET1K_V2。
for rep in 1 2; do
  OUT=data/rep$rep/X_resnet50.npy
  if [ ! -f "$OUT" ]; then
    say "[resnet50] rep$rep ..."
    python scripts/extract_resnet50.py --rep "$rep" --device cuda >> "$LOG" 2>&1 || { say "!! rep$rep resnet50 失败"; exit 1; }
  else
    say "[resnet50] rep$rep 已存在，跳过"
  fi
done

# ---------- 6. GHIST 数据（ghist_data 格式，outs 在 cpfs） ----------
for rep in 1 2; do
  OUT=data/ghist_rep$rep
  if [ ! -f "$OUT/cell_gene_matrix_filtered.csv" ]; then
    say "[ghist-data] rep$rep ..."
    python scripts/build_ghist_data.py --rep "$rep" >> "$LOG" 2>&1 || { say "!! rep$rep ghist 失败"; exit 1; }
  else
    say "[ghist-data] rep$rep 已存在，跳过"
  fi
done

# ---------- 7. HIPT / SQUALL（权重缺失则跳过，到位后重跑） ----------
say "[check] HIPT 权重: $([ -f weights/pixel2gene/vit_256_small_dino.pth ] && echo 有 || echo 缺)"
say "[check] SQUALL 权重: $([ -f weights/squall/SQUALL_full.pth ] && echo 有 || echo 缺)"

say "==== 数据重建完成（未含 HIPT/SQUALL 特征）===="
