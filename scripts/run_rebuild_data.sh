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
# 快路径：extract_local_global --stage global --l1 256 == UNI2 on 256×256 中心 patch
# （resize 224 → CLS），但直接从 HE 图分块裁剪+并行，避免 np.stack 16 万张 PNG 的慢 IO。
for rep in 1 2; do
  OUT=data/rep$rep/X_uni2.npy
  if [ ! -f "$OUT" ]; then
    say "[uni2(快路径 l1=256)] rep$rep ..."
    python scripts/extract_local_global.py --rep "$rep" --stage global --l1 256 \
      --output "$OUT" --device cuda >> "$LOG" 2>&1 || { say "!! rep$rep uni2 失败"; exit 1; }
  else
    say "[uni2] rep$rep 已存在，跳过"
  fi
done

# 注：X_uni2_g{l1}（Local+Global op1 sweep）不在此预提取——等过滤后 LG sweep 在
# rep1_f/rep2_f 上按需提取（sweep._extract_global 自动做，从过滤后 metadata 裁剪）。
# X_uni2_l{l2}（Local）依赖裁剪源 l1，由 op2 sweep 在定出 best l1 后提取。

# ---------- 4. CTransPath ctx512（Path2Space，权重在 cpfs） ----------
# extract_ctranspath_context.py 无 --data_dir：data_dir 由 --rep 推导（~/HE2ST-pj/data/rep{N}）
for rep in 1 2; do
  OUT=data/rep$rep/X_ctranspath_ctx512.npy
  if [ ! -f "$OUT" ]; then
    say "[ctranspath] rep$rep ..."
    python scripts/extract_ctranspath_context.py --rep "$rep" \
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

# ---------- 7. HIPT cell 特征（Pixel2Gene cell；权重到位后自动补） ----------
if [ -f weights/pixel2gene/vit_256_small_dino.pth ]; then
  for rep in 1 2; do
    OUT=data/rep$rep/X_hipt_cell.npy
    if [ ! -f "$OUT" ]; then
      say "[hipt-cell] rep$rep ..."
      python scripts/extract_hipt_cell.py --rep "$rep" \
        --ckpt weights/pixel2gene/vit_256_small_dino.pth --device cuda >> "$LOG" 2>&1 \
        || { say "!! rep$rep hipt-cell 失败"; exit 1; }
    fi
  done
else
  say "[skip] HIPT 权重缺失，跳过 X_hipt_cell.npy（Pixel2Gene cell）"
fi

# ---------- 8. SQUALL 特征（解码器头；权重到位后自动补） ----------
if [ -f weights/squall/SQUALL_full.pth ]; then
  for rep in 1 2; do
    OUT=data/rep$rep/X_squall_tokens.npy
    if [ ! -f "$OUT" ]; then
      say "[squall-tokens] rep$rep ..."
      python scripts/extract_squall.py --rep "$rep" \
        --ckpt weights/squall/SQUALL_full.pth --save_tokens --device cuda >> "$LOG" 2>&1 \
        || { say "!! rep$rep squall 失败"; exit 1; }
    fi
  done
else
  say "[skip] SQUALL 权重缺失，跳过 X_squall_tokens.npy"
fi

say "==== 数据重建完成（$([ -f weights/pixel2gene/vit_256_small_dino.pth ] && echo 含HIPT || echo 缺HIPT)、$([ -f weights/squall/SQUALL_full.pth ] && echo 含SQUALL || echo 缺SQUALL)）===="
