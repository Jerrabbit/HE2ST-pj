#!/bin/bash
# 全数据集预处理编排：Rep1 patches(已在跑) → Rep1 UNI2 features → Rep2 patches → Rep2 UNI2 features
# 远程运行：nohup bash scripts/run_preprocess_all.sh > /dev/null 2>&1 < /dev/null &
# 日志：~/HE2ST-pj/logs/preprocess_all.log
set -u
LOG=~/HE2ST-pj/logs/preprocess_all.log
exec > "$LOG" 2>&1

PATCHES_LOG=~/HE2ST-pj/logs/preprocess_rep1_patches.log

echo "[$(date)] 等待 Rep1 patches 进程结束 ..."
while pgrep -f 'preprocess_he.py --rep 1 --stage patches' > /dev/null; do sleep 30; done
if ! grep -q '已写入' "$PATCHES_LOG" 2>/dev/null; then
    echo "[$(date)] ERROR: Rep1 patches 未正常完成，终止。"
    exit 1
fi
echo "[$(date)] Rep1 patches 完成"

echo "[$(date)] 开始 Rep1 UNI2 features ..."
python ~/HE2ST-pj/scripts/preprocess_he.py --rep 1 --stage features \
    --data_dir ~/HE2ST-pj/data/rep1 --output ~/HE2ST-pj/data/rep1/X_uni2.npy
if [ $? -ne 0 ]; then echo "[$(date)] ERROR: Rep1 features 失败"; exit 1; fi
echo "[$(date)] Rep1 features 完成"

echo "[$(date)] 开始 Rep2 patches ..."
python ~/HE2ST-pj/scripts/preprocess_he.py --rep 2 --stage patches --output_dir ~/HE2ST-pj/data/rep2
if [ $? -ne 0 ]; then echo "[$(date)] ERROR: Rep2 patches 失败"; exit 1; fi
echo "[$(date)] Rep2 patches 完成"

echo "[$(date)] 开始 Rep2 UNI2 features ..."
python ~/HE2ST-pj/scripts/preprocess_he.py --rep 2 --stage features \
    --data_dir ~/HE2ST-pj/data/rep2 --output ~/HE2ST-pj/data/rep2/X_uni2.npy
if [ $? -ne 0 ]; then echo "[$(date)] ERROR: Rep2 features 失败"; exit 1; fi
echo "[$(date)] ALL DONE"
