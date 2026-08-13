#!/bin/bash
# 部署：把本地 HE2ST-pj 代码 scp 到远程 gpu-server（远程非 git 克隆，需直接拷贝）。
# 用法：bash scripts/deploy.sh
#
# 前置：远程主机 key 需在 ~/.ssh/known_hosts（若提示 host key 变更，先人工核对指纹）：
#     ssh-keygen -R 10.193.2.99 -f ~/.ssh/known_hosts   # 移除旧条目
#     ssh-keyscan -H 10.193.2.99 >> ~/.ssh/known_hosts  # 添加新 key
set -euo pipefail

LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_ROOT="~/HE2ST-pj"
HOST="gpu-server"

# 逐目录 scp（Windows 无 rsync，用 scp + 排除 __pycache__）
for dir in common scripts methods tests; do
    echo "[deploy] scp $dir -> $HOST:$REMOTE_ROOT/$dir ..."
    ssh "$HOST" "mkdir -p $REMOTE_ROOT/$dir"
    scp -r -q "$LOCAL_ROOT/$dir/" "$HOST:$REMOTE_ROOT/$dir/"
done
echo "[deploy] 完成。"

echo ""
echo "远程运行（相邻切片情形，示例）："
echo "    cd ~/HE2ST-pj"
echo "    python scripts/train.py --method uni2_mlp --train_dir ~/HE2ST-pj/data/rep1 \\"
echo "        --valid_dir ~/HE2ST-pj/data/rep2 --epochs 50 --lr 1e-3 \\"
echo "        --gene_norm log1p_zscore --output_dir ~/HE2ST-pj/outputs/uni2_mlp"
