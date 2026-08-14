"""统一测试/评估入口：所有方法共用同一评估流程（PCC / SPCC / Top-k / AUROC）。

用法（远程服务器）：
    python scripts/test.py --method uni2_mlp --ckpt outputs/uni2_mlp/best.pt \
        --test_dir ~/HE2ST-pj/data/rep2 --gene_norm log1p_zscore
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根目录

import torch

from common.benchmark.harness import evaluate
from common.data.dataset import FeatureDataset, HESTDataset
from common.data.expression import load_stats_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HE→ST 统一测试/评估入口")
    p.add_argument("--method", required=True, help="方法名，对应 methods/ 下文件夹")
    p.add_argument("--ckpt", required=True, help="模型权重路径（best.pt）")
    p.add_argument("--test_dir", required=True, help="测试集数据目录")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--gene_norm", choices=["log1p_zscore", "log1p_norm_total", "none"],
                   default="log1p_zscore")
    p.add_argument("--gene_file", default=None, help="公共基因列表文件")
    p.add_argument("--stats_file", default=None,
                   help="训练集表达归一化统计量 json（None 时用测试集自身统计量拟合）")
    p.add_argument("--pretrained_weights", default=None,
                   help="BLEEP 等专属：backbone 预训练权重路径（远程无网时替代 timm HF 下载）")
    p.add_argument("--variant", default=None,
                   help="方法专属变体（如 pixel2gene cell/spot）")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("config") or {}

    mod = importlib.import_module(f"methods.{args.method}")
    build_kwargs = {}
    if args.method == "bleep" and args.pretrained_weights:
        build_kwargs["pretrained_weights"] = args.pretrained_weights
    if args.method == "pixel2gene" and args.variant:
        build_kwargs["variant"] = args.variant
    model = mod.build_model(num_genes=cfg.get("num_genes", 313), **build_kwargs)
    model.load_state_dict(ckpt["model"])
    if hasattr(mod, "post_load"):
        # 方法需要从 checkpoint 恢复非参数状态（如 BLEEP 的参考检索库）
        model = mod.post_load(model, ckpt)

    gene_list = None
    if args.gene_file:
        with open(args.gene_file) as f:
            gene_list = [line.strip() for line in f if line.strip()]

    ref_stats = None
    if args.stats_file and os.path.exists(args.stats_file):
        ref_stats = load_stats_json(args.stats_file)
    elif ref_stats is None:
        # 默认复用同输出目录下的训练集统计量（防止在测试集上自拟合泄漏）
        train_stats = os.path.join(args.output_dir, "train_stats.json")
        if os.path.exists(train_stats):
            ref_stats = load_stats_json(train_stats)

    if getattr(model, "input_type", "patch") == "feature":
        feature_file = getattr(model, "feature_file", None)  # 方法自带特征文件（如 X_ctranspath.npy）
        ds = FeatureDataset(args.test_dir, feature_path=feature_file,
                            gene_list=gene_list, gene_norm=args.gene_norm,
                            ref_stats=ref_stats)
    else:
        ds = HESTDataset(args.test_dir, gene_list=gene_list, gene_norm=args.gene_norm,
                         ref_stats=ref_stats)

    from torch.utils.data import DataLoader
    # 测试是单趟评估，num_workers=0 避免多进程 IPC 在受限 shell 下触发
    # "Too many open files"（远程 nohup/ssh 环境文件描述符上限可能较低）
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    stats = ref_stats or ds.stats
    results = evaluate(model, loader, args.device, args.gene_norm, stats)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
