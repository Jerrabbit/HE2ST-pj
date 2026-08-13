"""SpatialEx 专用评估：加载 best.pt → 整片超图推理 → 统一指标。

用法（远程服务器）：
    python scripts/test_spatialex.py --test_dir ~/HE2ST-pj/data/rep2 \
        --checkpoint outputs/spatialex/best.pt \
        --gene_norm log1p_zscore --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根目录

import torch

from common.data.expression import load_expression, normalize_expression
from methods.spatialex import build_model, evaluate_slide


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SpatialEx 整片超图评估")
    p.add_argument("--test_dir", required=True, help="测试集数据目录（含 X_uni2.npy）")
    p.add_argument("--checkpoint", default="outputs/spatialex/best.pt",
                   help="训练保存的 best.pt")
    p.add_argument("--gene_norm", choices=["log1p_zscore", "log1p_norm_total", "none"],
                   default="log1p_zscore", help="与训练一致的表达归一化方式")
    p.add_argument("--stats_path", default=None,
                   help="训练集拟合的归一化统计量 json（None = 在测试集上拟合）")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default="outputs/spatialex", help="结果输出目录")
    return p.parse_args()


def _load_stats(test_dir: str, gene_norm: str, stats_path: str | None) -> dict | None:
    """读取训练集拟合的归一化统计量；未提供则在测试集上自拟合。"""
    if stats_path:
        with open(stats_path) as f:
            return json.load(f)
    expr_raw, _ = load_expression(test_dir)
    _, stats = normalize_expression(expr_raw, gene_norm)
    return stats


def main() -> None:
    args = parse_args()
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"找不到 checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("config", {})
    num_genes = cfg.get("num_genes", None) or ckpt["model"]["predictor.linear.weight"].shape[0]
    in_dim = cfg.get("in_dim", ckpt["model"]["predictor.mlp.0.weight"].shape[1])
    hidden_dim = cfg.get("hidden_dim", ckpt["model"]["predictor.mlp.0.weight"].shape[0])
    num_layers = cfg.get("num_layers", 2)

    model = build_model(num_genes=num_genes, in_dim=in_dim, hidden_dim=hidden_dim,
                        num_layers=num_layers)
    model.load_state_dict(ckpt["model"])
    print(f"SpatialEx 模型就绪: num_genes={num_genes} in_dim={in_dim} "
          f"hidden_dim={hidden_dim} num_layers={num_layers}", flush=True)

    stats = _load_stats(args.test_dir, args.gene_norm, args.stats_path)
    results = evaluate_slide(model, args.test_dir, args.gene_norm, stats,
                             args.device, args.output_dir)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"结果已保存: {os.path.join(args.output_dir, 'test_results.json')}")


if __name__ == "__main__":
    main()
