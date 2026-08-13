"""ST-Net：DenseNet121 + 线性输出层（bias=平均表达）。"""
from __future__ import annotations

import torch

from common.benchmark.harness import fit
from .model import STNet

__all__ = ["STNet"]


def build_model(num_genes: int = 313, **kwargs):
    return STNet(num_genes=num_genes, **kwargs)


def _compute_mean_expr(train_loader, device: str) -> torch.Tensor:
    """训练集平均表达（用于官方 bias 初始化，log1p 归一化空间）。"""
    total, n = None, 0
    with torch.no_grad():
        for batch in train_loader:
            e = batch["gene_expr"].to(device)
            if total is None:
                total = e.sum(0)
            else:
                total = total + e.sum(0)
            n += e.size(0)
    return total / max(n, 1)


def train_function(model, train_loader, valid_loader, args, stats) -> dict:
    """ST-Net 训练：先按官方初始化 bias=平均表达，再走统一 MSE 回归流程。"""
    model = model.to(args.device)
    mean_expr = _compute_mean_expr(train_loader, args.device)
    model.set_bias_init(mean_expr)
    print(f"[ST-Net] bias 初始化为训练集平均表达（{model.num_genes} 基因）",
          flush=True)

    return fit(
        model, train_loader, valid_loader, args.epochs, args.lr, args.device,
        out_dir=args.output_dir, weight_decay=args.weight_decay,
        gene_norm=args.gene_norm, eval_stats=stats,
        config={"method": "st_net", "num_genes": model.num_genes,
                "gene_norm": args.gene_norm},
    )
