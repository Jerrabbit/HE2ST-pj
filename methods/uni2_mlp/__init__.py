"""UNI2+MLP 基线及其改进版（variant='improved'）。

- baseline：统一 MLPHead（1536→512→256→313）。
- improved：bias 均值初始化（ST-Net）+ 残差连接 + SiLU（参考其它模型 MLP 设计）。
"""
from __future__ import annotations

import torch

from common.benchmark.harness import fit
from .model import FEATURE_DIM, UNI2MLP, UNI2MLPImproved

__all__ = ["UNI2MLP", "UNI2MLPImproved", "FEATURE_DIM"]


def build_model(num_genes: int = 313, variant: str = "baseline", **kwargs):
    if variant == "improved":
        return UNI2MLPImproved(num_genes=num_genes, **kwargs)
    return UNI2MLP(num_genes=num_genes, **kwargs)


def train_function(model, train_loader, valid_loader, args, stats) -> dict:
    """UNI2+MLP 训练：改进版先做 bias 均值初始化，再统一 fit（含早停）。"""
    if isinstance(model, UNI2MLPImproved):
        total, n = None, 0
        with torch.no_grad():
            for batch in train_loader:
                e = batch["gene_expr"]
                total = e.sum(0) if total is None else total + e.sum(0)
                n += e.size(0)
        mean_expr = total / max(n, 1)
        model.set_bias_init(mean_expr)
        print(f"[UNI2MLP-improved] bias 初始化为训练集平均表达", flush=True)

    return fit(
        model, train_loader, valid_loader, args.epochs, args.lr, args.device,
        out_dir=args.output_dir, weight_decay=args.weight_decay,
        gene_norm=args.gene_norm, eval_stats=stats,
        config={"method": "uni2_mlp", "num_genes": model.num_genes,
                "gene_norm": args.gene_norm},
        patience=args.patience,
    )
