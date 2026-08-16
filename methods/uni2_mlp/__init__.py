"""UNI2+MLP 基线及改进版。

变体（--variant）：
- baseline：统一 MLPHead（1536→512→256→313）。
- improved：bias 均值初始化（ST-Net）+ 残差连接 + SiLU（参考其它模型 MLP 设计）。
- local_global：Local-Global 双尺度改进，输入 concat[UNI2(Global l1), UNI2(Local l2)]
  → MLPHead（3072 维；特征文件由 --feature_file 逗号指定）。
- global_only / local_only：消融（只用 Global 或只用 Local，1536 维）。
"""
from __future__ import annotations

import torch

from common.benchmark.harness import fit
from .local_global import FEATURE_DIM, LocalGlobalMLP
from .model import UNI2MLP, UNI2MLPImproved

__all__ = ["UNI2MLP", "UNI2MLPImproved", "LocalGlobalMLP", "FEATURE_DIM"]


def build_model(num_genes: int = 313, variant: str = "baseline",
                l1: int = 512, l2: int = 56, in_dim: int | None = None, **kwargs):
    """统一模型工厂。variant 见模块 docstring。

    local_global / global_only / local_only 的 in_dim 由特征文件数决定
    （Local+Global=3072，消融=1536）；特征文件名由 train.py --feature_file 覆盖。
    """
    if variant == "improved":
        return UNI2MLPImproved(num_genes=num_genes, **kwargs)
    if variant in ("local_global", "global_only", "local_only"):
        n_files = {"local_global": 2, "global_only": 1, "local_only": 1}[variant]
        dim = in_dim or FEATURE_DIM * n_files
        return LocalGlobalMLP(num_genes=num_genes, in_dim=dim, l1=l1, l2=l2, **kwargs)
    return UNI2MLP(num_genes=num_genes, **kwargs)


def train_function(model, train_loader, valid_loader, args, stats) -> dict:
    """UNI2+MLP 训练：improved 版先做 bias 均值初始化，再统一 fit（含早停）。

    local_global 变体直接用统一 fit（MLP 头训练，编码器 UNI2 冻结）。
    """
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
                "gene_norm": args.gene_norm,
                "l1": getattr(model, "l1", None), "l2": getattr(model, "l2", None)},
        patience=args.patience,
    )
