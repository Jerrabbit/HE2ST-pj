"""Pixel2Gene：HIPT-4K 特征 + 回归（spot 级官方 / cell 级方案 B）。

- variant='spot'（默认）：官方伪 Visium spot 级，HIPT-4K 576 维 + 官方 ForwardSumModel 头。
- variant='cell'：方案 B，level-1 ViT-256 per-cell 384 维 + 官方 ForwardSumModel 头
  （n_inp=384 适配，与 spot 级同架构；官方有头不换统一 MLP）。
特征分别由 scripts/extract_hipt.py / scripts/extract_hipt_cell.py 预生成。
"""
from __future__ import annotations

from .model import Pixel2GeneModel, Pixel2GeneCellModel

__all__ = ["Pixel2GeneModel", "Pixel2GeneCellModel", "build_model"]


def build_model(num_genes: int = 313, variant: str = "spot", **kwargs):
    if variant == "cell":
        return Pixel2GeneCellModel(num_genes=num_genes, **kwargs)
    return Pixel2GeneModel(num_genes=num_genes, **kwargs)
