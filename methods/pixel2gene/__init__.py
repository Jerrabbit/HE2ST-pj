"""Pixel2Gene：HIPT-4K 层级特征 + 统一 MLP 回归（伪 Visium spot 级适配）。

特征由 scripts/extract_hipt.py 预生成（data_dir/X_hipt.npy，576 维/细胞）。
训练走标准 harness fit（input_type='feature' + 统一 MLPHead），无自定义 train_function。
"""
from __future__ import annotations

from .model import Pixel2GeneModel

__all__ = ["Pixel2GeneModel", "build_model"]


def build_model(num_genes: int = 313, **kwargs):
    return Pixel2GeneModel(num_genes=num_genes, **kwargs)
