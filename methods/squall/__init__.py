"""SQUALL：冻结多模态基础模型特征 + 统一 MLP 回归（per-cell 适配）。

特征由 scripts/extract_squall.py 预生成（data_dir/X_squall.npy，1024 维/细胞）。
训练走标准 harness fit（input_type='feature' + 统一 MLPHead），无自定义 train_function。
"""
from __future__ import annotations

from .model import SQUALLModel

__all__ = ["SQUALLModel", "build_model"]


def build_model(num_genes: int = 313, **kwargs):
    return SQUALLModel(num_genes=num_genes, **kwargs)
