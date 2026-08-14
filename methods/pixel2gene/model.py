"""Pixel2Gene：HIPT-4K 层级特征 + 统一 MLP 回归（伪 Visium spot 级适配）。

官方：Pixel2Gene（Zhang et al. 2024）用 HIPT-4K（256→4096 层级 ViT）提取整片
组织学层级特征，从 H&E 回归预测基因表达。

本仓库适配（保留官方架构，仅数据/粒度层适配）：
    - 伪 Visium 分箱（100µm 六角网格）把细胞聚成 spot（官方 format_xenium/bin_pseudo_visium）；
    - 每个 spot 取 2048×2048 上下文区域（8×8 个 256 patch，真实的层级上下文），
      HIPT-4K 提取特征：concat[mean256(384), cls4k(192)] = 576 维（官方 asset_dict 同款）；
    - 每细胞继承其 spot 的 576 维特征，统一 MLPHead 回归（harness fit）。
    - 特征由 scripts/extract_hipt.py 预生成 → data_dir/X_hipt.npy (N, 576)。

HIPT-4K 模型代码在 methods/pixel2gene/hipt/（官方原样移植，einops 已替换为原生 torch）。
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

from common.models.mlp_head import MLPHead

__all__ = ["Pixel2GeneModel", "HIPTFeatureDim"]

HIPTFeatureDim = 576  # concat[mean256 (384), cls4k (192)]


class Pixel2GeneModel(nn.Module):
    """HIPT-4K 特征 (B, 576) → 归一化表达 (B, G)。input_type='feature'。"""

    input_type = "feature"
    feature_file = "X_hipt.npy"

    def __init__(
        self,
        num_genes: int,
        hidden_dims: list[int] = (512, 256),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_genes = num_genes
        self.head = MLPHead(
            input_dim=HIPTFeatureDim, hidden_dims=hidden_dims,
            output_dim=num_genes, dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """HIPT 特征 (B, 576) → (B, G) 归一化表达预测。"""
        return self.head(x)


def build_hipt_4k(model256_path: str, model4k_path: str, device: str = "cuda"):
    """构造官方 HIPT_4K（加载 DINO 权重）。需要 hipt 目录在 sys.path。"""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "hipt"))
    from hipt_4k import HIPT_4K

    return HIPT_4K(
        model256_path=model256_path, model4k_path=model4k_path,
        device256=torch.device(device), device4k=torch.device(device),
    )
