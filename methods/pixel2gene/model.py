"""Pixel2Gene：HIPT-4K 层级特征 + 官方 ForwardSumModel 回归头（伪 Visium spot 级适配）。

官方：Pixel2Gene（Zhang et al. 2024）用 HIPT-4K（256→4096 层级 ViT）提取整片
组织学层级特征，从 H&E 回归预测基因表达。

本仓库适配（保留官方架构，仅数据/粒度层适配）：
    - 伪 Visium 分箱（100µm 六角网格）把细胞聚成 spot（官方 format_xenium/bin_pseudo_visium）；
    - 每个 spot 取 2048×2048 上下文区域（8×8 个 256 patch，真实的层级上下文），
      HIPT-4K 提取特征：concat[mean256(384), cls4k(192)] = 576 维（官方 asset_dict 同款）；
    - 每细胞继承其 spot 的 576 维特征，**预测头沿用官方 impute_filter_train.py 的
      ForwardSumModel**（net_lat 576→256×4 + net_out 256→G ELU 输出头）。
      课题要求 6 的统一 MLP 仅用于"只产生 embedding 需外接头"的方法，Pixel2Gene 自带头故沿用。
    - 特征由 scripts/extract_hipt.py 预生成 → data_dir/X_hipt.npy (N, 576)。

HIPT-4K 模型代码在 methods/pixel2gene/hipt/（官方原样移植，einops 已替换为原生 torch）。
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

from common.models.mlp_head import MLPHead

__all__ = ["Pixel2GeneModel", "Pixel2GeneCellModel", "ForwardSumModel",
           "FeedForward", "ELU", "HIPTFeatureDim"]

HIPTFeatureDim = 576  # concat[mean256 (384), cls4k (192)]


class FeedForward(nn.Module):
    """官方 impute_filter_train.py FeedForward：Linear + activation(+residual)。"""

    def __init__(self, n_inp: int, n_out: int, activation: nn.Module | None = None,
                 residual: bool = False):
        super().__init__()
        self.linear = nn.Linear(n_inp, n_out)
        self.activation = activation if activation is not None else nn.LeakyReLU(0.1, inplace=True)
        self.residual = residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        y = self.activation(y)
        if self.residual:
            y = y + x
        return y


class ELU(nn.Module):
    """官方 ELU(alpha,beta)：nn.ELU(alpha=alpha) + beta（原样复刻）。"""

    def __init__(self, alpha: float, beta: float):
        super().__init__()
        self.activation = nn.ELU(alpha=alpha, inplace=True)
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x) + self.beta


class ForwardSumModel(nn.Module):
    """官方 Pixel2Gene 预测头（impute_filter_train.py ForwardSumModel 的回归部分）。

    net_lat: 576→256→256→256→256（FeedForward, LeakyReLU）
    net_out: 256→G（FeedForward, ELU(0.01,0.01) 输出头）
    """

    def __init__(self, n_inp: int = HIPTFeatureDim, n_out: int = 313):
        super().__init__()
        self.net_lat = nn.Sequential(
            FeedForward(n_inp, 256),
            FeedForward(256, 256),
            FeedForward(256, 256),
            FeedForward(256, 256),
        )
        self.net_out = FeedForward(256, n_out, activation=ELU(alpha=0.01, beta=0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net_out(self.net_lat(x))


class Pixel2GeneModel(nn.Module):
    """HIPT-4K 特征 (B, 576) → 归一化表达 (B, G)。input_type='feature'。"""

    input_type = "feature"
    feature_file = "X_hipt.npy"

    def __init__(self, num_genes: int):
        super().__init__()
        self.num_genes = num_genes
        self.head = ForwardSumModel(n_inp=HIPTFeatureDim, n_out=num_genes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """HIPT 特征 (B, 576) → (B, G) 归一化表达预测。"""
        return self.head(x)


class Pixel2GeneCellModel(nn.Module):
    """Pixel2Gene **方案 B（cell-level）**：level-1 ViT-256 特征 + 统一 MLP。

    只用 HIPT 的 level-1（vit_256_small_dino），每细胞自己的 256×256 patch →
    [CLS] 384 维特征（X_hipt_cell.npy）。**真正的 per-cell 预测**（无 spot 内封顶），
    作为课题 cell-level benchmark 的 Pixel2Gene 版本。按课题要求 6 统一规则
    （纯 embedding 需外接头）接统一 MLPHead(384 → G)。
    """

    input_type = "feature"
    feature_file = "X_hipt_cell.npy"

    def __init__(self, num_genes: int, hidden_dims: list[int] = (512, 256),
                 dropout: float = 0.1):
        super().__init__()
        self.num_genes = num_genes
        self.head = MLPHead(
            input_dim=384, hidden_dims=hidden_dims,
            output_dim=num_genes, dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ViT-256 特征 (B, 384) → (B, G) 归一化表达预测。"""
        return self.head(x)


def build_hipt_4k(model256_path: str, model4k_path: str, device: str = "cuda"):
    """构造官方 HIPT_4K（加载 DINO 权重）。需要 hipt 目录在 sys.path。"""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "hipt"))
    from hipt_4k import HIPT_4K

    return HIPT_4K(
        model256_path=model256_path, model4k_path=model4k_path,
        device256=torch.device(device), device4k=torch.device(device),
    )
