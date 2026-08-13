"""UNI2+MLP 基线模型。

架构：UNI2(224×224 图像块) → 特征(1536) → 统一 MLPHead → 基因表达预测

本方法为特征输入（input_type='feature'），配合 FeatureDataset 使用：
数据管线 `scripts/preprocess_he.py --stage features` 已把 UNI2 特征存为
data_dir/X_uni2.npy (N, 1536)。

预期结论：仅利用 Foundation Model 特征 + 简单 MLP，即可在多个 Benchmark 中
稳定超过 SpatialEx、GHIST 等复杂空间模型。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from common.models.mlp_head import MLPHead

FEATURE_DIM = 1536  # UNI2 [CLS] token 维度


class UNI2MLP(nn.Module):
    """UNI2 特征 + 统一 MLP 头基线。

    参数：
        num_genes: 预测的公共基因数
        mlp_hidden_dims: MLP 隐藏层维度（与其它方法统一，默认 [512, 256]）
    """

    input_type = "feature"

    def __init__(
        self,
        num_genes: int,
        mlp_hidden_dims: list[int] = (512, 256),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_genes = num_genes
        self.head = MLPHead(FEATURE_DIM, list(mlp_hidden_dims), num_genes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 (B, 1536) UNI2 特征 → 输出 (B, num_genes) 归一化表达预测。"""
        return self.head(x)
