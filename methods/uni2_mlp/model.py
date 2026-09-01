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

from common.models.mlp_head import MLPHead, RefMLPHead

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


class ResidualBlock(nn.Module):
    """残差 MLP 块（参考 Pixel2Gene ForwardSum 的残差设计）。"""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class UNI2MLPImproved(nn.Module):
    """UNI2+MLP **改进版**：针对 MLP 的改进，参考其它模型设计。

    改进点（相对统一 MLPHead）：
        1. **bias 初始化为训练集平均表达**（ST-Net 关键技巧，官方 run_spatial.py 的
           last.bias = mean_expression）——给预测一个强起点；
        2. **残差连接**（Pixel2Gene ForwardSum 的 residual FeedForward）；
        3. **SiLU 激活**（Phoenix/STFlow 等现代架构常用，优于 LeakyReLU）。

    架构：
        Linear(1536→768) → SiLU → Dropout
        → ResidualBlock(768) × 2
        → Linear(768→313)，bias = 训练集平均表达
    """

    input_type = "feature"

    def __init__(
        self,
        num_genes: int,
        hidden_dim: int = 768,
        n_res_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_genes = num_genes
        layers = [
            nn.Linear(FEATURE_DIM, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        ]
        for _ in range(n_res_blocks):
            layers.append(ResidualBlock(hidden_dim, dropout))
        layers.append(nn.Linear(hidden_dim, num_genes))
        self.head = nn.Sequential(*layers)

    @torch.no_grad()
    def set_bias_init(self, mean_expr: torch.Tensor) -> None:
        """最后一层 bias = 训练集平均表达（ST-Net 关键技巧）。"""
        last = self.head[-1]
        last.bias.copy_(mean_expr.to(last.bias.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 (B, 1536) UNI2 特征 → 输出 (B, num_genes) 归一化表达预测。"""
        return self.head(x)


class UNI2MLPRef(nn.Module):
    """UNI2+MLP **参考架构版**（methods/uni2_mlp/MLP架构参考.txt，已验证更好性能）。

    改进：①特征提取加 LayerNorm（CLS/中心 token 各做 LN，见 extract 的 layer_norm）；
         ②MLP 头 = RefMLPHead（LayerNorm→512→GELU→Dropout→313→Softplus）。
    **Softplus 输出恒正 → 训练/评测用 gene_norm=log1p（正数目标空间）**。
    """

    input_type = "feature"

    def __init__(self, num_genes: int, hidden_dim: int = 512, dropout: float = 0.1,
                 use_softplus: bool = True):
        super().__init__()
        self.num_genes = int(num_genes)
        self.head = RefMLPHead(FEATURE_DIM, hidden_dim, self.num_genes, dropout,
                               use_softplus)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 (B, 1536) UNI2 特征 → 输出 (B, num_genes) 表达预测（恒正若 Softplus）。"""
        return self.head(x)
