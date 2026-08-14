"""DeepPT：预训练 ResNet50 + 自编码器压缩 + MLP 预测（spot/cell 级适配，方案 A）。

官方代码：D:\\hest_data\\codes\\DeepPT（12AE/model_AE.py、13DeepPT_train/model_MLP.py）
官方流程（slide-level bulk）：ResNet50 tile 特征 → AE 压缩到 512 → MLP(512→512→基因)
预测整片 bulk 转录组。

本仓库适配（README 方案 A，单细胞粒度）：
    UNI2 特征(1536) → AE 编码器压缩到 512 → 统一 MLPHead(512→512→256→G) 回归。
    AE 用重构损失在训练集 UNI2 特征上预训练（与官方 12AE 相同），再与 head 联合训练。
    MLP 头与 UNI2+MLP 基线统一（common/models/mlp_head.py，课题要求 6）。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from common.models.mlp_head import MLPHead

__all__ = ["DeepPTModel", "AutoEncoder"]


class AutoEncoder(nn.Module):
    """官方 model_AE.py 结构：encoder Linear(in→hidden)→ReLU，decoder Linear(hidden→in)→ReLU。"""

    def __init__(self, n_inputs: int = 1536, n_hiddens: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_inputs, n_hiddens),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(n_hiddens, n_inputs),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor):
        """x (B, in) → (recon (B,in), latent (B, hidden))。"""
        z = self.encoder(x)
        return self.decoder(z), z


class DeepPTModel(nn.Module):
    """AE 编码器 → 统一 MLP 头（spot/cell 级表达回归）。input_type='feature'。"""

    input_type = "feature"

    def __init__(
        self,
        num_genes: int,
        feat_dim: int = 1536,
        ae_dim: int = 512,
        hidden_dims: list[int] = (512, 256),
    ):
        super().__init__()
        self.num_genes = num_genes
        self.feat_dim = feat_dim
        self.ae = AutoEncoder(n_inputs=feat_dim, n_hiddens=ae_dim)
        self.head = MLPHead(
            input_dim=ae_dim, hidden_dims=hidden_dims, output_dim=num_genes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """UNI2 特征 (B, 1536) → (B, G) 归一化表达预测。"""
        z = self.ae.encoder(x)
        return self.head(z)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """AE 重构（预训练用）：(B, in) → (B, in)。"""
        return self.ae(x)[0]
