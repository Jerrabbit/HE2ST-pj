"""DeepPT：预训练 ResNet50 + 自编码器压缩 + MLP 预测（spot/cell 级适配，方案 A）。

官方代码：D:\\hest_data\\codes\\DeepPT（12AE/model_AE.py、13DeepPT_train/model_MLP.py）
官方流程（slide-level bulk）：ResNet50 tile 特征 → AE 压缩到 512 → MLP(512→512→基因)
预测整片 bulk 转录组。

本仓库适配（README 方案 A，单细胞粒度）：
    UNI2 特征(1536) → AE 编码器压缩到 512 → 官方 MLP_regression(512→512→G) 回归。
    AE 用重构损失在训练集 UNI2 特征上预训练（与官方 12AE 相同），再与 head 联合训练。
    **预测头沿用官方 model_MLP.py 的 MLP_regression**（Linear→Dropout→Linear，无激活），
    课题要求 6 的统一 MLP 仅用于"只产生 embedding 需外接头"的方法，DeepPT 自带头故沿用。
"""
from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["DeepPTModel", "AutoEncoder", "MLP_regression"]


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


class MLP_regression(nn.Module):
    """官方 13DeepPT_train/model_MLP.py MLP_regression（checkpoint 兼容，勿改结构）。

    Linear(n_inputs→n_hiddens) → Dropout → Linear(n_hiddens→n_outputs)。
    官方注释掉了 ReLU（2020.03.26 为正值表达设计）；无激活即为官方原样。
    """

    def __init__(self, n_inputs: int, n_hiddens: int, n_outputs: int,
                 dropout: float = 0.2, bias_init: torch.Tensor | None = None):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Linear(n_inputs, n_hiddens),
            nn.Dropout(dropout),
        )
        self.layer1 = nn.Linear(n_hiddens, n_outputs)
        if bias_init is not None:
            self.layer1.bias = bias_init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer1(self.layer0(x))


class DeepPTModel(nn.Module):
    """AE 编码器 → 官方 MLP_regression 头（spot/cell 级表达回归）。input_type='feature'。"""

    input_type = "feature"

    def __init__(
        self,
        num_genes: int,
        feat_dim: int = 1536,
        ae_dim: int = 512,
        head_hiddens: int = 512,
        head_dropout: float = 0.2,
    ):
        super().__init__()
        self.num_genes = num_genes
        self.feat_dim = feat_dim
        self.ae = AutoEncoder(n_inputs=feat_dim, n_hiddens=ae_dim)
        self.head = MLP_regression(
            n_inputs=ae_dim, n_hiddens=head_hiddens,
            n_outputs=num_genes, dropout=head_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """UNI2 特征 (B, 1536) → (B, G) 归一化表达预测。"""
        z = self.ae.encoder(x)
        return self.head(z)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """AE 重构（预训练用）：(B, in) → (B, in)。"""
        return self.ae(x)[0]
