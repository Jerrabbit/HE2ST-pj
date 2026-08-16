"""SQUALL：冻结多模态基础模型特征 + 统一 MLP 回归。

官方：SQUALL（Zhang et al. 2026）多模态病理基础模型（PLIP/UNI/Virchow 编码 +
Transformer + 模态混合专家 MoME），在大规模 HE-ST 上预训练后冻结推理预测基因表达。

本仓库适配：
    - 冻结 SQUALL_full.pth（zongxu/SQUALL），per-cell 224×224 patch → forward_rgb
      → (B, 196, 1024) token 嵌入 → mean-pool → 1024 维特征（X_squall.npy）；
    - SQUALL 对 313 基因无自带头（官方 decoder 输出 expr_chans=15757，与我们的
      公共基因不匹配，且架构必须与 checkpoint 严格一致），故按课题要求 6 统一规则
      接统一 MLPHead（1024 → 基因）。这正是"模型仅形成 embedding、需外接头"的情形。
    - 特征由 scripts/extract_squall.py 预生成 → data_dir/X_squall.npy (N, 1024)。

SQUALL 模型代码在 methods/squall/Squall.py（官方原样，utils/sksurv 等重依赖已 stub，
架构不变）。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from common.models.mlp_head import MLPHead

__all__ = ["SQUALLModel", "SQUALLDecoderHead", "SQUALLFeatureDim"]

SQUALLFeatureDim = 1024  # mean-pool 后的嵌入维度


class SQUALLModel(nn.Module):
    """SQUALL 特征 (B, 1024) → 归一化表达 (B, G)。input_type='feature'。"""

    input_type = "feature"
    feature_file = "X_squall.npy"

    def __init__(self, num_genes: int, hidden_dims: list[int] = (512, 256),
                 dropout: float = 0.1):
        super().__init__()
        self.num_genes = num_genes
        self.head = MLPHead(
            input_dim=SQUALLFeatureDim, hidden_dims=hidden_dims,
            output_dim=num_genes, dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SQUALL 特征 (B, 1024) → (B, G) 归一化表达预测。"""
        return self.head(x)


class SQUALLDecoderHead(nn.Module):
    """冻结 SQUALL 编码器 token 特征 + **训练**官方解码器头（TransformerDecoder → 313）。

    与 SQUALLModel（统一 MLP 于 mean-pool 1024 特征）对比：本模型在冻结编码器的
    **196×1024 token 嵌入**（`X_squall_tokens.npy`，`extract_squall.py --save_tokens`）
    上训练官方 `TransformerDecoder`（depth=4）+ 313 线性头 —— 即"训练解码器头"，
    按项目原则（编码器冻结、头训练）与统一 MLP 头公平对齐。

    结构：token (B,196,1024) → TransformerDecoder（训练）→ mean-pool tokens → Linear(1024→G)。
    """

    input_type = "feature"
    feature_file = "X_squall_tokens.npy"

    def __init__(
        self,
        num_genes: int,
        embed_dim: int = 1024,
        decoder_depth: int = 4,
        num_heads: int = 16,
        drop_path_rate: float = 0.1,
        if_rpb: bool = True,
    ):
        super().__init__()
        self.num_genes = int(num_genes)
        from methods.squall.transformer import TransformerDecoder
        self.decoder = TransformerDecoder(
            embed_dim=embed_dim, depth=decoder_depth, num_heads=num_heads,
            drop_path_rate=drop_path_rate, if_rpb=if_rpb,
        )
        self.head = nn.Linear(embed_dim, self.num_genes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """token 特征 (B, 196, 1024) → 解码器 → mean-pool → (B, G) 归一化表达预测。"""
        z = self.decoder(x)              # (B, 196, 1024)
        return self.head(z.mean(1))      # (B, G)
