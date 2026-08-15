"""Denoiser（官方 STFlow stflow/model/denoiser.py 纯 torch 移植）。

- TimestepEmbedder：正弦时间嵌入 + MLP。
- Denoiser：SpatialTransformer backbone，条件特征经 image_transform 投影后与
  时间嵌入相加，作为 transformer 的 token 特征。
架构与官方一致（无预训练权重，从头训）。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .config import SpatialConfig
from .transformer import SpatialTransformer


class TimestepEmbedder(nn.Module):
    """正弦时间嵌入 + MLP。"""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[..., None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class Denoiser(nn.Module):
    def __init__(self, config) -> None:
        super(Denoiser, self).__init__()

        # 外层 config（feature_dim/hidden_dim/pairwise_hidden_dim 等）→ 内层 SpatialConfig
        self.backbone = SpatialTransformer(
            SpatialConfig(
                n_genes=config.n_genes,
                feature_dim=config.feature_dim,
                hidden_dim=config.hidden_dim,
                d_edge_model=config.pairwise_hidden_dim,
                n_layers=config.n_layers,
                n_heads=config.n_heads,
                dropout=config.dropout,
                attn_dropout=config.attn_dropout,
                n_neighbors=config.n_neighbors,
                act=config.activation,
            )
        )
        self.loss_func = nn.MSELoss()

        self.fourier_proj = TimestepEmbedder(config.hidden_dim)
        self.image_transform = nn.Linear(config.feature_dim, config.hidden_dim)

    def inference(self, noisy_exp, img_features, coords, t_steps, predict=False):
        # noisy_exp: [B, n_cells, n_genes]
        # img_features: [B, n_cells, n_features]
        # coords: [B, n_cells, 2]
        # t_steps: [B]
        img_features = self.image_transform(img_features)
        time_emb = self.fourier_proj(t_steps)[:, None].expand(
            noisy_exp.shape[0], noisy_exp.shape[1], -1
        )
        features = img_features + time_emb

        prediction = self.backbone(
            gene_exp=noisy_exp,
            features=features,
            coords=coords,
        )
        return prediction

    def forward(self, exp, img_features, coords, labels, t_steps):
        prediction = self.inference(exp, img_features, coords, t_steps)
        pad_mask = img_features.sum(-1) == 0
        loss = self.loss_func(prediction[~pad_mask], labels[~pad_mask])
        return prediction, loss
