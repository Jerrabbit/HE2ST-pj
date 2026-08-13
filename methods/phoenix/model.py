"""Phoenix：(Latent) Flow Matching 生成模型（按论文概念实现）。

论文：Pan-cancer virtual spatial transcriptomics from routine histology with
Phoenix（bioRxiv 2026）。**本地无官方代码**，本实现依据论文方法概念
（Latent Flow Matching 生成模型，由组织学图像推断单细胞基因表达）构建：

1. 表达自编码器：G 维表达 → L 维潜在 → G 维重构（latent representation）。
2. 潜在空间的 flow matching：以 UNI2 组织学特征为条件，学习把高斯噪声
   输送到目标潜在分布的时间相关向量场。
3. 推理：从噪声出发按 Euler 步 ODE 解到潜在，再经解码器还原表达。

训练（见 __init__.py 的 train_function）：
    总损失 = 重构损失（AE）+ λ·流匹配损失（velocity MSE）
    流匹配目标：t~U(0,1)，z_t = (1-t)·ε + t·z，v = z - ε，
    去噪器从 (z_t, t, image_feature) 预测 v。

适配本仓库：input_type='feature'（用 UNI2 特征作条件），
输出 (B, G) 归一化表达预测，与统一 harness 评估一致。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestepEmbedder(nn.Module):
    """正弦时间嵌入 + MLP（同 STFlow model/denoiser.py 的 TimestepEmbedder）。"""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t[..., None].float() * freqs
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[..., :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class ExpressionEncoder(nn.Module):
    """G 维表达 → L 维潜在（MLP）。"""

    def __init__(self, num_genes: int, latent_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_genes, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ExpressionDecoder(nn.Module):
    """L 维潜在 → G 维表达（MLP）。"""

    def __init__(self, latent_dim: int, hidden_dim: int, num_genes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_genes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class FlowDenoiser(nn.Module):
    """时间条件去噪器：从 (z_t, t, image_feature) 预测流速度 v。"""

    def __init__(self, latent_dim: int, hidden_dim: int, feature_dim: int):
        super().__init__()
        self.cond_proj = nn.Linear(feature_dim, hidden_dim)
        self.time_emb = TimestepEmbedder(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(
        self, z_t: torch.Tensor, t: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        c = self.cond_proj(condition)
        te = self.time_emb(t)
        x = torch.cat([z_t, c, te], dim=-1)
        return self.net(x)


class Phoenix(nn.Module):
    """Phoenix：潜在空间 flow matching 生成模型。

    参数：
        num_genes: 表达维度（公共基因数）
        feature_dim: 组织学特征维度（UNI2 = 1536）
        latent_dim: 潜在空间维度
        hidden_dim: 去噪器/编解码器隐藏维度
        flow_weight: 流匹配损失权重（相对重构损失）
        n_sample_steps: 推理 Euler 步数
        seed: 推理噪声的固定随机种子（保证评估可复现）
    """

    input_type = "feature"

    def __init__(
        self,
        num_genes: int,
        feature_dim: int = 1536,
        latent_dim: int = 128,
        hidden_dim: int = 256,
        flow_weight: float = 1.0,
        n_sample_steps: int = 20,
        seed: int = 0,
    ):
        super().__init__()
        self.num_genes = num_genes
        self.latent_dim = latent_dim
        self.flow_weight = flow_weight
        self.n_sample_steps = n_sample_steps
        self.seed = seed

        self.encoder = ExpressionEncoder(num_genes, latent_dim, hidden_dim)
        self.decoder = ExpressionDecoder(latent_dim, hidden_dim, num_genes)
        self.denoiser = FlowDenoiser(latent_dim, hidden_dim, feature_dim)

    # ---------- 训练侧 ----------
    def _flow_target(self, z: torch.Tensor, t: torch.Tensor, eps: torch.Tensor):
        z_t = (1 - t[:, None]) * eps + t[:, None] * z
        v = z - eps
        return z_t, v

    def training_loss(
        self,
        gene_expr: torch.Tensor,
        condition: torch.Tensor,
        lambd: float | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """计算重构损失 + 流匹配损失。

        返回 (总损失, 明细 dict)。
        """
        z = self.encoder(gene_expr)
        rec = self.decoder(z)
        rec_loss = F.mse_loss(rec, gene_expr)

        t = torch.rand(gene_expr.size(0), device=gene_expr.device)
        eps = torch.randn_like(z)
        z_t, v_target = self._flow_target(z, t, eps)
        v_pred = self.denoiser(z_t, t, condition)
        fm_loss = F.mse_loss(v_pred, v_target)

        lambd = lambd if lambd is not None else self.flow_weight
        total = rec_loss + lambd * fm_loss
        return total, {"rec_loss": rec_loss, "fm_loss": fm_loss}

    # ---------- 推理侧 ----------
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """条件特征 (B, D) → 采样生成表达 (B, G)。

        Euler 步 ODE：噪声 ε 起，沿学到的向量场解到 t=1，再解码。
        噪声用固定 seed 的 generator 采样，保证评估可复现。
        """
        device = x.device
        B = x.size(0)
        gen = torch.Generator(device=device).manual_seed(self.seed)
        z = torch.randn(B, self.latent_dim, device=device, generator=gen)

        ts = torch.linspace(0.0, 1.0, self.n_sample_steps + 1, device=device)
        for i in range(self.n_sample_steps):
            t = ts[i].expand(B)
            v = self.denoiser(z, t, x)
            z = z + (ts[i + 1] - ts[i]) * v
        return self.decoder(z)
