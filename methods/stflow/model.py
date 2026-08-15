"""STFlow：整片切片流匹配（Flow Matching）生成模型（官方架构纯 torch 移植）。

官方：Huang et al. 2025（arXiv:2506.05361）。整片切片流匹配：
    - 病理基础模型（UNI）提取 tile 特征作为条件；
    - 去噪器（SpatialTransformer）拟合线性插值流 z_t = (1-t)·ε + t·z（z=表达，ε=噪声），
      速度场 v = z - ε，时间步嵌入；
    - 推理用 Euler 步 ODE 从高斯先验采样生成表达。

本仓库适配（保留官方架构，仅数据/粒度层）：
    - 用 UNI2 特征作条件（官方 feature_encoder=uni_v1_official 的等价替代，
      UNI2 1536 维，官方 gigapath 槽位 feature_dim=1536）。
    - 表达空间 = 本仓库统一归一化空间（log1p_zscore）；先验用高斯 N(0,1)。
    - **ROI 级**：用 common/data/slide_tiling.tile_rois 把细胞切成空间 ROI，
      每个 ROI 作为 SpatialTransformer 的一批（[1, N_cells, G] + 特征 + 坐标）。
    - 推理：从噪声起按 Euler 步解到 t=1，逐 ROI 预测，映射回 per-cell。

模型代码在 methods/stflow/（fa.py / transformer.py / denoiser.py 官方移植，
einops、torch_geometric、scvi 已替换为纯 torch / 高斯先验）。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import STFlowConfig
from .denoiser import Denoiser


class STFlow(nn.Module):
    """STFlow：表达空间的流匹配，SpatialTransformer 去噪器，ROI 级推理。"""

    input_type = "feature"

    def __init__(
        self,
        num_genes: int,
        feature_dim: int = 1536,
        hidden_dim: int = 256,
        edge_dim: int = 64,
        n_layers: int = 4,
        n_heads: int = 4,
        n_neighbors: int = 8,
        dropout: float = 0.1,
        n_sample_steps: int = 20,
        seed: int = 0,
    ):
        super().__init__()
        self.num_genes = num_genes
        self.feature_dim = feature_dim
        self.n_sample_steps = n_sample_steps
        self.seed = seed

        config = STFlowConfig(
            n_genes=num_genes, feature_dim=feature_dim, hidden_dim=hidden_dim,
            pairwise_hidden_dim=edge_dim, n_layers=n_layers, n_heads=n_heads,
            dropout=dropout, n_neighbors=n_neighbors,
        )
        self.denoiser = Denoiser(config)

    # ---------- 训练侧（ROI 级） ----------
    def flow_loss(
        self,
        gene_expr: torch.Tensor,   # (1, N, G) 归一化表达
        img_features: torch.Tensor,  # (1, N, F) UNI2 条件
        coords: torch.Tensor,        # (1, N, 2)
    ) -> tuple[torch.Tensor, dict]:
        """线性插值流匹配损失（官方 interpolant 语义，高斯先验，zscore 空间）。

        z_t = (1-t)·ε + t·z，v_target = z - ε；去噪器从 (z_t, t, 特征, 坐标) 预测 v。
        """
        B, N, G = gene_expr.shape
        z = gene_expr
        eps = torch.randn_like(z)
        t = torch.rand(B, device=gene_expr.device)          # 每个 ROI 一个 t
        z_t = (1 - t[:, None, None]) * eps + t[:, None, None] * z
        v_target = z - eps                                   # (B, N, G)

        v_pred = self.denoiser.inference(z_t, img_features, coords, t)
        loss = nn.functional.mse_loss(v_pred, v_target)
        return loss, {"flow_loss": loss}

    # ---------- 推理侧（ROI 级） ----------
    @torch.no_grad()
    def sample_roi(
        self,
        img_features: torch.Tensor,  # (1, N, F)
        coords: torch.Tensor,        # (1, N, 2)
        device: str = "cuda",
    ) -> torch.Tensor:
        """对一个 ROI 做 Euler ODE 采样：高斯噪声 → t=1 → (1, N, G) 归一化表达。"""
        B, N, F = img_features.shape
        gen = torch.Generator(device=device).manual_seed(self.seed)
        z = torch.randn(B, N, self.num_genes, device=device, generator=gen)

        ts = torch.linspace(0.0, 1.0, self.n_sample_steps + 1, device=device)
        for i in range(self.n_sample_steps):
            t = ts[i].expand(B)
            v = self.denoiser.inference(z, img_features, coords, t)
            z = z + (ts[i + 1] - ts[i]) * v
        return z
