"""STFlow：整片切片流匹配（Flow Matching）生成模型（官方架构/目标/推理对齐官方）。

官方：Huang et al. 2025（arXiv:2506.05361）。整片切片流匹配：
    - 病理基础模型（UNI）提取 tile 特征作为条件；
    - 去噪器（SpatialTransformer）拟合线性插值流 z_t = (1-t)·ε + t·z（z=表达，ε=噪声），
      **直接预测干净表达 z**（官方 denoiser.py: MSE(pred, labels=gene_exp)，非速度场）；
    - 推理按官方 test.py 语义：t 从 0.01 起，interpolant.denoise 步进，
      返回最后一步的模型预测 pred（不积分到 t=1）。

本仓库适配（保留官方架构，仅数据/粒度层）：
    - 用 UNI2 特征作条件（官方 feature_encoder=uni_v1_official 的等价替代，
      UNI2 1536 维，官方 gigapath 槽位 feature_dim=1536）。
    - 表达空间 = log1p（zinb 先验，官方 normalize 语义）；**313 基因**（保留 benchmark 可比性，
      官方为 50 HVG）。
    - **ROI 级**：用 common/data/slide_tiling.tile_rois 把细胞切成空间 ROI，
      每个 ROI 作为 SpatialTransformer 的一批（[1, N_cells, G] + 特征 + 坐标）。
    - 训练目标/推理公式/默认超参对齐官方（denoising + denoise 公式 + 128/128/0.2/0.2/swiglu）。

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
        hidden_dim: int = 128,           # 官方默认 128
        edge_dim: int = 128,             # 官方 pairwise_hidden_dim 128
        n_layers: int = 4,
        n_heads: int = 4,
        n_neighbors: int = 8,
        dropout: float = 0.2,            # 官方默认 0.2
        attn_dropout: float = 0.2,       # 官方默认 0.2
        activation: str = "swiglu",      # 官方默认 swiglu
        n_sample_steps: int = 5,         # 官方默认 5
        prior: str = "zinb",             # 官方默认 prior_sampler=zinb
        seed: int = 0,
    ):
        super().__init__()
        self.num_genes = num_genes
        self.feature_dim = feature_dim
        self.n_sample_steps = n_sample_steps
        self.prior = prior
        self.seed = seed

        config = STFlowConfig(
            n_genes=num_genes, feature_dim=feature_dim, hidden_dim=hidden_dim,
            pairwise_hidden_dim=edge_dim, n_layers=n_layers, n_heads=n_heads,
            dropout=dropout, attn_dropout=attn_dropout, n_neighbors=n_neighbors,
            activation=activation,
        )
        self.denoiser = Denoiser(config)

    # ---------- 先验采样 ----------
    def _sample_prior(self, shape: tuple, device: str) -> torch.Tensor:
        """按 self.prior 采样 ODE 起点（官方 interpolant 语义）。"""
        if self.prior == "zinb":
            return self._zinb_prior(shape, device)
        return torch.randn(shape, device=device)

    @staticmethod
    def _zinb_prior(shape: tuple, device: str) -> torch.Tensor:
        """官方 zinb 先验（train.py 固定参数）：ZINB(total_count=1, logits=0.1,
        zi_logits=0)，采样后 log1p（官方 interpolant.normalize=True 语义）。

        官方 scvi ZeroInflatedNegativeBinomial 参数：p=sigmoid(0.1)=0.525，
        零膨胀概率 sigmoid(0)=0.5。纯 torch 实现（无需 scvi）。
        """
        from torch.distributions import NegativeBinomial

        nb = NegativeBinomial(
            total_count=torch.ones(shape, device=device),
            probs=torch.full(shape, 0.525, device=device),
        ).sample()
        zi = (torch.rand(shape, device=device) < 0.5).float()
        return torch.log1p(nb * (1 - zi))

    # ---------- 训练侧（ROI 级） ----------
    def flow_loss(
        self,
        gene_expr: torch.Tensor,   # (1, N, G) 归一化表达
        img_features: torch.Tensor,  # (1, N, F) UNI2 条件
        coords: torch.Tensor,        # (1, N, 2)
    ) -> tuple[torch.Tensor, dict]:
        """官方去噪训练目标：模型预测**干净表达 z**，MSE(pred, z)。

        官方 train.py L85-92: `noisy_exp, t = corrupt_exp(gene_exp); pred, loss =
        model(noisy_exp, ..., labels=gene_exp)`；denoiser.py L93-96:
        `loss = MSE(prediction, labels)`。插值 z_t = (1-t)·ε + t·z，t~U(0,1)。
        """
        B, N, G = gene_expr.shape
        z = gene_expr
        eps = self._sample_prior(z.shape, z.device)
        t = torch.rand(B, device=gene_expr.device)          # 每个 ROI 一个 t
        z_t = (1 - t[:, None, None]) * eps + t[:, None, None] * z
        pred = self.denoiser.inference(z_t, img_features, coords, t, predict=True)
        loss = nn.functional.mse_loss(pred, z)
        return loss, {"flow_loss": loss}

    # ---------- 推理侧（ROI 级） ----------
    @torch.no_grad()
    def sample_roi(
        self,
        img_features: torch.Tensor,  # (1, N, F)
        coords: torch.Tensor,        # (1, N, 2)
        device: str = "cuda",
    ) -> torch.Tensor:
        """官方 test.py 推理语义：t 从 0.01 起，denoise 步进，**返回最后一步的 pred**。

        官方 test.py L62-79：ts=linspace(0.01, 1.0, n)；逐对 (t1,t2)：
            pred = model.inference(exp_t1, ..., t1, predict=True)
            最后一步 break，其余 `exp_t1 = denoise(pred, exp_t1, t1, d_t)`
                = exp_t1 + d_t·(pred - exp_t1)/(1 - t1)（interpolant.py L32-38）。
        sample = pred（模型在倒数第二个 t 处的干净表达预测）。
        """
        B, N, F = img_features.shape
        torch.manual_seed(self.seed)
        z = self._sample_prior((B, N, self.num_genes), device)

        ts = torch.linspace(0.01, 1.0, self.n_sample_steps, device=device)
        ts = ts[:, None].expand(self.n_sample_steps, B)      # (steps, B)
        pred = None
        for i, (t1, t2) in enumerate(zip(ts[:-1], ts[1:])):
            pred = self.denoiser.inference(z, img_features, coords, t1, predict=True)
            if i == self.n_sample_steps - 2:
                break                                          # 最后一步不 denoise
            dt = t2 - t1
            z = z + dt[:, None, None] * (pred - z) / (1.0 - t1[:, None, None])
        return pred
