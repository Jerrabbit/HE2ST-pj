"""Phoenix：(Latent) Flow Matching 生成模型（官方架构移植）。

官方：Phoenix（Tran/Gindra et al. 2026，Peng Lab，D:\\hest_data\\codes\\Phoenix）。
本文件按官方 phoenix/models/flow_simple.py 的 FlowTransformerModel 移植：

    基因表达按**每基因 1 维 token**（d_genes=1，x 形状 (B, n_genes, 1)）
      → SwiGLU 投影到 d_model + 位置嵌入
      → 图像特征（UNI2 1536 维）经 RMSNorm + SwiGLU 投影到 d_cross + 位置嵌入
        （官方用 DINOv2 ViT-Giant 提取，本仓库用预提取 UNI2 特征等价替代，
        两者输出维度都是 1536）
      → 2 个 ClassicalTransformerBlock 处理图像条件
      → n_layers 个 FlowTransformerBlock：
          zattn（基因 token 自注意力，adaLN 时间调节）
        + xattn（交叉注意力到图像特征）
        + SwiGLU FFN（adaLN）
      → FlowTransformerHead → 输出流速度 v（与 x 同形状）

训练：线性插值流匹配 z_t = (1-t)·ε + t·z（z=表达 token，ε=噪声），v = z - ε，
     MSE(v_pred, v_target)。推理：Euler ODE 从噪声解到 t=1。

条件 UNI2 特征 (B, 1536) 作为 (B, 1, 1536) 单 token 输入（per-cell 条件）。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .flow_transformer import FlowTransformerConfig, FlowTransformerModel


class Phoenix(nn.Module):
    """Phoenix：基因 token 流匹配生成模型。input_type='feature'。"""

    input_type = "feature"

    def __init__(
        self,
        num_genes: int,
        feature_dim: int = 1536,
        d_model: int = 512,
        d_cross: int = 512,
        n_heads: int = 8,
        n_layers: int = 8,
        d_genes: int = 1,
        n_sample_steps: int = 20,
        seed: int = 0,
    ):
        super().__init__()
        self.num_genes = num_genes
        self.feature_dim = feature_dim
        self.n_sample_steps = n_sample_steps
        self.seed = seed

        cfg = FlowTransformerConfig(
            d_genes=d_genes, d_image=feature_dim,
            d_model=d_model, d_cross=d_cross,
            n_heads=n_heads, n_layers=n_layers,
            qkv_bias=False, ffn_bias=False, ffn_mult=4,
            attn_drop=0.0, proj_drop=0.0,
            n_classes=0, cls_drop=0.1,
            checkpoint=False,
        )
        self.flow = FlowTransformerModel(cfg, vision_model=None)

    # ---------- 训练侧 ----------
    def _flow_target(self, z: torch.Tensor, t: torch.Tensor, eps: torch.Tensor):
        z_t = (1 - t[:, None, None]) * eps + t[:, None, None] * z
        v = z - eps
        return z_t, v

    def training_loss(
        self,
        gene_expr: torch.Tensor,   # (B, G) 归一化表达
        condition: torch.Tensor,   # (B, F) UNI2 条件
        lambd: float | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """流匹配损失：MSE(去噪器预测速度, 真实速度)。"""
        B, G = gene_expr.shape
        z = gene_expr.unsqueeze(-1)            # (B, G, 1) 基因 token
        c = condition.unsqueeze(1)             # (B, 1, F) 条件 token

        t = torch.rand(B, device=gene_expr.device)
        eps = torch.randn_like(z)
        z_t, v_target = self._flow_target(z, t, eps)
        v_pred = self.flow(z_t, t, c)          # (B, G, 1)

        fm_loss = nn.functional.mse_loss(v_pred, v_target)
        return fm_loss, {"fm_loss": fm_loss}

    # ---------- 推理侧 ----------
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """条件特征 (B, D) → 采样生成表达 (B, G)。Euler 步 ODE，固定 seed 可复现。"""
        device = x.device
        B = x.size(0)
        G = self.num_genes
        c = x.unsqueeze(1)                      # (B, 1, F)

        gen = torch.Generator(device=device).manual_seed(self.seed)
        z = torch.randn(B, G, 1, device=device, generator=gen)

        ts = torch.linspace(0.0, 1.0, self.n_sample_steps + 1, device=device)
        for i in range(self.n_sample_steps):
            t = ts[i].expand(B)
            v = self.flow(z, t, c)
            z = z + (ts[i + 1] - ts[i]) * v
        return z.squeeze(-1)                    # (B, G)
