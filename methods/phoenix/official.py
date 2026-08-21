"""Phoenix 官方权重支持：零样本 + 微调（纯 torch，无 apex/flash_attn）。

已验证（2026-08-22）：
- `flow_model.pth` 的 196 个 flow 键（x_projection/c_norm/c_projection/px/pc_embedding/
  t_embedding/layers/blocks/head）与 methods/phoenix/flow_transformer.py（flow_simple
  移植）的 state_dict **完全一致**（missing=0, extra=0），可 strict 加载。
- DINOv2 ViT-Giant（`vit_giant_patch14_reg4_dinov2`）从 `pytorch_model.bin` 加载。

目标空间：官方 stats_table 的 mean/std 量级小（median ~0.09/0.29）→ flow 在
**log1p 空间**训练。反归一化 `clip(x)*std+mean` 得 log1p，评估前需 `expm1` 转 raw counts。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .flow_transformer import FlowTransformerConfig, FlowTransformerModel

DINO_MODEL_NAME = "vit_giant_patch14_reg4_dinov2"
# 官方推理图像归一化（phoenix README H5PYDataset transform，非 ImageNet）
IMG_MEAN = torch.tensor([0.707223, 0.578729, 0.703617]).view(1, 3, 1, 1)
IMG_STD = torch.tensor([0.211883, 0.230117, 0.177517]).view(1, 3, 1, 1)


def build_flow(d_model: int = 512, n_layers: int = 8) -> FlowTransformerModel:
    """构建 flow transformer（flow_simple 移植，与 flow_model.pth 键匹配）。"""
    cfg = FlowTransformerConfig(
        d_genes=1, d_image=1536, d_model=d_model, d_cross=512,
        n_heads=8, n_layers=n_layers, qkv_bias=False, ffn_bias=False, ffn_mult=4,
        attn_drop=0.0, proj_drop=0.0, n_classes=0, cls_drop=0.1, checkpoint=False,
    )
    return FlowTransformerModel(cfg, vision_model=None)


def load_flow_weights(flow: nn.Module, path: str, device: str = "cpu") -> nn.Module:
    """从 flow_model.pth 加载 flow 权重（剔除 vision_model.* 前缀键，strict 加载）。"""
    sd = torch.load(path, map_location=device, weights_only=False)
    flow_sd = {k: v for k, v in sd.items() if not k.startswith("vision_model")}
    flow.load_state_dict(flow_sd, strict=True)
    return flow


def build_dino(flow_weights_path: str, device: str = "cpu") -> nn.Module:
    """构建 DINOv2 ViT-Giant（冻结），从 flow_model.pth 的 vision_model.* 键加载。

    注意：`pytorch_model.bin` 是 518 分辨率原版（pos_embed 1369），不能直接用于 224；
    flow_model.pth 里的 vision_model 是 **224 分辨率**（pos_embed 256，2026-08-22 核实），
    与官方训练一致，故从 flow_model.pth 提取。
    """
    import timm

    sd = torch.load(flow_weights_path, map_location="cpu", weights_only=False)
    dino_sd = {k[len("vision_model."):]: v for k, v in sd.items()
               if k.startswith("vision_model.")}
    m = timm.create_model(DINO_MODEL_NAME, pretrained=False, img_size=224, num_classes=0,
                          global_pool="token", init_values=1e-5, dynamic_img_size=False)
    missing, unexpected = m.load_state_dict(dino_sd, strict=False)
    if missing or unexpected:
        print(f"[Phoenix] DINOv2 加载: missing={len(missing)} unexpected={len(unexpected)}",
              flush=True)
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval().to(device)


class PhoenixOfficial(nn.Module):
    """官方 Phoenix：冻结 DINOv2 + flow（微调时 flow 可训练）。input_type='patch'。

    forward 输入 (B,3,224,224) **已用官方 tissue 归一化**的图像；内部先 DINOv2 提
    256 token 条件，再逐 t 采样（Euler ODE）输出 (B, n_genes) log1p 空间预测。
    """

    input_type = "patch"

    def __init__(self, num_genes: int, flow_weights: str,
                 dino_weights: str | None = None,
                 device: str = "cuda", n_sample_steps: int = 50, finetune: bool = False):
        super().__init__()
        self.num_genes = int(num_genes)
        self.n_sample_steps = n_sample_steps
        # DINOv2 从 flow_model.pth 的 vision_model.* 加载（224 分辨率，与训练一致）
        self.dino = build_dino(flow_weights, device)
        self.flow = build_flow()
        load_flow_weights(self.flow, flow_weights, device)
        self.flow.to(device)
        for p in self.flow.parameters():
            p.requires_grad_(finetune)
        self.flow.eval() if not finetune else self.flow.train()

    def _condition(self, x: torch.Tensor) -> torch.Tensor:
        """图像 (B,3,224,224) → DINOv2 patch token 条件 (B, T, 1536)。"""
        return self.dino.forward_features(x)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B,3,224,224) → (B, num_genes) log1p 空间预测（Euler 采样）。"""
        device = x.device
        c = self._condition(x)
        B = x.size(0)
        z = torch.randn(B, self.num_genes, 1, device=device)
        ts = torch.linspace(0.0, 1.0, self.n_sample_steps + 1, device=device)
        for i in range(self.n_sample_steps):
            t = ts[i].expand(B)
            v = self.flow(z, t, c)
            z = z + (ts[i + 1] - ts[i]) * v
        return z.squeeze(-1)

    def training_loss(self, gene_expr: torch.Tensor, x: torch.Tensor,
                      lambd: float | None = None) -> tuple[torch.Tensor, dict]:
        """流匹配损失（微调用）：目标为 log1p_zscore 空间表达。"""
        B, G = gene_expr.shape
        z = gene_expr.unsqueeze(-1)          # (B, G, 1)
        c = self._condition(x)               # (B, T, 1536)
        t = torch.rand(B, device=gene_expr.device)
        eps = torch.randn_like(z)
        z_t = (1 - t[:, None, None]) * eps + t[:, None, None] * z
        v_target = z - eps
        v_pred = self.flow(z_t, t, c)
        loss = nn.functional.mse_loss(v_pred, v_target)
        return loss, {"fm_loss": float(loss.item())}


class PhoenixFlowOnly(nn.Module):
    """flow 微调（DINOv2 token 已缓存，不经过 DINOv2）。input_type='feature'。

    输入 x: (B, 261, 1536) DINOv2 patch token（由 extract_phoenix_dino.py 预提取）。
    """

    input_type = "feature"
    feature_file = "X_phoenix_dino.npy"

    def __init__(self, num_genes: int, flow_weights: str,
                 device: str = "cuda", n_sample_steps: int = 50):
        super().__init__()
        self.num_genes = int(num_genes)
        self.n_sample_steps = n_sample_steps
        self.flow = build_flow()
        load_flow_weights(self.flow, flow_weights, device)
        self.flow.to(device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 261, 1536) → (B, num_genes) log1p_zscore 预测（Euler 采样）。"""
        B = x.size(0)
        c = x
        z = torch.randn(B, self.num_genes, 1, device=x.device)
        ts = torch.linspace(0.0, 1.0, self.n_sample_steps + 1, device=x.device)
        for i in range(self.n_sample_steps):
            t = ts[i].expand(B)
            v = self.flow(z, t, c)
            z = z + (ts[i + 1] - ts[i]) * v
        return z.squeeze(-1)

    def training_loss(self, gene_expr: torch.Tensor, x: torch.Tensor,
                      lambd: float | None = None) -> tuple[torch.Tensor, dict]:
        """流匹配损失（目标 log1p_zscore 表达，条件为缓存 DINOv2 token）。"""
        B, G = gene_expr.shape
        z = gene_expr.unsqueeze(-1)
        c = x
        t = torch.rand(B, device=gene_expr.device)
        eps = torch.randn_like(z)
        z_t = (1 - t[:, None, None]) * eps + t[:, None, None] * z
        v_pred = self.flow(z_t, t, c)
        loss = nn.functional.mse_loss(v_pred, z - eps)
        return loss, {"fm_loss": float(loss.item())}


def denorm_to_log1p(x: np.ndarray, stats: dict) -> np.ndarray:
    """flow 输出（normalized）→ log1p counts（官方语义 clip*std+mean）。"""
    mean, std = stats["mean"], stats["std"]
    return np.clip(x, 0, None) * std + mean


def denorm_to_raw(x: np.ndarray, stats: dict) -> np.ndarray:
    """flow 输出 → raw counts（log1p → expm1）。"""
    return np.expm1(np.clip(denorm_to_log1p(x, stats), -30, 30)).astype(np.float32)
