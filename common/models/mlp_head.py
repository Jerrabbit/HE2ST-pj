"""统一 MLP 预测头，所有需要接 MLP 的方法复用（课题要求 6）。

课题要求 6：各方法若需接 MLP，层数、架构必须相同。本模块固化为唯一实现。
架构（与远程现有 baseline/uni_mlp.py 一致，已在初步实验验证）：
    Linear(in→512) → BatchNorm1d → LeakyReLU(0.1) → Dropout(0.1)
    → Linear(512→256) → BatchNorm1d → LeakyReLU(0.1) → Dropout(0.1)
    → Linear(256→out)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MLPHead(nn.Module):
    """共享 MLP 预测头。

    参数：
        input_dim: 输入特征维度（UNI2 为 1536）
        hidden_dims: 隐藏层维度（默认 [512, 256]）
        output_dim: 预测的公共基因数
        dropout: dropout 率（默认 0.1）
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int] = (512, 256),
        output_dim: int = 313,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = list(hidden_dims)
        self.output_dim = output_dim
        self.dropout = dropout

        layers = []
        prev = input_dim
        for h in self.hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 (B, input_dim) → 输出 (B, output_dim) 基因表达预测。"""
        return self.mlp(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, device: str | None = None) -> torch.Tensor:
        """推理模式预测（无梯度）。"""
        if device is not None:
            x = x.to(device)
        self.eval()
        return self.forward(x)


class RefMLPHead(nn.Module):
    """参考架构 MLP 头（methods/uni2_mlp/MLP架构参考.txt，已验证更好性能）。

    架构（参考文件的 active 版本）：
        LayerNorm(input_dim) → Linear(input→512) → GELU → Dropout
        → Linear(512→output_dim) → **Softplus**（输出恒正 → 需正数目标空间，如 log1p）

    也可用参考文件注释版（无 LayerNorm/Softplus 的 2 层 MLP），见 `use_softplus=False`。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        output_dim: int = 313,
        dropout: float = 0.1,
        use_softplus: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        if use_softplus:
            self.head.append(nn.Softplus())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 (B, input_dim) → 输出 (B, output_dim) 基因表达预测（恒正若 Softplus）。"""
        return self.head(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor, device: str | None = None) -> torch.Tensor:
        if device is not None:
            x = x.to(device)
        self.eval()
        return self.forward(x)
