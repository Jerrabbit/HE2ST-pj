"""UNI2+MLP 基线模型。

架构：UNI2(224×224 图像块) → 特征 → 统一 MLPHead → 基因表达预测
预期结论：仅利用 Foundation Model 特征 + 简单 MLP，即可在多个 Benchmark 中
稳定超过 SpatialEx、GHIST 等复杂空间模型——性能提升来自信息表示而非复杂结构。
"""
from __future__ import annotations


class UNI2MLP:
    """UNI2 + 统一 MLP 头基线。

    参数：
        num_genes: 预测的公共基因数
        mlp_hidden_dims: MLP 隐藏层维度（须与 common/models/mlp_head.py 统一）
    """

    def __init__(self, num_genes: int, mlp_hidden_dims: list[int] | None = None):
        self.num_genes = num_genes
        self.mlp_hidden_dims = mlp_hidden_dims
        raise NotImplementedError("待实现：UNI2 特征提取 + 统一 MLPHead")
