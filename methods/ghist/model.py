"""GHIST：UNet 分割 + 细胞类型 + 多组件表达预测（官方 Framework 移植 + benchmark 包装）。

官方：Outeiral et al. 2024（Cell），D:\\hest_data\\codes\\GHIST。
架构（严格照官方 framework.py 的 Framework，零改动）：
    整片 H&E + 核分割 mask
      → UNet Backbone（分割 + 中间特征 hd1/h1）
      → 逐核 mask 内特征聚合 → cell 特征向量（768 维）
      → Embed → cell embedding
      → avgexp 模式：MLPSoftmax 加权 n_ref 参考细胞表达 → 表达
      → 邻域模式：estimate_comp（组成）+ CrossAttention 细化
      → 细胞型：mlp_hist（H&E） + mlp_genes（表达）
      → 免疫 / 浸润变体（独立分支）

本仓库适配（数据/接口层，架构零改动）：
    - 整片图方法（非 per-cell patch），无逐 batch forward(x)；
      数据管线读 data_dir 的 ghist_data 格式（he_image.tif + he_image_nuclei_seg.tif
      + cell_gene_matrix），train_function / evaluate_slide 见 __init__.py。
    - 训练损失照官方：分割 CE + 细胞型 CE + 表达 MSE + 免疫/浸润 MSE + 组成 KLDiv。
    - 评估走统一指标（PCC/SPCC/Top-k/AUROC），归一化空间与 harness 一致。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .framework import Framework  # noqa: 官方 Framework


class GHISTModel(nn.Module):
    """GHIST Framework 的 benchmark 包装。

    整片图方法：input_type='patch'（但输入是整片 H&E + mask，走 train_function /
    evaluate_slide，不提供逐 batch forward(x)）。
    """

    input_type = "patch"

    def __init__(
        self,
        num_genes: int,
        n_classes: int = 9,
        emb_dim: int = 256,
        device: str = "cuda",
        n_ref: int = 512,
        use_avgexp: bool = True,
        use_celltype: bool = True,
        use_neighb: bool = True,
        in_channels: int = 3,
    ):
        super().__init__()
        self.num_genes = num_genes
        self.n_classes = n_classes
        self.core = Framework(
            n_classes=n_classes, n_genes=num_genes, emb_dim=emb_dim,
            device=device, n_ref=n_ref,
            use_avgexp=use_avgexp, use_celltype=use_celltype,
            use_neighb=use_neighb, in_channels=in_channels,
        )

    def forward(self, x):
        raise NotImplementedError(
            "GHIST 为整片图方法（UNet 分割 + 逐核聚合，需要整片 H&E + 核 mask），"
            "无逐 batch forward(x)。请使用 methods.ghist.train_function / evaluate_slide。"
        )
