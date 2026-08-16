"""ST-Net：DenseNet121 + 线性输出层，按官方实现复刻。

官方代码：D:\\hest_data\\codes\\ST-Net（stnet/utils/nn.py、cmd/run_spatial.py）
架构：
    torchvision DenseNet121（ImageNet 预训练，全卷积直到全局平均池）
        → 替换 classifier：Linear(1024 → G)，权重置零，**bias 初始化为
          训练集平均表达**（官方 run_spatial.py:223-235，任务 gene 时
          last.bias.data = mean_expression）
    损失：MSE（官方 sum((pred-gene)^2)/outputs），目标为 log1p 归一化表达。
    训练：Adam/SGD 微调（官方随机初始化分类头权重，bias 用均值）。

适配本仓库：
    - input_type='patch'，用 HESTDataset；DenseNet 全卷积到全局池，
      任意输入尺寸均可（不强制 resize）。
    - 表达目标 gene_norm='log1p_norm_total'（接近官方的 normalize_total+log1p）。
    - bias=平均表达：train_function 里先从训练集计算均值再初始化。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchvision

__all__ = ["STNet"]


class STNet(nn.Module):
    """DenseNet121 → Linear(G)。bias 初始化为平均表达（官方 ST-Net 关键设置）。"""

    input_type = "patch"

    def __init__(
        self,
        num_genes: int,
        pretrained: bool = True,
        bias_init: torch.Tensor | None = None,
        finetune: bool = True,
    ):
        super().__init__()
        self.num_genes = num_genes
        self.finetune = finetune
        if pretrained:
            weights = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1
        else:
            weights = None
        backbone = torchvision.models.densenet121(weights=weights)
        in_features = backbone.classifier.in_features  # 1024
        backbone.classifier = nn.Linear(in_features, num_genes)
        if not finetune:
            # 冻结 DenseNet 特征提取器（只训 Linear 头），与冻结特征的 UNI2+MLP 公平对比
            for p in backbone.features.parameters():
                p.requires_grad = False
        with torch.no_grad():
            backbone.classifier.weight.zero_()
            if bias_init is not None:
                backbone.classifier.bias.copy_(bias_init)
        self.backbone = backbone

    @torch.no_grad()
    def set_bias_init(self, mean_expr: torch.Tensor) -> None:
        """官方初始化：权重置零，bias=平均表达（在训练集上计算）。"""
        w = self.backbone.classifier.weight
        w.zero_()
        self.backbone.classifier.bias.copy_(mean_expr.to(w.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """patch (B,3,H,W) → (B,G) 表达预测（log1p 归一化空间）。"""
        return self.backbone(x)
