"""Local-Global Dual-scale Representation（UNI2+MLP 基线的改进）。

架构：`concat[UNI2(op1 放缩中心) CLS, UNI2 中心 patch token 复用(op2)] → 统一 MLPHead`

- **op1（Global branch）**：以目标细胞为中心取 l1×l1 图像块，resize 到 224×224，
  用 UNI2 提 [CLS] 特征，负责 tissue architecture、microenvironment。
- **op2（Local branch）**：**复用 op1 同一次 forward 的 patch token 网格**——
  UNI2 patch14 无重叠，224×224 一次 forward 出 16×16=256 个 patch token；
  中心裁剪 l2×l2（l2=14k）对应正是该网格中心 k×k 子块，取子块 token 的
  mean-pool 作为 Local 特征，负责 morphology、nucleus、cell neighborhood。

**单次 forward 约束**：Local 分支**不再二次 forward**（不做"裁剪→resize→再提"），
而是从 Global 的 token 序列直接切出中心子块——op2 sweep 的全部 l2 值（56..112）
都免费复用同一份 token，成本为零。

实现方式：UNI2 为冻结编码器（预提取特征文件），模型只做 `concat → MLPHead`。
特征由 `scripts/extract_local_global.py` 预提取（`--stage local` 单次产出全部 l2 文件）：
    - Global：`X_uni2_g{l1}.npy`（l1×l1 块 → resize 224 → UNI2 [CLS]，1536 维/细胞）
    - Local：`X_uni2_l{l2}.npy`（同一 forward 中心 k×k patch token mean-pool，1536 维/细胞）
模型输入 = concat[g, l]（3072 维）或仅其一（消融，1536 维）。

调参流程（每种泛化情形分别进行）：
    1. **op1 sweep**（只测 op1 / Global）→ 从 >224 边长逐步缩小（小步长，多取值），
       30 epoch 取 best val_PCC，绘 PCC–l1 曲线，选 best l1。
    2. **op2 sweep**（op1+op2，固定 best l1）→ 尝试 l2 = 4..8 × 14，选最佳 PCC 对应 l2。
    3. **最终训练**：best l1 + best l2，50 epoch 报告全部指标。
    4. **消融**：只用 Local、只用 Global、完整 Local+Global。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from common.models.mlp_head import MLPHead

FEATURE_DIM = 1536  # UNI2 [CLS] token 维度


class LocalGlobalMLP(nn.Module):
    """Local-Global 双尺度特征 + 统一 MLP 头（可训练）。

    特征文件由 `feature_files` 指定（预提取，冻结 UNI2）：
        - Local+Global（完整）：[X_uni2_g{l1}.npy, X_uni2_l{l2}.npy]，输入 3072 维
        - Global-only（消融）：[X_uni2_g{l1}.npy]，输入 1536 维
        - Local-only（消融）：[X_uni2_l{l2}.npy]，输入 1536 维

    参数：
        num_genes: 预测的公共基因数
        feature_files: 特征文件列表（相对 data_dir 或绝对路径）
        in_dim: 输入特征维数（1536×文件数）
        mlp_hidden_dims / dropout: 统一 MLPHead 参数（与其它方法一致）
        l1 / l2: 记录调参用的块/裁剪边长（仅作配置记录）
    """

    input_type = "feature"
    feature_files = ["X_uni2_g512.npy", "X_uni2_l56.npy"]

    def __init__(
        self,
        num_genes: int,
        feature_files: list[str] | None = None,
        in_dim: int = FEATURE_DIM * 2,
        mlp_hidden_dims: tuple[int, int] = (512, 256),
        dropout: float = 0.1,
        l1: int = 512,
        l2: int = 56,
        norm_concat: bool = False,
    ):
        super().__init__()
        self.num_genes = int(num_genes)
        self.l1 = int(l1)
        self.l2 = int(l2)
        if feature_files:
            self.feature_files = list(feature_files)
        self.head = MLPHead(in_dim, list(mlp_hidden_dims), self.num_genes, dropout)
        # 对比变体：concat 后先 LayerNorm 再进 MLP（默认无，验证是否有增益）
        self.norm_concat = norm_concat
        self.norm = nn.LayerNorm(in_dim) if norm_concat else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """concat[Global, Local] 特征 (B, 1536*n_files) → (B, num_genes) 归一化表达预测。"""
        return self.head(self.norm(x))
