"""Local-Global Dual-scale Representation（UNI2+MLP 基线的改进）。

架构：concat[UNI2(op1 放缩中心), UNI2(op2 中心裁剪)] → 统一 MLPHead

- op1（Global branch）：以目标细胞为中心取不同尺寸图像块，resize 到 224×224，
  负责 tissue architecture、microenvironment。
- op2（Local branch）：在 op1 得到的 224×224 块上中心裁剪小区域
  （边长须为 14 的倍数以适配 UNI2），该区域 token 与整块 token 做 concat，
  负责 morphology、nucleus、cell neighborhood。

调参流程（每种泛化情形分别进行）：
    1. op1 sweep（只测 op1）→ 从 >224 边长逐步缩小，绘 PCC–c1 曲线，选 best c1
    2. op2 sweep（op1+op2，固定 best c1）→ 尝试不同 c2（14 的倍数），选最佳 PCC 对应 c2
    3. 最终训练：best c1、c2 组合，报告全部指标
"""
from __future__ import annotations


class LocalGlobalModule:
    """Local-Global 双尺度特征：concat[UNI2(op1 全局), UNI2(op2 局部)] → MLPHead。

    参数：
        num_genes: 预测的公共基因数
        c1: op1 取块边长（放缩中心，≥224，由 op1 sweep 决定）
        c2: op2 中心裁剪边长（须为 14 的倍数，由 op2 sweep 决定）
    """

    def __init__(self, num_genes: int, c1: int = 336, c2: int = 224):
        self.num_genes = num_genes
        self.c1 = c1
        self.c2 = c2
        raise NotImplementedError("待实现：op1 放缩中心 + op2 中心裁剪 + token concat + 统一 MLPHead")
