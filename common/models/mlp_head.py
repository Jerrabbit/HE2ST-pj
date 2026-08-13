"""统一 MLP 预测头，所有需要接 MLP 的方法复用（课题要求 6）。

课题要求 6：各方法若需接 MLP，层数、架构必须相同；
UNI2+MLP 基线需认真设计 MLP（层数、激活函数等），
确认后的架构在本模块固化为唯一实现，供所有方法共用。
"""
from __future__ import annotations


class MLPHead:
    """共享 MLP 预测头。

    设计要点（需在初步实验确定后固化，勿在各方法中自行改架构）：
        - 层数、隐藏层维度、激活函数、dropout 对所有方法一致
        - 输出维度 = 预测的公共基因数
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.1,
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        raise NotImplementedError("待实现：统一架构的 MLP（所有方法共用）")
