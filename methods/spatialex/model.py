"""SpatialEx：基于超图卷积（HGNN）的 HE→ST 表达预测，按官方实现复刻。

官方代码：D:\\hest_data\\codes\\SpatialEx\\SpatialEx\\model.py（Predictor_spot、HGNN）
架构（Predictor_spot，逐层照抄官方，无架构改动）：
    mlp:   Linear(in_dim → hidden_dim) → LeakyReLU(0.1) → BatchNorm1d(hidden_dim)
    mod:   HGNN(in_dim=hidden_dim, num_hidden=hidden_dim, out_dim=hidden_dim,
                num_layers=2, dropout=0, activation='prelu')
           HGNN.forward(X, H)：
               X = torch.sparse.mm(H, W1(dropout(X))) → prelu
               X = torch.sparse.mm(H, W2(dropout(X)))        （num_layers=2）
    linear: Linear(hidden_dim, out_dim)
    forward(graph, he_rep, x, agg_mtx=None, selection=None)：
        he_rep = mlp(he_rep)；enc = mod(he_rep, graph)
        x_prime = leaky_relu(linear(leaky_relu(enc)))
        返回 (loss, x_prime, enc)

本仓库适配（仅设备与损失目标，架构不变）：
    - 设备：去除官方 DGI_SAGE/HyperSAGE 中硬编码的 cuda=True；Predictor_spot 路径
      只用 torch.sparse.mm 的 HGNN，图/特征统一由调用方放到指定 device。
    - 损失目标：默认 cell 级 MSE（在统一归一化空间，与其他方法公平可比）；
      use_spot_agg=True 时复刻官方 Generate_pseudo_spot 的 visium 伪 spot 聚合损失
      （见 methods/spatialex/__init__.py 的 build_pseudo_spot_agg）。
    - 整片图方法：cell 的预测依赖邻居超图卷积，SpatialExModel.forward 不支持逐 cell
      批处理（BatchNorm1d + 超图卷积需要整片/ROI 上下文），推理统一走
      evaluate_slide / predict_roi。因此 forward 直接抛 NotImplementedError，
      防止误用逐 cell 路径。

Xenium 规模适配：全切片图 O(N^2) 不可行，按物理坐标切成重叠 ROI（约 1024–2048 细胞），
每个 ROI 内从全切片超图 H 子选择（H[roi][:, roi]，与官方 Xenium_HBRC_overlap 一致）
再 hpnn 归一化，作为训练/推理的基本单元。细节见 __init__.py。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["HGNN", "Predictor_spot", "SpatialExModel"]


def create_activation(name):
    """官方 utils.py create_activation（原样复刻）。"""
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "elu":
        return nn.ELU()
    elif name == "leaky_relu":
        return nn.LeakyReLU()
    elif name == "prelu":
        return nn.PReLU()
    else:
        return None


class HGNN(nn.Module):
    """超图卷积层（官方 model.py HGNN，原样复刻）。"""

    def __init__(self, in_dim, num_hidden, out_dim, num_layers, dropout, activation):
        super(HGNN, self).__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.activation = create_activation(activation)
        self.mlp = nn.ModuleList()
        self.dropout = dropout

        if num_layers == 1:
            self.W1 = nn.Linear(in_dim, out_dim)
        elif num_layers == 2:
            self.W1 = nn.Linear(in_dim, num_hidden)
            self.W2 = nn.Linear(num_hidden, out_dim)
        elif self.num_layers > 2:
            for i in range(self.num_layers - 2):
                self.mlp.append(nn.Linear(num_hidden, num_hidden))

        self.dropout = nn.Dropout(dropout)

    def forward(self, X, H):
        if self.num_layers == 1:
            X = torch.sparse.mm(H, self.W1(self.dropout(X)))
            X = self.activation(X)
        elif self.num_layers == 2:
            X = torch.sparse.mm(H, self.W1(self.dropout(X)))
            X = self.activation(X)
            X = torch.sparse.mm(H, self.W2(self.dropout(X)))
        else:
            X = torch.sparse.mm(H, self.W1(self.dropout(X)))
            X = self.activation(X)
            for i in range(self.num_layers - 2):
                X = torch.sparse.mm(H, self.mlp[i](self.dropout(X)))
                X = self.activation(X)
            X = torch.sparse.mm(H, self.W2(self.dropout(X)))

        return X


class Predictor_spot(nn.Module):
    """SpatialEx 基线预测器（官方 model.py Predictor_spot，原样复刻）。

    agg=True 时损失作用于 visium 伪 spot 聚合表达（官方 Xenium 流程）；
    agg=False 时损失为 cell 级 MSE（本 benchmark 默认）。
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float = 0.1,
        loss_fn="mse",
        activation="prelu",
        agg=True,
    ):
        super(Predictor_spot, self).__init__()

        dropout = 0  # 官方硬编码：HGNN 内部 dropout=0
        self.agg = agg
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(hidden_dim),
        )

        self.mod = HGNN(
            in_dim=hidden_dim,
            num_hidden=hidden_dim,
            out_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
        )

        self.linear = nn.Linear(hidden_dim, out_dim)

        if loss_fn == "mse":
            self.criterion = nn.MSELoss()
        else:
            print("not implement")

    def forward(self, graph, he_rep, x, agg_mtx=None, selection=None):
        he_rep = self.mlp(he_rep)
        enc = self.mod(he_rep, graph)
        x_prime = F.leaky_relu(self.linear(F.leaky_relu(enc)))
        if self.agg:
            loss = self.criterion(torch.sparse.mm(agg_mtx, x_prime[selection]), x)
        else:
            loss = self.criterion(x_prime, x)
        return loss, x_prime, enc

    def predict(self, graph, he_rep):
        he_rep = self.mlp(he_rep)
        enc = self.mod(he_rep, graph)
        x_prime = F.leaky_relu(self.linear(F.leaky_relu(enc)))
        return x_prime


class SpatialExModel(nn.Module):
    """SpatialEx 包装类：整片超图方法，训练/推理统一按 ROI 走。

    参数：
        num_genes: 基因数（输出维度）
        in_dim: H&E 特征维度（本仓库用 UNI2 特征，1536）
        hidden_dim: 隐层宽度（官方 512）
        num_layers: HGNN 层数（官方 2）
        dropout: 传给 Predictor_spot（官方内部强制置 0，此处仅为接口一致性）
        use_spot_agg: 是否用官方 visium 伪 spot 聚合损失（默认 False = cell 级 MSE）
    """

    input_type = "feature"
    feature_file = "X_uni2.npy"

    def __init__(
        self,
        num_genes: int,
        in_dim: int = 1536,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_spot_agg: bool = False,
    ):
        super().__init__()
        self.num_genes = num_genes
        self.use_spot_agg = use_spot_agg
        self.predictor = Predictor_spot(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=num_genes,
            num_layers=num_layers,
            dropout=dropout,
            loss_fn="mse",
            activation="prelu",
            agg=use_spot_agg,
        )

    def forward(self, x):
        """SpatialEx 是整片超图方法，不支持逐 cell 前向。

        cell 预测依赖邻居（超图卷积），必须按整片/ROI 的图来推理。请使用
        methods.spatialex.evaluate_slide 或 predict_roi(graph, he_rep)。
        """
        raise NotImplementedError(
            "SpatialEx 是整片超图方法：请使用 evaluate_slide / predict_roi(graph, he_rep)，"
            "不要对单个 cell/批做前向。"
        )

    def predict_roi(self, graph, he_rep):
        """单个 ROI 内全部细胞的表达预测（mlp→HGNN→linear，无损失）。

        graph: (n_roi, n_roi) 稀疏超图传播矩阵（已 hpnn 归一化）
        he_rep: (n_roi, in_dim) H&E 特征
        返回 (n_roi, num_genes) 归一化空间表达预测。
        """
        return self.predictor.predict(graph, he_rep)
