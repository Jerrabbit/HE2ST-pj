"""Hist2ST：Transformer + 图神经网络(GNN) 混合架构，从 H&E 组织学图像预测空间基因表达。

官方代码：D:\\hest_data\\codes\\Hist2ST\\（HIST2ST.py / transformer.py / gcn.py / NB_module.py）
本文件为官方架构的忠实复刻（纯 nn.Module，去掉 pytorch_lightning）。

架构（严格按官方，无任何改动）：
    patch_embedding = Conv2d(3, 32, patch_size=7, stride=7)   # (N,3,112,112) → (N,32,16,16)
    x_embed / y_embed = Embedding(n_pos=64, dim=1024)          # 细胞坐标位置嵌入
    ViT(mixer_transformer)：
        layer1 = convmixer_block × 2（深度可分离卷积 + 逐点卷积）
        layer2 = attn_block × 8（PreNorm + 自注意力 + FFN）
        layer3 = gs_block × 4（图卷积聚合，gcn=True, policy='mean'）
        jknet  = LSTM 跳跃连接（JKN）聚合 layer3 各层输出后取均值
    gene_head = LayerNorm(1024) → Linear(1024, n_genes)
    前向：patch → patch_embedding → convmixer → 展平 → (+位置嵌入) → 自注意力 → GNN → 基因头
    训练目标：MSE（归一化空间）；可选 ZINB/NB（NB_module，默认关）

适配本仓库（仅数据/接口层，无架构改动）：
    - 去掉 pytorch_lightning：Hist2ST 为纯 nn.Module，训练循环见 methods/hist2st/__init__.py
      train_function（等价于官方 training_step：MSE + 可选 ZINB）。
    - calcADJ → 纯 kNN（k=4，cKDTree，见 __init__.py _build_adj），prune='NA' 语义：
      官方 prune='Grid' 的距离阈值 ≤2.0 为 Visium 点阵间距设计，不适用于 Xenium µm 坐标，
      因此采用纯 kNN（A[i][最近 k 个 j]=1，自环排除）作为正确适配。
    - 位置嵌入：ROI 局部坐标归一化到 [0, n_pos) 取整（官方数据本就在该范围，见 _normalize_centers）。
    - patch 输入为 [0,1] float、不做 ImageNet 归一化（官方加载原始像素 im 后仅 /255 处理，
      convmixer 直接从原始值学习；见 __init__.py _load_slide）。
    - ZINB / bake 默认关（benchmark 统一在归一化空间评估，语义干净）。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision import transforms as tv_transforms

__all__ = [
    "SelectItem",
    "PreNorm",
    "FeedForward",
    "Attention",
    "attn_block",
    "convmixer_block",
    "mixer_transformer",
    "ViT",
    "gs_block",
    "MeanAct",
    "DispAct",
    "NB_loss",
    "ZINB_loss",
    "Hist2ST",
    "Hist2STModel",
]


# --------------------------------------------------------------------------- #
# transformer.py（官方原样，仅把 einops.rearrange 换成等价 reshape/permute）
# --------------------------------------------------------------------------- #
class SelectItem(nn.Module):
    """取容器输出第 item_index 项（官方用于 LSTM → 取 output 序列）。"""

    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """多头自注意力（官方 transformer.py Attention，等价实现无 einops）。"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # einops 'b n (h d) -> b h n d'
        q, k, v = (t.view(b, n, h, -1).permute(0, 2, 1, 3) for t in qkv)
        dots = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = self.attend(dots)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        # einops 'b h n d -> b n (h d)'
        out = out.permute(0, 2, 1, 3).reshape(b, n, -1)
        return self.to_out(out)


class attn_block(nn.Module):
    """自注意力块：PreNorm(Attn) + 残差 + PreNorm(FFN) + 残差。"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
        self.ff = PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))

    def forward(self, x):
        x = self.attn(x) + x
        x = self.ff(x) + x
        return x


class convmixer_block(nn.Module):
    """ConvMixer 块：深度可分离卷积（残差）+ 逐点卷积（官方 convmixer_block）。"""

    def __init__(self, dim, kernel_size):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x):
        x = self.dw(x) + x
        x = self.pw(x)
        return x


class mixer_transformer(nn.Module):
    """ConvMixer → 自注意力 → 图卷积（GNN）+ JKN(LSTM) 聚合。"""

    def __init__(
        self,
        channel=32,
        kernel_size=5,
        dim=1024,
        depth1=2,
        depth2=8,
        depth3=4,
        heads=8,
        dim_head=64,
        mlp_dim=1024,
        dropout=0.0,
        policy="mean",
        gcn=True,
    ):
        super().__init__()
        self.layer1 = nn.Sequential(
            *[convmixer_block(channel, kernel_size) for _ in range(depth1)],
        )
        self.layer2 = nn.Sequential(
            *[attn_block(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth2)]
        )
        self.layer3 = nn.ModuleList([gs_block(dim, dim, policy, gcn) for _ in range(depth3)])
        self.jknet = nn.Sequential(
            nn.LSTM(dim, dim, 2),
            SelectItem(0),
        )
        self.down = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, 1),
            nn.Flatten(),
        )

    def forward(self, x, ct, adj):
        x = self.down(self.layer1(x))
        g = x.unsqueeze(0)
        g = self.layer2(g + ct).squeeze(0)
        jk = []
        for layer in self.layer3:
            g = layer(g, adj)
            jk.append(g.unsqueeze(0))
        g = torch.cat(jk, 0)
        g = self.jknet(g).mean(0)
        return g


class ViT(nn.Module):
    """Dropout 包一层 mixer_transformer（官方 ViT）。"""

    def __init__(
        self,
        channel=32,
        kernel_size=5,
        dim=1024,
        depth1=2,
        depth2=8,
        depth3=4,
        heads=8,
        mlp_dim=1024,
        dim_head=64,
        dropout=0.0,
        policy="mean",
        gcn=True,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.transformer = mixer_transformer(
            channel, kernel_size, dim,
            depth1, depth2, depth3,
            heads, dim_head, mlp_dim, dropout,
            policy, gcn,
        )

    def forward(self, x, ct, adj):
        x = self.dropout(x)
        x = self.transformer(x, ct, adj)
        return x


# --------------------------------------------------------------------------- #
# gcn.py（官方 gs_block 原样）
# --------------------------------------------------------------------------- #
class gs_block(nn.Module):
    """图卷积块：邻接聚合 → 可选拼接自身 → Linear+ReLU → L2 归一化。"""

    def __init__(self, feature_dim, embed_dim, policy="mean", gcn=False, num_sample=10):
        super().__init__()
        self.gcn = gcn
        self.policy = policy
        self.embed_dim = embed_dim
        self.feat_dim = feature_dim
        self.num_sample = num_sample
        self.weight = nn.Parameter(
            torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim)
        )
        init.xavier_uniform_(self.weight)

    def forward(self, x, Adj):
        neigh_feats = self.aggregate(x, Adj)
        if not self.gcn:
            combined = torch.cat([x, neigh_feats], dim=1)
        else:
            combined = neigh_feats
        combined = F.relu(self.weight.mm(combined.T)).T
        combined = F.normalize(combined, 2, 1)
        return combined

    def aggregate(self, x, Adj):
        adj = Adj
        if not self.gcn:
            n = len(adj)
            adj = adj - torch.eye(n, device=adj.device)
        if self.policy == "mean":
            num_neigh = adj.sum(1, keepdim=True)
            mask = adj.div(num_neigh)
            to_feats = mask.mm(x)
        elif self.policy == "max":
            indexs = [i.nonzero() for i in adj == 1]
            to_feats = []
            for feat in [x[i.squeeze()] for i in indexs]:
                if len(feat.size()) == 1:
                    to_feats.append(feat.view(1, -1))
                else:
                    to_feats.append(torch.max(feat, 0)[0].view(1, -1))
            to_feats = torch.cat(to_feats, 0)
        return to_feats


# --------------------------------------------------------------------------- #
# NB_module.py（官方原样）
# --------------------------------------------------------------------------- #
class MeanAct(nn.Module):
    """ZINB mean 激活：clamp(exp(x), 1e-5, 1e6)。"""

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)


class DispAct(nn.Module):
    """ZINB dispersion 激活：clamp(softplus(x), 1e-4, 1e4)。"""

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)


def NB_loss(x, h_r, h_p):
    ll = torch.lgamma(torch.exp(h_r) + x) - torch.lgamma(torch.exp(h_r))
    ll += h_p * x - torch.log(torch.exp(h_p) + 1) * (x + torch.exp(h_r))
    loss = -torch.mean(torch.sum(ll, axis=-1))
    return loss


def ZINB_loss(x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
    eps = 1e-10
    if isinstance(scale_factor, float):
        scale_factor = np.full((len(mean),), scale_factor)
    scale_factor = scale_factor[:, None]
    mean = mean * scale_factor

    t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
    t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
    nb_final = t1 + t2

    nb_case = nb_final - torch.log(1.0 - pi + eps)
    zero_nb = torch.pow(disp / (disp + mean + eps), disp)
    zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
    result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

    if ridge_lambda > 0:
        ridge = ridge_lambda * torch.square(pi)
        result += ridge
    result = torch.mean(result)
    return result


# --------------------------------------------------------------------------- #
# HIST2ST.py（官方 Hist2ST，去掉 pytorch_lightning，纯 nn.Module）
# --------------------------------------------------------------------------- #
class Hist2ST(nn.Module):
    """官方 Hist2ST 核心模型（纯 nn.Module 复刻）。

    前向：forward(patches, centers, adj) → (pred, extra, h)
        patches: (1, N, 3, fig_size, fig_size) [0,1] float
        centers: (1, N, 2) int64，取值 [0, n_pos)
        adj:     (N, N) float 稠密邻接（二进制，含/不含自环按 gcn 处理）
        pred:    (N, n_genes) 归一化空间表达预测（官方将 patches reshape 为 B*N 后
                 不恢复 batch 维，故为 (N,G) 而非 (1,N,G)）
        extra:   None（zinb=0）或 (mean,disp,pi)（zinb>0 且 nb=False）或 (r,p)（nb=True）
        h:       (N, dim) 基因头输入特征
    """

    def __init__(
        self,
        learning_rate=1e-5,
        fig_size=112,
        label=None,
        dropout=0.2,
        n_pos=64,
        kernel_size=5,
        patch_size=7,
        n_genes=785,
        depth1=2,
        depth2=8,
        depth3=4,
        heads=16,
        channel=32,
        zinb=0,
        nb=False,
        bake=0,
        lamb=0,
        policy="mean",
    ):
        super().__init__()
        dim = (fig_size // patch_size) ** 2 * channel // 8
        self.learning_rate = learning_rate
        self.nb = nb
        self.zinb = zinb
        self.bake = bake
        self.lamb = lamb
        self.label = label
        self.patch_embedding = nn.Conv2d(3, channel, patch_size, patch_size)
        self.x_embed = nn.Embedding(n_pos, dim)
        self.y_embed = nn.Embedding(n_pos, dim)
        self.vit = ViT(
            channel=channel, kernel_size=kernel_size, heads=heads,
            dim=dim, depth1=depth1, depth2=depth2, depth3=depth3,
            mlp_dim=dim, dropout=dropout, policy=policy, gcn=True,
        )
        self.channel = channel
        self.patch_size = patch_size
        self.n_genes = n_genes
        self.dim = dim
        self.n_pos = n_pos
        self.fig_size = fig_size
        if self.zinb > 0:
            if self.nb:
                self.hr = nn.Linear(dim, n_genes)
                self.hp = nn.Linear(dim, n_genes)
            else:
                self.mean = nn.Sequential(nn.Linear(dim, n_genes), MeanAct())
                self.disp = nn.Sequential(nn.Linear(dim, n_genes), DispAct())
                self.pi = nn.Sequential(nn.Linear(dim, n_genes), nn.Sigmoid())
        if self.bake > 0:
            self.coef = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Linear(dim, 1),
            )
            # 官方 bake 自蒸馏的增强（HIST2ST.py:132-135 原样）
            self.tf = tv_transforms.Compose([
                tv_transforms.RandomGrayscale(0.1),
                tv_transforms.RandomRotation(90),
                tv_transforms.RandomHorizontalFlip(0.2),
            ])
        self.gene_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, n_genes),
        )

    def forward(self, patches, centers, adj, aug=False):
        B, N, C, H, W = patches.shape
        patches = patches.reshape(B * N, C, H, W)
        patches = self.patch_embedding(patches)
        centers_x = self.x_embed(centers[:, :, 0])
        centers_y = self.y_embed(centers[:, :, 1])
        ct = centers_x + centers_y
        h = self.vit(patches, ct, adj)
        x = self.gene_head(h)
        extra = None
        if self.zinb > 0:
            if self.nb:
                r = self.hr(h)
                p = self.hp(h)
                extra = (r, p)
            else:
                m = self.mean(h)
                d = self.disp(h)
                p = self.pi(h)
                extra = (m, d, p)
        if aug:
            h = self.coef(h)
        return x, extra, h

    def aug(self, patch, center, adj):
        """bake 个增强视图的自蒸馏输入（官方 HIST2ST.py:160-166 原样）。"""
        bake_x = []
        for _ in range(self.bake):
            new_patch = self.tf(patch.squeeze(0)).unsqueeze(0)
            x, _extra, h = self(new_patch, center, adj, True)
            bake_x.append((x.unsqueeze(0), h.unsqueeze(0)))
        return bake_x

    def distillation(self, bake_x):
        """softmax 加权聚合增强视图（官方 HIST2ST.py:167-173 原样）。"""
        new_x, coef = zip(*bake_x)
        coef = torch.cat(coef, 0)
        new_x = torch.cat(new_x, 0)
        coef = F.softmax(coef, dim=0)
        new_x = (new_x * coef).sum(0)
        return new_x


class Hist2STModel(nn.Module):
    """benchmark 统一接口包装：input_type='patch'，核心为官方 Hist2ST。

    注意：Hist2ST 是**全切片图方法**（对 ROI 内所有细胞做自注意力 + GNN，
    需要整组细胞 + 局部邻接图），不存在逐 batch 的 forward(x)。
    `forward(x)` 因此不可用，请走 methods.hist2st.evaluate_slide / predict_slide
    （在 __init__.py 中实现，负责 ROI 切片 + 邻接构建 + 预测对齐）。

    测试提速说明：官方架构 fig_size=112 → dim=1024。CPU 冒烟测试可用
    fig_size=56（dim=256，patch_size=7 仍整除），默认保持官方 fig_size=112。
    """

    input_type = "patch"

    def __init__(
        self,
        num_genes,
        fig_size=112,
        n_pos=64,
        learning_rate=1e-5,
        dropout=0.2,
        kernel_size=5,
        patch_size=7,
        depth1=2,
        depth2=8,
        depth3=4,
        heads=16,
        channel=32,
        zinb=0,
        nb=False,
        bake=0,
        lamb=0,
        policy="mean",
    ):
        super().__init__()
        self.num_genes = num_genes
        self.fig_size = fig_size
        self.n_pos = n_pos
        self.core = Hist2ST(
            learning_rate=learning_rate, fig_size=fig_size, n_pos=n_pos,
            kernel_size=kernel_size, patch_size=patch_size, n_genes=num_genes,
            depth1=depth1, depth2=depth2, depth3=depth3, heads=heads,
            channel=channel, dropout=dropout, zinb=zinb, nb=nb, bake=bake,
            lamb=lamb, policy=policy,
        )

    def forward(self, x):
        raise NotImplementedError(
            "Hist2ST 为全切片图方法（自注意力+GNN over ROI cells，需要整组细胞+邻接图），"
            "无逐 batch forward(x)。请使用 methods.hist2st.evaluate_slide / predict_slide。"
        )
