"""SpatialTransformer（官方 STFlow stflow/model/transformer.py 纯 torch 移植）。

- einops 替换为等价 native torch（view/reshape/[:, None, :]/torch.einsum）。
- torch_geometric 的 to_dense_batch 替换为本地实现（远程无 torch_geometric）。
- **对齐官方**：GeneUpdate 无 softplus（官方无 non_negative 约束）；activation=swiglu 时
  mlp_attn/edge_trans/W_output/TransformerBlock.mlp 用 timm SwiGLUPacked（官方同）；
  mlp_attn 的基因维从官方硬编码 50 按实际 n_genes=313 适配（消息含基因表达差特征）。
架构零改动（与官方 checkpoint 兼容：无预训练权重，从头训）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, SwiGLUPacked

from .fa import FrameAveraging


def get_activation(activation="gelu"):
    return {
        "gelu": nn.GELU,
        "silu": nn.SiLU,
        "relu": nn.ReLU,
    }[activation]


def to_dense_batch(x: torch.Tensor, batch: torch.Tensor, fill_value=0.0,
                   max_num_nodes: int | None = None) -> torch.Tensor:
    """torch_geometric.utils.to_dense_batch 的本地等价实现。

    x: [N_total, D]，batch: [N_total] int。按 batch 分组为 [B, N_max, D] 填充。
    本仓库适配里每 batch=1（单 ROI），等价于 unsqueeze(0)。
    """
    num_nodes = int(batch.max()) + 1
    max_num = max_num_nodes if max_num_nodes is not None else num_nodes
    D = x.size(1)
    out = torch.full((num_nodes, max_num, D), fill_value, dtype=x.dtype, device=x.device)
    for b in range(num_nodes):
        idx = batch == b
        cnt = int(idx.sum())
        if cnt:
            out[b, :cnt] = x[idx]
    return out


class GeneUpdate(nn.Module):
    def __init__(self, d_model, n_genes, proj_drop=0.):
        super(GeneUpdate, self).__init__()
        self.output = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(proj_drop),
            nn.Linear(d_model, n_genes),
            nn.Dropout(proj_drop),
        )

    def forward(self, features):
        return self.output(features)


class MLPAttnEdgeAggregation(FrameAveraging):
    def __init__(self, d_model, d_edge_model, n_genes, n_heads=1, proj_drop=0.,
                 attn_drop=0., activation='gelu'):
        super(MLPAttnEdgeAggregation, self).__init__(dim=2)

        self.d_head, self.d_edge_head, self.n_heads = d_model // n_heads, d_edge_model // n_heads, n_heads

        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 3),
        )

        # 官方硬编码 50（其 50 基因面板）；按实际 n_genes 适配（message 含基因表达差特征）。
        # 官方 swiglu 分支用 SwiGLUPacked（transformer.py L61-73），本地对齐。
        if activation == "swiglu":
            self.mlp_attn = SwiGLUPacked(
                in_features=self.d_head * 2 + self.d_edge_head + n_genes,
                hidden_features=d_model, out_features=1, drop=proj_drop, norm_layer=nn.LayerNorm
            )
            self.edge_trans = SwiGLUPacked(
                in_features=self.dim + 1, hidden_features=d_edge_model, out_features=d_edge_model,
                drop=proj_drop, norm_layer=nn.LayerNorm
            )
            self.W_output = SwiGLUPacked(
                in_features=d_model + d_edge_model, hidden_features=d_model, out_features=d_model,
                drop=proj_drop, norm_layer=nn.LayerNorm
            )
        else:
            self.mlp_attn = Mlp(
                in_features=self.d_head * 2 + self.d_edge_head + n_genes,
                hidden_features=d_model, out_features=1, drop=proj_drop, norm_layer=nn.LayerNorm
            )
            self.edge_trans = Mlp(
                in_features=self.dim + 1, hidden_features=d_edge_model, out_features=d_edge_model,
                act_layer=get_activation(activation), drop=proj_drop, norm_layer=nn.LayerNorm
            )
            self.W_output = Mlp(
                in_features=d_model + d_edge_model, hidden_features=d_model, out_features=d_model,
                act_layer=get_activation(activation), drop=proj_drop, norm_layer=nn.LayerNorm
            )

        self.attn_dropout = nn.Dropout(attn_drop)

    def forward(self, gene_exp, token_embs, coords, neighbor_indices, neighbor_masks=None):
        n_tokens, n_neighbors = token_embs.size(0), neighbor_indices.size(1)
        n_heads, d_head, d_edge_head = self.n_heads, self.d_head, self.d_edge_head

        q_s, k_s, v_s = self.layernorm_qkv(token_embs).chunk(3, dim=-1)
        # einops 'n (h d) -> n h d'
        q_s, k_s, v_s = map(lambda x: x.view(n_tokens, n_heads, -1), (q_s, k_s, v_s))

        # pairwise representation with FA
        radial_coords = coords[neighbor_indices] - coords.unsqueeze(dim=1)  # [N, N_neighbor, 2]
        radial_coord_norm = radial_coords.norm(dim=-1).unsqueeze(-1)  # [N, N_neighbor, 1]

        frame_feats, _, _ = self.create_frame(radial_coords, neighbor_masks)  # [N*8, N_neighbors, 3]
        frame_feats = frame_feats.view(n_tokens, self.n_frames, n_neighbors, -1)

        radial_coord_norm = radial_coord_norm.unsqueeze(dim=1).expand(n_tokens, self.n_frames, n_neighbors, -1)
        frame_feats = self.edge_trans(torch.cat([frame_feats, radial_coord_norm], dim=-1)).mean(dim=1)

        # gene expression features
        gene_exp_diff = gene_exp[neighbor_indices] - gene_exp.unsqueeze(dim=1)
        gene_exp_feats_expand = gene_exp_diff[..., None, :].expand(n_tokens, n_neighbors, n_heads, -1)

        # attention map
        q_s = q_s.unsqueeze(dim=1).expand(n_tokens, n_neighbors, n_heads, d_head)
        frame_feats = frame_feats.view(n_tokens, n_neighbors, n_heads, d_edge_head)
        message = torch.cat([q_s, k_s[neighbor_indices], frame_feats, gene_exp_feats_expand], dim=-1)

        attn_map = self.mlp_attn(message).squeeze(-1)
        if neighbor_masks is not None:
            attn_map.masked_fill_(neighbor_masks.unsqueeze(dim=-1), -1e9)
        attn_map = self.attn_dropout(nn.Softmax(dim=-1)(attn_map.transpose(1, 2)))  # [N, n_heads, N_neighbor]

        # context aggregation
        v_s_neighs = v_s[neighbor_indices].view(n_tokens, -1, n_heads, d_head)
        scalar_context = torch.einsum('nhm,nmhd->nhd', attn_map, v_s_neighs).view(n_tokens, -1)
        edge_context = torch.einsum('nhm,nmhd->nhd', attn_map, frame_feats).view(n_tokens, -1)
        return self.W_output(torch.cat([scalar_context, edge_context], dim=-1))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_edge_model, n_genes, n_heads=1, activation="gelu",
                 attn_drop=0., proj_drop=0., mlp_ratio=4.0):
        super(TransformerBlock, self).__init__()

        self.attn = MLPAttnEdgeAggregation(
            d_model=d_model, d_edge_model=d_edge_model, n_genes=n_genes,
            n_heads=n_heads, proj_drop=proj_drop, attn_drop=attn_drop,
            activation=activation
        )
        # 官方 swiglu 分支用 SwiGLUPacked（transformer.py L150-153），本地对齐。
        if activation == "swiglu":
            self.mlp = SwiGLUPacked(
                in_features=d_model, hidden_features=int(d_model * mlp_ratio),
                drop=proj_drop, norm_layer=nn.LayerNorm
            )
        else:
            self.mlp = Mlp(
                in_features=d_model, hidden_features=int(d_model * mlp_ratio),
                act_layer=get_activation(activation), drop=proj_drop, norm_layer=nn.LayerNorm
            )
        self.gene_updater = GeneUpdate(d_model, n_genes, proj_drop=proj_drop)

    def forward(self, gene_exp, token_embs, coords, neighbor_indices):
        context_token_embs = self.attn(gene_exp, token_embs, coords, neighbor_indices)
        token_embs = token_embs + context_token_embs
        token_embs = token_embs + self.mlp(token_embs)
        gene_exp = self.gene_updater(token_embs)
        return gene_exp, token_embs


class SpatialTransformer(nn.Module):
    def __init__(self, config):
        super(SpatialTransformer, self).__init__()

        self.n_neighbors = config.n_neighbors

        self.blks = nn.ModuleList([
            TransformerBlock(config.d_model, config.d_edge_model,
                             n_genes=config.n_genes, n_heads=config.n_heads,
                             activation=config.act, attn_drop=config.attn_dropout,
                             proj_drop=config.dropout)
            for _ in range(config.n_layers)
        ])

    def _build_graph(self, coords, batch_idx, n_neighbors, exclude_self=True):
        # coords: [N, 2], batch_idx: [N], n_neighbors: int
        exclude_self_mask = torch.eye(coords.shape[0], dtype=torch.bool, device=coords.device)
        batch_mask = batch_idx.unsqueeze(0) == batch_idx.unsqueeze(1)

        rel_pos = coords[:, None, :] - coords[None, :, :]  # einops 等价
        rel_dist = rel_pos.norm(dim=-1).detach()
        if exclude_self:
            rel_dist.masked_fill_(exclude_self_mask | ~batch_mask, 1e9)
        else:
            rel_dist.masked_fill_(~batch_mask, 1e9)

        dist_values, nearest_indices = rel_dist.topk(n_neighbors, dim=-1, largest=False)
        return nearest_indices

    def forward(self, gene_exp, features, coords):
        # gene_exp: [B, N_cells, N_genes], features: [B, N_cells, -1], coords: [B, N_cells, 2]
        B, N_cells, N_genes = gene_exp.shape[0], gene_exp.shape[1], gene_exp.shape[-1]
        device = features.device

        pad_mask = features.sum(dim=-1) == 0  # [B, N_cells]
        batch_idx = torch.arange(B, device=device).unsqueeze(-1).repeat(1, N_cells)[~pad_mask]

        features = features[~pad_mask]
        coords = coords[~pad_mask]
        gene_exp = gene_exp[~pad_mask]

        nearest_indices = self._build_graph(
            coords, batch_idx, min(self.n_neighbors, N_cells), exclude_self=True
        )

        all_gene_exp = []
        for blk in self.blks:
            gene_exp, features = blk(gene_exp, features, coords, nearest_indices)
            all_gene_exp.append(gene_exp)
        gene_exp = torch.stack(all_gene_exp, dim=0).mean(dim=0)

        gene_exp = to_dense_batch(gene_exp, batch=batch_idx, fill_value=0.0,
                                  max_num_nodes=N_cells)  # [B, N_cells, N_genes]
        return gene_exp
