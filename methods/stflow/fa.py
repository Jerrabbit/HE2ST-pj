"""FrameAveraging（官方 STFlow stflow/model/fa.py 纯 torch 移植，einops 已替换）。

官方原样，仅把 einops rearrange 换成等价 reshape。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FrameAveraging(nn.Module):
    def __init__(self, dim=3, backward=False):
        super(FrameAveraging, self).__init__()

        self.dim = dim
        self.n_frames = 2 ** dim
        self.ops = self.create_ops(dim)  # [2^dim, dim]
        self.backward = backward

    def create_ops(self, dim):
        colon = slice(None)
        accum = []
        directions = torch.tensor([-1, 1])

        for ind in range(dim):
            dim_slice = [None] * dim
            dim_slice[ind] = colon
            accum.append(directions[dim_slice])

        accum = torch.broadcast_tensors(*accum)
        operations = torch.stack(accum, dim=-1)
        # einops '... d -> (...) d' 等价：展平前面的维度
        operations = operations.reshape(-1, dim)
        return operations

    def create_frame(self, X, mask=None):
        assert X.shape[-1] == self.dim, \
            f'expected points of dimension {self.dim}, but received {X.shape[-1]}'

        if mask is None:
            mask = torch.ones(*X.shape[:-1], device=X.device).bool()
        mask = mask.unsqueeze(-1)
        center = (X * mask).sum(dim=1) / mask.sum(dim=1)
        X = X - center.unsqueeze(1) * mask  # [B,N,dim]
        X_ = X.masked_fill(~mask, 0.)

        C = torch.bmm(X_.transpose(1, 2), X_)  # [B,dim,dim] (Cov)
        if not self.backward:
            C = C.detach()

        _, eigenvectors = torch.linalg.eigh(C, UPLO='U')  # [B,dim,dim]
        F_ops = self.ops.unsqueeze(1).unsqueeze(0).to(X.device) * eigenvectors.unsqueeze(1)
        h = torch.einsum('boij,bpj->bopi', F_ops.transpose(2, 3), X)

        h = h.view(X.size(0) * self.n_frames, X.size(1), self.dim)
        return h, F_ops.detach(), center

    def invert_frame(self, X, mask, F_ops, center):
        X = torch.einsum('boij,bopj->bopi', F_ops, X)
        X = X.mean(dim=1)  # frame averaging
        X = X + center.unsqueeze(1)
        if mask is None:
            return X
        return X * mask.unsqueeze(-1)
