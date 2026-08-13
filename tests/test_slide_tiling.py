"""slide_tiling 共享模块测试：ROI 切片 + kNN 邻接 + 超图构建。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 项目根目录

from common.data.slide_tiling import (
    build_hypergraph,
    knn_adjacency,
    normalize_hypergraph_hpnn,
    tile_rois,
)


def _coords(n=500, extent=1000.0, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, extent, (n, 2))


def test_tile_rois_covers_all():
    coords = _coords()
    rois = tile_rois(coords, roi_size=200, stride=100, min_cells=5)
    union = np.unique(np.concatenate(rois))
    assert len(union) == len(coords)  # 每个细胞至少属于一个 ROI


def test_knn_adjacency_shape_and_degree():
    coords = _coords()
    A = knn_adjacency(coords, k=8)
    assert A.shape == (len(coords), len(coords))
    deg = np.asarray(A.sum(axis=1)).ravel()
    assert (deg >= 1).all() and (deg <= 8).all()
    # 无自环
    assert A.diagonal().sum() == 0


def test_hypergraph_hpnn_normalization():
    coords = _coords(n=200)
    H = build_hypergraph(coords, k=7, self_loop=True)
    assert H.shape == (200, 200)
    assert (np.asarray(H.sum(axis=0)) > 0).all()  # 无空超边
    M = normalize_hypergraph_hpnn(H)
    assert M.shape == (200, 200)
    assert np.isfinite(M.data).all()


def test_sparse_to_torch():
    import torch
    from common.data.slide_tiling import sparse_to_torch
    coords = _coords(n=100)
    A = knn_adjacency(coords, k=5)
    T = sparse_to_torch(A, "cpu")
    assert isinstance(T, torch.Tensor) and T.is_sparse
    assert tuple(T.shape) == (100, 100)
