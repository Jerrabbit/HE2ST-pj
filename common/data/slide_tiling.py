"""滑动-图基础设施：ROI 切片 + 局部邻接/超图构建（slide-level 图方法共用）。

Xenium 切片可达 16 万+细胞，全切片图 O(N^2) 不可行，统一把切片按物理坐标
切成重叠 ROI，在每个 ROI 内构建局部图，作为方法（SpatialEx、Hist2ST 等）
训练与推理的基本单元。

纯 numpy/scipy 实现（无 torch 依赖），供各方法在 train_function / predict_slide
中调用。图语义约定与官方一致：
    - kNN 邻接图（Hist2ST 的 calcADJ 稀疏化版）：A[i,j]=1 当 j 是 i 的 k 近邻
    - 空间超图（SpatialEx）：H[i,j]=1 当 i 是超边 j（=细胞 j 及其 k 近邻）的成员
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree


def tile_rois(
    coords: np.ndarray,
    roi_size: float,
    stride: float,
    min_cells: int = 32,
) -> list[np.ndarray]:
    """按物理坐标网格把切片切成重叠 ROI。

    参数：
        coords: (N, 2) 细胞物理坐标
        roi_size: ROI 正方形边长（物理单位）
        stride: 相邻 ROI 中心间距（重叠量 = roi_size - stride）
        min_cells: ROI 内细胞数少于该值则丢弃
    返回：
        rois: list[np.ndarray[int]] 每个 ROI 的全局细胞索引
    """
    if coords.shape[1] != 2:
        raise ValueError(f"coords 需为 (N,2)，got {coords.shape}")
    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    # 生成 ROI 中心网格（覆盖全图）
    xs = np.arange(xmin, xmax + stride, stride)
    ys = np.arange(ymin, ymax + stride, stride)
    cx, cy = np.meshgrid(xs, ys)
    centers = np.stack([cx.ravel(), cy.ravel()], axis=1)
    half = roi_size / 2.0

    rois: list[np.ndarray] = []
    for c in centers:
        mask = (
            (coords[:, 0] >= c[0] - half)
            & (coords[:, 0] <= c[0] + half)
            & (coords[:, 1] >= c[1] - half)
            & (coords[:, 1] <= c[1] + half)
        )
        idx = np.flatnonzero(mask)
        if len(idx) >= min_cells:
            rois.append(idx)
    if not rois:
        # 极端情况：全图一个 ROI
        rois = [np.arange(len(coords))]
    return rois


def knn_adjacency(
    coords: np.ndarray, k: int = 8, self_loop: bool = False
) -> sparse.csr_matrix:
    """kNN 邻接图（二进制，(N,N) CSR）。用 cKDTree，O(N log N) 可扩展 16 万细胞。"""
    n = len(coords)
    tree = cKDTree(coords)
    # k+1 含自身；若 n <= k，则全部相连
    nn = min(k + 1, n)
    dist, idx = tree.query(coords, k=nn)
    idx = np.asarray(idx, dtype=np.int64)
    if idx.ndim == 1:  # k=0 或 n=1
        idx = idx[:, None]
    rows = np.repeat(np.arange(n), idx.shape[1])
    cols = idx.ravel()
    # 去掉自环（除非显式要求）
    if not self_loop:
        keep = rows != cols
        rows, cols = rows[keep], cols[keep]
    data = np.ones(len(rows), dtype=np.float32)
    A = sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    A.eliminate_zeros()
    return A


def build_hypergraph(coords: np.ndarray, k: int = 7, self_loop: bool = True) -> sparse.csr_matrix:
    """SpatialEx 风格空间超图 H（kNN 超边）。

    超边 j = {细胞 j 及其 k 个近邻}，H[i,j]=1 当 i ∈ 超边 j。
    返回 (N, N) CSR 稀疏超图关联矩阵（与官方 Build_hypergraph 语义一致）。
    """
    A = knn_adjacency(coords, k=k, self_loop=self_loop)
    # 官方：H = Build_graph(...).T；Build_graph 的 A[i,j]=1 当 j 是 i 的近邻，
    # 因此 H = A.T 的第 j 列 = {j 及其近邻} 即为超边 j。
    H = A.T.tocsr()
    # 官方 Build_graph 含自环（每个节点自身也是邻居），保证超边非空
    return H


def normalize_hypergraph_hpnn(H: sparse.spmatrix) -> sparse.csr_matrix:
    """SpatialEx 官方 hpnn 归一化 → 传播矩阵 Dv H W De H.T Dv。"""
    DE = np.asarray(H.sum(axis=0)).ravel()          # 超边度数
    DV = np.asarray(H.sum(axis=1)).ravel()          # 节点度数
    DE_inv = sparse.diags(1.0 / np.maximum(DE, 1e-8))
    DV_inv_sqrt = sparse.diags(1.0 / np.sqrt(np.maximum(DV, 1e-8)))
    M = DV_inv_sqrt @ H @ DE_inv @ (H.T) @ DV_inv_sqrt
    return M.tocsr()


def normalize_graph_gcn(A: sparse.spmatrix) -> sparse.csr_matrix:
    """GCN 对称归一化 D^-1/2 A D^-1/2。"""
    D = np.asarray(A.sum(axis=1)).ravel()
    D_inv_sqrt = sparse.diags(1.0 / np.sqrt(np.maximum(D, 1e-8)))
    return (D_inv_sqrt @ A @ D_inv_sqrt).tocsr()


def sparse_to_torch(M: sparse.spmatrix, device: str = "cpu"):
    """scipy 稀疏矩阵 → torch 稀疏张量（float32）。"""
    import torch
    M = M.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack([M.row, M.col]).astype(np.int64))
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    return torch.sparse_coo_tensor(indices, values, shape).to(device)
