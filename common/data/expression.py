"""基因表达数据的读取与归一化（所有方法共用，保证公平比较）。

约定：data_dir/gene_expression.npy 存 **raw counts**（整数，(N, G)），
data_dir/gene_names.txt 每行一个基因名。

归一化统计量（z-score 的 mean/std、norm_total 的中位数库大小）必须只在
**训练集**上拟合，再应用到验证/测试集，避免测试集信息泄漏。
"""
from __future__ import annotations

import os

import numpy as np


def load_expression(data_dir: str) -> tuple[np.ndarray, list[str]]:
    """读取 raw counts 表达矩阵与基因名。

    返回：
        expr: (N, G) float32 raw counts
        gene_names: [G] 基因名列表
    """
    expr_path = os.path.join(data_dir, "gene_expression.npy")
    if not os.path.exists(expr_path):
        raise FileNotFoundError(f"gene_expression.npy 不存在于 {expr_path}，请先运行预处理")
    expr = np.load(expr_path).astype(np.float32)

    names_path = os.path.join(data_dir, "gene_names.txt")
    if os.path.exists(names_path):
        with open(names_path) as f:
            gene_names = [line.strip() for line in f]
    else:
        gene_names = [f"gene_{i}" for i in range(expr.shape[1])]
    return expr, gene_names


def subset_genes(
    expr_raw: np.ndarray, gene_list: list[str], gene_names: list[str]
) -> np.ndarray:
    """按基因名子集（保证与其它方法使用相同公共基因，公平比较）。"""
    idx = [gene_names.index(g) for g in gene_list]
    return expr_raw[:, idx]


def normalize_expression(
    expr_raw: np.ndarray,
    gene_norm: str = "log1p_zscore",
    ref_stats: dict | None = None,
) -> tuple[np.ndarray, dict]:
    """按统一约定归一化表达矩阵。

    参数：
        expr_raw: (N, G) raw counts
        gene_norm: 'log1p_zscore'（默认）| 'log1p_norm_total' | 'none'
        ref_stats: 训练集拟合的统计量（None 表示在 expr_raw 上拟合）

    返回：
        (expr, stats)：
            expr: (N, G) 归一化表达矩阵
            stats: 使用的统计量 dict（供后续数据集复用）
                - zscore: {'means': (1,G), 'stds': (1,G)}
                - norm_total: {'median_lib': float}
                - none: {}
    """
    if gene_norm == "log1p_norm_total":
        X = expr_raw.copy()
        lib = X.sum(axis=1, keepdims=True)
        lib[lib == 0] = 1
        if ref_stats is None:
            median_lib = float(np.median(lib))
        else:
            median_lib = ref_stats["median_lib"]
        X = X / lib * median_lib
        return np.log1p(X).astype(np.float32), {"median_lib": median_lib}
    if gene_norm == "none":
        return expr_raw, {}
    # log1p_zscore（默认）
    expr = np.log1p(expr_raw)
    if ref_stats is None:
        means = expr.mean(axis=0, keepdims=True)
        stds = expr.std(axis=0, keepdims=True)
        stds[stds < 1e-8] = 1.0
    else:
        means, stds = ref_stats["means"], ref_stats["stds"]
    return ((expr - means) / stds).astype(np.float32), {"means": means, "stds": stds}
