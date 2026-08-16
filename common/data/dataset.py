"""通用 H&E+ST 配对数据集（所有方法复用）。

数据目录约定（与 common/data/preprocess.py 输出一致）：
    data_dir/
        metadata.csv          (cell_id, x_centroid, y_centroid, patch_path)
        gene_expression.npy   (N, G) **raw counts** 表达矩阵
        gene_names.txt        每行一个基因名
        patches/cell_{id}.png 256×256 H&E patch

提供两个数据集：
- HESTDataset：patch 输入（CNN/Transformer 类方法，如 ST-Net、Hist2ST、UNI2+MLP 等）
- FeatureDataset：预提取特征输入（特征型方法，如 DeepPT、SpatialEx、GHIST、UNI2+MLP 特征版）

归一化统一走 common/data/expression.py：z-score/norm_total 统计量在训练集上拟合。
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .expression import load_expression, normalize_expression

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def _load_targets(
    data_dir: str,
    gene_list: list[str] | None,
    gene_norm: str,
    ref_stats: dict | None,
):
    """读取表达矩阵并按统一约定归一化。

    返回：
        expr_norm: (N, G) float32 归一化表达矩阵
        gene_names: 当前使用的基因名列表
        stats: 归一化统计量（供测试集复用）
    """
    expr_raw, gene_names = load_expression(data_dir)
    if gene_list is not None:
        expr_raw = expr_raw[:, [gene_names.index(g) for g in gene_list]]
        gene_names = list(gene_list)
    expr_norm, stats = normalize_expression(expr_raw, gene_norm, ref_stats)
    return expr_norm, gene_names, stats


class HESTDataset(Dataset):
    """由 (H&E patch, ST 表达向量) 配对组成的数据集（patch 输入）。

    参数：
        data_dir: 数据集目录（见模块 docstring）
        gene_list: 基因子集（None = 全部基因）
        gene_norm: 'log1p_zscore'（默认）| 'log1p_norm_total' | 'none'
        ref_stats: 训练集归一化统计量（None = 在本数据集上拟合）
        img_size: 目标图像尺寸（0 = 保持原图）
        debug: 仅加载前 100 个样本
    """

    def __init__(
        self,
        data_dir: str,
        gene_list: list[str] | None = None,
        gene_norm: str = "log1p_zscore",
        ref_stats: dict | None = None,
        img_size: int = 0,
        debug: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.gene_norm = gene_norm
        self.ref_stats = ref_stats

        meta_path = os.path.join(data_dir, "metadata.csv")
        self.metadata = pd.read_csv(meta_path)
        if debug and len(self.metadata) > 100:
            self.metadata = self.metadata.iloc[:100]

        self.expr_all, self.gene_list, self.stats = _load_targets(
            data_dir, gene_list, gene_norm, ref_stats
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        row = self.metadata.iloc[idx]
        image = Image.open(row["patch_path"]).convert("RGB")
        if self.img_size > 0:
            image = image.resize((self.img_size, self.img_size), Image.BILINEAR)

        img = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD

        coords = np.array([row["x_centroid"], row["y_centroid"]], dtype=np.float32)
        return {
            "patch": img,                                # (3, H, W) 已 ImageNet 归一化
            "gene_expr": torch.from_numpy(self.expr_all[idx].copy()),  # (G,)
            "coords": torch.from_numpy(coords),          # (2,)
            "cell_id": row["cell_id"],
        }


class FeatureDataset(Dataset):
    """由 (预提取特征, ST 表达向量) 配对组成的数据集（特征输入）。

    特征文件：data_dir/X_uni2.npy（或通过 feature_path 指定），(N, D)。
    适用于 DeepPT、SpatialEx、GHIST、UNI2+MLP（特征版）等使用预提取特征的方法。

    参数：
        data_dir: 数据集目录（含 metadata.csv、gene_expression.npy、特征 .npy）
        feature_path: 特征文件路径（默认 data_dir/X_uni2.npy）
        gene_list / gene_norm / ref_stats: 同 HESTDataset
        debug: 仅加载前 100 个样本
    """

    def __init__(
        self,
        data_dir: str,
        feature_path: str | list[str] | None = None,
        gene_list: list[str] | None = None,
        gene_norm: str = "log1p_zscore",
        ref_stats: dict | None = None,
        debug: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir

        # 支持多特征文件（Local+Global）：str 或 list[str]，按序加载并沿最后一维 concat
        if feature_path is None:
            feature_path = os.path.join(data_dir, "X_uni2.npy")
        if isinstance(feature_path, str):
            feature_path = [feature_path]
        feats = []
        for fp in feature_path:
            if os.path.dirname(fp) == "":  # 裸文件名 → 视为相对 data_dir
                fp = os.path.join(data_dir, fp)
            if not os.path.exists(fp):
                raise FileNotFoundError(f"特征文件不存在: {fp}")
            feats.append(np.load(fp).astype(np.float32))  # (N, D_i)
        self.features = np.concatenate(feats, axis=1)      # (N, sum D_i)

        self.metadata = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
        if debug and len(self.metadata) > 100:
            self.metadata = self.metadata.iloc[:100]

        self.expr_all, self.gene_list, self.stats = _load_targets(
            data_dir, gene_list, gene_norm, ref_stats
        )

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        row = self.metadata.iloc[idx]
        coords = np.array([row["x_centroid"], row["y_centroid"]], dtype=np.float32)
        return {
            "feature": torch.from_numpy(self.features[idx].copy()),      # (D,)
            "gene_expr": torch.from_numpy(self.expr_all[idx].copy()),    # (G,)
            "coords": torch.from_numpy(coords),                          # (2,)
            "cell_id": row["cell_id"],
        }
