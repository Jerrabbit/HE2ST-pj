"""通用 H&E+ST 配对数据集（所有方法复用）。

数据目录约定（与现有 xenium_rep1/rep2 一致，见 common/data/preprocess.py）：
    data_dir/
        metadata.csv          (cell_id, x_centroid, y_centroid, image_col, image_row, patch_path)
        gene_expression.npy   (N, G) 表达矩阵
        gene_names.txt        每行一个基因名
        patches/cell_{id}.png 256×256 H&E patch
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class HESTDataset(Dataset):
    """由 (H&E patch, ST 表达向量) 配对组成的数据集。

    参数：
        data_dir: 数据集目录（见模块 docstring）
        gene_list: 基因子集（None = 全部基因）
        gene_norm: 表达归一化：
            'log1p_zscore'   log1p 后按基因 z-score（默认）
            'log1p_norm_total' 库大小归一化后 log1p（SpatialEx 原始做法）
            'none'           不处理
        img_size: 目标图像尺寸（0 = 保持原图）
        debug: 仅加载前 100 个样本
    """

    def __init__(
        self,
        data_dir: str,
        gene_list: list[str] | None = None,
        gene_norm: str = "log1p_zscore",
        img_size: int = 0,
        debug: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.gene_norm = gene_norm

        meta_path = os.path.join(data_dir, "metadata.csv")
        self.metadata = pd.read_csv(meta_path)
        if debug and len(self.metadata) > 100:
            self.metadata = self.metadata.iloc[:100]

        expr_path = os.path.join(data_dir, "gene_expression.npy")
        if not os.path.exists(expr_path):
            raise FileNotFoundError(f"gene_expression.npy 不存在于 {expr_path}，请先运行预处理")
        self.expr_all = np.load(expr_path).astype(np.float32)  # (N, G)

        gene_names_path = os.path.join(data_dir, "gene_names.txt")
        if os.path.exists(gene_names_path):
            with open(gene_names_path) as f:
                self.gene_names = [line.strip() for line in f]
        else:
            self.gene_names = [f"gene_{i}" for i in range(self.expr_all.shape[1])]

        if gene_list is not None:
            self.gene_list = list(gene_list)
            idx = [self.gene_names.index(g) for g in self.gene_list]
            self.expr_all = self.expr_all[:, idx]
        else:
            self.gene_list = self.gene_names

        if gene_norm == "log1p_zscore":
            expr = np.log1p(self.expr_all)
            self.means = expr.mean(axis=0, keepdims=True)
            self.stds = expr.std(axis=0, keepdims=True)
            self.stds[self.stds < 1e-8] = 1.0
            self.expr_all = ((expr - self.means) / self.stds).astype(np.float32)
        elif gene_norm == "log1p_norm_total":
            self._apply_norm_total_log1p()

    def _apply_norm_total_log1p(self) -> None:
        """SpatialEx 原始做法：库大小归一化（各细胞总counts对齐到中位数）后 log1p。"""
        X = self.expr_all
        lib = X.sum(axis=1, keepdims=True)
        lib[lib == 0] = 1
        X = X / lib * np.median(lib)
        self.expr_all = np.log1p(X).astype(np.float32)

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
