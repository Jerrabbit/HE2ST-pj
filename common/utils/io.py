"""通用 I/O 工具：特征/结果保存与加载。

特征提取耗时较长，建议缓存为 .npy 复用。
"""
from __future__ import annotations

import numpy as np


def save_features(features: np.ndarray, path: str) -> None:
    """保存特征矩阵（.npy），供训练/测试复用。"""
    raise NotImplementedError("待实现")


def load_features(path: str) -> np.ndarray:
    """加载特征矩阵。"""
    raise NotImplementedError("待实现")
