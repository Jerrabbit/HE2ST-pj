"""UNI2 Foundation Model 特征提取封装。

课题要求：UNI+MLP 方法使用 UNI2（远程服务器上已有 UNI1、UNI2 权重文件）。
UNI2 输入 224×224 图像块，输出 token / [CLS] 特征，作为下游 MLP 的输入。
"""
from __future__ import annotations

import numpy as np


class UNI2FeatureExtractor:
    """加载 UNI2 权重并提取图像块特征。

    使用说明：
        - 权重路径：远程服务器（本地调试时需自行下载）
        - 输入：(B, 3, 224, 224) 图像块
        - 输出：特征向量 (B, D)
    """

    def __init__(self, weight_path: str, device: str = "cuda"):
        self.weight_path = weight_path
        self.device = device
        raise NotImplementedError("待实现：加载 UNI2 模型（参考官方 D:\\hest_data\\codes 代码）")

    def extract(self, patches: np.ndarray) -> np.ndarray:
        """提取图像块特征。返回 (B, D) 特征矩阵。"""
        raise NotImplementedError("待实现")
