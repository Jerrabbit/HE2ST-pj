"""CTransPath 特征提取器（官方 Path2Space 的 CTransPathExtractor 适配版）。

CTransPath（Wang et al. 2022）是 Swin Transformer 变体，输出 768 维 tile 特征。
本模块复刻官方 ge_model/path2space/features.py 的推理管线：
    224×224 resize（BILINEAR, antialias=False）→ ToTensor → ImageNet Normalize
    → 冻结 CTransPath → 768 维特征

用途：Path2Space 方法需要 CTransPath 特征（X_ctranspath.npy），与 UNI2 特征并列。
权重：官方 ctranspath.pth（键 'model' 或裸 state_dict）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

CTRANSPATH_MEAN = (0.485, 0.456, 0.406)
CTRANSPATH_STD = (0.229, 0.224, 0.225)
CTRANSPATH_FEATURE_DIM = 768
CTRANSPATH_TILE_SIZE = 224


def _load_ctranspath_model(weights_path: str, device: torch.device):
    """从官方 ctranspath.pth 加载冻结 CTransPath。"""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))  # 项目根目录
    from methods.path2space.frozen.ctrans import CTransPath

    model = CTransPath(num_classes=0).to(device)
    state = torch.load(str(weights_path), map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    return model


class CTransPathExtractor:
    """冻结 CTransPath 批量特征提取器（官方 features.py 语义）。"""

    def __init__(
        self,
        weights_path: str | os.PathLike,
        device: str | torch.device | None = None,
        batch_size: int = 128,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.model = _load_ctranspath_model(weights_path, self.device)
        self.transform = transforms.Compose([
            transforms.Resize(CTRANSPATH_TILE_SIZE, antialias=False),
            transforms.ToTensor(),
            transforms.Normalize(mean=CTRANSPATH_MEAN, std=CTRANSPATH_STD),
        ])

    @torch.no_grad()
    def extract(self, tiles) -> np.ndarray:
        """对 PIL 图迭代器批量提特征，返回 (N, 768) float32。"""
        out: list[np.ndarray] = []
        batch: list[Image.Image] = []
        for tile in tiles:
            batch.append(tile)
            if len(batch) == self.batch_size:
                out.append(self._extract_batch(batch))
                batch.clear()
        if batch:
            out.append(self._extract_batch(batch))
        if not out:
            return np.zeros((0, CTRANSPATH_FEATURE_DIM), dtype=np.float32)
        return np.concatenate(out, axis=0).astype(np.float32, copy=False)

    @torch.no_grad()
    def _extract_batch(self, batch: list[Image.Image]) -> np.ndarray:
        x = torch.stack([self.transform(t) for t in batch]).to(self.device).float()
        y = self.model(x)
        return y.cpu().numpy()


def extract_features_from_patches(
    patch_paths: list[str],
    weights_path: str,
    batch_size: int = 128,
    device: str | torch.device | None = None,
    num_workers: int = 0,
) -> np.ndarray:
    """从 patch 文件路径列表提取 CTransPath 特征（供预处理脚本复用）。"""
    extractor = CTransPathExtractor(weights_path, device, batch_size)

    def _tiles():
        for p in patch_paths:
            yield Image.open(p).convert("RGB")

    return extractor.extract(_tiles())
