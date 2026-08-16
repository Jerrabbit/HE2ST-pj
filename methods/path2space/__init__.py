"""Path2Space：深度学习回归模型，从 H&E 全切片图像直接预测空间基因表达。

本方法为**推理专用**（官方冻结模型，154 个 MLP 集成 + CTransPath 特征），
官方权重与 CTransPath 特征需在部署时生成。使用流程见 scripts/test_path2space.py。

模型接口：
    build_model(num_genes, ensemble_dir, genes_txt, gene_names, ...)
        → Path2SpaceModel（input_type='feature'，feature_file='X_ctranspath.npy'）
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))  # 项目根目录

import torch

from common.benchmark.harness import evaluate
from common.data.dataset import FeatureDataset
from .model import Path2SpaceMLP, Path2SpaceModel

__all__ = ["Path2SpaceModel", "Path2SpaceMLP", "build_model"]


def build_model(num_genes: int = 313, variant: str = "train", feature_file=None, **kwargs):
    """统一模型工厂（scripts/train.py 调用）。

    - variant='train'（默认）：**可训练版** Path2SpaceMLP —— 冻结 CTransPath
      特征 + 训练官方 MLP 头（项目设置"编码器冻结、MLP 训练"）。特征来自
      data_dir/X_ctranspath_ctx512.npy（Macenko + 大上下文，extract_ctranspath_context.py）。
    - variant='frozen'：官方冻结 154-MLP 集成（推理专用，用 test_path2space*.py 评估）。
    """
    if variant == "frozen":
        return Path2SpaceModel(num_genes=num_genes, **kwargs)
    model = Path2SpaceMLP(num_genes=num_genes, **kwargs)
    if feature_file:
        model.feature_file = feature_file
    return model


def evaluate_frozen(
    model: Path2SpaceModel,
    test_dir: str,
    gene_names: list[str] | None = None,
    batch_size: int = 512,
    device: str | torch.device | None = None,
    output_dir: str = "outputs",
) -> dict:
    """对测试集跑冻结 Path2Space 并返回统一指标。

    gene_norm 固定为 'none'：表达以 raw counts 语义比较（模型已转 raw）。
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = FeatureDataset(test_dir, feature_path=os.path.join(test_dir, model.feature_file),
                        gene_list=gene_names, gene_norm="none")
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4)
    results = evaluate(model, loader, device, gene_norm="none", stats=None)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "test_results.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results
