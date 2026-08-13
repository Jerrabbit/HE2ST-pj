"""Phoenix 冒烟测试：flow matching 训练损失 + 生成式推理（CPU，小数据）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 项目根目录

from common.benchmark.harness import evaluate
from common.data.dataset import FeatureDataset
from methods.phoenix import build_model, train_function


def _make_fixture(root: Path, n: int = 80, g: int = 10, d: int = 16, seed: int = 0):
    """构造最小特征数据集（特征 + 表达 + 元数据）。"""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    expr = rng.poisson(2.0, size=(n, g)).astype(np.float32)
    np.save(root / "gene_expression.npy", expr)
    (root / "gene_names.txt").write_text("\n".join(f"gene{i}" for i in range(g)))
    # 特征与表达有相关性，便于生成模型学到信号
    feat = (expr @ rng.normal(size=(g, d))).astype(np.float32)
    feat += rng.normal(size=(n, d)).astype(np.float32)
    np.save(root / "X_uni2.npy", feat)
    rows = [{"cell_id": f"c{i}", "x_centroid": i, "y_centroid": i,
             "patch_path": ""} for i in range(n)]
    pd.DataFrame(rows).to_csv(root / "metadata.csv", index=False)


class _Args:
    def __init__(self, output_dir):
        self.device = "cpu"
        self.output_dir = str(output_dir)
        self.lr = 1e-3
        self.weight_decay = 0.0
        self.epochs = 2
        self.gene_norm = "log1p_zscore"


def test_phoenix_training_loss_shape():
    model = build_model(num_genes=10, feature_dim=16, latent_dim=8,
                        hidden_dim=32)
    expr = torch.randn(6, 10)
    cond = torch.randn(6, 16)
    loss, detail = model.training_loss(expr, cond)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(v) for v in detail.values())


def test_phoenix_forward_deterministic():
    model = build_model(num_genes=10, feature_dim=16, latent_dim=8, hidden_dim=32,
                        n_sample_steps=5)
    model.eval()
    cond = torch.randn(4, 16)
    with torch.no_grad():
        out1 = model(cond)
        out2 = model(cond)
    assert out1.shape == (4, 10)
    # 固定 seed 的生成应可复现
    assert torch.allclose(out1, out2, atol=1e-5)


def test_phoenix_train_function(tmp_path):
    _make_fixture(tmp_path / "train")
    _make_fixture(tmp_path / "test", seed=1)
    ds = FeatureDataset(str(tmp_path / "train"), gene_norm="log1p_zscore")
    ds_test = FeatureDataset(str(tmp_path / "test"), gene_norm="log1p_zscore",
                             ref_stats=ds.stats)
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    loader_test = DataLoader(ds_test, batch_size=16, shuffle=False)

    import methods.phoenix as phoenix
    model = build_model(num_genes=10, feature_dim=16, latent_dim=8, hidden_dim=32,
                        n_sample_steps=5)
    hist = phoenix.train_function(model, loader, loader_test, _Args(tmp_path),
                                  ds.stats)
    assert len(hist) == 2
    ckpt = torch.load(tmp_path / "best.pt", map_location="cpu")
    assert "model" in ckpt
    # 恢复权重后评估应能跑通
    m2 = build_model(num_genes=10, feature_dim=16, latent_dim=8, hidden_dim=32,
                     n_sample_steps=5)
    m2.load_state_dict(ckpt["model"])
    res = evaluate(m2, loader_test, "cpu", "log1p_zscore", ds.stats)
    assert "PCC" in res and "AUROC" in res
