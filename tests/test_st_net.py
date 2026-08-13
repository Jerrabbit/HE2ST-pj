"""ST-Net 冒烟测试：DenseNet121 前向 + bias 初始化 + 完整训练流程（CPU，小数据）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 项目根目录

from common.benchmark.harness import evaluate
from common.data.dataset import HESTDataset
from methods.st_net import build_model, train_function


def _make_fixture(root: Path, n: int = 30, g: int = 6, seed: int = 0):
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    expr = rng.poisson(2.0, size=(n, g)).astype(np.float32)
    np.save(root / "gene_expression.npy", expr)
    (root / "gene_names.txt").write_text("\n".join(f"gene{i}" for i in range(g)))
    (root / "patches").mkdir()
    rows = []
    for i in range(n):
        Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(
            root / "patches" / f"cell_{i}.png"
        )
        rows.append({"cell_id": f"c{i}", "x_centroid": i, "y_centroid": i,
                     "patch_path": str(root / "patches" / f"cell_{i}.png")})
    pd.DataFrame(rows).to_csv(root / "metadata.csv", index=False)


class _Args:
    def __init__(self, output_dir):
        self.device = "cpu"
        self.output_dir = str(output_dir)
        self.lr = 1e-3
        self.weight_decay = 0.0
        self.epochs = 2
        self.gene_norm = "log1p_norm_total"


def test_stnet_forward():
    model = build_model(num_genes=6, pretrained=False)
    assert model.input_type == "patch"
    x = torch.randn(4, 3, 64, 64)
    out = model(x)
    assert out.shape == (4, 6)


def test_stnet_bias_init():
    model = build_model(num_genes=6, pretrained=False)
    mean = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    model.set_bias_init(mean)
    assert torch.allclose(model.backbone.classifier.bias, mean)
    assert torch.all(model.backbone.classifier.weight == 0)


def test_stnet_train_function(tmp_path):
    _make_fixture(tmp_path / "train")
    _make_fixture(tmp_path / "test", seed=1)
    ds = HESTDataset(str(tmp_path / "train"), gene_norm="log1p_norm_total")
    ds_test = HESTDataset(str(tmp_path / "test"), gene_norm="log1p_norm_total",
                          ref_stats=ds.stats)
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    loader_test = DataLoader(ds_test, batch_size=8, shuffle=False)

    import methods.st_net as stnet
    model = build_model(num_genes=6, pretrained=False)
    hist = stnet.train_function(model, loader, loader_test, _Args(tmp_path),
                                ds.stats)
    assert len(hist) == 2
    ckpt = torch.load(tmp_path / "best.pt", map_location="cpu")
    m2 = build_model(num_genes=6, pretrained=False)
    m2.load_state_dict(ckpt["model"])
    res = evaluate(m2, loader_test, "cpu", "log1p_norm_total", ds.stats)
    assert "PCC" in res and "AUROC" in res
