"""BLEEP 冒烟测试：对比损失、参考库检索、完整 train_function 流程（CPU）。"""
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
from methods.bleep import build_model
from methods.bleep.model import BLEEP, clip_soft_target_loss


def _make_fixture(root: Path, n: int = 60, g: int = 8, seed: int = 0):
    """构造最小 patch 数据集（与 test_datasets 一致）。"""
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
        self.epochs = 2
        self.gene_norm = "log1p_zscore"


def test_clip_soft_target_loss_shape():
    b = torch.randn(8, 256)
    loss = clip_soft_target_loss(b, b)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_bleep_forward_retrieval(tmp_path):
    _make_fixture(tmp_path / "train")
    _make_fixture(tmp_path / "test", seed=1)
    ds = HESTDataset(str(tmp_path / "train"), gene_norm="log1p_zscore")
    loader = DataLoader(ds, batch_size=16, shuffle=False)

    model = build_model(num_genes=8, pretrained=False, image_backbone="resnet18",
                        top_k=5)
    assert model.input_type == "patch"

    # 对比训练几步
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    for batch in loader:
        img = model.image_embed(batch["patch"])
        spot = model.spot_embed(batch["gene_expr"])
        loss = clip_soft_target_loss(spot, img)
        loss.backward()
        opt.step()
        opt.zero_grad()
    assert torch.isfinite(loss)

    # 建参考库并检索评估
    model.eval()
    with torch.no_grad():
        expr = torch.cat([b["gene_expr"] for b in loader])
        model.build_reference(expr)
    ds_test = HESTDataset(str(tmp_path / "test"), gene_norm="log1p_zscore",
                          ref_stats=ds.stats)
    loader_test = DataLoader(ds_test, batch_size=16, shuffle=False)
    res = evaluate(model, loader_test, "cpu", "log1p_zscore", ds.stats)
    assert "PCC" in res and "AUROC" in res


def test_bleep_train_function(tmp_path):
    _make_fixture(tmp_path / "train")
    _make_fixture(tmp_path / "test", seed=1)
    ds = HESTDataset(str(tmp_path / "train"), gene_norm="log1p_zscore")
    ds_test = HESTDataset(str(tmp_path / "test"), gene_norm="log1p_zscore",
                          ref_stats=ds.stats)
    loader = DataLoader(ds, batch_size=16, shuffle=True)
    loader_test = DataLoader(ds_test, batch_size=16, shuffle=False)

    import methods.bleep as bleep
    model = build_model(num_genes=8, pretrained=False, image_backbone="resnet18",
                        top_k=5)
    hist = bleep.train_function(model, loader, loader_test, _Args(tmp_path), ds.stats)
    assert len(hist) == 2
    ckpt = torch.load(tmp_path / "best.pt", map_location="cpu")
    assert "reference" in ckpt

    # post_load 恢复参考库后即可检索
    m2 = build_model(num_genes=8, pretrained=False, image_backbone="resnet18",
                     top_k=5)
    m2.load_state_dict(ckpt["model"])
    m2 = bleep.post_load(m2, ckpt)
    assert m2.reference is not None
    out = m2(ds_test[0]["patch"].unsqueeze(0))
    assert out.shape == (1, 8)
