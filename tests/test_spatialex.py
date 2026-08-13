"""SpatialEx 冒烟测试：HGNN/Predictor_spot 前向 + 完整 train_function + evaluate_slide（CPU）。

用小参数（hidden_dim=32，num_genes=8，特征 16 维）在 200 细胞的小切片上验证：
    1. HGNN 前向输出形状正确
    2. Predictor_spot 返回 (loss, x_prime, enc) 且形状正确
    3. train_function 完整流程（epochs=2）→ best.pt 存在 → evaluate_slide 出指标
    4. evaluate_slide 在测试集上返回有限 PCC/SPCC/AUROC
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 项目根目录

from common.data.expression import load_expression, normalize_expression
from common.data.slide_tiling import build_hypergraph, normalize_hypergraph_hpnn, sparse_to_torch
from methods.spatialex import build_model, evaluate_slide, train_function
from methods.spatialex.model import HGNN, Predictor_spot

N = 200
G = 8
D = 16
H = 32
REGION = 2000.0


def _make_fixture(root: Path, n: int = N, g: int = G, d: int = D, seed: int = 0):
    """构造最小特征数据集：X_uni2.npy + 表达 + 2D 坐标（2000×2000µm 网格）。"""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    expr = rng.poisson(2.0, size=(n, g)).astype(np.float32)
    np.save(root / "gene_expression.npy", expr)
    (root / "gene_names.txt").write_text("\n".join(f"gene{i}" for i in range(g)))
    feats = rng.normal(0, 1, (n, d)).astype(np.float32)
    np.save(root / "X_uni2.npy", feats)
    rows = []
    for i in range(n):
        rows.append({"cell_id": f"c{i}",
                     "x_centroid": float(rng.uniform(0, REGION)),
                     "y_centroid": float(rng.uniform(0, REGION)),
                     "patch_path": f"patches/cell_{i}.png"})
    pd.DataFrame(rows).to_csv(root / "metadata.csv", index=False)


def _fit_stats(data_dir: Path, gene_norm: str) -> dict:
    """在给定数据目录上拟合表达归一化统计量（与 scripts/train.py 一致）。"""
    expr_raw, _ = load_expression(str(data_dir))
    _, stats = normalize_expression(expr_raw, gene_norm)
    return stats


class _Args:
    def __init__(self, train_dir, valid_dir, output_dir, epochs=2,
                 gene_norm="log1p_norm_total"):
        self.train_dir = str(train_dir)
        self.valid_dir = str(valid_dir)
        self.output_dir = str(output_dir)
        self.device = "cpu"
        self.epochs = epochs
        self.lr = 1e-3
        self.weight_decay = 0.0
        self.gene_norm = gene_norm


def _small_graph(n_cells: int, seed: int = 0, k: int = 5):
    """构造 (n_cells, n_cells) hpnn 归一化的稀疏超图 torch 张量。"""
    coords = np.random.default_rng(seed).uniform(0, REGION, (n_cells, 2)).astype(np.float32)
    H = build_hypergraph(coords, k=k, self_loop=True)
    M = normalize_hypergraph_hpnn(H)
    return sparse_to_torch(M, "cpu")


def test_hgnn_forward():
    n_cells = 40
    Gt = _small_graph(n_cells)
    hgnn = HGNN(in_dim=D, num_hidden=H, out_dim=H, num_layers=2,
                dropout=0, activation="prelu")
    out = hgnn(torch.randn(n_cells, D), Gt)
    assert out.shape == (n_cells, H)
    assert torch.isfinite(out).all()


def test_predictor_spot_forward():
    n_cells = 40
    Gt = _small_graph(n_cells, seed=1)
    pred = Predictor_spot(in_dim=D, hidden_dim=H, out_dim=G, num_layers=2,
                          dropout=0.1, loss_fn="mse", activation="prelu", agg=False)
    loss, x_prime, enc = pred(Gt, torch.randn(n_cells, D), torch.randn(n_cells, G))
    assert x_prime.shape == (n_cells, G)
    assert enc.shape == (n_cells, H)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_train_function(tmp_path):
    train_dir = tmp_path / "train"
    valid_dir = tmp_path / "valid"
    _make_fixture(train_dir, seed=0)
    _make_fixture(valid_dir, seed=1)
    stats = _fit_stats(train_dir, "log1p_norm_total")

    # loaders 仅用于接口一致性（train_function 内部直接从 train_dir/valid_dir 构建图）
    from common.data.dataset import FeatureDataset
    ds = FeatureDataset(str(train_dir), gene_norm="log1p_norm_total", ref_stats=stats)
    ds_v = FeatureDataset(str(valid_dir), gene_norm="log1p_norm_total", ref_stats=stats)
    loader = DataLoader(ds, batch_size=64, shuffle=True)
    loader_v = DataLoader(ds_v, batch_size=64, shuffle=False)

    model = build_model(num_genes=G, in_dim=D, hidden_dim=H, num_layers=2)
    hist = train_function(model, loader, loader_v, _Args(train_dir, valid_dir, tmp_path / "out"), stats)
    assert len(hist) == 2
    assert "PCC" in hist[0] and "AUROC" in hist[0]

    ckpt = torch.load(tmp_path / "out" / "best.pt", map_location="cpu")
    assert "model" in ckpt and "history" in ckpt and "config" in ckpt
    assert ckpt["config"]["method"] == "spatialex"

    m2 = build_model(num_genes=G, in_dim=D, hidden_dim=H, num_layers=2)
    m2.load_state_dict(ckpt["model"])
    res = evaluate_slide(m2, str(valid_dir), "log1p_norm_total", stats, "cpu",
                         str(tmp_path / "out"))
    assert "PCC" in res and "AUROC" in res
    assert (tmp_path / "out" / "test_results.json").exists()


def test_evaluate_slide_metrics(tmp_path):
    test_dir = tmp_path / "test"
    _make_fixture(test_dir, seed=3)
    stats = _fit_stats(test_dir, "log1p_norm_total")

    model = build_model(num_genes=G, in_dim=D, hidden_dim=H, num_layers=2)
    res = evaluate_slide(model, str(test_dir), "log1p_norm_total", stats, "cpu",
                         str(tmp_path / "res"))
    for key in ("PCC", "SPCC", "AUROC"):
        assert key in res
        assert math.isfinite(res[key]), f"{key} 应有限，got {res[key]}"
    assert (tmp_path / "res" / "test_results.json").exists()
