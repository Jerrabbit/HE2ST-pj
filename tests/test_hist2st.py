"""Hist2ST 冒烟测试：核心前向、calcADJ 语义、完整 train_function 流程、evaluate_slide 指标（CPU）。

用 fig_size=56（dim=256）做 CPU 测试提速（官方默认 fig_size=112 → dim=1024）；
仅测试期缩小，默认 build_model 仍是官方架构。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 项目根目录

from common.data.expression import load_expression, normalize_expression
from common.data.slide_tiling import knn_adjacency
import methods.hist2st as hist2st
from methods.hist2st import build_model

G = 8


def _make_fixture(root: Path, n: int = 60, g: int = G, seed: int = 0, region: float = 500.0):
    """构造最小 patch 数据集：~n 个细胞铺在 region×region 区域（保证 tiling ≥1 ROI）。

    patch 为 128×128 随机 RGB（模型内部 resize 到 fig_size）；cell_id 用纯数字
    以匹配约定 patches/cell_{id}.png。
    """
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    expr = rng.poisson(2.0, size=(n, g)).astype(np.float32)
    np.save(root / "gene_expression.npy", expr)
    (root / "gene_names.txt").write_text("\n".join(f"gene{i}" for i in range(g)))
    (root / "patches").mkdir()
    xs = rng.uniform(0, region, size=n)
    ys = rng.uniform(0, region, size=n)
    rows = []
    for i in range(n):
        Image.fromarray(rng.integers(0, 255, (128, 128, 3), dtype=np.uint8)).save(
            root / "patches" / f"cell_{i}.png")
        rows.append({"cell_id": str(i), "x_centroid": float(xs[i]), "y_centroid": float(ys[i]),
                     "patch_path": str(root / "patches" / f"cell_{i}.png")})
    pd.DataFrame(rows).to_csv(root / "metadata.csv", index=False)


class _Args:
    def __init__(self, output_dir, train_dir, valid_dir, gene_norm="log1p_norm_total"):
        self.device = "cpu"
        self.output_dir = str(output_dir)
        self.train_dir = str(train_dir)
        self.valid_dir = str(valid_dir)
        self.lr = 1e-3
        self.weight_decay = 0.0
        self.epochs = 2
        self.gene_norm = gene_norm


def test_hist2st_forward():
    """Hist2STModel.core 前向：(1,N,3,112,112)→(1,N,G)；图路径可用。"""
    model = build_model(num_genes=G, fig_size=56)  # dim = (56//7)^2 * 32//8 = 256
    assert model.input_type == "patch"
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 100, (6, 2))
    patches = torch.rand(1, 6, 3, 56, 56)
    centers = torch.from_numpy(hist2st._normalize_centers(coords, model.core.n_pos))
    centers = centers.unsqueeze(0)  # (1,6,2) long
    adj = torch.from_numpy(knn_adjacency(coords, k=4).toarray()).float()  # (6,6)
    pred, extra, h = model.core(patches, centers, adj)
    # 官方 forward：patches reshape B*N，pred 保持 (N,G)（无 batch 维）
    assert pred.shape == (6, G)
    assert h.shape == (6, 256)
    assert extra is None
    assert torch.isfinite(pred).all()
    # 包装 forward 应提示走 predict_slide
    with pytest.raises(NotImplementedError):
        model(torch.rand(1, 3, 56, 56))


def test_calcADJ_semantics():
    """kNN 邻接与官方 calcADJ prune='NA' 语义一致：每节点恰 k 个二元邻居、无自环。"""
    rng = np.random.default_rng(1)
    coords = rng.uniform(0, 100, (20, 2))
    A = knn_adjacency(coords, k=4).toarray()
    assert A.shape == (20, 20)
    assert np.all(A.sum(axis=1) == 4)          # 恰 k 个邻居
    assert np.all(np.diag(A) == 0)             # 无自环
    assert set(np.unique(A)) <= {0.0, 1.0}     # 二值
    for i in range(20):
        dists = np.linalg.norm(coords - coords[i], axis=1)
        expected = set(np.argsort(dists)[1:5])          # 最近 4 个非自身
        assert set(np.flatnonzero(A[i])) == expected
    assert (A == A.T).mean() > 0.5             # kNN 通常近似对称


def test_train_function(tmp_path):
    """完整 train_function：ROI 图训练 2 epoch → best.pt → reload → evaluate_slide。"""
    _make_fixture(tmp_path / "train", n=60)
    _make_fixture(tmp_path / "test", n=40, seed=1)
    expr_raw, _ = load_expression(str(tmp_path / "train"))
    _expr_norm, stats = normalize_expression(expr_raw, "log1p_norm_total")

    model = build_model(num_genes=G, fig_size=56)
    args = _Args(tmp_path / "out", tmp_path / "train", tmp_path / "test")
    hist = hist2st.train_function(model, None, None, args, stats)
    assert len(hist) == 2

    ckpt = torch.load(tmp_path / "out" / "best.pt", map_location="cpu")
    assert "model" in ckpt and "history" in ckpt
    assert ckpt["config"]["method"] == "hist2st"

    m2 = build_model(num_genes=G, fig_size=56)
    m2.load_state_dict(ckpt["model"])
    res = hist2st.evaluate_slide(m2, str(tmp_path / "test"), "log1p_norm_total",
                                 stats, "cpu", str(tmp_path / "res"))
    assert "PCC" in res and "AUROC" in res
    assert (tmp_path / "res" / "test_results.json").exists()


def test_evaluate_slide_metrics(tmp_path):
    """evaluate_slide 在 tiny 测试集上返回有限 PCC/SPCC/AUROC。"""
    _make_fixture(tmp_path / "test", n=40, g=6, seed=2)
    expr_raw, _ = load_expression(str(tmp_path / "test"))
    _expr_norm, stats = normalize_expression(expr_raw, "log1p_norm_total")
    model = build_model(num_genes=6, fig_size=56)
    res = hist2st.evaluate_slide(model, str(tmp_path / "test"), "log1p_norm_total",
                                 stats, "cpu", output_dir=None)
    for key in ("PCC", "SPCC", "top10", "top50", "top100", "AUROC"):
        assert key in res
    assert np.isfinite(res["PCC"])
    assert np.isfinite(res["SPCC"])
    assert np.isfinite(res["AUROC"])
