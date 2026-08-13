"""HESTDataset / FeatureDataset / 表达归一化 的单元测试。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from common.data.dataset import FeatureDataset, HESTDataset
from common.data.expression import (
    load_expression,
    normalize_expression,
    subset_genes,
)


@pytest.fixture
def data_dir(tmp_path):
    """构造最小数据集目录：raw counts 表达 + 元数据 + patch + 特征。"""
    n, g = 50, 6
    rng = np.random.default_rng(0)
    expr = rng.poisson(2.0, size=(n, g)).astype(np.float32)
    np.save(tmp_path / "gene_expression.npy", expr)
    (tmp_path / "gene_names.txt").write_text("\n".join(f"gene{i}" for i in range(g)))

    rows = []
    (tmp_path / "patches").mkdir()
    for i in range(n):
        (tmp_path / "patches" / f"cell_{i}.png").write_bytes(b"")
        Image.fromarray((rng.integers(0, 255, (32, 32, 3), dtype=np.uint8))).save(
            tmp_path / "patches" / f"cell_{i}.png"
        )
        rows.append({"cell_id": f"c{i}", "x_centroid": i, "y_centroid": i,
                     "patch_path": str(tmp_path / "patches" / f"cell_{i}.png")})
    pd.DataFrame(rows).to_csv(tmp_path / "metadata.csv", index=False)

    np.save(tmp_path / "X_uni2.npy", rng.normal(size=(n, 8)).astype(np.float32))
    return tmp_path


def test_load_expression(data_dir):
    expr, names = load_expression(str(data_dir))
    assert expr.shape == (50, 6)
    assert names == [f"gene{i}" for i in range(6)]
    assert expr.dtype == np.float32


def test_normalize_zscore_roundtrip():
    rng = np.random.default_rng(0)
    x = rng.poisson(2.0, size=(100, 4)).astype(np.float32)
    z, stats = normalize_expression(x, "log1p_zscore")
    assert z.shape == x.shape
    assert abs(z.mean()) < 0.2 and abs(z.std() - 1.0) < 0.2
    # 逆变换近似还原 raw counts
    back = np.expm1(z * stats["stds"] + stats["means"])
    assert np.allclose(back, x, atol=1e-1)


def test_ref_stats_no_leakage():
    rng = np.random.default_rng(1)
    train = rng.poisson(3.0, size=(80, 4)).astype(np.float32)
    test = rng.poisson(1.0, size=(20, 4)).astype(np.float32)  # 不同分布
    _, stats = normalize_expression(train, "log1p_zscore")
    z_test, _ = normalize_expression(test, "log1p_zscore", stats)
    # 用训练统计量，测试集应大致保持方差 1
    assert abs(np.log1p(test).std() / stats["stds"].ravel().mean()) < 2.0


def test_hest_dataset(data_dir):
    ds = HESTDataset(str(data_dir), gene_norm="none")
    item = ds[0]
    assert item["patch"].shape == (3, 32, 32)
    assert item["gene_expr"].shape == (6,)
    assert ds.stats == {}


def test_hest_dataset_gene_subset(data_dir):
    ds = HESTDataset(str(data_dir), gene_list=["gene0", "gene2"], gene_norm="none")
    assert ds.gene_list == ["gene0", "gene2"]
    assert ds[0]["gene_expr"].shape == (2,)


def test_feature_dataset(data_dir):
    ds = FeatureDataset(str(data_dir), gene_norm="none")
    item = ds[0]
    assert item["feature"].shape == (8,)
    assert item["gene_expr"].shape == (6,)
    assert len(ds) == 50


def test_hest_dataset_zscore(data_dir):
    ds = HESTDataset(str(data_dir), gene_norm="log1p_zscore")
    assert "means" in ds.stats and "stds" in ds.stats
    assert abs(ds.expr_all.mean()) < 0.2
    assert abs(ds.expr_all.std() - 1.0) < 0.2
