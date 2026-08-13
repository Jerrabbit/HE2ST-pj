"""Path2Space 冒烟测试：MLP 集成加载 + 基因映射 + expm1 转 raw + 冻结评估路径。

用小参数伪造官方集成（n_ik=2 × n_il=2、n_genes_all=5）验证：
    1. MLPEnsemble 从 result_{ik}_{il}_0/model_trained.pth 加载并正确集成
    2. _resolve_out_indices 把公共基因映射到官方输出位置
    3. Path2SpaceModel.forward 输出 raw counts 语义（log1p → expm1）
    4. evaluate_frozen 走 FeatureDataset + harness evaluate 全链路
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 项目根目录

import methods.path2space as p2s
from methods.path2space.model import (
    MLPEnsemble,
    MLP_regression_relu_two,
    Path2SpaceModel,
    _resolve_out_indices,
)


def _make_ensemble_dir(root: Path, n_ik=2, n_il=2, n_in=8, n_hid=8, n_out=5,
                       seed=0):
    """生成带已知权重的小型官方集成（result_{ik}_{il}_0/model_trained.pth）。"""
    rng = np.random.default_rng(seed)
    for ik in range(n_ik):
        for il in range(n_il):
            model = MLP_regression_relu_two(n_in, n_hid, n_out)
            with torch.no_grad():
                for p in model.parameters():
                    p.copy_(torch.as_tensor(rng.normal(0, 0.1, p.shape), dtype=torch.float32))
            ckpt_dir = root / f"result_{ik}_{il}_0"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_dir / "model_trained.pth")
    return root


def _make_fixture(root: Path, n: int = 20, g: int = 5, seed: int = 0):
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    expr = rng.poisson(2.0, size=(n, g)).astype(np.float32)
    np.save(root / "gene_expression.npy", expr)
    (root / "gene_names.txt").write_text("\n".join(f"gene{i}" for i in range(g)))
    feats = rng.normal(0, 1, (n, 8)).astype(np.float32)
    np.save(root / "X_ctranspath.npy", feats)
    rows = []
    for i in range(n):
        rows.append({"cell_id": f"c{i}", "x_centroid": i, "y_centroid": i,
                     "patch_path": f"patches/cell_{i}.png"})
    pd.DataFrame(rows).to_csv(root / "metadata.csv", index=False)


def test_mlp_ensemble_load_and_predict(tmp_path):
    ens_dir = _make_ensemble_dir(tmp_path / "ens")
    ens = MLPEnsemble(ens_dir, n_inputs=8, n_genes=5, n_ik_folds=2,
                      n_il_folds=2, device="cpu")
    assert len(ens.models) == 4
    x = np.random.default_rng(1).normal(0, 1, (6, 8))
    pred = ens.predict(x)
    assert pred.shape == (6, 5)
    assert np.all(np.isfinite(pred))


def test_mlp_ensemble_matches_direct_mean(tmp_path):
    ens_dir = _make_ensemble_dir(tmp_path / "ens", seed=7)
    ens = MLPEnsemble(ens_dir, n_inputs=8, n_genes=5, n_ik_folds=2,
                      n_il_folds=2, device="cpu")
    x = np.random.default_rng(3).normal(0, 1, (4, 8)).astype(np.float32)
    pred = ens.predict(x)
    # 手工复算：先 il 均值再 ik 均值（官方集成顺序）
    direct = np.zeros((4, 5), dtype=np.float64)
    xt = torch.as_tensor(x)
    for ik in range(2):
        il_sum = np.zeros((4, 5), dtype=np.float64)
        for il in range(2):
            il_sum += ens.models[(ik, il)](xt).detach().cpu().numpy()
        direct += il_sum / 2
    direct /= 2
    assert np.allclose(pred, direct, atol=1e-5)


def test_resolve_out_indices(tmp_path):
    genes_txt = tmp_path / "genes.txt"
    genes_txt.write_text("\n".join(["a", "b", "c", "d", "e"]))
    gene_names = ["e", "a", "c"]
    idx = _resolve_out_indices(str(genes_txt), gene_names, n_genes_all=5)
    assert idx.tolist() == [4, 0, 2]
    # 缺失基因应报错
    with pytest.raises(ValueError):
        _resolve_out_indices(str(genes_txt), ["a", "zzz"], 5)


def test_path2space_forward_raw_counts(tmp_path):
    ens_dir = _make_ensemble_dir(tmp_path / "ens", seed=4)
    genes_txt = tmp_path / "genes.txt"
    genes_txt.write_text("\n".join(["gene0", "gene1", "gene2", "gene3", "gene4"]))
    model = Path2SpaceModel(num_genes=3, ensemble_dir=str(ens_dir),
                            genes_txt=str(genes_txt),
                            gene_names=["gene4", "gene0", "gene2"],
                            n_inputs=8, n_genes_all=5, n_ik_folds=2,
                            n_il_folds=2, device="cpu")
    assert model.input_type == "feature"
    assert model.feature_file == "X_ctranspath.npy"
    assert model.out_indices.tolist() == [4, 0, 2]
    x = torch.randn(5, 8)
    out = model(x)
    assert out.shape == (5, 3)
    # 输出应为 raw counts 语义（≥0，且由 expm1 产生）
    assert torch.all(out >= 0)
    assert torch.all(out < 1e6)  # 无溢出


def test_evaluate_frozen(tmp_path):
    _make_fixture(tmp_path / "test")
    ens_dir = _make_ensemble_dir(tmp_path / "ens", seed=11)
    model = Path2SpaceModel(num_genes=5, ensemble_dir=str(ens_dir),
                            genes_txt=None, gene_names=[f"gene{i}" for i in range(5)],
                            n_inputs=8, n_genes_all=5, n_ik_folds=2,
                            n_il_folds=2, device="cpu")
    res = p2s.evaluate_frozen(model, str(tmp_path / "test"),
                              gene_names=[f"gene{i}" for i in range(5)],
                              batch_size=8, device="cpu",
                              output_dir=str(tmp_path / "out"))
    assert "PCC" in res and "AUROC" in res
    assert (tmp_path / "out" / "test_results.json").exists()
