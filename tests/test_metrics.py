"""评估指标单元测试。

运行方式（在 HE2ST-pj 根目录）：
    conda activate myenv
    python -m pytest tests/test_metrics.py -v
"""
import numpy as np
import pytest

from common.eval.metrics import auroc, pcc, spcc, topk_accuracy


def test_pcc_perfect_correlation():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert pcc(x, x) == pytest.approx(1.0)


def test_pcc_negative_correlation():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert pcc(x, -x) == pytest.approx(-1.0)


def test_pcc_constant_returns_nan():
    x = np.array([1.0, 1.0, 1.0])
    assert np.isnan(pcc(x, x))


def test_spcc_monotonic():
    x = np.array([1.0, 3.0, 8.0, 2.0, 5.0])
    assert spcc(x, x) == pytest.approx(1.0)


def test_topk_accuracy_exact_match():
    y_true = np.array([0.1, 0.9, 0.3, 0.7, 0.5])
    y_pred = np.array([0.2, 1.0, 0.4, 0.8, 0.6])  # 排名顺序与 true 一致
    assert topk_accuracy(y_true, y_pred, k=3) == pytest.approx(1.0)


def test_topk_accuracy_partial_match():
    y_true = np.array([0.9, 0.1, 0.1, 0.1, 0.1])  # top-1 是 index0
    y_pred = np.array([0.1, 0.9, 0.1, 0.1, 0.1])  # top-1 是 index1，完全未重合
    assert topk_accuracy(y_true, y_pred, k=1) == pytest.approx(0.0)


def test_auroc_perfect_separation():
    y_true = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    y_pred = np.array([0.0, 0.1, 0.8, 0.9, 0.95])
    assert auroc(y_true, y_pred) == pytest.approx(1.0)


def test_auroc_worse_than_random_penalized():
    y_true = np.array([0.0, 0.0, 0.0, 1.0, 1.0])
    y_pred = np.array([1.0, 0.9, 0.8, 0.1, 0.0])  # 完全反向
    assert auroc(y_true, y_pred) == pytest.approx(0.0)


def test_input_mismatch_raises():
    with pytest.raises(ValueError):
        pcc(np.array([1.0, 2.0]), np.array([1.0]))
