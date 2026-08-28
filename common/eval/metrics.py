"""统一评估指标，所有方法共用，保证公平比较。

课题要求 9 规定最终评估指标必须包括三类：
- 回归指标：PCC、SPCC
- 排序指标：Top-k 准确率（预测最高表达基因与真实的重合率）
- 分类指标：AUROC（衡量模型预测基因"是否表达"的能力）
"""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score


def _check_inputs(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """统一校验输入，返回一维 float64 数组。"""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("输入为空数组")
    if y_true.size != y_pred.size:
        raise ValueError(f"长度不一致：y_true={y_true.size}, y_pred={y_pred.size}")
    return y_true, y_pred


def pcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson 相关系数（回归指标）。常量数组时返回 nan。"""
    y_true, y_pred = _check_inputs(y_true, y_pred)
    if np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0]):
        return float("nan")
    return float(pearsonr(y_true, y_pred).statistic)


def spcc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman 秩相关系数（回归指标）。常量数组时返回 nan。"""
    y_true, y_pred = _check_inputs(y_true, y_pred)
    if np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0]):
        return float("nan")
    return float(spearmanr(y_true, y_pred).statistic)


def topk_accuracy(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """Top-k 准确率：预测表达最高的 k 个基因与真实表达最高的 k 个基因的重合率。

    输入为单个样本（细胞/spot）的逐基因表达向量。
    """
    y_true, y_pred = _check_inputs(y_true, y_pred)
    if k <= 0:
        raise ValueError("k 必须为正整数")
    k = min(k, y_true.size)
    true_topk = set(np.argsort(y_true)[-k:])
    pred_topk = set(np.argsort(y_pred)[-k:])
    return len(true_topk & pred_topk) / k


def auroc(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.0) -> float:
    """AUROC（分类指标）：衡量模型区分基因"是否表达"的能力。

    将 y_true 按 threshold 二值化（>threshold 视为表达），
    以 y_pred 作为连续打分计算 ROC-AUC。
    """
    y_true, y_pred = _check_inputs(y_true, y_pred)
    labels = (y_true > threshold).astype(int)
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float(roc_auc_score(labels, y_pred))


def ssim_2d(x: np.ndarray, y: np.ndarray, data_range: float | None = None) -> float:
    """二维图像 SSIM（Wang et al. 2004 标准公式，Gaussian 窗，纯 numpy/scipy 实现）。

    与 skimage.metrics.structural_similarity 的 gaussian_weights=True 默认一致：
        SSIM = mean( (2μxμy+C1)(2σxy+C2) / ((μx²+μy²+C1)(σx²+σy²+C2)) )
        C1=(K1·L)², C2=(K2·L)², K1=0.01, K2=0.03, L=data_range
    局部统计量用 σ=1.5 的高斯核加权（等价 win_size=7）。

    用于 HE→ST 的**空间表达图**相似度：把细胞表达栅格化成空间图后逐基因计算，
    衡量"预测的空间表达结构"与"真实空间表达结构"的一致程度（非逐像素 PCC）。
    """
    from scipy.ndimage import gaussian_filter

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError(f"SSIM 输入须为同形状 2D 数组: {x.shape} vs {y.shape}")
    if data_range is None:
        data_range = float(max(x.max(), y.max()) - min(x.min(), y.min()))
    if not np.isfinite(data_range) or data_range <= 0:
        return float("nan")
    K1, K2 = 0.01, 0.03
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    sigma = 1.5
    mu_x = gaussian_filter(x, sigma=sigma)
    mu_y = gaussian_filter(y, sigma=sigma)
    mu_x_sq, mu_y_sq = mu_x * mu_x, mu_y * mu_y
    sigma_x_sq = gaussian_filter(x * x, sigma=sigma) - mu_x_sq
    sigma_y_sq = gaussian_filter(y * y, sigma=sigma) - mu_y_sq
    sigma_xy = gaussian_filter(x * y, sigma=sigma) - mu_x * mu_y
    num = (2.0 * mu_x * mu_y + C1) * (2.0 * sigma_xy + C2)
    den = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    with np.errstate(divide="ignore", invalid="ignore"):
        ssim_map = num / den
    return float(np.nanmean(ssim_map))
