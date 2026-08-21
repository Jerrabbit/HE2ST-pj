"""统一 benchmark harness：所有方法共用同一训练/评估流程（课题要求 4、9）。

方法模型统一接口（各方法在 methods/<name>/model.py 中实现）：
    class SomeModel(nn.Module):
        input_type = 'patch' | 'feature'      # 决定数据加载
        def forward(self, x) -> Tensor        # x=(B,3,H,W) 或 (B,D)；输出 (B,G) 归一化表达预测

评估语义（与 HEST 通用 benchmark 一致）：
    - PCC / SPCC：逐基因（跨细胞）相关性，取均值
    - cell_PCC：逐细胞（跨基因，log1p 计数空间）相关性，取均值
    - Top-k：逐细胞，预测与真实 top-k 高表达基因的重合率，取均值（默认 k=10..100）
    - AUROC：逐基因，以 raw counts>0 为标签，取均值
PCC/SPCC/cell_PCC 在归一化空间（预测与真值同尺度，逐基因单调不变）；Top-k/AUROC
    按 gene_norm 将预测与真值逆变换回 raw counts 语义后计算。
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..eval.metrics import auroc, pcc, spcc, topk_accuracy  # noqa: F401（单基因实现供其它模块/测试复用）

EPS = 1e-8


def _extract_input(model: nn.Module, batch: dict) -> torch.Tensor:
    """按模型的 input_type 从 batch 中取输入。"""
    if getattr(model, "input_type", "patch") == "feature":
        return batch["feature"]
    return batch["patch"]


def predict(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    gene_norm: str = "log1p_zscore",
    stats: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """对 dataloader 全部样本做预测。

    返回：
        y_true_norm: (N, G) 归一化空间真值（用于 PCC/SPCC，与预测同空间）
        y_true_raw:  (N, G) 逆变换后的 raw counts 真值（用于 Top-k/AUROC 标签）
        y_pred:      (N, G) 归一化空间预测（用于 PCC/SPCC）
        y_pred_raw:  (N, G) 逆变换后的 raw counts 语义预测（用于 Top-k/AUROC）
    """
    model.to(device)  # 保证模型与输入同设备（test.py 从 CPU checkpoint 加载后直接评估也能跑）
    model.eval()
    true_list, pred_list = [], []
    with torch.no_grad():
        for batch in dataloader:
            x = _extract_input(model, batch).to(device)
            pred = model(x).cpu().numpy()
            true_list.append(batch["gene_expr"].cpu().numpy())
            pred_list.append(pred)
    y_true_norm = np.concatenate(true_list, axis=0)
    y_pred = np.concatenate(pred_list, axis=0)
    y_true_raw = _invert_normalization(y_true_norm, gene_norm, stats)
    y_pred_raw = _invert_normalization(y_pred, gene_norm, stats)
    return y_true_norm, y_true_raw, y_pred, y_pred_raw


def _invert_normalization(
    pred: np.ndarray, gene_norm: str, stats: dict | None
) -> np.ndarray:
    """把归一化空间的预测逆变换回 raw counts 语义（单调变换，供排序/分类指标用）。"""
    if gene_norm == "log1p_zscore" and stats is not None:
        means, stds = stats["means"], stats["stds"]
        log1p = pred * stds + means
    else:
        log1p = pred
    return np.expm1(np.clip(log1p, -30.0, 30.0)).astype(np.float32)


def compute_metrics_vectorized(
    y_true_norm: np.ndarray,
    y_pred: np.ndarray,
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    topk_ks: tuple | None = None,
    auroc_threshold: float = 0.0,
    details: bool = False,
) -> dict:
    """与逐基因/逐细胞循环完全等价的向量化指标计算（大切片评估大幅加速）。

    - PCC/SPCC：逐基因（跨细胞）。PCC 用归一化空间；SPCC 是秩相关，对单调变换
      不变，两空间等价。秩用 scipy rankdata(axis=0) 一次算全矩阵。
    - **cell_PCC（新增）**：逐细胞（跨基因）。每个细胞对全部基因的预测与真实表达
      计算 Pearson，再对所有细胞取均值。空间选择：**log1p 计数空间**（log1p(raw counts)，
      即自然"表达谱"空间，避免逐基因 z-score 缩放扭曲谱形）。常量细胞 → nan，nanmean 聚合。
    - Top-k：逐细胞（argpartition 取每行前 k，集合与 argsort[-k:] 相同）。
      默认 k = 10,20,...,100（课题要求计算 Top-k 随 k 变化的曲线）。
    - AUROC：逐基因，平均秩公式——与 sklearn roc_auc_score 数值一致（已验证
      max|diff|≈1e-16），比逐基因调用快约 6 倍。

    语义与 common/eval/metrics.py 的单基因函数完全一致（常量列 → nan，nanmean 聚合）。
    details=True 时额外返回逐基因数组 gene_pccs/gene_spccs/gene_aurocs（供 CSV 导出）。
    """
    from scipy.stats import rankdata

    if topk_ks is None:
        topk_ks = tuple(range(10, 101, 10))  # 10,20,...,100

    N, G = y_true_norm.shape
    with np.errstate(divide="ignore", invalid="ignore"):
        # PCC：逐基因 Pearson
        Xc = y_true_norm - y_true_norm.mean(0, keepdims=True)
        Yc = y_pred - y_pred.mean(0, keepdims=True)
        denom = np.sqrt((Xc**2).sum(0) * (Yc**2).sum(0))
        pccs = (Xc * Yc).sum(0) / denom

        # SPCC：对秩做 Pearson
        rx = rankdata(y_true_norm, axis=0)
        ry = rankdata(y_pred, axis=0)
        rxc = rx - rx.mean(0, keepdims=True)
        ryc = ry - ry.mean(0, keepdims=True)
        denom2 = np.sqrt((rxc**2).sum(0) * (ryc**2).sum(0))
        spccs = (rxc * ryc).sum(0) / denom2

        # AUROC：平均秩公式（== roc_auc_score，处理平局）
        R = rankdata(y_pred_raw, axis=0)
        pos = y_true_raw > auroc_threshold
        n_pos = pos.sum(0).astype(np.float64)
        n_neg = N - n_pos
        rank_sum = np.where(pos, R, 0.0).sum(0)
        aurocs = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        aurocs = np.where((n_pos == 0) | (n_neg == 0), np.nan, aurocs)

        # cell-level PCC：逐细胞（跨基因），log1p 计数空间
        xt = np.log1p(np.clip(y_true_raw, 0, None))
        xp = np.log1p(np.clip(y_pred_raw, 0, None))
        xc = xt - xt.mean(1, keepdims=True)
        yc = xp - xp.mean(1, keepdims=True)
        denom_c = np.sqrt((xc**2).sum(1) * (yc**2).sum(1))
        cell_pccs = (xc * yc).sum(1) / denom_c

        # Top-k 逐细胞（k≥G 时全集重合，acc=1.0，与原 topk_accuracy 语义一致）
        topk = {}
        for k in topk_ks:
            kk = min(int(k), G)
            if kk >= G:
                topk[f"top{k}"] = 1.0
                continue
            ti = np.argpartition(-y_true_raw, kk, axis=1)[:, :kk]
            pi = np.argpartition(-y_pred_raw, kk, axis=1)[:, :kk]
            th = np.zeros((N, G), dtype=bool)
            th[np.arange(N)[:, None], ti] = True
            ph = np.zeros((N, G), dtype=bool)
            ph[np.arange(N)[:, None], pi] = True
            topk[f"top{k}"] = float(((th & ph).sum(1) / kk).mean())

    result = {
        "PCC": float(np.nanmean(pccs)),
        "SPCC": float(np.nanmean(spccs)),
        "cell_PCC": float(np.nanmean(cell_pccs)),
        **topk,
        "AUROC": float(np.nanmean(aurocs)),
    }
    if details:
        result["gene_pccs"] = pccs
        result["gene_spccs"] = spccs
        result["gene_aurocs"] = aurocs
    return result


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    gene_norm: str = "log1p_zscore",
    stats: dict | None = None,
    topk_ks: tuple | None = None,
    auroc_threshold: float = 0.0,
    details: bool = False,
) -> dict:
    """完整评估，返回全部指标（课题要求 9）。

    stats 为表达归一化统计量（测试集使用训练集统计量时，与训练一致）。
    details=True 时额外返回逐基因数组 gene_pccs/gene_spccs/gene_aurocs（供 CSV 导出）。
    """
    y_true_norm, y_true_raw, y_pred, y_pred_raw = predict(
        model, dataloader, device, gene_norm, stats
    )
    return compute_metrics_vectorized(
        y_true_norm, y_pred, y_true_raw, y_pred_raw, topk_ks, auroc_threshold,
        details=details,
    )


def load_gene_names(data_dir: str) -> list[str] | None:
    """从数据目录读取 gene_names.txt（公共基因列表）。"""
    p = os.path.join(data_dir, "gene_names.txt")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return [ln.strip() for ln in f if ln.strip()]


def scalar_results(results: dict) -> dict:
    """从评估结果里提取标量摘要（丢弃逐基因 numpy 数组，供 JSON 序列化）。"""
    return {k: v for k, v in results.items()
            if isinstance(v, (int, float, np.floating, np.integer))}


def save_eval_results_csv(
    csv_path: str,
    results: dict,
    gene_names: list[str] | None = None,
    topk_ks: tuple | None = None,
) -> dict:
    """把评估结果保存为 CSV（结果需来自 evaluate(details=True)）。

    输出两个文件：
        {csv_path}              摘要：PCC / SPCC / cell_PCC / AUROC / top10..top100
        {csv_path}_genes.csv    逐基因：gene, PCC, SPCC, AUROC
    """
    import csv

    if topk_ks is None:
        topk_ks = tuple(range(10, 101, 10))
    summary_keys = ["PCC", "SPCC", "cell_PCC", "AUROC"]
    summary_keys += [f"top{k}" for k in topk_ks]

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k in summary_keys:
            if k in results:
                w.writerow([k, float(results[k])])

    gene_path = f"{csv_path}_genes.csv"
    if gene_names is not None and "gene_pccs" in results:
        with open(gene_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["gene", "PCC", "SPCC", "AUROC"])
            for i, g in enumerate(gene_names):
                w.writerow([
                    g,
                    float(results["gene_pccs"][i]),
                    float(results["gene_spccs"][i]),
                    float(results["gene_aurocs"][i]),
                ])
    return {"summary": csv_path, "genes": gene_path}


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader | None,
    epochs: int,
    lr: float,
    device: str = "cuda",
    out_dir: str = "outputs",
    weight_decay: float = 0.0,
    loss_fn: nn.Module | None = None,
    gene_norm: str = "log1p_zscore",
    eval_stats: dict | None = None,
    verbose: bool = True,
    config: dict | None = None,
    patience: int = 10,
) -> dict:
    """通用回归训练（MSE 目标，Adam 优化器）。

    用于输出连续表达预测的方法（UNI2+MLP、ST-Net、Path2Space、DeepPT、GHIST、
    SpatialEx 等）。损失作用于归一化空间的预测。
    特殊训练目标的方法（BLEEP 对比、Hist2ST 负二项、STFlow 流匹配）在各自文件夹内
    实现自己的训练循环，复用本模块的 evaluate()。

    返回：
        history: {epoch: {loss, val_PCC, ...}}
    """
    if loss_fn is None:
        loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    os.makedirs(out_dir, exist_ok=True)
    best_pcc, best_state = -float("inf"), None
    no_improve = 0
    history = []
    model = model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total, n = 0.0, 0
        for batch in train_loader:
            x = _extract_input(model, batch).to(device)
            y = batch["gene_expr"].to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            total += loss.item() * y.size(0)
            n += y.size(0)
        train_loss = total / max(n, 1)

        rec = {"epoch": epoch, "train_loss": train_loss}
        if valid_loader is not None:
            ev = evaluate(model, valid_loader, device, gene_norm, eval_stats)
            rec.update(ev)
            pcc = ev["PCC"]
            if pcc == pcc and pcc > best_pcc + 1e-4:  # 非 nan 且更优
                best_pcc = pcc
                best_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                no_improve = 0
            else:
                no_improve += 1
        history.append(rec)
        if verbose:
            print(
                f"[epoch {epoch}/{epochs}] loss={train_loss:.4f} "
                + (f"val_PCC={rec.get('PCC', float('nan')):.4f}" if valid_loader else ""),
                flush=True,
            )
        if valid_loader is not None and no_improve >= patience:
            print(f"[early stop] val_PCC {patience} 个 epoch 未提升，在 epoch {epoch} 停止", flush=True)
            break

    if best_state is not None:
        ckpt = os.path.join(out_dir, "best.pt")
        torch.save({"model": best_state, "history": history, "config": config}, ckpt)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return history
