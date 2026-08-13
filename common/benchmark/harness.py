"""统一 benchmark harness：所有方法共用同一训练/评估流程（课题要求 4、9）。

方法模型统一接口（各方法在 methods/<name>/model.py 中实现）：
    class SomeModel(nn.Module):
        input_type = 'patch' | 'feature'      # 决定数据加载
        def forward(self, x) -> Tensor        # x=(B,3,H,W) 或 (B,D)；输出 (B,G) 归一化表达预测

评估语义（与 HEST 通用 benchmark 一致）：
    - PCC / SPCC：逐基因（跨细胞）相关性，取均值
    - Top-k：逐细胞，预测与真实 top-k 高表达基因的重合率，取均值
    - AUROC：逐基因，以 raw counts>0 为标签，取均值
PCC/SPCC 在归一化空间（预测与真值同尺度，逐基因单调不变）；Top-k/AUROC
    按 gene_norm 将预测与真值逆变换回 raw counts 语义后计算。
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..eval.metrics import auroc, pcc, spcc, topk_accuracy

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


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda",
    gene_norm: str = "log1p_zscore",
    stats: dict | None = None,
    topk_ks: tuple = (10, 50, 100),
    auroc_threshold: float = 0.0,
) -> dict:
    """完整评估，返回全部指标（课题要求 9）。

    stats 为表达归一化统计量（测试集使用训练集统计量时，与训练一致）。
    """
    y_true_norm, y_true_raw, y_pred, y_pred_raw = predict(
        model, dataloader, device, gene_norm, stats
    )
    G = y_true_raw.shape[1]

    # 逐基因 PCC / SPCC（跨细胞）。PCC 用归一化空间（同一尺度、无偏）；
    # SPCC 是秩相关，对单调变换不变，两空间等价。
    pccs, spccs = [], []
    for g in range(G):
        t, p = y_true_norm[:, g], y_pred[:, g]
        pccs.append(pcc(t, p))
        spccs.append(spcc(t, p))
    pcc_mean = float(np.nanmean(pccs)) if pccs else float("nan")
    spcc_mean = float(np.nanmean(spccs)) if spccs else float("nan")

    # 逐细胞 Top-k
    topk = {}
    for k in topk_ks:
        vals = [topk_accuracy(t, p, k) for t, p in zip(y_true_raw, y_pred_raw)]
        topk[f"top{k}"] = float(np.mean(vals))

    # 逐基因 AUROC（raw counts>threshold 为表达标签）
    aurocs = []
    for g in range(G):
        aurocs.append(auroc(y_true_raw[:, g], y_pred_raw[:, g], auroc_threshold))
    auroc_mean = float(np.nanmean(aurocs)) if aurocs else float("nan")

    return {
        "PCC": pcc_mean,
        "SPCC": spcc_mean,
        **topk,
        "AUROC": auroc_mean,
    }


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
            if pcc == pcc and pcc > best_pcc:  # 非 nan 且更优
                best_pcc = pcc
                best_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
        history.append(rec)
        if verbose:
            print(
                f"[epoch {epoch}/{epochs}] loss={train_loss:.4f} "
                + (f"val_PCC={rec.get('PCC', float('nan')):.4f}" if valid_loader else ""),
                flush=True,
            )

    if best_state is not None:
        ckpt = os.path.join(out_dir, "best.pt")
        torch.save({"model": best_state, "history": history, "config": config}, ckpt)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return history
