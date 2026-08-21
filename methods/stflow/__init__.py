"""STFlow：整片切片流匹配生成模型（ROI 级适配）。

数据管线：metadata → 坐标/UNI2 特征/归一化表达 → tile_rois 切 ROI →
每 ROI 作 SpatialTransformer 的一批（[1,N,G] + 特征 + 坐标）训练流匹配；
评估走 evaluate_slide（逐 ROI 采样 → 首 ROI 优先对齐回 per-cell → 统一指标）。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch

from common.benchmark.harness import (
    _invert_normalization,
    compute_metrics_vectorized,
    load_gene_names,
    scalar_results,
)
from common.data.expression import load_expression, normalize_expression
from common.data.slide_tiling import tile_rois
from .model import STFlow

__all__ = ["STFlow", "build_model", "train_function", "evaluate_slide"]

ROI_TARGET_CELLS = 256
ROI_MIN_CELLS = 32


def build_model(num_genes: int = 313, **kwargs):
    return STFlow(num_genes=num_genes, **kwargs)


# ---------------------------------------------------------------------------
# 数据管线
# ---------------------------------------------------------------------------
def _load_slide(data_dir: str, gene_norm: str, stats: dict | None):
    meta = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
    coords = meta[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)
    features = np.load(os.path.join(data_dir, "X_uni2.npy")).astype(np.float32)
    expr_raw, _ = load_expression(data_dir)
    expr_norm, _ = normalize_expression(expr_raw, gene_norm, stats)
    return coords, features, expr_norm


def _tile(coords: np.ndarray, target_cells: int = ROI_TARGET_CELLS,
          min_cells: int = ROI_MIN_CELLS) -> list[np.ndarray]:
    """按细胞密度自动调 ROI 尺寸，使每 ROI 约 target_cells 个细胞。"""
    n = len(coords)
    if n < min_cells:
        return [np.arange(n)]
    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    ext_x = max(xmax - xmin, 1e-6)
    ext_y = max(ymax - ymin, 1e-6)
    density = n / (ext_x * ext_y)
    roi_size = float(np.sqrt(target_cells / max(density, 1e-12)))
    roi_size = min(roi_size, max(ext_x, ext_y))
    roi_size = max(roi_size, 1.0)
    stride = roi_size * 0.5
    return tile_rois(coords, roi_size, stride, min_cells=min_cells)


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_slide(model, test_dir: str, gene_norm: str, stats: dict | None,
                   device: str = "cuda", output_dir: str | None = None,
                   details: bool = False) -> dict:
    """整测试切片逐 ROI 采样生成 → 首 ROI 优先对齐回 per-cell → 统一指标。

    details=True 时额外返回逐基因数组 gene_pccs/gene_spccs/gene_aurocs 与
    _gene_names（供 CSV 导出）。
    """
    model = model.to(device)
    model.eval()
    coords, features, expr_norm = _load_slide(test_dir, gene_norm, stats)
    N, G = expr_norm.shape
    rois = _tile(coords)

    y_pred = np.full((N, G), np.nan, dtype=np.float32)
    covered = np.zeros(N, dtype=bool)
    for roi in rois:
        if len(roi) < 2:
            continue
        f = torch.from_numpy(features[roi]).unsqueeze(0).to(device)
        c = torch.from_numpy(coords[roi]).unsqueeze(0).to(device)
        pred = model.sample_roi(f, c, device).squeeze(0).cpu().numpy()  # (N_roi, G)
        new = ~covered[roi]
        y_pred[roi[new]] = pred[new]
        covered[roi[new]] = True

    keep = covered
    n_dropped = int((~keep).sum())
    if n_dropped:
        print(f"[STFlow] 警告: {n_dropped} 个细胞未覆盖，已从指标剔除", flush=True)
    if int(keep.sum()) == 0:
        nan = {"PCC": float("nan"), "SPCC": float("nan"),
               "top10": float("nan"), "top50": float("nan"),
               "top100": float("nan"), "AUROC": float("nan")}
        results = nan
    else:
        y_true_raw = _invert_normalization(expr_norm[keep], gene_norm, stats)
        y_pred_raw = _invert_normalization(y_pred[keep], gene_norm, stats)
        results = compute_metrics_vectorized(expr_norm[keep], y_pred[keep],
                                             y_true_raw, y_pred_raw,
                                             details=details)
    if details:
        results["_gene_names"] = load_gene_names(test_dir)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "test_results.json"), "w") as f:
            json.dump(scalar_results(results), f, ensure_ascii=False, indent=2)
    return results


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def _serialize_stats(stats: dict | None) -> dict | None:
    if stats is None:
        return None
    out = {}
    for k, v in stats.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, np.generic):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def train_function(model, train_loader, valid_loader, args, stats) -> dict:
    """STFlow 训练：逐 ROI 流匹配损失，每 epoch 验证切片评估，val_PCC 早停。"""
    device = args.device
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=getattr(args, "weight_decay", 0.0),
    )
    os.makedirs(args.output_dir, exist_ok=True)

    coords, features, expr_norm = _load_slide(args.train_dir, args.gene_norm, stats)
    rng = np.random.default_rng(0)

    best_pcc, best_state = -float("inf"), None
    no_improve = 0
    patience = int(getattr(args, "patience", 10))
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        rois = _tile(coords)
        rng.shuffle(rois)
        total, n = 0.0, 0
        for roi in rois:
            if len(roi) < 2:
                continue
            g = torch.from_numpy(expr_norm[roi]).unsqueeze(0).to(device)
            f = torch.from_numpy(features[roi]).unsqueeze(0).to(device)
            c = torch.from_numpy(coords[roi]).unsqueeze(0).to(device)
            optimizer.zero_grad()
            loss, _det = model.flow_loss(g, f, c)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(roi)
            n += len(roi)
        train_loss = total / max(n, 1)

        rec = {"epoch": epoch, "train_loss": train_loss}
        if valid_loader is not None or True:  # 用 evaluate_slide 评估验证切片
            ev = evaluate_slide(model, args.valid_dir, args.gene_norm, stats,
                                device, os.path.join(args.output_dir, f"val_epoch{epoch}"))
            rec.update(ev)
            if ev["PCC"] == ev["PCC"] and ev["PCC"] > best_pcc + 1e-4:
                best_pcc = ev["PCC"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
        history.append(rec)
        print(f"[STFlow epoch {epoch}/{args.epochs}] loss={train_loss:.4f} "
              f"val_PCC={rec.get('PCC', float('nan')):.4f}", flush=True)
        if no_improve >= patience:
            print(f"[STFlow early stop] val_PCC {patience} 个 epoch 未提升，"
                  f"在 epoch {epoch} 停止", flush=True)
            break

    if best_state is not None:
        torch.save({
            "model": best_state,
            "history": history,
            "config": {"method": "stflow", "num_genes": model.num_genes,
                       "gene_norm": args.gene_norm,
                       "prior": getattr(model, "prior", "gaussian"),
                       "n_sample_steps": getattr(model, "n_sample_steps", 20),
                       "stats": _serialize_stats(stats)},
        }, os.path.join(args.output_dir, "best.pt"))
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return history
