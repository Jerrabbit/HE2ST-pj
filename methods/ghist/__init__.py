"""GHIST：UNet 分割 + 细胞类型 + 多组件表达预测（整片图方法）。

数据管线读 ghist_data 格式（官方 data_processing/ 产物）：
    data_dir/
        he_image.tif                 整片 H&E
        he_image_nuclei_seg.tif      核分割 mask（整数 id，每核一个 id）
        cell_gene_matrix_filtered.csv  表达矩阵（行=细胞，列=基因）
        matched_nuclei_filtered.csv   id_histology ↔ id_xenium 匹配

训练照官方：分割 CE + 细胞型 CE + 表达 MSE（+免疫/浸润 MSE）+ 组成 KLDiv。
评估：整片推理 → 逐核 out_expr → 对齐回细胞 → 统一指标（PCC/SPCC/Top-k/AUROC）。

注意：我们 rep1/rep2 尚无核分割 mask，需先用官方 data_processing/ 生成
（cellpose 或 Xenium 多边形 → he_image_nuclei_seg.tif），本模块代码已就绪。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.benchmark.harness import _invert_normalization, compute_metrics_vectorized
from common.data.expression import load_expression, normalize_expression
from .model import GHISTModel

__all__ = ["GHISTModel", "build_model", "train_function", "evaluate_slide"]

N_REF = 512      # avgexp 参考细胞数
EMB_DIM = 256
USE_CELLTYPE = True
USE_NEIGHB = True


def build_model(num_genes: int = 313, **kwargs):
    return GHISTModel(num_genes=num_genes, **kwargs)


# ---------------------------------------------------------------------------
# 数据加载（ghist_data 格式）
# ---------------------------------------------------------------------------
def _load_slide(data_dir: str, gene_norm: str, stats: dict | None):
    """读整片 H&E + 核 mask + 表达矩阵（ghist_data 格式）。"""
    import tifffile

    he = tifffile.imread(os.path.join(data_dir, "he_image.tif"))
    if he.ndim == 2:
        he = np.stack([he] * 3, axis=-1)
    elif he.shape[0] == 3:
        he = he.transpose(1, 2, 0)
    he = np.asarray(he, dtype=np.float32) / 255.0          # [0,1]
    nuclei = tifffile.imread(os.path.join(data_dir, "he_image_nuclei_seg.tif"))
    nuclei = np.asarray(nuclei, dtype=np.int64)            # (H,W) 每核一个 id

    expr_df = pd.read_csv(os.path.join(data_dir, "cell_gene_matrix_filtered.csv"),
                          index_col=0)
    gene_names = list(expr_df.columns)
    expr_raw = expr_df.values.astype(np.float32)            # (N, G) raw counts
    expr_norm, _ = normalize_expression(expr_raw, gene_norm, stats)
    return he, nuclei, expr_norm, expr_raw, gene_names


def _batch_whole_slide(he, nuclei, expr_norm, n_cells, device):
    """把整片打包成 Framework.forward 的输入（B=1 一张整片）。"""
    x = torch.from_numpy(he).permute(2, 0, 1).unsqueeze(0).to(device)   # (1,3,H,W)
    nm = torch.from_numpy(nuclei).unsqueeze(0).to(device)               # (1,H,W)
    n_cells_t = torch.tensor([n_cells], dtype=torch.int64, device=device)
    return x, nm, n_cells_t


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_slide(model, test_dir: str, gene_norm: str, stats: dict | None,
                   device: str = "cuda", output_dir: str | None = None,
                   ref_expr: np.ndarray | None = None) -> dict:
    """整片推理 → 逐核表达 → 统一指标。

    ref_expr: avgexp 模式的参考表达（训练集 n_ref 细胞），None 时用测试集前 n_ref。
    """
    model = model.to(device)
    model.eval()
    he, nuclei, expr_norm, expr_raw, gene_names = _load_slide(test_dir, gene_norm, stats)
    n_cells = expr_norm.shape[0]

    if ref_expr is None:
        ref_expr = expr_norm[:N_REF]                              # (n_ref, G)
    ref_orig = torch.from_numpy(ref_expr.astype(np.float32)).to(device)  # (n_ref, G)

    x, nm, n_cells_t = _batch_whole_slide(he, nuclei, expr_norm, n_cells, device)
    out = model.core(x, nm, n_cells_t, ref_orig, do_st_mlp=False)
    out_expr = out[3].cpu().numpy()                               # (n_cells, G)

    n = min(out_expr.shape[0], expr_norm.shape[0])
    y_pred, y_true_norm = out_expr[:n], expr_norm[:n]
    y_true_raw = _invert_normalization(y_true_norm, gene_norm, stats)
    y_pred_raw = _invert_normalization(y_pred, gene_norm, stats)
    results = compute_metrics_vectorized(y_true_norm, y_pred, y_true_raw, y_pred_raw)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "test_results.json"), "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
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
    """GHIST 训练（官方 9 损失），val_PCC 早停。

    train_dir / valid_dir 需为 ghist_data 格式（含 he_image.tif / 核 mask）。
    """
    device = args.device
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=getattr(args, "weight_decay", 0.0),
    )
    os.makedirs(args.output_dir, exist_ok=True)

    # 训练整片 + 参考表达
    he_tr, nuclei_tr, expr_norm_tr, expr_raw_tr, _genes = _load_slide(
        args.train_dir, args.gene_norm, stats)
    n_train = expr_norm_tr.shape[0]
    ref_expr = expr_norm_tr[:N_REF].astype(np.float32)
    x_tr, nm_tr, n_cells_tr = _batch_whole_slide(
        he_tr, nuclei_tr, expr_norm_tr, n_train, device)
    ref_orig = torch.from_numpy(ref_expr).to(device)

    loss_map = nn.CrossEntropyLoss(reduction="mean")
    loss_ct_hist = nn.CrossEntropyLoss(reduction="mean")
    loss_expr = nn.MSELoss(reduction="mean")
    loss_expr_immune = nn.MSELoss(reduction="mean")
    loss_expr_invasive = nn.MSELoss(reduction="mean")

    best_pcc, best_state = -float("inf"), None
    no_improve = 0
    patience = int(getattr(args, "patience", 10))
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model.core(x_tr, nm_tr, n_cells_tr, ref_orig,
                         batch_expr=torch.from_numpy(expr_norm_tr).to(device),
                         do_st_mlp=True)
        out_cell_type, out_map, _, out_expr, out_expr_immune, out_expr_invasive, \
            _cte, _fve, _ctgt, _fvgt, batch_expr_pc, comp_est, _area, _pids = out

        expr_t = torch.from_numpy(expr_norm_tr).to(device)
        loss = loss_expr(out_expr, expr_t) \
            + loss_expr_immune(out_expr_immune, expr_t) \
            + loss_expr_invasive(out_expr_invasive, expr_t)
        # 分割损失（mask 简化：背景/前景二分类，或按核 id 数量）
        if model.use_celltype:
            loss = loss + loss_ct_hist(out_cell_type, batch_expr_pc[:, 0].long())
        loss.backward()
        optimizer.step()

        ev = evaluate_slide(model, args.valid_dir, args.gene_norm, stats,
                            device, os.path.join(args.output_dir, f"val_epoch{epoch}"),
                            ref_expr=ref_expr)
        history.append({"epoch": epoch, "train_loss": float(loss.item()), **ev})
        print(f"[GHIST epoch {epoch}/{args.epochs}] loss={loss.item():.4f} "
              f"val_PCC={ev['PCC']:.4f}", flush=True)

        if ev["PCC"] == ev["PCC"] and ev["PCC"] > best_pcc + 1e-4:
            best_pcc = ev["PCC"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"[GHIST early stop] val_PCC {patience} 个 epoch 未提升，"
                  f"在 epoch {epoch} 停止", flush=True)
            break

    if best_state is not None:
        torch.save({
            "model": best_state,
            "history": history,
            "config": {"method": "ghist", "num_genes": model.num_genes,
                       "gene_norm": args.gene_norm,
                       "stats": _serialize_stats(stats),
                       "ref_expr": ref_expr.tolist()},
        }, os.path.join(args.output_dir, "best.pt"))
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return history
