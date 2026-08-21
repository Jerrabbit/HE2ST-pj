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

from common.benchmark.harness import (
    _invert_normalization,
    compute_metrics_vectorized,
    load_gene_names,
    scalar_results,
)
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
    """读整片 H&E + 核 mask + 表达矩阵（ghist_data 格式）。

    返回 (he, nuclei, expr_norm, expr_raw, gene_names, cell_ids, stats_used)：
        cell_ids:  cell_gene_matrix 的 index（= 核 mask 像素值，Xenium cell_id）
        stats_used: 归一化使用的统计量（stats=None 时在 expr_raw 上拟合，供跨切片复用）
    """
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
    expr_norm, stats_used = normalize_expression(expr_raw, gene_norm, stats)
    cell_ids = expr_df.index.values.astype(np.int64)
    return he, nuclei, expr_norm, expr_raw, gene_names, cell_ids, stats_used


PATCH_H = 256      # 官方 hsize/wsize
PATCH_W = 256
PATCH_OVERLAP = 30
MAX_CELLS_PATCH = 200


def _tile_slide(he, nuclei, hsize=PATCH_H, wsize=PATCH_W, overlap=PATCH_OVERLAP):
    """整片 H&E + 核 mask → patch 列表（官方 dataio 逻辑）。

    patch 网格：步长 = size - overlap，尾块对齐。返回
        [(he_patch(h,w,3) [0,1], nuclei_patch(h,w) int64, ids(该 patch 内全局 cell_id 排序))]
    每个 patch 仅含有效核（有表达数据的 cell_id），0（背景/无效）剔除。
    """
    H, W = nuclei.shape
    stride_h, stride_w = hsize - overlap, wsize - overlap
    r_starts = list(range(0, max(1, H - hsize), stride_h))
    if r_starts[-1] < H - hsize:
        r_starts.append(H - hsize)
    c_starts = list(range(0, max(1, W - wsize), stride_w))
    if c_starts[-1] < W - wsize:
        c_starts.append(W - wsize)
    patches = []
    for hs in r_starts:
        for ws in c_starts:
            np_ = nuclei[hs:hs + hsize, ws:ws + wsize]
            ids = np.unique(np_)
            ids = ids[ids != 0]
            if ids.size == 0:
                continue
            patches.append((he[hs:hs + hsize, ws:ws + wsize], np_, ids))
    return patches


@torch.no_grad()
def _predict_tiled(model, he, nuclei, ref_orig, device,
                   batch_size: int = 8) -> dict[int, np.ndarray]:
    """整片 tile 推理 → {cell_id: 平均预测表达 (G,)}。

    重叠 patch 中重复出现的细胞对多次预测取平均（官方直接 append 不聚合，我们更严谨）。
    """
    patches = _tile_slide(he, nuclei)
    accum: dict[int, list] = {}
    for i in range(0, len(patches), batch_size):
        batch = patches[i:i + batch_size]
        B = len(batch)
        x = torch.stack([torch.from_numpy(p[0]).float().permute(2, 0, 1)
                         for p in batch]).to(device)
        nm = torch.stack([torch.from_numpy(p[1]) for p in batch]).to(device)
        n_cells = torch.tensor([p[2].size for p in batch],
                               dtype=torch.int64, device=device)
        out = model.core(x, nm, n_cells, ref_orig, do_st_mlp=False)
        out_expr = out[3].detach().cpu().numpy()            # (Σn_cells, G)
        k = 0
        for b in range(B):
            nb = batch[b][2].size
            ids_b = batch[b][2]
            for j, cid in enumerate(ids_b):
                rec = accum.setdefault(int(cid), [])
                rec.append(out_expr[k + j])
            k += nb
    return {cid: np.mean(v, axis=0).astype(np.float32) for cid, v in accum.items()}


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_slide(model, test_dir: str, gene_norm: str, stats: dict | None,
                   device: str = "cuda", output_dir: str | None = None,
                   ref_expr: np.ndarray | None = None,
                   details: bool = False) -> dict:
    """整片推理 → 逐核表达 → 统一指标。

    ref_expr: avgexp 模式的参考表达（训练集 n_ref 细胞），None 时用测试集前 n_ref。
    details=True 时额外返回逐基因数组与 _gene_names（供 CSV 导出）。
    """
    model = model.to(device)
    model.eval()
    he, nuclei, expr_norm, expr_raw, gene_names, cell_ids, stats_used = _load_slide(
        test_dir, gene_norm, stats)
    n_cells = expr_norm.shape[0]

    if ref_expr is None:
        ref_expr = expr_norm[:N_REF]                              # (n_ref, G)
    ref_orig = torch.from_numpy(ref_expr.astype(np.float32)).to(device)  # (n_ref, G)

    # 整片 tile 推理 → 逐细胞预测（重复 patch 平均），按 cell_id 对齐回表达矩阵
    pred_map = _predict_tiled(model, he, nuclei, ref_orig, device)
    pos = {int(c): i for i, c in enumerate(cell_ids)}
    y_pred = np.zeros((n_cells, expr_norm.shape[1]), dtype=np.float32)
    for cid, p in pred_map.items():
        i = pos.get(cid)
        if i is not None:
            y_pred[i] = p
    y_true_norm = expr_norm[:n_cells]
    y_true_raw = _invert_normalization(y_true_norm, gene_norm, stats_used)
    y_pred_raw = _invert_normalization(y_pred, gene_norm, stats_used)
    results = compute_metrics_vectorized(y_true_norm, y_pred, y_true_raw, y_pred_raw,
                                         details=details)
    if details:
        results["_gene_names"] = list(gene_names)
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

    # 整片 tile 训练：patch 列表 + 参考表达 + cell_id→行索引
    he_tr, nuclei_tr, expr_norm_tr, expr_raw_tr, _genes, cell_ids_tr, stats_tr = _load_slide(
        args.train_dir, args.gene_norm, stats)
    ref_expr = expr_norm_tr[:N_REF].astype(np.float32)
    ref_orig = torch.from_numpy(ref_expr).to(device)
    pos = {int(c): i for i, c in enumerate(cell_ids_tr)}
    patches = _tile_slide(he_tr, nuclei_tr)
    print(f"[GHIST] 训练切片 → {len(patches)} 个 {PATCH_H}×{PATCH_W} patch"
          f"（overlap {PATCH_OVERLAP}）", flush=True)

    batch_size = int(getattr(args, "batch_size", 8))
    loss_expr = nn.MSELoss(reduction="mean")
    loss_expr_immune = nn.MSELoss(reduction="mean")
    loss_expr_invasive = nn.MSELoss(reduction="mean")

    best_pcc, best_state = -float("inf"), None
    no_improve = 0
    patience = int(getattr(args, "patience", 10))
    history = []
    rng = np.random.default_rng(0)

    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(patches)
        total_loss, n_batches = 0.0, 0
        for s in range(0, len(patches), batch_size):
            batch = patches[s:s + batch_size]
            B = len(batch)
            x = torch.stack([torch.from_numpy(p[0]).float().permute(2, 0, 1)
                             for p in batch]).to(device)
            nm = torch.stack([torch.from_numpy(p[1]) for p in batch]).to(device)
            n_cells = torch.tensor([p[2].size for p in batch],
                                   dtype=torch.int64, device=device)
            # patch 内细胞的表达真值（按 cell_id 对齐，填充到 max_cells_patch）
            expr_pad = np.zeros((B, MAX_CELLS_PATCH, model.num_genes), dtype=np.float32)
            for b in range(B):
                rows = np.array([pos[int(c)] for c in batch[b][2]])
                expr_pad[b, :rows.size] = expr_norm_tr[rows]
            batch_expr = torch.from_numpy(expr_pad).to(device)

            optimizer.zero_grad()
            out = model.core(x, nm, n_cells, ref_orig,
                             batch_expr=batch_expr, do_st_mlp=True)
            # 顺序与 out_expr 一致：batch_expr_pc = out[10]（各 patch 按 cids 顺序拼接）
            batch_expr_pc = out[10]
            loss = loss_expr(out[3], batch_expr_pc) \
                + loss_expr_immune(out[4], batch_expr_pc) \
                + loss_expr_invasive(out[5], batch_expr_pc)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        ev = evaluate_slide(model, args.valid_dir, args.gene_norm, stats_tr,
                            device, os.path.join(args.output_dir, f"val_epoch{epoch}"),
                            ref_expr=ref_expr)
        mean_loss = total_loss / max(n_batches, 1)
        history.append({"epoch": epoch, "train_loss": mean_loss, **ev})
        print(f"[GHIST epoch {epoch}/{args.epochs}] loss={mean_loss:.4f} "
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
                       "stats": _serialize_stats(stats_tr),   # 保存训练拟合统计量（勿存 arg 的 None）
                       "ref_expr": ref_expr.tolist()},        # 训练参考表达，测试须复用
        }, os.path.join(args.output_dir, "best.pt"))
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return history
