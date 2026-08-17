"""Hist2ST：Transformer + 图神经网络(GNN) 混合架构，从 H&E 组织学图像预测空间基因表达。

官方代码：D:\\hest_data\\codes\\Hist2ST\\（HIST2ST.py / graph_construction.py / transformer.py / gcn.py / NB_module.py）
架构见 methods/hist2st/model.py。

本模块提供：
    build_model(num_genes=313, **kwargs) → Hist2STModel
    train_function(model, train_loader, valid_loader, args, stats) → history
    evaluate_slide(model, test_dir, gene_norm, stats, device, output_dir) → dict

Hist2ST 为**全切片图方法**：在 ROI（物理坐标网格切片）内对细胞做自注意力 + GNN。
    - 训练：整张切片按 ROI 切片（tile_rois 自动调参，每 ROI ~512 细胞），
      每个 ROI 作为一个 batch（该 ROI 内细胞共享局部 kNN 邻接图）。
    - 评估：整测试切片切片 ROI → 逐 ROI 预测 → 按"首 ROI 优先"对齐回原细胞顺序。

数据管线（自建，harness 传入的 train_loader/valid_loader 仅用于接口一致）：
    metadata.csv → 坐标 (N,2)；patches/cell_{id}.png → (N,3,fig_size,fig_size) [0,1] 像素
    gene_expression.npy（raw counts）→ normalize_expression（默认 gene_norm='log1p_norm_total'）

适配（相对官方，仅数据/接口层，无架构改动）：
    - calcADJ → 纯 kNN（k=4，cKDTree），prune='NA' 语义
      （官方 prune='Grid' 距离阈值 ≤2.0 为 Visium 点阵设计，不适用于 Xenium µm 坐标）。
    - 位置嵌入：ROI 局部坐标归一化到 [0, n_pos) 取整（官方数据本就在该范围）。
    - patch 输入为 [0,1] float、不做 ImageNet 归一化（官方加载原始像素 im 后 /255）。
    - ZINB / bake 默认关（benchmark 统一在归一化空间评估，语义干净）。

收敛结论（2026-08-14，相邻切片基准）：
    官方设计 350 epoch + lr 1e-5 + ZINB + 自蒸馏；统一协议（50 epoch, lr 1e-3, 纯 MSE）
    下 6 epoch 后 val_PCC≈0（loss 卡在归一化表达方差）。超参探针（lr 3e-3/1e-2 ×
    ZINB 0/0.25，各 5 epoch）全部无法收敛：纯 MSE val_PCC≈0，ZINB 最高峰值 0.027 且
    过拟合回落。根因：该架构从原始 patch 从头训练（不用预训练特征），学组织特征太慢。
    如实记为 null result。探针日志：logs/hist2st_probe/。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from common.benchmark.harness import _invert_normalization, compute_metrics_vectorized
from common.data.expression import load_expression, normalize_expression
from common.data.slide_tiling import knn_adjacency, tile_rois
from .model import Hist2STModel, NB_loss, ZINB_loss

__all__ = ["Hist2STModel", "build_model", "train_function", "evaluate_slide"]

# 邻接 kNN 邻居数（官方默认 --neighbor 4）
KNN_K = 4
# ROI 目标细胞数（自注意力在 N≈256-1024 内可行；GPU 上 N≤1024 没问题）
ROI_TARGET_CELLS = 512
ROI_MIN_CELLS = 32


def build_model(num_genes: int = 313, **kwargs):
    return Hist2STModel(num_genes=num_genes, **kwargs)


# --------------------------------------------------------------------------- #
# 数据管线
# --------------------------------------------------------------------------- #
def _resolve_patch_path(data_dir: str, row) -> str:
    """metadata 的 patch_path 存在则用，否则按约定 data_dir/patches/cell_{id}.png。"""
    pp = row.get("patch_path")
    if isinstance(pp, str) and pp:
        if os.path.exists(pp):
            return pp
        cand = os.path.join(data_dir, pp)
        if os.path.exists(cand):
            return cand
    return os.path.join(data_dir, "patches", f"cell_{row['cell_id']}.png")


def _load_slide(data_dir: str, gene_norm: str, ref_stats: dict | None, fig_size: int):
    """读取一张切片的坐标 / patch / 表达。

    返回：
        coords:    (N, 2) float32 物理坐标
        patches:   (N, 3, fig_size, fig_size) float32，[0,1] 像素（/255，不做 ImageNet 归一化）
        expr_norm: (N, G) float32 归一化表达（ref_stats 复用训练集统计量）
        expr_raw:  (N, G) float32 raw counts（供 ZINB 损失使用）
    """
    meta = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
    coords = meta[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)
    cell_ids = meta["cell_id"].astype(str).tolist()
    expr_raw, _gene_names = load_expression(data_dir)
    expr_norm, _stats = normalize_expression(expr_raw, gene_norm, ref_stats)
    patches = np.zeros((len(cell_ids), 3, fig_size, fig_size), dtype=np.float32)
    for i, cid in enumerate(cell_ids):
        path = _resolve_patch_path(data_dir, meta.iloc[i])
        img = Image.open(path).convert("RGB")
        if img.size != (fig_size, fig_size):
            img = img.resize((fig_size, fig_size), Image.BILINEAR)
        patches[i] = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return coords, patches, expr_norm, expr_raw


def _tile_slide(coords, target_cells: int = ROI_TARGET_CELLS,
                stride_frac: float = 0.5, min_cells: int = ROI_MIN_CELLS):
    """自动调 ROI 尺寸使每 ROI 约 target_cells 个细胞（启发式，文档见模块 docstring）。

    roi_size 依据局部密度估计：roi_size = sqrt(target_cells / density)，并封顶为切片边长，
    保证稀疏数据退化为整切片 ROI（`tile_rois` 自身也有 <min_cells 全图回退）。
    """
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
    stride = roi_size * stride_frac
    return tile_rois(coords, roi_size, stride, min_cells=min_cells)


def _normalize_centers(coords, n_pos: int = 64) -> np.ndarray:
    """ROI 局部坐标归一化到 [0, n_pos) 并取整（官方位置嵌入要求 int 索引）。

    官方数据本就在 Embedding(n_pos) 兼容范围；对 Xenium µm 坐标做此必要适配。
    """
    mn = coords.min(axis=0)
    mx = coords.max(axis=0)
    span = mx - mn + 1e-6
    norm = (coords - mn) / span * (n_pos - 1)
    return np.clip(np.round(norm).astype(np.int64), 0, n_pos - 1)


def _build_adj(coords, k: int = KNN_K) -> torch.Tensor:
    """kNN 邻接（二进制 (N,N) 稠密 torch float），calcADJ prune='NA' 语义。

    knn_adjacency 用 cKDTree O(N log N)；官方 calcADJ 是 O(N^2) 稠密距离矩阵，
    Xenium 大 ROI 不可行。语义一致：A[i][最近 k 个 j]=1，自环排除。
    """
    A = knn_adjacency(coords, k=k, self_loop=False).toarray().astype(np.float32)
    return torch.from_numpy(A)


# --------------------------------------------------------------------------- #
# 预测 / 评估
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _predict_roi(model, sub_patches, sub_coords, device) -> np.ndarray:
    """对一个 ROI（子 patches + 局部邻接）做整组前向，返回 (N,G) 预测。"""
    patches = torch.from_numpy(sub_patches).unsqueeze(0).to(device)          # (1,N,3,H,W)
    centers = torch.from_numpy(_normalize_centers(sub_coords, model.core.n_pos))
    centers = centers.unsqueeze(0).to(device)                                # (1,N,2) long
    adj = _build_adj(sub_coords).to(device)                                  # (N,N) float
    pred, _extra, _h = model.core(patches, centers, adj)
    return pred.cpu().numpy().astype(np.float32)                             # (N,G)


def _compute_metrics(y_true_norm: np.ndarray, y_pred: np.ndarray,
                     gene_norm: str, stats: dict | None) -> dict:
    """与 common/benchmark/harness.evaluate() 完全一致的指标语义（向量化）。

    PCC/SPCC 在归一化空间逐基因；Top-k/AUROC 经 _invert_normalization 逆变换到
    raw counts 语义后计算（与 harness 相同）。
    """
    y_true_raw = _invert_normalization(y_true_norm, gene_norm, stats)
    y_pred_raw = _invert_normalization(y_pred, gene_norm, stats)
    return compute_metrics_vectorized(y_true_norm, y_pred, y_true_raw, y_pred_raw)


def _predict_slide_arrays(model, coords, patches, expr_norm, gene_norm: str,
                          stats: dict | None, device: str = "cuda",
                          output_dir: str | None = None) -> dict:
    """对**已加载的切片数组**做整切片 ROI 图推理评估（evaluate_slide 的核心）。

    覆盖对齐策略：**首 ROI 优先**（first-ROI-wins）——重叠 ROI 中先预测到该细胞的
    ROI 负责其预测，保证确定性；稀疏边缘未覆盖细胞用 kNN 邻居补一个 ROI。
    结果与 evaluate_slide 逐字节一致，仅省去重复的 patch 磁盘加载
    （训练期验证切片每 epoch 复用同一份数组，避免每 epoch 重读 10 万+ 张 PNG）。
    """
    model = model.to(device)
    model.eval()
    N, G = expr_norm.shape
    rois = _tile_slide(coords)
    y_pred = np.full((N, G), np.nan, dtype=np.float32)
    covered = np.zeros(N, dtype=bool)
    for roi in rois:
        if len(roi) < 2:
            continue
        pred = _predict_roi(model, patches[roi], coords[roi], device)
        new = ~covered[roi]
        y_pred[roi[new]] = pred[new]
        covered[roi[new]] = True
    # 稀疏边缘补充：未覆盖细胞 ∪ 各自 kNN 邻居
    missing = np.flatnonzero(~covered)
    if len(missing) > 0:
        A = knn_adjacency(coords, k=KNN_K, self_loop=True).toarray()
        union = np.flatnonzero(A[missing].sum(axis=0) > 0)
        union = np.union1d(union, missing)
        if len(union) >= 2:
            pred = _predict_roi(model, patches[union], coords[union], device)
            fill = ~covered[union]
            y_pred[union[fill]] = pred[fill]
            covered[union[fill]] = True
    keep = covered
    n_dropped = int((~keep).sum())
    if n_dropped > 0:
        print(f"[Hist2ST] 警告: {n_dropped} 个细胞未覆盖，已从指标剔除", flush=True)
    if int(keep.sum()) == 0:
        print("[Hist2ST] 警告: 没有细胞被覆盖，无法计算指标（返回 NaN）", flush=True)
        nan = {"PCC": float("nan"), "SPCC": float("nan"),
               "top10": float("nan"), "top50": float("nan"),
               "top100": float("nan"), "AUROC": float("nan")}
        results = nan
    else:
        results = _compute_metrics(expr_norm[keep], y_pred[keep], gene_norm, stats)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "test_results.json"), "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    return results


def evaluate_slide(model, test_dir: str, gene_norm: str, stats: dict | None,
                   device: str = "cuda", output_dir: str | None = None) -> dict:
    """整切片 ROI 图推理评估，返回与 harness evaluate() 相同的指标 dict。

    = _load_slide + _predict_slide_arrays（后者可在训练期用缓存数组复用）。
    """
    coords, patches, expr_norm, _expr_raw = _load_slide(test_dir, gene_norm, stats, model.fig_size)
    return _predict_slide_arrays(model, coords, patches, expr_norm, gene_norm,
                                 stats, device, output_dir)


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
def _zinb_loss_from_roi(expr_raw_roi: np.ndarray, extra, stats: dict | None, device: str):
    """按 raw counts + size factors（lib/median_lib）计算 ZINB 损失（默认关闭）。"""
    mean, disp, pi = extra
    lib = expr_raw_roi.sum(axis=1, keepdims=True)
    median_lib = stats.get("median_lib") if stats else None
    if median_lib is None:
        median_lib = float(np.median(lib))
    sfs = torch.from_numpy((lib / max(float(median_lib), 1e-8)).astype(np.float32)).to(device)
    x = torch.from_numpy(expr_raw_roi).to(device)
    return ZINB_loss(x, mean, disp, pi, scale_factor=sfs)


def _serialize_stats(stats: dict | None) -> dict | None:
    """把统计量转成可 torch.load(weights_only=True) 的纯 Python 类型。"""
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
    """Hist2ST 自定义训练：ROI 图 batch（等价官方 training_step），MSE + 可选 ZINB。

    harness 传入的 train_loader/valid_loader 仅用于接口一致；数据管线由本函数从
    args.train_dir / args.valid_dir 自建（ROI 切片 + 局部邻接）。每 epoch 用
    evaluate_slide 在验证切片上评估，按 val_PCC 保存 best.pt。
    """
    device = args.device
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=getattr(args, "weight_decay", 0.0),
    )
    os.makedirs(args.output_dir, exist_ok=True)

    train_dir = getattr(args, "train_dir")
    valid_dir = getattr(args, "valid_dir", None)
    gene_norm = args.gene_norm
    zinb_coef = float(getattr(args, "zinb_coef", 0.0))

    coords, patches, expr_norm, expr_raw = _load_slide(train_dir, gene_norm, stats, model.fig_size)
    # 验证切片只加载一次（patches 是 ~17GB 数组），跨 epoch 复用避免每 epoch 重读磁盘
    valid_slide = None
    if valid_dir is not None:
        valid_slide = _load_slide(valid_dir, gene_norm, stats, model.fig_size)
        print(f"[Hist2ST] 验证切片已预加载: {len(valid_slide[0])} cells "
              f"({valid_slide[1].nbytes/1e9:.1f} GB)", flush=True)
    rng = np.random.default_rng(0)

    best_pcc, best_state = -float("inf"), None
    no_improve = 0
    patience = int(getattr(args, "patience", 10))
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        rois = _tile_slide(coords)
        rng.shuffle(rois)
        total, n = 0.0, 0
        for roi in rois:
            if len(roi) < 2:
                continue
            sub_patches = torch.from_numpy(patches[roi]).unsqueeze(0).to(device)   # (1,N,3,H,W)
            sub_centers = torch.from_numpy(_normalize_centers(coords[roi], model.core.n_pos))
            sub_centers = sub_centers.unsqueeze(0).to(device)                      # (1,N,2) long
            sub_adj = _build_adj(coords[roi]).to(device)                           # (N,N) float
            sub_expr = torch.from_numpy(expr_norm[roi]).to(device)                 # (N,G)
            optimizer.zero_grad()
            pred, extra, _h = model.core(sub_patches, sub_centers, sub_adj)
            loss = F.mse_loss(pred, sub_expr)   # pred 官方为 (N,G)（reshape B*N 后未恢复 batch 维）
            if model.core.zinb > 0 and zinb_coef > 0:
                zinb_loss = _zinb_loss_from_roi(expr_raw[roi], extra, stats, device)
                loss = loss + model.core.zinb * zinb_loss
            if model.core.bake > 0:
                # 官方 bake 自蒸馏：增强视图聚合预测 → 原预测（HIST2ST.py:182-186）
                bake_x = model.core.aug(sub_patches, sub_centers, sub_adj)
                new_pred = model.core.distillation(bake_x)
                bake_loss = F.mse_loss(new_pred, pred)
                loss = loss + model.core.lamb * bake_loss
            loss.backward()
            optimizer.step()
            total += loss.item() * sub_expr.size(0)
            n += sub_expr.size(0)
        train_loss = total / max(n, 1)

        rec = {"epoch": epoch, "train_loss": train_loss}
        if valid_dir is not None:
            # 复用预加载的验证切片数组（coords, patches, expr_norm；expr_raw 不参与评估）
            ev = _predict_slide_arrays(model, *valid_slide[:3], gene_norm, stats, device,
                                       output_dir=os.path.join(args.output_dir, f"val_epoch{epoch}"))
            rec.update(ev)
            if ev["PCC"] == ev["PCC"] and ev["PCC"] > best_pcc + 1e-4:
                best_pcc = ev["PCC"]
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
        history.append(rec)
        print(f"[Hist2ST epoch {epoch}/{args.epochs}] loss={train_loss:.4f} "
              + (f"val_PCC={rec.get('PCC', float('nan')):.4f}" if valid_dir else ""), flush=True)
        if valid_dir is not None and no_improve >= patience:
            print(f"[Hist2ST early stop] val_PCC {patience} 个 epoch 未提升，"
                  f"在 epoch {epoch} 停止", flush=True)
            break

    if best_state is not None:
        torch.save({
            "model": best_state,
            "history": history,
            "config": {"method": "hist2st", "num_genes": model.num_genes,
                       "fig_size": model.fig_size, "n_pos": model.n_pos,
                       "gene_norm": gene_norm, "stats": _serialize_stats(stats),
                       "zinb": float(getattr(model.core, "zinb", 0)),
                       "bake": int(getattr(model.core, "bake", 0)),
                       "lamb": float(getattr(model.core, "lamb", 0))},
        }, os.path.join(args.output_dir, "best.pt"))
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    return history
