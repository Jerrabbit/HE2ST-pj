"""SpatialEx：高参数空间多组学，基于组织学锚定的整合（超图卷积，官方复刻）。

官方代码：D:\\hest_data\\codes\\SpatialEx\\SpatialEx\\（model.py / preprocess.py / utils.py / SpatialEx.py）
架构（Predictor_spot，见 model.py）：
    mlp → HGNN(num_layers=2, prelu) → linear；整片超图卷积，cell 预测依赖邻居。

本仓库统一 benchmark 适配（保证公平，与其他方法共享预处理/评估语义）：
    1. 特征输入：data_dir/X_uni2.npy（UNI2 特征，与其他特征型方法一致）。
    2. 超图：复用 common/data/slide_tiling.py —— k=7 自环 kNN 超图 H（Build_hypergraph，
       与官方 Build_hypergraph_spatial_and_HE 的 num_neighbors=7 语义一致），
       hpnn 归一化（normalize_hypergraph_hpnn，与官方 normalize_graph hpnn 一致）。
    3. Xenium 规模：全切片图 O(N^2) 不可行，切成重叠 ROI（约 1024–2048 细胞）。
       ROI 内从全切片超图子选择 H[roi][:, roi] 再 hpnn 归一化 —— 与官方
       Xenium_HBRC_overlap 完全一致（官方也是先建全图，再对每个 ROI 子选择归一化）。
    4. 损失目标：默认 cell 级 MSE（统一归一化空间，与其他方法可比）；use_spot_agg=True
       时复刻官方 Generate_pseudo_spot（visium 交错网格，x_interval=100，
       y_interval=100*sqrt(3)）构建聚合矩阵，对伪 spot 聚合表达算 MSE。
    5. 评估：evaluate_slide 走整片超图推理（ROI 重叠处取多次预测均值），指标与
       harness.evaluate 完全一致（PCC/SPCC 归一化空间逐基因，Top-k/AUROC 逆变换回
       raw counts 语义）。

ROI 自调参启发式（_autotune_roi_size）：
    以细胞密度估算边长，使每 ROI 约 target=1536（1024–2048 区间中点）个细胞；
    限制每 ROI 不超过 max_cells=4096（超边矩阵规模），且至少含 min_cells=32 个细胞
    （BatchNorm 需要 batch>1）。stride ≈ 0.5*roi_size（50% 重叠）。由此 ROI 数
    ≈ 4N/target，对 16 万+细胞切片也是线性可扩展的。
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import sparse
from scipy.spatial import cKDTree

from common.benchmark.harness import _invert_normalization, compute_metrics_vectorized
from common.data.expression import load_expression, normalize_expression
from common.data.slide_tiling import (
    build_hypergraph,
    normalize_hypergraph_hpnn,
    sparse_to_torch,
    tile_rois,
)
from .model import SpatialExModel

__all__ = ["SpatialExModel", "build_model", "train_function", "evaluate_slide"]

K_NEIGHBORS = 7  # 官方 num_neighbors=7
CELL_PER_ROI_MIN = 32
CELL_PER_ROI_LOW = 1024
CELL_PER_ROI_HIGH = 2048
CELL_PER_ROI_MAX = 4096


def build_model(num_genes: int = 313, **kwargs):
    """构造 SpatialEx 模型。kwargs 透传 SpatialExModel（in_dim/hidden_dim/num_layers 等）。"""
    return SpatialExModel(num_genes=num_genes, **kwargs)


# ---------------------------------------------------------------------------
# 数据与图构建
# ---------------------------------------------------------------------------
def _load_slide(
    data_dir: str,
    gene_norm: str,
    stats: dict | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """加载切片：坐标 (N,2)、UNI2 特征 (N,D)、归一化表达 (N,G)。

    stats 为训练集拟合的归一化统计量（None 表示在测试集上自拟合）。
    """
    meta = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
    coords = meta[["x_centroid", "y_centroid"]].values.astype(np.float32)
    features = np.load(os.path.join(data_dir, "X_uni2.npy")).astype(np.float32)
    if len(coords) != len(features):
        raise ValueError(
            f"metadata 与 X_uni2.npy 行数不一致: {len(coords)} vs {len(features)}"
        )
    expr_raw, _ = load_expression(data_dir)
    expr_norm, _ = normalize_expression(expr_raw, gene_norm, stats)
    return coords, features, expr_norm


def _autotune_roi_size(
    coords: np.ndarray,
    min_cells: int = CELL_PER_ROI_MIN,
    target_low: int = CELL_PER_ROI_LOW,
    target_high: int = CELL_PER_ROI_HIGH,
    max_cells: int = CELL_PER_ROI_MAX,
) -> float:
    """按细胞密度自动选择 ROI 边长，使每 ROI 约 target 个细胞（启发式，见模块 docstring）。"""
    xmin, ymin = coords.min(axis=0)
    xmax, ymax = coords.max(axis=0)
    span_x = max(float(xmax - xmin), 1.0)
    span_y = max(float(ymax - ymin), 1.0)
    n = len(coords)
    density = n / (span_x * span_y)
    density = max(density, 1e-9)
    target = 0.5 * (target_low + target_high)
    roi_size = float(np.sqrt(target / density))
    roi_size = min(roi_size, float(np.sqrt(max_cells / density)))  # 不超 max_cells
    roi_size = max(roi_size, float(np.sqrt(min_cells / density)))  # 至少 min_cells
    return roi_size


def _tile_rois(coords: np.ndarray, min_cells: int = CELL_PER_ROI_MIN) -> list[np.ndarray]:
    """自调参 ROI 切片；若仍有 ROI 超过 max_cells 则整体缩小再切（防御性）。"""
    roi_size = _autotune_roi_size(coords, min_cells=min_cells)
    stride = 0.5 * roi_size
    rois = tile_rois(coords, roi_size, stride, min_cells)
    max_n = max((len(r) for r in rois), default=0)
    if max_n > CELL_PER_ROI_MAX:
        factor = float(np.sqrt(CELL_PER_ROI_MAX / max_n))
        rois = tile_rois(coords, roi_size * factor, stride * factor, min_cells)
    return rois


def _sub_graph_tensor(H: sparse.csr_matrix, idx: np.ndarray, device: str):
    """从全切片超图 H 子选择 ROI 并 hpnn 归一化 → torch 稀疏张量。

    与官方 Xenium_HBRC_overlap 一致：sub_graph = normalize_graph(H[roi][:, roi])。
    """
    sub_H = H[np.ix_(idx, idx)]
    M = normalize_hypergraph_hpnn(sub_H)
    return sparse_to_torch(M, device)


def build_pseudo_spot_agg(coords: np.ndarray) -> sparse.csr_matrix:
    """复刻官方 utils.py Generate_pseudo_spot（all_in=True）的 visium 交错网格聚合矩阵。

    返回 (n_spots, N) 稀疏矩阵：agg[s, i]=1 当细胞 i 属于最近伪 spot s。
    伪 spot 网格：x_interval=100，y_interval=100*sqrt(3)，第二套网格偏移 (50, y_interval/2)。
    仅供 use_spot_agg=True 的损失变体使用。
    """
    x = coords[:, 0]
    y = coords[:, 1]
    x_interval = 100.0
    y_interval = 100.0 * np.sqrt(3)

    spot_x1 = np.arange(0.0, x.max() + x_interval, x_interval)
    spot_y1 = np.arange(0.0, y.max() + y_interval, y_interval)
    gx1, gy1 = np.meshgrid(spot_x1, spot_y1)
    spot1 = np.stack([gx1.ravel(), gy1.ravel()], axis=1)

    spot_x2 = np.arange(50.0, x.max() + x_interval, x_interval)
    spot_y2 = np.arange(y_interval / 2.0, y.max() + y_interval, y_interval)
    gx2, gy2 = np.meshgrid(spot_x2, spot_y2)
    spot2 = np.stack([gx2.ravel(), gy2.ravel()], axis=1)

    spot = np.vstack([spot1, spot2])
    tree = cKDTree(spot)
    _, indices = tree.query(coords)  # 每个细胞最近的伪 spot（all_in=True，全部纳入）

    rows = indices.astype(np.int64)
    cols = np.arange(len(coords), dtype=np.int64)
    data = np.ones(len(coords), dtype=np.float32)
    agg = sparse.coo_matrix((data, (rows, cols)), shape=(rows.max() + 1, len(coords)))
    return agg.tocsr()


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
def _compute_metrics(
    y_true_norm: np.ndarray,
    y_pred: np.ndarray,
    gene_norm: str,
    stats: dict | None,
    topk_ks: tuple = (10, 50, 100),
) -> dict:
    """与 harness.evaluate 完全一致的指标计算（向量化，PCC/SPCC 归一化空间，Top-k/AUROC 逆变换 raw）。"""
    y_true_raw = _invert_normalization(y_true_norm, gene_norm, stats)
    y_pred_raw = _invert_normalization(y_pred, gene_norm, stats)
    return compute_metrics_vectorized(y_true_norm, y_pred, y_true_raw, y_pred_raw, topk_ks)


def evaluate_slide(
    model: SpatialExModel,
    test_dir: str,
    gene_norm: str,
    stats: dict | None,
    device: str = "cpu",
    output_dir: str = "outputs",
) -> dict:
    """整片超图推理 + 统一指标（与 harness.evaluate 语义一致）。

    对每个 ROI 做超图卷积预测；ROI 重叠处同一 cell 被多次预测，取各次预测的
    平均值作为最终预测（文档化约定：均值对重叠边界更平滑）。

    返回并保存 {"PCC", "SPCC", "top10", "top50", "top100", "AUROC"} 到
    output_dir/test_results.json。
    """
    model.to(device)  # 保证模型与图/特征同设备（test_spatialex.py 从 CPU checkpoint 加载）
    model.eval()
    coords, features, expr_norm = _load_slide(test_dir, gene_norm, stats)
    H = build_hypergraph(coords, k=K_NEIGHBORS, self_loop=True)
    rois = _tile_rois(coords)

    y_pred = np.zeros_like(expr_norm, dtype=np.float32)
    y_count = np.zeros(len(coords), dtype=np.float32)
    with torch.no_grad():
        for idx in rois:
            G = _sub_graph_tensor(H, idx, device)
            sub_feat = torch.from_numpy(features[idx]).to(device)
            x_prime = model.predict_roi(G, sub_feat)
            y_pred[idx] += x_prime.detach().cpu().numpy()
            y_count[idx] += 1.0
    y_count[y_count == 0] = 1.0
    y_pred /= y_count[:, None]

    results = _compute_metrics(expr_norm, y_pred, gene_norm, stats)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "test_results.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def train_function(model, train_loader, valid_loader, args, stats) -> list:
    """SpatialEx 自定义训练：整片超图 → 重叠 ROI 内 cell 级 MSE。

    train_loader / valid_loader 仅用于接口一致性（训练数据直接从 args.train_dir
    / args.valid_dir 加载并构建超图）。每 epoch 在验证集上跑 evaluate_slide，
    保存最优 best.pt（schema 与 harness.fit 一致：{model, history, config}）。
    """
    device = args.device
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=getattr(args, "weight_decay", 0.0)
    )
    os.makedirs(args.output_dir, exist_ok=True)

    coords, features, expr_norm = _load_slide(args.train_dir, args.gene_norm, stats)
    H = build_hypergraph(coords, k=K_NEIGHBORS, self_loop=True)
    rois = _tile_rois(coords)
    agg_full = None
    if model.use_spot_agg:
        agg_full = build_pseudo_spot_agg(coords)

    roi_data = [(idx, _sub_graph_tensor(H, idx, device)) for idx in rois]
    print(
        f"[SpatialEx] train {len(coords)} cells -> {len(roi_data)} ROIs "
        f"(use_spot_agg={model.use_spot_agg})",
        flush=True,
    )

    best_pcc, best_state = -float("inf"), None
    no_improve = 0
    patience = int(getattr(args, "patience", 10))
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = np.random.permutation(len(roi_data))
        total, n = 0.0, 0
        for r in order:
            idx, G = roi_data[r]
            sub_feat = torch.from_numpy(features[idx]).to(device)
            sub_expr = torch.from_numpy(expr_norm[idx]).to(device)
            optimizer.zero_grad()
            if model.use_spot_agg:
                sub_agg = sparse_to_torch(agg_full[:, idx], device)
                agg_exp = torch.from_numpy(
                    (agg_full[:, idx] @ expr_norm[idx]).astype(np.float32)
                ).to(device)
                selection = torch.arange(len(idx), device=device)
                loss, _, _ = model.predictor(
                    G, sub_feat, agg_exp, agg_mtx=sub_agg, selection=selection
                )
            else:
                loss, _, _ = model.predictor(G, sub_feat, sub_expr)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
            n += len(idx)
        train_loss = total / max(n, 1)

        ev = evaluate_slide(
            model, args.valid_dir, args.gene_norm, stats, device, args.output_dir
        )
        history.append({"epoch": epoch, "train_loss": train_loss, **ev})
        print(
            f"[SpatialEx epoch {epoch}/{args.epochs}] loss={train_loss:.4f} "
            f"val_PCC={ev['PCC']:.4f}",
            flush=True,
        )
        if ev["PCC"] == ev["PCC"] and ev["PCC"] > best_pcc + 1e-4:
            best_pcc = ev["PCC"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"[SpatialEx early stop] val_PCC {patience} 个 epoch 未提升，"
                  f"在 epoch {epoch} 停止", flush=True)
            break

    if best_state is not None:
        in_dim = model.predictor.mlp[0].in_features
        hidden_dim = model.predictor.mlp[0].out_features
        torch.save(
            {
                "model": best_state,
                "history": history,
                "config": {
                    "method": "spatialex",
                    "num_genes": model.num_genes,
                    "in_dim": in_dim,
                    "hidden_dim": hidden_dim,
                    "num_layers": model.predictor.mod.num_layers,
                    "use_spot_agg": model.use_spot_agg,
                    "gene_norm": args.gene_norm,
                },
            },
            os.path.join(args.output_dir, "best.pt"),
        )
    return history
