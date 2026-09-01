"""统一 benchmark harness：所有方法共用同一训练/评估流程（课题要求 4、9）。

方法模型统一接口（各方法在 methods/<name>/model.py 中实现）：
    class SomeModel(nn.Module):
        input_type = 'patch' | 'feature'      # 决定数据加载
        def forward(self, x) -> Tensor        # x=(B,3,H,W) 或 (B,D)；输出 (B,G) 归一化表达预测

评估语义（与 HEST 通用 benchmark 一致）：
    - PCC / SPCC：逐基因（跨细胞）相关性，取均值
    - cell_PCC：逐细胞（跨基因，log1p 计数空间）相关性，取均值
    - SSIM：逐基因空间表达图（坐标栅格化）的结构相似度，取均值（需 coords）
    - Top-k：逐细胞，预测与真实 top-k 高表达基因的重合率，取均值（默认 k=10..100；
      topk_ks='full' 时算全 k=1..G 曲线）
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
    true_list, pred_list, coord_list = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            x = _extract_input(model, batch).to(device)
            pred = model(x).cpu().numpy()
            true_list.append(batch["gene_expr"].cpu().numpy())
            pred_list.append(pred)
            coord_list.append(batch["coords"].cpu().numpy())
    y_true_norm = np.concatenate(true_list, axis=0)
    y_pred = np.concatenate(pred_list, axis=0)
    coords = np.concatenate(coord_list, axis=0)
    y_true_raw = _invert_normalization(y_true_norm, gene_norm, stats)
    y_pred_raw = _invert_normalization(y_pred, gene_norm, stats)
    return y_true_norm, y_true_raw, y_pred, y_pred_raw, coords


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


def _rank_desc(a: np.ndarray) -> np.ndarray:
    """每行内按值**降序**的秩（0 = 最大）。输入 (N, G)，输出同形状 int32。"""
    order = np.argsort(-a, axis=1, kind="stable")
    r = np.empty_like(order, dtype=np.int32)
    np.put_along_axis(r, order, np.arange(a.shape[1])[None, :], axis=1)
    return r


def _topk_curve_full(y_true_raw: np.ndarray, y_pred_raw: np.ndarray):
    """一次性计算全部 k=1..G 的 Top-k 准确率（enter-rank 方法，O(N·G)）。

    对每个细胞，基因 g 同时进入"真值 top-k"与"预测 top-k"当且仅当
        max(true_rank[g], pred_rank[g]) < k  （rank 从 0 计，k 从 1 计）
    记 enter[g] = max(两 rank)，则 |S_k ∩ P_k| = #{g : enter[g] ≤ k-1}。
    对每行 enter 排序后 searchsorted 一次得到所有 k 的重合数，除以 k 即准确率。

    返回 (ks (G,) int, acc (G,) float64)：acc[j] = Top-(j+1) 准确率。
    """
    tr = _rank_desc(y_true_raw)
    pr = _rank_desc(y_pred_raw)
    enter = np.maximum(tr, pr)
    es = np.sort(enter, axis=1).astype(np.int64)    # (N, G) 每行升序
    N, G = es.shape
    # 把每行加上互不重叠的大偏移 → 展平后全局有序，可用一次 searchsorted 算所有行的插入点
    row_off = (np.arange(N, dtype=np.int64) * G)[:, None]
    flat = (es + row_off).ravel()                   # 全局升序
    queries = (row_off + np.arange(1, G + 1, dtype=np.int64)[None, :]).ravel()
    p = np.searchsorted(flat, queries, side="left").reshape(N, G)
    overlap = p - row_off                           # (N,G) |S_{j+1} ∩ P_{j+1}|，j=0..G-1
    ks = np.arange(1, G + 1)
    acc = overlap.astype(np.float64) / ks[None, :]  # (N, G)
    return ks, acc.mean(axis=0)


def _ssim_grid_dims(coords: np.ndarray, grid_size: int = 224) -> tuple[int, int]:
    """按坐标包围盒纵横比决定栅格尺寸（最长边 = grid_size，保证纵横比）。"""
    w = float(coords[:, 0].max() - coords[:, 0].min())
    h = float(coords[:, 1].max() - coords[:, 1].min())
    scale = grid_size / max(w, h, 1e-9)
    return max(int(round(w * scale)), 2), max(int(round(h * scale)), 2)


def _rasterize_bins(coords: np.ndarray, gw: int, gh: int) -> np.ndarray:
    """细胞坐标 → 栅格 bin 编号（(N,) int64），同一栅格内多细胞取均值聚合。"""
    x, y = coords[:, 0], coords[:, 1]
    bx = np.floor((x - x.min()) / max(x.max() - x.min(), 1e-9) * (gw - 1))
    by = np.floor((y - y.min()) / max(y.max() - y.min(), 1e-9) * (gh - 1))
    bx = np.clip(bx, 0, gw - 1).astype(np.int64)
    by = np.clip(by, 0, gh - 1).astype(np.int64)
    return by * gw + bx


def _spatial_ssim(
    y_true_img: np.ndarray,
    y_pred_img: np.ndarray,
    coords: np.ndarray,
    grid_size: int = 224,
) -> np.ndarray:
    """逐基因空间表达图 SSIM（均值聚合到栅格 → ssim_2d），返回 (G,) 数组。

    空间语义：把每个基因的跨细胞表达值按坐标栅格化成空间图（空栅格补 0），
    用 SSIM 衡量预测图与真实图的结构相似度——捕捉 PCC 无法反映的**空间结构**一致性。
    **输入统一为 log1p(raw counts) 公共空间**（跨方法可比，避免 zscore 虚高）；
    栅格纵横比保持、最长边 = grid_size。
    """
    from ..eval.metrics import ssim_2d

    N, G = y_true_img.shape
    gw, gh = _ssim_grid_dims(coords, grid_size)
    bins = _rasterize_bins(coords, gw, gh)
    cnt = np.bincount(bins, minlength=gw * gh).astype(np.float64)
    cnt[cnt == 0] = 1.0
    ssims = np.full(G, np.nan, dtype=np.float64)
    for g in range(G):
        st = (np.bincount(bins, weights=y_true_img[:, g], minlength=gw * gh) / cnt).reshape(gh, gw)
        sp = (np.bincount(bins, weights=y_pred_img[:, g], minlength=gw * gh) / cnt).reshape(gh, gw)
        dr = float(max(st.max(), sp.max()) - min(st.min(), sp.min()))
        ssims[g] = ssim_2d(st, sp, data_range=dr)
    return ssims


def compute_metrics_vectorized(
    y_true_norm: np.ndarray,
    y_pred: np.ndarray,
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    topk_ks: tuple | None = None,
    auroc_threshold: float = 0.0,
    details: bool = False,
    coords: np.ndarray | None = None,
    ssim_grid: int = 224,
    gene_idx: np.ndarray | None = None,
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

    topk_ks_default = tuple(range(10, 101, 10))  # 10,20,...,100（README 展示用 k）
    if topk_ks is None:
        topk_ks = topk_ks_default

    N, G = y_true_norm.shape
    if gene_idx is not None:
        # 只对指定基因子集（如 zero-shot：模型基因 panel ∩ 数据集 313 基因）计算全部指标，
        # 其余基因（填 0 的部分）不参与评测。
        gi = np.asarray(gene_idx, dtype=int)
        y_true_norm = y_true_norm[:, gi]
        y_pred = y_pred[:, gi]
        y_true_raw = y_true_raw[:, gi]
        y_pred_raw = y_pred_raw[:, gi]
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
        if topk_ks == "full":
            # 全 k=1..G 曲线（enter-rank 一次算完），存 _topk_ks/_topk_acc 供 CSV
            ks, acc = _topk_curve_full(y_true_raw, y_pred_raw)
            for k in topk_ks_default:
                topk[f"top{k}"] = float(acc[k - 1]) if k <= G else 1.0
            result_extra = {"_topk_ks": ks, "_topk_acc": acc}
        else:
            result_extra = {}
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

    # SSIM：空间表达图逐基因结构相似度（需要坐标；无坐标则跳过）。
    # **统一在 log1p(counts) 公共空间计算**（所有方法一致），保证跨方法可比——
    # zscore 空间会归一化掉幅度/偏移偏差导致 SSIM 虚高（实测弱预测 0.3× 幅度时
    # zscore=1.0 vs log1p=0.43）。PCC/SPCC/Top-k/AUROC 对单调/仿射不变，不受此影响。
    if coords is not None:
        ssim_true = np.log1p(np.clip(y_true_raw, 0, None))
        ssim_pred = np.log1p(np.clip(y_pred_raw, 0, None))
        ssims = _spatial_ssim(ssim_true, ssim_pred, coords, ssim_grid)
        result_extra["SSIM"] = float(np.nanmean(ssims))
        if details:
            result_extra["gene_ssims"] = ssims

    result = {
        "PCC": float(np.nanmean(pccs)),
        "SPCC": float(np.nanmean(spccs)),
        "cell_PCC": float(np.nanmean(cell_pccs)),
        **topk,
        "AUROC": float(np.nanmean(aurocs)),
        **result_extra,
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
    ssim: bool = False,
) -> dict:
    """完整评估，返回全部指标（课题要求 9）。

    stats 为表达归一化统计量（测试集使用训练集统计量时，与训练一致）。
    details=True 时额外返回逐基因数组 gene_pccs/gene_spccs/gene_aurocs/gene_ssims。
    ssim=True 时计算空间表达图 SSIM（需要数据集提供 coords；训练期验证默认关闭以省时）。
    topk_ks="full" 时计算全部 k=1..G 的 Top-k 曲线（_topk_ks/_topk_acc 供 CSV）。
    """
    y_true_norm, y_true_raw, y_pred, y_pred_raw, coords = predict(
        model, dataloader, device, gene_norm, stats
    )
    return compute_metrics_vectorized(
        y_true_norm, y_pred, y_true_raw, y_pred_raw, topk_ks, auroc_threshold,
        details=details, coords=coords if ssim else None,
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
    summary_keys = ["PCC", "SPCC", "cell_PCC", "SSIM", "AUROC"]
    summary_keys += [f"top{k}" for k in topk_ks]

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k in summary_keys:
            if k in results:
                w.writerow([k, float(results[k])])

    # 全 k=1..G Top-k 曲线（k, accuracy 两列，供画连续曲线）
    curve_path = None
    if "_topk_ks" in results and "_topk_acc" in results:
        curve_path = os.path.join(os.path.dirname(csv_path) or ".", "topk_curve.csv")
        with open(curve_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["k", "accuracy"])
            for k, a in zip(results["_topk_ks"], results["_topk_acc"]):
                w.writerow([int(k), float(a)])

    gene_path = f"{csv_path}_genes.csv"
    gene_pcc_path = f"{csv_path}_gene_pcc.csv"
    if gene_names is not None and "gene_pccs" in results:
        has_ssim = "gene_ssims" in results
        with open(gene_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["gene", "PCC", "SPCC", "AUROC", "SSIM"])
            for i, g in enumerate(gene_names):
                w.writerow([
                    g,
                    float(results["gene_pccs"][i]),
                    float(results["gene_spccs"][i]),
                    float(results["gene_aurocs"][i]),
                    float(results["gene_ssims"][i]) if has_ssim else "",
                ])
        # 专门输出逐基因 PCC（每方法一个 gene_pcc.csv），供误差棒（mean±std）与下游分析
        with open(gene_pcc_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["gene", "PCC"])
            for i, g in enumerate(gene_names):
                w.writerow([g, float(results["gene_pccs"][i])])
    out = {"summary": csv_path, "genes": gene_path, "gene_pcc": gene_pcc_path}
    if curve_path:
        out["topk_curve"] = curve_path
    return out


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
