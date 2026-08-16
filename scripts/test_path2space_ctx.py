"""Path2Space 大上下文特征评估（冻结 154-MLP 集成 + 可选空间平滑）。

与 scripts/test_path2space.py 的区别：
    1) 指定新的 context 特征文件（如 X_ctranspath_ctx512.npy，由
       extract_ctranspath_context.py 生成，含 Macenko 归一化 + 大上下文 tile）；
    2) 可选 --smooth 应用官方 KDTree 空间平滑（对 log1p 预测后处理，radius 单位 µm，
       官方 radius=2 网格坐标 ≈ 200µm）；
    3) 无平滑 / 有平滑各输出一套指标，便于对比。

评估语义与统一 harness 完全一致（gene_norm='none'，raw counts 语义）：
    PCC/SPCC 逐基因、Top-k 逐细胞、AUROC 逐基因（raw counts>0 为标签）。

用法（远程 myenv1）：
    # 无平滑
    python scripts/test_path2space_ctx.py --test_dir ~/HE2ST-pj/data/rep2 \
        --features data/rep2/X_ctranspath_ctx512.npy \
        --ensemble_dir <mlp_ensemble> --genes_txt <genes.txt> \
        --output_dir outputs/bench_path2space_ctx512
    # 有平滑
    python scripts/test_path2space_ctx.py ... --smooth --smooth_radius_um 200 \
        --output_dir outputs/bench_path2space_ctx512_s200
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根目录

import numpy as np
import pandas as pd
import torch

from common.benchmark.harness import compute_metrics_vectorized
from common.data.expression import load_expression

PATH2SPACE_ENSEMBLE_HINT = ("/cpfs01/projects-HDD/cfff-d7ff0c9cdf2f_HDD/"
                            "hjr_24300980068/HE2ST/path2space/mlp_ensemble")
PATH2SPACE_GENES_HINT = ("/cpfs01/projects-HDD/cfff-d7ff0c9cdf2f_HDD/"
                         "hjr_24300980068/HE2ST/path2space/genes.txt")
UM_PER_PX = 0.364  # Xenium H&E 实测 MPP（µm/px）


def smooth_by_kdtree(pred_log1p: np.ndarray, coords_um: np.ndarray,
                     radius_um: float) -> np.ndarray:
    """KDTree 空间平滑（官方 smoothing.smooth_genes_kdtree 的向量化版本）。

    输入/输出均为 log1p 预测 (N, G)。radius 单位 µm（像素坐标 × UM_PER_PX）。
    无邻居的细胞保持原值。官方在 log1p 空间平滑，故此处一致。
    """
    from scipy.sparse import csr_matrix
    from scipy.spatial import cKDTree

    N = pred_log1p.shape[0]
    tree = cKDTree(coords_um)
    nbrs = tree.query_ball_point(coords_um, radius_um)
    rows, cols = [], []
    for i, ns in enumerate(nbrs):
        rows.extend([i] * len(ns))
        cols.extend(ns)
    deg = np.array([len(ns) for ns in nbrs], dtype=np.float64)
    S = csr_matrix((np.ones(len(rows), dtype=np.float64), (rows, cols)),
                   shape=(N, N))
    smoothed = S.dot(pred_log1p) / deg[:, None]
    smoothed[deg == 0] = pred_log1p[deg == 0]
    return smoothed


def main() -> None:
    p = argparse.ArgumentParser(description="Path2Space 大上下文特征评估")
    p.add_argument("--test_dir", required=True, help="测试集数据目录")
    p.add_argument("--features", required=True, help="context 特征文件路径（.npy，(N,768)）")
    p.add_argument("--ensemble_dir", default=PATH2SPACE_ENSEMBLE_HINT)
    p.add_argument("--genes_txt", default=PATH2SPACE_GENES_HINT)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--device", default="cuda")
    p.add_argument("--smooth", action="store_true", help="应用 KDTree 空间平滑")
    p.add_argument("--smooth_radius_um", type=float, default=200.0,
                   help="平滑半径（µm，官方 radius=2 网格 ≈200µm）")
    p.add_argument("--n_cells", type=int, default=None, help="调试/验证：只评估前 N 个细胞")
    p.add_argument("--output_dir", default="outputs")
    args = p.parse_args()

    from methods.path2space.model import Path2SpaceModel

    feats = np.load(args.features).astype(np.float32)
    expr_raw, gene_names = load_expression(args.test_dir)
    if args.n_cells:
        feats = feats[: args.n_cells]
        expr_raw = expr_raw[: args.n_cells]
    print(f"[P2S-ctx] 特征 {feats.shape}，真值 {expr_raw.shape}", flush=True)
    num_genes = len(gene_names)

    model = Path2SpaceModel(
        num_genes=num_genes, ensemble_dir=args.ensemble_dir,
        genes_txt=args.genes_txt, gene_names=gene_names,
        output_is_log1p=True, device=args.device,
    )
    print(f"[P2S-ctx] 冻结模型就绪: {len(model.out_indices)} 个公共基因有输出映射",
          flush=True)

    # 冻结集成预测 → (N, 313) raw counts
    pred_raw = []
    with torch.no_grad():
        for i in range(0, feats.shape[0], args.batch_size):
            x = torch.as_tensor(feats[i:i + args.batch_size], dtype=torch.float32)
            pred_raw.append(model(x).cpu().numpy())
    y_pred_raw = np.concatenate(pred_raw, axis=0).astype(np.float32)
    y_true_raw = np.asarray(expr_raw, dtype=np.float32)
    print(f"[P2S-ctx] 预测 {y_pred_raw.shape}（raw counts 语义）", flush=True)

    # 空间平滑（官方在 log1p 空间平滑）
    if args.smooth:
        meta = pd.read_csv(os.path.join(args.test_dir, "metadata.csv"))
        coords_um = meta[["x_centroid", "y_centroid"]].values.astype(np.float64) * UM_PER_PX
        y_pred_raw = np.expm1(np.clip(
            smooth_by_kdtree(np.log1p(y_pred_raw), coords_um, args.smooth_radius_um),
            -30.0, 30.0)).astype(np.float32)
        print(f"[P2S-ctx] 已应用空间平滑 radius={args.smooth_radius_um}µm", flush=True)

    results = compute_metrics_vectorized(y_true_raw, y_pred_raw, y_true_raw, y_pred_raw)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
