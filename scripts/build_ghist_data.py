"""GHIST 数据管线构建：Xenium 核多边形 → HE 图像素核分割 mask（ghist_data 格式）。

官方 GHIST 数据格式（methods/ghist 读取）：
    he_image.tif                 H&E 整片（HE 像素帧）
    he_image_nuclei_seg.tif      核分割 mask（uint32，像素值 = 全局 cell_id）
    cell_gene_matrix_filtered.csv  表达矩阵（index = cell_id，列 = 基因，raw counts）
    matched_nuclei_filtered.csv  id_histology(mask id) ↔ 核面积
    celltype_filtered.csv        细胞类型（我们无 Xenium 细胞型，写二值 1）
    avgexp.csv                   参考表达（前 N_REF 细胞的均值向量）

坐标对齐：Xenium 核多边形坐标（µm）→ HE 像素，用 h5ad 中
obsm['spatial']→obsm['image_coor'] 拟合的精确 2D 仿射（含 ~0.28° 旋转，
残差 <0.5px，已验证）。mask 像素值 = Xenium cell_id，与 cell_gene_matrix 索引一致。

用法（远程 myenv1）：
    python scripts/build_ghist_data.py --rep 1
    python scripts/build_ghist_data.py --rep 2
调试：--max_cells 限制细胞数（验证用，输出截断文件会污染正式数据，正式跑勿加）。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE = "/cpfs01/projects-HDD/cfff-d7ff0c9cdf2f_HDD/hjr_24300980068/HE2ST/datasets"
REP_H5AD = {
    1: os.path.join(BASE, "Human_Breast_Cancer_Rep1_uni_resolution64_full.h5ad"),
    2: os.path.join(BASE, "Human_Breast_Cancer_Rep2_uni_resolution64_full.h5ad"),
}
REP_HE = {
    1: os.path.join(BASE, "he_images", "Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.ome.tif"),
    2: os.path.join(BASE, "he_images", "Xenium_FFPE_Human_Breast_Cancer_Rep2_he_image.ome.tif"),
}
REP_BOUND = {
    1: os.path.join(BASE, "Xenium_FFPE_Human_Breast_Cancer_Rep1_outs", "outs", "nucleus_boundaries.csv.gz"),
    2: os.path.join(BASE, "Xenium_FFPE_Human_Breast_Cancer_Rep2_outs", "outs", "nucleus_boundaries.csv.gz"),
}
REP_DATA = {1: os.path.expanduser("~/HE2ST-pj/data/rep1"),
            2: os.path.expanduser("~/HE2ST-pj/data/rep2")}
OUT = os.path.expanduser("~/HE2ST-pj/data/ghist_rep")   # OUT + str(rep)

N_REF = 512      # avgexp 参考细胞数（与 methods/ghist N_REF 一致）
MIN_AREA = 10    # 官方 min_nuc_area：小于该像素面积的核丢弃


def fit_affine(h5ad_path: str, n_fit: int = 20000) -> np.ndarray:
    """从 h5ad 拟合 Xenium(spatial)→HE(image_coor) 的 2D 仿射（含旋转）。

    返回 3×3 齐次矩阵 M，使 [HE_x, HE_y, 1] = M @ [X, Y, 1]。
    """
    import anndata as ad

    adata = ad.read_h5ad(h5ad_path, backed="r")
    S = np.asarray(adata.obsm["spatial"][:n_fit]).astype(float)
    H = np.asarray(adata.obsm["image_coor"][:n_fit]).astype(float)
    M_aug = np.c_[S, np.ones(len(S))]
    A, *_ = np.linalg.lstsq(M_aug, H, rcond=None)     # (3,2)
    M = np.eye(3)
    M[:2, :] = A.T
    pred = M_aug @ A
    err = np.abs(pred - H)
    print(f"[GHIST] 仿射拟合残差: col std={err[:,0].std():.3f} row std={err[:,1].std():.3f} "
          f"max={err.max():.3f}", flush=True)
    return M


def transform_poly(vertices: np.ndarray, M: np.ndarray) -> np.ndarray:
    """多边形顶点 (K,2) [vx,vy] → HE 像素 (K,2) [col,row]。"""
    aug = np.c_[vertices, np.ones(len(vertices))]
    return (aug @ M[:2, :].T)


def main() -> None:
    p = argparse.ArgumentParser(description="GHIST 数据管线构建")
    p.add_argument("--rep", type=int, choices=[1, 2], required=True)
    p.add_argument("--max_cells", type=int, default=None,
                   help="调试：限制细胞数（会污染正式数据，仅验证用）")
    p.add_argument("--out_dir", default=None,
                   help="输出目录（默认 ~/HE2ST-pj/data/ghist_rep{rep}；验证时指定临时目录）")
    args = p.parse_args()

    import pandas as pd
    import tifffile
    import cv2

    from common.data.preprocess import load_he_image

    rep = args.rep
    out_dir = args.out_dir or (OUT + str(rep))
    os.makedirs(out_dir, exist_ok=True)

    # 1) 精确仿射
    M = fit_affine(REP_H5AD[rep])

    # 2) HE 图像 + metadata + 表达
    he = load_he_image(REP_HE[rep])                       # (H,W,3) uint8
    H, W = he.shape[:2]
    print(f"[GHIST] rep{rep}: HE {he.shape}", flush=True)
    meta = pd.read_csv(os.path.join(REP_DATA[rep], "metadata.csv"))
    expr = np.load(os.path.join(REP_DATA[rep], "gene_expression.npy"))  # (N,G) raw
    with open(os.path.join(REP_DATA[rep], "gene_names.txt")) as f:
        gene_names = [l.strip() for l in f if l.strip()]
    if args.max_cells:
        meta = meta.iloc[:args.max_cells]
        expr = expr[:args.max_cells]
    cell_ids = meta["cell_id"].astype(int).tolist()
    print(f"[GHIST] {len(cell_ids)} cells, 表达 {expr.shape}, 基因 {len(gene_names)}", flush=True)

    # 3) 核多边形（µm），过滤到 metadata cell_ids
    nb = pd.read_csv(REP_BOUND[rep])
    cid_set = set(cell_ids)
    nb = nb[nb["cell_id"].isin(cid_set)]
    grouped = {cid: g[["vertex_x", "vertex_y"]].values
               for cid, g in nb.groupby("cell_id")}
    print(f"[GHIST] 匹配到多边形核: {len(grouped)}", flush=True)

    # 4) 光栅化：像素值 = cell_id
    mask = np.zeros((H, W), dtype=np.int32)
    n_skipped = 0
    for cid in cell_ids:
        verts = grouped.get(cid)
        if verts is None or len(verts) < 3:
            n_skipped += 1
            continue
        pts = transform_poly(verts.astype(float), M)
        # 去掉越界顶点，核主体在图像外则跳过
        inside = ((pts[:, 0] >= 0) & (pts[:, 0] < W) &
                  (pts[:, 1] >= 0) & (pts[:, 1] < H))
        if inside.sum() < 3:
            n_skipped += 1
            continue
        pts = np.clip(pts, [0, 0], [W - 1, H - 1]).astype(np.int32)
        # 面积过滤
        area = cv2.contourArea(pts)
        if area < MIN_AREA:
            n_skipped += 1
            continue
        cv2.fillPoly(mask, [pts.reshape(-1, 1, 2)], int(cid))
    print(f"[GHIST] 跳过核: {n_skipped}（越界/过小/无多边形）", flush=True)
    print(f"[GHIST] mask 唯一核数: {len(np.unique(mask)) - 1}", flush=True)

    # 5) 对齐验证：mask 质心 vs 变换后多边形质心（应为 <1px，验证光栅化本身）
    ys, xs = np.where(mask > 0)
    print(f"[GHIST] mask 非零范围: 行 {ys.min()}-{ys.max()} 列 {xs.min()}-{xs.max()}", flush=True)
    sample = [c for c in cell_ids[:100] if (mask == c).any()]
    if sample:
        dx, dy = [], []
        for c in sample:
            cy_, cx_ = np.where(mask == c)
            mcx, mcy = cx_.mean(), cy_.mean()
            poly_ref = transform_poly(grouped[c].astype(float), M).mean(0)  # 变换后多边形质心
            dx.append(abs(mcx - poly_ref[0]))
            dy.append(abs(mcy - poly_ref[1]))
        print(f"[GHIST] 对齐验证({len(sample)}核): mask质心 vs 多边形质心 "
              f"dx max={max(dx):.2f} mean={np.mean(dx):.2f} "
              f"dy max={max(dy):.2f} mean={np.mean(dy):.2f}", flush=True)

    # 6) 写出
    tifffile.imwrite(os.path.join(out_dir, "he_image.tif"), he)
    tifffile.imwrite(os.path.join(out_dir, "he_image_nuclei_seg.tif"),
                     mask.astype(np.uint32), photometric="minisblack")
    # 表达矩阵：index = cell_id（与 mask 像素值一致，Framework 按 sorted mask id 迭代）
    df_expr = pd.DataFrame(expr, index=cell_ids, columns=gene_names)
    df_expr.to_csv(os.path.join(out_dir, "cell_gene_matrix_filtered.csv"))
    # matched_nuclei：mask id(==cell_id) + 核面积（单次 np.unique 计数，勿逐核扫描大 mask）
    uniq, counts = np.unique(mask, return_counts=True)
    area_map = dict(zip(uniq.tolist(), counts.astype(float)))
    pd.DataFrame({"id_histology": cell_ids, "id_xenium": cell_ids,
                  "area_pix": [area_map.get(c, 0.0) for c in cell_ids]}).to_csv(
        os.path.join(out_dir, "matched_nuclei_filtered.csv"), index=False)
    # celltype：无 Xenium 细胞型，写二值 1（官方 fallback：types_patch = >0 ? 1 : 0）
    pd.DataFrame({"ct": [1] * len(cell_ids)}, index=cell_ids).to_csv(
        os.path.join(out_dir, "celltype_filtered.csv"))
    # avgexp：前 N_REF 细胞的均值向量（官方参考表达）
    pd.DataFrame(expr[:N_REF].mean(0, keepdims=True), index=[0],
                 columns=gene_names).to_csv(os.path.join(out_dir, "avgexp.csv"))
    with open(os.path.join(out_dir, "genes.txt"), "w") as f:
        f.write("\n".join(gene_names) + "\n")

    print(f"[GHIST] 已写出 {out_dir}: he_image.tif / he_image_nuclei_seg.tif / "
          f"cell_gene_matrix_filtered.csv / matched_nuclei_filtered.csv / "
          f"celltype_filtered.csv / avgexp.csv", flush=True)


if __name__ == "__main__":
    main()
