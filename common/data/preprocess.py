"""HE 图像预处理：H&E patch 提取、保存、组织筛选、颜色归一化。

管线（已按真实数据验证）：
    h5ad(image_col/image_row = H&E 像素坐标) → 从 he_image.ome.tif 裁 patch → 保存
课题要求 4：所有方法使用同一预处理模块，保证公平比较。
"""
from __future__ import annotations

import os
import sys

import numpy as np


def load_he_image(he_path: str) -> np.ndarray:
    """加载 H&E 全切片图像为 numpy 数组 (H, W, 3) uint8。

    使用 tifffile 读取 ome.tif（内存约 2.2GB，单次加载即可反复裁 patch）。
    """
    import tifffile

    img = tifffile.imread(he_path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[0] == 3:  # (C,H,W) → (H,W,C)
        img = img.transpose(1, 2, 0)
    return np.asarray(img, dtype=np.uint8)


def extract_patches(
    image: np.ndarray,
    centers_px: np.ndarray,
    patch_size: int = 256,
    skip_out_of_bounds: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """以细胞中心 (col, row) 像素坐标从 H&E 图像裁剪方形 patch。

    参数：
        image: H&E 图像 (H, W, 3)
        centers_px: 中心坐标 (N, 2)，第 0 列为 col(x)，第 1 列为 row(y)
        patch_size: patch 边长（偶数）
        skip_out_of_bounds: 越界样本跳过（True）或裁剪（False）
    返回：
        (patches (M, patch_size, patch_size, 3), valid_centers (M, 2))
    """
    h, w = image.shape[:2]
    half = patch_size // 2
    patches, valid = [], []
    for col, row in centers_px.astype(int):
        left, top = col - half, row - half
        if left < 0 or top < 0 or left + patch_size > w or top + patch_size > h:
            if skip_out_of_bounds:
                continue
            left = max(0, left)
            top = max(0, top)
        patch = image[top:top + patch_size, left:left + patch_size]
        if patch.shape[:2] != (patch_size, patch_size):
            if skip_out_of_bounds:
                continue
            patch = np.pad(patch, ((0, patch_size - patch.shape[0]),
                                   (0, patch_size - patch.shape[1]), (0, 0)))
        patches.append(patch)
        valid.append((col, row))
    return np.stack(patches), np.asarray(valid)


def save_patches(
    patches: np.ndarray,
    centers_px: np.ndarray,
    cell_ids: np.ndarray,
    output_dir: str,
    prefix: str = "cell",
) -> str:
    """保存 patch 为 PNG + 生成 metadata.csv。

    目录结构（与现有 xenium_rep1 兼容，供 HESTDataset 读取）：
        output_dir/patches/cell_{id}.png
        output_dir/metadata.csv  (cell_id, x_centroid, y_centroid, image_col, image_row, patch_path)

    返回：
        metadata.csv 路径
    """
    from PIL import Image

    patches_dir = os.path.join(output_dir, "patches")
    os.makedirs(patches_dir, exist_ok=True)
    rows = []
    for i in range(len(patches)):
        name = f"{prefix}_{cell_ids[i]}.png"
        path = os.path.join(patches_dir, name)
        Image.fromarray(patches[i]).save(path)
        rows.append({
            "cell_id": cell_ids[i],
            "x_centroid": centers_px[i, 0],
            "y_centroid": centers_px[i, 1],
            "patch_path": path,
        })
    meta_path = os.path.join(output_dir, "metadata.csv")
    import pandas as pd

    pd.DataFrame(rows).to_csv(meta_path, index=False)
    return meta_path


def extract_patches_from_h5ad(
    h5ad_path: str,
    he_path: str,
    output_dir: str,
    patch_size: int = 256,
    max_cells: int | None = None,
) -> str:
    """完整预处理：h5ad 坐标 → H&E 裁 patch → 保存（patches + metadata.csv）。

    参数：
        h5ad_path: *_uni_resolution64_full.h5ad
        he_path: H&E 图像 .ome.tif
        output_dir: 输出目录（patches/ 与 metadata.csv）
        patch_size: patch 边长
        max_cells: 调试用，限制细胞数
    返回：
        (metadata.csv 路径, 保留的 cell ids (M,) 数组)
    """
    import anndata as ad

    adata = ad.read_h5ad(h5ad_path, backed="r")
    n = min(len(adata), max_cells) if max_cells else len(adata)
    cell_ids = np.asarray(adata.obs_names[:n], dtype=object)
    centers = adata.obs.loc[adata.obs_names[:n], ["image_col", "image_row"]].to_numpy(dtype=np.float64)

    print(f"加载 H&E 图像 {he_path} ...", flush=True)
    image = load_he_image(he_path)
    print(f"提取 {n} 个 patch ...", flush=True)
    patches, valid = extract_patches(image, centers, patch_size)
    # 反查越界被跳过的细胞，只保留成功提取的 cell id
    valid_set = {(int(a), int(b)) for a, b in valid}
    keep = np.array([(int(c[0]), int(c[1])) in valid_set for c in centers])
    kept_ids = cell_ids[keep]
    meta_path = save_patches(patches, valid, kept_ids, output_dir)
    print(f"完成：{len(patches)}/{n} 个 patch 保存到 {output_dir}", flush=True)
    return meta_path, kept_ids
