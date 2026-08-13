"""ST 与 H&E 图像的空间对齐。

已验证结论：h5ad 中 obs[['image_col','image_row']]（= obsm['image_coor']）就是各
细胞在 H&E 图像中的像素坐标，由 Xenium 空间坐标（obsm['spatial']，单位 μm）经仿射
变换得到（缩放≈1/HE_MPP≈2.749 px/μm，含 y 反射，几乎无旋转；最小二乘拟合残差
<0.5px）。因此无需重新配准，直接从 h5ad 读坐标即可。

课题要求 8 的"两个版本对齐"（按坐标 vs 统一 MPP）在此实现为：
- align_by_coords：直接按现成坐标（适合单张切片，MPP 可能不同）
- align_by_mpp：将不同切片的 patch/坐标统一到相同 MPP（多切片情形必须用）
"""
from __future__ import annotations

import numpy as np


def fit_affine_transform(src_pts: np.ndarray, dst_pts: np.ndarray) -> np.ndarray:
    """最小二乘拟合仿射变换。

    参数：
        src_pts: 源点 (N, 2)
        dst_pts: 目标点 (N, 2)
    返回：
        2x3 矩阵 M，满足 dst ≈ [src, 1] @ M
    """
    src = np.asarray(src_pts, dtype=np.float64)
    A = np.column_stack([src, np.ones(len(src))])
    M, *_ = np.linalg.lstsq(A, np.asarray(dst_pts, dtype=np.float64), rcond=None)
    return M


def apply_affine_transform(pts: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """应用 2x3 仿射矩阵。

    参数：
        pts: 点集 (N, 2)
        transform: 2x3 矩阵 M
    返回：
        (N, 2) 变换后的点
    """
    pts = np.asarray(pts, dtype=np.float64)
    return np.column_stack([pts, np.ones(len(pts))]) @ transform


def fit_xenium_he_transform(h5ad_path: str, sample_size: int = 5000) -> np.ndarray:
    """从 h5ad 拟合 Xenium 空间坐标(μm) → H&E 像素坐标的仿射矩阵。

    参数：
        h5ad_path: 预处理后的 *_uni_resolution64_full.h5ad 路径
        sample_size: 用于拟合的采样点数
    返回：
        2x3 仿射矩阵
    """
    import anndata as ad

    adata = ad.read_h5ad(h5ad_path, backed="r")
    spatial = np.asarray(adata.obsm["spatial"], dtype=np.float64)       # (N,2) μm
    image_coor = np.asarray(adata.obsm["image_coor"], dtype=np.float64)  # (N,2) H&E px
    if sample_size and len(spatial) > sample_size:
        idx = np.random.default_rng(0).choice(len(spatial), sample_size, replace=False)
        spatial, image_coor = spatial[idx], image_coor[idx]
    return fit_affine_transform(spatial, image_coor)


def xenium_um_to_he_px(coords_um: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Xenium 空间坐标(μm) → H&E 像素坐标 (col, row)。"""
    return apply_affine_transform(coords_um, transform)


def align_by_coords(coords: np.ndarray) -> np.ndarray:
    """版本 1：按现成坐标直接使用（不统一 MPP）。

    参数：
        coords: H&E 像素坐标 (N, 2)，来自 h5ad image_col/image_row
    返回：
        原坐标（int 取整）
    """
    return np.round(np.asarray(coords)).astype(int)


def align_by_mpp(
    coords_px: np.ndarray,
    src_mpp: float,
    target_mpp: float,
) -> np.ndarray:
    """版本 2：统一 MPP 后对齐（多切片情形必须使用）。

    将坐标按 MPP 比例换算到目标 MPP 下的像素坐标：
        dst_px = coords_px * (src_mpp / target_mpp)

    参数：
        coords_px: 源 MPP 下的像素坐标 (N, 2)
        src_mpp: 源图像 MPP（μm/像素）
        target_mpp: 统一目标 MPP
    返回：
        目标 MPP 下的像素坐标
    """
    return np.round(np.asarray(coords_px, dtype=np.float64) * (src_mpp / target_mpp)).astype(int)
