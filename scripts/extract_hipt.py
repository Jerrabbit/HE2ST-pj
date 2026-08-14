"""HIPT-4K 特征提取 → data_dir/X_hipt.npy（Pixel2Gene 输入，576 维/细胞）。

流程（伪 Visium spot 级适配）：
    1. 读整片 H&E tif（远程 /cpfs01/.../he_images/）；
    2. 伪 Visium 分箱（100µm 六角网格，官方 format_xenium/bin_pseudo_visium 语义）
       把细胞聚成 spot，过滤细胞数过少的空 spot；
    3. 每个 spot 取 2048×2048 上下文区域（8×8 个 256 patch，真实层级上下文），
       HIPT-4K 提取 concat[mean256(384), cls4k(192)] = 576 维（官方 asset_dict 同款）；
    4. 每细胞继承其 spot 的 576 维特征，写入 data_dir/X_hipt.npy (N, 576)。

用法（远程服务器）：
    python scripts/extract_hipt.py --rep 2 \
        --model256 ~/HE2ST-pj/weights/pixel2gene/vit_256_small_dino.pth \
        --model4k  ~/HE2ST-pj/weights/pixel2gene/vit_4096_xs_dino.pth
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial import cKDTree

Image.MAX_IMAGE_PIXELS = None

sys.path.insert(0, os.path.expanduser("~/HE2ST-pj/methods/pixel2gene/hipt"))

from hipt_model_utils import eval_transforms  # noqa: E402

MPP = 0.363788           # µm/px（Rep1/Rep2 已实测）
SPOT_SPACING_UM = 100.0  # 官方 Visium 中心距
CONTEXT_PX = 2048        # HIPT 上下文区域边长（256 的倍数）
MIN_CELLS_PER_SPOT = 10

REP_TIF = {
    1: "/cpfs01/projects-HDD/cfff-d7ff0c9cdf2f_HDD/hjr_24300980068/HE2ST/"
       "datasets/he_images/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.ome.tif",
    2: "/cpfs01/projects-HDD/cfff-d7ff0c9cdf2f_HDD/hjr_24300980068/HE2ST/"
       "datasets/he_images/Xenium_FFPE_Human_Breast_Cancer_Rep2_he_image.ome.tif",
}


def bin_pseudo_visium(coords_px: np.ndarray, spacing_px: float) -> tuple[np.ndarray, np.ndarray]:
    """把细胞像素坐标分箱到伪 Visium 六角网格。

    官方 bin_pseudo_visium 语义：两套错位网格（x 步长 spacing，y 步长 spacing*sqrt(3)），
    每细胞取最近 spot 中心。返回 (spot_id (N,), centers (M,2))。
    """
    row_spacing = spacing_px * np.sqrt(3)
    xmin, ymin = coords_px.min(0)
    xmax, ymax = coords_px.max(0)

    # 套 1：覆盖全范围
    n_x = max(int((xmax - xmin) // spacing_px) + 1, 2)
    n_y = max(int((ymax - ymin) // row_spacing) + 1, 2)
    cx1, cy1 = np.meshgrid(np.linspace(xmin, xmax, n_x),
                           np.linspace(ymin, ymax, n_y))
    centers1 = np.stack([cx1.ravel(), cy1.ravel()], axis=1)
    # 套 2：错位 offset (spacing/2, row_spacing/2)
    cx2, cy2 = np.meshgrid(np.linspace(xmin + spacing_px / 2, xmax - spacing_px / 2, n_x - 1),
                           np.linspace(ymin + row_spacing / 2, ymax - row_spacing / 2, n_y - 1))
    centers2 = np.stack([cx2.ravel(), cy2.ravel()], axis=1)
    centers = np.vstack([centers1, centers2])

    tree = cKDTree(centers)
    _, spot_id = tree.query(coords_px)
    return spot_id.astype(np.int64), centers


def _crop_context(img: np.ndarray, cx: float, cy: float) -> np.ndarray:
    """取 (cx, cy) 为中心、CONTEXT_PX 边长的区域；越界用 edge pad。"""
    half = CONTEXT_PX // 2
    H, W = img.shape[:2]
    x0, y0 = int(cx - half), int(cy - half)
    x1, y1 = int(cx + half), int(cy + half)
    pad_l = max(0, -x0); pad_t = max(0, -y0)
    x0p, y0p = max(0, x0), max(0, y0)
    x1p, y1p = min(W, x1), min(H, y1)
    crop = img[y0p:y1p, x0p:x1p]
    if crop.shape[0] != CONTEXT_PX or crop.shape[1] != CONTEXT_PX:
        crop = np.pad(crop, (
            (pad_t, CONTEXT_PX - crop.shape[0] - pad_t),
            (pad_l, CONTEXT_PX - crop.shape[1] - pad_l),
            (0, 0),
        ), mode="edge")
    return crop


def main() -> None:
    p = argparse.ArgumentParser(description="HIPT-4K 特征提取")
    p.add_argument("--rep", type=int, choices=[1, 2], required=True)
    p.add_argument("--model256", required=True, help="vit_256_small_dino.pth")
    p.add_argument("--model4k", required=True, help="vit_4096_xs_dino.pth")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_spots", type=int, default=None, help="调试：限制 spot 数")
    args = p.parse_args()

    import torch
    import tifffile
    from hipt_4k import HIPT_4K

    data_dir = os.path.expanduser(f"~/HE2ST-pj/data/rep{args.rep}")
    meta = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
    # metadata x_centroid/y_centroid = 像素坐标 (col,row)=(x,y)（save_patches 存的就是 centers_px）
    coords = meta[["x_centroid", "y_centroid"]].to_numpy(dtype=np.float32)

    spacing_px = SPOT_SPACING_UM / MPP
    spot_id, centers = bin_pseudo_visium(coords, spacing_px)
    # 过滤空 spot（细胞数过少），保留有效 spot
    keep_spots = np.flatnonzero(np.bincount(spot_id, minlength=len(centers)) >= MIN_CELLS_PER_SPOT)
    valid_mask = np.isin(spot_id, keep_spots)
    spot_id = spot_id[valid_mask]
    coords_valid = coords[valid_mask]
    n_dropped = int((~valid_mask).sum())
    if n_dropped:
        print(f"[HIPT] 丢弃 {n_dropped} 个离群/空 spot 细胞", flush=True)
    remap = {old: i for i, old in enumerate(keep_spots)}
    spot_id = np.array([remap[s] for s in spot_id], dtype=np.int64)
    spot_centers = centers[keep_spots]
    print(f"[HIPT] {len(coords_valid)} 细胞 → {len(spot_centers)} 个 spot "
          f"(spacing={spacing_px:.0f}px, context={CONTEXT_PX}px)", flush=True)
    if args.max_spots:
        spot_centers = spot_centers[:args.max_spots]
        spot_id = spot_id[spot_id < len(spot_centers)]

    print(f"[HIPT] 加载整片 H&E {REP_TIF[args.rep]} ...", flush=True)
    img = tifffile.imread(REP_TIF[args.rep])
    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    img = np.asarray(img, dtype=np.uint8)
    print(f"[HIPT] 图像 {img.shape}", flush=True)

    model = HIPT_4K(model256_path=args.model256, model4k_path=args.model4k,
                    device256=torch.device(args.device), device4k=torch.device(args.device))
    model.eval()

    spot_feats = np.zeros((len(spot_centers), 576), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i, (cx, cy) in enumerate(spot_centers):
            crop = _crop_context(img, cx, cy)
            x = eval_transforms()(Image.fromarray(crop)).unsqueeze(0).to(args.device)
            asset = model.forward_asset_dict(x)
            spot_feats[i] = asset["features_mean256_cls4k"]  # (1,576)
            if (i + 1) % 100 == 0:
                el = (time.time() - t0) / (i + 1)
                print(f"[HIPT] {i+1}/{len(spot_centers)} spots "
                      f"({el:.2f}s/spot, 剩余 ~{(len(spot_centers)-i-1)*el/60:.1f} 分钟)",
                      flush=True)
    print(f"[HIPT] 特征提取完成，用时 {(time.time()-t0)/60:.1f} 分钟", flush=True)

    # 每细胞继承其 spot 特征 → (N, 576)
    cell_feats = spot_feats[spot_id]
    out = os.path.join(data_dir, "X_hipt.npy")
    np.save(out, cell_feats)
    print(f"[HIPT] 已保存 {out} 形状 {cell_feats.shape}", flush=True)
    # 保存 spot 信息（供评估/回溯）
    np.savez(os.path.join(data_dir, "hipt_spots.npz"),
             spot_id=spot_id, spot_centers=spot_centers, spot_feats=spot_feats)


if __name__ == "__main__":
    main()
