"""构造"保留空细胞"的数据目录（rep*_e），用于实证空细胞对 PCC 的影响。

背景：同学用"只去不在H&E上的细胞、保留空细胞"的数据跑 ref MLP+ref 特征得 PCC>0.4，
我们用去空细胞的数据得 0.3572。本脚本把 h5ad 阶段被排除的空细胞（total_counts<10，
不在现数据中）加回，构造 rep*_e（= 现数据 + 空细胞）。

做法（rep=1/2）：
    1. 从原始 Xenium outs cells.csv 找空细胞（tc<10 且不在现数据 cell_id）。
    2. 仿射(由 h5ad spatial→image_col/image_row 拟合) 把空细胞 μm 坐标映射到 H&E 像素。
    3. 从 H&E 裁 l1×l1 patch，用 extract_reference 提忠实特征（g{l1}_ref + l{l2}_ref，LN）。
    4. 追加到现数据 → 输出 data/rep{rep}_e/（metadata+expr+特征全 append）。
    空细胞表达设为 0（其 tc≤9，近似足够；关键是它们作为"零表达"样本存在）。

用法（远程 myenv1）：
    python3 scripts/add_empty_cells.py --rep 1 --l1 112 --l2_list 28 42 56 70 84 98
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

BASE = "/cpfs01/projects-HDD/cfff-d7ff0c9cdf2f_HDD/hjr_24300980068/HE2ST/datasets"
REP_H5AD = {1: f"{BASE}/Human_Breast_Cancer_Rep1_uni_resolution64_full.h5ad",
            2: f"{BASE}/Human_Breast_Cancer_Rep2_uni_resolution64_full.h5ad"}
REP_HE = {1: f"{BASE}/he_images/Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.ome.tif",
          2: f"{BASE}/he_images/Xenium_FFPE_Human_Breast_Cancer_Rep2_he_image.ome.tif"}
REP_CELLS = {1: f"{BASE}/Xenium_FFPE_Human_Breast_Cancer_Rep1_outs/outs/cells.csv.gz",
             2: f"{BASE}/Xenium_FFPE_Human_Breast_Cancer_Rep2_outs/outs/cells.csv.gz"}
UNI2_WEIGHTS = "/cpfs01/projects-HDD/cfff-d7ff0c9cdf2f_HDD/hjr_24300980068/HE2ST/uni2_model/pytorch_model.bin"


def fit_affine(src, dst):
    A = np.column_stack([src, np.ones(len(src))])
    M, *_ = np.linalg.lstsq(A, dst, rcond=None)
    return M


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rep", type=int, choices=[1, 2], required=True)
    p.add_argument("--l1", type=int, default=112)
    p.add_argument("--l2_list", nargs="+", type=int, default=[28, 42, 56, 70, 84, 98])
    p.add_argument("--src_dir", default=None)  # 现数据目录，默认 data/rep{rep}
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    src = args.src_dir or os.path.expanduser(f"~/HE2ST-pj/data/rep{args.rep}")
    out = os.path.expanduser(f"~/HE2ST-pj/data/rep{args.rep}_e")
    # 现数据的 cell_id
    cur_meta = pd.read_csv(os.path.join(src, "metadata.csv"))
    cur_ids = set(cur_meta["cell_id"].astype(str))
    cur_expr = np.load(os.path.join(src, "gene_expression.npy"))
    G = cur_expr.shape[1]

    # 空细胞 = cells.csv 中 tc<10 且不在现数据
    cells = pd.read_csv(REP_CELLS[args.rep], compression="gzip")
    cells["cell_id"] = cells["cell_id"].astype(str)
    sub = cells[~cells["cell_id"].isin(cur_ids)]
    empty = sub[sub["total_counts"] < 10].copy()
    print(f"[add_empty] rep{args.rep}: 现数据 {len(cur_meta)}，空细胞 {len(empty)}", flush=True)
    if len(empty) == 0:
        raise SystemExit("无空细胞需加回")

    # 仿射 μm→H&E px（用 h5ad 的 spatial vs image_col/image_row 拟合）
    import anndata as ad
    a = ad.read_h5ad(REP_H5AD[args.rep], backed="r")
    sp = np.asarray(a.obsm["spatial"])
    ic = a.obs["image_col"].to_numpy(float)
    ir = a.obs["image_row"].to_numpy(float)
    a.file.close()
    M = fit_affine(sp, np.column_stack([ic, ir]))
    um = empty[["x_centroid", "y_centroid"]].to_numpy(float)
    px = np.column_stack([um, np.ones(len(um))]) @ M
    col = px[:, 0]; row = px[:, 1]
    # 验证空细胞在 H&E 内
    he = _load_he(REP_HE[args.rep])
    H, W = he.shape[:2]
    inside = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    print(f"  空细胞在 H&E 内: {inside.sum()}/{len(empty)}", flush=True)
    empty = empty[inside]
    col, row = col[inside], row[inside]

    # 裁 patch + 忠实特征
    from common.features.uni2 import UNI2FeatureExtractor
    extractor = UNI2FeatureExtractor(UNI2_WEIGHTS, device=args.device)
    l1 = args.l1
    ratios = [l2 / 224.0 for l2 in args.l2_list]
    g_all, l_all = [], {r: [] for r in ratios}
    half = l1 // 2
    BS = 256
    for s in range(0, len(empty), BS):
        cols = col[s:s + BS]; rows = row[s:s + BS]
        patches = np.stack([_crop(he, int(c), int(r), l1) for c, r in zip(cols, rows)])
        g, loc = extractor.extract_reference(patches, ratios, 128, layer_norm=True)
        g_all.append(g)
        for r, v in loc.items():
            l_all[r].append(v)
    g = np.concatenate(g_all, axis=0).astype(np.float32)
    print(f"[add_empty] 空细胞特征: Global {g.shape}", flush=True)

    # 构造 rep*_e 目录
    os.makedirs(out, exist_ok=True)
    # metadata（追加空细胞；patch_path 用占位）
    n_empty = len(empty)
    empty_rows = pd.DataFrame({
        "cell_id": empty["cell_id"].values,
        "x_centroid": col.round(0).astype(int),
        "y_centroid": row.round(0).astype(int),
        "patch_path": [f"EMPTY_{i}" for i in range(n_empty)],
    })
    new_meta = pd.concat([cur_meta, empty_rows], ignore_index=True)
    new_meta.to_csv(os.path.join(out, "metadata.csv"), index=False)
    # 表达（空细胞=0）
    new_expr = np.zeros((n_empty, G), dtype=cur_expr.dtype)
    np.save(os.path.join(out, "gene_expression.npy"), np.concatenate([cur_expr, new_expr], axis=0))
    # 基因名
    import shutil
    gn = os.path.join(src, "gene_names.txt")
    if os.path.exists(gn):
        shutil.copy(gn, os.path.join(out, "gene_names.txt"))
    # 追加特征（g{l1}_ref + 各 l{l2}_ref）
    def _append_feat(fname, new_arr):
        old_p = os.path.join(src, fname)
        new_arr = new_arr.astype(np.float32)
        if os.path.exists(old_p):
            old = np.load(old_p)
            combined = np.concatenate([old, new_arr], axis=0)
        else:
            combined = new_arr
        np.save(os.path.join(out, fname), combined)
    _append_feat(f"X_uni2_g{l1}_ref.npy", g)
    for r, v in l_all.items():
        l2 = int(round(r * 224))
        _append_feat(f"X_uni2_l{l2}_ref.npy", np.concatenate(v, axis=0).astype(np.float32))
    print(f"[add_empty] 完成: {out}（{len(new_meta)} 细胞 = 原 {len(cur_meta)} + 空 {n_empty}）", flush=True)


def _crop(image, col, row, size):
    h, w = image.shape[:2]
    half = size // 2
    left, top = col - half, row - half
    right, bottom = col + half, row + half
    pad_l, pad_t = max(0, -left), max(0, -top)
    pad_r, pad_b = max(0, right - w), max(0, bottom - h)
    left, top = max(0, left), max(0, top)
    tile = image[top:bottom, left:right]
    if pad_l or pad_t or pad_r or pad_b:
        tile = np.pad(tile, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode="edge")
    return tile


def _load_he(path):
    import tifffile
    im = tifffile.imread(path)
    if im.ndim == 2:
        im = np.stack([im] * 3, axis=-1)
    elif im.shape[0] == 3:
        im = im.transpose(1, 2, 0)
    return im


if __name__ == "__main__":
    main()
