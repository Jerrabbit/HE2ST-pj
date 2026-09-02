"""忠实复刻 img_feature_extractor 参考的 LG 特征提取。

与 extract_local_global.py 的区别：Local 特征取 **intermediates[-1]**（forward_intermediates
最后一层中间 NCHW patch 特征图）按参考 center 公式裁剪 + LayerNorm + mean-pool；Global =
feature_emb[:,0]（CLS）LayerNorm。全部经 common.features.uni2.extract_reference 实现。

产出（data_dir/）：
    X_uni2_g{l1}_ref.npy   Global CLS（LN 后）
    X_uni2_l{l2}_ref.npy   Local 中心裁剪 mean-pool（LN 后），每个 l2 一个文件
一次 forward 复用 intermediates → 全部 l2 免费。

用法（远程 myenv1）：
    python3 scripts/extract_faithful.py --rep 1 --data_dir data/rep1 \
        --l1 112 --l2_list 28 42 56 70 84 98 112 --device cuda
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from common.data.preprocess import load_he_image

BASE = "/cpfs01/projects-HDD/cfff-d7ff0c9cdf2f_HDD/hjr_24300980068/HE2ST/datasets"
REP_HE = {
    1: os.path.join(BASE, "he_images", "Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.ome.tif"),
    2: os.path.join(BASE, "he_images", "Xenium_FFPE_Human_Breast_Cancer_Rep2_he_image.ome.tif"),
}
UNI2_WEIGHTS = "/cpfs01/projects-HDD/cfff-d7ff0c9cdf2f_HDD/hjr_24300980068/HE2ST/uni2_model/pytorch_model.bin"

_IMG, _CTR, _L1 = None, None, None


def _crop_centered(image, col, row, size):
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


def _worker(idx):
    return [_crop_centered(_IMG, c, r, _L1) for c, r in _CTR[idx]]


def main():
    p = argparse.ArgumentParser(description="忠实复刻参考的 LG 特征提取")
    p.add_argument("--rep", type=int, choices=[1, 2], required=True)
    p.add_argument("--data_dir", default=None)
    p.add_argument("--l1", type=int, default=112)
    p.add_argument("--l2_list", nargs="+", type=int, default=[56])
    p.add_argument("--ckpt", default=UNI2_WEIGHTS)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max_cells", type=int, default=None)
    args = p.parse_args()

    data_dir = args.data_dir or os.path.expanduser(f"~/HE2ST-pj/data/rep{args.rep}")
    meta = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
    if args.max_cells:
        meta = meta.iloc[: args.max_cells]
    centers = meta[["x_centroid", "y_centroid"]].values.astype(int)
    print(f"[faithful] rep{args.rep}: {len(centers)} cells, l1={args.l1}, l2_list={args.l2_list}", flush=True)

    image = load_he_image(REP_HE[args.rep])
    from common.features.uni2 import UNI2FeatureExtractor
    extractor = UNI2FeatureExtractor(args.ckpt, device=args.device)

    global _IMG, _CTR, _L1
    _IMG, _CTR, _L1 = image, centers, args.l1
    import multiprocessing as mp
    pool = mp.get_context("fork").Pool(args.workers) if args.workers > 1 else None

    center_ratios = [l2 / 224.0 for l2 in args.l2_list]
    g_all = []
    l_all = {r: [] for r in center_ratios}
    CHUNK = 4096
    for s in range(0, len(centers), CHUNK):
        idx = np.arange(s, min(s + CHUNK, len(centers)))
        if pool is not None:
            sub = max(1, len(idx) // (args.workers * 2))
            tiles = [t for c in pool.map(_worker,
                     [idx[i:i + sub] for i in range(0, len(idx), sub)]) for t in c]
        else:
            tiles = _worker(idx)
        arr = np.stack(tiles)
        g, loc = extractor.extract_reference(arr, center_ratios, args.batch_size,
                                             layer_norm=True)
        g_all.append(g)
        for r, v in loc.items():
            l_all[r].append(v)
        print(f"[faithful] 进度 {min(s + CHUNK, len(centers))}/{len(centers)}", flush=True)
    if pool is not None:
        pool.close(); pool.join()

    g = np.concatenate(g_all, axis=0).astype(np.float32)
    np.save(os.path.join(data_dir, f"X_uni2_g{args.l1}_ref.npy"), g)
    print(f"[faithful] 已保存 Global {g.shape}", flush=True)
    for r, v in l_all.items():
        feats = np.concatenate(v, axis=0).astype(np.float32)
        l2 = int(round(r * 224))
        np.save(os.path.join(data_dir, f"X_uni2_l{l2}_ref.npy"), feats)
        print(f"[faithful] 已保存 Local l2={l2} {feats.shape}", flush=True)


if __name__ == "__main__":
    main()
