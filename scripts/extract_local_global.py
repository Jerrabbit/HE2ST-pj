"""UNI2+MLP Local-Global 双尺度特征提取（Global / Local 两阶段）。

Global branch（op1 放缩中心）：
    l1×l1 块（以细胞为中心）→ resize 224 → UNI2 → X_uni2_g{l1}.npy（1536 维/细胞）
Local branch（op2 中心裁剪）：
    l1×l1 块 → resize 224 → 中心裁剪 l2×l2 → resize 224 → UNI2 → X_uni2_l{l2}.npy

l1 从 >224 逐步缩小（op1 sweep）；l2 为 14 的倍数（op2 sweep，固定 best l1）。
UNI2 为冻结编码器（权重 /cpfs01/.../HE2ST/uni2_model/pytorch_model.bin）。

用法（远程 myenv1）：
    # Global 特征（某个 l1）
    python scripts/extract_local_global.py --rep 2 --stage global --l1 512 \
        --output data/rep2/X_uni2_g512.npy --workers 4
    # Local 特征（固定 best l1，某个 l2）
    python scripts/extract_local_global.py --rep 2 --stage local --l1 512 --l2 56 \
        --output data/rep2/X_uni2_l56.npy --workers 4
调试：--max_cells 200 先验证形状。
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根目录

import numpy as np
import pandas as pd
import torch
from PIL import Image

from common.data.preprocess import load_he_image

BASE = "/cpfs01/projects-HDD/cfff-d7ff0c9cdf2f_HDD/hjr_24300980068/HE2ST/datasets"
REP_HE = {
    1: os.path.join(BASE, "he_images", "Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.ome.tif"),
    2: os.path.join(BASE, "he_images", "Xenium_FFPE_Human_Breast_Cancer_Rep2_he_image.ome.tif"),
}
UNI2_WEIGHTS = "/cpfs01/projects-HDD/cfff-d7ff0c9cdf2f_HDD/hjr_24300980068/HE2ST/uni2_model/pytorch_model.bin"


def _crop_centered(image: np.ndarray, col: int, row: int, size: int) -> np.ndarray:
    """以 (col, row) 为中心裁 size×size 块，越界用边缘复制填充。"""
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


def _crop_global(image: np.ndarray, col: int, row: int, l1: int) -> np.ndarray:
    """Global：l1×l1 块（UNI2 提取时内部 resize 224）。"""
    return _crop_centered(image, col, row, l1)


def _crop_local(image: np.ndarray, col: int, row: int, l1: int, l2: int) -> np.ndarray:
    """Local：l1 块 → resize 224 → 中心裁剪 l2×l2（UNI2 提取时内部 resize 224）。"""
    block = _crop_centered(image, col, row, l1)
    block224 = np.array(Image.fromarray(block).resize((224, 224), Image.BILINEAR))
    half = l2 // 2
    c = 224 // 2
    return block224[c - half:c + half, c - half:c + half]


# --------------------------------------------------------------------------- #
# 并行 crop+resize（fork 继承整片图，与 extract_ctranspath_context.py 相同模式）
# --------------------------------------------------------------------------- #
_IMG, _CTR, _L1, _L2, _STAGE = None, None, None, None, None


def _worker(idx):
    if _STAGE == "local":
        return [_crop_local(_IMG, c, r, _L1, _L2) for c, r in _CTR[idx]]
    return [_crop_global(_IMG, c, r, _L1) for c, r in _CTR[idx]]


def main() -> None:
    p = argparse.ArgumentParser(description="UNI2+MLP Local-Global 特征提取")
    p.add_argument("--rep", type=int, choices=[1, 2], required=True)
    p.add_argument("--stage", choices=["global", "local"], required=True)
    p.add_argument("--l1", type=int, default=512, help="Global 块边长（放缩中心）")
    p.add_argument("--l2", type=int, default=56, help="Local 中心裁剪边长（14 的倍数）")
    p.add_argument("--ckpt", default=UNI2_WEIGHTS, help="UNI2 权重路径")
    p.add_argument("--output", default=None, help="输出 .npy 路径")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4, help="crop+resize 并行进程数（fork）")
    p.add_argument("--max_cells", type=int, default=None, help="调试：限制细胞数")
    args = p.parse_args()

    data_dir = os.path.expanduser(f"~/HE2ST-pj/data/rep{args.rep}")
    meta = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
    if args.max_cells:
        meta = meta.iloc[: args.max_cells]
    centers = meta[["x_centroid", "y_centroid"]].values.astype(int)
    print(f"[LG] {args.stage}: {len(centers)} cells, l1={args.l1}, l2={args.l2}", flush=True)

    image = load_he_image(REP_HE[args.rep])
    print(f"[LG] 图像 {image.shape}", flush=True)

    from common.features.uni2 import UNI2FeatureExtractor
    extractor = UNI2FeatureExtractor(args.ckpt, device=args.device)

    # fork 前设置全局（子进程继承，避免 pickle 2GB 图）
    global _IMG, _CTR, _L1, _L2, _STAGE
    _IMG, _CTR, _L1, _L2, _STAGE = image, centers, args.l1, args.l2, args.stage
    import multiprocessing as mp
    mp_ctx = mp.get_context("fork")
    pool = mp_ctx.Pool(args.workers) if args.workers > 1 else None

    CHUNK = 4096
    feats_all = []
    for s in range(0, len(centers), CHUNK):
        idx = np.arange(s, min(s + CHUNK, len(centers)))
        if pool is not None:
            sub = max(1, len(idx) // (args.workers * 2))
            tiles = [t for c in pool.map(_worker,
                     [idx[i:i + sub] for i in range(0, len(idx), sub)]) for t in c]
        else:
            tiles = _worker(idx)
        arr = np.stack(tiles)
        feats_all.append(extractor.extract(arr, batch_size=args.batch_size))
        done = min(s + CHUNK, len(centers))
        print(f"[LG] 进度 {done}/{len(centers)}", flush=True)

    if pool is not None:
        pool.close(); pool.join()
    feats = np.concatenate(feats_all, axis=0).astype(np.float32)
    out = args.output or os.path.join(
        data_dir, f"X_uni2_{'g' if args.stage == 'global' else 'l'}"
                 f"{args.l1 if args.stage == 'global' else args.l2}.npy")
    np.save(out, feats)
    print(f"[LG] 已保存 {out} 形状 {feats.shape}", flush=True)


if __name__ == "__main__":
    main()
