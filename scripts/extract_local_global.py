"""UNI2+MLP Local-Global 双尺度特征提取（**每细胞单次 forward**）。

Global branch（op1 放缩中心）：
    l1×l1 块（以细胞为中心）→ resize 224 → UNI2 [CLS] → X_uni2_g{l1}.npy（1536 维/细胞）

Local branch（op2 中心裁剪，**复用同一次 forward 的 patch token**）：
    l1×l1 块 → resize 224 → UNI2 全部 token（16×16 patch 网格）
    → 中心裁剪 l2×l2 对应 patch 网格中心 k×k 子块（k = l2/14）
    → mean-pool → X_uni2_l{l2}.npy（1536 维/细胞）

UNI2 为 patch14 无重叠：224×224 一次 forward 出 16×16 = 256 个 patch token，
中心裁剪（l2 为 14 的倍数）对应正是该网格中心子块，**无需二次 forward**。
`--stage local` 单次提取即可产出 `--l2_list` 内全部 l2 特征文件（op2 sweep 免费复用）。

l1 从 >224 逐步缩小（op1 sweep）；l2 = 4..8 × 14（op2 sweep，固定 best l1）。
UNI2 为冻结编码器（权重 /cpfs01/.../HE2ST/uni2_model/pytorch_model.bin）。

用法（远程 myenv1）：
    # Global 特征（某个 l1）
    python scripts/extract_local_global.py --rep 1 --stage global --l1 512 \
        --output data/rep1/X_uni2_g512.npy --workers 4
    # Local 特征（固定 best l1，单次 forward 产出全部 l2 文件）
    python scripts/extract_local_global.py --rep 1 --stage local --l1 336 \
        --l2_list 56 70 84 98 112 --workers 4
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

PATCH_GRID = 16  # 224/14
MAX_K = 8        # 中心 8×8 patch token（对应 l2 ≤ 112）


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
    """Global/Local 共用：l1×l1 块（UNI2 提取时内部 resize 224）。"""
    return _crop_centered(image, col, row, l1)


# --------------------------------------------------------------------------- #
# 并行 crop+resize（fork 继承整片图，与 extract_ctranspath_context.py 相同模式）
# --------------------------------------------------------------------------- #
_IMG, _CTR, _L1 = None, None, None


def _worker(idx):
    return [_crop_global(_IMG, c, r, _L1) for c, r in _CTR[idx]]


def main() -> None:
    p = argparse.ArgumentParser(description="UNI2+MLP Local-Global 特征提取（单次 forward）")
    p.add_argument("--rep", type=int, choices=[1, 2], required=True)
    p.add_argument("--data_dir", default=None,
                   help="数据目录（含 metadata.csv），默认 ~/HE2ST-pj/data/rep{N}。"
                        "过滤后数据（rep1_f/rep2_f）须显式传入，否则会按未过滤细胞集提取导致特征错位。")
    p.add_argument("--stage", choices=["global", "local"], required=True,
                   help="global=CLS 特征；local=中心 k×k patch token mean（复用单次 forward）")
    p.add_argument("--l1", type=int, default=512, help="Global 块边长（放缩中心）")
    p.add_argument("--l2", type=int, default=56, help="Local 中心裁剪边长（14 的倍数，单 l2 时用）")
    p.add_argument("--l2_list", nargs="+", type=int, default=None,
                   help="一次提取产出多个 l2 特征文件（op2 sweep 用）")
    p.add_argument("--ckpt", default=UNI2_WEIGHTS, help="UNI2 权重路径")
    p.add_argument("--output", default=None, help="仅 global 阶段：输出 .npy 路径")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--workers", type=int, default=4, help="crop+resize 并行进程数（fork）")
    p.add_argument("--max_cells", type=int, default=None, help="调试：限制细胞数")
    p.add_argument("--layernorm", action="store_true",
                   help="特征提取加 LayerNorm（参考 img_feature_extractor：CLS 与中心 patch "
                        "token 各做 LayerNorm(1536, eps=1e-6)）")
    args = p.parse_args()

    data_dir = args.data_dir or os.path.expanduser(f"~/HE2ST-pj/data/rep{args.rep}")
    meta = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
    if args.max_cells:
        meta = meta.iloc[: args.max_cells]
    centers = meta[["x_centroid", "y_centroid"]].values.astype(int)
    print(f"[LG] {args.stage}: {len(centers)} cells, l1={args.l1}", flush=True)

    image = load_he_image(REP_HE[args.rep])
    print(f"[LG] 图像 {image.shape}", flush=True)

    from common.features.uni2 import UNI2FeatureExtractor
    extractor = UNI2FeatureExtractor(args.ckpt, device=args.device,
                                     layer_norm=args.layernorm)

    # fork 前设置全局（子进程继承，避免 pickle 2GB 图）
    global _IMG, _CTR, _L1
    _IMG, _CTR, _L1 = image, centers, args.l1
    import multiprocessing as mp
    mp_ctx = mp.get_context("fork")
    pool = mp_ctx.Pool(args.workers) if args.workers > 1 else None

    if args.stage == "local":
        # l2 校验：14 的倍数、k≤8（中心 8×8 覆盖全部 sweep 值）
        l2s = args.l2_list or [args.l2]
        ks = []
        for l2 in l2s:
            if l2 % 14:
                raise SystemExit(f"l2={l2} 必须是 14 的倍数（patch14 对齐）")
            if l2 // 14 > MAX_K:
                raise SystemExit(f"l2={l2} 超过中心 8×8 上限（l2≤112）")
            ks.append(l2 // 14)
        print(f"[LG] local l2s={l2s} (k={ks})", flush=True)

    CHUNK = 4096
    if args.stage == "global":
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
        out = args.output or os.path.join(data_dir, f"X_uni2_g{args.l1}.npy")
        np.save(out, feats)
        print(f"[LG] 已保存 {out} 形状 {feats.shape}", flush=True)
    else:
        # 单次 forward：复用 patch token 网格，一次产出全部 l2 文件
        accum = {l2: [] for l2 in l2s}
        for s in range(0, len(centers), CHUNK):
            idx = np.arange(s, min(s + CHUNK, len(centers)))
            if pool is not None:
                sub = max(1, len(idx) // (args.workers * 2))
                tiles = [t for c in pool.map(_worker,
                         [idx[i:i + sub] for i in range(0, len(idx), sub)]) for t in c]
            else:
                tiles = _worker(idx)
            arr = np.stack(tiles)
            toks = extractor.extract_tokens(arr, batch_size=args.batch_size)  # (B,265,1536)
            for l2, k in zip(l2s, ks):
                center = extractor.center_patch_tokens(toks, k)   # (B, k*k, 1536)
                accum[l2].append(center.mean(1))                  # (B, 1536) mean-pool
            done = min(s + CHUNK, len(centers))
            print(f"[LG] 进度 {done}/{len(centers)}", flush=True)
        if pool is not None:
            pool.close(); pool.join()
        for l2 in l2s:
            feats = np.concatenate(accum[l2], axis=0).astype(np.float32)
            out = os.path.join(data_dir, f"X_uni2_l{l2}.npy")
            np.save(out, feats)
            print(f"[LG] 已保存 {out} 形状 {feats.shape}", flush=True)


if __name__ == "__main__":
    main()
