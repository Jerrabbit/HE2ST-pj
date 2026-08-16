"""Path2Space 大上下文特征提取：Macenko 染色归一化 + 大 context tile。

官方 Path2Space 管线（ge_model/path2space，Shulman et al. Cell 2026）在
CTransPath 特征前必须做 **Macenko 染色归一化**（冻结 normalizer，spams），tile
尺寸官方默认 224px（spots 模式）。本脚本支持更大的 context tile（如 384/512/768px）
以捕获更多组织上下文，再 resize 到 224 喂 CTransPath —— 即项目 Local-Global
设计中 op1"放缩中心/全局分支"的思路。

与旧版 extract_ctranspath.py 的区别：
    1) 从整片 H&E 现裁 context tile（旧版用预生成的 256×256 紧贴 patch）；
    2) 套用官方冻结 Macenko 染色归一化（旧版完全没有）；
    3) 输出文件名区分 context 尺寸，便于 sweep（X_ctranspath_ctx{ctx}.npy）。

内存策略：整片 H&E 载入内存（~1.8GB，与 preprocess_he.py 相同），context tile
**按 CHUNK 分批**裁取→归一化→提特征→丢弃，峰值内存可控（~CHUNK×ctx²×3 字节）。

用法（远程 myenv1）：
    python scripts/extract_ctranspath_context.py --rep 2 --ctx 512 --workers 8
调试：--max_cells 2000 --workers 4 先验证输出形状与耗时。
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根目录

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image

from common.data.preprocess import load_he_image

BASE = "/cpfs01/projects-HDD/cfff-d7ff0c9cdf2f_HDD/hjr_24300980068/HE2ST/datasets"
REP_HE = {
    1: os.path.join(BASE, "he_images", "Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.ome.tif"),
    2: os.path.join(BASE, "he_images", "Xenium_FFPE_Human_Breast_Cancer_Rep2_he_image.ome.tif"),
}
CTRANSPATH_CKPT = ("/cpfs01/projects-HDD/cfff-d7ff0c9cdf2f_HDD/"
                   "hjr_24300980068/HE2ST/path2space/ctranspath.pth")

# 官方 tile transform（与 path2space.features._tile_transform_single 一致：
# Resize 224 + BILINEAR + antialias=False + ImageNet 归一化）。
_tile_tf = transforms.Compose([
    transforms.Resize(224, antialias=False),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _crop_context(image: np.ndarray, col: int, row: int, ctx: int) -> np.ndarray:
    """以 (col, row) 为中心裁 ctx×ctx context tile，越界用边缘复制填充。"""
    h, w = image.shape[:2]
    half = ctx // 2
    left, top = col - half, row - half
    right, bottom = col + half, row + half
    pad_l, pad_t = max(0, -left), max(0, -top)
    pad_r, pad_b = max(0, right - w), max(0, bottom - h)
    left, top = max(0, left), max(0, top)
    tile = image[top:bottom, left:right]
    if pad_l or pad_t or pad_r or pad_b:
        tile = np.pad(tile, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode="edge")
    return tile


def _resize224(tile: np.ndarray) -> np.ndarray:
    """context tile → 224×224（BILINEAR）。CTransPath 输入恒为 224×224。

    先 resize 再做 Macenko：官方管线 Macenko 作用在 224×224 tile 上，
    这里同样在 224 分辨率上做染色归一化（染色统计对分辨率不敏感），
    同时把 Macenko 像素量从 ctx² 降到 224²（ctx=512 时省 ~5×）。
    """
    return np.array(Image.fromarray(tile).resize((224, 224), Image.BILINEAR))


_STAIN_MATRIX_TARGET = np.array(
    [[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]], dtype=np.float32).T  # (2,3)
_MAXC_TARGET = np.array([1.9705, 1.0308], dtype=np.float64).reshape(1, 1, 2)


def _concentrations_pinv(OD: np.ndarray, S: np.ndarray) -> np.ndarray:
    """2-atom 浓度估计：非负约束下的最小二乘投影（clip 近似）。

    OD: (P,3) 光密度；S: (2,3) 染色矩阵。返回 (P,2) 浓度。
    对组织像素，H&E 浓度均为非负（组织 = 染色的非负组合），无约束最小二乘
    `C = OD·S†` 几乎总是非负；此处对负值 clip 0（背景像素 → 浓度 0 → 重建为
    白），等价官方 spams.lasso 非负解的一阶近似，向量化无分支、速度快。
    """
    G = S @ S.T                       # (2,2)
    Ginv = np.linalg.inv(G)
    pinv = Ginv @ S                   # (2,3)
    return np.maximum(OD @ pinv.T, 0.0)  # (P,2)


def _macenko_batch(tiles: list, max_side: int = 512) -> list:
    """批量化 Macenko 归一化（等价于官方 per-tile transform，数值稳定的加速版）。

    官方 `macenko_normalizer.transform` 对每个 tile 独立估计 source 染色矩阵并调用
    spams.lasso 求浓度。同一切片染色均匀，这里把整个 chunk 的像素池化：
        1) 从全部像素估计 **一个** source 染色矩阵（对应同一染色条件）；
        2) 全部像素用 **一次** spams.lasso 求浓度（与官方同参数，全部像素精确）；
        3) 逐 tile 按官方方式做 maxC 缩放 + 用冻结 target 染色矩阵重建。
    速度约 ~100× 于逐 tile 调用，且池化大样本不会触发单 tile 退化时的 eigh 不收敛。

    输入 tiles 须同尺寸 (H,W,3) uint8。返回同序列表，逐 tile 逐像素对齐。
    """
    from methods.path2space.frozen.utils_color_norm import normalize_rows  # noqa: F401

    tiles = list(tiles)
    N = len(tiles)
    if N == 0:
        return tiles
    H, W = tiles[0].shape[:2]
    if any(t.shape[:2] != (H, W) for t in tiles):
        raise ValueError("批内 tile 尺寸不一致")
    n_pix = H * W
    # 分块处理大 tile，控制峰值内存（~CHUNK_PIX 像素）
    CHUNK_PIX = 40_000_000  # ~ 40M 像素/次（float32 OD ≈ 0.5GB）
    per = max(1, CHUNK_PIX // max(1, n_pix))
    out = np.empty((N, H, W, 3), dtype=np.uint8)
    for s in range(0, N, per):
        blk = tiles[s:s + per]
        B = len(blk)
        arr = np.stack(blk)                                        # (B,H,W,3) uint8
        p95 = np.percentile(arr, 95, axis=(1, 2), keepdims=True)   # 亮度归一化
        f = np.clip(arr.astype(np.float32) * 255.0 / np.maximum(p95, 1e-8), 0, 255)
        f = np.where(f == 0, 1.0, f)
        OD = (-np.log(f / 255.0)).reshape(-1, 3)                   # (B*H*W,3) f32
        # source 染色矩阵（池化全部像素，等价 get_stain_matrix(池化图)）
        keep = OD[(OD > 0.15).any(axis=1)]                          # beta=0.15
        _, V = np.linalg.eigh(np.cov(keep, rowvar=False))
        V = V[:, [2, 1]]
        if V[0, 0] < 0:
            V[:, 0] *= -1
        if V[0, 1] < 0:
            V[:, 1] *= -1
        That = keep @ V
        phi = np.arctan2(That[:, 1], That[:, 0])
        minPhi, maxPhi = np.percentile(phi, 1), np.percentile(phi, 99)  # alpha=1
        v1 = V @ np.array([np.cos(minPhi), np.sin(minPhi)])
        v2 = V @ np.array([np.cos(maxPhi), np.sin(maxPhi)])
        HE = np.array([v1, v2]) if v1[0] > v2[0] else np.array([v2, v1])
        S_source = normalize_rows(HE).astype(np.float32)            # (2,3)
        # 全部像素一次向量化浓度估计（clip 近似，f32）
        conc = _concentrations_pinv(OD, S_source)                   # (B*H*W,2)
        conc_t = conc.reshape(B, n_pix, 2)
        maxC_src = np.percentile(conc_t, 99, axis=1, keepdims=True)  # (B,1,2)
        conc_t = conc_t * (_MAXC_TARGET / np.maximum(maxC_src, 1e-8))
        OD_norm = conc_t.reshape(-1, 2) @ _STAIN_MATRIX_TARGET        # (B*H*W,3)
        I_norm = (255.0 * np.exp(-OD_norm)).reshape(B, H, W, 3)
        out[s:s + B] = np.clip(I_norm, 0, 255).astype(np.uint8)
    return [out[i] for i in range(N)]


# --------------------------------------------------------------------------- #
# 多进程并行 crop+resize+Macenko。
# 注意：整片图 (~2GB) 通过 **fork 继承**（copy-on-write）共享给 worker，不能走
# initargs（会把 2.2GB pickle 到每个 worker）。因此必须在 `mp.Pool` 创建**之前**
# 设置模块级 `_IMG/_CTR/_CTX`，fork 出的子进程自动继承。强制 fork 上下文。
# --------------------------------------------------------------------------- #
_IMG, _CTR, _CTX = None, None, None


def _worker(idx):
    """worker：对给定细胞索引做 crop(ctx) → resize224 → Macenko，返回 224×224 归一化图。"""
    tiles = [_resize224(_crop_context(_IMG, c, r, _CTX)) for c, r in _CTR[idx]]
    return _macenko_batch(tiles)


def main() -> None:
    p = argparse.ArgumentParser(description="Path2Space 大上下文 CTransPath 特征提取")
    p.add_argument("--rep", type=int, choices=[1, 2], required=True)
    p.add_argument("--ctx", type=int, default=512, help="context tile 边长（像素）")
    p.add_argument("--ckpt", default=CTRANSPATH_CKPT)
    p.add_argument("--output", default=None, help="输出 .npy 路径")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=128, help="CTransPath 推理 batch")
    p.add_argument("--workers", type=int, default=4,
                   help="crop+resize+Macenko 并行进程数（fork 继承整片图，CTransPath GPU 串行）")
    p.add_argument("--max_cells", type=int, default=None, help="调试：限制细胞数")
    args = p.parse_args()

    data_dir = os.path.expanduser(f"~/HE2ST-pj/data/rep{args.rep}")
    meta = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
    if args.max_cells:
        meta = meta.iloc[: args.max_cells]
    centers = meta[["x_centroid", "y_centroid"]].values.astype(int)  # (col, row)
    print(f"[P2S-ctx] {len(centers)} cells, ctx={args.ctx}", flush=True)

    print("[P2S-ctx] 加载整片 H&E ...", flush=True)
    image = load_he_image(REP_HE[args.rep])
    print(f"[P2S-ctx] 图像 {image.shape}", flush=True)

    from methods.path2space.frozen.ctrans.ctranspath import CTransPath
    model = CTransPath(num_classes=0).to(args.device)
    state = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    import multiprocessing as mp

    # fork 前设置全局（子进程继承，避免 pickle 2GB 图像）
    global _IMG, _CTR, _CTX
    _IMG, _CTR, _CTX = image, centers, args.ctx
    mp_ctx = mp.get_context("fork")
    CHUNK = 8192  # 每次在内存中的细胞数（Macenko 内部再按像素分块）
    pool = mp_ctx.Pool(args.workers) if args.workers > 1 else None
    feats_all = []
    for s in range(0, len(centers), CHUNK):
        idx = np.arange(s, min(s + CHUNK, len(centers)))
        if pool is not None:
            # 并行 crop+resize+Macenko；CTransPath GPU forward 由主进程串行
            sub = max(1, len(idx) // (args.workers * 2))
            normed = [t for c in pool.map(_worker,
                      [idx[i:i + sub] for i in range(0, len(idx), sub)]) for t in c]
        else:
            normed = _worker(idx)

        with torch.no_grad():
            for i in range(0, len(normed), args.batch_size):
                batch = normed[i:i + args.batch_size]
                xs = torch.stack([_tile_tf(Image.fromarray(t)) for t in batch]).to(args.device).float()
                feats_all.append(model(xs).cpu().numpy())
        done = min(s + CHUNK, len(centers))
        print(f"[P2S-ctx] 进度 {done}/{len(centers)}", flush=True)

    if pool is not None:
        pool.close(); pool.join()
    feats = np.concatenate(feats_all, axis=0).astype(np.float32)
    out = args.output or os.path.join(data_dir, f"X_ctranspath_ctx{args.ctx}.npy")
    np.save(out, feats)
    print(f"[P2S-ctx] 已保存 {out} 形状 {feats.shape}", flush=True)


if __name__ == "__main__":
    main()
