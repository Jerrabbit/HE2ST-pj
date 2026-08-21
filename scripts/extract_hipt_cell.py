"""Pixel2Gene 方案 B（cell-level）：level-1 ViT-256 特征提取 → X_hipt_cell.npy。

只用 HIPT 的 level-1 编码器（vit_256_small_dino），每细胞自己的 256×256 patch →
[CLS] 384 维特征（真正 per-cell，无 spot 内封顶）。归一化与官方一致
（ToTensor + Normalize(0.5,0.5)，即 HIPT eval_transforms）。

用法（远程服务器）：
    python scripts/extract_hipt_cell.py --rep 2 \
        --ckpt ~/HE2ST-pj/weights/pixel2gene/vit_256_small_dino.pth
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.expanduser("~/HE2ST-pj/methods/pixel2gene/hipt"))

from hipt_model_utils import get_vit256  # noqa: E402

CKPT = os.path.expanduser("~/HE2ST-pj/weights/pixel2gene/vit_256_small_dino.pth")
CLS_DIM = 384  # ViT-256 [CLS] 维度


def main() -> None:
    p = argparse.ArgumentParser(description="ViT-256 cell-level 特征提取")
    p.add_argument("--rep", type=int, choices=[1, 2], required=True)
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--max_cells", type=int, default=None, help="调试：限制细胞数")
    args = p.parse_args()

    model = get_vit256(pretrained_weights=args.ckpt).to(args.device)
    model.eval()
    print(f"[HIPT-cell] ViT-256 加载成功", flush=True)

    data_dir = os.path.expanduser(f"~/HE2ST-pj/data/rep{args.rep}")
    meta = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
    if args.max_cells:
        meta = meta.iloc[:args.max_cells]
    n = len(meta)
    print(f"[HIPT-cell] 提取 {n} 个细胞 ...", flush=True)

    tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # 并行读 PNG（DataLoader num_workers>0）：顺序 Image.open 在 cpfs HDD 上是 IO 瓶颈
    # （实测 256 batch 顺序读 ~2s/张，2.5h 才 4096 cell；8 workers 可 ~8× 加速）。
    from torch.utils.data import DataLoader, Dataset

    class PatchDataset(Dataset):
        def __init__(self, paths):
            self.paths = list(paths)

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            return tf(Image.open(self.paths[i]).convert("RGB"))

    dl = DataLoader(PatchDataset(meta["patch_path"].tolist()),
                    batch_size=args.batch_size, shuffle=False,
                    num_workers=8, pin_memory=True, drop_last=False)
    feats = np.zeros((n, CLS_DIM), dtype=np.float32)
    done = 0
    for x in dl:
        x = x.to(args.device)                           # (B,3,256,256)
        with torch.no_grad():
            fea_all = model.forward_all(x)              # (B, 257, 384)
        feats[done:done + x.size(0)] = fea_all[:, 0].cpu().numpy()  # [CLS] token
        done += x.size(0)
        if done % 4096 == 0:
            print(f"  {done}/{n}", flush=True)

    out = os.path.join(data_dir, "X_hipt_cell.npy")
    np.save(out, feats)
    print(f"[HIPT-cell] 已保存 {out} 形状 {feats.shape}", flush=True)


if __name__ == "__main__":
    main()
