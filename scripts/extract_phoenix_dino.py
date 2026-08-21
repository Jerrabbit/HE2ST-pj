"""预提取 DINOv2 patch token（Phoenix 微调用）：X_phoenix_dino.npy (N, 261, 1536) fp16。

DINOv2 冻结，token 与细胞一一对应 → 预提取一次，微调 flow 时直接读缓存，
避免每 epoch 重复 1.1B 前向（否则 50ep 需 20+ 小时）。

用法（远程）：
    python scripts/extract_phoenix_dino.py --rep 1 --output data/rep1/X_phoenix_dino.npy
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根目录


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DINOv2 token 预提取")
    p.add_argument("--rep", type=int, choices=[1, 2], required=True)
    p.add_argument("--flow_weights", default="methods/phoenix/flow_model.pth")
    p.add_argument("--output", default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_cells", type=int, default=None)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    from methods.phoenix.official import IMG_MEAN, IMG_STD, build_dino

    args = parse_args()
    device = args.device
    data_dir = os.path.expanduser(f"~/HE2ST-pj/data/rep{args.rep}")
    meta = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
    if args.max_cells:
        meta = meta.iloc[: args.max_cells]
    n = len(meta)
    out = args.output or os.path.join(data_dir, "X_phoenix_dino.npy")

    tf = transforms.Compose([
        transforms.Resize((224, 224), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(tuple(IMG_MEAN[0, :, 0, 0].tolist()),
                             tuple(IMG_STD[0, :, 0, 0].tolist())),
    ])

    class PatchDS(Dataset):
        def __init__(self, paths):
            self.paths = list(paths)

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            return tf(Image.open(self.paths[i]).convert("RGB"))

    dino = build_dino(args.flow_weights, device)   # 冻结，从 flow_model.pth 加载
    print(f"[DINO] {n} cells → {out}", flush=True)

    dl = DataLoader(PatchDS(meta["patch_path"].tolist()), batch_size=args.batch_size,
                    shuffle=False, num_workers=8, pin_memory=True)
    # fp16 mmap 写出（261 token × 1536 × 2 bytes/cell）
    feat = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16,
                                     shape=(n, 261, 1536))
    done = 0
    with torch.no_grad():
        for x in dl:
            c = dino.forward_features(x.to(device)).cpu().numpy().astype(np.float16)
            feat[done:done + c.shape[0]] = c
            done += c.shape[0]
            if done % 8192 == 0:
                print(f"[DINO] {done}/{n}", flush=True)
    feat.flush()
    print(f"[DINO] 已保存 {out} 形状 ({n}, 261, 1536) fp16", flush=True)


if __name__ == "__main__":
    main()
