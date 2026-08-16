"""SQUALL 特征提取 → data_dir/X_squall.npy（1024 维/细胞）。

冻结 SQUALL_full.pth，per-cell 256×256 patch → resize 224×224（保留 0-255 原始值，
与官方教程 io.imread 一致）→ forward_rgb → (B, 196, 1024) token 嵌入 → mean-pool → (B, 1024)。

用法（远程服务器）：
    python scripts/extract_squall.py --rep 2 \
        --ckpt ~/HE2ST-pj/weights/squall/SQUALL_full.pth
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, os.path.expanduser("~/HE2ST-pj"))  # methods.squall 可导入

CONFIG = os.path.expanduser("~/HE2ST-pj/codes/squall/SQUALL_Tutorial/config.yaml")
CKPT = os.path.expanduser("~/HE2ST-pj/weights/squall/SQUALL_full.pth")
RES = 0.5          # 分辨率（官方教程用 0.5）
IMG_SIZE = 224     # SQUALL img_size
FEAT_DIM = 1024


def load_squall(ckpt_path: str, device: str = "cuda"):
    """构建 Squall 并加载冻结权重（strict=True，架构必须与 checkpoint 一致）。"""
    import yaml

    from methods.squall.Squall import Squall

    class AttrDict(dict):
        def __getattr__(self, k):
            try:
                return self[k]
            except KeyError:
                raise AttributeError(k)

    with open(CONFIG) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model = Squall(AttrDict(config["model"]))
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    print(f"[SQUALL] 冻结模型加载成功, params="
          f"{sum(p.numel() for p in model.parameters())/1e6:.0f}M", flush=True)
    return model


def main() -> None:
    p = argparse.ArgumentParser(description="SQUALL 特征提取")
    p.add_argument("--rep", type=int, choices=[1, 2], required=True)
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_cells", type=int, default=None, help="调试：限制细胞数")
    args = p.parse_args()

    model = load_squall(args.ckpt, args.device)
    data_dir = os.path.expanduser(f"~/HE2ST-pj/data/rep{args.rep}")
    meta = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
    if args.max_cells:
        meta = meta.iloc[:args.max_cells]
    n = len(meta)
    print(f"[SQUALL] 提取 {n} 个细胞 ...", flush=True)

    feats = np.zeros((n, FEAT_DIM), dtype=np.float32)
    for i in range(0, n, args.batch_size):
        j0, j1 = i, min(i + args.batch_size, n)
        B = j1 - j0
        batch = []
        for j in range(j0, j1):
            img = Image.open(meta.iloc[j]["patch_path"]).convert("RGB")
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            # 官方教程喂 rgb/255（0-1 归一化）；0-255 直接喂会错位特征
            batch.append(torch.from_numpy(np.asarray(img, dtype=np.float32)
                                          / 255.0).permute(2, 0, 1))
        x = torch.stack(batch).to(args.device)                     # (B,3,224,224) 0-1
        res = torch.full((B, 1), RES, dtype=torch.float32, device=args.device)
        with torch.no_grad():
            z = model.forward_rgb(x, res)                           # (B,196,1024)
            feats[j0:j1] = z.mean(1).cpu().numpy()
        if ((i // args.batch_size) % 100) == 0:
            print(f"  {j1}/{n}", flush=True)

    out = os.path.join(data_dir, "X_squall.npy")
    np.save(out, feats)
    print(f"[SQUALL] 已保存 {out} 形状 {feats.shape}", flush=True)


if __name__ == "__main__":
    main()
