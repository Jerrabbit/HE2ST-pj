"""DeepPT 忠实版特征提取：ResNet50(ImageNet V2) 提取每细胞 patch 特征（2048-d）。

官方 DeepPT（D:\\hest_data\\codes\\DeepPT\\11slide_processing）：
    utils_preprocessing.py Feature_Extraction:
        resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        → conv1..layer4 → avgpool → flatten → 2048-d（去 fc）
    main_processing.py: tile → Resize(224) → ImageNet mean/std 归一化 → batch 提取

本脚本按官方流程对每个细胞的 256×256 patch 提取 ResNet50 特征（per-cell 适配，
粒度与官方 bulk tile 相同，仅聚合对象从 tile 换成细胞）。

用法（远程 myenv1，GPU）：
    python scripts/extract_resnet50.py --rep 1
    python scripts/extract_resnet50.py --rep 2
输出：data/rep{rep}/X_resnet50.npy (N, 2048)
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description="ResNet50(ImageNet V2) 特征提取（DeepPT 忠实版）")
    p.add_argument("--rep", type=int, choices=[1, 2], required=True)
    p.add_argument("--ckpt", default=None,
                   help="官方 ResNet50_IMAGENET1K_V2.pt 路径（默认 torchvision IMAGENET1K_V2）")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_cells", type=int, default=None)
    args = p.parse_args()

    import torch
    import torchvision.transforms as T
    from torchvision.models import resnet50
    from PIL import Image
    import pandas as pd

    data_dir = os.path.expanduser(f"~/HE2ST-pj/data/rep{args.rep}")
    meta = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
    if args.max_cells:
        meta = meta.iloc[:args.max_cells]
    print(f"[ResNet50] rep{args.rep}: {len(meta)} cells", flush=True)

    # 官方 Feature_Extraction：ResNet50 conv1..layer4 → avgpool → flatten
    if args.ckpt:
        model = resnet50(weights=None)
        sd = torch.load(args.ckpt, map_location="cpu")
        if any(k.startswith("resnet.") for k in sd):
            # 官方 .pt 是 Feature_Extraction 模块 state_dict（key 带 resnet. 前缀），剥前缀
            sd = {k.replace("resnet.", "", 1): v for k, v in sd.items()}
        model.load_state_dict(sd)
        print(f"[ResNet50] 加载官方权重 {args.ckpt}", flush=True)
    else:
        from torchvision.models import ResNet50_Weights
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        print("[ResNet50] 使用 torchvision IMAGENET1K_V2 权重", flush=True)
    model = model.eval().to(args.device)

    # 官方预处理：Resize(224) + ImageNet 归一化
    pre = T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # DataLoader 多进程并行读图（集群网络存储上 27 万张 PNG 逐个 open 是 CPU/IO 瓶颈）
    class PatchDataset(torch.utils.data.Dataset):
        def __init__(self, paths):
            self.paths = paths
        def __getitem__(self, i):
            return pre(Image.open(self.paths[i]).convert("RGB"))
        def __len__(self):
            return len(self.paths)

    dl = torch.utils.data.DataLoader(
        PatchDataset(meta["patch_path"].tolist()),
        batch_size=args.batch_size, num_workers=8, pin_memory=True,
    )

    feats = []
    done = 0
    for x in dl:
        with torch.no_grad():
            x = x.to(args.device)
            # conv1..layer4 → avgpool → flatten（去 fc，2048-d）
            x = model.conv1(x); x = model.bn1(x); x = model.relu(x); x = model.maxpool(x)
            x = model.layer1(x); x = model.layer2(x); x = model.layer3(x); x = model.layer4(x)
            x = model.avgpool(x)
            f = torch.flatten(x, 1).cpu().numpy()
        feats.append(f)
        done += len(x)
        print(f"[ResNet50] 进度 {done}/{len(meta)}", flush=True)

    feats = np.concatenate(feats, axis=0).astype(np.float32)
    out = os.path.join(data_dir, "X_resnet50.npy")
    np.save(out, feats)
    print(f"[ResNet50] 已保存 {out} 形状 {feats.shape}", flush=True)


if __name__ == "__main__":
    main()
