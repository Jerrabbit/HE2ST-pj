"""Phoenix 官方权重**零样本**评估（2026-09-03 重启，用于交集基因评测）。

流程：Rep2 256×256 patch → resize 224（bicubic）+ 官方 tissue 归一化
→ DINOv2 ViT-Giant 提 token → flow ODE 采样 → 官方 stats 反归一化（log1p）→ raw counts
→ 只对**与官方基因面板的交集基因**（xenium_human_breast 280 ⊂ 本仓库 313）计算指标
（gene_idx），其余基因填 0 不参与评测；含 SSIM（坐标栅格化）与全 Top-k 曲线。

用法（远程 myenv1）：
    python scripts/test_phoenix_zero_shot.py --test_dir data/rep2 \
        --output_dir outputs/bench_phoenix_zero_shot
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根目录


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phoenix 官方权重零样本评估")
    p.add_argument("--test_dir", required=True)
    p.add_argument("--flow_weights", default="methods/phoenix/flow_model.pth")
    p.add_argument("--dino_weights", default="methods/phoenix/pytorch_model.bin")
    p.add_argument("--gene_list", default="methods/phoenix/xenium_human_breast.npy")
    p.add_argument("--stats", default="methods/phoenix/nest_breast_stats_table.npz")
    p.add_argument("--n_sample_steps", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_cells", type=int, default=None)
    p.add_argument("--output_dir", default="outputs/bench_phoenix_zero_shot")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    import json

    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms

    from common.benchmark.harness import (
        compute_metrics_vectorized, load_gene_names, save_eval_results_csv)
    from methods.phoenix.official import IMG_MEAN, IMG_STD, PhoenixOfficial, denorm_to_raw

    args = parse_args()
    device = args.device

    # 官方推理图像变换（224 bicubic + tissue 归一化）
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

    meta = pd.read_csv(os.path.join(args.test_dir, "metadata.csv"))
    if args.max_cells:
        meta = meta.iloc[: args.max_cells]
    print(f"[Phoenix-zero] {len(meta)} cells, {args.n_sample_steps} 步", flush=True)

    # 官方基因集 + stats
    genes_off = [str(g) for g in np.load(args.gene_list, allow_pickle=True)]
    stats = np.load(args.stats)
    stats = {"mean": stats["mean"].astype(np.float32), "std": stats["std"].astype(np.float32)}
    assert len(genes_off) == stats["mean"].shape[0]

    # 本仓库 313 公共基因 → 官方基因集映射（覆盖子集）
    our_genes = load_gene_names(args.test_dir) or []
    name_to_off = {g: i for i, g in enumerate(genes_off)}
    covered = [(c, name_to_off[g]) for c, g in enumerate(our_genes) if g in name_to_off]
    print(f"[Phoenix-zero] 公共 {len(our_genes)} 基因中覆盖 {len(covered)}", flush=True)
    out_cols = np.array([c for c, _ in covered])
    off_cols = np.array([i for _, i in covered])

    model = PhoenixOfficial(num_genes=len(genes_off), flow_weights=args.flow_weights,
                            dino_weights=args.dino_weights, device=device,
                            n_sample_steps=args.n_sample_steps)
    model.eval()
    print(f"[Phoenix-zero] 模型就绪（DINOv2 + flow 官方权重）", flush=True)

    dl = DataLoader(PatchDS(meta["patch_path"].tolist()), batch_size=args.batch_size,
                    shuffle=False, num_workers=4, pin_memory=True)
    pred_log1p_all = []
    with torch.no_grad():
        for i, x in enumerate(dl):
            x = x.to(device)
            z = model(x)                       # (B, 280) log1p 空间
            pred_log1p_all.append(z.cpu().numpy())
            if (i * args.batch_size) % 16384 < args.batch_size:
                print(f"[Phoenix-zero] 进度 {min((i+1)*args.batch_size, len(meta))}/{len(meta)}",
                      flush=True)
    pred_log1p = np.concatenate(pred_log1p_all, axis=0)[: len(meta)]  # (N, 280)

    # 反归一化 → raw counts
    pred_raw_full = denorm_to_raw(pred_log1p, stats)                  # (N, 280) raw
    # 散射到 313 列（未覆盖基因填 0）
    pred = np.zeros((len(meta), len(our_genes)), dtype=np.float32)
    pred[:, out_cols] = pred_raw_full[:, off_cols]

    # 真值 raw counts（313 列）
    expr = np.load(os.path.join(args.test_dir, "gene_expression.npy"))
    true_raw = expr[: len(meta)].astype(np.float32)

    # 统一指标：**只对交集基因(out_cols)计算**（gene_idx），含 SSIM（坐标栅格化）与全 Top-k
    coords = meta[["x_centroid", "y_centroid"]].to_numpy(float)
    results = compute_metrics_vectorized(true_raw, pred, true_raw, pred,
                                         topk_ks="full", details=True, coords=coords,
                                         gene_idx=out_cols)
    results["_gene_names"] = [our_genes[c] for c in out_cols]  # 只含覆盖(交集)基因
    results["covered_genes"] = len(covered)
    json_results = {k: v for k, v in results.items()
                    if not k.startswith("_") and not isinstance(v, list)
                    and not isinstance(v, np.ndarray)}
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    csv_files = save_eval_results_csv(os.path.join(args.output_dir, "eval_metrics.csv"),
                                      results, gene_names=results["_gene_names"])
    print(json.dumps(json_results, ensure_ascii=False, indent=2))
    print(f"CSV: {csv_files['summary']} / {csv_files['genes']}")


if __name__ == "__main__":
    main()
