"""Phoenix 官方权重微调后测试：Rep2 全量，统一协议（log1p_zscore 评估）。

用法：
    python scripts/test_phoenix_finetune.py --ckpt outputs/bench_phoenix_finetune/best.pt \
        --test_dir data/rep2 --output_dir outputs/bench_phoenix_finetune
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根目录


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phoenix 微调模型测试")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--flow_weights", default="methods/phoenix/flow_model.pth")
    p.add_argument("--dino_weights", default="methods/phoenix/pytorch_model.bin")
    p.add_argument("--output_dir", default="outputs/bench_phoenix_finetune")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_cells", type=int, default=None)
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

    from common.benchmark.harness import evaluate, save_eval_results_csv
    from common.data.expression import normalize_expression
    from methods.phoenix.official import IMG_MEAN, IMG_STD, PhoenixOfficial

    args = parse_args()
    device = args.device
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    num_genes = cfg.get("num_genes", 313)
    test_steps = cfg.get("test_steps", 50)
    stats = cfg.get("stats")
    if stats:
        stats = {k: (np.asarray(v, dtype=np.float64) if isinstance(v, list) else v)
                 for k, v in stats.items()}

    tf = transforms.Compose([
        transforms.Resize((224, 224), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(tuple(IMG_MEAN[0, :, 0, 0].tolist()),
                             tuple(IMG_STD[0, :, 0, 0].tolist())),
    ])

    class PatchExprDS(Dataset):
        def __init__(self, paths, expr_norm):
            self.paths = list(paths)
            self.expr = expr_norm

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            return {"patch": tf(Image.open(self.paths[i]).convert("RGB")),
                    "gene_expr": torch.from_numpy(self.expr[i].copy())}

    meta = pd.read_csv(os.path.join(args.test_dir, "metadata.csv"))
    if args.max_cells:
        meta = meta.iloc[: args.max_cells]
    expr_raw = np.load(os.path.join(args.test_dir, "gene_expression.npy"))
    expr_norm, _ = normalize_expression(expr_raw[: len(meta)], "log1p_zscore", stats)
    gene_names = None
    gn_path = os.path.join(args.test_dir, "gene_names.txt")
    if os.path.exists(gn_path):
        with open(gn_path) as f:
            gene_names = [ln.strip() for ln in f if ln.strip()]
    ds = PatchExprDS(meta["patch_path"].tolist(), expr_norm)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = PhoenixOfficial(num_genes=num_genes, flow_weights=args.flow_weights,
                            dino_weights=args.dino_weights, device=device,
                            n_sample_steps=test_steps)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[Phoenix-test] 加载 best.pt，{test_steps} 步采样，{len(meta)} 细胞", flush=True)

    results = evaluate(model, dl, device, "log1p_zscore", stats, details=True)
    json_results = {k: v for k, v in results.items() if not isinstance(v, np.ndarray)}
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    csv_files = save_eval_results_csv(os.path.join(args.output_dir, "eval_metrics.csv"),
                                      results, gene_names=gene_names)
    print(json.dumps(json_results, ensure_ascii=False, indent=2))
    print(f"CSV: {csv_files['summary']}")


if __name__ == "__main__":
    main()
