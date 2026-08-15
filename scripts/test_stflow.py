"""STFlow 专用评估：加载 best.pt，对测试切片逐 ROI 采样生成评估。

用法（远程服务器）：
    python scripts/test_stflow.py --ckpt outputs/bench_stflow/best.pt \
        --test_dir data/rep2 --output_dir outputs/bench_stflow
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根目录


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="STFlow 测试集评估")
    p.add_argument("--ckpt", required=True, help="best.pt 路径")
    p.add_argument("--test_dir", required=True, help="测试集数据目录")
    p.add_argument("--output_dir", default="outputs/stflow")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    import torch

    from methods.stflow import STFlow, evaluate_slide

    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    config = ckpt.get("config", {})
    num_genes = config.get("num_genes", 313)
    gene_norm = config.get("gene_norm", "log1p_zscore")
    stats = config.get("stats")

    model = STFlow(num_genes=num_genes)
    model.load_state_dict(ckpt["model"])
    model.to(args.device)
    print(f"[STFlow] 加载 best.pt: num_genes={num_genes} gene_norm={gene_norm}",
          flush=True)

    results = evaluate_slide(model, args.test_dir, gene_norm, stats,
                             args.device, args.output_dir)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"结果已保存: {os.path.join(args.output_dir, 'test_results.json')}")


if __name__ == "__main__":
    main()
