"""GHIST 专用评估：加载 best.pt → 整片 tiling 推理 → 统一指标 + CSV。

用法（远程服务器）：
    python scripts/test_ghist.py --ckpt outputs/bench_ghist/best.pt \
        --test_dir data/ghist_rep2 --output_dir outputs/bench_ghist
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根目录


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GHIST 测试集评估")
    p.add_argument("--ckpt", required=True, help="best.pt 路径")
    p.add_argument("--test_dir", required=True, help="测试集 ghist_data 目录")
    p.add_argument("--output_dir", default="outputs/ghist")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    import json

    import numpy as np
    import torch

    import methods.ghist as ghist
    from common.benchmark.harness import save_eval_results_csv, scalar_results

    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = ckpt.get("config", {})
    num_genes = config.get("num_genes", 313)
    gene_norm = config.get("gene_norm", "log1p_zscore")
    stats = config.get("stats")
    if stats:
        # _serialize_stats 把 np 数组转成了 list，这里还原
        stats = {k: (np.asarray(v, dtype=np.float64) if isinstance(v, list) else v)
                 for k, v in stats.items()}
    ref_expr = config.get("ref_expr")
    if ref_expr is not None:
        ref_expr = np.asarray(ref_expr, dtype=np.float32)

    model = ghist.build_model(num_genes=num_genes)
    model.load_state_dict(ckpt["model"])
    model.to(args.device)
    print(f"[GHIST] 加载 best.pt: num_genes={num_genes} gene_norm={gene_norm} "
          f"ref_expr={None if ref_expr is None else ref_expr.shape}", flush=True)

    results = ghist.evaluate_slide(model, args.test_dir, gene_norm, stats,
                                   args.device, args.output_dir,
                                   ref_expr=ref_expr, details=True)
    print(json.dumps(scalar_results(results), ensure_ascii=False, indent=2))
    csv_files = save_eval_results_csv(
        os.path.join(args.output_dir, "eval_metrics.csv"),
        results, gene_names=results.get("_gene_names"))
    print(f"结果已保存: {os.path.join(args.output_dir, 'test_results.json')}")
    print(f"CSV 已保存: {csv_files['summary']} / {csv_files['genes']}")


if __name__ == "__main__":
    main()
