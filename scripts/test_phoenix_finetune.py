"""Phoenix 官方权重微调后测试：Rep2 全量，统一协议（log1p_zscore 评估）。

与训练一致的 token 式推理：加载 PhoenixFlowOnly（DINOv2 冻结、缓存 token），
从缓存 DINOv2 token 采样（Euler，n_sample_steps），避免 111k patch 的 DINOv2 前向
与 cpfs 小文件读。

用法（远程 myenv1）：
    python scripts/test_phoenix_finetune.py --ckpt outputs/bench_phoenix_finetune/best.pt \
        --test_dir data/rep2 --output_dir outputs/bench_phoenix_finetune \
        --tokens_dir /tmp/dino_tokens
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根目录


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phoenix 微调模型测试（token 式，与训练一致）")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--flow_weights", default="methods/phoenix/flow_model.pth")
    p.add_argument("--output_dir", default="outputs/bench_phoenix_finetune")
    p.add_argument("--tokens_dir", default=None,
                   help="DINOv2 缓存 token 目录（如 /tmp/dino_tokens，cpfs mmap 随机读会 D-state 卡死，"
                        "务必放本地盘）。缺省在 test_dir 内找 X_phoenix_dino.npy。")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_cells", type=int, default=None)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    import json

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset

    from common.benchmark.harness import evaluate, save_eval_results_csv
    from common.data.expression import normalize_expression
    from methods.phoenix.official import PhoenixFlowOnly

    # 111k 细胞单次长 evaluate 循环，file_descriptor 共享策略会耗尽 FD
    # （"Too many open files"）；file_system 策略把 tensor 落到 /tmp，无持久 FD。
    torch.multiprocessing.set_sharing_strategy("file_system")

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

    class TokenExprDS(Dataset):
        """(DINOv2 缓存 token (261,1536) fp16 → fp32, gene_expr 归一化, 坐标)。与训练一致。"""

        def __init__(self, tokens_path, expr_norm, coords):
            self.tokens = np.load(tokens_path, mmap_mode="r")   # (N, 261, 1536) fp16
            self.expr = expr_norm
            self.coords = coords
            assert len(self.tokens) == len(expr_norm) == len(coords), \
                f"token {self.tokens.shape} vs expr {expr_norm.shape} vs coords {coords.shape} 行数不匹配"

        def __len__(self):
            return len(self.expr)

        def __getitem__(self, i):
            return {"feature": torch.from_numpy(self.tokens[i].astype(np.float32).copy()),
                    "gene_expr": torch.from_numpy(self.expr[i].copy()),
                    "coords": torch.from_numpy(self.coords[i].copy())}

    def _tok_path(dirname: str) -> str:
        if args.tokens_dir:
            return os.path.join(args.tokens_dir, os.path.basename(dirname),
                                "X_phoenix_dino.npy")
        return os.path.join(dirname, "X_phoenix_dino.npy")

    import pandas as pd
    meta = pd.read_csv(os.path.join(args.test_dir, "metadata.csv"))
    if args.max_cells:
        meta = meta.iloc[: args.max_cells]
    coords_all = meta[["x_centroid", "y_centroid"]].to_numpy(np.float32)

    expr_raw = np.load(os.path.join(args.test_dir, "gene_expression.npy"))
    if args.max_cells:
        expr_raw = expr_raw[: args.max_cells]
    expr_norm, _ = normalize_expression(expr_raw, "log1p_zscore", stats)
    gene_names = None
    gn_path = os.path.join(args.test_dir, "gene_names.txt")
    if os.path.exists(gn_path):
        with open(gn_path) as f:
            gene_names = [ln.strip() for ln in f if ln.strip()]

    tok_path = _tok_path(args.test_dir)
    if not os.path.exists(tok_path):
        raise SystemExit(f"缺少 DINOv2 缓存 token：{tok_path}\n"
                         f"请先运行 extract_phoenix_dino.py，或指定 --tokens_dir /tmp/dino_tokens")
    ds = TokenExprDS(tok_path, expr_norm, coords_all)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = PhoenixFlowOnly(num_genes=num_genes, flow_weights=args.flow_weights,
                            device=device, n_sample_steps=test_steps)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.n_sample_steps = test_steps
    print(f"[Phoenix-test] 加载 best.pt（token 式），{test_steps} 步采样，{len(ds)} 细胞，"
          f"token {ds.tokens.shape}", flush=True)

    results = evaluate(model, dl, device, "log1p_zscore", stats,
                       topk_ks="full", details=True, ssim=True)
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
