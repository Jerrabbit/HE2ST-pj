"""Phoenix 官方权重**微调**：Rep1 训练 → Rep2 测试。

- 加载官方 flow_model.pth（flow 可训练）+ DINOv2（冻结）。
- 输入为 patch（DINOv2 在内部提 256 token 条件），目标为 log1p_zscore（训练集统计量）。
- 训练：流匹配 MSE；验证：Rep2 子集少步采样取 val_PCC 早停；测试：Rep2 全量完整步数。

用法（远程 myenv1）：
    python scripts/train_phoenix_finetune.py --train_dir data/rep1 --valid_dir data/rep2 \
        --output_dir outputs/bench_phoenix_finetune
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根目录


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phoenix 官方权重微调")
    p.add_argument("--train_dir", required=True)
    p.add_argument("--valid_dir", required=True)
    p.add_argument("--flow_weights", default="methods/phoenix/flow_model.pth")
    p.add_argument("--dino_weights", default="methods/phoenix/pytorch_model.bin")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--train_max_cells", type=int, default=None)
    p.add_argument("--eval_subset", type=int, default=8000, help="验证采样子集")
    p.add_argument("--eval_steps", type=int, default=5, help="验证用采样步数（早停信号）")
    p.add_argument("--test_steps", type=int, default=50, help="最终测试采样步数")
    p.add_argument("--output_dir", default="outputs/bench_phoenix_finetune")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def _load_patches_meta(data_dir: str, max_cells: int | None):
    import pandas as pd

    meta = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
    if max_cells:
        meta = meta.iloc[: max_cells]
    return meta


def main() -> None:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset, Subset

    from common.benchmark.harness import evaluate
    from common.data.expression import normalize_expression, save_stats_json
    from methods.phoenix.official import PhoenixFlowOnly

    args = parse_args()
    device = args.device

    class TokenExprDS(Dataset):
        """(DINOv2 缓存 token (261,1536) fp16 → fp32, gene_expr 归一化)。"""

        def __init__(self, tokens_path, expr_norm):
            self.tokens = np.load(tokens_path, mmap_mode="r")   # (N, 261, 1536) fp16
            self.expr = expr_norm
            assert len(self.tokens) == len(expr_norm)

        def __len__(self):
            return len(self.expr)

        def __getitem__(self, i):
            return {"feature": torch.from_numpy(self.tokens[i].astype(np.float32).copy()),
                    "gene_expr": torch.from_numpy(self.expr[i].copy())}

    # 表达归一化（log1p_zscore，训练集统计量）
    expr_tr = np.load(os.path.join(args.train_dir, "gene_expression.npy"))
    expr_va = np.load(os.path.join(args.valid_dir, "gene_expression.npy"))
    expr_tr_norm, stats = normalize_expression(expr_tr, "log1p_zscore")
    expr_va_norm, _ = normalize_expression(expr_va, "log1p_zscore", stats)
    num_genes = expr_tr.shape[1]
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[Phoenix-finetune] genes={num_genes} 训练集 {expr_tr.shape[0]} 验证集 {expr_va.shape[0]}",
          flush=True)
    save_stats_json(stats, os.path.join(args.output_dir, "train_stats.json"))

    tr_tokens = os.path.join(args.train_dir, "X_phoenix_dino.npy")
    va_tokens = os.path.join(args.valid_dir, "X_phoenix_dino.npy")
    if not os.path.exists(tr_tokens) or not os.path.exists(va_tokens):
        raise SystemExit(
            f"缺少 DINOv2 缓存 token：{tr_tokens} / {va_tokens}\n"
            f"请先运行：python scripts/extract_phoenix_dino.py --rep 1/2")

    tr_ds = TokenExprDS(tr_tokens, expr_tr_norm[: args.train_max_cells
                       if args.train_max_cells else expr_tr.shape[0]])
    va_ds = TokenExprDS(va_tokens, expr_va_norm)
    if len(va_ds) > args.eval_subset:
        idx = np.random.default_rng(0).choice(len(va_ds), args.eval_subset, replace=False)
        va_ds = Subset(va_ds, idx)

    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       num_workers=4, pin_memory=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=4, pin_memory=True)

    model = PhoenixFlowOnly(num_genes=num_genes, flow_weights=args.flow_weights,
                            device=device, n_sample_steps=args.test_steps)
    print(f"[Phoenix-finetune] 官方 flow 权重已加载（DINOv2 token 缓存），flow 可训练", flush=True)

    optimizer = torch.optim.AdamW(model.flow.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    best_pcc, best_state, no_improve = -float("inf"), None, 0
    history = []
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.flow.train()
        total, n = 0.0, 0
        for batch in tr_dl:
            x = batch["feature"].to(device)
            y = batch["gene_expr"].to(device)
            optimizer.zero_grad()
            loss, _ = model.training_loss(y, x)
            loss.backward()
            optimizer.step()
            total += loss.item() * y.size(0)
            n += y.size(0)
        train_loss = total / max(n, 1)

        # 验证：少步采样
        model.flow.eval()
        model.n_sample_steps = args.eval_steps
        ev = evaluate(model, va_dl, device, "log1p_zscore", stats)
        model.n_sample_steps = args.test_steps
        history.append({"epoch": epoch, "train_loss": train_loss, **ev})
        print(f"[Phoenix-finetune ep{epoch}/{args.epochs}] loss={train_loss:.4f} "
              f"val_PCC={ev['PCC']:.4f}", flush=True)

        if ev["PCC"] == ev["PCC"] and ev["PCC"] > best_pcc + 1e-4:
            best_pcc = ev["PCC"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= args.patience:
            print(f"[Phoenix-finetune] early stop @ ep{epoch}", flush=True)
            break

    if best_state is not None:
        torch.save({"model": best_state,
                    "config": {"method": "phoenix_finetune", "num_genes": num_genes,
                               "gene_norm": "log1p_zscore",
                               "test_steps": args.test_steps,
                               "stats": {k: v.tolist() if hasattr(v, "tolist") else v
                                         for k, v in stats.items()}}},
                   os.path.join(args.output_dir, "best.pt"))
    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"[Phoenix-finetune] 训练完成，best val_PCC={best_pcc:.4f}，"
          f"测试: python scripts/test_phoenix_finetune.py --ckpt {args.output_dir}/best.pt "
          f"--test_dir {args.valid_dir} --output_dir {args.output_dir}")


if __name__ == "__main__":
    main()
