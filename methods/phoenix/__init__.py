"""Phoenix：潜在空间 flow matching 生成模型（论文概念实现，无官方代码）。"""
from __future__ import annotations

import os

import torch

from common.benchmark.harness import evaluate
from .model import Phoenix

__all__ = ["Phoenix"]


def build_model(num_genes: int = 313, **kwargs):
    return Phoenix(num_genes=num_genes, **kwargs)


EVAL_SUBSET = 25000       # 每 epoch 验证用随机子集，控制采样开销；最终测试仍全量
EVAL_SAMPLE_STEPS = 10    # 训练期验证用较少 Euler 步（早停信号足够）；最终测试用完整 n_sample_steps


def train_function(model, train_loader, valid_loader, args, stats) -> dict:
    """Phoenix 自定义训练：流匹配损失。

    验证：68.8M 流 transformer 的 20 步 Euler 采样在 111k 细胞上很慢，
    每 epoch 用 EVAL_SUBSET 随机子集评估（早停信号仍有效），最终测试用全量。
    """
    import numpy as np
    from torch.utils.data import DataLoader, Subset

    device = args.device
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    os.makedirs(args.output_dir, exist_ok=True)

    # 预构建验证子集（固定 indices，保证每 epoch 评估一致）
    eval_loader = None
    if valid_loader is not None:
        ds = valid_loader.dataset
        n_valid = len(ds)
        if n_valid > EVAL_SUBSET:
            idx = np.random.default_rng(0).choice(n_valid, EVAL_SUBSET, replace=False)
            eval_loader = DataLoader(Subset(ds, idx), batch_size=args.batch_size,
                                     shuffle=False)
        else:
            eval_loader = valid_loader

    best_pcc, best_state = -float("inf"), None
    no_improve = 0
    patience = int(getattr(args, "patience", 10))
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for batch in train_loader:
            cond = batch["feature"].to(device)
            expr = batch["gene_expr"].to(device)
            optimizer.zero_grad()
            loss, _ = model.training_loss(expr, cond)
            loss.backward()
            optimizer.step()
            total += loss.item() * expr.size(0)
            n += expr.size(0)
        train_loss = total / max(n, 1)

        # 训练期验证用较少 Euler 步（只影响早停信号，不影响最终测试质量）
        orig_steps = model.n_sample_steps
        model.n_sample_steps = EVAL_SAMPLE_STEPS
        ev = evaluate(model, eval_loader, device, args.gene_norm, stats)
        model.n_sample_steps = orig_steps
        history.append({"epoch": epoch, "train_loss": train_loss, **ev})
        print(f"[Phoenix epoch {epoch}/{args.epochs}] loss={train_loss:.4f} "
              f"val_PCC={ev['PCC']:.4f}", flush=True)

        if ev["PCC"] == ev["PCC"] and ev["PCC"] > best_pcc + 1e-4:
            best_pcc = ev["PCC"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"[Phoenix early stop] val_PCC {patience} 个 epoch 未提升，"
                  f"在 epoch {epoch} 停止", flush=True)
            break

    if best_state is not None:
        torch.save({"model": best_state,
                    "config": {"method": "phoenix", "num_genes": model.num_genes,
                               "gene_norm": args.gene_norm}},
                   os.path.join(args.output_dir, "best.pt"))
    return history
