"""Phoenix：潜在空间 flow matching 生成模型（论文概念实现，无官方代码）。"""
from __future__ import annotations

import os

import torch

from common.benchmark.harness import evaluate
from .model import Phoenix

__all__ = ["Phoenix"]


def build_model(num_genes: int = 313, **kwargs):
    return Phoenix(num_genes=num_genes, **kwargs)


def train_function(model, train_loader, valid_loader, args, stats) -> dict:
    """Phoenix 自定义训练：重构损失 + 流匹配损失联合优化。

    args 额外字段（通过 --kwargs 或默认）：
        latent_dim, hidden_dim, flow_weight, n_sample_steps
    """
    device = args.device
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    os.makedirs(args.output_dir, exist_ok=True)

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

        ev = evaluate(model, valid_loader, device, args.gene_norm, stats)
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
