"""DeepPT：AE 压缩 + 统一 MLP 回归（spot/cell 级适配，方案 A）。

训练流程：先 AE 重构预训练（官方 12AE），再走统一 MSE 回归（harness fit）。
"""
from __future__ import annotations

import torch

from common.benchmark.harness import fit
from .model import DeepPTModel

__all__ = ["DeepPTModel", "build_model", "train_function"]

AE_PRETRAIN_EPOCHS = 5
AE_PRETRAIN_LR = 1e-3


def build_model(num_genes: int = 313, **kwargs):
    return DeepPTModel(num_genes=num_genes, **kwargs)


def _pretrain_ae(model: DeepPTModel, train_loader, device: str, epochs: int) -> None:
    """AE 重构预训练：在训练集 UNI2 特征上最小化重构 MSE（官方 12AE 语义）。"""
    model.train()
    opt = torch.optim.AdamW(model.ae.parameters(), lr=AE_PRETRAIN_LR)
    for ep in range(1, epochs + 1):
        total, n = 0.0, 0
        for batch in train_loader:
            x = batch["feature"].to(device)
            opt.zero_grad()
            recon = model.reconstruct(x)
            loss = torch.nn.functional.mse_loss(recon, x)
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
            n += x.size(0)
        print(f"[DeepPT AE epoch {ep}/{epochs}] recon_loss={total/max(n,1):.4f}",
              flush=True)


def train_function(model, train_loader, valid_loader, args, stats) -> dict:
    """DeepPT 训练：先 AE 重构预训练，再走统一 MSE 回归（fit，含早停）。"""
    model = model.to(args.device)
    _pretrain_ae(model, train_loader, args.device, AE_PRETRAIN_EPOCHS)

    return fit(
        model, train_loader, valid_loader, args.epochs, args.lr, args.device,
        out_dir=args.output_dir, weight_decay=args.weight_decay,
        gene_norm=args.gene_norm, eval_stats=stats,
        config={"method": "deeppt", "num_genes": model.num_genes,
                "gene_norm": args.gene_norm},
        patience=args.patience,
    )
