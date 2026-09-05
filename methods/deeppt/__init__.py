"""DeepPT：AE 压缩 + 统一 MLP 回归（spot/cell 级适配，方案 A）。

训练流程：先 AE 重构预训练（官方 12AE），再走统一 MSE 回归（harness fit）。
"""
from __future__ import annotations

import torch

from common.benchmark.harness import fit
from .model import DeepPTModel

__all__ = ["DeepPTModel", "build_model", "train_function"]

AE_PRETRAIN_EPOCHS = 20
AE_PRETRAIN_LR = 1e-4   # 官方 12AE：Adam lr=1e-4


def build_model(num_genes: int = 313, **kwargs):
    return DeepPTModel(num_genes=num_genes, **kwargs)


def _pretrain_ae(model: DeepPTModel, train_loader, device: str, epochs: int) -> None:
    """AE 重构预训练：在训练集 UNI2 特征上最小化重构 MSE（官方 12AE 语义）。"""
    model.train()
    opt = torch.optim.Adam(model.ae.parameters(), lr=AE_PRETRAIN_LR)   # 官方 12AE 用 Adam
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
    """DeepPT 训练：AE 重构预训练 → **冻结 AE** → 只训官方 MLP_regression 头（fit 早停）。

    对齐官方流程：12AE 单独训至收敛后 `features_compression` 冻结 encoder，头只在固定压缩
    特征上训练（原实现 AE 与头联合微调，已按忠实度审计改为训后冻结）。
    """
    model = model.to(args.device)
    _pretrain_ae(model, train_loader, args.device, AE_PRETRAIN_EPOCHS)
    for p in model.ae.parameters():
        p.requires_grad_(False)          # 冻结 AE，头在固定压缩特征上训练（官方语义）
    model.ae.eval()

    return fit(
        model, train_loader, valid_loader, args.epochs, args.lr, args.device,
        out_dir=args.output_dir, weight_decay=args.weight_decay,
        gene_norm=args.gene_norm, eval_stats=stats,
        config={"method": "deeppt", "num_genes": model.num_genes,
                "gene_norm": args.gene_norm},
        patience=args.patience,
    )
