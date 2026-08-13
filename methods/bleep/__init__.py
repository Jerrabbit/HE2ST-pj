"""BLEEP：双模态对比学习 + 参考集检索。"""
from __future__ import annotations

import os

import torch

from common.benchmark.harness import evaluate
from .model import BLEEP, clip_soft_target_loss

__all__ = ["BLEEP"]


def build_model(num_genes: int = 313, **kwargs):
    return BLEEP(num_genes=num_genes, **kwargs)


def post_load(model: BLEEP, ckpt: dict) -> BLEEP:
    """从 checkpoint 恢复参考检索库（state_dict 不含非参数 reference）。"""
    if ckpt.get("reference"):
        model.reference = {
            "spot_emb": ckpt["reference"]["spot_emb"],
            "spot_expr": ckpt["reference"]["spot_expr"],
        }
    return model


def _build_reference(model: BLEEP, train_loader, device: str) -> None:
    """把训练集全部表达向量过 spot_projection 构建参考库。"""
    expr_parts = []
    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            expr_parts.append(batch["gene_expr"].to(device))
    expr = torch.cat(expr_parts, dim=0)
    model.build_reference(expr)   # spot_emb 存 GPU，spot_expr 存 float32
    model.reference["spot_expr"] = model.reference["spot_expr"].cpu()
    model.reference["spot_emb"] = model.reference["spot_emb"].cpu()


def train_function(model, train_loader, valid_loader, args, stats) -> dict:
    """BLEEP 自定义训练：批内软目标对比损失 + 每 epoch 重建参考库并评估。"""
    device = args.device
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    os.makedirs(args.output_dir, exist_ok=True)

    best_pcc, best_state, best_ref = -float("inf"), None, None
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for batch in train_loader:
            x = batch["patch"].to(device)
            e = batch["gene_expr"].to(device)
            optimizer.zero_grad()
            img_emb = model.image_embed(x)
            spot_emb = model.spot_embed(e)
            loss = clip_soft_target_loss(spot_emb, img_emb, temperature=1.0)
            loss.backward()
            optimizer.step()
            total += loss.item() * e.size(0)
            n += e.size(0)
        train_loss = total / max(n, 1)

        # 每 epoch 用训练集重建参考库，再在验证集上检索评估
        _build_reference(model, train_loader, device)
        ev = evaluate(model, valid_loader, device, args.gene_norm, stats)
        history.append({"epoch": epoch, "train_loss": train_loss, **ev})
        print(f"[BLEEP epoch {epoch}/{args.epochs}] loss={train_loss:.4f} "
              f"val_PCC={ev['PCC']:.4f}", flush=True)

        if ev["PCC"] == ev["PCC"] and ev["PCC"] > best_pcc:
            best_pcc = ev["PCC"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_ref = {k: v.clone() for k, v in model.reference.items()}

    if best_state is not None:
        torch.save({"model": best_state, "reference": best_ref,
                    "config": {"method": "bleep", "num_genes": model.num_genes,
                               "gene_norm": args.gene_norm}},
                   os.path.join(args.output_dir, "best.pt"))
    return history
