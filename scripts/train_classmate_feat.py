#!/usr/bin/env python
"""同学特征 + 我们 MLP 头（原版 3 层 MLPHead / 参考 RefMLPHead）—— 是否特征问题的归因实验。

- 特征：outputs/classmate_feat/rep{1,2}/{DAVID_BLIP2_features,UNI_features}.npy（已对齐到我们
  rep1/rep2 cell_id 序，与 0.3240 等完全同细胞同目标）。
- 头/目标：
    --arch mlp  → MLPHead(3 层+BN) + log1p_zscore   （对照：我们 UNI2 CLS 0.3240）
    --arch ref  → RefMLPHead(LN→512→GELU→Dropout→Softplus) + log1p （对照：0.3160）
- 统一协议：rep1 训练 → rep2 验证/测试，AdamW lr1e-3、批 2048、50ep、patience10、best val_PCC；
  最终测试全量指标（PCC/SPCC/cell_PCC/SSIM/full Top-k/AUROC）。

用法（远程）：
    python3 scripts/train_classmate_feat.py --arch mlp \
      --train_feat outputs/classmate_feat/rep1/UNI_features.npy \
      --test_feat  outputs/classmate_feat/rep2/UNI_features.npy \
      --train_dir data/rep1 --test_dir data/rep2 \
      --output_dir outputs/bench_cc_UNI_mlp
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from common.benchmark.harness import evaluate, scalar_results, save_eval_results_csv
from common.data.expression import normalize_expression
from common.models.mlp_head import MLPHead, RefMLPHead


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["mlp", "ref", "bn2"], default="mlp")
    p.add_argument("--train_dir", required=True)
    p.add_argument("--test_dir", required=True)
    p.add_argument("--train_feat", required=True)
    p.add_argument("--test_feat", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


class FeatDS(Dataset):
    """(特征行, log1p_zscore|log1p 表达, 坐标)，行序 = 我们 rep 序（与特征 npy 对齐）。"""

    def __init__(self, feat, expr_norm, coords):
        self.feat = feat
        self.expr = expr_norm
        self.coords = coords

    def __len__(self):
        return len(self.expr)

    def __getitem__(self, i):
        return {"feature": torch.from_numpy(self.feat[i]),
                "gene_expr": torch.from_numpy(self.expr[i]),
                "coords": torch.from_numpy(self.coords[i])}


def _load(data_dir, feat_path, ref_stats=None, gene_norm="log1p_zscore"):
    import numpy as np
    meta = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
    coords = meta[["x_centroid", "y_centroid"]].to_numpy(np.float32)
    raw = np.load(os.path.join(data_dir, "gene_expression.npy")).astype(np.float32)
    feat = np.load(feat_path)                       # (N, d) 已对齐
    assert len(feat) == len(raw) == len(meta), (feat.shape, raw.shape, len(meta))
    norm, stats = normalize_expression(raw, gene_norm, ref_stats)
    return feat, norm, coords, stats


def main() -> None:
    args = parse_args()
    device = args.device
    gene_norm = "log1p" if args.arch == "ref" else "log1p_zscore"   # ref(Softplus)→log1p
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    tr_f, tr_n, tr_c, stats = _load(args.train_dir, args.train_feat, None, gene_norm)
    te_f, te_n, te_c, _ = _load(args.test_dir, args.test_feat, stats, gene_norm)
    in_dim = tr_f.shape[1]
    num_genes = tr_n.shape[1]

    if args.arch == "ref":
        model = RefMLPHead(in_dim, hidden_dim=512, output_dim=num_genes, use_softplus=True)
    elif args.arch == "bn2":
        # 参考 txt 注释版：Linear→256 → BN → LeakyReLU → Dropout → Linear(n_genes)，无 LN/Softplus
        model = torch.nn.Sequential(
            torch.nn.Linear(in_dim, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(256, num_genes),
        )
    else:
        model = MLPHead(in_dim, hidden_dims=(512, 256), output_dim=num_genes, dropout=0.1)
    model.input_type = "feature"     # harness predict 按 input_type 取 batch["feature"]
    model.to(device)
    print(f"[cc] arch={args.arch} in_dim={in_dim} genes={num_genes} gene_norm={gene_norm} "
          f"train={len(tr_n)} test={len(te_n)}", flush=True)

    tr_ds = FeatDS(tr_f, tr_n, tr_c)
    te_ds = FeatDS(te_f, te_n, te_c)
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
                       pin_memory=True)
    te_dl = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False, num_workers=4,
                       pin_memory=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    best_pcc, best_state, no_improve, history = -float("inf"), None, 0, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for b in tr_dl:
            x = b["feature"].to(device)
            y = b["gene_expr"].to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * y.size(0)
            n += y.size(0)
        model.eval()
        ev = evaluate(model, te_dl, device, gene_norm,
                      stats if gene_norm == "log1p_zscore" else None, ssim=False)
        history.append({"epoch": epoch, "mse": total / max(n, 1), **scalar_results(ev)})
        print(f"[cc ep{epoch}] mse={total/max(n,1):.4f} val_PCC={ev['PCC']:.4f}", flush=True)
        if ev["PCC"] == ev["PCC"] and ev["PCC"] > best_pcc + 1e-4:
            best_pcc, no_improve = ev["PCC"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[cc] early stop @ ep{epoch}", flush=True)
                break

    model.load_state_dict(best_state or {k: v.detach().cpu() for k, v in model.state_dict().items()})
    model.to(device).eval()
    results = evaluate(model, te_dl, device, gene_norm,
                       stats if gene_norm == "log1p_zscore" else None,
                       topk_ks="full", details=True, ssim=True)
    from common.benchmark.harness import load_gene_names
    results["_gene_names"] = load_gene_names(args.test_dir)
    with open(os.path.join(out_dir, "test_results.json"), "w") as f:
        json.dump(scalar_results(results), f, ensure_ascii=False, indent=2)
    save_eval_results_csv(os.path.join(out_dir, "eval_metrics.csv"), results,
                          gene_names=results["_gene_names"])
    print(json.dumps(scalar_results(results), ensure_ascii=False, indent=2), flush=True)
    print(f"[cc] best val_PCC={best_pcc:.4f} 已保存 {out_dir}/test_results.json", flush=True)


if __name__ == "__main__":
    main()
