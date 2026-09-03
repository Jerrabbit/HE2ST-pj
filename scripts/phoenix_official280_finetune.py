#!/usr/bin/env python
"""Phoenix flow **保留官方 280 面板**的就地微调（不改架构 / 输出仍 280，2026-09-03）。

与之前的 Phoenix 微调（`train_phoenix_finetune.py`，把 flow 位置 0..312 **重接线**到我们
313 面板列序、输出 313）对比：
- 本脚本**保持官方 280 个输出位置不变**（num_genes=280），位置 p 仍是官方基因 p；
- 训练目标 = rep1 表达中**按基因名匹配**到官方面板 p 的那一列（280 全覆盖，官方 280 ⊂ 我们
  313，零样本已证），只这 280 个"重叠"位置学到监督；我们面板多出的 33 个非官方基因
  无输出位置 → 预测为 0 / 不参与评测；
- 评测**只在交集基因（280）上算指标**（数组本身即 280 交集子集；README 已注明），
  可直接与 zero-shot（280 交集 PCC≈0）对比。

架构/协议与既有微调一致：DINOv2 冻结（用缓存 token `X_phoenix_dino.npy` (N,261,1536)），
flow（px_embedding+blocks+head）从官方 flow_model.pth 初始化并**训练**；目标空间
log1p_zscore（训练集统计量）；rep2 子集少步采样 PCC 早停；最终测试完整步数。

用法（远程 myenv1）：
    python scripts/phoenix_official280_finetune.py train \
        --train_dir data/rep1 --valid_dir data/rep2 \
        --output_dir outputs/bench_phoenix_official280_ft
    python scripts/phoenix_official280_finetune.py test \
        --test_dir data/rep2 \
        --ckpt outputs/bench_phoenix_official280_ft/best.pt \
        --output_dir outputs/bench_phoenix_official280_ft
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根目录

OFFICIAL_GENES = os.path.expanduser("~/HE2ST-pj/methods/phoenix/xenium_human_breast.npy")
FLOW_WEIGHTS = os.path.expanduser("~/HE2ST-pj/methods/phoenix/flow_model.pth")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phoenix flow 保留官方 280 面板就地微调")
    sub = p.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--flow_weights", default=FLOW_WEIGHTS)
    common.add_argument("--official_genes", default=OFFICIAL_GENES,
                        help="官方 280 基因名（按 flow 位置 0..279 顺序）")
    common.add_argument("--device", default="cuda")
    common.add_argument("--tokens_dir", default=None,
                        help="DINOv2 token 缓存目录（cpfs mmap 随机读 D-state 卡死时可复制本地盘后传）")
    common.add_argument("--batch_size", type=int, default=32)
    common.add_argument("--train_max_cells", type=int, default=None)
    common.add_argument("--eval_subset", type=int, default=8000, help="验证采样子集")
    common.add_argument("--eval_steps", type=int, default=5, help="验证用采样步数")
    common.add_argument("--test_steps", type=int, default=50, help="最终测试采样步数")
    common.add_argument("--output_dir", required=True)

    p_train = sub.add_parser("train", parents=[common])
    p_train.add_argument("--train_dir", required=True)
    p_train.add_argument("--valid_dir", required=True)
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--patience", type=int, default=10)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--weight_decay", type=float, default=0.0)

    p_test = sub.add_parser("test", parents=[common])
    p_test.add_argument("--test_dir", required=True)
    p_test.add_argument("--ckpt", required=True)
    p_test.add_argument("--test_max_cells", type=int, default=None)
    return p.parse_args()


def _tok_path(data_dir: str, tokens_dir: str | None) -> str:
    if tokens_dir:
        return os.path.join(tokens_dir, os.path.basename(data_dir.rstrip("/")), "X_phoenix_dino.npy")
    return os.path.join(data_dir, "X_phoenix_dino.npy")


def official_to_our_cols(official_path: str, our_gene_names: list[str]) -> list[int]:
    """官方位置 p (0..279) → 我们面板列号（按基因名匹配，找不到→-1）。"""
    import numpy as np

    off = [str(g) for g in np.load(official_path, allow_pickle=True)]
    idx = {g: i for i, g in enumerate(our_gene_names)}
    return [idx.get(g, -1) for g in off]


def _prepare(data_dir: str, tokens_dir: str | None, our_cols_official: list[int],
             ref_stats: dict | None, max_cells: int | None):
    """读 meta/表达，把 (N,313) 抽成官方顺序 (N,280)，归一化。返回 (norm, coords, n, stats)。"""
    import numpy as np
    import pandas as pd

    meta = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
    n = len(meta) if max_cells is None else min(max_cells, len(meta))
    raw = np.load(os.path.join(data_dir, "gene_expression.npy"))[:n]
    sub = raw[:, our_cols_official].astype(np.float32)          # (N,280) 官方顺序
    from common.data.expression import normalize_expression
    norm, stats = normalize_expression(sub, "log1p_zscore", ref_stats)
    coords = meta[["x_centroid", "y_centroid"]].to_numpy(np.float32)[:n]
    return norm, coords, n, stats


def build_token_dataset(data_dir: str, tokens_dir: str | None, expr_norm,
                        coords, n: int):
    import numpy as np
    import torch
    from torch.utils.data import Dataset

    class TokenExprDS(Dataset):
        """(X_phoenix_dino.npy 行 fp16→fp32, log1p_zscore 表达(官方 280 顺序), 坐标)。"""

        def __init__(self, mmap, expr_norm, coords):
            self.tokens = mmap
            self.expr = expr_norm
            self.coords = coords

        def __len__(self):
            return len(self.expr)

        def __getitem__(self, i):
            return {"feature": torch.from_numpy(self.tokens[i].astype(np.float32).copy()),
                    "gene_expr": torch.from_numpy(self.expr[i]),
                    "coords": torch.from_numpy(self.coords[i])}

    mmap = np.load(_tok_path(data_dir, tokens_dir), mmap_mode="r")
    return TokenExprDS(mmap[:n], expr_norm, coords)


def main() -> None:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset

    from common.benchmark.harness import load_gene_names
    from common.benchmark.harness import evaluate, save_eval_results_csv, scalar_results
    from methods.phoenix.official import PhoenixFlowOnly

    args = parse_args()
    device = args.device

    # 官方位置 0..279 → 我们面板列号（按基因名）。zero-shot 已证 280/313 覆盖 → 全覆盖。
    # 官方 280 位置 → 我们面板列号映射只在 train 需要（存进 ckpt）；test 完全依赖 ckpt
    if args.mode == "train":
        our_names = load_gene_names(args.train_dir)
        cols_official = official_to_our_cols(args.official_genes, our_names)
        n_matched = sum(1 for c in cols_official if c >= 0)
        if n_matched != len(cols_official):
            raise SystemExit(f"官方 {len(cols_official)} 基因中 {n_matched} 能匹配到本仓库基因名；"
                             f"预期全覆盖(280)。请检查 gene 命名。")
        official_names = [our_names[c] for c in cols_official]
        print(f"[Phoenix-280] 官方 280 位置全部按基因名匹配；只监督/评测这 280 个交集基因，"
              f"输出仍 280（33 个非官方基因无输出位置 → 0/忽略）", flush=True)

    if args.mode == "train":
        os.makedirs(args.output_dir, exist_ok=True)
        tr_norm, tr_coords, n_tr, stats = _prepare(
            args.train_dir, args.tokens_dir, cols_official, None, args.train_max_cells)
        va_norm, va_coords, n_va, _ = _prepare(
            args.valid_dir, args.tokens_dir, cols_official, stats, None)

        tr_ds = build_token_dataset(args.train_dir, args.tokens_dir, tr_norm, tr_coords, n_tr)
        va_ds = build_token_dataset(args.valid_dir, args.tokens_dir, va_norm, va_coords, n_va)
        if len(va_ds) > args.eval_subset:
            idx = np.random.default_rng(0).choice(len(va_ds), args.eval_subset, replace=False)
            va_ds = Subset(va_ds, idx)
        tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=4, pin_memory=True)
        va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)

        model = PhoenixFlowOnly(num_genes=280, flow_weights=args.flow_weights,
                                device=device, n_sample_steps=args.test_steps)
        print(f"[Phoenix-280] 训练集 {n_tr} 验证集 {n_va}(取 {args.eval_subset})，"
              f"flow 可训练（DINOv2 冻结）", flush=True)
        opt = torch.optim.AdamW(model.flow.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
        best_pcc, best_state, no_improve, history = -float("inf"), None, 0, []

        for epoch in range(1, args.epochs + 1):
            model.flow.train()
            total, nb = 0.0, 0
            for batch in tr_dl:
                x = batch["feature"].to(device)
                y = batch["gene_expr"].to(device)
                opt.zero_grad()
                loss, _ = model.training_loss(y, x)
                loss.backward()
                opt.step()
                total += loss.item() * y.size(0)
                nb += y.size(0)
            model.flow.eval()
            model.n_sample_steps = args.eval_steps
            ev = evaluate(model, va_dl, device, "log1p_zscore", stats, ssim=False)
            model.n_sample_steps = args.test_steps
            history.append({"epoch": epoch, "train_loss": total / max(nb, 1),
                            **scalar_results(ev)})
            print(f"[Phoenix-280 ep{epoch}/{args.epochs}] loss={total/max(nb,1):.4f} "
                  f"val_PCC={ev['PCC']:.4f}", flush=True)
            if ev["PCC"] == ev["PCC"] and ev["PCC"] > best_pcc + 1e-4:
                best_pcc, no_improve = ev["PCC"], 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"[Phoenix-280] early stop @ ep{epoch}", flush=True)
                    break

        torch.save({"model": best_state if best_state is not None
                    else {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "config": {"method": "phoenix_official280_ft", "num_genes": 280,
                               "gene_names": official_names,
                               "covered_our_cols": cols_official,
                               "gene_norm": "log1p_zscore",
                               "eval_steps": args.eval_steps, "test_steps": args.test_steps,
                               "stats": {k: v.tolist() for k, v in stats.items()}}},
                   os.path.join(args.output_dir, "best.pt"))
        with open(os.path.join(args.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        print(f"[Phoenix-280] 训练完成 best val_PCC={best_pcc:.4f}\n"
              f"  测试: python scripts/phoenix_official280_finetune.py test "
              f"--test_dir {args.valid_dir} --ckpt {args.output_dir}/best.pt "
              f"--output_dir {args.output_dir}", flush=True)

    else:  # test
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        stats = {"means": np.asarray(cfg["stats"]["means"], dtype=np.float32),
                 "stds": np.asarray(cfg["stats"]["stds"], dtype=np.float32)}
        cols_official = cfg["covered_our_cols"]
        te_norm, te_coords, n_te, _ = _prepare(
            args.test_dir, args.tokens_dir, cols_official, stats, args.test_max_cells)
        te_ds = build_token_dataset(args.test_dir, args.tokens_dir, te_norm, te_coords, n_te)
        te_dl = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
        model = PhoenixFlowOnly(num_genes=cfg["num_genes"], flow_weights=args.flow_weights,
                                device=device, n_sample_steps=cfg["test_steps"])
        model.load_state_dict(ck["model"])
        model.eval()
        results = evaluate(model, te_dl, device, "log1p_zscore", stats,
                           topk_ks="full", details=True, ssim=True)
        results["_gene_names"] = cfg["gene_names"]
        os.makedirs(args.output_dir, exist_ok=True)
        json_results = scalar_results(results)
        with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        csv_files = save_eval_results_csv(os.path.join(args.output_dir, "eval_metrics.csv"),
                                          results, gene_names=cfg["gene_names"])
        print(json.dumps(json_results, ensure_ascii=False, indent=2), flush=True)
        print(f"[Phoenix-280] CSV: {csv_files['summary']}", flush=True)


if __name__ == "__main__":
    main()
