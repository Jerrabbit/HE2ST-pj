"""SQUALL 官方解码器评估（forward_rgb_to_expr → 15757 基因 → 映射 313 → 指标）。

官方推理路径：rgb(0-1) → encoder.forward_rgb → decoder.forward_rgb_to_expr
→ (B, expr_size, expr_size, 15757) 表达网格 → 空间 mean → 15757 维/细胞。
用 gene_token_homologs.csv 把 15757 通道映射到本仓库 313 公共基因，统一指标评估。

对比：冻结特征 + 训练头（X_squall.npy → MLP）的结果在 bench_squall 0.2116；
本脚本给出"官方解码器直接推理"的结果，检验 SQUALL 完整模型是否被低估。

注意输入预处理：官方教程喂 rgb/255（0-1），与 extract_squall.py 的 0-255 不同。
解码器输出尺度（log1p 或 raw）用诊断判断，脚本同时报告两种解释的指标。

用法（远程 myenv1）：
    python scripts/test_squall_decoder.py --test_dir data/rep2 \
        --max_cells 5000 --output_dir outputs/bench_squall_decoder
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根目录

import numpy as np
import pandas as pd
import torch
from PIL import Image

from common.benchmark.harness import compute_metrics_vectorized
from common.data.expression import load_expression

CONFIG = os.path.expanduser("~/HE2ST-pj/codes/squall/SQUALL_Tutorial/config.yaml")
CKPT = os.path.expanduser("~/HE2ST-pj/weights/squall/SQUALL_full.pth")
GENE_MAP = os.path.expanduser("~/HE2ST-pj/codes/squall/gene_token_homologs.csv")
RES = 0.5
IMG_SIZE = 224
EXPR_CHANS = 15757


def load_squall(ckpt_path: str, device: str = "cuda"):
    """构建 Squall 并加载冻结权重（strict=True）。"""
    import yaml

    from methods.squall.Squall import Squall

    class AttrDict(dict):
        def __getattr__(self, k):
            try:
                return self[k]
            except KeyError:
                raise AttributeError(k)

    with open(CONFIG) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    model = Squall(AttrDict(config["model"]))
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    return model


def load_gene_to_idx(csv_path: str) -> dict[str, int]:
    """gene_token_homologs.csv：首列 = 15757 通道索引，HGNC_symbol = 基因名。"""
    df = pd.read_csv(csv_path)
    return dict(zip(df["HGNC_symbol"], df.iloc[:, 0].astype(int)))


def main() -> None:
    p = argparse.ArgumentParser(description="SQUALL 官方解码器评估")
    p.add_argument("--test_dir", required=True)
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--gene_map", default=GENE_MAP)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_cells", type=int, default=5000, help="验证用子集；None=全量")
    p.add_argument("--output_dir", default="outputs/bench_squall_decoder")
    args = p.parse_args()

    model = load_squall(args.ckpt, args.device)
    print(f"[SQUALL-decoder] 冻结模型加载成功", flush=True)

    # 基因映射：本仓库基因名 → 15757 通道索引
    gene_names_path = os.path.join(args.test_dir, "gene_names.txt")
    with open(gene_names_path) as f:
        gene_names = [ln.strip() for ln in f if ln.strip()]
    gene_to_idx = load_gene_to_idx(args.gene_map)
    covered = {g: gene_to_idx[g] for g in gene_names if g in gene_to_idx}
    print(f"[SQUALL-decoder] 313 基因中 {len(covered)} 个在 SQUALL 15757 基因表中", flush=True)

    meta = pd.read_csv(os.path.join(args.test_dir, "metadata.csv"))
    if args.max_cells:
        meta = meta.iloc[: args.max_cells]
    n = len(meta)
    expr_raw, _ = load_expression(args.test_dir)
    expr_raw = expr_raw[:n]

    cols = {g: i for i, g in enumerate(covered)}  # 覆盖基因 → 输出列
    preds = np.zeros((n, len(covered)), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, args.batch_size):
            j0, j1 = i, min(i + args.batch_size, n)
            B = j1 - j0
            batch = []
            for j in range(j0, j1):
                img = Image.open(meta.iloc[j]["patch_path"]).convert("RGB").resize(
                    (IMG_SIZE, IMG_SIZE), Image.BILINEAR)
                batch.append(torch.from_numpy(np.asarray(img, dtype=np.float32)
                                              / 255.0).permute(2, 0, 1))  # 0-1
            x = torch.stack(batch).to(args.device)                       # (B,3,224,224)
            res = torch.full((B, 1), RES, dtype=torch.float32, device=args.device)
            expr = model.forward_rgb_to_expr(x, res)                     # (B,56,56,15757)
            expr = expr.mean(1).mean(1).cpu().numpy()                    # (B,15757) 空间 mean
            for g, c in cols.items():
                preds[j0:j1, c] = expr[:, gene_to_idx[g]]

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "pred_decoder.npy"), preds)

    # 诊断：解码器输出尺度
    print(f"[SQUALL-decoder] 输出范围 [{preds.min():.4f}, {preds.max():.4f}] "
          f"非负占比 {(preds >= 0).mean():.3f}", flush=True)

    # 覆盖基因子集上的真值
    valid_cols = [i for i, g in enumerate(gene_names) if g in covered]
    covered_names = [gene_names[i] for i in valid_cols]
    y_true = expr_raw[:, valid_cols]
    coords = meta[["x_centroid", "y_centroid"]].to_numpy(float)

    # 解释 A：解码器输出即 raw counts 语义
    resA = compute_metrics_vectorized(y_true, preds, y_true, preds,
                                      topk_ks="full", details=True, coords=coords)
    # 解释 B：解码器输出为 log1p → expm1 回 raw
    preds_log = np.expm1(np.clip(preds, -30, 30)).astype(np.float32)
    resB = compute_metrics_vectorized(y_true, preds_log, y_true, preds_log,
                                      topk_ks="full", details=True, coords=coords)
    print(f"[SQUALL-decoder] 解释A(输出=raw): PCC={resA['PCC']:.4f} | "
          f"解释B(输出=log1p→raw): PCC={resB['PCC']:.4f}", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    out = {"n_cells": n, "covered_genes": len(covered),
           "as_raw": {k: v for k, v in resA.items() if isinstance(v, (int, float))},
           "as_log1p": {k: v for k, v in resB.items() if isinstance(v, (int, float))},
           "output_range": [float(preds.min()), float(preds.max())],
           "nonneg_frac": float((preds >= 0).mean())}
    with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    # 保存标准 CSV（取解释 B=log1p→raw 的语义，与其它方法 log1p 目标一致；基因 CSV 为覆盖子集）
    from common.benchmark.harness import save_eval_results_csv, scalar_results
    save_eval_results_csv(os.path.join(args.output_dir, "eval_metrics.csv"),
                          resB, gene_names=covered_names)
    print(json.dumps(scalar_results(resB), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
