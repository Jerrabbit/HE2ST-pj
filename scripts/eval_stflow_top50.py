"""STFlow Top50 HVG 评估（对照原论文协议）。

原论文（Huang et al. 2025, arXiv:2506.05361）在 Top50 高变基因上评估，
平均 PCC ~0.3-0.4。本脚本用同一协议验证我们的 STFlow 实现：
    1) 加载已训练 best.pt，对 rep2 做 ROI 流匹配采样 → 逐细胞预测；
    2) 用 rep1 训练集 raw counts 的 log1p 方差选 Top50 高变基因；
    3) 在这 50 个基因上逐基因 Pearson（归一化空间）→ 平均 PCC。
同时输出全 313 基因平均 PCC 作对照。

用法（远程 myenv1，GPU）：
    python scripts/eval_stflow_top50.py \
        --ckpt outputs/bench_stflow/best.pt \
        --train_dir data/rep1 --test_dir data/rep2 \
        --stats outputs/bench_stflow/train_stats.json
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description="STFlow Top50 HVG 评估")
    p.add_argument("--ckpt", required=True, help="best.pt 路径")
    p.add_argument("--train_dir", required=True, help="训练集目录（选 HVG）")
    p.add_argument("--test_dir", required=True, help="测试集目录（评估）")
    p.add_argument("--stats", default=None, help="train_stats.json（归一化统计量）")
    p.add_argument("--gene_norm", choices=["log1p_zscore", "log1p_norm_total", "log1p", "none"],
                   default=None, help="覆盖模型配置的归一化空间（默认读 best.pt config）")
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    import json
    import torch
    from scipy.stats import pearsonr

    from common.data.expression import load_expression, load_stats_json
    from methods.stflow import build_model, _load_slide, _tile

    # 1) 加载模型
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build_model(
        num_genes=ckpt["config"].get("num_genes", 313),
        prior=ckpt["config"].get("prior", "gaussian"),
        n_sample_steps=ckpt["config"].get("n_sample_steps", 20),
    )
    model.load_state_dict(ckpt["model"])
    model = model.to(args.device).eval()
    gene_norm = args.gene_norm or ckpt["config"].get("gene_norm", "log1p_zscore")
    print(f"[STFlow] 模型加载: num_genes={model.num_genes} "
          f"n_sample_steps={model.n_sample_steps} gene_norm={gene_norm}", flush=True)

    stats = load_stats_json(args.stats) if args.stats else None

    # 2) rep2 逐 ROI 采样（与模型训练同归一化空间）
    coords, features, expr_norm = _load_slide(args.test_dir, gene_norm, stats)
    N, G = expr_norm.shape
    y_pred = np.full((N, G), np.nan, dtype=np.float32)
    covered = np.zeros(N, dtype=bool)
    for roi in _tile(coords):
        if len(roi) < 2:
            continue
        f = torch.from_numpy(features[roi]).unsqueeze(0).to(args.device)
        c = torch.from_numpy(coords[roi]).unsqueeze(0).to(args.device)
        pred = model.sample_roi(f, c, args.device).squeeze(0).cpu().numpy()
        new = ~covered[roi]
        y_pred[roi[new]] = pred[new]
        covered[roi[new]] = True
    keep = covered
    print(f"[STFlow] 覆盖 {int(keep.sum())}/{N} 细胞", flush=True)

    def gene_mean_pcc(y_true, y_pred_, name):
        pccs = []
        for g in range(G):
            if y_pred_[:, g].std() == 0 or y_true[:, g].std() == 0:
                continue
            r, _ = pearsonr(y_true[:, g], y_pred_[:, g])
            if r == r:
                pccs.append(r)
        print(f"[STFlow] {name}: 有效基因 {len(pccs)} 平均 PCC = {np.mean(pccs):.4f}", flush=True)
        return pccs

    # 3) Top50 HVG：rep1 raw counts 的 log1p 方差
    raw_tr, _ = load_expression(args.train_dir)
    hvg_var = np.log1p(raw_tr).var(0)
    topk_idx = np.argsort(hvg_var)[-args.top_k:]
    print(f"[STFlow] Top{args.top_k} HVG 方差范围: "
          f"{hvg_var[topk_idx].min():.3f}-{hvg_var[topk_idx].max():.3f}", flush=True)

    # 4) Top50 平均 PCC + 全基因对照
    y_true, y_p = expr_norm[keep], y_pred[keep]
    all_pccs = gene_mean_pcc(y_true, y_p, "全基因")
    topk_pccs = []
    for g in topk_idx:
        if y_p[:, g].std() == 0 or y_true[:, g].std() == 0:
            continue
        r, _ = pearsonr(y_true[:, g], y_p[:, g])
        if r == r:
            topk_pccs.append(r)
    print(f"[STFlow] Top{args.top_k} HVG 平均 PCC = {np.mean(topk_pccs):.4f} "
          f"（有效 {len(topk_pccs)} 基因）", flush=True)


if __name__ == "__main__":
    main()
