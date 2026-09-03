#!/usr/bin/env python
"""SQUALL 官方解码器**就地微调**（不改架构 / 输出维度保持 15757，2026-09-03）。

实验动机（与"自建 313 头"对比）：
- 之前 SQUALL 的训练（SQUALLModel / SQUALLDecoderHead，bench 0.2812 / 0.3281）都是
  **冻结编码器 + 自建新头，输出改为 313**（官方 decoder 15757 被丢弃）。
- 本脚本不同：保留官方 **encoder → decoder_rgb → increase_dim_rgb(15757)** 完整路径，
  只对映射到本仓库 313 面板的**交集基因（264）**所在输出通道给监督（其余 ~15493 通道
  无目标、不参与训练），微调 decoder_rgb + fc1 + 交集通道对应的 fc2 行 —— 真正
  "不改架构、输出仍 15757、只有重叠基因学到信息、其余预测为 0（评测时忽略）"。

实现要点：
- **编码器冻结**：训练不加载 encoder，直接吃预提取 token `X_squall_tokens.npy`
  (N,196,1024)（`scripts/extract_squall.py --save_tokens`），符合统一规则
  （冻结编码器、训练头/解码器）。
- **省显存 + 与官方同运算顺序**：fc2 线性层理论上可与 56×56 空间均值交换，但 fp32 大数下
  "先均值再过 fc2" 会引入 ~1e-2 舍入差；故实现保持官方顺序——对 `relu(fc1)` 上采样出的
  3136(=56×56) 栅格 cell 过 **fc2 子集（264 行）**再均值，与官方 (B,56,56,15757)→mean
  在所选通道上数值一致，且只物化 264 列、不产生 15757。
- **评测只在交集基因上算指标**（预测/真值数组本身就是 264 交集子集；README 已注明）。

数据/协议：rep1(训练) → rep2(验证子集早停 + 全量测试)；目标空间 log1p_zscore
（统计量只在训练集拟合）；验证 rep2 子集 PCC 早停（patience），与统一协议一致。

用法（远程，conda 环境含 torch/pandas/scipy，SQUALL 官方 config/ckpt 在项目内）：
    python scripts/squall_decoder_inplace_finetune.py train \
        --train_dir data/rep1 --valid_dir data/rep2 \
        --output_dir outputs/bench_squall_decoder_inplace_ft
    python scripts/squall_decoder_inplace_finetune.py test \
        --test_dir data/rep2 \
        --ckpt outputs/bench_squall_decoder_inplace_ft/best.pt \
        --output_dir outputs/bench_squall_decoder_inplace_ft
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根目录

CONFIG = os.path.expanduser("~/HE2ST-pj/codes/squall/SQUALL_Tutorial/config.yaml")
CKPT = os.path.expanduser("~/HE2ST-pj/weights/squall/SQUALL_full.pth")
GENE_MAP = os.path.expanduser("~/HE2ST-pj/codes/squall/SQUALL_Tutorial/gene_token_homologs.csv")
EXPR_CHANS = 15757  # 官方表达通道数（increase_dim_rgb 输出宽），脚本不改它


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SQUALL 官方解码器就地微调（15757 通道，交集基因监督）")
    sub = p.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=CONFIG)
    common.add_argument("--ckpt_src", default=CKPT, help="官方 SQUALL_full.pth")
    common.add_argument("--gene_map", default=GENE_MAP)
    common.add_argument("--device", default="cuda")
    common.add_argument("--tokens_dir", default=None,
                        help="token 缓存目录（cpfs mmap 随机读若卡 D-state，可复制到本地盘后传此路径）")
    common.add_argument("--batch_size", type=int, default=32)
    common.add_argument("--train_max_cells", type=int, default=None)
    common.add_argument("--eval_subset", type=int, default=8000, help="验证采样子集（早停信号）")
    common.add_argument("--output_dir", required=True)

    p_train = sub.add_parser("train", parents=[common])
    p_train.add_argument("--train_dir", required=True)
    p_train.add_argument("--valid_dir", required=True)
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--patience", type=int, default=10)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--weight_decay", type=float, default=1e-2)

    p_test = sub.add_parser("test", parents=[common])
    p_test.add_argument("--test_dir", required=True)
    p_test.add_argument("--ckpt", required=True, help="train 保存的 best.pt")
    p_test.add_argument("--test_max_cells", type=int, default=None)
    return p.parse_args()


# ----------------------------------------------------------------------------- helpers

def _tok_path(data_dir: str, tokens_dir: str | None) -> str:
    """按现有约定定位 token 文件：给了 tokens_dir 就用其子目录（同名），否则在 data_dir。"""
    if tokens_dir:
        return os.path.join(tokens_dir, os.path.basename(data_dir.rstrip("/")), "X_squall_tokens.npy")
    return os.path.join(data_dir, "X_squall_tokens.npy")


def _norm_expr(raw: np.ndarray, ref_stats: dict | None = None):
    from common.data.expression import normalize_expression
    return normalize_expression(raw, "log1p_zscore", ref_stats)


def load_covered(data_dir: str, gene_map_path: str):
    """返回本仓库基因里能映射到 SQUALL 15757 通道的交集。

    Returns:
        our_cols: (K,) 交集基因在 data_dir 313 面板中的列号（升序）
        channels: (K,) 对应 SQUALL 输出通道号（gene_token_homologs.csv 首列 id）
        names:    (K,) 交集基因名（our 面板顺序，与 our_cols 对应）
    """
    import numpy as np
    import pandas as pd
    from common.benchmark.harness import load_gene_names

    our = load_gene_names(data_dir)
    df = pd.read_csv(gene_map_path)
    # 首列 = 15757 通道 id；HGNC_symbol = 基因名（test_squall_decoder 同约定）
    sym2ch: dict[str, int] = {}
    for sym, ch in zip(df["HGNC_symbol"].astype(str), df.iloc[:, 0]):
        sym = sym.strip()
        if sym and sym != "nan":
            try:
                sym2ch[sym] = int(float(ch))
            except (ValueError, TypeError):
                continue
    pairs = [(c, sym2ch[g]) for c, g in enumerate(our) if g in sym2ch]
    pairs.sort()  # 按 our 面板列号升序 → K 个交集基因
    our_cols = np.array([c for c, _ in pairs], dtype=int)
    channels = np.array([ch for _, ch in pairs], dtype=int)
    assert channels.min() >= 0 and channels.max() < EXPR_CHANS, \
        f"通道号越界 [{channels.min()}, {channels.max()}] vs 15757"
    return our_cols, channels, [our[c] for c in our_cols]


def build_token_dataset(data_dir: str, tokens_dir: str | None, expr_norm: np.ndarray,
                        coords: np.ndarray, start: int, stop: int):
    import numpy as np
    import torch
    from torch.utils.data import Dataset

    class TokExprDS(Dataset):
        """(X_squall_tokens.npy 行, log1p_zscore 表达, 坐标)，行序与 metadata 对齐。"""

        def __init__(self, tokens_mmap, expr_norm, coords):
            self.tokens = tokens_mmap   # (N,196,1024) fp32 mmap
            self.expr = expr_norm
            self.coords = coords

        def __len__(self):
            return len(self.expr)

        def __getitem__(self, i):
            return {"feature": torch.from_numpy(self.tokens[i].astype(np.float32).copy()),
                    "gene_expr": torch.from_numpy(self.expr[i]),
                    "coords": torch.from_numpy(self.coords[i])}

    mmap = np.load(_tok_path(data_dir, tokens_dir), mmap_mode="r")
    return TokExprDS(mmap[start:stop], expr_norm[start:stop], coords[start:stop])


def load_metadata(data_dir: str):
    import numpy as np
    import pandas as pd

    meta = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
    coords = meta[["x_centroid", "y_centroid"]].to_numpy(np.float32)
    return meta, coords


# ----------------------------------------------------------------------------- model

class SquallDecoderInplace(torch.nn.Module):
    """官方 decoder 就地微调：decoder_rgb + increase_dim_rgb(fc1/relu) + fc2 交集行。

    输入 (B,196,1024) token → decoder_rgb → fc1→relu → 空间均值(线性 fc2 可交换)
    → fc2_sub → (B,K) log1p_zscore 预测（只含交集基因 K 列）。输出宽只取 K 个通道，
    与官方 15757 全量输出在所选通道上逐位一致（未选通道不产生、不训练 → 训练时
    等价于全 15757 + loss mask，评测"其他预测为 0/忽略"）。
    """

    input_type = "feature"
    feature_file = "X_squall_tokens.npy"

    def __init__(self, decoder, channels):
        super().__init__()
        import numpy as np
        import torch

        ch = np.asarray(channels, dtype=int)
        fc2 = decoder.increase_dim_rgb.fc2          # Linear(C0//up → 15757)
        # 官方 fc2 的交集行 → 本模块 fc2_sub（其余行不动、也不进入可训练参数）
        fc2_sub = torch.nn.Linear(fc2.in_features, len(ch), bias=fc2.bias is not None)
        with torch.no_grad():
            fc2_sub.weight.copy_(fc2.weight[ch])
            if fc2.bias is not None:
                fc2_sub.bias.copy_(fc2.bias[ch])
        self.fc2_sub = fc2_sub
        # 只保留需要的官方子模块；fc2（全量）与 decoder_expr 不参与 forward/训练 → 摘除
        self.decoder = decoder
        decoder.increase_dim_rgb.fc2 = None
        decoder.decoder_expr = None

    def forward(self, x):
        """(B,196,1024) → (B,K)，与官方 forward_rgb_to_expr 的通道子集**同运算顺序**。

        先对 3136(=56×56) 个栅格 cell 过 fc2_sub 再均值，而不是先均值再过 fc2 ——
        与官方 (B,56,56,15757)→mean 在所选通道上数值一致（fc2 线性层与均值数学上可交换，
        但 fp32 大数下先均值会引入 ~1e-2 舍入差，故保持官方运算顺序）。只物化 K=264 列，
        不产生 15757。
        """
        z = self.decoder.decoder_rgb(x)             # (B,196,1024)
        B, L, _ = z.shape
        H = int(round(L ** 0.5))
        assert H * H == L, f"token 数非平方：{L}"
        inc = self.decoder.increase_dim_rgb
        up = inc.up_scale
        h = inc.relu(inc.fc1(z.reshape(B, H, H, -1)))       # (B,H,H,up*C0)
        grid = h.reshape(B, H * H * up * up, -1)            # (B, 3136, C0//up)
        return self.fc2_sub(grid).mean(dim=1)               # (B,K)


def load_decoder_inplace(config_path: str, ckpt_path: str, channels,
                         device: str) -> SquallDecoderInplace:
    """构建官方 Squall（严格加载权重）→ 摘取 decoder → 就地微调模块。"""
    import yaml

    from methods.squall.Squall import Squall

    class AttrDict(dict):
        def __getattr__(self, k):
            try:
                return self[k]
            except KeyError:
                raise AttributeError(k)

    with open(config_path) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    model = Squall(AttrDict(cfg["model"]))
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    dec = model.decoder  # 官方 SquallDecoder（decoder_rgb + increase_dim_rgb）
    wrapper = SquallDecoderInplace(dec, channels)
    # 释放 encoder 与中间对象（训练只用预提取 token），减少 CPU/GPU 内存
    del model, dec, state
    import gc
    gc.collect()
    return wrapper.to(device)


def eval_intersection(model, loader, device, stats) -> dict:
    """在交集基因上算全量指标（loader 的 gene_expr 宽 = K，全部为交集，无需 gene_idx）。"""
    from common.benchmark.harness import evaluate
    return evaluate(model, loader, device, "log1p_zscore", stats, details=True, ssim=False)


# ----------------------------------------------------------------------------- main

def _prepare(data_dir: str, tokens_dir: str | None, our_cols: np.ndarray,
             ref_stats: dict | None, max_cells: int | None):
    """读 meta/表达，抽交集列 + 归一化。返回 (expr_norm (N,K), raw (N,K), coords (N,2), n)。"""
    import numpy as np
    meta, coords = load_metadata(data_dir)
    n = len(meta) if max_cells is None else min(max_cells, len(meta))
    raw = np.load(os.path.join(data_dir, "gene_expression.npy"))[:n][:, our_cols].astype(np.float32)
    norm, stats = _norm_expr(raw, ref_stats) if ref_stats is not None else _norm_expr(raw)
    return norm, raw, coords[:n], n, stats


def main() -> None:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset

    from common.benchmark.harness import evaluate, save_eval_results_csv, scalar_results

    args = parse_args()
    device = args.device

    # 交集基因映射只在 train 需要（存进 ckpt）；test 完全依赖 ckpt 记录，可离线重建
    if args.mode == "train":
        our_cols, channels, names = load_covered(args.train_dir, args.gene_map)
        K = len(our_cols)
        print(f"[Squall-Inplace] 交集基因 {K}/313（通道号 min/max={channels.min()}/{channels.max()}），"
              f"官方输出仍 {EXPR_CHANS} 宽，只监督/评测这 {K} 个", flush=True)

    if args.mode == "train":
        os.makedirs(args.output_dir, exist_ok=True)

        tr_norm, tr_raw, tr_coords, n_tr, stats = _prepare(
            args.train_dir, args.tokens_dir, our_cols, None, args.train_max_cells)
        va_norm, va_raw, va_coords, n_va, _ = _prepare(
            args.valid_dir, args.tokens_dir, our_cols, stats, None)

        tr_ds = build_token_dataset(args.train_dir, args.tokens_dir, tr_norm, tr_coords, 0, n_tr)
        va_ds = build_token_dataset(args.valid_dir, args.tokens_dir, va_norm, va_coords, 0, n_va)
        if len(va_ds) > args.eval_subset:
            idx = np.random.default_rng(0).choice(len(va_ds), args.eval_subset, replace=False)
            va_ds = Subset(va_ds, idx)

        tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=4, pin_memory=True)
        va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)

        model = load_decoder_inplace(args.config, args.ckpt_src, channels, device)
        print(f"[Squall-Inplace] 训练集 {n_tr} 验证集 {n_va}(取 {args.eval_subset})，"
              f"K={K}，可训练参数 {sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_fn = torch.nn.MSELoss()
        best_pcc, best_state, no_improve, history = -float("inf"), None, 0, []

        for epoch in range(1, args.epochs + 1):
            model.train()
            total, n = 0.0, 0
            for batch in tr_dl:
                x = batch["feature"].to(device)
                y = batch["gene_expr"].to(device)
                opt.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                opt.step()
                total += loss.item() * y.size(0)
                n += y.size(0)
            model.eval()
            ev = evaluate(model, va_dl, device, "log1p_zscore", stats, ssim=False)
            history.append({"epoch": epoch, "train_mse": total / max(n, 1), **scalar_results(ev)})
            print(f"[Squall-Inplace ep{epoch}/{args.epochs}] mse={total/max(n,1):.4f} "
                  f"val_PCC={ev['PCC']:.4f}", flush=True)
            if ev["PCC"] == ev["PCC"] and ev["PCC"] > best_pcc + 1e-4:
                best_pcc, no_improve = ev["PCC"], 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"[Squall-Inplace] early stop @ ep{epoch}", flush=True)
                    break

        # 保存：模型 + 通道/名字 + 归一化统计量（测试时重建只需这些）
        torch.save({"model": best_state if best_state is not None
                    else {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "config": {"method": "squall_decoder_inplace_ft", "num_channels": K,
                               "channels": channels.tolist(), "gene_names": names,
                               "covered_our_cols": our_cols.tolist(),
                               "gene_norm": "log1p_zscore",
                               "stats": {k: v.tolist() for k, v in stats.items()}}},
                   os.path.join(args.output_dir, "best.pt"))
        with open(os.path.join(args.output_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=2)
        print(f"[Squall-Inplace] 训练完成 best val_PCC={best_pcc:.4f}\n"
              f"  测试: python scripts/squall_decoder_inplace_finetune.py test "
              f"--test_dir {args.valid_dir} --ckpt {args.output_dir}/best.pt "
              f"--output_dir {args.output_dir}", flush=True)

    else:  # test
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        cfg, stats = ck["config"], ck["config"]["stats"]
        stats = {"means": np.asarray(stats["means"], dtype=np.float32),
                 "stds": np.asarray(stats["stds"], dtype=np.float32)}
        # 用 ckpt 记录的通道/名字重建（与训练一致），并核对数据交集一致
        channels = np.asarray(cfg["channels"], dtype=int)
        our_cols_c = np.asarray(cfg["covered_our_cols"], dtype=int)
        K = len(channels)
        te_norm, te_raw, te_coords, n_te, _ = _prepare(
            args.test_dir, args.tokens_dir, our_cols_c, stats, args.test_max_cells)
        te_ds = build_token_dataset(args.test_dir, args.tokens_dir, te_norm, te_coords, 0, n_te)
        te_dl = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=4, pin_memory=True)
        model = load_decoder_inplace(args.config, args.ckpt_src, channels, args.device)
        model.load_state_dict(ck["model"])
        model.eval()
        results = evaluate(model, te_dl, args.device, "log1p_zscore", stats,
                           topk_ks="full", details=True, ssim=True)
        results["_gene_names"] = cfg["gene_names"]
        os.makedirs(args.output_dir, exist_ok=True)
        json_results = scalar_results(results)
        with open(os.path.join(args.output_dir, "test_results.json"), "w") as f:
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        csv_files = save_eval_results_csv(os.path.join(args.output_dir, "eval_metrics.csv"),
                                          results, gene_names=cfg["gene_names"])
        print(json.dumps(json_results, ensure_ascii=False, indent=2), flush=True)
        print(f"[Squall-Inplace] CSV: {csv_files['summary']}", flush=True)


if __name__ == "__main__":
    main()
