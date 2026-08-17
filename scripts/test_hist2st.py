"""Hist2ST 专用评估：加载 best.pt，对测试切片做整切片 ROI 图推理评估。

用法（远程服务器）：
    python scripts/test_hist2st.py --ckpt outputs/hist2st/best.pt --test_dir ~/HE2ST-pj/data/rep2 \
        --output_dir outputs/hist2st
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根目录


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hist2ST 测试集评估（加载 best.pt）")
    p.add_argument("--ckpt", required=True, help="best.pt 路径（含 model/history/config）")
    p.add_argument("--test_dir", required=True, help="测试集数据目录")
    p.add_argument("--output_dir", default="outputs/hist2st")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main() -> None:
    import json

    import torch

    import methods.hist2st as hist2st
    from methods.hist2st.model import Hist2STModel

    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    config = ckpt.get("config", {})
    num_genes = config.get("num_genes", 313)
    fig_size = config.get("fig_size", 112)
    n_pos = config.get("n_pos", 64)
    gene_norm = config.get("gene_norm", "log1p_norm_total")
    stats = config.get("stats")
    if stats is None and gene_norm == "log1p_zscore":
        print("警告: best.pt 未存归一化统计量，log1p_zscore 逆变换会退化为 log1p 语义", flush=True)

    # zinb>0 时训练侧模型含 ZINB core（core.mean/disp/pi/coef），test 侧构造需一致；
    # 旧 checkpoint 的 config 未存 zinb，从 state_dict 自动检测（ZINB core 存在即需建）。
    zinb = config.get("zinb", 0)
    if not zinb:
        sd = ckpt["model"]
        if any(k.startswith("core.mean.") or k.startswith("core.disp.")
               or k.startswith("core.pi.") or k.startswith("core.coef.") for k in sd):
            zinb = 1
            print("[Hist2ST] 从 state_dict 检测到 ZINB core，按 zinb>0 构造模型", flush=True)

    model = Hist2STModel(num_genes=num_genes, fig_size=fig_size, n_pos=n_pos, zinb=zinb)
    model.load_state_dict(ckpt["model"])
    model.to(args.device)
    print(f"[Hist2ST] 加载 best.pt: num_genes={num_genes} fig_size={fig_size} "
          f"gene_norm={gene_norm}", flush=True)

    results = hist2st.evaluate_slide(model, args.test_dir, gene_norm, stats,
                                     args.device, args.output_dir)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"结果已保存: {os.path.join(args.output_dir, 'test_results.json')}")


if __name__ == "__main__":
    main()
