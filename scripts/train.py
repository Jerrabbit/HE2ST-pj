"""统一训练入口：所有方法共用同一训练流程，保证公平比较（课题要求 4、6）。

各方法模型统一接口（methods/<name>/model.py）：
    class SomeModel(nn.Module):
        input_type = 'patch' | 'feature'      # 决定用 HESTDataset 还是 FeatureDataset
        def forward(self, x) -> (B, G) 归一化表达预测

用法（远程服务器）：
    python scripts/train.py --method uni2_mlp --train_dir ~/HE2ST-pj/data/rep1 \
        --valid_dir ~/HE2ST-pj/data/rep2 --epochs 50 --lr 1e-3 --output_dir outputs/uni2_mlp
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根目录

from common.benchmark.harness import fit
from common.data.dataset import FeatureDataset, HESTDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HE→ST 统一训练入口")
    p.add_argument("--method", required=True, help="方法名，对应 methods/ 下文件夹")
    p.add_argument("--train_dir", required=True, help="训练集数据目录")
    p.add_argument("--valid_dir", required=True, help="验证集数据目录")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--gene_norm", choices=["log1p_zscore", "log1p_norm_total", "none"],
                   default="log1p_zscore")
    p.add_argument("--gene_file", default=None, help="公共基因列表文件（每行一个基因名）")
    p.add_argument("--img_size", type=int, default=0, help="patch 输入时 resize 到该尺寸（0=原图）")
    p.add_argument("--output_dir", default="outputs", help="模型与日志输出目录")
    p.add_argument("--device", default="cuda")
    p.add_argument("--debug", action="store_true", help="仅前 100 样本快速验证")
    return p.parse_args()


def _make_dataset(model, data_dir, gene_list, gene_norm, ref_stats, img_size, debug):
    if getattr(model, "input_type", "patch") == "feature":
        feature_file = getattr(model, "feature_file", None)  # 方法自带特征文件（如 X_ctranspath.npy）
        return FeatureDataset(data_dir, feature_path=feature_file, gene_list=gene_list,
                              gene_norm=gene_norm, ref_stats=ref_stats, debug=debug)
    return HESTDataset(data_dir, gene_list=gene_list, gene_norm=gene_norm,
                       ref_stats=ref_stats, img_size=img_size, debug=debug)


def _load_method(method_name: str, num_genes: int):
    """导入方法模块并构造模型。"""
    mod = importlib.import_module(f"methods.{method_name}")
    return mod.build_model(num_genes=num_genes)


def main() -> None:
    args = parse_args()

    gene_list = None
    if args.gene_file:
        with open(args.gene_file) as f:
            gene_list = [line.strip() for line in f if line.strip()]

    # 训练集上拟合表达归一化统计量，验证/测试集复用（避免泄漏）
    probe = HESTDataset(args.train_dir, gene_list=gene_list, gene_norm="none", debug=True)
    num_genes = len(probe.gene_list)
    gene_list = probe.gene_list
    print(f"公共基因数: {num_genes}", flush=True)

    model = _load_method(args.method, num_genes)
    print(f"[{args.method}] input_type={getattr(model, 'input_type', 'patch')} "
          f"参数量={sum(p.numel() for p in model.parameters())}", flush=True)

    train_ds = _make_dataset(model, args.train_dir, gene_list, args.gene_norm,
                             None, args.img_size, args.debug)
    stats = train_ds.stats
    valid_ds = _make_dataset(model, args.valid_dir, gene_list, args.gene_norm,
                             stats, args.img_size, args.debug)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    mod = importlib.import_module(f"methods.{args.method}")
    if hasattr(mod, "train_function"):
        # 方法自带训练流程（如 BLEEP 对比学习、Hist2ST 负二项），复用同一数据加载与评估
        mod.train_function(model, train_loader, valid_loader, args, stats)
        print(f"训练完成（自定义流程），最优模型保存于 {args.output_dir}/best.pt")
    else:
        fit(model, train_loader, valid_loader, args.epochs, args.lr, args.device,
            out_dir=args.output_dir, weight_decay=args.weight_decay,
            gene_norm=args.gene_norm, eval_stats=stats,
            config={"method": args.method, "num_genes": num_genes,
                    "gene_norm": args.gene_norm, "gene_list": gene_list})
        print(f"训练完成，最优模型保存于 {args.output_dir}/best.pt")


if __name__ == "__main__":
    main()
