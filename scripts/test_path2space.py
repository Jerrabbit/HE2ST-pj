"""Path2Space 专用评估：冻结官方模型（154-MLP 集成 + CTransPath 特征）。

Path2Space 是推理专用方法（无训练），走独立评估路径。需要：
    1. 测试集数据目录（含 X_ctranspath.npy，由 extract_ctranspath.py 生成）
    2. 官方权重（ensemble_dir 含 result_{ik}_{il}_0/model_trained.pth）
    3. 官方基因表 genes.txt

用法（远程服务器）：
    python scripts/test_path2space.py --test_dir ~/HE2ST-pj/data/rep2 \
        --ensemble_dir <官方 mlp_ensemble 解压目录> \
        --genes_txt <官方 genes.txt>
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根目录

PATH2SPACE_ENSEMBLE_HINT = "/cpfs01/projects-HDD/cfff-d7ff0c9cdf2f_HDD/hjr_24300980068/HE2ST/path2space/mlp_ensemble"
PATH2SPACE_GENES_HINT = "/cpfs01/projects-HDD/cfff-d7ff0c9cdf2f_HDD/hjr_24300980068/HE2ST/path2space/genes.txt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Path2Space 冻结模型评估")
    p.add_argument("--test_dir", required=True, help="测试集数据目录")
    p.add_argument("--ensemble_dir", default=PATH2SPACE_ENSEMBLE_HINT,
                   help="官方 154-MLP 集成目录")
    p.add_argument("--genes_txt", default=PATH2SPACE_GENES_HINT,
                   help="官方 14068 基因表")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default="outputs/path2space")
    p.add_argument("--output_is_log1p", type=bool, default=True,
                   help="官方输出是否为 log1p 尺度（默认 True，转 raw counts）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import json

    import methods.path2space as p2s
    from methods.path2space.model import Path2SpaceModel

    if not os.path.isdir(args.ensemble_dir):
        raise FileNotFoundError(f"找不到官方集成目录: {args.ensemble_dir}")
    gene_names = None
    gn_path = os.path.join(args.test_dir, "gene_names.txt")
    if os.path.exists(gn_path):
        with open(gn_path) as f:
            gene_names = [ln.strip() for ln in f if ln.strip()]
    else:
        print(f"警告: {gn_path} 不存在，将从 --genes_txt 前 num_genes 推断", flush=True)

    num_genes = len(gene_names) if gene_names else 313
    model = Path2SpaceModel(
        num_genes=num_genes, ensemble_dir=args.ensemble_dir,
        genes_txt=args.genes_txt, gene_names=gene_names,
        output_is_log1p=args.output_is_log1p, device=args.device,
    )
    print(f"Path2Space 冻结模型就绪: {len(model.out_indices)} 个公共基因有输出映射",
          flush=True)

    results = p2s.evaluate_frozen(
        model, args.test_dir, gene_names=gene_names,
        batch_size=args.batch_size, device=args.device,
        output_dir=args.output_dir,
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"结果已保存: {os.path.join(args.output_dir, 'test_results.json')}")


if __name__ == "__main__":
    main()
