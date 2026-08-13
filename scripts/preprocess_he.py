"""H&E 数据预处理入口：h5ad → H&E patch → 保存；可选 UNI2 特征提取。

用法（远程服务器，正式跑时先停止保活脚本 pkill -f gpu_keepalive.py）：
    # 1) 提取 H&E patches（从 h5ad 的 image_col/image_row 坐标）
    python scripts/preprocess_he.py --rep 1 --stage patches --output_dir ~/HE2ST-pj/data/rep1

    # 2) 用 UNI2 提取特征（需要 GPU）
    python scripts/preprocess_he.py --rep 1 --stage features \
        --data_dir ~/HE2ST-pj/data/rep1 \
        --output X_rep1_uni2.npy

    # 调试：--max_cells 限制细胞数
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根目录

BASE = "/cpfs01/projects-HDD/cfff-d7ff0c9cdf2f_HDD/hjr_24300980068/HE2ST/datasets"
REP_H5AD = {
    1: os.path.join(BASE, "Human_Breast_Cancer_Rep1_uni_resolution64_full.h5ad"),
    2: os.path.join(BASE, "Human_Breast_Cancer_Rep2_uni_resolution64_full.h5ad"),
}
REP_HE = {
    1: os.path.join(BASE, "he_images", "Xenium_FFPE_Human_Breast_Cancer_Rep1_he_image.ome.tif"),
    2: os.path.join(BASE, "he_images", "Xenium_FFPE_Human_Breast_Cancer_Rep2_he_image.ome.tif"),
}
UNI2_WEIGHTS = "/cpfs01/projects-HDD/cfff-d7ff0c9cdf2f_HDD/hjr_24300980068/HE2ST/uni2_model/pytorch_model.bin"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="H&E 数据预处理")
    p.add_argument("--stage", choices=["patches", "features"], required=True)
    p.add_argument("--rep", type=int, choices=[1, 2], default=1, help="Rep 编号")
    p.add_argument("--patch_size", type=int, default=256)
    p.add_argument("--max_cells", type=int, default=None, help="调试：限制细胞数")
    p.add_argument("--output_dir", default=None, help="patches 阶段输出目录")
    p.add_argument("--data_dir", default=None, help="features 阶段输入数据集目录")
    p.add_argument("--output", default=None, help="features 阶段输出 .npy 路径")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def stage_patches(args: argparse.Namespace) -> None:
    from common.data.preprocess import extract_patches_from_h5ad

    out = args.output_dir or os.path.expanduser(f"~/HE2ST-pj/data/rep{args.rep}")
    _, kept_ids = extract_patches_from_h5ad(
        REP_H5AD[args.rep], REP_HE[args.rep], out,
        patch_size=args.patch_size, max_cells=args.max_cells,
    )
    # 同时导出表达矩阵与基因名，供 HESTDataset 读取；只保留成功提 patch 的细胞
    import anndata as ad
    import numpy as np
    import pandas as pd

    import scipy.sparse as sp

    adata = ad.read_h5ad(REP_H5AD[args.rep], backed="r")
    if adata.raw is not None:
        X = adata.raw.X
    else:
        X = adata.X
    keep_pos = pd.Index(adata.obs_names).get_indexer(kept_ids)
    if sp.issparse(X):
        expr = np.asarray(X[keep_pos].todense()).astype(np.float32)
    else:
        expr = np.asarray(X[keep_pos]).astype(np.float32)
    # h5ad 的 X 已做 log1p（uns['log1p']）。逆变换恢复 raw counts，
    # 与既有 xenium_rep1/gene_expression.npy 约定一致（Dataset 再自行 log1p_zscore）。
    expr = np.round(np.expm1(expr)).astype(np.float32)
    np.save(os.path.join(out, "gene_expression.npy"), expr)
    with open(os.path.join(out, "gene_names.txt"), "w") as f:
        f.write("\n".join(adata.var_names) + "\n")
    nnz = expr[expr != 0]
    print(f"gene_expression.npy {expr.shape} 已写入 {out}")
    print(f"raw counts 诊断: max={expr.max():.3f} nnz均值={nnz.mean() if len(nnz) else 0:.4f} "
          f"整数={bool(np.allclose(expr, np.round(expr)))}")


def stage_features(args: argparse.Namespace) -> None:
    import numpy as np

    from common.features.uni2 import UNI2FeatureExtractor

    data_dir = args.data_dir or os.path.expanduser(f"~/HE2ST-pj/data/rep{args.rep}")
    meta = os.path.join(data_dir, "metadata.csv")
    patches_dir = os.path.join(data_dir, "patches")
    if not os.path.exists(meta):
        raise FileNotFoundError(f"请先运行 --stage patches 生成 {data_dir}")

    import pandas as pd
    from PIL import Image

    df = pd.read_csv(meta)
    if args.max_cells:
        df = df.iloc[:args.max_cells]
    print(f"加载 {len(df)} 张 patch ...")
    patches = np.stack([np.array(Image.open(p).convert("RGB")) for p in df["patch_path"]])

    extractor = UNI2FeatureExtractor(UNI2_WEIGHTS, device=args.device)
    print("UNI2 特征提取 ...")
    feats = extractor.extract(patches, batch_size=args.batch_size)
    out = args.output or os.path.join(data_dir, "X_uni2.npy")
    np.save(out, feats)
    print(f"特征已保存: {out} 形状 {feats.shape}")


def main() -> None:
    args = parse_args()
    if args.stage == "patches":
        stage_patches(args)
    else:
        stage_features(args)


if __name__ == "__main__":
    main()
