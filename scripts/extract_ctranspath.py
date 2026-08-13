"""提取 CTransPath 特征（Path2Space 方法专用）。

为每个数据目录的 patches 生成 X_ctranspath.npy（768 维），镜像
preprocess_he.py 的 UNI2 特征提取流程，但用官方 CTransPath 冻结模型。

用法（远程服务器）：
    python scripts/extract_ctranspath.py --data_dir ~/HE2ST-pj/data/rep1 \
        --weights_path <官方 ctranspath.pth 路径>
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根目录

CTRANSPATH_WEIGHTS_HINT = "/cpfs01/projects-HDD/cfff-d7ff0c9cdf2f_HDD/hjr_24300980068/HE2ST/path2space/ctranspath.pth"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CTransPath 特征提取")
    p.add_argument("--data_dir", required=True, help="数据集目录（含 metadata.csv + patches/）")
    p.add_argument("--weights_path", default=CTRANSPATH_WEIGHTS_HINT,
                   help="官方 ctranspath.pth 路径")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_cells", type=int, default=None, help="调试：限制细胞数")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import numpy as np
    import pandas as pd
    from PIL import Image

    from common.features.ctranspath import CTransPathExtractor

    data_dir = args.data_dir
    meta = os.path.join(data_dir, "metadata.csv")
    if not os.path.exists(meta):
        raise FileNotFoundError(f"缺少 metadata.csv: {data_dir}")
    if not os.path.exists(args.weights_path):
        raise FileNotFoundError(f"找不到 ctranspath.pth: {args.weights_path}")

    df = pd.read_csv(meta)
    if args.max_cells:
        df = df.iloc[:args.max_cells]
    print(f"加载 {len(df)} 张 patch ...", flush=True)

    extractor = CTransPathExtractor(args.weights_path, device=args.device,
                                    batch_size=args.batch_size)
    print("CTransPath 特征提取 ...", flush=True)
    feats = extractor.extract(
        (Image.open(p).convert("RGB") for p in df["patch_path"])
    )
    out = os.path.join(data_dir, "X_ctranspath.npy")
    np.save(out, feats)
    print(f"特征已保存: {out} 形状 {feats.shape}", flush=True)


if __name__ == "__main__":
    main()
