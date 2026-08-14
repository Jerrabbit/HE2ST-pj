"""CTransPath 特征提取 → data_dir/X_ctranspath.npy（768 维/细胞，Path2Space 输入）。

Path2Space 是冻结模型（154-MLP 集成），输入为 CTransPath 提取的 768 维 tile 特征。
本脚本用官方 path2space-companion 的 CTransPathExtractor（frozen/ctrans + ctranspath.pth）
对 per-cell 256×256 patch（resize 224）提取特征。

用法（远程服务器）：
    python scripts/extract_ctranspath.py --rep 2 \
        --ckpt /cpfs01/.../HE2ST/path2space/ctranspath.pth
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.expanduser("~/HE2ST-pj/path2space/extract"))

from path2space.features import CTransPathExtractor  # noqa: E402
from PIL import Image  # noqa: E402

CTRANSPATH_CKPT = ("/cpfs01/projects-HDD/cfff-d7ff0c9cdf2f_HDD/"
                   "hjr_24300980068/HE2ST/path2space/ctranspath.pth")


def main() -> None:
    p = argparse.ArgumentParser(description="CTransPath 特征提取")
    p.add_argument("--rep", type=int, choices=[1, 2], required=True)
    p.add_argument("--ckpt", default=CTRANSPATH_CKPT)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--max_cells", type=int, default=None, help="调试：限制细胞数")
    args = p.parse_args()

    data_dir = os.path.expanduser(f"~/HE2ST-pj/data/rep{args.rep}")
    meta = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
    if args.max_cells:
        meta = meta.iloc[:args.max_cells]
    print(f"[CTransPath] 提取 {len(meta)} 个 patch ...", flush=True)

    extractor = CTransPathExtractor(args.ckpt, device=args.device,
                                    batch_size=args.batch_size)

    def tiles():
        for _, row in meta.iterrows():
            yield Image.open(row["patch_path"]).convert("RGB")

    feats = extractor.extract(tiles())
    out = os.path.join(data_dir, "X_ctranspath.npy")
    np.save(out, feats)
    print(f"[CTransPath] 已保存 {out} 形状 {feats.shape}", flush=True)


if __name__ == "__main__":
    main()
