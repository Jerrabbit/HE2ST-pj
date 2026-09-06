#!/usr/bin/env python
"""把同学上传特征（dataset-final/hbc{1,2_filter}.h5ad, obsm）对齐到我们 rep1/rep2 的细胞序。

输出 outputs/classmate_feat/rep{N}/{key}.npy（按我们 data/rep{N}/metadata.csv 的 cell_id 序，
与 rep1→rep2 基准同细胞同目标，便于"是否特征问题"的同口径归因）。无匹配的细胞剔除并打印数量。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REP_H5 = {1: "hbc1.h5ad", 2: "hbc2_filter.h5ad"}
KEYS = ["DAVID_BLIP2_features", "UNI_features"]
BASE_H5 = "/cpfs01/projects-HDD/cfff-d7ff0c9cdf2f_HDD/hjr_24300980068/dataset-final"
OURS = "/cpfs01/projects-HDD/cfff-d7ff0c9cdf2f_HDD/hjr_24300980068/HE2ST-pj/data/rep{N}"
OUT = "/cpfs01/projects-HDD/cfff-d7ff0c9cdf2f_HDD/hjr_24300980068/HE2ST-pj/outputs/classmate_feat/rep{N}"


def main() -> None:
    import anndata as ad

    for rep in (1, 2):
        h5 = os.path.join(BASE_H5, REP_H5[rep])
        a = ad.read_h5ad(h5, backed="r")
        ids = np.asarray(a.obs["cell_id"] if "cell_id" in a.obs.columns
                         else a.obs_names).astype(np.int64)
        # 我们的目标细胞序（与基准完全相同）
        meta = pd.read_csv(OURS.format(N=rep) + "/metadata.csv")
        ours_ids = meta["cell_id"].to_numpy(dtype=np.int64)
        pos = {int(c): i for i, c in enumerate(ours_ids)}
        # h5 行 → 我们位置（仅取交集，按我们序）
        match = np.array([pos.get(int(c), -1) for c in ids])
        rows = np.flatnonzero(match >= 0)
        if rows.size == 0:
            print(f"rep{rep}: 0 match!", flush=True)
            continue
        # rows 按我们序排序后取行索引 → 一次性 gather 最快（sort order gives our positions ascending）
        order = np.argsort(match[rows])
        h5_rows = rows[order]
        our_pos = match[rows][order]          # == arange(len) 若全匹配
        print(f"rep{rep}: h5={ids.size} 匹配={h5_rows.size} (我们 {ours_ids.size}); "
              f"缺失={ours_ids.size - h5_rows.size}", flush=True)
        out_dir = OUT.format(N=rep)
        os.makedirs(out_dir, exist_ok=True)
        for k in KEYS:
            F = a.obsm[k]                       # (N, d)
            if F.shape[0] != ids.size:
                print(f"  !! {k} 行数 {F.shape[0]} != obs {ids.size}", flush=True)
                continue
            Fe = np.asarray(F[h5_rows], dtype=np.float32)
            out = os.path.join(out_dir, k + ".npy")
            np.save(out, Fe)
            print(f"  {k} -> {out} 形状 {Fe.shape}", flush=True)
        a.file.close()


if __name__ == "__main__":
    main()
