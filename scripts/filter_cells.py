"""细胞过滤：去除低表达/空白细胞，产出过滤后数据目录（特征文件按行切片复用）。

QC 指标（标准 Xenium 质控）：
    n_genes = 表达>0 的基因数；umis = 总 UMI（raw counts 求和）
    保留 n_genes >= --min_genes 且 umis >= --min_umis 的细胞。

产出 out_dir/：
    metadata.csv / gene_expression.npy / gene_names.txt / patches/（保留细胞）
    X_*.npy：源目录中首维==N 的特征文件按保留 mask 切行（2D/3D 均可，mmap 分块写，
    避免大文件（如 SQUALL token 131GB）整载入内存）。

用法（远程 myenv1）：
    # 先 --dry-run 看各阈值保留比例，再正式执行
    python scripts/filter_cells.py --src_dir data/rep1 --out_dir data/rep1_f \
        --min_genes 200 --min_umis 500 --dry-run
    python scripts/filter_cells.py --src_dir data/rep1 --out_dir data/rep1_f \
        --min_genes 200 --min_umis 500
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil

import numpy as np


def _load_src(src_dir: str):
    import pandas as pd

    meta = pd.read_csv(os.path.join(src_dir, "metadata.csv"))
    expr = np.load(os.path.join(src_dir, "gene_expression.npy"))
    gene_names = None
    gn_path = os.path.join(src_dir, "gene_names.txt")
    if os.path.exists(gn_path):
        with open(gn_path) as f:
            gene_names = [ln.strip() for ln in f if ln.strip()]
    return meta, expr, gene_names


def qc_mask(expr: np.ndarray, min_genes: int, min_umis: int) -> np.ndarray:
    n_genes = (expr > 0).sum(axis=1)
    umis = expr.sum(axis=1)
    return (n_genes >= min_genes) & (umis >= min_umis)


def _subset_feature(src_path: str, out_path: str, idx: np.ndarray):
    """把特征文件按 idx 切行写新文件（支持 2D/3D，mmap 分块避免整载大文件）。"""
    arr = np.load(src_path, mmap_mode="r")
    if arr.ndim not in (2, 3):
        print(f"  跳过 {os.path.basename(src_path)}：ndim={arr.ndim}", flush=True)
        return
    out = np.lib.format.open_memmap(out_path, mode="w+", dtype=arr.dtype,
                                    shape=(len(idx),) + arr.shape[1:])
    CHUNK = 200_000
    for s in range(0, len(idx), CHUNK):
        b = idx[s:s + CHUNK]
        out[s:s + len(b)] = arr[b]
    out.flush()
    print(f"  {os.path.basename(src_path)}: ({arr.shape[0]},{arr.shape[1:]}) "
          f"→ ({len(idx)},{arr.shape[1:]})", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="细胞过滤（低表达/空白剔除）")
    p.add_argument("--src_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--min_genes", type=int, default=200, help="最少检测基因数")
    p.add_argument("--min_umis", type=int, default=500, help="最少总 UMI")
    p.add_argument("--dry_run", action="store_true", help="只打印统计，不写文件")
    p.add_argument("--copy_patches", action="store_true", default=True,
                   help="复制保留细胞的 patch（默认开）")
    p.add_argument("--no_copy_patches", action="store_true",
                   help="不复制 patch（metadata 保留原路径引用；原 rep1/rep2 patches 持久，"
                        "patch 类方法可直读，省数十分钟 HDD 小文件拷贝）")
    p.add_argument("--subset_features", action="store_true", default=True,
                   help="按行切片 X_*.npy 特征文件（默认开）")
    p.add_argument("--exclude_features", default=None,
                   help="逗号分隔：跳过这些特征文件的切片（如大文件 X_phoenix_dino.npy，"
                        "11 方法不需要，避免数小时无谓 IO）")
    p.add_argument("--ghist_src", default=None,
                   help="GHIST 数据目录（可选，过滤其 CSV 表达/核/avgexp）")
    p.add_argument("--ghist_out", default=None,
                   help="过滤后 GHIST 数据目录（与 --ghist_src 成对给出）")
    args = p.parse_args()

    meta, expr, gene_names = _load_src(args.src_dir)
    N = expr.shape[0]
    mask = qc_mask(expr, args.min_genes, args.min_umis)
    idx = np.flatnonzero(mask)
    n_genes = (expr > 0).sum(1)
    umis = expr.sum(1)

    print(f"[filter] {args.src_dir}: {N} 细胞 → 保留 {len(idx)} "
          f"({len(idx) / max(N, 1):.1%})  [min_genes>={args.min_genes}, "
          f"min_umis>={args.min_umis}]", flush=True)
    print(f"  n_genes 分布: p50={np.median(n_genes):.0f} p10={np.percentile(n_genes,10):.0f} "
          f"p5={np.percentile(n_genes,5):.0f}", flush=True)
    print(f"  umis 分布:   p50={np.median(umis):.0f} p10={np.percentile(umis,10):.0f} "
          f"p5={np.percentile(umis,5):.0f}", flush=True)
    if args.dry_run:
        # 目标：去掉底部 ~10% 低表达/空白细胞 → 用 p10 阈值估算联合保留比例
        g10 = float(np.percentile(n_genes, 10))
        u10 = float(np.percentile(umis, 10))
        keep10 = int(((n_genes >= g10) & (umis >= u10)).sum())
        print(f"  [p10 方案] min_genes={g10:.0f} & min_umis={u10:.0f} → "
              f"保留 {keep10}/{N} ({keep10 / max(N, 1):.1%})", flush=True)
        return

    os.makedirs(args.out_dir, exist_ok=True)

    # 表达 + 基因名
    out_expr = np.lib.format.open_memmap(
        os.path.join(args.out_dir, "gene_expression.npy"), mode="w+",
        dtype=expr.dtype, shape=(len(idx), expr.shape[1]))
    out_expr[:] = expr[idx]
    out_expr.flush()
    if gene_names is not None:
        with open(os.path.join(args.out_dir, "gene_names.txt"), "w") as f:
            f.write("\n".join(gene_names) + "\n")

    # metadata + patches
    keep_meta = meta.iloc[idx].copy()
    if args.copy_patches and not args.no_copy_patches:
        out_patches = os.path.join(args.out_dir, "patches")
        os.makedirs(out_patches, exist_ok=True)
        new_paths = []
        n_missing = 0
        for _, row in keep_meta.iterrows():
            src_p = row["patch_path"]
            if not os.path.isabs(src_p):
                # patch_path 可能相对项目根（preprocess_he 的 output_dir 是相对路径
                # 如 'data/rep1/patches/cell_X.png'），也可能相对 src_dir → 两个候选都试
                cand = os.path.join(args.src_dir, src_p)
                src_p = src_p if os.path.exists(src_p) else cand
            name = os.path.basename(src_p)
            dst = os.path.join(out_patches, name)
            if os.path.exists(src_p):
                if not os.path.exists(dst):
                    shutil.copy2(src_p, dst)
                new_paths.append(os.path.join("patches", name))
            else:  # patch 缺失（特征类方法不需要）→ 保留原路径引用
                n_missing += 1
                new_paths.append(str(row["patch_path"]))
        if n_missing:
            print(f"  [filter] 警告: {n_missing} 个 patch 缺失，已保留原路径引用"
                  f"（patch 类方法 ST-Net/BLEEP/Hist2ST 需要完整 patches/）", flush=True)
        keep_meta["patch_path"] = new_paths
    keep_meta.to_csv(os.path.join(args.out_dir, "metadata.csv"), index=False)

    # 特征文件按行切片（首维 == N 才切；--exclude_features 跳过）
    exclude = set()
    if args.exclude_features:
        exclude = set(f.strip() for f in args.exclude_features.split(",") if f.strip())
    if args.subset_features:
        for fp in sorted(glob.glob(os.path.join(args.src_dir, "X_*.npy"))):
            name = os.path.basename(fp)
            if name in exclude:
                print(f"  跳过 {name}（--exclude_features）", flush=True)
                continue
            try:
                arr = np.load(fp, mmap_mode="r")
            except Exception as e:
                print(f"  跳过 {name}: {e}", flush=True)
                continue
            if arr.shape[0] == N:
                _subset_feature(fp, os.path.join(args.out_dir, name), idx)

    # GHIST 数据过滤（可选）：按保留 cell_id 子集 CSV + 重算 avgexp
    if args.ghist_src and args.ghist_out:
        _filter_ghist(args.ghist_src, args.ghist_out, keep_meta["cell_id"].values)

    print(f"[filter] 完成: {args.out_dir}", flush=True)


N_REF = 512  # avgexp 参考细胞数（与 methods/ghist 一致）


def _filter_ghist(src: str, out: str, retained_cell_ids) -> None:
    """按保留 cell_id 过滤 GHIST 数据（表达/核/细胞型 CSV），重算 avgexp，复制 H&E 与核 mask。"""
    import shutil

    import pandas as pd

    keep = set(int(c) for c in retained_cell_ids)
    os.makedirs(out, exist_ok=True)

    expr = pd.read_csv(os.path.join(src, "cell_gene_matrix_filtered.csv"), index_col=0)
    expr = expr[expr.index.isin(keep)]
    expr.to_csv(os.path.join(out, "cell_gene_matrix_filtered.csv"))

    mn = pd.read_csv(os.path.join(src, "matched_nuclei_filtered.csv"))
    mn = mn[mn["id_histology"].isin(keep)]
    mn.to_csv(os.path.join(out, "matched_nuclei_filtered.csv"), index=False)

    ct = pd.read_csv(os.path.join(src, "celltype_filtered.csv"), index_col=0)
    ct = ct[ct.index.isin(keep)]
    ct.to_csv(os.path.join(out, "celltype_filtered.csv"))

    # avgexp：前 N_REF 保留细胞的均值向量（官方参考表达语义）
    n_ref = min(N_REF, len(expr))
    avgexp = expr.iloc[:n_ref].mean(axis=0).to_frame().T
    avgexp.to_csv(os.path.join(out, "avgexp.csv"), index=False)

    for name in ("he_image.tif", "he_image_nuclei_seg.tif"):
        s = os.path.join(src, name)
        if os.path.exists(s):
            shutil.copy2(s, os.path.join(out, name))
    print(f"[filter] GHIST: {src} → {out}（保留 {len(expr)} 细胞）", flush=True)


if __name__ == "__main__":
    main()
