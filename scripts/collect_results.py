"""汇总过滤后 benchmark 结果：主表 CSV + 过滤前后对比 CSV + Top-k 随 k 曲线。

数据来源：
    过滤后：outputs/bench_*_f/eval_metrics.csv（新评估模块，含 cell_PCC、top10..top100）
    过滤前：outputs/bench_*/test_results.json（旧评估模块，含 PCC/SPCC/AUROC/top10/50/100）

用法：
    python scripts/collect_results.py [--outputs outputs] [--out_dir outputs/filtered_summary]
"""
from __future__ import annotations

import argparse
import csv
import json
import os

import numpy as np

# 方法 → (过滤后目录, 过滤前目录)  （过滤前目录存 test_results.json 或 eval_metrics.csv）
METHODS = [
    ("Local+Global",   "bench_uni2_lg_final_f",   "bench_uni2_lg_final"),
    ("UNI2+MLP",       "bench_uni2_mlp_f",        "bench_uni2_mlp"),
    ("SQUALL(dec)",    "bench_squall_decoder_f",  "bench_squall_decoder"),
    ("GHIST",          "bench_ghist_f",           "bench_ghist"),
    ("SpatialEx",      "bench_spatialex_f",       "bench_spatialex"),
    ("Pixel2Gene",     "bench_pixel2gene_cell_f", "bench_pixel2gene_cell_log1p"),
    ("Path2Space",     "bench_path2space_f",      "bench_path2space_train"),
    ("DeepPT",         "bench_deeppt_resnet50_f", "bench_deeppt_resnet50"),
    ("ST-Net",         "bench_st_net_f",          "bench_st_net_frozen"),
    ("Hist2ST",        "bench_hist2st_f",         "bench_hist2st_official"),
    ("BLEEP",          "bench_bleep_f",           "bench_bleep_frozen"),
    ("STFlow",         "bench_stflow_f",          "bench_stflow_zinb"),
    ("Global-only",    "bench_uni2_global_f",     "bench_uni2_g112_ablation"),
    ("Local-only",     "bench_uni2_local_f",      "bench_uni2_l56_ablation"),
]

TOPK_KS = list(range(10, 101, 10))


def load_filtered(outputs: str, dirname: str) -> dict:
    """读过滤后 eval_metrics.csv → {metric: value}。"""
    p = os.path.join(outputs, dirname, "eval_metrics.csv")
    if not os.path.exists(p):
        return {}
    out = {}
    with open(p) as f:
        for row in csv.DictReader(f):
            try:
                out[row["metric"]] = float(row["value"])
            except (TypeError, ValueError):
                pass
    return out


def load_original(outputs: str, dirname: str) -> dict:
    """读过滤前 test_results.json → {metric: value}。"""
    for cand in (os.path.join(outputs, dirname, "test_results.json"),
                 os.path.join(outputs, dirname, "eval_metrics.csv")):
        if not os.path.exists(cand):
            continue
        if cand.endswith(".json"):
            with open(cand) as f:
                d = json.load(f)
            return {k: v for k, v in d.items() if isinstance(v, (int, float))}
        with open(cand) as f:
            return {r["metric"]: float(r["value"]) for r in csv.DictReader(f)
                    if r.get("value")}
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--out_dir", default="outputs/filtered_summary")
    ap.add_argument("--plot_topk", action="store_true", default=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rows_f, rows_before = [], []
    for name, d_f, d_old in METHODS:
        rf = load_filtered(args.outputs, d_f)
        ro = load_original(args.outputs, d_old)
        if rf:
            rows_f.append((name, rf))
        if ro:
            rows_before.append((name, ro))

    # ---- 过滤后主表 CSV（全指标） ----
    topk_keys = [f"top{k}" for k in TOPK_KS]
    cols = ["method", "PCC", "SPCC", "cell_PCC", "AUROC"] + topk_keys
    master = os.path.join(args.out_dir, "filtered_master.csv")
    with open(master, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for name, r in sorted(rows_f, key=lambda x: -x[1].get("PCC", -1)):
            w.writerow([name] + [f"{r.get(c, float('nan')):.4f}" for c in cols[1:]])
    print(f"过滤后主表已存: {master}")

    # ---- 过滤前后 PCC 对比 CSV ----
    cmp_csv = os.path.join(args.out_dir, "before_after_pcc.csv")
    rows_before_d = dict(rows_before)
    with open(cmp_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "before_PCC", "after_PCC", "delta"])
        for name, r in sorted(rows_f, key=lambda x: -x[1].get("PCC", -1)):
            before = rows_before_d.get(name, {}).get("PCC", float("nan"))
            after = r.get("PCC", float("nan"))
            w.writerow([name,
                        f"{before:.4f}" if before == before else "nan",
                        f"{after:.4f}" if after == after else "nan",
                        f"{after - before:+.4f}" if (before == before and after == after) else "nan"])
    print(f"过滤前后对比已存: {cmp_csv}")

    # ---- Top-k 曲线 ----
    if args.plot_topk:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 6))
        for name, r in rows_f:
            ys = [r.get(f"top{k}", np.nan) for k in TOPK_KS]
            if all(y == y for y in ys):
                ax.plot(TOPK_KS, ys, "o-", lw=1.6, ms=3, label=name)
        ax.set_xlabel("k (top-k genes)")
        ax.set_ylabel("top-k accuracy")
        ax.set_title("Top-k accuracy vs k (filtered rep1->rep2)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="center right", bbox_to_anchor=(1.30, 0.5))
        fig.tight_layout()
        png = os.path.join(args.out_dir, "topk_vs_k_filtered.png")
        fig.savefig(png, dpi=150, bbox_inches="tight")
        print(f"Top-k 曲线已存: {png}")

    # 控制台摘要
    print("\n===== 过滤后 PCC 排序 =====")
    for name, r in sorted(rows_f, key=lambda x: -x[1].get("PCC", -1)):
        print(f"  {name:14s} PCC={r.get('PCC', float('nan')):.4f}  "
              f"cell_PCC={r.get('cell_PCC', float('nan')):.4f}  "
              f"AUROC={r.get('AUROC', float('nan')):.4f}")


if __name__ == "__main__":
    main()
