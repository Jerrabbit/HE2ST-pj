"""为全部 12 种方法绘制 PCC / SPCC / AUROC 柱状图（普通柱状图）。

新跑方法（7 个）读本地 outputs/bench_*_unf/test_results.json；
未跑方法（5 个）用 README 中"最初未过滤"基准的值。全部同风格实心柱，
色调沿用之前误差棒 PCC 图的清新浅色（黄、绿、蓝为主）。
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ---- CJK 字体 ----
for fp in [r"C:/Windows/Fonts/msyh.ttc", r"C:/Windows/Fonts/msyhbd.ttc",
           r"C:/Windows/Fonts/simhei.ttf", r"C:/Windows/Fonts/simsun.ttc"]:
    try:
        fm.fontManager.addfont(fp)
        break
    except Exception:
        continue
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

INK = "#0b0b0b"; MUTED = "#898781"; GRID = "#e1e0d9"; SURFACE = "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "axes.titlecolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "font.size": 11, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.family": "sans-serif",
})

# ---- 12 种方法：清新浅色（黄/绿/蓝为主，与误差棒 PCC 图一致） ----
METHOD_COLORS = {
    "UNI2+MLP":   "#a9c7ef",   # 浅蓝
    "Pixel2Gene": "#fbe39a",   # 浅黄
    "SpatialEx":  "#b4e6da",   # 浅青绿
    "Path2Space": "#f6d689",   # 浅暗黄
    "DeepPT":     "#b9e4bc",   # 浅绿
    "ST-Net":     "#8fb7e6",   # 中浅蓝
    "STFlow":     "#f6c4c9",   # 浅粉
    "Phoenix":    "#d8c9f0",   # 浅紫
    "SQUALL":     "#a8d8d9",   # 浅青
    "GHIST":      "#c9e8c0",   # 浅薄荷
    "Hist2ST":    "#f3d9a8",   # 浅杏
    "BLEEP":      "#cfe0f2",   # 浅长春花蓝
}

# ---- 旧未过滤占位值（未跑方法） ----
PLACEHOLDER = {
    "BLEEP":   (0.2131, 0.2056, 0.666),
    "Phoenix": (0.2055, 0.1585, 0.610),
    "GHIST":   (0.3164, 0.2952, 0.700),
    "Hist2ST": (0.2139, 0.2046, 0.670),
    "SQUALL":  (0.3281, 0.2873, 0.742),
}
METHOD_DIRS = {
    "UNI2+MLP":   "bench_uni2_mlp_unf",
    "Pixel2Gene": "bench_pixel2gene_cell_unf",
    "SpatialEx":  "bench_spatialex_unf",
    "Path2Space": "bench_path2space_unf",
    "STFlow":     "bench_stflow_unf",
    "DeepPT":     "bench_deeppt_resnet50_unf",
    "ST-Net":     "bench_st_net_unf",
}


def get_metrics(name):
    """返回 (PCC, SPCC, AUROC)。"""
    d = METHOD_DIRS.get(name)
    if d:
        p = os.path.join("outputs", d, "test_results.json")
        if os.path.exists(p):
            r = json.load(open(p))
            return (r["PCC"], r["SPCC"], r["AUROC"])
    ph = PLACEHOLDER.get(name)
    return ph if ph else (float("nan"), float("nan"), float("nan"))


METHODS = ["UNI2+MLP", "SQUALL", "GHIST", "SpatialEx", "Pixel2Gene", "Path2Space",
           "DeepPT", "STFlow", "ST-Net", "BLEEP", "Hist2ST", "Phoenix"]
DATA = {n: get_metrics(n) for n in METHODS}

CHART_CFG = [
    ("PCC", "benchmark_pcc_bar_all12.png"),
    ("SPCC", "benchmark_spcc_bar_all12.png"),
    ("AUROC", "benchmark_auroc_bar_all12.png"),
]

for metric, fname in CHART_CFG:
    idx = {"PCC": 0, "SPCC": 1, "AUROC": 2}[metric]
    items = sorted([(n, DATA[n][idx]) for n in METHODS], key=lambda t: -t[1])
    names = [t[0] for t in items]
    vals = [t[1] for t in items]

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(names))
    for xi, (n, v) in enumerate(zip(names, vals)):
        ax.bar(xi, v, color=METHOD_COLORS[n], width=0.62, zorder=3)
        ax.text(xi, v + 0.004, f"{v:.4f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9.5, rotation=40, ha="right")
    ax.set_ylabel(metric)
    ax.set_ylim(0, max(vals) * 1.16)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"已生成 {fname}")
    for n, v in items:
        print(f"  {n:12s} {metric}={v:.4f}")
