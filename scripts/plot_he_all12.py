"""为全部 12 种方法绘制 PCC / SPCC / AUROC 柱状图（普通柱状图）。

新跑方法（7 个，本轮 rep1→rep2 未过滤）读本地 outputs/bench_*_unf/test_results.json；
未跑方法（5 个）用 README 中"最初未过滤"基准的值（SQUALL/GHIST/Hist2ST/BLEEP/Phoenix）。
实心 = 本轮新跑；浅色+斜线 = 暂用旧未过滤值。无误差棒/散点。
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from matplotlib.patches import Patch

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

# ---- 12 种方法：名称 → (输出目录 or None=旧值, 旧值(PCC,SPCC,AUROC)) ----
# 新跑 7 方法读本地 test_results.json；未跑 5 方法用 README 旧未过滤值
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
    """返回 (PCC, SPCC, AUROC, is_new)。"""
    d = METHOD_DIRS.get(name)
    if d:
        p = os.path.join("outputs", d, "test_results.json")
        if os.path.exists(p):
            r = json.load(open(p))
            return (r["PCC"], r["SPCC"], r["AUROC"], True)
    ph = PLACEHOLDER.get(name)
    if ph:
        return (ph[0], ph[1], ph[2], False)
    return (float("nan"), float("nan"), float("nan"), False)


METHODS = ["UNI2+MLP", "SQUALL", "GHIST", "SpatialEx", "Pixel2Gene", "Path2Space",
           "DeepPT", "STFlow", "ST-Net", "BLEEP", "Hist2ST", "Phoenix"]
DATA = {n: get_metrics(n) for n in METHODS}

# 每个指标的柱色（实心=新跑；浅色+斜线=旧值）
CHART_CFG = [
    ("PCC", "#2a78d6", "#a9c7ef", "benchmark_pcc_bar_all12.png"),
    ("SPCC", "#1baf7a", "#b4e6da", "benchmark_spcc_bar_all12.png"),
    ("AUROC", "#eda100", "#f6d689", "benchmark_auroc_bar_all12.png"),
]

for metric, solid, light, fname in CHART_CFG:
    idx = {"PCC": 0, "SPCC": 1, "AUROC": 2}[metric]
    items = [(n, DATA[n][idx], DATA[n][3]) for n in METHODS]
    items.sort(key=lambda t: -t[1])                       # 按指标降序
    names = [t[0] for t in items]
    vals = [t[1] for t in items]
    newf = [t[2] for t in items]

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    x = np.arange(len(names))
    for xi, (v, is_new) in enumerate(zip(vals, newf)):
        color = solid if is_new else light
        hatch = None if is_new else "//"
        ax.bar(xi, v, color=color, hatch=hatch, width=0.62, zorder=3,
               edgecolor=INK if not is_new else "none", linewidth=0.5)
        ax.text(xi, v + 0.004, f"{v:.4f}", ha="center", va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9.5, rotation=40, ha="right")
    ax.set_ylabel(metric)
    ax.set_ylim(0, max(vals) * 1.16)
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Patch(facecolor=solid, label="本轮新跑"),
                       Patch(facecolor=light, hatch="//", edgecolor=INK, label="暂用旧未过滤值")],
              loc="lower right", frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"已生成 {fname}")
    for n, v, is_new in items:
        print(f"  {n:12s} {metric}={v:.4f}{'  (新)' if is_new else '  (旧占位)'}")
