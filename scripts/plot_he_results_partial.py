"""绘制已完成的 6 个方法（rep1→rep2 未过滤新跑）的：
  1) PCC 带误差棒柱状图（柱=均值，误差棒=±1 std 逐基因 PCC）
  2) Top-k 准确率随 k 变化的连续曲线（k=10..313）

数据来源：outputs/bench_*_unf/{test_results.json, eval_metrics.csv_gene_pcc.csv, topk_curve.csv}
配色：dataviz 参考调色板。输出到项目根目录。
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# ---- CJK 字体（中文标签） ----
for fp in [r"C:/Windows/Fonts/msyh.ttc", r"C:/Windows/Fonts/msyhbd.ttc",
           r"C:/Windows/Fonts/simhei.ttf", r"C:/Windows/Fonts/simsun.ttc"]:
    try:
        fm.fontManager.addfont(fp)
        break
    except Exception:
        continue
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---- dataviz 参考调色板 ----
BLUE = "#2a78d6"; ORANGE = "#eb6834"; AQUA = "#1baf7a"; YELLOW = "#eda100"
MAGENTA = "#e87ba4"; GREEN = "#008300"; INK = "#0b0b0b"; MUTED = "#898781"
GRID = "#e1e0d9"; SURFACE = "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "axes.titlecolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "font.size": 11, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.family": "sans-serif",
})

# ---- 方法定义（Top-k 曲线用 dataviz 槽位色；柱状图单独用黄绿蓝配色） ----
METHODS = [
    ("UNI2+MLP", "bench_uni2_mlp_unf", BLUE),
    ("Pixel2Gene", "bench_pixel2gene_cell_unf", ORANGE),
    ("SpatialEx", "bench_spatialex_unf", AQUA),
    ("Path2Space", "bench_path2space_unf", YELLOW),
    ("DeepPT", "bench_deeppt_resnet50_unf", MAGENTA),
    ("ST-Net", "bench_st_net_unf", GREEN),
]
# PCC 柱状图：黄、绿、蓝为主，很浅的柱色 + 稍深的散点色
BAR_COLORS = {
    "UNI2+MLP": "#a9c7ef",     # 极浅蓝
    "Pixel2Gene": "#fbe39a",   # 极浅黄
    "SpatialEx": "#b4e6da",    # 极浅青绿
    "Path2Space": "#f6d689",   # 极浅暗黄
    "DeepPT": "#b9e4bc",       # 极浅绿
    "ST-Net": "#8fb7e6",       # 浅蓝
}
SCATTER_COLORS = {
    "UNI2+MLP": "#2a78d6",     # 深蓝
    "Pixel2Gene": "#d19a00",   # 深黄
    "SpatialEx": "#149a74",    # 深青绿
    "Path2Space": "#b98200",   # 深暗黄
    "DeepPT": "#2f8f5a",       # 深绿
    "ST-Net": "#2a5f9e",       # 深蓝
}
OUT_ROOT = "outputs"


def load_method(dirname):
    base = os.path.join(OUT_ROOT, dirname)
    # 逐基因 PCC：mean / std（误差棒）
    pccs = []
    with open(os.path.join(base, "eval_metrics.csv_gene_pcc.csv")) as f:
        r = csv.reader(f)
        next(r)  # header
        for row in r:
            if row and row[1] not in ("", "nan"):
                pccs.append(float(row[1]))
    pccs = np.array(pccs)
    # Top-k 曲线
    ks, acc = [], []
    with open(os.path.join(base, "topk_curve.csv")) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if row:
                ks.append(int(row[0])); acc.append(float(row[1]))
    ks = np.array(ks); acc = np.array(acc)
    # test_results.json 汇总（PCC 标量，跨检查用）
    with open(os.path.join(base, "test_results.json")) as f:
        res = json.load(f)
    return {"name": None, "pccs": pccs, "ks": ks, "acc": acc, "res": res}


data = {name: load_method(d) for name, d, _ in METHODS}
for name in data:
    data[name]["name"] = name

# ================= 1. PCC 带误差棒柱状图（±1std + 逐基因 PCC 散点） =================
names = [name for name, _, _ in METHODS]
means = [data[n]["pccs"].mean() for n in names]
stds = [data[n]["pccs"].std() for n in names]
order = np.argsort(-np.array(means))  # 按 PCC 降序
names_s = [names[i] for i in order]
means_s = [means[i] for i in order]
stds_s = [stds[i] for i in order]
cols = [BAR_COLORS[n] for n in names_s]

fig, ax = plt.subplots(figsize=(8.4, 5.0))
x = np.arange(len(names_s))
# 误差棒只画上侧（mean + 1std）；柱半透明，让下方散点透出
yerr_up = np.array([np.zeros(len(stds_s)), stds_s])
ax.bar(x, means_s, yerr=yerr_up, color=cols, width=0.6, alpha=0.55,
       error_kw=dict(ecolor=INK, lw=1.2, capsize=4), zorder=3)
# 逐基因 PCC 散点（横向轻微抖动，展示分布；颜色比柱深以保持可见）
rng = np.random.default_rng(0)
for xi, n in zip(x, names_s):
    pccs = data[n]["pccs"]
    jx = rng.uniform(-0.18, 0.18, size=len(pccs))
    ax.scatter(xi + jx, pccs, s=6, color=SCATTER_COLORS[n], alpha=0.45,
               linewidths=0, zorder=2)
# 数值标签放在柱顶内侧（避开误差棒线，zorder 最上层）
for xi, m in zip(x, means_s):
    ax.text(xi, m - 0.012, f"{m:.4f}", ha="center", va="top", fontsize=9,
            color=INK, zorder=5)
ax.set_xticks(x)
ax.set_xticklabels(names_s, fontsize=10)
ax.set_ylabel("PCC")
ymax = max(data[n]["pccs"].max() for n in names)
ax.set_ylim(0, ymax + 0.08)
ax.grid(axis="x", visible=False)
fig.tight_layout()
fig.savefig("benchmark_pcc_bar_unf_partial.png", dpi=150)
plt.close(fig)
print("已生成 benchmark_pcc_bar_unf_partial.png")
for n, m, s in zip(names_s, means_s, stds_s):
    print(f"  {n:12s} {m:.4f} ± {s:.3f}（散点 min {data[n]['pccs'].min():.3f} ~ max {data[n]['pccs'].max():.3f}）")

# ================= 2. Top-k 准确率连续曲线 =================
fig, ax = plt.subplots(figsize=(9.5, 5.2))
for name, _, color in METHODS:
    d = data[name]
    m = d["ks"] >= 10                      # 曲线从 k=10 开始
    ax.plot(d["ks"][m], d["acc"][m], color=color, lw=2, label=name)
ax.set_xlabel("k（Top-k 前 k 个基因）")
ax.set_ylabel("Top-k 准确率")
ax.set_title("Top-k 准确率随 k 变化的连续曲线（k=10..313）")
ax.set_xlim(10, 313)                        # 横轴到 313
ax.set_ylim(0.3, 1.0)                       # 纵轴到 1（k=313 时 acc=1.0）
ax.legend(loc="lower right", frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig("topk_accuracy_unf_partial.png", dpi=150)
plt.close(fig)
print("已生成 topk_accuracy_unf_partial.png")
