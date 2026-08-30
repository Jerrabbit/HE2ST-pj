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

# ---- 方法定义（固定颜色，跨图一致） ----
METHODS = [
    ("UNI2+MLP", "bench_uni2_mlp_unf", BLUE),
    ("Pixel2Gene (cell)", "bench_pixel2gene_cell_unf", ORANGE),
    ("SpatialEx", "bench_spatialex_unf", AQUA),
    ("Path2Space", "bench_path2space_unf", YELLOW),
    ("DeepPT (R50)", "bench_deeppt_resnet50_unf", MAGENTA),
    ("ST-Net", "bench_st_net_unf", GREEN),
]
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

# ================= 1. PCC 带误差棒柱状图 =================
names = [name for name, _, _ in METHODS]
means = [data[n]["pccs"].mean() for n in names]
stds = [data[n]["pccs"].std() for n in names]
order = np.argsort(-np.array(means))  # 按 PCC 降序
names_s = [names[i] for i in order]
means_s = [means[i] for i in order]
stds_s = [stds[i] for i in order]
name_to_color = {name: c for name, _, c in METHODS}
cols = [name_to_color[n] for n in names_s]  # 固定颜色随实体

fig, ax = plt.subplots(figsize=(8.6, 4.8))
bars = ax.barh(range(len(names_s)), means_s, xerr=stds_s, color=cols, height=0.6,
               error_kw=dict(ecolor=INK, lw=1.2, capsize=4), zorder=3)
for i, (m, s) in enumerate(zip(means_s, stds_s)):
    ax.text(m + s + 0.004, i, f"{m:.4f}±{s:.3f}", va="center", fontsize=9, color=INK)
ax.set_yticks(range(len(names_s)))
ax.set_yticklabels(names_s, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("PCC（rep1→rep2 未过滤，mean±1std 逐基因 PCC）")
ax.set_title("已完成 6 方法 PCC（带误差棒）")
ax.set_xlim(0, max(m + s for m, s in zip(means_s, stds_s)) + 0.06)
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig("benchmark_pcc_bar_unf_partial.png", dpi=150)
plt.close(fig)
print("已生成 benchmark_pcc_bar_unf_partial.png")
for n, m, s in zip(names_s, means_s, stds_s):
    print(f"  {n:18s} {m:.4f} ± {s:.4f}")

# ================= 2. Top-k 准确率连续曲线 =================
fig, ax = plt.subplots(figsize=(9.5, 5.2))
for name, _, color in METHODS:
    d = data[name]
    ax.plot(d["ks"], d["acc"], color=color, lw=2, label=name)
ax.set_xlabel("k（Top-k 前 k 个基因）")
ax.set_ylabel("Top-k 准确率")
ax.set_title("Top-k 准确率随 k 变化的连续曲线（k=10..313）")
ax.set_xlim(0, 313)
ax.set_ylim(0.35, 0.75)
ax.legend(loc="lower right", frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig("topk_accuracy_unf_partial.png", dpi=150)
plt.close(fig)
print("已生成 topk_accuracy_unf_partial.png")
