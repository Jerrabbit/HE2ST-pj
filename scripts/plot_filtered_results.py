"""绘制过滤后数据集的 LG 两步调参曲线 + 各方法 PCC 柱状图。

数据来自远程 outputs/sweep_op1_f/op1_results.csv、outputs/sweep_op2_f/op2_results.csv
与 outputs/bench_*_f/test_results.json（过滤后）。配色用 dataviz 参考调色板
（蓝 #2a78d6 / 橙 #eb6834 / 墨 #0b0b0b / 次级 #898781）。

输出：
    local_global_op1_sweep_filtered.png
    local_global_op2_sweep_filtered.png
    benchmark_pcc_bar_filtered.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ---- CJK 字体（中文标签） ----
for fp in [r"C:/Windows/Fonts/msyh.ttc", r"C:/Windows/Fonts/msyhbd.ttc",
           r"C:/Windows/Fonts/simhei.ttf", r"C:/Windows/Fonts/simsun.ttc"]:
    try:
        fm.fontManager.addfont(fp)
        break
    except Exception:
        continue
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun",
                                   "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ---- dataviz 参考调色板 ----
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "axes.titlecolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "font.size": 11, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.family": "sans-serif",
})

# ================= op1 sweep：PCC vs l1（Global-only） =================
l1s = [448,420,392,364,336,308,280,252,224,196,168,140,112,84,56,28]
op1 = [0.3230,0.3246,0.3274,0.3292,0.3322,0.3339,0.3364,0.3391,
       0.3434,0.3438,0.3471,0.3492,0.3510,0.3474,0.3309,0.2856]
best_l1 = 112

fig, ax = plt.subplots(figsize=(8, 4.6))
ax.plot(l1s, op1, "o-", color=BLUE, markersize=6, linewidth=2,
        label="Global-only best val_PCC (30ep)")
ax.scatter([best_l1], [op1[l1s.index(best_l1)]], s=110, facecolors="none",
           edgecolors=ORANGE, linewidths=2.5, zorder=5)
ax.annotate(f"best l1=112\nPCC {op1[l1s.index(best_l1)]:.4f}",
            xy=(best_l1, op1[l1s.index(best_l1)]),
            xytext=(best_l1 + 18, op1[l1s.index(best_l1)] - 0.006),
            color=INK, fontsize=10)
ax.axhline(0.3364, color=MUTED, linestyle="--", linewidth=1)
ax.text(436, 0.3370, "UNI2+MLP 基线 0.3364", color=MUTED, fontsize=9, ha="right")
ax.set_xlabel("l1（Global 视野块边长 px）")
ax.set_ylabel("best val_PCC（30ep）")
ax.set_title("op1 sweep（过滤后）：Global-only，PCC vs l1")
ax.set_xlim(0, 470)
ax.legend(loc="lower left", frameon=False)
fig.tight_layout()
fig.savefig("local_global_op1_sweep_filtered.png", dpi=150)
plt.close(fig)

# ================= op2 sweep：PCC vs l2（固定 l1=112） =================
l2s = [28,42,56,70,84,98,112]
op2 = [0.3849,0.3862,0.3859,0.3838,0.3819,0.3779,0.3748]
best_l2 = 42

fig, ax = plt.subplots(figsize=(7.5, 4.6))
ax.plot(l2s, op2, "o-", color=BLUE, markersize=6, linewidth=2,
        label="Local+Global best val_PCC (30ep, l1=112)")
ax.scatter([best_l2], [op2[l2s.index(best_l2)]], s=110, facecolors="none",
           edgecolors=ORANGE, linewidths=2.5, zorder=5)
ax.annotate(f"best l2=42\nPCC {op2[l2s.index(best_l2)]:.4f}",
            xy=(best_l2, op2[l2s.index(best_l2)]),
            xytext=(best_l2 + 4, op2[l2s.index(best_l2)] - 0.0045),
            color=INK, fontsize=10)
ax.axhline(0.3510, color=MUTED, linestyle="--", linewidth=1)
ax.text(26, 0.3516, "Global-only(l1=112) 0.3510", color=MUTED, fontsize=9)
ax.set_xlabel("l2（Local 中心裁剪边长 px）")
ax.set_ylabel("best val_PCC（30ep）")
ax.set_title("op2 sweep（过滤后）：Local+Global，PCC vs l2（固定 l1=112）")
ax.set_xlim(22, 118)
ax.legend(loc="lower left", frameon=False)
fig.tight_layout()
fig.savefig("local_global_op2_sweep_filtered.png", dpi=150)
plt.close(fig)

# ================= 各方法 PCC 柱状图（排除 STFlow/Phoenix，含 LG） =================
methods = [
    ("UNI2+MLP Local+Global (l1=112, l2=42)", 0.3830, True),
    ("SQUALL (解码器头)", 0.3495, False),
    ("UNI2+MLP (基线)", 0.3364, False),
    ("Pixel2Gene (cell)", 0.3074, False),
    ("SpatialEx", 0.3065, False),
    ("Path2Space", 0.2946, False),
    ("GHIST", 0.2926, False),
    ("DeepPT (ResNet50)", 0.2791, False),
    ("ST-Net", 0.2520, False),
    ("BLEEP", 0.2322, False),
    ("Hist2ST", 0.2049, False),
]
methods = sorted(methods, key=lambda m: -m[1])  # 按 PCC 降序

fig, ax = plt.subplots(figsize=(9.2, 4.8))
names = [m[0] for m in methods]
vals = [m[1] for m in methods]
cols = [ORANGE if m[2] else BLUE for m in methods]
bars = ax.barh(range(len(methods)), vals, color=cols, height=0.62, zorder=3)
ax.set_yticks(range(len(methods)))
ax.set_yticklabels(names, fontsize=9.5)
ax.invert_yaxis()
ax.set_xlabel("PCC（过滤后，rep1_f 训练 → rep2_f 测试）")
ax.set_title("过滤后 benchmark：各方法 PCC（排除 STFlow/Phoenix）")
ax.set_xlim(0, 0.42)
for i, v in enumerate(vals):
    ax.text(v + 0.004, i, f"{v:.4f}", va="center", fontsize=9, color=INK)
# 图例
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=ORANGE, label="Local+Global（本仓库改进）"),
                   Patch(facecolor=BLUE, label="其他方法")],
          loc="lower right", frameon=False, fontsize=9)
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig("benchmark_pcc_bar_filtered.png", dpi=150)
plt.close(fig)

print("已生成 3 张图：")
for f in ["local_global_op1_sweep_filtered.png",
          "local_global_op2_sweep_filtered.png",
          "benchmark_pcc_bar_filtered.png"]:
    print("  -", f)
