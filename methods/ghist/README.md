# GHIST — 可行性评估

> 文档顺序第 6 个方法。状态：**待定（需整片图+分割管线），暂未实现**。

## 方法概述

GHIST（Outeiral et al., 2024，Cell）在**单细胞分辨率**从 H&E 预测空间基因表达：
- UNet（Backbone）对 H&E 图做**细胞核分割 + 细胞类型分类**；
- 结合细胞核形态（面积、边界）、邻居组件（cell-graph）、细胞类型与平均表达，多组件融合；
- 预测每个细胞的基因表达（回归，表达式在 0–5 尺度）。

核心亮点正是"Local（细胞形态/核） + Global（细胞邻域 graph）"，与我们的
Local-Global 假设高度相关，是重要对比方法。

## 官方代码位置

`D:\hest_data\codes\GHIST\`
- `model/`：`model.py`（Framework）、`backbone.py`（UNet）、`modules.py`、`layers.py`
- `data_processing/`：1_get_xenium_nuclei_seg_image.py、3_segment_nuclei_he_image.py（cellpose）
- `train.py` / `inference.py`、`configs/config_demo.json`

## 所需资源（阻塞点）

| 资源 | 状态 | 说明 |
|---|---|---|
| 整张切片 H&E 图像（tif） | ⚠️ **本地有、未上传远程** | 本地 `D:\hest_data\datasets\breast\*.ome.tif`（S1–S4，含 Top/Mid/Bot）；远程 rep1/rep2 只有 per-cell 256×256 patch，**无整片图**，需 scp 上传 |
| 细胞核分割图 `he_image_nuclei_seg.tif` | ❌ 需生成 | 官方用 cellpose 或 Xenium 自带分割，把核轮廓转成 mask |
| 细胞类型标注 | ❌ 需准备 | Xenium 有细胞分型（但来自转录组，需与 H&E 对齐） |
| `torchstain`（HE 染色归一化）等依赖 | 未装 | requirements.txt 额外包 |

> 说明：per-cell patch 本身就是从整片 H&E 裁剪的，用户直觉"图不是已经有了吗"是对的——
> 但 GHIST 的输入是**全分辨率整片 tif + 核分割 mask**，不是局部 patch，因此整片图
> 是独立于 patches 的必需资源（本地已有，缺的是上传到远程）。

## 与本 benchmark 的兼容性

- **输入不匹配**：GHIST 是**基于整片图像 + 分割图**的细胞级方法，不是 per-cell patch 输入。
  我们的 patch（256×256，以细胞为中心）无法直接喂入 UNet 整片分割流程。
- **分割需求**：需先对每张切片跑细胞核分割（cellpose 或 Xenium 提供的核 mask 转 tif），
  且 UNet 的 celltype 分支需要标签（可用 Xenium 细胞分型做弱监督/伪标签）。
- **评估可对齐**：输出是 per-cell 表达（0–5 尺度），可映射回我们的 raw counts / 归一化
  空间评估（PCC/AUROC 均适用）。**这是少数能在单细胞粒度公平对比的复杂空间模型之一，
  值得实现。**

## 建议

1. 本轮以**可行性文档**记录。
2. 实现路径（若启动）：
   - 从本地 `D:\hest_data\datasets\breast\*.ome.tif` 上传每张切片的整片 H&E 到远程；
   - 用 Xenium 细胞多边形生成核分割 mask（避免额外跑 cellpose）；
   - UNet 分割+分型、cell-graph、邻居组件全部照官方架构实现；
   - 输出转统一归一化空间评估。工作量大（数据管线为主，模型代码可直接复用官方）。
