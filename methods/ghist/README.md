# GHIST — 实现状态

> 文档顺序第 6 个方法。状态：**已实现（官方 Framework 完整移植 + benchmark 统一接口），
> 待数据管线（核分割 mask）就绪后可在我们 rep1/rep2 上运行**。

## 已实现（2026-08-15）

- `methods/ghist/framework.py / backbone.py / modules.py / layers.py / intialisation.py`：
  官方 Framework（UNet 分割 + 逐核聚合 + avgexp/offset/细胞型/邻域组成多组件头）原样移植。
- `methods/ghist/model.py`：GHISTModel benchmark 包装（input_type='patch' 整片图方法）。
- `methods/ghist/__init__.py`：build_model + train_function（官方 9 损失 + val_PCC 早停）
  + evaluate_slide（整片推理 → 统一指标），读 ghist_data 格式。
- 合成数据前向验证通过（out_expr / celltype / comp 输出正确）。

## 数据管线待补（在我们 rep1/rep2 上运行的前提）

我们 rep1/rep2 尚无 `he_image_nuclei_seg.tif`（核分割 mask）。需先用官方
`data_processing/` 生成（cellpose 或 Xenium 多边形 → mask + matched_nuclei）。
远程已有 GHIST 在其它数据集上的官方运行（`/cpfs01/.../GHIST/`，PCC 0.210,
AUROC 0.596），但那是不同数据集，不能直接对齐我们的 rep2。

## 官方参考指标（其它数据集上）

PCC 0.210, SPCC 0.201, Top-10 0.370, Top-50 0.473, Top-100 0.521, AUROC 0.596

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
| 整张切片 H&E 图像（tif） | ✅ **当前基准 Rep1/Rep2 整片图已在远程** | 远程 `/cpfs01/projects-HDD/cfff-d7ff0c9cdf2f_HDD/hjr_24300980068/HE2ST/datasets/he_images/Xenium_FFPE_Human_Breast_Cancer_Rep{1,2}_he_image.ome.tif`（MPP 0.363788 µm/px 已实测） |
| 细胞核分割图 `he_image_nuclei_seg.tif` | ❌ 需生成 | 官方用 cellpose 或 Xenium 自带分割，把核轮廓转成 mask |
| 细胞类型标注 | ❌ 需准备 | Xenium 有细胞分型（但来自转录组，需与 H&E 对齐） |
| `torchstain`（HE 染色归一化）等依赖 | 未装 | requirements.txt 额外包 |

> 说明：per-cell patch 本身就是从整片 H&E 裁剪的，用户直觉"图不是已经有了吗"是对的——
> 但 GHIST 的输入是**全分辨率整片 tif + 核分割 mask**，不是局部 patch，因此整片图
> 是独立于 patches 的必需资源。当前相邻切片基准的 **Rep1/Rep2 整片图已在远程集群**
> （`/cpfs01/.../he_images/`），无需上传。另有**多切片基准**数据
> `D:\hest_data\datasets\breast\Human_Breast_Biomarkers_S*_*.ome.tif`（S1–S4）在本地，
> 未上传远程，供将来同癌种多切片 benchmark 使用。

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
   - 整片 H&E：当前基准直接用远程 `/cpfs01/.../he_images/` 下的 Rep1/Rep2 tif；
     多切片基准的 S1–S4 需从本地 `D:\hest_data\datasets\breast\*.ome.tif` 上传；
   - 用 Xenium 细胞多边形生成核分割 mask（避免额外跑 cellpose）；
   - UNet 分割+分型、cell-graph、邻居组件全部照官方架构实现；
   - 输出转统一归一化空间评估。工作量大（数据管线为主，模型代码可直接复用官方）。
