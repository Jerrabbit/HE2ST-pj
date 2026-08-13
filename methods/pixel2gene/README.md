# Pixel2Gene — 可行性评估

> 文档顺序第 5 个方法。状态：**待定（需权重），暂未实现**。

## 方法概述

Pixel2Gene（Zhang et al., 2024，Nature）用 HIPT-4K（Hierarchical Image Pyramid Transformer）
在整张组织学切片上提取 4096-token 层级特征，通过自监督蒸馏与组织学-表达对齐，
从 H&E 直接重建/预测空间基因表达。特征侧重**大尺度组织上下文**（4K tile 层级）。

## 官方代码位置

`D:\hest_data\codes\Pixel2Gene\`
- `scripts/hipt_4k.py`、`hipt_model_utils.py`、`extract_features.py`：HIPT-4K 特征提取
- `scripts/demo_train.sh`、`scripts/demo_predict.sh`：训练/预测入口
- `format_xenium/`：Xenium 数据格式化（superpixel binning、伪 Visium 分箱）

## 所需资源（阻塞点）

| 资源 | 状态 | 说明 |
|---|---|---|
| HIPT-4K 预训练权重（ViT-Small 4096 + ViT-Base 256 层级） | ❌ 本地/远程均未找到 | 原论文权重托管于 GitHub/官方路径，需下载（真正的硬阻塞） |
| HIPT 依赖包（`hipt_4k.py` 依赖 `timm`、`apex`?） | 部分 | `hipt_4k.py` 在本仓库，但依赖外部 `hipt_model_utils` 权重 |
| 整张切片图像（Xenium 全分辨率 tif） | ⚠️ **本地有、未上传远程** | 本地 `D:\hest_data\datasets\breast\*.ome.tif`（S1–S4，含 Top/Mid/Bot）；远程只有 per-cell patch，需 scp 上传 |

> 说明：Pixel2Gene 输入是**整片 4K tile 层级**（HIPT-4K），不是局部 cell patch——
> cell patch 只是 H&E 的局部裁剪，无法直接替代整片图。整片图本地已有，缺的是上传。

## 与本 benchmark 的兼容性

- **输入不匹配**：Pixel2Gene 需要整张 H&E 全分辨率图像（4K tile 输入），而我们的
  基准数据是 per-cell 256×256 patch。cell patch 无法直接拼出整片图像（重叠/对齐复杂）。
- **输出粒度**：官方在 superpixel/pseudo-spot 粒度预测，非单细胞。
- **若坚持实现**：需 (a) 从本地 `D:\hest_data\datasets\breast\*.ome.tif` 上传整片 H&E
  到远程；(b) 下载 HIPT-4K 权重（真正的硬阻塞）；(c) 按官方 binning 流程把单细胞
  聚合到 pseudo-spot 再评估。工作量中等，当前缺的是权重（整片图本地已有）。

## 建议

1. 本轮以**可行性文档**记录（本文件），不写实现。
2. 若后续上传整片图像并获取 HIPT 权重，可再按 `format_xenium/` 流程实现，
   评估粒度需与其它方法对齐（pseudo-spot 聚合后比较）。
