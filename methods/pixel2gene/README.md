# Pixel2Gene — 实现状态

> 文档顺序第 5 个方法。状态：**已实现（HIPT-4K 移植 + 特征提取 + MLP 回归），待权重上传后跑基准**。

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
| 整张切片图像（Xenium 全分辨率 tif） | ✅ **当前基准 Rep1/Rep2 整片图已在远程** | 远程 `/cpfs01/.../he_images/Xenium_FFPE_Human_Breast_Cancer_Rep{1,2}_he_image.ome.tif`；多切片基准 S1–S4 在本地 `D:\hest_data\datasets\breast\*.ome.tif`，未上传 |

> 说明：Pixel2Gene 输入是**整片 4K tile 层级**（HIPT-4K），不是局部 cell patch——
> cell patch 只是 H&E 的局部裁剪，无法直接替代整片图。当前基准的 Rep1/Rep2 整片图
> **已在远程集群**（`/cpfs01/.../he_images/`），无需上传。

## 与本 benchmark 的兼容性

- **输入不匹配**：Pixel2Gene 需要整张 H&E 全分辨率图像（4K tile 输入），而我们的
  基准数据是 per-cell 256×256 patch。cell patch 无法直接拼出整片图像（重叠/对齐复杂）。
- **输出粒度**：官方在 superpixel/pseudo-spot 粒度预测，非单细胞。
- **若坚持实现**：需 (a) 整片 H&E 用远程 `/cpfs01/.../he_images/` 下的 Rep1/Rep2 tif
  （已在集群，无需上传）；(b) 下载 HIPT-4K 权重（真正的硬阻塞）；(c) 按官方 binning
  流程把单细胞聚合到 pseudo-spot 再评估。工作量中等，当前缺的是权重。

## 实现进度（2026-08-14）

- **HIPT-4K 模型已移植**：`methods/pixel2gene/hipt/`（官方 hipt_4k.py / hipt_model_utils.py /
  vision_transformer.py / vision_transformer4k.py 原样复制，einops 已替换为等价原生 torch，
  远程无 einops；reshape 等价性已本地验证）。
- **权重已下载本地**：`D:\hest_data\codes\Pixel2Gene\checkpoints\vit_256_small_dino.pth`(704MB) +
  `vit_4096_xs_dino.pth`(396MB)，上传远程 `~/HE2ST-pj/weights/pixel2gene/` 中。
- **特征提取**：`scripts/extract_hipt.py` —— 伪 Visium 分箱（100µm 六角网格，官方
  bin_pseudo_visium 语义）+ 每 spot 取 2048² 上下文（8×8 个 256 patch，真实层级上下文）
  → HIPT-4K concat[mean256(384), cls4k(192)]=576 维 → 每细胞继承 spot 特征 → X_hipt.npy。
  rep1 实测：3519 有效 spot，覆盖 96.6% 细胞，无边缘越界。
- **回归模型**：`methods/pixel2gene/model.py` Pixel2GeneModel（input_type='feature'，
  统一 MLPHead(576→G)，走标准 harness fit）。
- **验证结论**：HIPT-4K 前向本地通过（2048² → cls4k(1,192)）；bin 分箱真实坐标验证通过。

## 下一步（权重上传完成后）

1. `python scripts/extract_hipt.py --rep 1/2 --model256 ... --model4k ...` 生成 X_hipt.npy；
2. `python scripts/train.py --method pixel2gene ...`（标准 harness fit）；
3. `python scripts/test.py --method pixel2gene ...`。

## 建议

1. 本轮以**实现中**记录（权重上传中）。
2. 若后续获取更完整的 4K 上下文（4096² tile）可扩展，但当前 2048² 上下文已保留层级结构。
