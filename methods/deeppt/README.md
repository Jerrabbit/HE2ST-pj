# DeepPT — 可行性评估

> 文档顺序第 9 个方法。状态：**待定（粒度不匹配，适配后简单可行）**。

## 方法概述

DeepPT（Hoang et al., 2024，Nature Cancer）：
1. 用预训练 ResNet50 对整片切片的 tile 提取特征；
2. 训练**自编码器（AE）**把 tile 特征压缩到 512 维（缓解病理学-表达域差距）；
3. 训练 **MLP（512→512→基因数）** 从压缩特征**直接预测该切片 bulk 转录组**。
   流程上还有 k-fold 集成（ik×il 折叠取均值）。

## 官方代码位置

`D:\hest_data\codes\DeepPT\`（压缩包：`11slide_processing.zip`、`12AE.zip`、`13DeepPT_train.zip`）
- `13DeepPT_train/1main_train.py`：MLP 训练入口（n_inputs=512, n_hiddens=512, dropout=0.2）
- `13DeepPT_train/model_MLP.py`、`utils.py`
- `ResNet50_IMAGENET1K_V2.pt`：预训练 ResNet50 权重（本地已有）

## 粒度不匹配（核心问题）

官方 **DeepPT 预测的是整张切片的 bulk RNA 表达**（每个 tile 特征 → 聚合 → 每切片一个表达向量），
而我们的 benchmark 是 **per-cell（spot 级）表达**。直接按官方流程无法在 spot 级评估。

## 与本 benchmark 的兼容性

- **架构可复用**：ResNet50 特征提取 + AE 压缩（512 维）+ MLP 是简单可移植结构，
  完全可以在**单细胞粒度**训练（每 cell 的 UNI2 特征或 ResNet50 patch 特征 → AE 压缩 →
  MLP → 该 cell 的表达）。这样保持官方"AE 压缩 + MLP"架构，只是把聚合对象从切片换成细胞。
- **两种方案**：
  - **方案 A（推荐，spot 级适配）**：`input_type='feature'`，AE(1536→512) + MLP(512→G)，
    在统一归一化空间回归，走 harness `fit()`。公平、简单、可对比。
  - **方案 B（严格 bulk）**：按官方逐切片聚合，输出 per-slice 表达——与我们的 spot 级
    评估体系不兼容，除非把 benchmark 额外增加 slide 级评估。不建议本轮做。
- **AE 训练**：方案 A 中 AE 用重构损失在训练集 UNI2 特征上预训练，再与 MLP 联合或分阶段。

## 建议

1. 本轮以**可行性文档**记录。
2. 若启动：按方案 A 实现为普通特征回归方法（工作量小，结构简单），
   MLP 层数与 UNI2+MLP 基线保持统一（`common/models/mlp_head.py`）。
