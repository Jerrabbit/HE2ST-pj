# SQUALL — 可行性评估

> 文档顺序第 10 个方法。状态：**待定（需 HF 权重，获取后可行）**。

## 方法概述

SQUALL（Zhang et al., 2026）是**多模态病理学基础模型**：
- 多编码器（PLIP / UNI / Virchow）提取 H&E tile 特征；
- Transformer + **互补掩码** + **分辨率感知位置编码**；
- **模态混合专家（MoME）**融合表达 token 与组织 token；
- 在大规模 HE-ST 数据上预训练后，可冻结推理或微调预测基因表达。

与我们的核心假设（Foundation Model 特征即可强）高度相关，是重要对比方法。

## 官方代码位置

`D:\hest_data\codes\SQUALL-release\`
- `models/Squall.py`、`build.py`、`builder.py`、`ViT_utils.py`、`UNI.py`、`Virchow.py`、`PLIP.py`
- `SQUALL_Tutorial/Tutorial_inference.ipynb`：推理示例（HE tif + expr.pt）
- `gene_token_homologs.csv`：基因 token 同源映射

## 所需资源（阻塞点）

| 资源 | 状态 | 说明 |
|---|---|---|
| SQUALL 预训练权重 | ❌ | HuggingFace `zongxu/SQUALL`，需联网下载 |
| 基础编码器权重 | ⚠️ UNI 有 | UNI（我们有）、Virchow、PLIP 需 HF 下载 |
| 推理输入格式 | ❌ | 官方以整片 tif tile + 特定分辨率输入（`posX_15_posY_25_inhouse_HE.tif`） |
| 权重体积 | - | 多编码器+大模型，显存要求高 |

## 与本 benchmark 的兼容性

- **输入需适配**：官方以整片图像 tile 输入；我们可改用 per-cell patch（256×256）
  或复用 UNI2 特征作为主编码器输入，但 SQUALL 的位置编码/MoME 是按 tile 网格设计的，
  需按 cell patch 流重新对齐。
- **权重可得性**：HF 需联网。若远程服务器可访问 HF 且显存充足，可下载 SQUALL 权重
  做**冻结推理**（或轻量微调），走统一归一化空间评估。

## 建议

1. 本轮以**可行性文档**记录。
2. 若启动：确认服务器可访问 HF → 下载 SQUALL 权重 → 按官方 Tutorial 适配 cell patch
   输入 → 冻结推理 + 统一评估。若权重不可得，则记录为"权重阻塞"。
