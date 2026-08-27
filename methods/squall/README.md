# SQUALL — 已接入（忠实度专项审查确认：高）

> 状态：✅ **已实现并接入 benchmark**（2026-08-27 忠实度审查后更新）。

## 方法概述

SQUALL（Zhang et al., 2026）是**多模态病理学基础模型**：
- 多编码器（PLIP / UNI / Virchow）提取 H&E tile 特征；
- Transformer + **互补掩码** + **分辨率感知位置编码**；
- **模态混合专家（MoME）**融合表达 token 与组织 token；
- 在大规模 HE-ST 数据上预训练。

## 官方权重

`SQUALL_full.pth`（HF `zongxu/SQUALL` 原版权重，用户确认）。本地 `methods/squall/`
以**严格模式**加载：实测 0 missing / 0 unexpected。`forward_rgb→(B,196,1024)`、
`forward_rgb_to_expr→(B,56,56,15757)` 与官方教程一致。

## 忠实度专项审查（2026-08-27，详见根 README"方法实现忠实度专项审查"节）

- `transformer.py` 与官方**逐字节一致**；`Squall.py` 仅 import stub + 补 `forward_rgb_to_expr`
  （官方发布版缺该方法，本地补的是官方教程代码库版本，不改变架构/权重）。
- 配置直接读官方 `SQUALL_Tutorial/config.yaml`。
- **0-1 输入**：本地 `extract_squall.py` 输入 `/255.0`（0-1），实测 0-255 会使解码输出负相关。
- 预训练 `SquallDecoder.forward` 引用未定义变量 `z_rgb`（官方原版同样如此，仅预训练路径
  可达，本项目不用）。

## 本仓库适配（项目原则：冻结编码器 + 训练头）

- **特征提取**：per-cell 256×256 patch → resize 224 → `forward_rgb` → mean-pool 1024 特征
  （`X_squall.npy`）或保留 196×1024 token（`X_squall_tokens.npy`，`--save_tokens`）。
- **统一 MLP 头**：`SQUALLModel`（1024 → 512 → 256 → G），结果 0.2812。
- **训练解码器头**：`SQUALLDecoderHead`（官方 `TransformerDecoder` depth=4 从头训练，
  mean-pool + Linear→313），结果 0.3281。
- 基准表以解码器头 0.3281 为主（官方架构解码器头可训练时最强）。
