# STFlow — 可行性评估

> 文档顺序第 11 个方法。状态：**待定（需病理基础模型权重 + 复杂自训练），暂未实现**。

## 方法概述

STFlow（Huang et al., 2025，arXiv:2506.05361）：**整张切片流匹配**（Flow Matching）生成式方法：
- 病理基础模型（UNI / GigaPath）提取 tile 特征作为条件；
- 训练**去噪器（denoiser）**拟合线性插值流 `z_t = (1-t)ε + t·z`（z=表达，ε=噪声），
  速度场 `v = z - ε`，时间步嵌入；
- 推理用 Euler-step ODE 从先验（gaussian / zinb）采样逐步生成表达；
- 关键卖点：**可扩展**到全切片规模，生成粒度可到 spot/cell。

**我们已实现的 Phoenix 就是 Latent Flow Matching**（linear interpolant + timestep embedder +
Euler ODE），机制高度同源 —— STFlow 的实现可大量复用 Phoenix 的流匹配骨架。

## 官方代码位置

`D:\hest_data\codes\STFlow\stflow\`
- `model/`：denoiser 实现
- `data/`：dataloader（patch + 表达 + 先验）
- `flow/`：`interpolant.py`、`noise.py`（线性插值 + 噪声先验）
- `app/flow/`：STFlow 训练管线
- 依赖 UNI / GigaPath 权重（HF `MahmoodLab/UNI`、`prov-gigapath`）

## 所需资源（阻塞点）

| 资源 | 状态 | 说明 |
|---|---|---|
| UNI / GigaPath 基础模型权重 | ⚠️ UNI 有（UNI1/UNI2 在服务器） | GigaPath 需 HF 下载 |
| 自训练流程 | ❌ 需搭建 | 官方对每数据集自训练 FM（denoiser 从头训） |
| 显存/算力 | 需评估 | 大模型 + 流匹配训练，耗卡 |

## 与本 benchmark 的兼容性

- **完全兼容（机制上）**：输入是我们已有的 patch/UNI2 特征 + 表达（raw counts），
  输出 per-cell 表达，可走统一归一化空间评估。
- **实现路径**：
  1. 复用 Phoenix 的流匹配骨架（interpolant / denoiser / Euler ODE / 先验采样）；
  2. 条件用 UNI2 特征（替代官方整片 tile 条件），denoiser 结构照 STFlow 官方；
  3. 先验 gaussian/zinb 采样照官方 `flow/noise.py`；
  4. 评估与 Phoenix 一致（采样确定性：固定 seed）。

## 建议

1. 本轮以**可行性文档**记录。
2. 若启动：作为 Phoenix 的兄弟实现（`input_type='feature'`），把条件从单细胞特征升级为
   含邻域上下文（可复用 slide_tiling 的 kNN 聚合特征作为条件，呼应 Local-Global 主线）。
   工作量中等，风险在于训练收敛与先验选择。
