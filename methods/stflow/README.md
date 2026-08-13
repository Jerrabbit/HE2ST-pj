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
| UNI / GigaPath 基础模型权重 | ✅ **UNI 够用，无需 GigaPath** | 官方 `--feature_encoder` 默认 `uni_v1_official`（UNI）；GigaPath 只是可选枚举之一（`uni_v1_official\|resnet50_trunc\|ciga\|gigapath`） |
| UNI2 特征 | ✅ 已有 | 我们的 `X_uni2.npy` 是 1536-d，正好填官方 `feature_dim` 映射的 **gigapath→1536** 槽位，无需真下载 GigaPath 权重 |
| 自训练流程 | ❌ 需搭建 | 官方对每数据集自训练 FM（denoiser 从头训） |
| 显存/算力 | 需评估 | 大模型 + 流匹配训练，耗卡 |

> **为什么文档里出现 GigaPath？** 因为 STFlow 官方 `train.py` 把 `--feature_encoder`
> 作为命令行可选项，GigaPath 只是其中之一（对应 feature_dim=1536）。官方默认用 UNI
> （feature_dim=1024）。我们用 1536-d 的 UNI2 特征时，把 feature_encoder 设为
> `gigapath` 仅是为了拿到 1536 的 feature_dim，**并不需要 GigaPath 的权重**。
> 官方从 `<embed_dataroot>/<dataset>/<feature_encoder>/fp32/<sample_id>.h5` 读预计算
> tile 特征——我们只需把 UNI2 特征按该目录结构写成 .h5 即可接入。

## 与本 benchmark 的兼容性

- **完全兼容（机制上）**：输入是我们已有的 patch/UNI2 特征 + 表达（raw counts），
  输出 per-cell 表达，可走统一归一化空间评估。
- **实现路径**：
  1. 复用 Phoenix 的流匹配骨架（interpolant / denoiser / Euler ODE / 先验采样）；
  2. 条件用 UNI2 特征（替代官方整片 tile 条件，feature_encoder=`gigapath` 取
     feature_dim=1536，权重本身用我们的 UNI2），denoiser 结构照 STFlow 官方；
  3. 先验 gaussian/zinb 采样照官方 `flow/noise.py`；
  4. 评估与 Phoenix 一致（采样确定性：固定 seed）。

## 建议

1. 本轮以**可行性文档**记录。
2. 若启动：作为 Phoenix 的兄弟实现（`input_type='feature'`），把条件从单细胞特征升级为
   含邻域上下文（可复用 slide_tiling 的 kNN 聚合特征作为条件，呼应 Local-Global 主线）。
   工作量中等，风险在于训练收敛与先验选择。
