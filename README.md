# HE2ST-pj

**HE→ST 任务**：从 H&E 组织学图像预测空间转录组（ST）基因表达。课题总体要求见本地文档 `D:\Desktop\其他学习资料\大二下\复芏\HE2ST-pj\课题总体情况及实现要求.txt`。

核心研究问题：
1. 简单有效的模型能否在 HE→ST 任务中达到甚至超过复杂空间模型？
2. Local（局部组织形态）与 Global（组织上下文）信息是否互补？如何简单利用两者？
3. 如何建立覆盖不同泛化场景的标准 Benchmark，公平比较现有方法？

## 目录结构

```
HE2ST-pj/
├── common/        # 所有方法共用模块（数据预处理、特征、模型、评估、工具）
├── methods/       # 每个方法一个独立文件夹（12 种方法）
│   └── uni2_mlp/  #   UNI2+MLP 基线 + Local-Global 双尺度改进
├── scripts/       # 统一入口：train / test / 特征提取（extract_*）
└── tests/         # 单元测试
```

## 开发约定

- **公平比较**：所有方法共用 `common/` 下的数据预处理与评估模块，保证预处理和评估方式一致。
- **核心原则（编码器冻结、MLP 训练）**：特征提取用到的编码器（UNI2 / CTransPath / HIPT / resnet50 / DenseNet 等）一律**冻结**（通常以预提取特征形式参与），接的 MLP / 预测头**训练**。该设置保证公平对比，并排除"编码器微调"带来的不公平优势（详见"关键结论信号"）。
- **官方实现**：每个方法严格按原论文架构和官方代码实现，禁止自行改架构；官方源码/权重参考 `D:\hest_data\codes`。UNI+MLP 使用 UNI2 权重。
- **统一 MLP 头**：纯 embedding 需外接头的方法使用 `common/models/mlp_head.py` 的统一架构；自带官方预测头的方法沿用官方头（如 DeepPT MLP_regression、Pixel2Gene ForwardSum、Path2Space MLP_regression_relu_two）。
- **MPP 对齐**：多切片情形必须统一 MPP（`common/data/alignment.py` 含坐标对齐与 MPP 统一对齐两个版本）。
- **评估指标**：回归（PCC、SPCC）、排序（Top-k 准确率）、分类（AUROC），见 `common/eval/metrics.py`。

## 环境

- 本地调试：conda 环境 `myenv`
- 远程服务器 `gpu-server`：conda 环境 `myenv1`，代码本地编写后上传运行

## 测试

```bash
conda activate myenv
python -m pytest tests/ -v
```

## 状态（2026-08-16 更新）

- **12 种方法全部实现**（BLEEP、Phoenix、Path2Space、SpatialEx、Pixel2Gene、GHIST、ST-Net、
  Hist2ST、DeepPT、SQUALL、STFlow、UNI2+MLP），各带独立文件夹 + 统一 harness。阻塞项详见各
  `methods/<name>/README.md`。
- **"编码器冻结、MLP 训练"原则已确立**（用户明确）：据此 ST-Net / BLEEP 的基准结果改用
  **冻结编码器版**（ST-Net 0.2386、BLEEP 0.2131），微调版仅作参考（见下）。
- **Path2Space 重训 ✅ 完成**：冻结 CTransPath（官方 ctranspath.pth）+ 训练官方 MLP 头
  （`methods/path2space` 可训练版 `Path2SpaceMLP`）。正式 rep1→rep2 **PCC 0.2780**
  （冻结 154-MLP 集成 0.0411 → 训练头 0.2780，见下）。
- **Local-Global 双尺度模块已实现**（单次 forward token 复用，2026-08-17 重构）：
  `methods/uni2_mlp/local_global.py` 已就绪，`op1 sweep` 进行中（见下）。
- 相邻切片基准（rep1 训练 → rep2 测试）结果见下节；之后进行三层次泛化评测。

## 相邻切片基准结果（rep1 → rep2，2026-08-16 更新）

协议：50 epoch + val_PCC patience=10 早停（取 best 模型），lr=1e-3，AdamW，
`log1p_zscore` 归一化（统计量只在训练集拟合，测试集复用，防泄漏），统一评估（PCC/SPCC
归一化空间逐基因，Top-k/AUROC 逆变换回 raw counts 语义）。完整指标在各
`outputs/bench_*/test_results.json`。

### 合规结果（编码器冻结 + 头部训练，按 PCC 排序）

> 完整柱状图见 [`benchmark_pcc_bar.png`](benchmark_pcc_bar.png)（12 方法 PCC，基线标红）。

| 方法 | 编码器 | PCC | SPCC | Top-10 | Top-50 | Top-100 | AUROC | 备注 |
|---|---|---|---|---|---|---|---|---|
| **UNI2+MLP Local+Global**（改进） | UNI2 冻结 | **0.3712** | 0.3071 | 0.539 | 0.588 | 0.630 | 0.759 | l1=112+l2=56，核心创新 |
| **UNI2+MLP**（基线） | UNI2 冻结 | 0.3245 | 0.2852 | 0.510 | 0.575 | 0.626 | 0.739 | 超 UNI1 基线 0.312 |
| **DeepPT** | UNI2 冻结 | 0.3206 | 0.2834 | 0.507 | 0.577 | 0.629 | 0.738 | 官方 MLP_regression 头（UNI2 特征版；ResNet50 忠实版 0.2628 见下） |
| **GHIST** | UNet 从头 | 0.3164 | 0.2952 | 0.500 | 0.525 | 0.516 | 0.700 | 官方 Framework（核 mask + 图），从零训练 |
| **Pixel2Gene**（cell 级） | HIPT 冻结 | 0.2699 | 0.2600 | 0.465 | 0.541 | 0.629 | 0.718 | 官方 ForwardSum 头（384-d 适配）；统一 MLP 版 0.3085 作参考 |
| **SpatialEx** | UNI2 冻结 | 0.2964 | 0.2686 | 0.493 | 0.561 | 0.616 | 0.727 | 超官方 SpatialEx(UNI1) 0.256 |
| **SQUALL** | 冻结 555M | 0.2812 | 0.2581 | 0.481 | 0.560 | 0.622 | 0.715 | 0-1 输入修复后复核值（见下） |
| **Path2Space**（重训训练头） | CTransPath 冻结 | 0.2780 | 0.2555 | 0.476 | 0.561 | 0.625 | 0.714 | 训练官方 MLP 头适配 per-cell（见下） |
| **ST-Net**（冻结 DenseNet） | DenseNet 冻结 | 0.2386 | 0.2318 | 0.457 | 0.545 | 0.620 | 0.694 | 微调版 0.3619 仅作参考 |
| **BLEEP**（冻结 resnet50） | resnet50 冻结 | 0.2131 | 0.2056 | 0.440 | 0.525 | 0.601 | 0.666 | 微调版 0.3235 仅作参考 |
| Pixel2Gene（spot 级） | HIPT 冻结 | 0.1687 | 0.1729 | 0.379 | 0.510 | 0.622 | 0.644 | spot 内异质性封顶 |
| Phoenix v2 | 流模型 | 0.1509 | 0.1304 | 0.409 | 0.474 | 0.539 | 0.592 | 官方 FlowTransformerModel |
| Phoenix v1 | 流模型 | 0.1001 | 0.0982 | 0.318 | 0.432 | 0.522 | 0.573 | 313 基因适配有限 |
| STFlow | 流模型 | 0.0847 | 0.0697 | 0.302 | 0.422 | 0.533 | 0.552 | whole-slide flow matching |
| Path2Space（冻结集成） | CTransPath 冻结 | 0.0411 | 0.0354 | 0.148 | 0.323 | 0.432 | 0.526 | 冻结 154-MLP 集成不迁移（见下） |
| Hist2ST（官方配置） | 从头 | 0.2139 | 0.2046 | 0.431 | 0.531 | 0.611 | 0.670 | 独立协议（见下），统一协议下不收敛 |

> 统一协议：50ep + 早停 + log1p_zscore（统计量仅在训练集拟合）。Hist2ST 为**独立协议**
> （官方 100ep + lr1e-5 + ZINB + bake 自蒸馏），故单独一行不参与统一协议排序。全部原始
> 数值见各 `outputs/bench_*/test_results.json`。

### Path2Space 重训（✅ 完成，正式 rep1→rep2）

- **做法**：冻结 CTransPath（官方 ctranspath.pth）+ **训练**官方 MLP 头（`Path2SpaceMLP`，
  架构同官方 `MLP_regression_relu_two`，768→768→313）。特征走官方管线：**Macenko 染色
  归一化 + ctx512 大上下文**（`extract_ctranspath_context.py`，Macenko 与官方逐像素
  一致 corr≈0.9994）。
- **正式 rep1→rep2**（rep1 164k 训练 → rep2 111k 测试）：早停于 epoch 13（10ep 未提升），
  best val_PCC **0.2780**；全量指标 **PCC 0.2780 / SPCC 0.2555 / top10 0.476 /
  top50 0.561 / top100 0.625 / AUROC 0.713**。
- **对比**：冻结集成 ~0.02-0.04（无论 Macenko/上下文/平滑如何调都 ~0）→ 训练头
  **+0.24**。根因：冻结模型在 **spot 级目标**上训练，无法适配 **per-cell** 目标；
  训练头可适配。跨切片（rep1→rep2）下有效，说明是表示适配而非过拟合。
- **同切片验证**（rep2 分裂，80k/31k）：val_PCC **0.2725**，与跨切片 0.2780 一致。

### DeepPT ResNet50 忠实版（对比论文差异归因）

官方 DeepPT（Hoang 2024, Nature Cancer）用 **ResNet50-ImageNet → AE(2048→512) → 官方 MLP_regression(512→512→G)** 预测整片 bulk 表达。为对照
SpatialEx 论文（其报告 DeepPT 在相同乳腺癌 Xenium 数据上 **~0.205**，低于 SpatialEx），
按官方流程实现忠实版（per-cell 适配）：官方 `ResNet50_IMAGENET1K_V2.pt` 提 2048-d 特征
→ AE 重构预训练 → 官方 MLP 头（`outputs/bench_deeppt_resnet50`）。

**结果对比**：

| 版本 | 特征 | PCC | 说明 |
|---|---|---|---|
| DeepPT（UNI2 特征版） | UNI2 冻结 | 0.3206 | 与 UNI2+MLP 基线 0.3245 接近 |
| **DeepPT（ResNet50 忠实版）** | ResNet50-ImageNet | **0.2628** | 官方特征 + 官方头（本 benchmark 统一协议） |
| SpatialEx 论文 DeepPT | ResNet50 | ~0.205 | 论文 280 基因面板等独立协议 |

**归因**：UNI2 → ResNet50 使 PCC **0.3206 → 0.2628（−0.06）**，特征提取器是主要因素；
剩余差距（0.2628 vs 0.205）来自协议差异（313 vs 280 基因、per-cell 粒度、归一化/训练设定）。
同时印证：**表示（Representation）强于架构**——相同官方头下，UNI2 特征（0.32）显著优于
ResNet50（0.26）。

### STFlow 低分归因（实现校验，2026-08-19）

STFlow（Huang et al. 2025）全基因 PCC 0.0847 偏低。按原论文协议做两层校验：

| 评估 | 全 313 基因 | Top50 HVG |
|---|---|---|
| zscore 空间 | 0.0847 | 0.1846 |
| **log1p 空间（官方 normalize）** | 0.0758 | **0.2569** |
| 论文 | — | 0.3-0.4 |

**结论**：① 基因集是主因之一（Top50 HVG 显著高于全基因）；② **归一化空间是关键差异**
（log1p 比 zscore 的 Top50 HVG 高 +0.07）；③ 剩余差距（0.257 vs 0.3-0.4）来自
**数据粒度**（Xenium per-cell 稀疏噪声 vs 论文 Visium spot 聚合）与先验
（官方默认 zinb，我们用 gaussian）。架构（SpatialTransformer + 流匹配）忠实官方。

### 参考结果（编码器微调 / 结构变体，非统一冻结协议，仅作参考）

| 方法 | PCC | SPCC | Top-50 | AUROC | 说明 |
|---|---|---|---|---|---|
| ST-Net（微调 DenseNet） | 0.3619 | 0.307 | 0.587 | 0.760 | 优势几乎全来自编码器微调（冻结后 0.2386） |
| BLEEP（微调 resnet50） | 0.3235 | 0.283 | 0.560 | 0.732 | 0.26 vs 0.32 之谜已解决：旧 0.2594 是 test.py 缺 `--img_size 224` 的 bug；`--img_size 224` 全量测试确认 **0.3235**（冻结后 0.2131） |
| UNI2+MLP improved（仅改 MLP） | 0.3248 | 0.284 | 0.576 | 0.736 | ≈ 基线 0.3245 → MLP 结构改进无收益，提升来自表示而非结构 |

## 当前进度（2026-08-18）

### 已完成

- **12 种 benchmark 方法全部有最终合规结果**（见上表）；批任务全部收尾：
  Hist2ST 官方 **0.2139**、BLEEP 全量 **0.3235**、Path2Space 正式 **0.2780**、
  SQUALL 0-1 复核 **0.2812**、DeepPT ResNet50 忠实版 **0.2628**（见 DeepPT 章节）。
- **GHIST 数据管线** ✅ 完成：Xenium 核多边形 → HE 像素核分割 mask（精确 2D 仿射，
  残差 <0.5px，对齐 <2px），`data/ghist_rep1/rep2`（164000/111345 核）；tiling 训练
  （官方 256×256 patch, overlap 30）已跑通。

### 运行中

- **Local+Global 最终训练 + 消融**：best l1=112 + l2=56，50ep 完整指标（PCC/SPCC/
  Top-k/AUROC）+ 消融（Global-only / Local-only / Local+Global）。

### 已完成补充

- **op2 sweep ✅ 完成**：固定 l1=112，完整 l2 曲线为**倒 U 型**（28:0.3718 / 42:0.3718 /
  **56:0.3727 峰值** / 70:0.3697 / 84:0.3671 / 98:0.3639 / 112:0.3601）。**best l2=56**，
  扩展测试（28/42）确认 56 是最佳——Local 视野过小（28/42）或过大（≥70）均下降。
  曲线见 `outputs/sweep_op2/` 与 `outputs/sweep_op2_ext/`。
- **Local+Global 最终训练 + 消融 ✅ 完成**（50ep，best l1=112 + l2=56）：

  | 配置 | PCC | SPCC | Top-50 | AUROC |
  |---|---|---|---|---|
  | Global-only（l1=112） | 0.3366 | 0.291 | 0.579 | 0.744 |
  | **Local+Global**（concat） | **0.3712** | 0.307 | 0.588 | 0.759 |
  | **Local-only**（l2=56） | **0.3732** | 0.310 | 0.590 | 0.762 |

  **消融结论**：Local 分支主导（Local-only ≈ Local+Global，Global 的 CLS 几乎无增量）。
  原因：中心 token 经自注意力已携带全局上下文（单次 forward 设计固有属性），Local 特征
  实为"局部区域 + 全局上下文"，CLS 与其冗余。最终成绩 **Local+Global 0.3712**
  （vs 基线 0.3245，+0.047）。

### 已完成补充

- **op1 sweep ✅ 完成（16 个 l1 全跑完）**：PCC-l1 曲线为**倒 U 型**，峰值 **l1=112
  （0.3365）**，比标准 224 基线（0.3245）高 +0.012；过大（>140）或过小（<84）均下降。
  结论：Global 分支最优视野 ≈40.7 µm（l1=112）。曲线见 `outputs/sweep_op1/op1_curve.png`。
- **STFlow 实现校验 ✅ 完成**：见上文 STFlow 归因专节（log1p 空间 Top50 HVG 0.2569）。

### 已完成补充

- **GHIST** ✅ 完成：训练（best val_PCC 0.3164@ep28）+ 正式测试 **PCC 0.3164**
  （修复 ref_expr 传递 bug 后与训练一致）；从零训练的整片图方法追平 UNI2 特征系。
- **STFlow Top50 HVG 校验** ✅：全 313 基因 0.0847 → **Top50 HVG 0.1846**，
  基因集是低分主因之一，但距论文 0.3-0.4 仍有差距（继续查归一化空间/数据粒度）。

## 关键结论信号

1. **"编码器冻结、MLP 训练"下 UNI2+MLP（0.3245）领先**：简单 Foundation 特征 + 可训练头，
   超过冻结的领域模型（ST-Net 冻结 0.239、BLEEP 冻结 0.213、Path2Space 冻结 0.04）。
2. **编码器微调是 ST-Net / BLEEP 高分的唯一来源**：ST-Net 0.362→0.239（−0.12）、BLEEP
   0.3235→0.213。公平对比（编码器冻结）下简单方法领先 → 支持"性能来自表示而非微调"。
3. **Path2Space 冻结不迁移、训练头可迁移**：冻结集成无论 Macenko / 大上下文 / 空间平滑
   都 ~0.02；换成**训练**官方 MLP 头后同切片 **0.27+**。说明低分是"冻结模型无法适配
   per-cell 目标"，不是 CTransPath 特征差。
4. **UNI2+MLP improved（仅改 MLP）= 0.3248 ≈ 基线 0.3245**：MLP 架构改进无收益 →
   提升主要来自表示（Representation）而非模型结构；下一步核心是 Local-Global 双尺度输入。
5. **Pixel2Gene cell 级（0.309）显著高于 spot 级（0.169）**：per-cell 特征克服了 spot 内
   异质性封顶。

## 方法实现清单与忠实度分析（2026-08-16）

> 统一训练协议：50 epoch + val_PCC patience=10 早停（取 best），lr=1e-3，AdamW，
> `log1p_zscore` 归一化（统计量只在训练集拟合）。特例方法（Hist2ST/Phoenix/STFlow/
> Path2Space 冻结）走各自官方训练流程，见备注。
> **项目原则**：编码器冻结 + MLP 训练（ST-Net/BLEEP 用 `--no_finetune`）。

| 方法 | 架构 | 编码器 | 头部/模型 | 官方忠实度 | PCC | 指标评估 |
|---|---|---|---|---|---|---|
| **UNI2+MLP**（基线） | 特征回归 | UNI2 冻结（1536-d CLS） | 统一 MLPHead 1536→512→256→313 | 本仓库基线 | **0.3245** | 合理，作为对比基准 |
| **DeepPT** | 特征回归 | UNI2 冻结 | AE(1536→512)+官方 `MLP_regression`（Linear→Dropout→Linear） | ✅ 官方头原样；AE 为单细胞适配 | 0.3206 | 合理 |
| **Pixel2Gene** | 特征回归 | HIPT 冻结 | 官方 `ForwardSumModel`（576→256×4 FFN+ELU 输出头） | ✅ 官方头；cell 级为方案 B | 0.3085 / 0.169 | cell 级合理，spot 级受异质性封顶 |
| **SpatialEx** | 超图 GNN | UNI2 冻结 | MLP→HGNN→Linear，超图 kNN k=7 | ✅ 官方架构；cell-level MSE 为可选适配 | 0.2964 | 合理，超官方(UNI1)0.256 |
| **ST-Net** | CNN 回归 | DenseNet **冻结**（`--no_finetune`） | Linear 回归头 | ✅ 官方 DenseNet 架构；冻结为项目原则 | 0.2386 | 合理（冻结）；微调 0.3619 仅参考 |
| **BLEEP** | 对比学习 | resnet50 **冻结** | 对比投影头 | ✅ 官方架构；冻结为项目原则 | 0.2131 | 合理（冻结）；微调 0.3235 仅参考 |
| **SQUALL** | Transformer 多模态 | 冻结 555M 特征 | 统一 MLP（官方 TransformerDecoder 头待训，task #18） | ⚠️ 移植（未用官方解码器） | 0.2812 | 冻结解码器不迁移（~0.02），训练头合理；0-1 输入修复后复核值 |
| **Phoenix** | 流匹配（生成） | 流模型 | 官方 `FlowTransformerModel` | ✅ v2 官方架构 | 0.1509 / 0.100 | 生成式采样不适配 per-cell 回归 |
| **STFlow** | 流匹配（生成） | UNI2 冻结 | `SpatialTransformer` 去噪器（ROI 级） | ✅ 官方架构纯 torch 移植 | 0.0847 | 全基因低分因基因集+粒度：Top50 HVG 校验 **0.257（log1p 空间）**，论文 0.3-0.4 差距来自 per-cell vs spot（见下） |
| **Path2Space** | MLP 集成 | CTransPath 冻结 | 官方 `MLP_regression_relu_two`（**训练**头） | ✅ 重训方案（冻结集成 ~0.04 不迁移） | **0.2780** | 重训头适配 per-cell，跨切片有效 |
| **GHIST** | UNet+图 | UNet 从头 | Framework 图模型 | ✅ 官方 Framework 移植 | **0.3164** | 数据管线 + tiling 完成；从零训练追平 UNI2 特征系 |
| **Hist2ST** | 图 Transformer | 从头 | Convmixer+Transformer+GNN | ✅ 官方架构 | null→**0.2139**（官方配置） | 协议是根因，官方配置有真实学习 |

### 忠实度与指标评估要点

1. **特征回归类方法（UNI2+MLP / DeepPT / Pixel2Gene / SpatialEx / SQUALL / Path2Space）**
   均以**冻结的 Foundation 特征**为输入、训练各自头部——结构最忠实、指标 0.21-0.32，反映"表示 + 简单头"的力量。
2. **编码器微调类（ST-Net / BLEEP）**：官方架构本身微调编码器；按项目"冻结原则"改用 `--no_finetune`
   （0.239 / 0.213）。微调版（0.362 / 0.324）作为"微调增益"的参考证据，不计入合规表。
3. **生成式方法（Phoenix / STFlow）**：忠实复现官方流匹配架构，但**生成式采样不适合 per-cell 回归**，
   指标 0.08-0.15 属方法性质使然（与基线同特征对比：UNI2+MLP 0.32 vs STFlow 0.08）。
4. **从头学习方法（Hist2ST / GHIST）**：无预训练特征，统一协议下难收敛（Hist2ST null），
   官方配置重训才有学习（**0.2139**）；GHIST 因需核分割+细胞型管线尚未运行。
5. **Path2Space**：冻结 154-MLP 集成在 Xenium per-cell 上 ~0.02-0.04（spot 级训练目标不迁移）；
   **重训**官方 MLP 头（冻结 CTransPath）后正式 rep1→rep2 **0.2780**。
6. **SQUALL**：官方解码器推理（`forward_rgb_to_expr` → 15757 基因）在 per-cell 上 ~0.02
   （解码器输出近常量，不迁移）；冻结 555M 编码器特征 + 训练统一 MLP = **0.2812**
   （0-1 输入修复后复核值）才是有效表示。
   注：早期特征提取误用 0-255 输入（官方教程为 0-1），冻结特征基线待用 0-1 复核。

## 数据预处理与评估统一流程（公平性保障）

### 数据预处理（共用 `common/data/`，所有方法读同一数据目录格式）

1. **h5ad → 数据目录**（`preprocess.py`）：`extract_patches_from_h5ad` 从 h5ad 坐标
   （image_col/image_row）在整片 H&E 上裁 256×256 patch，导出 `patches/cell_{id}.png`、
   `gene_expression.npy`（**raw counts**，log1p 逆变换回整数）、`gene_names.txt`、
   `metadata.csv`（含 x_centroid/y_centroid 像素坐标）→ **所有方法输入一致**。
2. **特征提取**（per-method，编码器不同故分开，均冻结）：UNI2、CTransPath、HIPT、
   SQUALL、Local+Global → `data_dir/X_*.npy`；方法只读特征。
3. **归一化**（`expression.py`）：默认 `log1p_zscore`，统计量（mean/std）只在**训练集**
   拟合、测试集复用（`save_stats_json` 防泄漏）。
4. **MPP 对齐**（`alignment.py`）：`align_by_coords`（按坐标）与 `align_by_mpp`（统一 MPP）
   两版本（当前 rep1/rep2 同 MPP，天然一致）。
5. **数据集**（`dataset.py`）：`HESTDataset`（patch 输入）与 `FeatureDataset`（特征输入，
   支持多文件 concat，Local+Global 用）。

### 评估（共用 `common/benchmark/harness.py` + `common/eval/metrics.py`）

1. **统一指标**（`metrics.py`）：PCC/SPCC（归一化空间逐基因）、Top-k（逐细胞 raw counts
   语义）、AUROC（逐基因 raw counts>0）。全部经 `compute_metrics_vectorized`（`harness.py`）。
2. **统一协议**：50 epoch + val_PCC patience=10 早停 + lr=1e-3 + AdamW + log1p_zscore；
   `fit()` 统一训练，`evaluate()` 统一评估。
3. **各方法评估路径**（最终指标都走同一个 `compute_metrics_vectorized`）：
   - **标准 harness**（BLEEP/DeepPT/Phoenix/SQUALL/ST-Net/uni2_mlp/Pixel2Gene/Path2Space
     训练版）：`evaluate(model, loader, ...)`。
   - **整片图方法**（SpatialEx/Hist2ST/STFlow/GHIST）：`evaluate_slide` 整片建图/ROI 推理，
     指标用同一个 `compute_metrics_vectorized`（语义逐字节一致）。
   - **冻结推理**（Path2Space 冻结 / SQUALL 解码器）：各自 test 脚本，同样调用
     `compute_metrics_vectorized`。
4. **归一化语义**：PCC/SPCC 在归一化空间（统计量来自训练集）；Top-k/AUROC 经
   `_invert_normalization` 逆变换回 raw counts 语义。

### 检查结论

- **所有方法最终指标均来自 `compute_metrics_vectorized`**（或其调用者 evaluate/evaluate_slide），
  指标语义一致 → **公平比较成立**。
- 数据预处理共用同一数据目录格式、归一化与统计量防泄漏；特征提取因编码器不同而分开
  （各方法按其官方编码器提取，属合理差异），输入粒度和输出语义统一。

## 下一步实验计划

按优先级：

1. **Local+Global 双尺度正式实验**（收尾中）：~~op1 sweep~~ ✅（best l1=112, 0.3365）、
   ~~op2 sweep~~ ✅（best l2=56, 0.3727）→ **最终 50ep 完整指标 + 消融进行中**
   （Global-only / Local-only / Local+Global）。
2. ~~GHIST 训练收尾~~ ✅ 完成：**PCC 0.3164**（已补基准表）。
3. ~~STFlow 实现校验~~ ✅ 完成：log1p 空间 Top50 HVG **0.2569**（见归因专节）。
4. **官方预测头补齐**（用户原则：官方有头不换统一 MLP）：
   - ~~Pixel2Gene cell 级改用官方 ForwardSumModel~~ ✅ 完成（PCC 0.2699）；
   - **SQUALL 官方 TransformerDecoder 头训练**（`--save_tokens` → `SQUALLDecoderHead`）。
5. **三层次泛化评测**：同切片左右半 → 相邻切片（MPP 统一）→ 同癌种多切片。
6. **多组学验证**：肾癌切片（基因+蛋白双组学）。
5. **跨癌种验证**：结直肠/肺癌/卵巢训练 → 乳腺癌测试。
6. 同步更新 README/CLAUDE.md 与 GitHub。

## Hist2ST 收敛失败说明

官方设计为 350 epoch + lr 1e-5(Adam) + ZINB + 自蒸馏（bake），从原始 patch 从头训练
组织特征。统一协议（50 epoch、MSE）下 6 epoch 后 val_PCC≈0、loss 卡在归一化表达方差
（模型退化为预测每基因均值）。超参探针（子采样 20k 细胞，扫 lr 3e-3/1e-2 × ZINB
0/0.25，各 5 epoch）显示：纯 MSE 全部 val_PCC≈0；加 ZINB 最高峰值仅 0.027 且随后
过拟合回落。结论：该架构在 50-epoch 统一协议下无法收敛到有意义结果，如实记为
null result（探针日志：`logs/hist2st_probe/`）。不收敛的根因是**不使用预训练特征**
（从原始 patch 从头学），与简单方法用 UNI2 预训练特征形成对比。

**官方配置重训结果（✅ 完成）**：`--epochs 100 --lr 1e-5 --zinb 0.25 --zinb_coef 0.25
--bake 5 --lamb 0.5`，early stop at **epoch 66**，全量 **PCC 0.2139 / SPCC 0.205 /
top100 0.611 / AUROC 0.670**。证明统一协议 null 的根因是**协议**（50ep + MSE 无预训练
特征下无法收敛），而非架构本身——但即便如此，从头学方法（0.21）仍远低于简单
Foundation 特征方法（UNI2+MLP 0.32），再次支持"表示 > 架构/训练技巧"。

## 同步状态（2026-08-16）

- **GitHub main = `977bdf4`**，已包含全部最新代码（冻结变体、Hist2ST bake、Path2Space
  可训练版、并行特征提取、Local+Global 双尺度框架、README）；本地工作区干净。
- **远程代码 = 本地**（含 GHIST 实现、Path2Space 可训练版、并行提取脚本、Local+Global
  框架；逐文件核对一致）。
- 运行结果文件（`outputs/bench_*/`）不入库（gitignore），保存在远程。
