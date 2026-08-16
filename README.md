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
- **Path2Space 重训方案**：冻结 CTransPath（官方 ctranspath.pth）+ 训练官方 MLP 头
  （`methods/path2space` 可训练版 `Path2SpaceMLP`）。同切片验证 **val_PCC 0.27+**，对比
  冻结 154-MLP 集成 ~0.02。正式 rep1→rep2 运行进行中。
- **Local-Global 双尺度模块尚未实现**：`methods/uni2_mlp/local_global.py` 仍为
  `NotImplementedError`。当前 `variant='improved'` 只是 MLP 架构改进。
- 相邻切片基准（rep1 训练 → rep2 测试）结果见下节；之后进行三层次泛化评测。

## 相邻切片基准结果（rep1 → rep2，2026-08-16 更新）

协议：50 epoch + val_PCC patience=10 早停（取 best 模型），lr=1e-3，AdamW，
`log1p_zscore` 归一化（统计量只在训练集拟合，测试集复用，防泄漏），统一评估（PCC/SPCC
归一化空间逐基因，Top-k/AUROC 逆变换回 raw counts 语义）。完整指标在各
`outputs/bench_*/test_results.json`。

### 合规结果（编码器冻结 + 头部训练，按 PCC 排序）

| 方法 | 编码器 | PCC | SPCC | Top-50 | AUROC | 备注 |
|---|---|---|---|---|---|---|
| **UNI2+MLP**（基线） | UNI2 冻结 | **0.3245** | 0.2852 | 0.575 | 0.739 | 超 UNI1 基线 0.312 |
| **DeepPT** | UNI2 冻结 | 0.3206 | 0.2834 | 0.577 | 0.738 | 官方 MLP_regression 头 |
| **Pixel2Gene**（cell 级） | HIPT 冻结 | 0.3085 | 0.2775 | 0.572 | 0.733 | ViT-256 per-cell 特征 |
| **SpatialEx** | UNI2 冻结 | 0.2964 | 0.2686 | 0.561 | 0.727 | 超官方 SpatialEx(UNI1) 0.256 |
| **ST-Net**（冻结 DenseNet） | DenseNet 冻结 | 0.2386 | 0.2318 | 0.545 | 0.694 | 微调版 0.3619 仅作参考 |
| **BLEEP**（冻结 resnet50） | resnet50 冻结 | 0.2131 | 0.2056 | 0.525 | 0.666 | 微调版 ~0.32 仅作参考 |
| **SQUALL** | 冻结 555M | 0.2116 | 0.2103 | 0.533 | 0.674 | 冻结特征 + 统一 MLP |
| Pixel2Gene（spot 级） | HIPT 冻结 | 0.1687 | 0.1729 | 0.510 | 0.644 | spot 内异质性封顶 |
| Phoenix v2 | 流模型 | 0.1509 | 0.1304 | 0.474 | 0.592 | 官方 FlowTransformerModel |
| Phoenix v1 | 流模型 | 0.1001 | 0.0982 | 0.432 | 0.573 | 313 基因适配有限 |
| STFlow | 流模型 | 0.0847 | 0.0697 | 0.422 | 0.552 | whole-slide flow matching |
| Path2Space（冻结集成） | CTransPath 冻结 | 0.0411 | 0.0354 | 0.323 | 0.526 | 冻结 154-MLP 集成不迁移（见下） |
| Hist2ST | 从头 | — | — | — | — | 统一协议下不收敛（见下） |

> Top-10 / Top-100 列略去，完整指标见各 `outputs/bench_*/test_results.json`。

### Path2Space 重训（新增，进行中）

- **做法**：冻结 CTransPath（官方 ctranspath.pth）+ **训练**官方 MLP 头（`Path2SpaceMLP`，
  架构同官方 `MLP_regression_relu_two`，768→768→313）。特征走官方管线：**Macenko 染色
  归一化 + ctx512 大上下文**（`extract_ctranspath_context.py`，Macenko 与官方逐像素
  一致 corr≈0.9994）。
- **同切片验证**（rep2 分裂，80k 训练 / 31k 测试）：epoch 7 val_PCC **0.2725**。
- **对比**：冻结集成 ~0.02-0.04（无论 Macenko/上下文/平滑如何调都 ~0）。根因：冻结模型
  在 **spot 级目标**上训练，无法适配 **per-cell** 目标；训练头可适配。
- 正式 rep1→rep2 运行进行中，完成后更新本表。

### 参考结果（编码器微调，违反"冻结"原则，仅作参考）

| 方法 | PCC | 说明 |
|---|---|---|
| ST-Net（微调 DenseNet） | 0.3619 | 优势几乎全来自编码器微调（冻结后 0.2386） |
| BLEEP（微调 resnet50） | ~0.32* | 微调增益类似（冻结后 0.2131）；*为修正 --img_size 224 后的预期值 |

## 训练中（2026-08-16）

- **Hist2ST 官方配置重训**（`outputs/bench_hist2st_official`）：`--epochs 100 --lr 1e-5
  --zinb 0.25 --zinb_coef 0.25 --bake 5 --lamb 0.5`，约 **epoch 45/100**，val_PCC ~0.18。
  官方配置（低 lr + ZINB + bake 自蒸馏）下有真实学习，验证统一协议 null 的根因是协议
  而非架构；预计最终仍远低于 UNI2 系方法。
- **BLEEP 全量测试**（非冻结，`--img_size 224` 修正）：待跑，回答"0.26 vs 0.32"之谜
  （旧 0.2594 是 test.py 缺 `--img_size 224` 的 bug）。
- **Path2Space 正式 rep1→rep2**：rep1/rep2 的 Macenko+ctx512 特征提取（并行版）→ 训练
  `Path2SpaceMLP`。

## 关键结论信号

1. **"编码器冻结、MLP 训练"下 UNI2+MLP（0.3245）领先**：简单 Foundation 特征 + 可训练头，
   超过冻结的领域模型（ST-Net 冻结 0.239、BLEEP 冻结 0.213、Path2Space 冻结 0.04）。
2. **编码器微调是 ST-Net / BLEEP 高分的唯一来源**：ST-Net 0.362→0.239（−0.12）、BLEEP
   ~0.32→0.213。公平对比（编码器冻结）下简单方法领先 → 支持"性能来自表示而非微调"。
3. **Path2Space 冻结不迁移、训练头可迁移**：冻结集成无论 Macenko / 大上下文 / 空间平滑
   都 ~0.02；换成**训练**官方 MLP 头后同切片 **0.27+**。说明低分是"冻结模型无法适配
   per-cell 目标"，不是 CTransPath 特征差。
4. **UNI2+MLP improved（仅改 MLP）= 0.3248 ≈ 基线 0.3245**：MLP 架构改进无收益 →
   提升主要来自表示（Representation）而非模型结构；下一步核心是 Local-Global 双尺度输入。
5. **Pixel2Gene cell 级（0.309）显著高于 spot 级（0.169）**：per-cell 特征克服了 spot 内
   异质性封顶。

## Hist2ST 收敛失败说明

官方设计为 350 epoch + lr 1e-5(Adam) + ZINB + 自蒸馏（bake），从原始 patch 从头训练
组织特征。统一协议（50 epoch、MSE）下 6 epoch 后 val_PCC≈0、loss 卡在归一化表达方差
（模型退化为预测每基因均值）。超参探针（子采样 20k 细胞，扫 lr 3e-3/1e-2 × ZINB
0/0.25，各 5 epoch）显示：纯 MSE 全部 val_PCC≈0；加 ZINB 最高峰值仅 0.027 且随后
过拟合回落。结论：该架构在 50-epoch 统一协议下无法收敛到有意义结果，如实记为
null result（探针日志：`logs/hist2st_probe/`）。不收敛的根因是**不使用预训练特征**
（从原始 patch 从头学），与简单方法用 UNI2 预训练特征形成对比。当前正在用官方配置
重训（见上），已出现真实学习趋势。

## 同步状态（2026-08-16）

- **GitHub main = `1a8ed92`**，已包含全部最新代码（冻结变体、Hist2ST bake、Path2Space
  可训练版、并行特征提取、README）；本地工作区干净。
- **远程代码 = 本地**（含 GHIST 实现、Path2Space 可训练版、并行提取脚本；逐文件核对一致）。
- 代码与运行结果的同步缺口以本节为准（此前 GHIST 未 scp、8 个未提交文件等历史问题均已解决）。
