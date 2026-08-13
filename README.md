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
├── scripts/       # 统一入口：train / test / sweep（op1/op2 调参）
└── tests/         # 单元测试
```

## 开发约定

- **公平比较**：所有方法共用 `common/` 下的数据预处理与评估模块，保证预处理和评估方式一致。
- **官方实现**：每个方法严格按原论文架构和官方代码实现，禁止自行改架构；官方源码/权重参考 `D:\hest_data\codes`。UNI+MLP 使用 UNI2 权重。
- **统一 MLP 头**：各方法如需接 MLP，须使用 `common/models/mlp_head.py` 的统一架构。
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

## 状态

- 12 方法中 7 个已实现（BLEEP、Phoenix、Path2Space、SpatialEx、ST-Net、Hist2ST、UNI2+MLP），5 个仅可行性文档（Pixel2Gene、GHIST、DeepPT、SQUALL、STFlow，阻塞详见各 `methods/<name>/README.md`）。
- 相邻切片基准（rep1 训练 → rep2 测试）已完成，结果见下节。之后进行三层次泛化评测（同切片左右半 → 同癌种多切片）、多组学与跨癌种验证。

## 相邻切片基准结果（rep1 → rep2）

协议：50 epoch + val_PCC patience=10 早停（取 best 模型），lr=1e-3，AdamW，
`log1p_zscore` 归一化（统计量只在训练集拟合，测试集复用，防泄漏），统一评估（PCC/SPCC
归一化空间逐基因，Top-k/AUROC 逆变换回 raw counts 语义）。

| 方法 | PCC | SPCC | Top-10 | Top-50 | Top-100 | AUROC | 备注 |
|---|---|---|---|---|---|---|---|
| **UNI2+MLP**（基线） | **0.3245** | 0.2852 | 0.510 | 0.575 | 0.626 | 0.739 | 超 UNI1 基线 0.312 |
| **SpatialEx** | 0.2964 | 0.2686 | 0.493 | 0.561 | 0.616 | 0.727 | 超官方 SpatialEx(UNI1) 0.256 |
| Phoenix | 0.1001 | 0.0982 | 0.318 | 0.432 | 0.522 | 0.573 | 官方为 30 基因蛋白任务，313 基因适配有限 |
| Hist2ST | — | — | — | — | — | — | **统一协议下不收敛**（见下） |

**Hist2ST 收敛失败说明**：官方设计为 350 epoch + lr 1e-5(Adam) + ZINB + 自蒸馏（bake），
从原始 patch 从头训练组织特征。统一协议（50 epoch、MSE）下 6 epoch 后 val_PCC≈0、
loss 卡在归一化表达方差（模型退化为预测每基因均值）。超参探针（子采样 20k 细胞，扫
lr 3e-3/1e-2 × ZINB 0/0.25，各 5 epoch）显示：纯 MSE 全部 val_PCC≈0；加 ZINB 最高
峰值仅 0.027 且随后过拟合回落。结论：该架构在 50-epoch 统一协议下无法收敛到有意义
结果，如实记为 null result（探针日志：`logs/hist2st_probe/`）。不收敛的根因是
**不使用预训练特征**（从原始 patch 从头学），与简单方法用 UNI2 预训练特征形成对比。
