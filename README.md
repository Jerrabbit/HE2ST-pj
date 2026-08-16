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

- **12 种方法全部实现**（BLEEP、Phoenix、Path2Space、SpatialEx、Pixel2Gene、GHIST、ST-Net、
  Hist2ST、DeepPT、SQUALL、STFlow、UNI2+MLP），各带独立文件夹 + 统一 harness。阻塞项（缺
  官方权重、需核分割管线等）详见各 `methods/<name>/README.md`。
- **Local-Global 双尺度模块尚未实现**：`methods/uni2_mlp/local_global.py` 仍为
  `NotImplementedError`。当前 `variant='improved'` 只是 MLP 架构改进（bias 均值初始化 +
  残差 + SiLU），非双尺度输入。
- 相邻切片基准（rep1 训练 → rep2 测试）已完成，结果见下节。之后进行三层次泛化评测
  （同切片左右半 → 同癌种多切片）、多组学与跨癌种验证。

## 相邻切片基准结果（rep1 → rep2，2026-08-16 更新）

协议：50 epoch + val_PCC patience=10 早停（取 best 模型），lr=1e-3，AdamW，
`log1p_zscore` 归一化（统计量只在训练集拟合，测试集复用，防泄漏），统一评估（PCC/SPCC
归一化空间逐基因，Top-k/AUROC 逆变换回 raw counts 语义）。完整指标在各
`outputs/bench_*/test_results.json`。

| 方法 | PCC | SPCC | Top-10 | Top-50 | Top-100 | AUROC | 备注 |
|---|---|---|---|---|---|---|---|
| **ST-Net**（微调 DenseNet） | **0.3619** | 0.3070 | 0.537 | 0.587 | 0.628 | 0.759 | 优势主要来自编码器微调（见下） |
| **UNI2+MLP**（基线） | **0.3245** | 0.2852 | 0.510 | 0.575 | 0.626 | 0.739 | 超 UNI1 基线 0.312 |
| **UNI2+MLP improved**（仅改 MLP） | 0.3248 | 0.2836 | 0.511 | 0.576 | 0.628 | 0.736 | 与基线几乎无差（见下） |
| DeepPT | 0.3206 | 0.2834 | 0.507 | 0.577 | 0.629 | 0.738 | 官方 MLP_regression 头 |
| Pixel2Gene（cell 级） | 0.3085 | 0.2775 | 0.498 | 0.572 | 0.629 | 0.733 | ViT-256 per-cell 特征 |
| SpatialEx | 0.2964 | 0.2686 | 0.493 | 0.561 | 0.616 | 0.727 | 超官方 SpatialEx(UNI1) 0.256 |
| BLEEP（旧结果） | 0.2594 | 0.2373 | 0.452 | 0.535 | 0.603 | 0.687 | test.py 缺 --img_size 224 的 bug，待重跑 |
| ST-Net（冻结编码器） | 0.2386 | 0.2318 | 0.456 | 0.545 | 0.620 | 0.694 | 冻结后低于 UNI2+MLP |
| SQUALL | 0.2116 | 0.2103 | 0.433 | 0.533 | 0.610 | 0.674 | 冻结 555M 特征 + 统一 MLP |
| Pixel2Gene（spot 级） | 0.1687 | 0.1729 | 0.379 | 0.510 | 0.622 | 0.644 | HIPT spot 级特征，受 spot 内异质性封顶 |
| Phoenix v2 | 0.1509 | 0.1304 | 0.409 | 0.474 | 0.538 | 0.592 | 官方 FlowTransformerModel |
| Phoenix v1 | 0.1001 | 0.0982 | 0.318 | 0.432 | 0.522 | 0.573 | 官方为 30 基因蛋白任务，313 基因适配有限 |
| STFlow | 0.0847 | 0.0697 | 0.302 | 0.422 | 0.533 | 0.552 | whole-slide flow matching，ROI 级适配 |
| Path2Space | 0.0411 | 0.0354 | 0.148 | 0.323 | 0.432 | 0.526 | 缺官方权重，用 CTransPath 冻结特征 + 239/313 基因部分覆盖（缺失补 0） |
| Hist2ST | — | — | — | — | — | — | 统一协议下不收敛（见下） |

### 训练中（2026-08-16）

- **Hist2ST 官方配置重训**（`outputs/bench_hist2st_official`）：`--epochs 100 --lr 1e-5
  --zinb 0.25 --zinb_coef 0.25 --bake 5 --lamb 0.5`，已跑至 epoch 28/100，val_PCC 从
  0.088 稳步升至 **0.184**——官方配置（低 lr + ZINB + bake 自蒸馏）下确有真实学习，
  验证统一协议不收敛的根因是协议而非架构，但预计最终仍远低于 UNI2 系方法。
- **BLEEP 冻结编码器重训**（`outputs/bench_bleep_frozen`）：`--no_finetune --img_size
  224 --batch_size 256`，已跑至 epoch 36/50，val_PCC≈0.16（低于微调版，与 ST-Net
  结论一致：编码器微调是主要性能来源）。注意 BLEEP 旧 test_results.json（0.2594）
  是 test.py 缺 `--img_size 224` 的 bug 所致，正确值预计约 0.32（诊断见 memory）。

### 关键结论信号

1. **ST-Net 微调后最高（0.362），但优势几乎全来自编码器微调**——冻结编码器后 0.239
   低于 UNI2+MLP 0.3245。公平对比（固定特征）下简单方法仍领先。
2. **UNI2+MLP improved（仅改 MLP：bias 均值初始化 + 残差 + SiLU）= 0.3248，与基线
   0.3245 几乎无差** → 印证"性能提升主要来自表示（Representation）而非模型结构"，
   下一步核心是 Local-Global 双尺度输入（尚未实现）。
3. **用 UNI2/预训练特征的方法（uni2_mlp / deeppt / pixel2gene_cell ~0.31-0.32）普遍
   超过需整片/spot 上下文的复杂空间方法（spatialex 0.30、squall 0.21、phoenix
   0.10-0.15、stflow 0.08、path2space 0.04）**——支持"Foundation 特征 > 复杂空间建模"。
4. **Pixel2Gene cell 级（0.309）显著高于 spot 级（0.169）**：per-cell 特征克服了 spot
   内异质性封顶。

### Hist2ST 收敛失败说明

官方设计为 350 epoch + lr 1e-5(Adam) + ZINB + 自蒸馏（bake），从原始 patch 从头训练
组织特征。统一协议（50 epoch、MSE）下 6 epoch 后 val_PCC≈0、loss 卡在归一化表达方差
（模型退化为预测每基因均值）。超参探针（子采样 20k 细胞，扫 lr 3e-3/1e-2 × ZINB
0/0.25，各 5 epoch）显示：纯 MSE 全部 val_PCC≈0；加 ZINB 最高峰值仅 0.027 且随后
过拟合回落。结论：该架构在 50-epoch 统一协议下无法收敛到有意义结果，如实记为
null result（探针日志：`logs/hist2st_probe/`）。不收敛的根因是**不使用预训练特征**
（从原始 patch 从头学），与简单方法用 UNI2 预训练特征形成对比。当前正在用官方配置
重训（见上），已出现真实学习趋势。

## 同步状态（2026-08-16）

- **Git（origin/main）落后**：本地 1 个未推送 commit（81baeae BLEEP 修复）+ 8 个未提交
  文件（BLEEP `--img_size`/`--no_finetune`、Hist2ST bake/lamb、ST-Net `--no_finetune`、
  Path2Space 239 基因部分覆盖、UNI2+MLP improved 等）。这些未提交改动已在远程部署运行。
- **远程缺 GHIST 实现文件**：`methods/ghist/{backbone,framework,intialisation,layers,
  model,modules}.py` 在 git 但未 scp 到远程，GHIST 远程暂无法运行。
- 其余代码远程 = 本地工作副本（逐文件 sha1 一致）。
