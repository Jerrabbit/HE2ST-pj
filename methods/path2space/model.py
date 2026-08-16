"""Path2Space：官方冻结模型的 22×7 MLP 集成（按官方 ensemble.py 复刻）。

官方代码：D:\\hest_data\\codes\\Path2Space（model_mlp.py / ensemble.py）
架构（必须与官方 checkpoint 完全一致）：
    MLP_regression_relu_two：Linear(768→768) → ReLU → Dropout(0.2)
        → Linear(768→14068) → ReLU
    n_hiddens == n_inputs == 768（官方 CTransPath 特征维度）
    集成：22（ik-fold）× 7（il-fold）= 154 个 MLP 的均值。

适配本仓库：
    - 模型为**推理专用**（冻结），官方权重位于远程服务器，需提供
      ensemble_dir（含 result_{ik}_{il}_0/model_trained.pth）。
    - 输出 14068 固定基因，通过 genes.txt 映射到本仓库公共基因子集。
    - 输出尺度为官方训练的 log1p 表达；Path2SpaceModel 统一转成 raw counts
      语义（gene_norm='none' 评估）。
"""
from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn

__all__ = ["MLP_regression_relu_two", "MLPEnsemble", "Path2SpaceModel"]


class MLP_regression_relu_two(nn.Module):
    """两层 ReLU MLP（官方 model_mlp.py，checkpoint 兼容，勿改结构）。"""

    def __init__(self, n_inputs, n_hiddens, n_outputs, dropout=0.2,
                 bias_init=None):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Linear(n_inputs, n_hiddens),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.layer1 = nn.Sequential(
            nn.Linear(n_hiddens, n_outputs),
            nn.ReLU(),
        )
        if bias_init is not None:
            with torch.no_grad():
                self.layer1[0].bias = nn.Parameter(bias_init)

    def forward(self, x):
        return self.layer1(self.layer0(x))


class MLPEnsemble:
    """22×7 MLP 集成推理（官方 ensemble.py，先 il 内均值再 ik 外均值）。"""

    def __init__(self, ensemble_dir, n_inputs=768, n_genes=14068,
                 device="cpu", n_ik_folds=22, n_il_folds=7):
        self.ensemble_dir = str(ensemble_dir)
        self.n_inputs = int(n_inputs)
        self.n_genes = int(n_genes)
        self.n_ik = int(n_ik_folds)
        self.n_il = int(n_il_folds)
        self.device = torch.device(device)
        self.models: dict[tuple[int, int], MLP_regression_relu_two] = {}
        self._load_all()

    def _load_all(self):
        missing = []
        for ik in range(self.n_ik):
            for il in range(self.n_il):
                ckpt = os.path.join(
                    self.ensemble_dir, f"result_{ik}_{il}_0", "model_trained.pth"
                )
                if not os.path.exists(ckpt):
                    missing.append(ckpt)
                    continue
                model = MLP_regression_relu_two(
                    n_inputs=self.n_inputs, n_hiddens=self.n_inputs,
                    n_outputs=self.n_genes, dropout=0.2,
                ).to(self.device)
                model.load_state_dict(torch.load(ckpt, map_location=self.device))
                model.eval()
                self.models[(ik, il)] = model
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} 个 MLP checkpoint 缺失，第一个: {missing[0]}"
            )

    @torch.no_grad()
    def predict(self, features: np.ndarray) -> np.ndarray:
        """(N, 768) → (N, 14068)，先 il 均值再 ik 均值（官方顺序）。"""
        if features.ndim != 2 or features.shape[1] != self.n_inputs:
            raise ValueError(
                f"features 需为 (N, {self.n_inputs})，got {features.shape}"
            )
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        preds_ik = np.zeros((self.n_ik, x.size(0), self.n_genes), dtype=np.float64)
        for ik in range(self.n_ik):
            preds_il = np.zeros((self.n_il, x.size(0), self.n_genes),
                                dtype=np.float64)
            for il in range(self.n_il):
                preds_il[il] = self.models[(ik, il)](x).cpu().numpy()
            preds_ik[ik] = preds_il.mean(axis=0)
        return preds_ik.mean(axis=0)


class Path2SpaceModel(nn.Module):
    """把冻结的 Path2Space 集成包装为统一 harness 模型（input_type='feature'）。

    行为：
        - feature_file = 'X_ctranspath.npy'（需先由预处理脚本生成）
        - forward：768 维 CTransPath 特征 → 14068 基因（log1p 尺度）
          → 经 out_indices 取公共基因子集 → 若 output_is_log1p 则 expm1
          转成 raw counts 语义（评估用 gene_norm='none'）。
    注意：本模型无可训练参数（冻结），不参与 fit；用 scripts/test_path2space.py 评估。
    """

    input_type = "feature"
    feature_file = "X_ctranspath.npy"

    def __init__(
        self,
        num_genes: int,
        ensemble_dir: str,
        genes_txt: str | None = None,
        gene_names: list[str] | None = None,
        n_inputs: int = 768,
        n_genes_all: int = 14068,
        output_is_log1p: bool = True,
        device: str | torch.device | None = None,
        n_ik_folds: int = 22,
        n_il_folds: int = 7,
    ):
        super().__init__()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.num_genes = int(num_genes)
        self.ensemble = MLPEnsemble(
            ensemble_dir, n_inputs=n_inputs, n_genes=n_genes_all,
            device=self.device, n_ik_folds=n_ik_folds, n_il_folds=n_il_folds,
        )
        # 公共基因 → 14068 输出基因的索引映射（部分覆盖：仅含可预测基因）
        self.out_indices, self.out_cols = _resolve_out_indices(
            genes_txt, gene_names, n_genes_all
        )
        if len(self.out_indices) < num_genes:
            print(f"[Path2Space] 警告: {num_genes} 公共基因中仅 {len(self.out_indices)} 个"
                  f"在官方 14068 基因表中，其余 {num_genes - len(self.out_indices)} "
                  f"个无法预测（输出填 0，不贡献指标）", flush=True)
        self.output_is_log1p = output_is_log1p

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """CTransPath 特征 (B, 768) → (B, G) raw counts 语义预测。

        部分覆盖：未覆盖基因输出填 0（constant → 该基因 PCC 为 NaN 被跳过，
        对 Top-k 无贡献），仅在可预测的公共基因子集上体现 Path2Space 性能。
        """
        feat = x.cpu().numpy() if torch.is_tensor(x) else x
        pred = self.ensemble.predict(feat)                  # (B, 14068) log1p
        pred = pred[:, self.out_indices]                     # (B, covered)
        if self.output_is_log1p:
            pred = np.expm1(np.clip(pred, -30.0, 30.0))      # → raw counts
        B = feat.shape[0] if isinstance(feat, np.ndarray) else feat.size(0)
        full = np.zeros((B, self.num_genes), dtype=np.float32)
        full[:, self.out_cols] = pred                        # 散射回 313 列位置
        return torch.as_tensor(full, dtype=torch.float32, device=x.device)


def _resolve_out_indices(
    genes_txt: str | None, gene_names: list[str] | None, n_genes_all: int
) -> np.ndarray:
    """把公共基因名映射到官方 genes.txt 的输出位置。

    必须提供其一：genes_txt（官方基因列表文件）或 gene_names（本仓库基因名）。
    """
    if genes_txt is None and gene_names is None:
        raise ValueError("需提供 genes_txt 或 gene_names 之一")
    if genes_txt is not None:
        with open(genes_txt) as f:
            all_genes = [ln.strip() for ln in f if ln.strip()]
    else:
        all_genes = gene_names
        n_genes_all = len(all_genes)
    if gene_names is not None:
        common = gene_names
    else:
        raise ValueError("需提供 gene_names（本仓库公共基因名）")
    name_to_idx = {g: i for i, g in enumerate(all_genes)}
    # 部分覆盖：只保留在官方基因表中的公共基因（缺失的由 Path2SpaceModel 填 0）。
    # 返回双重映射：
    #   out_indices: 每个可预测公共基因在官方 14068 输出中的下标（用于从 pred 选列）
    #   out_cols:    每个可预测公共基因在 313 公共基因 full 输出中的列位置（用于散射）
    covered = [(col, name_to_idx[g]) for col, g in enumerate(common) if g in name_to_idx]
    if not covered:
        raise ValueError("无任何公共基因在 Path2Space 基因表中")
    out_indices = np.array([i for _, i in covered], dtype=np.int64)
    out_cols = np.array([c for c, _ in covered], dtype=np.int64)
    return out_indices, out_cols
