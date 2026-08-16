"""BLEEP：双模态对比学习（CLIP 风格），按官方实现复刻。

官方代码：D:\\hest_data\\codes\\BLEEP（models.py / modules.py / dataset.py）
架构：
    图像编码器 timm resnet50（avg-pool，无分类头，2048 维）
        → 共享投影头 ProjectionHead → 256 维联合嵌入
    表达向量（N 基因，log1p 库归一化）直接用
        → 同款投影头 → 256 维联合嵌入
    损失：批内软目标交叉熵（soft-target cross-entropy，temperature=1.0）
    推理：余弦相似度 k-NN 检索——query 图像嵌入对参考集表达嵌入做 top-k 加权平均
          （官方 weighted_average，k=50，权重 exp(-(d² - min_d² + 1))）

适配本仓库：
    - input_type='patch'，用 HESTDataset（img_size=224 resize，官方为 224 直接裁切）
    - 表达输入 = gene_norm='log1p_norm_total'（官方 normalize_total + log1p）
    - 修复官方 dataset.py 两个 bug（未设 is_train、重复 permute）
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECTION_DIM = 256
IMAGE_EMBED_DIM = 2048  # resnet50 avg-pool


class ProjectionHead(nn.Module):
    """共享投影头：Linear(d)→GELU→Linear(256)→Dropout(0.1)→残差→LayerNorm(256)。

    官方 modules.py ProjectionHead。
    """

    def __init__(self, embedding_dim: int, projection_dim: int = PROJECTION_DIM,
                 dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(projection_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 官方 modules.py:178-184：先投影到 projection_dim，残差加在投影空间
        projected = self.linear1(x)
        x = self.linear2(self.gelu(projected))
        x = self.dropout(x)
        x = x + projected
        return self.norm(x)


class ImageEncoder(nn.Module):
    """timm resnet50，avg-pool、无分类头，输出 2048 维。

    权重来源：timm 默认从 HuggingFace 下载（远程无网不可用）。传
    `pretrained_weights`（torchvision resnet50-0676ba61.pth 路径）时改为
    加载本地权重文件（strict=False，架构同为标准 ResNet50，只忽略 fc 层）。
    """

    def __init__(self, model_name: str = "resnet50", pretrained: bool = True,
                 pretrained_weights: str | None = None):
        super().__init__()
        import timm
        # 提供 pretrained_weights 时跳过 timm 内置下载（远程无网无法访问 HF），
        # 用 pretrained=False 建模型后手动加载本地权重文件。
        load_local = pretrained_weights is not None
        self.model = timm.create_model(
            model_name, pretrained=(pretrained and not load_local),
            num_classes=0, global_pool="avg",
        )
        if load_local:
            sd = torch.load(pretrained_weights, map_location="cpu")
            missing, unexpected = self.model.load_state_dict(sd, strict=False)
            if len(unexpected) > 2:
                raise ValueError(
                    f"pretrained_weights 键不匹配: missing={len(missing)} "
                    f"unexpected={len(unexpected)}"
                )
        self.out_dim = getattr(self.model, "num_features", IMAGE_EMBED_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def clip_soft_target_loss(spot_emb: torch.Tensor, image_emb: torch.Tensor,
                          temperature: float = 1.0) -> torch.Tensor:
    """批内软目标交叉熵（官方 models.py:34-43）。

    spot_emb / image_emb 为 L2 归一化的 256 维联合嵌入，(B, 256)。
    """
    logits = (spot_emb @ image_emb.T) / temperature                  # B x B
    images_similarity = image_emb @ image_emb.T
    spots_similarity = spot_emb @ spot_emb.T
    targets = F.softmax(((images_similarity + spots_similarity) / 2) / temperature, dim=-1)
    spots_loss = (-targets * F.log_softmax(logits, dim=-1)).sum(1)
    images_loss = (-targets.T * F.log_softmax(logits.T, dim=-1)).sum(1)
    return ((spots_loss + images_loss) / 2).mean()


class BLEEP(nn.Module):
    """BLEEP 模型 + 参考集检索推理。

    参数：
        num_genes: 表达向量维度（我们的公共基因数）
        image_backbone: timm 图像主干名
        pretrained: 是否用 ImageNet 预训练
        projection_dim: 联合嵌入维度（官方 256）
        top_k: 检索近邻数（官方 50）
        ref_topk_weighted: 用官方加权平均聚合还是直接 top-k 均值
    """

    input_type = "patch"

    def __init__(
        self,
        num_genes: int,
        image_backbone: str = "resnet50",
        pretrained: bool = True,
        pretrained_weights: str | None = None,
        projection_dim: int = PROJECTION_DIM,
        top_k: int = 50,
        ref_topk_weighted: bool = True,
        finetune: bool = True,
    ):
        super().__init__()
        self.num_genes = num_genes
        self.projection_dim = projection_dim
        self.top_k = top_k
        self.ref_topk_weighted = ref_topk_weighted

        self.image_encoder = ImageEncoder(image_backbone, pretrained, pretrained_weights)
        if not finetune:
            # 冻结 resnet50 编码器（只训投影头），与冻结特征的 UNI2+MLP 公平对比
            for p in self.image_encoder.parameters():
                p.requires_grad = False
        self.image_projection = ProjectionHead(self.image_encoder.out_dim, projection_dim)
        self.spot_projection = ProjectionHead(num_genes, projection_dim)
        # 参考集（训练后填充）：{'spot_emb': (N,256) 已 L2 归一化, 'spot_expr': (N,G)}
        self.reference = None

    # ---------- 训练侧 ----------
    def image_embed(self, x: torch.Tensor) -> torch.Tensor:
        """patch → (B,256) 图像嵌入（layer-norm 后，未 L2 归一化，与官方一致）。"""
        f = self.image_encoder(x)
        return self.image_projection(f)

    def spot_embed(self, expr: torch.Tensor) -> torch.Tensor:
        """表达向量 → (B,256) 表达嵌入（未 L2 归一化）。"""
        return self.spot_projection(expr)

    @torch.no_grad()
    def build_reference(self, expr: torch.Tensor) -> None:
        """从参考集表达矩阵构建检索库（存入 self.reference，嵌入做 L2 归一化）。"""
        spot_emb = self.spot_embed(expr.to(next(self.parameters()).device))
        self.reference = {
            "spot_emb": F.normalize(spot_emb, dim=-1),  # (N, 256)
            "spot_expr": expr.float(),                  # (N, G)
        }

    # ---------- 推理侧（检索） ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """查询 patch → (B, G) 预测表达（参考集检索聚合）。

        参考 self.reference 需先 build_reference() 填充。
        检索语义同官方 find_matches + weighted_average：
        query 与参考嵌入均 L2 归一化，相似度=点积；
        加权权重 w = exp(-(d² - min_d² + 1))，d² 为欧氏平方距离 = 2 - 2·sim。
        """
        if self.reference is None:
            raise RuntimeError("未构建参考集：请先 build_reference()")
        q = F.normalize(self.image_embed(x), dim=-1)                   # (B,256)
        sim = q @ self.reference["spot_emb"].to(x.device).T            # (B,N)
        top_k = min(self.top_k, sim.size(1))
        vals, idx = sim.topk(top_k, dim=-1)                            # (B,k)
        ref_expr = self.reference["spot_expr"].to(x.device)
        if self.ref_topk_weighted:
            # 官方 weighted_average：d² = ||s-q||² = 2-2·sim（均已归一化）
            dist_sq = 2.0 - 2.0 * vals
            w = torch.exp(-(dist_sq - dist_sq.min(dim=-1, keepdim=True).values + 1.0))
            w = w / w.sum(dim=-1, keepdim=True)
        else:
            w = torch.full_like(vals, 1.0 / top_k)
        # (B,k) @ (B,k,G) → (B,G)
        pred = (w.unsqueeze(-1) * ref_expr[idx]).sum(dim=1)
        return pred
