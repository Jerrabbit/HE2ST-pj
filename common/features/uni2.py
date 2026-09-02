"""UNI2 Foundation Model 特征提取封装。

UNI2 = ViT-Giant (patch 14, 224×224)，输出 1536 维 [CLS] token 特征。
架构与官方权重对应关系（已验证，strict 加载 0 缺失）：
    - depth=24, embed_dim=1536, num_heads=24
    - SwiGLU MLP（mlp_ratio=16/3，fc1 8192 / fc2 4096）
    - 8 个 register token（cls/reg 与 pos_embed 分离，no_embed_class）
    - layer_scale init=1e-5，global_pool='token'

远程权重路径：/cpfs01/.../HE2ST/uni2_model/pytorch_model.bin（config.json 同目录）。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class UNI2FeatureExtractor:
    """加载 UNI2 权重并提取图像块特征。

    参数：
        weight_path: pytorch_model.bin 路径
        device: 'cuda' 或 'cpu'
    """

    def __init__(self, weight_path: str, device: str = "cuda", layer_norm: bool = False):
        import timm
        from timm.layers import SwiGLUPacked

        self.device = device
        model = timm.create_model(
            "vit_giant_patch14_224",
            pretrained=False,
            num_classes=0,
            depth=24,
            embed_dim=1536,
            num_heads=24,
            mlp_ratio=16 / 3,
            mlp_layer=SwiGLUPacked,
            init_values=1e-5,
            global_pool="token",
            reg_tokens=8,
            no_embed_class=True,
            dynamic_img_size=True,
        )
        sd = torch.load(weight_path, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"UNI2 权重加载不完整: missing={missing}, unexpected={unexpected}")
        self.model = model.eval().to(device)
        self.n_prefix = int(getattr(model, "num_prefix_tokens", 1))  # CLS + register 前缀 token 数
        # 特征提取 LayerNorm（methods/uni2_mlp/特征提取layernorm参考.txt）：
        # 对 CLS 与 patch token 各自做 LayerNorm(1536, eps=1e-6)。新建层（weight=1/bias=0，
        # 非训练）→ 相当于对每个 token 特征向量做固定标准化（mean=0, std=1）。
        self.ln = nn.LayerNorm(1536, eps=1e-6).to(device) if layer_norm else None

    @torch.no_grad()
    def extract(self, patches: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """提取图像块特征。

        参数：
            patches: (B, H, W, 3) uint8 图像块数组，或 (B, 3, 224, 224) 归一化张量
            batch_size: 批大小
        返回：
            (B, 1536) 特征矩阵
        """
        feats = []
        for i in range(0, len(patches), batch_size):
            batch = patches[i:i + batch_size]
            x = self._preprocess(batch)
            f = self.model(x)
            if self.ln is not None:
                f = self.ln(f)
            feats.append(f.cpu().numpy())
        return np.concatenate(feats, axis=0)

    @torch.no_grad()
    def extract_tokens(self, patches: np.ndarray, batch_size: int = 128) -> np.ndarray:
        """提取全部 token 嵌入，供 Local 分支**单次 forward** 复用。

        与 `extract()`（只取 CLS）不同，本方法返回完整 token 序列
        `(B, T, 1536)`（T = 1 CLS + 8 register + 16×16 patch = 265）。
        patch14 无重叠，中心裁剪 l2=14k 对应的正是 patch 网格中心 k×k 子块，
        可直接从同一次 forward 的结果切出，无需二次 forward。

        参数：
            patches: (B, H, W, 3) uint8 图像块数组，或 (B, 3, 224, 224) 归一化张量
            batch_size: 批大小
        返回：
            (B, T, 1536) token 序列
        """
        feats = []
        for i in range(0, len(patches), batch_size):
            x = self._preprocess(patches[i:i + batch_size])
            f = self.model.forward_features(x)
            if self.ln is not None:
                f = self.ln(f)   # 对每个 token（CLS/reg/patch）独立 LayerNorm
            feats.append(f.cpu().numpy())
        return np.concatenate(feats, axis=0)

    @torch.no_grad()
    def extract_reference(self, patches: np.ndarray, center_ratios: list[float] | None = None,
                          batch_size: int = 128, layer_norm: bool = True):
        """**忠实复刻** img_feature_extractor 参考实现（forward_intermediates 路径）。

        与 extract/extract_tokens 的区别：Local 特征取 **intermediates[-1]**（最后一层中间
        输出的 NCHW 空间 patch 特征图，final LayerNorm 之前）的中心裁剪（参考 center 公式），
        而非 forward_features 的 final-norm 后 token。Global = feature_emb[:,0]（CLS，
        实测与 forward_features 完全一致）。

        参数：
            patches: (B,H,W,3) uint8 或 (B,3,224,224)
            center_ratios: 中心裁剪比例列表（0.25 ↔ l2=56 ↔ 16×16 网格中心 4×4）。
                可给多个 → 一次 forward 复用 intermediates 产出多个 l2 的 Local 特征。
            layer_norm: 对 CLS 与每个 patch token 做 LayerNorm(1536, eps=1e-6)
        返回：
            (g (B,1536), {ratio: l (B,1536)})：g=Global CLS；每个 ratio 一个 Local mean-pool。
        """
        import torch.nn as nn

        ln = nn.LayerNorm(1536, eps=1e-6).to(self.device) if layer_norm else None
        g_all = []
        l_all = {r: [] for r in (center_ratios or [0.25])}
        for i in range(0, len(patches), batch_size):
            x = self._preprocess(patches[i:i + batch_size])
            feature_emb, intermediates = self.model.forward_intermediates(
                x, return_prefix_tokens=False)
            patch_emb = intermediates[-1]                     # (B, C, H, W)
            B, C, Hp, Wp = patch_emb.shape
            center = feature_emb[:, 0] if feature_emb.ndim == 3 else feature_emb  # (B,C) CLS
            if ln is not None:
                center = ln(center)
            for ratio in l_all:
                # 参考的 center 裁剪公式（与 img_feature_extractor 逐行一致）
                cs_h = max(1, round(Hp * ratio)); cs_w = max(1, round(Wp * ratio))
                ch, cw = (Hp - 1) // 2, (Wp - 1) // 2
                hl, hh = cs_h // 2, cs_h - cs_h // 2
                wl, wh = cs_w // 2, cs_w - cs_w // 2
                hs, he = int(max(ch - hl, 0)), int(min(ch + hh, Hp))
                ws, we = int(max(cw - wl, 0)), int(min(cw + wh, Wp))
                pf = patch_emb[:, :, hs:he, ws:we]           # (B, C, h, w)
                pf = pf.permute(0, 2, 3, 1).reshape(B, -1, C)  # (B, h*w, C)
                if ln is not None:
                    pf = ln(pf)
                l_all[ratio].append(pf.mean(1).cpu().numpy())  # (B, C)
            g_all.append(center.cpu().numpy())
        g = np.concatenate(g_all, axis=0)
        loc = {r: np.concatenate(v, axis=0) for r, v in l_all.items()}
        return g, loc

    def center_patch_tokens(self, tokens: np.ndarray, k: int) -> np.ndarray:
        """从全 token 序列提取中心 k×k patch token 块（复用同一次 forward）。

        tokens: (B, T, 1536)。返回 (B, k*k, 1536)，k 为 patch 数（l2 = 14k，k≤8）。
        中心块对齐约定：start = (16-k)//2（偶数 k 时与像素裁剪精确一致）。
        """
        if tokens.shape[1] < self.n_prefix + 256:
            raise ValueError(
                f"token 序列不完整: {tokens.shape[1]} < {self.n_prefix}+256 "
                f"（需 CLS+register+patch 全序列，用 extract_tokens() 提取）")
        pt = tokens[:, self.n_prefix:].reshape(tokens.shape[0], 16, 16, -1)
        start = (16 - k) // 2
        blk = pt[:, start:start + k, start:start + k]  # (B, k, k, 1536)
        return blk.reshape(blk.shape[0], k * k, blk.shape[-1])

    def _preprocess(self, patches: np.ndarray) -> torch.Tensor:
        """图像块 → (B,3,224,224) 归一化张量。"""
        if patches.ndim == 4 and patches.shape[3] == 3:  # (B,H,W,3) uint8
            x = torch.from_numpy(patches).permute(0, 3, 1, 2).float() / 255.0
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
            x = (x - _IMAGENET_MEAN.to(x.device)) / _IMAGENET_STD.to(x.device)
        elif patches.ndim == 4 and patches.shape[1] == 3:  # 已 (B,3,H,W) 假定已归一化
            x = torch.from_numpy(patches).float()
        else:
            raise ValueError(f"不支持的输入形状: {patches.shape}")
        return x.to(self.device)
