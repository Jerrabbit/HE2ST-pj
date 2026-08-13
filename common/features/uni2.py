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

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class UNI2FeatureExtractor:
    """加载 UNI2 权重并提取图像块特征。

    参数：
        weight_path: pytorch_model.bin 路径
        device: 'cuda' 或 'cpu'
    """

    def __init__(self, weight_path: str, device: str = "cuda"):
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
            feats.append(self.model(x).cpu().numpy())
        return np.concatenate(feats, axis=0)

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
