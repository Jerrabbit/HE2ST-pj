"""SQUALL：冻结多模态基础模型特征 + 可训练头（per-cell 适配）。

- variant='mlp'（默认）：冻结编码器 mean-pool 1024 特征（X_squall.npy）+ 统一 MLPHead。
- variant='decoder'：冻结编码器 token 嵌入（X_squall_tokens.npy，196×1024）
  + **训练官方 TransformerDecoder 头**（SQUALLDecoderHead）——与统一 MLP 公平对齐。
- variant='frozen_decoder'：官方冻结解码器直接推理（用 test_squall_decoder.py 评估，
  不训练）。

训练走标准 harness fit（input_type='feature'），无自定义 train_function。
"""
from __future__ import annotations

from .model import SQUALLDecoderHead, SQUALLModel

__all__ = ["SQUALLModel", "SQUALLDecoderHead", "build_model"]


def build_model(num_genes: int = 313, variant: str = "mlp", **kwargs):
    if variant == "decoder":
        return SQUALLDecoderHead(num_genes=num_genes, **kwargs)
    return SQUALLModel(num_genes=num_genes, **kwargs)
