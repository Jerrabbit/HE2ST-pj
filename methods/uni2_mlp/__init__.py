"""UNI2+MLP 基线及其 Local-Global 双尺度改进。"""
from .model import FEATURE_DIM, UNI2MLP

__all__ = ["UNI2MLP", "FEATURE_DIM"]


def build_model(num_genes: int = 313, **kwargs):
    """统一模型工厂（scripts/train.py、test.py 调用）。"""
    return UNI2MLP(num_genes=num_genes, **kwargs)
