"""统一测试/评估入口：所有方法共用同一评估流程（PCC / SPCC / Top-k / AUROC）。

用法（远程服务器 myenv1）：
    python scripts/test.py --method uni2_mlp --ckpt outputs/uni2_mlp/best.pt --scenario adjacent
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根目录


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HE→ST 统一测试/评估入口")
    parser.add_argument("--method", required=True, help="方法名，对应 methods/ 下文件夹")
    parser.add_argument("--ckpt", required=True, help="模型权重路径")
    parser.add_argument(
        "--scenario",
        choices=["same_slide", "adjacent", "multi_slide"],
        help="泛化评测情形",
    )
    parser.add_argument("--output_dir", default="outputs", help="评估结果输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: 加载模型 → 推理 → common/eval/metrics.py 计算全部指标 → 保存结果
    raise NotImplementedError("待实现：统一评估流程，报告 PCC / SPCC / Top-k / AUROC")


if __name__ == "__main__":
    main()
