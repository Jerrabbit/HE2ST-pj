"""统一训练入口：所有方法共用同一训练流程，保证公平比较（课题要求 4）。

用法（远程服务器 myenv1）：
    python scripts/train.py --method uni2_mlp --scenario adjacent --output_dir outputs
"""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HE→ST 统一训练入口")
    parser.add_argument("--method", required=True, help="方法名，对应 methods/ 下文件夹（如 uni2_mlp）")
    parser.add_argument("--config", help="实验配置文件路径（可选）")
    parser.add_argument(
        "--scenario",
        choices=["same_slide", "adjacent", "multi_slide"],
        help="泛化评测情形：同切片左右半 / 相邻切片 / 多切片同癌种",
    )
    parser.add_argument("--output_dir", default="outputs", help="模型与日志输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: 按 args.method 分发到对应方法的训练实现，统一数据加载与评估流程
    raise NotImplementedError("待实现：按 --method 分发到对应方法，统一训练流程")


if __name__ == "__main__":
    main()
