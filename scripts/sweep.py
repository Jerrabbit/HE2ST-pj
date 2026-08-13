"""Local-Global 调参入口（课题要求中的 op1 / op2 sweep）。

流程：
    1. --stage op1：只测 op1（放缩中心），sweep 边长 c1（从 >224 逐步缩小），
       输出 PCC–c1 曲线，总结趋势确定最佳区间，选 best c1。
    2. --stage op2：固定 best c1，sweep 中心裁剪边长 c2（须为 14 的倍数），
       选最佳 PCC 对应 c2。

用法（远程服务器 myenv1）：
    python scripts/sweep.py --stage op1 --c1_values 448 392 336 280 224 --scenario adjacent
    python scripts/sweep.py --stage op2 --c1 336 --c2_values 224 210 196 182 168 154 140 --scenario adjacent
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根目录


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local-Global 双尺度调参 sweep")
    parser.add_argument("--stage", choices=["op1", "op2"], required=True, help="调参阶段")
    parser.add_argument("--c1", type=int, help="op2 阶段需固定的 c1（best c1）")
    parser.add_argument("--c1_values", nargs="+", type=int, help="op1 sweep 的边长序列（≥224）")
    parser.add_argument("--c2_values", nargs="+", type=int, help="op2 sweep 的裁剪边长序列（14 的倍数）")
    parser.add_argument(
        "--scenario",
        choices=["same_slide", "adjacent", "multi_slide"],
        help="泛化评测情形",
    )
    parser.add_argument("--output_dir", default="outputs", help="结果输出目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: 逐边长训练/测试并记录 PCC，输出 PCC–边长曲线图与最佳边长
    raise NotImplementedError("待实现：逐边长训练/测试并记录 PCC，绘制 PCC–边长曲线")


if __name__ == "__main__":
    main()
