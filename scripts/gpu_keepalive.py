"""GPU 保活脚本：持续做真实矩阵运算，防止云实例因 GPU 利用率过低被自动回收。

启动（远程 gpu-server，后台运行）：
    nohup python scripts/gpu_keepalive.py > logs/gpu_keepalive.log 2>&1 &

停止（正式开始跑模型、需要 GPU 时）：
    pkill -f gpu_keepalive.py

参数：
    --gpu        GPU 编号（默认 0）
    --mem_gb     目标占用显存（默认 30GB，自动避开显存不足）
    --interval   每轮连续计算的时长（秒），期间 GPU 保持满载
"""
from __future__ import annotations

import argparse
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU 保活：持续占用 GPU")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--mem_gb", type=float, default=30.0)
    parser.add_argument("--interval", type=float, default=300.0)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda")

    # 每个 float32 矩阵 (A/B/C) 各占可用显存的 1/3，避免 A@B 结果 C 溢出
    props = torch.cuda.get_device_properties(device)
    free_gb = props.total_memory / 1e9
    per_tensor_gb = min(args.mem_gb / 3.0, max(0.5, (free_gb - 2.0) / 3.0))
    n = int((per_tensor_gb * 1e9 / 4) ** 0.5)
    A = torch.randn(n, n, device=device)
    B = torch.randn(n, n, device=device)
    print(f"[keepalive] GPU {args.gpu} n={n} (per-tensor ~{per_tensor_gb:.1f}GB), "
          f"matmul loop interval={args.interval}s", flush=True)

    step = 0
    while True:
        step += 1
        start = time.time()
        while time.time() - start < args.interval:
            C = A @ B
            torch.cuda.synchronize()
        if step % 12 == 0:
            util = torch.cuda.utilization(args.gpu)
            print(f"[keepalive] alive, util={util}%, free="
                  f"{torch.cuda.mem_get_info(args.gpu)[0] / 1e9:.1f}GB", flush=True)


if __name__ == "__main__":
    main()
