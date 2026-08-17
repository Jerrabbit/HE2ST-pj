"""Local-Global 两段调参入口（op1 / op2 sweep，编排 extract + train）。

流程（每种泛化情形分别进行）：
    1. --stage op1（只测 Global）：sweep 边长 l1（从 >224 逐步缩小，小步长多取值），
       每 l1 先提取 Global 特征（训练+验证两切片）再训练 30 epoch 取 best val_PCC，
       绘 PCC–l1 曲线，确定最佳区间，选 best l1。
    2. --stage op2（固定 best l1，Global+Local）：sweep 中心裁剪边长 l2 = 4..8×14。
       Local 特征**复用 op1 同一次 forward 的 patch token**（单次提取产出全部 l2 文件），
       每 l2 训练 30 epoch 取 best val_PCC，选最佳 l2。
    3. 确定 best l1+l2 后，单独跑 50 epoch 完整训练（train.py --variant local_global）。

每个配置：30 epoch（调参），early stop 取 best val_PCC（来自 history.json）。

用法（远程 myenv1）：
    # op1：l1 从 448 缩到 28，步长 28（112 之下追加 84/56/28 三档）
    python scripts/sweep.py --stage op1 --train_rep 1 --valid_rep 2 \
        --train_dir data/rep1 --valid_dir data/rep2 \
        --l1_values 448 420 392 364 336 308 280 252 224 196 168 140 112 84 56 28 \
        --out_dir outputs/sweep_op1
    # op2：固定 best l1，l2 = 4..8 × 14
    python scripts/sweep.py --stage op2 --l1 336 --train_rep 1 --valid_rep 2 \
        --train_dir data/rep1 --valid_dir data/rep2 \
        --l2_values 56 70 84 98 112 --out_dir outputs/sweep_op2
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 项目根目录

REP_DIR = os.path.expanduser("~/HE2ST-pj/data")
DEFAULT_L1_VALUES = list(range(448, 27, -28))  # 448,420,...,112,84,56,28（小步长 28，下探 <112）
DEFAULT_L2_VALUES = [k * 14 for k in (4, 5, 6, 7, 8)]  # 56,70,84,98,112


def _run(cmd: list[str]) -> None:
    print(f"[sweep] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _best_val_pcc(out_dir: str) -> float:
    """从 history.json 取最佳验证 PCC。

    unified harness fit() 的 history 条目存 **PCC**（大写，见 common/benchmark/harness.py）；
    兼容历史脚本可能用的 val_PCC。无有效值返回 nan。
    """
    hist_path = os.path.join(out_dir, "history.json")
    with open(hist_path) as f:
        history = json.load(f)
    pccs = []
    for h in history:
        p = h.get("PCC")
        if p is None:
            p = h.get("val_PCC")
        if p is not None and p == p:  # 去 nan/None
            pccs.append(float(p))
    return max(pccs) if pccs else float("nan")


def _extract_global(args, rep: int, data_dir: str, l1: int) -> None:
    """Global 特征：未存在则提取（rep 决定用哪张切片图，两切片都需特征）。"""
    feat = os.path.join(data_dir, f"X_uni2_g{l1}.npy")
    if not os.path.exists(feat):
        _run([sys.executable, "scripts/extract_local_global.py", "--rep", str(rep),
              "--stage", "global", "--l1", str(l1),
              "--output", feat, "--device", "cuda"])


def _extract_local(args, rep: int, data_dir: str, l1: int, l2_values: list[int]) -> None:
    """Local 特征：单次 forward（token 复用）一次产出全部 l2 文件，未存在才提取。"""
    missing = [l2 for l2 in l2_values
               if not os.path.exists(os.path.join(data_dir, f"X_uni2_l{l2}.npy"))]
    if missing:
        _run([sys.executable, "scripts/extract_local_global.py", "--rep", str(rep),
              "--stage", "local", "--l1", str(l1),
              "--l2_list"] + [str(x) for x in l2_values] + ["--device", "cuda"])


def stage_op1(args) -> tuple[int, float]:
    """只测 Global（op1 放缩中心）：每 l1 提取特征（训练+验证两切片）+ 30ep 训练，绘 PCC–l1 曲线。"""
    l1_values = args.l1_values or DEFAULT_L1_VALUES
    os.makedirs(args.out_dir, exist_ok=True)
    results = {}
    for l1 in l1_values:
        _extract_global(args, args.train_rep, args.train_dir, l1)
        _extract_global(args, args.valid_rep, args.valid_dir, l1)
        out = os.path.join(args.out_dir, f"g{l1}")
        _run([sys.executable, "scripts/train.py", "--method", "uni2_mlp",
              "--variant", "global_only", "--feature_file", f"X_uni2_g{l1}.npy",
              "--train_dir", args.train_dir, "--valid_dir", args.valid_dir,
              "--epochs", str(args.epochs), "--output_dir", out, "--device", "cuda"])
        results[l1] = _best_val_pcc(out)
        print(f"[sweep-op1] l1={l1}: best val_PCC={results[l1]:.4f}", flush=True)

    _save_results(args, "op1", results)
    best_l1 = max(results, key=results.get)
    print(f"[sweep-op1] 最佳 l1 = {best_l1}（PCC {results[best_l1]:.4f}）", flush=True)
    return best_l1, results[best_l1]


def stage_op2(args) -> tuple[int, float]:
    """固定 best l1，Global+Local：单次 forward 提取 Local 特征（两切片）+ 每 l2 30ep 训练。"""
    if not args.l1:
        raise SystemExit("op2 阶段必须 --l1 指定 best l1")
    l2_values = args.l2_values or DEFAULT_L2_VALUES
    # 确保 Global 特征存在（best l1 通常来自 op1 sweep；独立跑 op2 时补）
    _extract_global(args, args.train_rep, args.train_dir, args.l1)
    _extract_global(args, args.valid_rep, args.valid_dir, args.l1)
    # Local：token 复用，两切片各一次 forward 即产出全部 l2 文件
    _extract_local(args, args.train_rep, args.train_dir, args.l1, l2_values)
    _extract_local(args, args.valid_rep, args.valid_dir, args.l1, l2_values)
    os.makedirs(args.out_dir, exist_ok=True)
    results = {}
    for l2 in l2_values:
        out = os.path.join(args.out_dir, f"l{l2}")
        _run([sys.executable, "scripts/train.py", "--method", "uni2_mlp",
              "--variant", "local_global",
              "--feature_file", f"X_uni2_g{args.l1}.npy,X_uni2_l{l2}.npy",
              "--train_dir", args.train_dir, "--valid_dir", args.valid_dir,
              "--epochs", str(args.epochs), "--output_dir", out, "--device", "cuda"])
        results[l2] = _best_val_pcc(out)
        print(f"[sweep-op2] l2={l2} (l1={args.l1}): best val_PCC={results[l2]:.4f}", flush=True)

    _save_results(args, "op2", results)
    best_l2 = max(results, key=results.get)
    print(f"[sweep-op2] 最佳 l2 = {best_l2}（PCC {results[best_l2]:.4f}）", flush=True)
    return best_l2, results[best_l2]


def _save_results(args, stage: str, results: dict) -> None:
    """保存结果 CSV + 绘制 PCC–边长曲线（平滑趋势图）。"""
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = sorted(results)
    ys = [results[x] for x in xs]
    csv_path = os.path.join(args.out_dir, f"{stage}_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param", "best_val_pcc"])
        w.writerows(zip(xs, ys))
    print(f"[sweep] 结果已保存 {csv_path}", flush=True)

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, "o-", label=f"{stage} best val_PCC")
    plt.xlabel("l1 (块边长)" if stage == "op1" else "l2 (裁剪边长)")
    plt.ylabel("best val_PCC (30ep)")
    plt.title(f"{stage} sweep (PCC vs 边长)")
    plt.grid(True, alpha=0.3)
    png = os.path.join(args.out_dir, f"{stage}_curve.png")
    plt.savefig(png, dpi=120)
    print(f"[sweep] 曲线图已保存 {png}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local-Global 两段调参 sweep")
    p.add_argument("--stage", choices=["op1", "op2"], required=True)
    p.add_argument("--train_rep", type=int, choices=[1, 2], default=1,
                   help="训练切片编号（特征提取读该切片图）")
    p.add_argument("--valid_rep", type=int, choices=[1, 2], default=2,
                   help="验证切片编号（特征提取读该切片图）")
    p.add_argument("--train_dir", required=True, help="训练集数据目录")
    p.add_argument("--valid_dir", required=True, help="验证集数据目录")
    p.add_argument("--l1", type=int, help="op2 阶段固定的 best l1")
    p.add_argument("--l1_values", nargs="+", type=int, help="op1 sweep 边长序列")
    p.add_argument("--l2_values", nargs="+", type=int, help="op2 裁剪边长序列（14 的倍数）")
    p.add_argument("--epochs", type=int, default=30, help="调参时每个配置的 epoch 数")
    p.add_argument("--out_dir", default="outputs/sweep", help="结果输出目录")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.stage == "op1":
        stage_op1(args)
    else:
        stage_op2(args)
