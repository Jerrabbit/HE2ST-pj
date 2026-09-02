"""忠实特征（intermediates[-1]）粗调参 sweep：op1(Global l1) + op2(Local l2)。

用**标准 MLPHead**（variant=global_only/local_global）+ log1p_zscore，
特征由 scripts/extract_faithful.py 提取（_ref 命名，LN 后）。
验证：忠实特征是否在非 112/56 的 l1/l2 有更优峰，能否超旧 LG 标准版 0.3712。

用法（远程 myenv1）：
    python3 scripts/sweep_faithful.py --l1_values 448 224 140 112 84 56 28 \
        --l2_values 28 42 56 70 84 98 --out_dir outputs/sweep_f_op1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GN = "log1p_zscore"
EPOCHS = 30


def _run(cmd):
    print(f"[sweep-f] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def _best_val_pcc(out_dir):
    hist = json.load(open(os.path.join(out_dir, "history.json")))
    pccs = [h["PCC"] for h in hist if "PCC" in h and h["PCC"] == h["PCC"]]
    return max(pccs) if pccs else float("nan")


def _ensure_global(rep, l1):
    """提 l1 的忠实 Global（若未提）。附赠 l2=28 的 local，避免空 l2_list。"""
    g = f"data/rep{rep}/X_uni2_g{l1}_ref.npy"
    if not os.path.exists(g):
        _run([sys.executable, "scripts/extract_faithful.py", "--rep", str(rep),
              "--data_dir", f"data/rep{rep}", "--l1", str(l1),
              "--l2_list", "28", "--device", "cuda"])


def _ensure_locals(rep, l1, l2_values):
    """固定 l1 提取全部 l2 的忠实 Local（单 forward 复用；顺带重提 Global 保证一致）。"""
    g = f"data/rep{rep}/X_uni2_g{l1}_ref.npy"
    l56 = f"data/rep{rep}/X_uni2_l56_ref.npy"
    if not os.path.exists(g) or not os.path.exists(l56):
        _run([sys.executable, "scripts/extract_faithful.py", "--rep", str(rep),
              "--data_dir", f"data/rep{rep}", "--l1", str(l1),
              "--l2_list"] + [str(x) for x in l2_values] + ["--device", "cuda"])


def stage_op1(l1_values, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    res = {}
    for l1 in l1_values:
        _ensure_global(1, l1)
        _ensure_global(2, l1)
        d = os.path.join(out_dir, f"g{l1}")
        _run([sys.executable, "scripts/train.py", "--method", "uni2_mlp",
              "--variant", "global_only", "--feature_file", f"X_uni2_g{l1}_ref.npy",
              "--train_dir", "data/rep1", "--valid_dir", "data/rep2",
              "--gene_norm", GN, "--epochs", str(EPOCHS), "--patience", "10",
              "--lr", "1e-3", "--batch_size", "2048",
              "--output_dir", d, "--device", "cuda"])
        res[l1] = _best_val_pcc(d)
        print(f"[sweep-f op1] l1={l1}: {res[l1]:.4f}", flush=True)
    _save(out_dir, "op1", res)
    best = max(res, key=res.get)
    print(f"[sweep-f op1] best l1={best} ({res[best]:.4f})", flush=True)
    return best, res[best]


def stage_op2(l1, l2_values, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    _ensure_locals(1, l1, l2_values)
    _ensure_locals(2, l1, l2_values)
    res = {}
    for l2 in l2_values:
        d = os.path.join(out_dir, f"l{l2}")
        _run([sys.executable, "scripts/train.py", "--method", "uni2_mlp",
              "--variant", "local_global",
              "--feature_file", f"X_uni2_g{l1}_ref.npy,X_uni2_l{l2}_ref.npy",
              "--train_dir", "data/rep1", "--valid_dir", "data/rep2",
              "--gene_norm", GN, "--epochs", str(EPOCHS), "--patience", "10",
              "--lr", "1e-3", "--batch_size", "2048",
              "--output_dir", d, "--device", "cuda"])
        res[l2] = _best_val_pcc(d)
        print(f"[sweep-f op2] l2={l2}: {res[l2]:.4f}", flush=True)
    _save(out_dir, "op2", res)
    best = max(res, key=res.get)
    print(f"[sweep-f op2] best l2={best} ({res[best]:.4f})", flush=True)
    return best, res[best]


def _save(out_dir, stage, res):
    with open(os.path.join(out_dir, f"{stage}_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param", "best_val_pcc"])
        w.writerows(sorted(res.items()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--l1_values", nargs="+", type=int, default=[448, 224, 140, 112, 84, 56, 28])
    p.add_argument("--l2_values", nargs="+", type=int, default=[28, 42, 56, 70, 84, 98])
    p.add_argument("--out_dir", default="outputs/sweep_f")
    args = p.parse_args()
    op1_dir = os.path.join(args.out_dir, "op1")
    best_l1, _ = stage_op1(args.l1_values, op1_dir)
    op2_dir = os.path.join(args.out_dir, "op2")
    best_l2, _ = stage_op2(best_l1, args.l2_values, op2_dir)
    print(f"[sweep-f] DONE: best l1={best_l1}, best l2={best_l2}")
