"""把过滤后保留细胞的 patches 复制到 /tmp（nvme 快盘），供 patch 类方法避开 cpfs D-state。

用法（远程）：
    python3 scripts/copy_patches_tmp.py
"""
from __future__ import annotations

import os
import shutil

import pandas as pd

BASE = os.path.expanduser("~/HE2ST-pj/data")
TMP = "/tmp/patches"


def main() -> None:
    for rep in (1, 2):
        meta = pd.read_csv(os.path.join(BASE, f"rep{rep}_f", "metadata.csv"))
        dst_dir = os.path.join(TMP, f"rep{rep}")
        os.makedirs(dst_dir, exist_ok=True)
        n, skipped = 0, 0
        for p in meta["patch_path"]:
            if not os.path.exists(p):
                skipped += 1
                continue
            name = os.path.basename(p)
            dst = os.path.join(dst_dir, name)
            if not os.path.exists(dst):
                shutil.copy2(p, dst)
            n += 1
            if n % 20000 == 0:
                print(f"rep{rep}: {n} copied (skipped {skipped})", flush=True)
        print(f"rep{rep}: done {n} (skipped {skipped})", flush=True)


if __name__ == "__main__":
    main()
