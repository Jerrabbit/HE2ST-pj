"""把 rep_f/metadata.csv 的 patch_path 改为 /tmp/patches/rep{N}/（复制完成后执行）。"""
import os
import pandas as pd

BASE = os.path.expanduser("~/HE2ST-pj/data")

for rep in (1, 2):
    mp = os.path.join(BASE, f"rep{rep}_f", "metadata.csv")
    m = pd.read_csv(mp)
    m["patch_path"] = ["/tmp/patches/rep%d/%s" % (rep, os.path.basename(p))
                       for p in m["patch_path"]]
    m.to_csv(mp, index=False)
    print(f"rep{rep}_f metadata 已更新（patch_path → /tmp/patches/rep{rep}）")
