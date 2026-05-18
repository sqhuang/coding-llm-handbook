#!/usr/bin/env python3
"""
Step 03 — 数据 pipeline：The Stack v2 Python + 私有 PR。

WHAT: 跑 datatrove 在 The Stack v2 Python subset 上做去重 + 启发式过滤 + 输出 parquet。
私有 PR 抽取走 `examples/phase4/extract_pr_sft.py`（不在本目录）。

REQUIRES:
  - hw:       0-1 GPU（datatrove CPU 为主；MinHash 海量样本时上 GPU 加速版）
  - software: datatrove >= 0.3.0
  - data:     HuggingFace HF_TOKEN（拉 bigcode/the-stack-v2-dedup）
  - env:      HF_TOKEN, HF_HOME（缓存目录），DATA_DIR
  - disk:     ~50GB free for The Stack v2 Python subset cache

THIS FILE: 在没装 datatrove 时给清晰错；装了的话调用主项目根 examples/phase1/run_pipeline.py
（即"主笔记"里抽出来的 222 行参考实现）。
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIPELINE = ROOT / "lib" / "run_pipeline.py"


def main() -> int:
    # 1) tools
    missing = []
    try:
        import datatrove  # noqa: F401
    except ImportError:
        missing.append("datatrove (`pip install 'datatrove>=0.3'`)")
    if not os.environ.get("HF_TOKEN"):
        missing.append("HF_TOKEN env var")
    if missing:
        print("ERROR: prerequisites missing:")
        for m in missing:
            print(f"  - {m}")
        print("\nNote: this step downloads ~50GB. Don't run it on a laptop SSD.")
        return 2

    # 2) point pipeline at our DATA_DIR
    data_dir = Path(os.environ.get("DATA_DIR", ROOT / "data"))
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"data dir : {data_dir}")
    print(f"pipeline : {PIPELINE}")

    if not PIPELINE.exists():
        print(f"ERROR: {PIPELINE} not found. Was lib/ deleted from the bundle?")
        return 3

    # 3) delegate to the pipeline
    print("\n→ running datatrove pipeline (this is the long part)\n")
    rc = subprocess.call([sys.executable, str(PIPELINE)], cwd=str(data_dir))
    if rc != 0:
        print(f"\npipeline exited {rc}")
        return rc

    print(f"\n✓ pipeline finished; check {data_dir}/output/final/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
