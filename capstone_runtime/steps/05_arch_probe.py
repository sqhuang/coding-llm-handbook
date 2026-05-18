#!/usr/bin/env python3
"""
Step 05 — 加载 Air-Base + MoE 路由健康度观察。

WHAT: 用 vLLM 启 GLM-4.5-Air-Base FP8，往里灌 1k 条 Python prompts，统计
      expert_load 直方图 + 各 expert 命中次数 + DSA top-k 命中分布。

REQUIRES:
  - hw:       8 × H100 80GB (TP=8, FP8)
  - software: vllm >= 0.6.3 (with FP8 + DSA support), GLM-4.5-Air-Base 权重
  - data:     ~200GB free for the model (HF_HOME)
  - env:      HF_TOKEN, HF_HOME, CUDA_VISIBLE_DEVICES

This file checks env/imports and prints the exact command to run, then exits.
The actual serve + probe must run on the GPU box.
"""
import os
import shutil
import sys


def main() -> int:
    fails = []
    if shutil.which("python") is None and shutil.which("python3") is None:
        fails.append("python not on PATH")
    if not os.environ.get("HF_TOKEN"):
        fails.append("HF_TOKEN env var not set (see env.example)")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        fails.append("CUDA_VISIBLE_DEVICES not set — expected 0,1,...,7")
    try:
        import vllm  # noqa: F401
        ok_vllm = True
    except ImportError:
        fails.append("vllm not installed — `pip install 'vllm>=0.6.3'` (needs CUDA)")
        ok_vllm = False
    try:
        import torch
        if not torch.cuda.is_available():
            fails.append("torch present but cuda unavailable")
        elif torch.cuda.device_count() < 8:
            fails.append(f"need >= 8 GPUs, have {torch.cuda.device_count()}")
    except ImportError:
        fails.append("torch not installed")

    if fails:
        print("preconditions not met:")
        for f in fails:
            print(f"  - {f}")
        print("\nrun on the H100 box once env is ready:")
    print("""
  # 1) start vllm server
  python -m vllm.entrypoints.openai.api_server \\
    --model zai-org/GLM-4.5-Air-Base \\
    --tensor-parallel-size 8 \\
    --quantization fp8 \\
    --max-model-len 32768 \\
    --port 8000 &

  # 2) probe routing (separate shell):
  python lib/probe_expert_load.py \\
    --base-url http://localhost:8000/v1 \\
    --prompts data/clean/python_sample_1k.jsonl \\
    --out logs/expert_load.json
""")
    print("expected: expert_load_var < 0.2 across the 1k prompts.")
    return 0 if not fails and ok_vllm else 2


if __name__ == "__main__":
    raise SystemExit(main())
