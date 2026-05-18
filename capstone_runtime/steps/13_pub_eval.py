#!/usr/bin/env python3
"""
Step 13 — HumanEval+ / MBPP+ / LiveCodeBench 公开评测。

REQUIRES:
  - hw:       1-2 × H100 (vLLM serve)
  - software: evalplus >= 0.3.1, livecodebench
  - env:      LLM_BASE_URL pointing at your model（step 15 output 或 ad-hoc vllm）
"""
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    fails = []
    for tool in ("evalplus.evaluate",):
        if shutil.which(tool) is None:
            try:
                __import__(tool.split(".")[0])
            except ImportError:
                fails.append(f"{tool} not installed (`pip install evalplus`)")
    if not os.environ.get("LLM_BASE_URL"):
        fails.append("LLM_BASE_URL not set — point at your local vllm/sglang endpoint")
    if not os.environ.get("LLM_MODEL"):
        fails.append("LLM_MODEL not set — name of the model in the endpoint")

    if fails:
        print("preconditions not met:")
        for f in fails:
            print(f"  - {f}")
        return 2

    base = os.environ["LLM_BASE_URL"]
    model = os.environ["LLM_MODEL"]
    print(f"""
run these in sequence on the target box (each ~30-60 min):

  evalplus.evaluate \\
    --dataset humaneval \\
    --model "{model}" \\
    --backend openai \\
    --base-url "{base}" \\
    --greedy false --temperature 0.2 --n-samples 20 \\
    --output-dir logs/eval_humaneval+

  evalplus.evaluate \\
    --dataset mbpp \\
    --model "{model}" --backend openai --base-url "{base}" \\
    --greedy false --temperature 0.2 --n-samples 20 \\
    --output-dir logs/eval_mbpp+

  # LiveCodeBench (clone https://github.com/LiveCodeBench/LiveCodeBench)
  python -m lcb_runner.runner.main \\
    --model "{model}" --release_version release_v3 \\
    --scenario codegeneration --evaluate \\
    --start_date 2025-10-01

aggregate to logs/eval_pub.csv with `python lib/aggregate_eval.py` (TODO).
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
