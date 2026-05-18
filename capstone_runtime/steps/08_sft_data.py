#!/usr/bin/env python3
"""
Step 08 — SFT 数据合成（OSS-Instruct 30k + Issue-PR 5k + agent 轨迹 1k）。

WHAT: 用 GLM-5.1 API（或本地 Air-Instruct）按 OSS-Instruct 模板把 seed 代码段
      改写成 (instruction, response) 对，并和 step 03/10 的真实样本混合。

REQUIRES:
  - hw:       0-1 GPU (API 调用或本地小模型 inference)
  - software: openai-python (or requests)
  - data:     data/clean/python_seeds.jsonl (200-1000 段 seed 代码)
  - env:      GLM_API_KEY 或 LLM_BASE_URL（本地）

THIS FILE: 校验环境 + 打印合成命令；真正的多线程合成器在 lib/oss_instruct_gen.py（TODO）。
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


OSS_INSTRUCT_PROMPT = """\
You will be given a random Python code snippet. Your job is to write a programming
problem that this snippet would be the answer to, then re-state the snippet as the
answer. Output a single JSON object with keys "instruction" and "response".

Constraints:
- The instruction MUST NOT mention any identifier from the snippet that would leak the answer.
- The instruction should be a self-contained problem statement (1-3 sentences).
- Response should be a clean Python solution, possibly improved from the snippet.

Snippet:
{snippet}
"""


def main() -> int:
    seeds = ROOT / "data" / "clean" / "python_seeds.jsonl"
    out = ROOT / "data" / "sft_oss_instruct.jsonl"

    if not (os.environ.get("GLM_API_KEY") or os.environ.get("LLM_BASE_URL")):
        print("ERROR: need GLM_API_KEY (cloud) or LLM_BASE_URL (local) for the teacher.")
        return 2
    if not seeds.exists():
        print(f"ERROR: {seeds} missing. Generate via:")
        print(f"  python lib/sample_seeds.py --in data/clean/ --out {seeds} --n 200")
        return 2

    print(f"seeds  : {seeds}")
    print(f"target : 30000 instructions → {out}")
    print(f"teacher: GLM-5.1 API")
    print()
    print("prompt template:")
    print("-" * 60)
    print(OSS_INSTRUCT_PROMPT)
    print("-" * 60)
    print()
    print("TODO: implement lib/oss_instruct_gen.py — multi-thread call_model + quality filter.")
    print("Skeleton: for each seed, sample 150 candidates @ temp=1.0, keep best 1 by:")
    print("  - instruction does not mention any identifier from snippet")
    print("  - response passes ast.parse")
    print("  - instruction length 30-300 chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
