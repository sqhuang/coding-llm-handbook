#!/usr/bin/env python3
"""
Step 16 — LoRA 合并 + prefix cache 调优 + bench.

REQUIRES:
  - hw:       same 8 × H100 as step 15
  - software: sglang bench_serving 或 vllm benchmark
  - env:      LLM_BASE_URL pointing at step 15 endpoint
"""
import os
import sys


def main() -> int:
    if not os.environ.get("LLM_BASE_URL"):
        print("ERROR: LLM_BASE_URL not set (step 15 must be running)")
        return 2

    print(f"""
benchmark grid (4 variants, each ~10 min):

  for variant in base +prefix-cache +chunked-prefill +speculative; do
    python -m sglang.bench_serving \\
        --backend sglang --host 0.0.0.0 --port 30000 \\
        --dataset-name sharegpt --num-prompt 1000 \\
        --output-file logs/bench_${{variant}}.json
  done

aggregate to logs/deploy_report.md with:
  - TTFT (p50/p95)
  - throughput tok/s
  - prefix hit rate
  - VRAM peak

decision: pick the variant with TTFT < 500ms @ p95 AND max throughput.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
