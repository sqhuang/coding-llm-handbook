#!/usr/bin/env python3
"""
Step 14 — 内部 SWE-Bench v0 跑 4 ckpt (base / midtrain / sft / rl) × 10 题 × 3 samples.

REQUIRES:
  - hw:       1-2 × H100 + docker
  - software: lib/mini_agent.py + internal_bench_v0/ (from step 02)
  - env:      LLM_BASE_URL（每跑一个 ckpt 切换一次）
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "internal_bench_v0"


def main() -> int:
    if not BENCH.exists():
        print(f"ERROR: {BENCH}/ missing — run step 02 first.")
        return 2
    if not (BENCH / "instances").iterdir():
        print(f"ERROR: {BENCH}/instances/ empty — collect real PRs first.")
        return 2
    if not os.environ.get("LLM_BASE_URL"):
        print("ERROR: LLM_BASE_URL not set; bring up step 15 endpoint first.")
        return 2

    print(f"""
loop:
  for ckpt in (base, midtrain, sft, rl):
    1. point step 15 deploy at this ckpt
    2. for each instance in {BENCH.relative_to(ROOT)}/instances/:
       for sample in (0, 1, 2):
         python lib/mini_agent.py --task <instance.json> --max-turn 25 \\
             --output logs/internal_eval/<ckpt>/<instance>__<sample>.json
    3. ./run.sh <instance> <patch> for each output; record exit code
    4. tally resolved_rate = passed / total, majority-vote per instance

output: logs/eval_internal.csv with columns
  ckpt, instance_id, sample, resolved (0/1), tokens, wall_sec
""")
    print("aggregate: `python lib/aggregate_internal.py` (TODO)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
