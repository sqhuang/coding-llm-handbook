#!/usr/bin/env python3
"""
Step 18 — mini_agent + RAG，跑 3 个真实任务的 demo。

REQUIRES:
  - hw:       step 15 endpoint up, step 17 Qdrant up
  - software: lib/mini_agent.py + lib/rag_search.py (TODO)
  - data:     一个真实 repo（可读可写）
  - env:      LLM_BASE_URL, QDRANT_URL
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MINI = ROOT / "lib" / "mini_agent.py"


def main() -> int:
    fails = []
    if not MINI.exists():
        fails.append(f"{MINI} missing")
    if not os.environ.get("LLM_BASE_URL"):
        fails.append("LLM_BASE_URL not set (step 15 endpoint)")
    if not os.environ.get("QDRANT_URL"):
        fails.append("QDRANT_URL not set (step 17 RAG)")
    if fails:
        print("preconditions not met:")
        for f in fails:
            print(f"  - {f}")
        return 2

    print(f"""
3 demo tasks (gradually harder):

  (a) fix an import bug in a small file
  (b) add a CLI option to an existing argparse-based tool
  (c) refactor a 50-line function into 2-3 smaller ones with same behavior

run (after extending mini_agent with a rag_search tool):

  python lib/mini_agent.py \\
      --task "Fix the ImportError in src/utils/io.py" \\
      --workspace /path/to/repo \\
      --tools bash,read_file,write_file,edit_file,run_tests,rag_search,finish \\
      --rag-url $QDRANT_URL \\
      --max-turn 20 --auto-compact 0.8

record per task:
  - success / fail
  - turns used
  - tokens consumed
  - which tool was the bottleneck

success criterion: ≥ 2 of 3 pass first run; record a failure analysis for the third.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
