#!/usr/bin/env python3
"""
Step 10 — Agent 轨迹合成（30 题 × 35 尝试 ≈ 1050 条）。

WHAT: 复用 lib/mini_agent.py 作为采集器：每题让"老师模型"（GLM-5.1 API 或本地
     Air-Instruct）跑 ReAct loop，把完整 (思考, tool_call, tool_result) 序列存 jsonl。

REQUIRES:
  - hw:       1 GPU（本地教师）或 0（API 教师）
  - software: lib/mini_agent.py 的依赖（openai, docker）
  - data:     30 个题面 + sandbox docker image（来自 step 02 + step 11）
  - env:      GLM_API_KEY 或 LLM_BASE_URL, GH_TOKEN

THIS FILE: 检查 sandbox + 模型连接，打印一行启动命令。
"""
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MINI_AGENT = ROOT / "lib" / "mini_agent.py"


def main() -> int:
    fails = []
    if not MINI_AGENT.exists():
        fails.append(f"{MINI_AGENT} missing — was lib/ removed from bundle?")
    if not shutil.which("docker"):
        fails.append("docker not on PATH — required for sandbox")
    if not (os.environ.get("GLM_API_KEY") or os.environ.get("LLM_BASE_URL")):
        fails.append("set GLM_API_KEY or LLM_BASE_URL for the teacher model")

    if fails:
        print("preconditions not met:")
        for f in fails:
            print(f"  - {f}")
        return 2

    print(f"""
ready. Trajectory collection plan:

  tasks            : 30 (10 from internal_bench_v0 + 20 from public repos)
  attempts/task    : 35
  max_turn         : 25
  output           : data/agent_traj.jsonl
  pass min         : keep at least 30 final-passing trajectories

run (the lib/mini_agent.py needs `--collect-mode` flag, TODO):

  python lib/mini_agent.py \\
      --task-dir internal_bench_v0/instances/ \\
      --attempts 35 --max-turn 25 \\
      --output data/agent_traj.jsonl

(If you haven't added --collect-mode yet, copy mini_agent.py to lib/agent_collector.py
 and wrap run_agent in a loop that writes msgs to jsonl on each finish.)
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
