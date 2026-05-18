#!/usr/bin/env python3
"""
Step 02 — 搭内部 SWE-Bench v0（10 题）。

WHAT: 拉公司 repo 的 merged + bug-label PR，对每个 PR 重建 base/merge commit，
跑 pytest，自动算出 F2P/P2P 测试集，产出 swebench-instance JSON。

REQUIRES:
  - hw:       0 GPU；docker（跑 sandbox）
  - software: git, docker, python3 (>=3.9)
  - data:     公司 repo 的 git 访问权限
  - env:      GH_TOKEN（如果是 GitHub）；或 GITLAB_TOKEN

THIS FILE: 创建初始目录结构 + 写一个 sample task JSON，让你看到 schema。
真正的批量抽取脚本见 `../examples/phase6/collect.py`（项目根 examples/）。

Usage:
    python steps/02_internal_bench.py --out internal_bench_v0/
"""
import argparse
import json
import shutil
import sys
from pathlib import Path


SAMPLE = {
    "instance_id": "internal__sample-001",
    "repo": "internal/foo",
    "base_commit": "abc123def456",
    "problem_statement": (
        "When `Order.compute_total()` is called with a discount > 100%, "
        "the result is negative instead of clamped to 0. See `src/order.py:compute_total`."
    ),
    "test_patch": (
        "--- a/tests/test_order.py\n+++ b/tests/test_order.py\n"
        "@@\n+def test_discount_over_100():\n+    o = Order(items=[Item(price=10)])\n"
        "+    assert o.compute_total(discount=1.5) == 0\n"
    ),
    "gold_patch": (
        "--- a/src/order.py\n+++ b/src/order.py\n"
        "@@\n-    return total - total*discount\n+    return max(0.0, total - total*discount)\n"
    ),
    "FAIL_TO_PASS": ["tests/test_order.py::test_discount_over_100"],
    "PASS_TO_PASS": [
        "tests/test_order.py::test_no_discount",
        "tests/test_order.py::test_50pct_discount",
    ],
    "_note": "Replace this with real instances via examples/phase6/collect.py",
}


RUN_SH = """#!/usr/bin/env bash
# Usage: ./run.sh <instance_id> <patch.diff>
set -euo pipefail
INSTANCE=${1:?instance_id required}
PATCH=${2:?patch file required}
TASK_JSON=instances/${INSTANCE}.json

# 1. checkout base_commit in a fresh worktree
BASE=$(jq -r .base_commit "$TASK_JSON")
REPO=$(jq -r .repo "$TASK_JSON")
WORK=$(mktemp -d)
git clone "git@github.com:${REPO}.git" "$WORK"
git -C "$WORK" checkout "$BASE"

# 2. apply test_patch (gold tests) + agent patch
jq -r .test_patch "$TASK_JSON" | git -C "$WORK" apply -
git -C "$WORK" apply "$PATCH"

# 3. run F2P + P2P tests, report
F2P=$(jq -r '.FAIL_TO_PASS | join(" ")' "$TASK_JSON")
P2P=$(jq -r '.PASS_TO_PASS | join(" ")' "$TASK_JSON")
docker run --rm -v "$WORK":/repo -w /repo python:3.11-slim \\
  bash -lc "pip install -q -r requirements.txt && pytest $F2P $P2P --tb=short"
"""


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="internal_bench_v0",
                   help="output dir relative to cwd")
    args = p.parse_args()

    if not shutil.which("git"):
        print("WARN: git not on PATH — install before adding real instances", file=sys.stderr)
    if not shutil.which("docker"):
        print("WARN: docker not on PATH — required for run.sh sandbox", file=sys.stderr)

    out = Path(args.out)
    (out / "instances").mkdir(parents=True, exist_ok=True)
    sample_path = out / "instances" / f"{SAMPLE['instance_id']}.json"
    sample_path.write_text(json.dumps(SAMPLE, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "run.sh").write_text(RUN_SH, encoding="utf-8")
    (out / "run.sh").chmod(0o755)
    (out / "README.md").write_text(
        "# internal_bench_v0\n\n"
        "10 instances to be collected via `examples/phase6/collect.py`.\n"
        "Sample format: `instances/internal__sample-001.json`.\n"
        "Run one: `./run.sh internal__<id> path/to/agent.patch`.\n",
        encoding="utf-8",
    )
    print(f"✓ scaffolded {out}/  (1 sample instance + run.sh + README)")
    print("next: run `python ../examples/phase6/collect.py --repo <org/repo> --since YYYY-MM-DD ...`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
