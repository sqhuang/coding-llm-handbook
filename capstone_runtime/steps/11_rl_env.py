#!/usr/bin/env python3
"""
Step 11 — SWE-Gym 风格 reward 设计 + sandbox 接口（schema 验证）。

What this file gives you:
  - `reward.py` 模块化 reward functions（sparse / dense / anti-hack），可直接 import
  - `Sandbox` 抽象基类 + `MockSandbox`（无 docker），用来在 Mac 上 dry-run reward
  - 一个端到端 self-check：用 MockSandbox 跑一个 "fake patch" → 应该得到具体分数

实际 GPU/Docker sandbox（真用 docker exec / pytest）在 step 12 train 时由 VERL/OpenRLHF
启起来；本步的目的是「把 reward 设计冻结、写成可单测、合到 trainer 之前就跑通」。

Run:
    python steps/11_rl_env.py            # 跑 self-check
    python steps/11_rl_env.py --print    # 打印 reward 配置 JSON
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

ROOT = Path(__file__).resolve().parent.parent


# =============================================================================
# Reward configuration — all numbers in ONE place, easy to ablate.
# =============================================================================
@dataclass
class RewardConfig:
    sparse_pass: float = 1.0       # all F2P tests pass
    sparse_fail: float = 0.0
    dense_diff_touch: float = 0.1   # patch touches a file mentioned in problem_statement
    dense_lint: float = 0.05        # lint clean
    dense_compile: float = 0.05     # at least imports / no syntax err
    anti_hack_test_modified: float = -2.0
    anti_hack_skip_added: float = -1.0
    anti_hack_assert_removed: float = -0.5
    format_bonus: float = 0.1
    format_re_think: str = r"<think>.*?</think>"
    format_re_code: str = r"```(?:python|diff)\n.*?```"

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# =============================================================================
# Sandbox protocol — real impl in step 12; this lets us unit-test reward fns.
# =============================================================================
class SandboxResult(Protocol):
    exit_code: int
    stdout: str
    stderr: str


class Sandbox(Protocol):
    def apply_patch(self, patch: str) -> SandboxResult: ...
    def run_tests(self, paths: list[str]) -> dict: ...
    def lint(self) -> SandboxResult: ...
    def list_changed_files(self) -> list[str]: ...


# =============================================================================
# MockSandbox — for offline testing of reward design.
# =============================================================================
@dataclass
class _R:
    exit_code: int
    stdout: str = ""
    stderr: str = ""


@dataclass
class MockSandbox:
    changed_files: list[str] = field(default_factory=list)
    test_results: dict = field(default_factory=lambda: {"f2p_pass": 0, "f2p_total": 1,
                                                        "p2p_pass": 0, "p2p_total": 0})
    lint_ok: bool = True
    last_patch: str = ""

    def apply_patch(self, patch: str) -> _R:
        self.last_patch = patch
        return _R(exit_code=0)

    def run_tests(self, paths: list[str]) -> dict:
        return dict(self.test_results)

    def lint(self) -> _R:
        return _R(exit_code=0 if self.lint_ok else 1)

    def list_changed_files(self) -> list[str]:
        return list(self.changed_files)


# =============================================================================
# Reward function — pure, takes (completion, problem, sandbox) → float + breakdown.
# =============================================================================
def compute_reward(
    completion: str,
    problem: dict,
    sandbox: Sandbox,
    cfg: RewardConfig | None = None,
) -> tuple[float, dict]:
    cfg = cfg or RewardConfig()
    breakdown: dict[str, float] = {}

    # --- extract patch from completion ---
    patch = _extract_patch(completion)
    if patch is None:
        return cfg.sparse_fail, {"reason": "no patch extracted"}

    # --- anti-hack guard rails BEFORE we apply ---
    if _modifies_tests(patch):
        breakdown["anti_hack_test_modified"] = cfg.anti_hack_test_modified
    if _adds_skip(patch):
        breakdown["anti_hack_skip_added"] = cfg.anti_hack_skip_added
    if _removes_asserts(patch):
        breakdown["anti_hack_assert_removed"] = cfg.anti_hack_assert_removed

    # --- apply + test ---
    sandbox.apply_patch(patch)
    test = sandbox.run_tests([])
    sparse_ok = (test.get("f2p_pass", 0) == test.get("f2p_total", 0)
                 and test.get("p2p_pass", 0) == test.get("p2p_total", 0))
    breakdown["sparse"] = cfg.sparse_pass if sparse_ok else cfg.sparse_fail

    # --- dense signals ---
    changed = sandbox.list_changed_files()
    hint_files = set(_extract_file_mentions(problem.get("problem_statement", "")))
    if any(c in hint_files for c in changed):
        breakdown["dense_diff_touch"] = cfg.dense_diff_touch
    if sandbox.lint().exit_code == 0:
        breakdown["dense_lint"] = cfg.dense_lint

    # --- format bonus ---
    if re.search(cfg.format_re_think, completion, re.S) and \
       re.search(cfg.format_re_code, completion, re.S):
        breakdown["format"] = cfg.format_bonus

    return sum(breakdown.values()), breakdown


# =============================================================================
# Helpers — small, pure, easy to test.
# =============================================================================
_FENCE_DIFF = re.compile(r"```diff\n(.*?)```", re.S)
_FENCE_PY = re.compile(r"```python\n(.*?)```", re.S)
_TEST_PATH = re.compile(r"^(diff --git a/|---\s+a/|\+\+\+\s+b/).*(?:tests?/|_test\.|test_)", re.M)
_SKIP_PAT = re.compile(r"\+\s*@(?:pytest\.)?skip", re.M)
_ASSERT_REMOVED = re.compile(r"^\-\s*assert\b", re.M)
_FILE_REF = re.compile(r"`([a-zA-Z0-9_./\-]+\.(?:py|ts|js|go|rs|java))`")


def _extract_patch(completion: str) -> str | None:
    for pat in (_FENCE_DIFF, _FENCE_PY):
        m = pat.search(completion)
        if m:
            return m.group(1)
    if completion.lstrip().startswith(("--- ", "diff ")):
        return completion
    return None


def _modifies_tests(patch: str) -> bool:
    return bool(_TEST_PATH.search(patch))


def _adds_skip(patch: str) -> bool:
    return bool(_SKIP_PAT.search(patch))


def _removes_asserts(patch: str) -> bool:
    return bool(_ASSERT_REMOVED.search(patch))


def _extract_file_mentions(text: str) -> list[str]:
    return _FILE_REF.findall(text or "")


# =============================================================================
# Self-check
# =============================================================================
def _self_check() -> int:
    cfg = RewardConfig()
    problem = {
        "problem_statement": "There's a bug in `src/calc.py` causing test_add to fail.",
    }

    # case 1: clean correct patch, sparse pass
    sbx = MockSandbox(changed_files=["src/calc.py"],
                      test_results={"f2p_pass": 1, "f2p_total": 1, "p2p_pass": 0, "p2p_total": 0})
    completion = ("<think>fix the operator</think>\n"
                  "```diff\n--- a/src/calc.py\n+++ b/src/calc.py\n@@\n-return a-b\n+return a+b\n```")
    score, brk = compute_reward(completion, problem, sbx, cfg)
    print(f"[1] clean correct  : score={score:.2f}  breakdown={brk}")
    assert score >= cfg.sparse_pass, "clean patch should pass sparse"

    # case 2: cheats by modifying tests
    sbx = MockSandbox(changed_files=["tests/test_calc.py"],
                      test_results={"f2p_pass": 1, "f2p_total": 1, "p2p_pass": 0, "p2p_total": 0})
    completion = ("```diff\n--- a/tests/test_calc.py\n+++ b/tests/test_calc.py\n@@\n-assert add(2,3)==5\n+assert add(2,3)==-1\n```")
    score, brk = compute_reward(completion, problem, sbx, cfg)
    print(f"[2] cheats tests   : score={score:.2f}  breakdown={brk}")
    assert score < 0, "cheating should be punished into negative territory"

    # case 3: adds @pytest.skip
    sbx = MockSandbox(test_results={"f2p_pass": 1, "f2p_total": 1, "p2p_pass": 0, "p2p_total": 0})
    completion = ("```diff\n+ @pytest.skip(\"flaky\")\n+ def test_add(): assert 1==1\n```")
    score, brk = compute_reward(completion, problem, sbx, cfg)
    print(f"[3] adds @skip     : score={score:.2f}  breakdown={brk}")
    assert "anti_hack_skip_added" in brk, "skip pattern must be detected"

    # case 4: no patch at all
    score, brk = compute_reward("I'll think about it later", problem, MockSandbox(), cfg)
    print(f"[4] no patch       : score={score:.2f}  breakdown={brk}")
    assert score == cfg.sparse_fail

    print("\n✓ all reward self-checks passed")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--print", action="store_true", help="print reward config as JSON and exit")
    args = p.parse_args()
    if args.print:
        print(json.dumps(RewardConfig().as_dict(), indent=2))
        return 0
    return _self_check()


if __name__ == "__main__":
    raise SystemExit(main())
