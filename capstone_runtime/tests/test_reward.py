"""Unit tests for the GRPO reward functions in step 11."""
from .conftest import load_step


def test_clean_correct_patch_gets_positive_reward():
    mod = load_step("11_rl_env.py")
    cfg = mod.RewardConfig()
    sbx = mod.MockSandbox(
        changed_files=["src/calc.py"],
        test_results={"f2p_pass": 1, "f2p_total": 1, "p2p_pass": 0, "p2p_total": 0},
    )
    completion = (
        "<think>fix the sign</think>\n"
        "```diff\n--- a/src/calc.py\n+++ b/src/calc.py\n"
        "@@\n-return a-b\n+return a+b\n```"
    )
    score, brk = mod.compute_reward(completion, {"problem_statement": "bug in `src/calc.py`"},
                                    sbx, cfg)
    assert score >= cfg.sparse_pass
    assert brk["sparse"] == cfg.sparse_pass
    assert "dense_diff_touch" in brk


def test_test_modification_penalty():
    mod = load_step("11_rl_env.py")
    sbx = mod.MockSandbox()
    completion = (
        "```diff\n--- a/tests/test_calc.py\n+++ b/tests/test_calc.py\n"
        "@@\n-assert add(2,3)==5\n+assert add(2,3)==-1\n```"
    )
    score, brk = mod.compute_reward(completion, {}, sbx)
    assert "anti_hack_test_modified" in brk
    assert score < 0


def test_skip_addition_penalty():
    mod = load_step("11_rl_env.py")
    sbx = mod.MockSandbox(test_results={"f2p_pass": 1, "f2p_total": 1,
                                        "p2p_pass": 0, "p2p_total": 0})
    completion = (
        "```diff\n+ @pytest.skip(\"flaky\")\n+ def test_thing(): assert 1==1\n```"
    )
    score, brk = mod.compute_reward(completion, {}, sbx)
    assert "anti_hack_skip_added" in brk


def test_no_patch_returns_fail():
    mod = load_step("11_rl_env.py")
    score, brk = mod.compute_reward("I'll do it later", {}, mod.MockSandbox())
    assert score == 0.0
    assert brk == {"reason": "no patch extracted"}


def test_format_bonus_requires_both_think_and_code():
    mod = load_step("11_rl_env.py")
    cfg = mod.RewardConfig()
    sbx = mod.MockSandbox(test_results={"f2p_pass": 0, "f2p_total": 1,
                                        "p2p_pass": 0, "p2p_total": 0})
    only_code = "```diff\n+a\n```"
    score_only, brk_only = mod.compute_reward(only_code, {}, sbx, cfg)
    assert "format" not in brk_only

    full = "<think>plan</think>\n```diff\n+a\n```"
    score_full, brk_full = mod.compute_reward(full, {}, sbx, cfg)
    assert brk_full.get("format") == cfg.format_bonus
