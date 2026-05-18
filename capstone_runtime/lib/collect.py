"""
Internal SWE-Bench v1 candidate collector. Phase 6 §11.9 抽取脚本（脚本 A）。

输入：GitHub repo + 时间窗 + 标签筛选条件。
输出：每个 PR 一条 instance JSON，含 problem_statement / test_patch / gold_patch / F2P / P2P。

Run:
    export GH_TOKEN=ghp_xxx
    python collect.py
注意：`extract_test_diff` / `extract_code_diff` / `strip_solution_hints` 是公司内部辅助函数，
需根据自家 monorepo 习惯实现（拆 test 目录 vs 非 test 目录、去掉 issue body 中"修法提示"等）。
"""
import json
import os
import subprocess
from pathlib import Path

import requests

GH_TOKEN = os.environ["GH_TOKEN"]
HEADERS = {"Authorization": f"Bearer {GH_TOKEN}"}


def fetch_candidate_prs(repo, since, until, label="bug"):
    url = "https://api.github.com/search/issues"
    q = f"repo:{repo} is:pr is:merged label:{label} merged:{since}..{until}"
    r = requests.get(url, params={"q": q, "per_page": 100}, headers=HEADERS)
    return r.json()["items"]


def checkout_and_run_tests(repo_dir, commit, test_patch=None):
    subprocess.run(["git", "-C", repo_dir, "checkout", "-f", commit], check=True)
    if test_patch:
        subprocess.run(["git", "-C", repo_dir, "apply", test_patch], check=True)
    subprocess.run(
        ["pytest", "--tb=no", "-q",
         "--json-report", "--json-report-file=/tmp/r.json"],
        cwd=repo_dir, capture_output=True, timeout=600,
    )
    return json.loads(Path("/tmp/r.json").read_text())["tests"]


def build_instance(repo_dir, pr):
    base = pr["base"]["sha"]                       # merge base, bug 仍在
    merge = pr["merge_commit_sha"]                 # 修复后
    test_patch = extract_test_diff(repo_dir, base, merge)  # 只保留测试文件改动
    gold_patch = extract_code_diff(repo_dir, base, merge)  # 只保留非测试改动

    before = checkout_and_run_tests(repo_dir, base, test_patch)
    after = checkout_and_run_tests(repo_dir, merge)

    f2p = [t["nodeid"] for t in after if t["outcome"] == "passed"
           and any(b["nodeid"] == t["nodeid"] and b["outcome"] == "failed" for b in before)]
    p2p = [t["nodeid"] for t in after if t["outcome"] == "passed"
           and any(b["nodeid"] == t["nodeid"] and b["outcome"] == "passed" for b in before)]

    return {
        "instance_id": f"internal__{pr['number']}",
        "repo": pr["base"]["repo"]["full_name"],
        "base_commit": base,
        "problem_statement": strip_solution_hints(pr["body"]),
        "test_patch": test_patch,
        "gold_patch": gold_patch,
        "FAIL_TO_PASS": f2p,
        "PASS_TO_PASS": p2p,
    }
