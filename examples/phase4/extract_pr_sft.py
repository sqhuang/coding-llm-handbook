"""
GitHub merged-PR → ChatML SFT jsonl. Phase 4 §10.9 抽取脚本片段。

形态：Issue → unified diff（四形态中的 (b)）。
要求：环境变量 GH_TOKEN 已设置（一个 PAT，scope=repo:read 即可）。

Run:
    export GH_TOKEN=ghp_xxx
    python extract_pr_sft.py
Outputs pr_sft.jsonl in the current directory.
"""
import json
import os
import re
import time
from pathlib import Path

import requests

GITHUB = "https://api.github.com"
HEADERS = {
    "Authorization": f"token {os.environ['GH_TOKEN']}",
    "Accept": "application/vnd.github+json",
}
BOT_AUTHORS = {"dependabot[bot]", "renovate[bot]", "mergify[bot]"}
CODE_EXTS = {".py", ".ts", ".tsx", ".go", ".java", ".rs", ".cpp", ".sql"}
CLOSES_RE = re.compile(r"(?:closes|fixes|resolves)\s+#(\d+)", re.I)


def fetch_merged_prs(repo, since):
    url = f"{GITHUB}/repos/{repo}/pulls"
    params = {"state": "closed", "per_page": 100,
              "sort": "updated", "direction": "desc"}
    while url:
        r = requests.get(url, headers=HEADERS, params=params)
        r.raise_for_status()
        for pr in r.json():
            if pr["merged_at"] and pr["merged_at"] >= since:
                yield pr
        url = r.links.get("next", {}).get("url")
        time.sleep(0.5)  # rate limit


def get_pr_detail(repo, num):
    pr = requests.get(f"{GITHUB}/repos/{repo}/pulls/{num}", headers=HEADERS).json()
    diff = requests.get(pr["diff_url"], headers=HEADERS).text
    revs = requests.get(f"{GITHUB}/repos/{repo}/pulls/{num}/comments", headers=HEADERS).json()
    return pr, diff, revs


def find_linked_issue(repo, pr):
    body = (pr.get("body") or "") + " " + (pr.get("head", {}).get("ref") or "")
    m = CLOSES_RE.search(body)
    if not m:
        return None
    iss = requests.get(f"{GITHUB}/repos/{repo}/issues/{m.group(1)}", headers=HEADERS).json()
    return iss if "title" in iss else None


def quality_ok(pr, diff):
    if pr["user"]["login"] in BOT_AUTHORS:
        return False
    if len(diff.splitlines()) > 5000:
        return False
    if not any(f".{e.lstrip('.')}" in diff for e in CODE_EXTS):
        return False
    # 去 whitespace 后是否还有内容
    non_ws = [l for l in diff.splitlines() if l.startswith(("+", "-")) and l[1:].strip()]
    return len(non_ws) >= 2


def to_chatml(issue, diff):
    user = (f"# Issue #{issue['number']}: {issue['title']}\n\n"
            f"{issue['body']}\n\n"
            f"请输出 unified diff 补丁。")
    asst = f"```diff\n{diff}\n```"
    return {"messages": [
        {"role": "system", "content": "你是 <公司名> coding agent。"},
        {"role": "user", "content": user},
        {"role": "assistant", "content": asst},
    ]}


def main(repo, since, out_path):
    with Path(out_path).open("w") as f:
        for pr in fetch_merged_prs(repo, since):
            full, diff, _ = get_pr_detail(repo, pr["number"])
            if not quality_ok(full, diff):
                continue
            iss = find_linked_issue(repo, full)
            if not iss:
                continue
            f.write(json.dumps(to_chatml(iss, diff), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main("your-org/your-repo", "2022-01-01T00:00:00Z", "pr_sft.jsonl")
