#!/usr/bin/env python3
"""
Step 01 — 选基座 + 现状评测（base API）。

What this does:
  1. Reads env: GLM_API_KEY / LLM_BASE_URL (your local Air-Base via vLLM/SGLang)
  2. For each configured model, calls the OpenAI-compatible /chat/completions
     endpoint on a tiny built-in 3-question evalset (HumanEval-style) and
     records pass/fail by simple `exec()` of the model's solution against
     reference asserts.
  3. Writes `logs/baseline_report.md`.

NOT a full HumanEval+ run — for that, install `evalplus` (Tier 2) and run:
    evalplus.evaluate --dataset humaneval --model <name> --backend openai \\
        --base-url $LLM_BASE_URL --tokens 4096

This script is the "minimum proof" form: needs only the `requests` lib + your
API keys; gives you a first signal in < 5 min instead of waiting for evalplus
to download HumanEval+ + benchmark 164 problems.

Usage:
    export GLM_API_KEY=...                # or skip to disable GLM-5.1 leg
    export LLM_BASE_URL=http://localhost:30000/v1
    export LLM_MODEL=glm-4.5-air-base
    python steps/01_baseline.py
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent

# ---------------- mini evalset ----------------
MINI_EVAL = [
    {
        "id": "add",
        "prompt": "Write a Python function `add(a, b)` that returns the sum of two integers.",
        "tests": "assert add(2, 3) == 5\nassert add(-1, 1) == 0\nassert add(0, 0) == 0\n",
    },
    {
        "id": "is_prime",
        "prompt": "Write a Python function `is_prime(n)` that returns True iff n is prime.",
        "tests": "assert is_prime(2) is True\nassert is_prime(11) is True\nassert is_prime(1) is False\nassert is_prime(15) is False\n",
    },
    {
        "id": "reverse_words",
        "prompt": "Write a Python function `reverse_words(s)` that reverses the order of words in a sentence.",
        "tests": "assert reverse_words('hello world') == 'world hello'\nassert reverse_words('a b c') == 'c b a'\nassert reverse_words('one') == 'one'\n",
    },
]

CODE_FENCE = re.compile(r"```(?:python\n)?(.*?)```", re.S)


def extract_code(text: str) -> str:
    m = CODE_FENCE.search(text or "")
    if m:
        return m.group(1).strip()
    return text or ""


def run_with_tests(code: str, tests: str, timeout: int = 10) -> tuple[bool, str]:
    prog = code + "\n\n" + tests
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(prog)
        path = f.name
    try:
        r = subprocess.run([sys.executable, path], capture_output=True,
                           timeout=timeout, text=True)
        return r.returncode == 0, (r.stderr or r.stdout)[:400]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)
    finally:
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass


def call_model(base_url: str, api_key: str, model: str, prompt: str) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a coding assistant. "
                                          "Reply with ONE python code block only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 512,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def eval_one_model(label: str, base_url: str, api_key: str, model: str) -> dict:
    print(f"\n>>> {label}  ({model} @ {base_url})")
    rows = []
    for ex in MINI_EVAL:
        try:
            raw = call_model(base_url, api_key, model, ex["prompt"])
        except Exception as e:
            print(f"  {ex['id']:>15}  ERROR  {e}")
            rows.append({"id": ex["id"], "ok": False, "err": str(e)[:200]})
            continue
        code = extract_code(raw)
        ok, log = run_with_tests(code, ex["tests"])
        glyph = "✓" if ok else "✗"
        print(f"  {ex['id']:>15}  {glyph}  {'pass' if ok else log[:80]}")
        rows.append({"id": ex["id"], "ok": ok, "completion": raw})
    passed = sum(1 for r in rows if r["ok"])
    return {"label": label, "model": model, "base_url": base_url,
            "passed": passed, "total": len(MINI_EVAL), "rows": rows}


def main() -> int:
    targets: list[dict] = []

    # GLM-5.1 API leg
    glm_key = os.environ.get("GLM_API_KEY")
    if glm_key:
        targets.append({
            "label": "GLM-5.1 (API)",
            "base_url": os.environ.get("GLM_BASE_URL", "https://api.z.ai/v1"),
            "api_key": glm_key,
            "model": "glm-5.1",
        })
    else:
        print("note: GLM_API_KEY not set — skipping GLM-5.1 API comparison")

    # Local-deployed model (from step 15, or any OpenAI-compat endpoint)
    local_url = os.environ.get("LLM_BASE_URL")
    if local_url:
        targets.append({
            "label": "Local model",
            "base_url": local_url,
            "api_key": os.environ.get("LLM_API_KEY", "sk-local"),
            "model": os.environ.get("LLM_MODEL", "glm-4.5-air-base"),
        })
    else:
        print("note: LLM_BASE_URL not set — skipping local model leg")

    if not targets:
        print("\nERROR: no model endpoints configured. Set GLM_API_KEY or LLM_BASE_URL.")
        print("Hint: copy env.example to .env, fill it in, then `source .env`.")
        return 2

    results = []
    for t in targets:
        try:
            results.append(eval_one_model(t["label"], t["base_url"], t["api_key"], t["model"]))
        except Exception as e:
            print(f"\nFATAL on {t['label']}: {e}")
            results.append({"label": t["label"], "model": t["model"], "error": str(e),
                            "passed": 0, "total": len(MINI_EVAL), "rows": []})

    # ---- report ----
    out = ROOT / "logs" / "baseline_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("# Baseline (mini 3-task evalset)\n\n")
        f.write("> 这是 step 01 的「轻量」评测——3 道题 ≤ 5 min。\n")
        f.write("> 真正的 HumanEval+ 跑法见 `steps/13_pub_eval.py`。\n\n")
        f.write("| 模型 | 通过 | 备注 |\n|---|---|---|\n")
        for r in results:
            note = r.get("error", "")
            f.write(f"| {r['label']} (`{r['model']}`) | {r['passed']}/{r['total']} | {note} |\n")
        f.write("\n## 思考题（手填）\n")
        f.write("- GLM-5.1 vs base 差距来源猜测：\n- 内部 5 题手测准备清单：\n")
    print(f"\n✓ wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
