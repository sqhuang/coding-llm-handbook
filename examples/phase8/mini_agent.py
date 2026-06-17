"""
Minimal coding agent talking to a local GLM-5.2 via OpenAI-compat API,
running tools inside a Docker sandbox. Phase 8 §5 抽取脚本（< 300 行）。

依赖：
    pip install openai docker tree-sitter-languages

Run:
    export LLM_BASE_URL=http://localhost:8000/v1
    export LLM_MODEL=glm-5.1
    python mini_agent.py
"""
import io
import json
import os
import tarfile
import textwrap
import time
import uuid

import docker
from openai import OpenAI

# ---------- 1. LLM client ----------
client = OpenAI(
    base_url=os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
    api_key=os.getenv("LLM_API_KEY", "sk-local"),
)
MODEL = os.getenv("LLM_MODEL", "glm-5.1")
CTX_WINDOW = int(os.getenv("LLM_CTX", "65536"))


# ---------- 2. Sandbox ----------
class Sandbox:
    def __init__(self, image="python:3.11-slim"):
        self.docker = docker.from_env()
        self.name = f"mini-agent-{uuid.uuid4().hex[:8]}"
        self.c = self.docker.containers.run(
            image, command="sleep infinity", detach=True,
            working_dir="/workspace", mem_limit="2g",
            tmpfs={"/tmp": "size=256m"}, name=self.name,
        )
        self.exec("mkdir -p /workspace && pip install -q pytest")

    def exec(self, cmd, cwd="/workspace", timeout=60):
        r = self.c.exec_run(["bash", "-lc", cmd], workdir=cwd, demux=True)
        out, err = r.output
        return {
            "exit_code": r.exit_code,
            "stdout": (out or b"").decode("utf-8", "replace"),
            "stderr": (err or b"").decode("utf-8", "replace"),
        }

    def put(self, path, content):
        data = content.encode()
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name=path.lstrip("/"))
            info.size = len(data)
            info.mtime = int(time.time())
            tar.addfile(info, io.BytesIO(data))
        buf.seek(0)
        self.c.put_archive("/", buf.read())

    def get(self, path):
        stream, _ = self.c.get_archive(path)
        buf = io.BytesIO(b"".join(stream))
        buf.seek(0)
        with tarfile.open(fileobj=buf) as tar:
            m = tar.getmembers()[0]
            return tar.extractfile(m).read().decode("utf-8", "replace")

    def close(self):
        try:
            self.c.kill()
        except Exception:
            pass
        try:
            self.c.remove()
        except Exception:
            pass


# ---------- 3. Tools ----------
TOOLS = [
    {"type": "function", "function": {"name": "read_file", "description": "Read a file. Returns content with line numbers.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer", "default": 1}, "end_line": {"type": "integer", "default": 400}}, "required": ["path"]}}},
    {"type": "function", "function": {"name": "write_file", "description": "Overwrite a file with content.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "edit_file", "description": "Replace exactly ONE occurrence of old_str with new_str in file.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old_str": {"type": "string"}, "new_str": {"type": "string"}}, "required": ["path", "old_str", "new_str"]}}},
    {"type": "function", "function": {"name": "bash", "description": "Run a shell command in sandbox.", "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}},
    {"type": "function", "function": {"name": "grep", "description": "ripgrep in /workspace. Returns file:line:match.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string", "default": "/workspace"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "run_tests", "description": "Run pytest in sandbox.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "default": "/workspace"}}}}},
    {"type": "function", "function": {"name": "finish", "description": "Finish task with summary.", "parameters": {"type": "object", "properties": {"summary": {"type": "string"}}, "required": ["summary"]}}},
]


def trunc(s, n=6000):
    if len(s) <= n:
        return s
    return s[: n // 2] + f"\n... <truncated {len(s)-n} chars> ...\n" + s[-n // 2:]


class ToolExecutor:
    def __init__(self, sbx: Sandbox):
        self.sbx = sbx

    def read_file(self, path, start_line=1, end_line=400):
        try:
            content = self.sbx.get(path)
        except Exception as e:
            return f"ERROR: {e}"
        lines = content.splitlines()
        sl, el = max(1, start_line), min(len(lines), end_line)
        numbered = "\n".join(f"{i:>5} | {l}" for i, l in enumerate(lines[sl-1:el], sl))
        return trunc(numbered)

    def write_file(self, path, content):
        self.sbx.put(path, content)
        return f"Wrote {len(content)} bytes to {path}"

    def edit_file(self, path, old_str, new_str):
        content = self.sbx.get(path)
        n = content.count(old_str)
        if n == 0:
            return "ERROR: old_str not found"
        if n > 1:
            return f"ERROR: old_str matches {n} places; make it unique"
        new = content.replace(old_str, new_str, 1)
        self.sbx.put(path, new)
        return "OK: applied edit"

    def bash(self, cmd):
        r = self.sbx.exec(cmd)
        return trunc(f"exit={r['exit_code']}\nstdout:\n{r['stdout']}\nstderr:\n{r['stderr']}")

    def grep(self, pattern, path="/workspace"):
        # use python re since rg may not be installed
        cmd = f"grep -rn -E {json.dumps(pattern)} {path} || true"
        return trunc(self.sbx.exec(cmd)["stdout"])

    def run_tests(self, path="/workspace"):
        r = self.sbx.exec(f"cd {path} && pytest -x --tb=short 2>&1 | tail -80")
        return r["stdout"] + r["stderr"]

    def finish(self, summary):
        return f"FINISHED: {summary}"

    def dispatch(self, name, args):
        fn = getattr(self, name, None)
        if not fn:
            return f"ERROR: no tool {name}"
        try:
            return fn(**args)
        except Exception as e:
            return f"ERROR in {name}: {e}"


# ---------- 4. Agent loop ----------
SYSTEM = """You are a Python coding agent. You fix bugs by:
1. Using grep/read_file to locate code.
2. Using edit_file or write_file to change it.
3. Using run_tests to verify.
4. Calling finish(summary) when done.

Rules:
- Always read before you edit.
- edit_file requires UNIQUE old_str; add surrounding context lines if needed.
- If tests fail, read the traceback and iterate.
- Keep each step small; do not dump huge files in write_file unless creating a new one."""


def run_agent(task: str, sbx: Sandbox, max_steps=25):
    tools = ToolExecutor(sbx)
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": task},
    ]
    for step in range(max_steps):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=msgs,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=2048,
        )
        m = resp.choices[0].message
        msgs.append(m.model_dump(exclude_none=True))

        if not m.tool_calls:
            print(f"[step {step}] (no tool) {m.content}")
            break

        for tc in m.tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except Exception:
                args = {}
            print(f"[step {step}] → {name}({list(args)[:3]})")
            result = tools.dispatch(name, args)
            print(f"            {result[:200].replace(chr(10),' ')}")
            msgs.append({"role": "tool", "tool_call_id": tc.id,
                         "name": name, "content": result})
            if name == "finish":
                return result
    return "MAX_STEPS_REACHED"


# ---------- 5. Demo: fix a small Python bug ----------
if __name__ == "__main__":
    sbx = Sandbox()
    try:
        sbx.put("/workspace/calc.py", textwrap.dedent("""
            def add(a, b):
                return a - b   # BUG: should be +

            def div(a, b):
                return a / b
        """).strip())
        sbx.put("/workspace/test_calc.py", textwrap.dedent("""
            from calc import add, div
            def test_add(): assert add(2,3) == 5
            def test_div(): assert div(6,2) == 3
        """).strip())
        task = "There is a bug in /workspace/calc.py that makes test_add fail. Fix it and make all tests pass."
        out = run_agent(task, sbx)
        print("===", out)
    finally:
        sbx.close()
