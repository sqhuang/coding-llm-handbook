# Phase 8：Coding Agent 应用层深度笔记

> 📅 主线快照：2026-04-22 · 上次核对：2026-04-30

> **⚡ 三句话要点**
> 1. 两条路径：**A · 外壳路径**首选 **Roo Code + LiteLLM proxy**，把本地 GLM-5.1 包成 OpenAI-compat 即插即用；**B · 自建路径**走 ReAct + Docker + 6 工具的 300 行骨架。
> 2. 自建 agent 的"三层栈"：**repo map**（tree-sitter 解析）→ **auto-compact**（长对话摘要）→ **Reflexion**（失败任务自我反思迭代），缺一就在大仓库上崩。
> 3. **Sandbox 必上**：Docker / Firejail / E2B 三选一，绝不让模型直接对宿主 fs 操作——一次错误的 `rm` 就毁掉测试环境。

> 面向中国 AI 研究者的工程化笔记。目标：(A) 把本地部署的 GLM-5.1 接进成熟的 agent 外壳；(B) 从零构建一个 minimal coding agent，吃透其内部原理。
>
> 本笔记假设你已完成 Phase 7（本地模型服务化），即已经把 GLM-5.1 / GLM-4.5-Air 以 OpenAI-compatible API 形式暴露在 `http://localhost:8000/v1`。

> **读者画像** · 想把本地模型变成"真能用"的 coding agent 的应用工程师；或想审计 Cline/Roo/OpenHands 内部原理的架构师。
> **前置知识** · phase7 已部署本地 OpenAI-compat endpoint；序.14 chat template + tool calling；phase4 §3 agent 轨迹数据。
> **学完能做** · 路径 A 把 GLM-5.1 接进 Cline/Roo 跑真项目；路径 B 自己写 ≤ 500 行 minimal coding agent 并搭 RAG 索引大仓库。

---

## 0. 为什么要单独有一个 "应用层" 阶段

过去两年 code model 的进步（DeepSeek-Coder、Qwen-Coder、GLM-Coder、Claude 3.5+、GPT-4.1）让 "单轮补全" 问题几乎饱和；真正决定用户体感的，是 **模型之外的脚手架**：

- 如何把整个 repo 喂给一个 32k~200k 上下文的模型？（repo map、embedding、LSP）
- 如何让模型真正执行代码而不是纸上谈兵？（sandbox、tool calling）
- 如何在出错时自我纠正？（ReAct、Reflexion、plan-execute）
- 如何在长对话中控制 token 预算？（auto-compact、工具结果截断）
- 如何既能本地跑又能安全？（Docker / E2B / Firecracker）

Anthropic 把这一层叫 "agent harness"。Cline/Roo/Kilo/Aider/OpenHands/SWE-Agent 都是对这层的不同答案。本笔记把这层拆开，让你能自己攒。

---

## 1. Coding Agent 架构全景图

### 1.1 分层视角

一个完整的 coding agent 可以被切成 6 层：

```mermaid
flowchart TB
  UI["6. 交互层<br/>CLI / VSCode 插件 / Web"]
  PLAN["5. 规划层<br/>ReAct · Plan-Execute · Reflexion"]
  CTX["4. 上下文管理<br/>Repo Map · 历史压缩 · 记忆"]
  MODEL["1. 模型层<br/>GLM-5.1 · Thinking + Tool Calling"]
  TOOL["3. 工具层<br/>bash · read · write · search · git"]
  SB["2. 沙箱层<br/>Docker · E2B · Firecracker"]
  FS["文件系统 + 网络 + 测试运行"]
  IDX["Repo 索引<br/>tree-sitter AST + embedding"]
  UI --> PLAN
  PLAN --> CTX
  PLAN --> MODEL
  CTX --> MODEL
  MODEL --> TOOL
  TOOL --> SB
  SB --> FS
  CTX --> IDX
```


```
┌───────────────────────────────────────────────┐
│ 6. 交互层 (Interaction)                         │
│   CLI / VSCode 插件 / Web / Chat UI            │
├───────────────────────────────────────────────┤
│ 5. 规划层 (Planning / Control)                  │
│   ReAct loop / Plan-Execute / Reflexion        │
├───────────────────────────────────────────────┤
│ 4. 上下文管理层 (Context Engine)                 │
│   Repo map / History compaction / Memory        │
├───────────────────────────────────────────────┤
│ 3. 工具层 (Tools)                               │
│   bash / read / write / search / browser / git │
├───────────────────────────────────────────────┤
│ 2. 沙箱层 (Sandbox / Execution)                 │
│   Docker / E2B / Firecracker / native          │
├───────────────────────────────────────────────┤
│ 1. 模型层 (Model)                               │
│   GLM-5.1 / GLM-4.5-Air (via OpenAI API)        │
└───────────────────────────────────────────────┘
```

### 1.2 六层职责与实现要点

**L1 模型层（Model）**
- 能力点：code + reasoning + tool calling + long context。
- GLM-5.1 支持 thinking tokens（`<think>…</think>`）和 OpenAI 风格的 `tools` 字段，这是接外壳的硬需求。
- 最小需求：`chat/completions` + `tools` + `stream` + `max_tokens ≥ 8k output`。

**L2 沙箱层（Sandbox）**
- 核心目标：文件系统隔离 + 网络可控 + 资源限制 + 可复现。
- 三条路线：
  - **Docker per-session**：一次会话一个容器，简单、本地可用、开源。
  - **E2B (code interpreter cloud)**：基于 Firecracker microVM，毫秒级冷启，API 化；收费。
  - **Firecracker 自建**：AWS 的 microVM，~125ms 启动，隔离强度堪比 VM，工程量大。
- 要点：`--network=none` 或白名单出口、`--read-only` 挂载系统镜像、`tmpfs` 给 `/tmp`。

**L3 工具层（Tools）**
- 最小工具集（这 6 个 + git 已能覆盖 80% 任务）：
  - `read_file(path, start?, end?)`：分片读，避免一次吐全文。
  - `write_file(path, content)`：覆盖；或 `edit_file(path, old, new)` 做 patch。
  - `list_dir(path)`：树形列。
  - `grep(pattern, path, regex?)`：ripgrep 包一层。
  - `bash(cmd, timeout, cwd)`：在沙箱里跑。
  - `run_tests(path)` 可选：等同 `bash("pytest ...")`，但分支返回结构化结果。
- 进阶：`apply_patch`（Aider 用 `SEARCH/REPLACE` 格式、Claude Code 用 unified diff）、`browser`（Playwright）、`lsp_goto_def`。

**L4 上下文管理层（Context Engine）**
- Repo map：Aider 式 "每个文件列出 top-k 签名"；tree-sitter 抓 AST 拿函数/类/导入。
- 历史压缩：Claude Code 的 "auto-compact" —— 对话超阈值时，用 LLM 自己总结前文，只保留 rolling summary + 最近 N 轮。
- 长期记忆：写入 `.agent/memory/*.md` 或向量库，新 session 启动时加载。
- 工具结果截断：stdout 超过 8k 行自动 head/tail 化，中间用 `... <truncated 1234 lines> ...`。

**L5 规划层（Planning）**
- **ReAct**：Thought → Action → Observation 单循环，最简单。
- **Plan-Execute**：先 LLM 出 plan（markdown todo list），再逐条执行，每条结束后 LLM 检查。
- **Reflexion**：失败时 LLM 写 "reflection"（我为什么错、下次怎么办），塞进下一轮 prompt。
- **Multi-agent (manager-worker)**：OpenHands 的 CodeActAgent + BrowsingAgent；Cline 的 Plan vs Act mode。

**L6 交互层（Interaction）**
- CLI：Aider、Claude Code CLI。
- VSCode/JetBrains 插件：Cline、Roo Code、Kilo Code、Continue。
- Web UI：OpenHands 默认的 React 前端。
- 选择逻辑：个人效率首选 IDE 插件；CI / batch 场景 CLI；共享给团队用 Web。

### 1.3 现代主流 agent 的分层映射

| 项目 | L6 交互 | L5 规划 | L4 上下文 | L3 工具 | L2 沙箱 | L1 模型 |
|---|---|---|---|---|---|---|
| Claude Code | CLI + VSCode | ReAct + auto-compact | repo + `CLAUDE.md` | bash/read/write/edit/glob/grep | 本地 (权限模型) | Claude 4.x |
| Cline | VSCode | Plan / Act 双模式 | file mention + 环境详情 | 全套 + browser | 本地 | 任意 OpenAI-compat |
| Roo Code | VSCode fork | 多 mode（Code/Ask/Architect/Debug） | 同上 + custom modes | 同上 + MCP | 本地 | 任意 |
| Kilo Code | VSCode | 合并 Cline + Roo | 同上 | 同上 | 本地 | 任意 |
| Aider | CLI | 无显式 plan，repo-map 驱动 | tree-sitter repo map | SEARCH/REPLACE edits + git | 本地 | 任意 |
| OpenHands | Web + CLI | CodeAct + multi-agent | event stream + 历史压缩 | bash/browser/ipython/editor | Docker runtime | 任意 |
| SWE-Agent | CLI | ACI (Agent-Computer Interface) | 线性历史 | 专门优化过的 file_viewer/edit | Docker | 任意 |

---

## 2. 路径 A：把 GLM-5.1 接进现成外壳

### 2.1 前置：确认你的 OpenAI-compat endpoint

假设 Phase 7 你用 `vllm` 或 `sglang` 部署：

```bash
# 例：vLLM
python -m vllm.entrypoints.openai.api_server \
  --model /models/GLM-5.1 \
  --served-model-name glm-5.1 \
  --port 8000 \
  --enable-auto-tool-choice \
  --tool-call-parser glm45 \
  --reasoning-parser glm45
```

关键点（接外壳时踩过坑的）：
1. `--enable-auto-tool-choice` 必须开，否则 Cline/Roo 发的 tools 字段会被忽略。
2. `--tool-call-parser` 指定 GLM 家族的 parser（vLLM 0.6+ 有 `glm45`，5.1 沿用同一格式；若你跑 SGLang，参数名为 `--tool-call-parser glm45`）。
3. `--reasoning-parser` 开启后，thinking 内容会被放进 `message.reasoning_content` 字段，Claude Code 和 Cline 的新版本都能识别并在 UI 上折叠显示。
4. 如果你的 GPU 显存紧，用 `--max-model-len 65536` 手动砍上下文；大多数 agent 外壳对 128k+ 有依赖，但 64k 也能跑大多数任务。

用 `curl` 先打一个健康检查：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5.1",
    "messages": [{"role":"user","content":"ping"}],
    "max_tokens": 16
  }'
```

### 2.2 接入 Cline（VSCode）

Cline 是最早的开源 autonomous coding agent VSCode 插件（2024 年中）。选它还是 Roo/Kilo？看 2.5 对比。

**步骤**：
1. VSCode 扩展市场搜 `Cline`，安装。
2. 点左侧活动栏 Cline 图标，打开设置。
3. `API Provider` 选 **OpenAI Compatible**（不是 "OpenAI"，那个会强制走官方域名）。
4. `Base URL` 填 `http://localhost:8000/v1`。
5. `API Key` 随便填一个非空字符串（比如 `sk-local`），vLLM 默认不校验；如果你开了 `--api-key`，就填对应值。
6. `Model ID` 填 `glm-5.1`（和 `--served-model-name` 一致）。
7. `Model Configuration` 展开，勾上 `Supports Tools`、`Supports Images`（如多模态）、`Supports Prompt Caching`（GLM 目前不支持，留空）。
8. 上下文窗口手动填 `65536` 或你的 `--max-model-len`。
9. 保存。顶部聊天框输入 `list the files in src/` 测一把，看右侧 "API Request" 能否正确触发 `list_files` 工具。

**常见坑**：
- 报 "Tool use block did not match" → 80% 是 `--tool-call-parser` 没选对。
- 报 "max_tokens exceeded" → Cline 默认给工具结果保留 8k，加上 system prompt 大概 15k 起步，上下文窗口 ≥ 32k 较稳。
- thinking content 渲染成正文 → 升级 Cline 到 3.0+，且 endpoint 返回 `reasoning_content`。

### 2.3 接入 Roo Code

Roo Code 是 Cline 的 fork（早期叫 Roo-Cline），主要增强：
- Custom Modes：除了 Code，还有 Architect、Ask、Debug、Orchestrator，每个 mode 可定义自己的 system prompt 和可用工具集。
- 更细的 approval 控制（每个工具独立白名单）。
- MCP (Model Context Protocol) 服务器支持更成熟。

**配置过程几乎和 Cline 一致**，唯一差别：
- 设置入口在 Roo 的齿轮图标里。
- 推荐先把 "Orchestrator" mode 关掉，避免它拉起 sub-task 吃掉额外 token。
- 如果你要接 MCP（比如 git、filesystem MCP server），Roo 的 UI 比 Cline 友好。

### 2.4 接入 Kilo Code

Kilo Code 是 Cline + Roo 的再 fork（2025 年），主打：
- UI 更简洁。
- 把 Cline 和 Roo 的各自最佳 feature 合并：Cline 的 Plan mode + Roo 的 Custom modes。
- 内置 marketplace 可一键装 prompt 模板。

配置同 Cline，略。

### 2.5 接入 Claude Code（通过自定义 endpoint）

官方 Claude Code 默认走 Anthropic API。要把它指到 GLM-5.1，有两条路：

**方案一：LiteLLM proxy（推荐）**
LiteLLM 能把 Anthropic `/v1/messages` 协议转成 OpenAI 协议：

```bash
pip install 'litellm[proxy]'
# 写 config.yaml
cat > config.yaml <<'EOF'
model_list:
  - model_name: claude-3-5-sonnet-20241022  # 名字要撞 Claude Code 认的
    litellm_params:
      model: openai/glm-5.1
      api_base: http://localhost:8000/v1
      api_key: sk-local
EOF
litellm --config config.yaml --port 4000
```

然后在启动 Claude Code 前设环境变量：
```bash
export ANTHROPIC_BASE_URL=http://localhost:4000
export ANTHROPIC_AUTH_TOKEN=sk-local
claude
```

**方案二：claude-code-router / claude-relay 类开源工具**
社区有多个 relay（`musistudio/claude-code-router`、`claude-bridge` 等），做法类似但对模型切换更友好。

**注意**：Claude Code 的 prompt 严重依赖 Claude 的 tool calling 格式（尤其 `str_replace_editor` 的 diff 风格）。GLM-5.1 能 follow，但成功率比 Claude 低 10-20 个百分点；重任务建议用 Cline/Roo 这种为任意模型设计的外壳。

### 2.6 各外壳能力差异对比

| 维度 | Claude Code | Cline | Roo Code | Kilo Code | Aider |
|---|---|---|---|---|---|
| IDE 插件 | VSCode+CLI | VSCode | VSCode | VSCode | 无 (CLI) |
| 自定义 modes | 无（有 subagents） | 无（Plan/Act） | 有 | 有 | 无 |
| 开源 | 部分 | 是 | 是 | 是 | 是 |
| 接本地模型 | 需 relay | 原生 | 原生 | 原生 | 原生 |
| 工具粒度 | 细 (8+) | 细 (10+) | 细 (10+ + MCP) | 细 + marketplace | 粗 (edit + shell) |
| Diff 风格 | str_replace (unified) | SEARCH/REPLACE | SEARCH/REPLACE | SEARCH/REPLACE | SEARCH/REPLACE / diff |
| Repo map | CLAUDE.md + glob | 环境详情 | 环境详情 | 环境详情 | tree-sitter repo-map (顶尖) |
| 上下文压缩 | auto-compact (好) | 手动 | 手动 + 配置 | 手动 + 配置 | 滚动（每轮只发相关文件） |
| 沙箱 | 权限提示 | 权限提示 | 细粒度 auto-approve | 同 Roo | 本地 (git 保护) |
| 适合谁 | 想要即开即用 | 通用首选 | 爱折腾工作流 | 前两者的平替 | 对 git workflow 洁癖 |

**给这位研究者的推荐**：**路径 A 用 Roo Code + LiteLLM + GLM-5.1**。理由：研究者通常要对比不同模型 / 切换 mode，Roo 的 custom modes + mcp 支持让实验成本最低；又不像 Kilo 那样迭代还不稳定。

---

## 3. 路径 B：自建 minimal coding agent 的核心组件

### 3.1 Tool schema（OpenAI 兼容）

任何 agent 的第一步是定义工具。用 JSON Schema，直接塞进 OpenAI `tools` 字段：

```python
TOOLS = [
  {
    "type": "function",
    "function": {
      "name": "read_file",
      "description": "Read a file from the sandbox. Returns content with line numbers.",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {"type": "string", "description": "Absolute path"},
          "start_line": {"type": "integer", "default": 1},
          "end_line": {"type": "integer", "default": 2000}
        },
        "required": ["path"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "write_file",
      "description": "Write/overwrite a file. Use for new files or full rewrites.",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {"type": "string"},
          "content": {"type": "string"}
        },
        "required": ["path", "content"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "edit_file",
      "description": "Replace exactly ONE occurrence of old_str with new_str in path.",
      "parameters": {
        "type": "object",
        "properties": {
          "path": {"type": "string"},
          "old_str": {"type": "string"},
          "new_str": {"type": "string"}
        },
        "required": ["path", "old_str", "new_str"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "bash",
      "description": "Run a shell command in sandbox. 60s timeout.",
      "parameters": {
        "type": "object",
        "properties": {
          "cmd": {"type": "string"},
          "cwd": {"type": "string", "default": "/workspace"}
        },
        "required": ["cmd"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "grep",
      "description": "ripgrep in sandbox. Returns matching lines with file:line prefix.",
      "parameters": {
        "type": "object",
        "properties": {
          "pattern": {"type": "string"},
          "path": {"type": "string", "default": "/workspace"}
        },
        "required": ["pattern"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "finish",
      "description": "Call when task is complete with a final message to user.",
      "parameters": {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"]
      }
    }
  }
]
```

**设计原则**：
- 描述里写行为的 *后置条件* 和 *副作用*，LLM 会更准确地选择。
- `edit_file` 要求 `old_str` 唯一匹配——这是 Claude Code 和 Aider 都吃过的亏，两次 diff 失败率高 30%+ 来自模糊匹配。
- `bash` 不要加太多安全正则黑名单；沙箱隔离才是正道。

### 3.2 Sandbox：Docker per-session vs E2B

**Docker per-session（本地推荐）**：

```python
import docker, tarfile, io

class DockerSandbox:
    def __init__(self, image="python:3.11-slim", workdir="/workspace"):
        self.client = docker.from_env()
        self.workdir = workdir
        self.container = self.client.containers.run(
            image,
            command="sleep infinity",
            detach=True,
            working_dir=workdir,
            network_mode="bridge",   # 或 "none" 断网
            mem_limit="2g",
            nano_cpus=2_000_000_000, # 2 cpus
            tmpfs={"/tmp": "size=512m"},
            remove=False,
        )
        self.exec(f"mkdir -p {workdir}")

    def exec(self, cmd, timeout=60, cwd=None):
        cwd = cwd or self.workdir
        result = self.container.exec_run(
            ["bash", "-lc", cmd],
            workdir=cwd,
            demux=True,
        )
        stdout, stderr = result.output
        return {
            "exit_code": result.exit_code,
            "stdout": (stdout or b"").decode("utf-8", "replace"),
            "stderr": (stderr or b"").decode("utf-8", "replace"),
        }

    def put(self, path, content: str):
        data = content.encode()
        tar_buf = io.BytesIO()
        with tarfile.open(fileobj=tar_buf, mode="w") as tar:
            info = tarfile.TarInfo(name=path.lstrip("/"))
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        tar_buf.seek(0)
        self.container.put_archive("/", tar_buf.read())

    def close(self):
        self.container.kill()
        self.container.remove()
```

**E2B（云端推荐）**：

```python
from e2b_code_interpreter import Sandbox
sbx = Sandbox()                 # ~200ms 启动
sbx.files.write("/code/app.py", "print('hi')")
res = sbx.commands.run("python /code/app.py")
print(res.stdout)
sbx.kill()
```

**对比**：

| 维度 | Docker | E2B | Firecracker 自建 |
|---|---|---|---|
| 冷启动 | 1-3s | 150-300ms | 125ms |
| 隔离强度 | 容器 (cgroups/namespace) | microVM | microVM |
| 成本 | 本地免费 | ~¥0.3/小时 | 自运维 |
| 持久化 | docker volume | 自带 filesystem snapshot | 自己做 |
| 并发 | 10-50 个 | 数千 | 数千 |
| 适合 | 本地实验、CI | 产品化、云 SaaS | 有 infra 团队 |

研究阶段无脑选 Docker。

### 3.3 Repo 索引

三层索引，按任务难度递增：

**(a) 目录 + 文件头（最轻）**
每个文件读前 50 行拼出 "结构感"；对小 repo 够用。

**(b) tree-sitter AST → 签名地图（Aider 风格）**
```python
from tree_sitter_languages import get_parser
parser = get_parser("python")

def extract_signatures(path: str) -> list[str]:
    src = open(path, "rb").read()
    tree = parser.parse(src)
    sigs = []
    def walk(node, depth=0):
        if node.type in ("function_definition", "class_definition"):
            header = src[node.start_byte:node.children[-1].start_byte].decode()
            sigs.append(header.splitlines()[0])
        for ch in node.children:
            walk(ch, depth+1)
    walk(tree.root_node)
    return sigs
```

然后对每个源文件生成：
```
# src/parser.py
class Lexer:
    def tokenize(self, src: str) -> list[Token]:
def parse(tokens: list[Token]) -> AST:
```

拼起来就是 Aider 的 "repo map"。Aider 还做了 PageRank —— 根据引用次数排序，低 rank 的文件只保留类签名，高 rank 的保留方法签名。

**(c) Embedding 检索**
- 模型：`BAAI/bge-code-v1`（code-specialized，中英文友好）、CodeBERT、或通用 `bge-m3`。
- 切片：按函数切（tree-sitter），不要按字符滑窗 —— 结果质量差一个数量级。
- 向量库：本地用 chromadb / sqlite-vec；生产用 qdrant。
- 触发时机：agent 发出 `semantic_search("implementation of auth middleware")` 工具调用时。

**(d) LSP 融合（进阶）**
接 pylsp / clangd / rust-analyzer 拿 goto-def、find-references。工具粒度最细，但配置繁琐。SWE-Agent 最新版、Continue.dev 都用了。

### 3.4 规划循环

**ReAct 模板**（最小可用）：

```
System: You are a coding agent. Use tools to solve the user task.
When you need information, call a tool. When done, call finish().

User: <task>

Assistant (round 1):
  (maybe thinking...)
  tool_call: grep(pattern="def login")
Tool:
  src/auth.py:42: def login(username, password):
Assistant (round 2):
  tool_call: read_file(path="src/auth.py", start_line=30, end_line=80)
Tool:
  ...
Assistant (round N):
  tool_call: finish(summary="Fixed null-check in login()")
```

**Plan-Execute（OpenHands / Cline Plan mode）**：

```
Phase 1 (Plan):
  User task → LLM produces numbered plan
  1. Read src/auth.py
  2. Locate login()
  3. Add null check
  4. Add unit test
  5. Run pytest

Phase 2 (Execute):
  For each step:
    ReAct sub-loop until step done
    After each step, LLM judges "done/continue/replan"
```

优点：长任务不跑偏；缺点：多一层 token 开销，简单任务过度设计。

**Reflexion（失败时补救）**：

```
if run_tests().exit_code != 0:
    reflection = llm.complete(
        f"Task: {task}\n"
        f"Actions I took: {action_history}\n"
        f"Test output: {test_output}\n"
        f"Write a short reflection on what went wrong and what I should try differently."
    )
    history.append({"role":"system","content": f"Reflection: {reflection}"})
    # restart ReAct loop
```

Reflexion 原论文在 HumanEval 上把 GPT-4 pass@1 从 80% 推到 91%，对固定预算下的任务尤其有用。

### 3.5 上下文压缩

三层策略，越靠上越激进：

1. **工具结果截断**（每次工具调用后立即做）
   ```python
   def truncate(s, max_chars=8000):
       if len(s) <= max_chars:
           return s
       head = s[: max_chars // 2]
       tail = s[-max_chars // 2 :]
       return f"{head}\n... <truncated {len(s)-max_chars} chars> ...\n{tail}"
   ```
   `bash` 的 stdout、`read_file` 的内容都过这个滤。

2. **历史摘要（auto-compact）**
   ```python
   if total_tokens(messages) > 0.8 * CTX_WINDOW:
       summary = llm.complete(
           "Summarize this conversation preserving: files touched, "
           "decisions made, open TODOs. Markdown, <500 words.",
           messages=messages[:-6]  # 保留最近 6 轮原文
       )
       messages = [
           messages[0],  # system
           {"role":"system","content": f"[Prior context summary]\n{summary}"},
           *messages[-6:]
       ]
   ```
   Claude Code 的实现几乎一模一样，阈值是 "剩余 context < 20%" 时触发。

3. **文件系统作为外部记忆**
   Agent 把中间产物写 `/workspace/.agent/notes/*.md`，下一步自己 read 回来。这绕开了 context 长度限制，把 LLM 当成可寻址的 "CPU"，文件系统是 "内存"。Devin、OpenHands 都是这套逻辑。

### 3.6 长期记忆

区分 session 内 vs. 跨 session：

- **Session 内**：滚动摘要 + 文件系统笔记，上面讲过了。
- **跨 session**：
  - 简单方案：`~/.agent/memory.md`，每次 finish 时 append "学到的项目事实"，新 session 开场 read 进去。这就是 Claude Code 的 `CLAUDE.md` 机制。
  - 向量方案：每条记忆 embed 入库，prompt 时按当前任务 top-k 检索。对极大 repo 必要，小项目反而是噪音。

---

## 4. 关键设计选择

### 4.1 Single-agent vs Multi-agent

| 选择 | 适用 | 代表 | 代价 |
|---|---|---|---|
| Single | 任务边界清晰、工具少 | SWE-Agent | 长任务易漂 |
| Multi (manager + worker) | 跨领域任务（写代码 + 浏览 + 数据分析） | OpenHands、AutoGen、MetaGPT | Token 翻倍、调试难 |

经验法则：**工具数 ≤ 10、单任务 < 50 轮，就用 single-agent**。绝大多数编码任务落在这个区间。

### 4.2 同步 vs 异步工具调用

OpenAI 协议在一条 assistant message 里可返回多个 tool_calls（并行），也可以一个一个来。

- GLM-5.1 支持并行 tool calling。
- 何时用并行：`read_file(a)` + `read_file(b)` + `grep(...)` 明显独立时。
- 何时用串行：`write_file` 之后的 `run_tests`，必须等前者完成。
- 实现：每轮收到 `tool_calls: [t1, t2, t3]` 就 `asyncio.gather` 三个 runner，然后把三个 `tool` 消息一起附加回 `messages`。

### 4.3 Thinking tokens 怎么用

GLM-5.1 的 reasoning mode 会吐 `<think>…</think>`，通过 `reasoning_content` 字段返回。

- **要不要给 agent 用 thinking**：复杂规划（planning 阶段、Reflexion 阶段）开；单纯调工具（工具选择）关 —— 否则每轮多 1-3k token，又贵又慢。
- **写法**：`extra_body={"chat_template_kwargs": {"enable_thinking": true}}`（vLLM + GLM 的约定），或直接在 system prompt 里写 "Use <think> tags to plan first"。
- **历史里保留 thinking 吗**：**不保留**。把 reasoning_content 丢弃，只留 action（tool_calls）和 final text。原因：历史里留 thinking 会引导模型把"说"当成"做"，且浪费 context。Claude 官方 Extended Thinking 文档也是这个规范。

---

## 5. 完整 minimal coding agent 实现（< 300 行）

以下代码可以直接跑。依赖：`pip install openai docker tree-sitter-languages`。

<!-- include: examples/phase8/mini_agent.py -->


运行：

```bash
export LLM_BASE_URL=http://localhost:8000/v1
export LLM_MODEL=glm-5.1
python mini_agent.py
```

**期望输出**（示意）：

```
[step 0] → bash({'cmd'})
            exit=0 stdout: ...
[step 1] → read_file({'path'})
            1 | def add(a, b): ...
[step 2] → edit_file({'path', 'old_str', 'new_str'})
            OK: applied edit
[step 3] → run_tests({'path'})
            ===== 2 passed in 0.02s =====
[step 4] → finish({'summary'})
=== FINISHED: Fixed add() to use + instead of -.
```

**这 300 行能展示的核心机制**：OpenAI tool calling、Docker sandbox、6 个原子工具、ReAct 循环、输出截断。缺的（课后题）：repo map、历史压缩、Reflexion、并行 tool call、多文件大 diff patch。加上大概要到 800-1000 行。

---

## 6. 评测你的 agent：SWE-Bench Lite

SWE-Bench 是现在事实上的 coding agent 标尺。Lite 子集是 300 个真实 Python bug（来自 12 个知名 OSS 项目），任务是把给定 repo 改到能通过隐藏测试。

**跑法**：
1. `pip install swebench`。
2. 用官方 `swebench.harness.run_evaluation` 的 Docker runner。每条样本官方给一个 Docker image。
3. 把你的 agent 包成一个 callable：输入 `(repo_path, problem_statement)`，输出一个 git diff patch。
4. 评测器会 apply patch 并跑测试。

**研究型 baseline 参考**（2024-2025）：
- Aider + Claude 3.5 Sonnet：~27% Lite。
- SWE-Agent + GPT-4：~18% Lite。
- OpenHands + Claude 3.5：~40%+ Lite。
- 纯 GLM-4.5-Air 走你的 minimal agent：大概 10-15%（没做 repo map / 反思）。
- 加上 repo map + Reflexion + plan-execute：可以到 20-25%。

**最有教益的做法**：先跑 10 条样本人肉看轨迹，找失败模式；大概率你的 agent 会死在 "同一个 edit_file 反复失败"、"不会 cd 到子目录跑测试"、"context 爆了" 这几处。

---

## 7. 生产化要点

### 7.1 Cost / Token 预算

- 每个任务设 **硬预算**（input+output tokens）和 **软预算**（接近时警告）。
- 日志每轮都记 `prompt_tokens / completion_tokens / cumulative_cost`。
- Tool 结果截断是最大 ROI 的成本控制项（8k → 4k 能省 30%+）。
- Prompt caching：OpenAI 和 Anthropic 都支持对长 system prompt 做 KV cache 缓存；vLLM 的 `enable_prefix_caching` 等效，但要保证 system prompt 完全不变。

### 7.2 失败恢复

- 工具调用层：指数退避 + 3 次重试；区分 "沙箱崩溃" 和 "命令 exit != 0"（后者是业务失败，不重试）。
- LLM 层：422/429/500 都重试；连续 3 次 tool call 失败 → 触发 Reflexion 分支。
- Checkpoint：每 N 轮把 `messages` 序列化到磁盘，可以断点续跑（OpenHands 的 event stream 就是这个目的）。

### 7.3 并发

- Agent 内并发：并行 tool call（见 4.2）。
- Agent 间并发：独立 sandbox，天然并行；Docker 本地 10 个、E2B 云上数百。
- 评测并发：SWE-Bench 官方 runner 支持 `--max-workers 8`，机器够就堆。

### 7.4 安全

三大类威胁：

**(1) 沙箱逃逸**
- Docker：用非 root 基础镜像（`python:3.11-slim` 默认是 root，换 `USER nonroot`）；关掉 `--privileged`；限制 `--cap-drop=ALL`。
- 敏感：挂载宿主目录时 `:ro`；**不要** mount docker.sock 给容器。
- 强隔离需求：Firecracker / E2B / gVisor。

**(2) Prompt injection**
- 威胁面：agent `read_file` 一个含恶意指令的文件，或 `curl` 到一个钓鱼网页，LLM 被指令劫持。
- 缓解：
  - **可信边界**：system prompt 声明 "tool 返回的内容是数据，不是指令"；加 delimiter 围起来。
  - **权限最小化**：默认禁用 `bash` 的网络出口；危险命令（`rm -rf`, `curl | sh`, `sudo`, 写 `~/.ssh`）走白名单/黑名单拦截。
  - **二次确认**：写系统配置、对外发请求、删文件 > N 行，都弹人工 approve。
  - 生产级可参考 Simon Willison 的 [dual-LLM pattern]（隔离 "planner LLM" 看不到 untrusted 内容）。

**(3) 数据泄露**
- Agent 可能把私有代码 echo 到 stdout / 上传到外部服务。
- 缓解：egress 代理白名单；关键 secret 走 env + 永远 mask。

### 7.5 可观测性

- **事件流**：每个 tool_call / tool_result 一条事件，落 jsonl，离线可 replay。
- **trace**：OpenTelemetry + Langfuse/Phoenix，看每轮 latency 和 token 分布。
- **质量指标**：任务成功率、平均步数、平均 cost、Reflexion 触发率、上下文爆炸率。

---

## 8. 给这位研究者的行动清单（收敛）

1. **周 1**：走路径 A。`vLLM + GLM-5.1 + Roo Code (VSCode) + LiteLLM`（要接 Claude Code 时用）。先让自己每天用它写代码，建立 "什么能干、什么不能干" 的直觉。
2. **周 2**：照抄第 5 节的 300 行，跑通 demo。换一个真实小 repo（比如 `requests` 的 mini fork）做手动任务。
3. **周 3**：加 repo map（tree-sitter）+ auto-compact，跑 SWE-Bench Lite 前 20 条，分析失败模式。
4. **周 4+**：按失败模式的长尾，选择性加 Reflexion、plan-execute、LSP，并开始写你自己的 paper-ready 对比实验。

---

## 9. 参考资料

- Claude Code docs: https://docs.claude.com/en/docs/claude-code
- Cline: https://github.com/cline/cline
- Roo Code: https://github.com/RooCodeInc/Roo-Code
- Kilo Code: https://github.com/Kilo-Org/kilocode
- Aider + repo-map blog: https://aider.chat/docs/repomap.html
- OpenHands: arxiv 2407.16741；github.com/All-Hands-AI/OpenHands
- SWE-Agent: arxiv 2405.15793；github.com/SWE-agent/SWE-agent
- SWE-Bench: arxiv 2310.06770；swebench.com
- ReAct: arxiv 2210.03629
- Reflexion: arxiv 2303.11366
- Plan-and-Execute (LangChain): https://blog.langchain.dev/planning-agents/
- E2B: https://e2b.dev
- Firecracker: https://firecracker-microvm.github.io
- tree-sitter: https://tree-sitter.github.io
- bge-code-v1: https://huggingface.co/BAAI/bge-code-v1
- smol-developer: https://github.com/smol-ai/developer
- LiteLLM proxy: https://docs.litellm.ai/docs/proxy/quick_start

---

## 10. 企业场景扩展：基于私有代码的 RAG 系统

Phase 1 §0.6 提过私有代码入模的三条路线：继续预训练 / SFT / RAG。**钱少想快见效**的就先做 RAG——不动模型权重、一周内能上线、增量更新便宜。但代码 RAG 和"把 PDF 切成 512 tokens 丢进 Chroma"这种文档 RAG 是两回事，下面把坑一条条拆开。

### 10.1 为什么代码 RAG 不能照抄文档 RAG

三个根本差别，每一个都能把一个抄教程搭起来的 RAG 卡死：

**(1) 语义单元不是段落，是函数/类**
文档 RAG 里段落是天然语义单元，500 token 切一刀问题不大。代码里一个函数可能 20 行也可能 800 行（想想 Django 里一些 `View.dispatch`）。固定窗口切块会把 `def foo():` 的签名和 body 切到两个 chunk 里，embed 出来谁都搜不到。**必须用 AST 感知的切块。**

**(2) 跨文件引用是一等信号**
问 "UserService 怎么做 auth？" 真正的答案可能是：
```
services/user.py 里 UserService.authenticate → 
  auth/jwt.py 里 verify_token → 
  models/user.py 里 User.check_password
```
纯向量检索只会返回名字里含 `authenticate` 的那一个 chunk，**call graph 信号完全丢了**。要么在 chunk metadata 里塞 `imports` / `called_by` 字段，要么上 GraphRAG。

**(3) Query 和语料跨模态**
用户 query 是自然语言："怎么加一个限流中间件？"，但语料是 Python 代码。`text-embedding-3-large` 这种通用 embedding 模型在跨模态上性能会掉 20-40%。**必须用 code-aware embedding**（bge-code-v1 / voyage-code-2 这类训练集里有 NL-code pair 的）。

### 10.2 系统架构

```mermaid
flowchart LR
  subgraph OFF["离线索引 · 几小时一次"]
    direction TB
    REPO["私有代码仓库"] --> SP["tree-sitter 切块<br/>按函数 / 类"]
    SP --> EMB["bge-code-v1 embedding"]
    EMB --> META["注入元数据<br/>path / team / mtime"]
    META --> VDB["向量库<br/>Qdrant + BM25"]
  end
  subgraph ONL["在线查询 · 毫秒级"]
    direction TB
    Q["用户 query"] --> REW["Query 改写 · HyDE"]
    REW --> RET["Hybrid Search<br/>vec + bm25 + 权限 filter"]
    RET --> TOP50["top-50 候选"]
    TOP50 --> RR["bge-reranker-v2"]
    RR --> TOP5["top-5 精选"]
    TOP5 --> GEN["GLM-5.1 生成<br/>+ 源引用归因"]
    GEN --> ANS["答案"]
  end
  VDB -.检索.-> RET
```


```
┌────────────────── 离线索引（几小时一次） ──────────────────┐
│                                                              │
│  git repo ──▶ tree-sitter 切块 ──▶ 元数据注入 ──▶ embedding │
│                (函数/类级别)        (path, lang,    │        │
│                                      team, mtime)   ▼        │
│                                                 ┌──────────┐ │
│                                                 │ 向量库    │ │
│                                                 │ Qdrant   │ │
│                                                 │ + BM25   │ │
│                                                 └──────────┘ │
└──────────────────────────────────────────────────────────────┘
                                                      ▲
┌────────────────── 在线查询（毫秒级） ──────────────│─────────┐
│                                                    │         │
│  user query ──▶ query 改写（HyDE/扩展）──▶ hybrid │         │
│                                            search │         │
│                                            (vec + │         │
│                                             bm25 + │         │
│                                             filter)│         │
│                                               │    │         │
│                                               ▼              │
│                                          top-50 候选         │
│                                               │              │
│                                               ▼              │
│                                        bge-reranker-v2       │
│                                               │              │
│                                               ▼              │
│                                          top-5 snippets      │
│                                               │              │
│                                               ▼              │
│                                   prompt 拼接（+ cite）      │
│                                               │              │
│                                               ▼              │
│                                         GLM-5.1 / Claude     │
│                                               │              │
│                                               ▼              │
│                                    带引用的回答              │
└──────────────────────────────────────────────────────────────┘
```

关键：**索引和查询是两个独立 pipeline**，索引慢一点没事（离线 batch），查询要 P99 < 1s。

### 10.3 切块策略对比

| 策略 | 实现成本 | 语义完整性 | 召回率 | 何时用 |
|------|----------|------------|--------|--------|
| 固定 token 窗口（512/1024） | 最低，10 行 | 差，切函数 | 低 | Demo 快速验证 |
| 滑动窗口 + 重叠（overlap=128） | 低 | 中等 | 中 | 文档混代码场景 |
| tree-sitter 按函数/类 | 中等，每语言一套 query | **高** | 高 | **生产首选** |
| tree-sitter + 父类/文件摘要前缀 | 中等 | 很高 | 很高 | 大 repo、层级深 |
| GraphRAG（实体+关系图） | 高，需 LLM 抽取实体 | 最高 | 最高 | 跨文件推理、长尾 query |
| 每文件一个 chunk | 低 | 看文件长度 | 小文件 OK | 脚本堆（非 OOP） |

**推荐起步方案**：tree-sitter 按 `function_definition / class_definition / method_definition` 切，每个 chunk 携带：
- `file_path`（绝对路径或仓库相对路径）
- `lang`（python / ts / go ...）
- `symbol`（函数/类名）
- `docstring`（如果有）
- `signature`（函数签名，有时比 body 更能命中自然语言 query）
- `imports`（top-of-file 的 import 列表，跨文件线索）
- `span`（起止行号，用于回答时做 cite）
- `team` / `visibility`（权限，关键！）

切块时有两个小 trick 能显著提升效果：
1. **prepend 文件路径和类名**：把 `# services/user.py → class UserService` 放到 chunk 文本最前面一起 embed，这样 embedding 里就带了路径语义。
2. **split oversize**：如果一个函数超过 1500 tokens（比如 Django migration），再用滑动窗口二次切一下。

### 10.4 Embedding 模型选型

| 模型 | 类型 | 维度 | 上下文 | NL-code 对齐 | 中文注释 | 成本 |
|------|------|------|--------|--------------|----------|------|
| **bge-code-v1** (BAAI) | 开源 | 1536 | 32k | 强 | **强（中英双语）** | 自部署 GPU |
| Jina-embeddings-v3 | 开源/API | 1024（Matryoshka 可裁） | 8k | 中强 | 强（89 语言） | 自部署或 API |
| voyage-code-2 | 闭源 API | 1536 | 16k | **最强** | 中 | $0.12/1M tokens |
| voyage-code-3 | 闭源 API | 1024/2048 | 32k | 最强 | 中 | $0.18/1M tokens |
| OpenAI text-embedding-3-large | 闭源 API | 3072（可裁到 256） | 8k | 中 | 中 | $0.13/1M tokens |
| UniXcoder | 开源 | 768 | 512 | 中（老） | 弱 | 自部署 |
| CodeBERT | 开源 | 768 | 512 | 中（老） | 弱 | 自部署 |

**给这位中国研究者的推荐：bge-code-v1**。理由：
1. 中英双语，私有 repo 里注释混中文的情况它能 handle，OpenAI 和 voyage 都偏英文。
2. 开源，合规没问题，私有代码不出内网。
3. 维度 1536 刚好——比 UniXcoder 的 768 更有区分度，比 text-embedding-3-large 的 3072 省一半显存。
4. 32k context 够长，整个函数几乎不会超。

**不推荐**：UniXcoder / CodeBERT 虽然是经典，但 2020-2022 年的训练集，对 Python 3.10+ / TypeScript 新语法、现代框架（FastAPI、Next.js App Router）基本没见过，效果比新模型差一档。

部署 bge-code-v1 可以用 vLLM 或 TEI（text-embeddings-inference），TEI 更省显存，一张 4090 能跑 2k QPS。

### 10.5 向量库深度对比

| 向量库 | 部署 | QPS (1kw 向量) | Filter | 混合搜索 | 多租户 | 内存 | 适用 |
|--------|------|----------------|--------|----------|--------|------|------|
| **Qdrant** | Docker 一把梭 | 2-5k | **极强**，payload 索引 | 有（内置 BM25 from 1.10） | collection 或 payload | 中 | **生产首选** |
| Milvus | 云原生，etcd+minio | 5-10k | 强 | 2.4+ 有 | 有 database 概念 | 高 | 十亿级、已有 k8s |
| Weaviate | Docker | 3-6k | 中 | **内置 hybrid** | 有 | 中 | 想要开箱即用 hybrid |
| pgvector | postgres 插件 | 500-2k | 强（SQL） | 要自己拼 tsvector | schema 隔离 | 低 | **已有 PG 的团队** |
| LanceDB | 嵌入式（无服务） | 1-3k | 中 | 有 | 无 | **极低** | **单机 POC** |
| Chroma | Python 库 | < 1k | 弱 | 弱 | 无 | 低 | Dev only |

**对这位用户的推荐路径**：
- **第 1 周 POC**：LanceDB。一行 `pip install lancedb`，不需要起服务，pandas 友好，demo 给老板看刚好。
- **第 1 个月上线**：Qdrant。Docker 一起，payload filter 对"按 team / 按 repo / 按语言过滤"非常友好，这是权限系统的基础。
- **规模超 5000 万向量 / 多部门共用**：迁 Milvus。有 k8s 运维预算再上。

**Chroma 警告**：很多教程用 Chroma 是因为它最简单，但它的 filter 是 post-filter（先取 top-k 再过滤），碰到"我只想搜 team-a 的代码"时如果 team-a 占比很小，filter 后可能返回空。**生产不要用 Chroma**。

### 10.6 混合检索：向量 + BM25 + Filter

**纯向量检索会死在这些 query**：
- 精确 token 匹配：搜 `CVE-2024-12345` 向量基本没用，BM25 完胜。
- 函数名搜索：搜 `parseJSON` 向量可能返回一堆"解析 JSON"的相关代码，BM25 直接命中该函数。
- 短 query：`OAuth2` 三个 token 的向量很不稳定，BM25 更鲁棒。

**Hybrid 公式（RRF，Reciprocal Rank Fusion）**：
```
score(doc) = Σ  1 / (k + rank_i(doc))       # k=60 常用
         i∈{vec, bm25}
```
Qdrant 1.10+、Weaviate、Milvus 2.4+ 都内置了。

**Metadata filter 是必须的**：
```python
# Qdrant 例子
client.search(
    collection_name="code",
    query_vector=q_vec,
    query_filter=Filter(
        must=[
            FieldCondition(key="team", match=MatchAny(any=user.teams)),
            FieldCondition(key="lang", match=MatchValue(value="python")),
        ]
    ),
    limit=50,
)
```

### 10.7 Reranker 层

检索召回要的是 recall（宁多勿漏），reranker 要的是 precision。典型 pipeline：

```
hybrid search top-50 (粗排, ~50ms)  →  reranker top-5 (精排, ~200ms)
```

| Reranker | 类型 | 延迟 | 效果 | 成本 |
|----------|------|------|------|------|
| **bge-reranker-v2-m3** | 开源，cross-encoder | 200ms / 50 pair | 强 | GPU 自部署 |
| Cohere Rerank 3 | API | 100ms | 强 | $2 / 1k search |
| Jina Reranker v2 | 开源/API | 200ms | 中强 | 同上 |
| GPT-4o-mini as reranker | LLM | 1-3s | 强，但慢 | token 贵 |

**何时不值得加 reranker**：top-5 召回已经足够好（NDCG@5 > 0.8）、延迟预算紧（< 500ms）。**何时必须加**：repo 里有大量重名/相似函数（比如每个模块都有 `get_config`），纯向量分不清。

### 10.8 Query 改写

自然语言 query 和代码之间有 gap，改写能显著提升召回：

**(1) HyDE (Hypothetical Document Embeddings)**
让 LLM 先**伪造一个答案**（一段假想的代码），再用这段伪造代码去搜。原理：代码 vs 代码的向量对齐比 NL vs 代码更好。
```python
# 原 query: "怎么给 FastAPI 加 rate limit？"
hyde_code = llm("生成一段可能的实现代码，不超过 20 行")
# hyde_code ≈ "from slowapi import Limiter\nlimiter = Limiter(...)..."
search(embedding(hyde_code))
```

**(2) Multi-query expansion**
让 LLM 把一个 query 拆成 3-5 个子 query，分别检索后合并。
```
原：怎么加限流中间件？
拆：
  - rate limit middleware FastAPI
  - slowapi Limiter 用法
  - 限流 装饰器 Python
```

**(3) Query → 函数签名**（代码专属）
```
query: "把字符串转成 snake_case"
改写: "def to_snake_case(s: str) -> str:"
```
然后搜这个签名。对精确查找效果非常好。

### 10.9 权限过滤（Access-Aware Retrieval）

**最容易被忽视的生产化坑**：不要在 prompt 层说 "你只能看 team-a 的代码"。LLM 没有真·权限，prompt injection 一下就穿透。

**必须在检索层就过滤**：
1. 索引时每个 chunk 打 `team`/`visibility`/`repo` 标签。
2. 查询时强制注入 filter：`team IN current_user.teams`。
3. Filter 走向量库的 payload 索引，不是 post-filter。

```python
# 危险（prompt 层控制）❌
prompt = f"你只能看 {user.team} 的代码，query: {q}"
docs = search(q, top=50)  # 检索到别人的代码
```

```python
# 正确（检索层过滤）✅
docs = search(q, top=50, filter={"team": {"$in": user.teams}})
prompt = f"context: {docs}\nquery: {q}"
```

进一步：
- **审计日志**：每次检索把 user、query、返回 chunk 的 id 记下来，方便事后追溯。
- **collection 隔离**：极度敏感的代码（金融核心、密钥管理）干脆放独立 collection，跨 collection 搜索需要显式授权。

### 10.10 上下文拼接 & 引用

基础模板：
```
你是代码助手。基于下面从本仓库检索到的片段回答问题。
每条引用都标注了 file:line，回答时请按 [1] [2] 方式给出来源。

[1] services/user.py:42-78
```python
{snippet_1}
```

[2] auth/jwt.py:10-35
```python
{snippet_2}
```

问题：{user_query}
```

**长上下文模型（GLM-5.1 200K / Claude 1M）**的机会：
- 可以把 top-20 而不是 top-5 塞进去，让模型自己筛。
- 可以顺便附上 **repo skeleton**（tree-sitter 生成的 repo map，`aider` 风格）给模型全局观。
- 但注意：即使 200K 窗口，超过 32K 之后精度也会下降（lost in the middle），关键 snippet 放最前或最后。

**引用归因（citation）**非常重要：
- 没引用的答案 → 用户不信任。
- 有 `file:line` 引用 → 用户点进去就能验证，信任度翻倍。
- 实现：每个 chunk 带 `(file_path, start_line, end_line)`，prompt 里强制要求"每个断言后面加 [file:line]"。

### 10.11 增量更新

全量 re-index 一次 500 万 chunk × 1024 维 embedding，在一张 A100 上大概 4-6 小时。天天重建太贵，生产必须做增量：

**Git webhook 触发**：
```python
# github webhook / gitlab webhook
@app.post("/hook/push")
def on_push(event):
    changed = event["commits"]  # [{added, modified, removed}]
    for f in changed.modified + changed.added:
        old_chunks = qdrant.scroll(filter={"file_path": f})
        qdrant.delete(ids=[c.id for c in old_chunks])
        new_chunks = tree_sitter_chunk(read(f))
        qdrant.upsert(embed(new_chunks))
    for f in changed.removed:
        qdrant.delete(filter={"file_path": f})
```

**失效粒度**：
- 文件级（最简单）：一个文件改了，删掉该文件所有 chunk，重新切重新 embed。一次 commit 改 5 个文件 → 几秒搞定。
- 函数级（精细）：用 `git diff` + tree-sitter 找出变动的函数，只 re-embed 这些函数。省 30-50% 工作量，但实现复杂。建议**上量后再优化**。

**全量节奏**：
- 月度全量重建一次：修复漂移、升级 embedding 模型、清理孤儿 chunk。
- 换 embedding 模型时必须全量（不同模型向量不可混）。

### 10.12 完整可运行 Python 脚本

下面这份 < 200 行可以直接跑，用 tree-sitter 切 + bge-code-v1 embed + Qdrant 存 + hybrid search + GLM 回答。

```python
# requirements: qdrant-client tree_sitter tree_sitter_languages \
#               sentence-transformers openai
import os, hashlib, glob
from tree_sitter_languages import get_parser
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition,
    MatchValue, MatchAny, SparseVectorParams, SparseVector,
)
from openai import OpenAI

# ---------- 1. 切块 ----------
PY_QUERY = """
(function_definition) @fn
(class_definition)    @cls
"""

def chunk_python(path: str, src: str):
    parser = get_parser("python")
    tree = parser.parse(src.encode())
    lang = parser.language
    query = lang.query(PY_QUERY)
    out = []
    for node, _name in query.captures(tree.root_node):
        text = src[node.start_byte:node.end_byte]
        if not text.strip():
            continue
        symbol = ""
        for ch in node.children:
            if ch.type == "identifier":
                symbol = src[ch.start_byte:ch.end_byte]
                break
        out.append({
            "file_path": path,
            "symbol": symbol,
            "start_line": node.start_point[0] + 1,
            "end_line":   node.end_point[0] + 1,
            "text": f"# {path} :: {symbol}\n{text}",
        })
    return out

# ---------- 2. 索引 ----------
EMBED_MODEL = SentenceTransformer("BAAI/bge-code-v1", device="cuda")
DIM = EMBED_MODEL.get_sentence_embedding_dimension()

qc = QdrantClient(url="http://localhost:6333")
COLL = "private_code"
if COLL not in [c.name for c in qc.get_collections().collections]:
    qc.create_collection(
        collection_name=COLL,
        vectors_config={"dense": VectorParams(size=DIM, distance=Distance.COSINE)},
        sparse_vectors_config={"bm25": SparseVectorParams()},
    )

def chunk_id(c): 
    return int(hashlib.md5(f"{c['file_path']}:{c['start_line']}".encode()).hexdigest()[:12], 16)

def bm25_sparse(text: str):  # 用 qdrant-client 自带的 fastembed BM25
    from fastembed import SparseTextEmbedding
    global _BM25
    _BM25 = getattr(globals(), "_BM25", None) or SparseTextEmbedding("Qdrant/bm25")
    v = next(_BM25.embed([text]))
    return SparseVector(indices=v.indices.tolist(), values=v.values.tolist())

def index_repo(root: str, team: str = "default"):
    points = []
    for fp in glob.glob(f"{root}/**/*.py", recursive=True):
        src = open(fp).read()
        for c in chunk_python(fp, src):
            dense = EMBED_MODEL.encode(c["text"]).tolist()
            sparse = bm25_sparse(c["text"])
            points.append(PointStruct(
                id=chunk_id(c),
                vector={"dense": dense, "bm25": sparse},
                payload={**c, "team": team, "lang": "python"},
            ))
        if len(points) >= 256:
            qc.upsert(COLL, points); points = []
    if points: qc.upsert(COLL, points)

# ---------- 3. 检索 ----------
def hybrid_search(query: str, user_teams, top=5):
    q_dense = EMBED_MODEL.encode(query).tolist()
    q_sparse = bm25_sparse(query)
    flt = Filter(must=[FieldCondition(
        key="team", match=MatchAny(any=user_teams))])
    res = qc.query_points(
        collection_name=COLL,
        prefetch=[
            {"query": q_dense,  "using": "dense", "limit": 50, "filter": flt},
            {"query": q_sparse, "using": "bm25",  "limit": 50, "filter": flt},
        ],
        query={"fusion": "rrf"},
        limit=top,
    )
    return res.points

# ---------- 4. 问答 ----------
llm = OpenAI(base_url="https://open.bigmodel.cn/api/paas/v4/",
             api_key=os.environ["GLM_API_KEY"])

def answer(query: str, user_teams=("default",)):
    hits = hybrid_search(query, list(user_teams), top=5)
    ctx = "\n\n".join(
        f"[{i+1}] {h.payload['file_path']}:{h.payload['start_line']}-{h.payload['end_line']}\n"
        f"```python\n{h.payload['text']}\n```"
        for i, h in enumerate(hits)
    )
    msg = [
        {"role": "system", "content": "你是私有代码助手，仅根据给定上下文回答，每个论断后用 [n] 引用来源。"},
        {"role": "user", "content": f"上下文：\n{ctx}\n\n问题：{query}"},
    ]
    r = llm.chat.completions.create(model="glm-4.6", messages=msg)
    return r.choices[0].message.content, hits

if __name__ == "__main__":
    # index_repo("/path/to/your/private/repo", team="backend")
    ans, _ = answer("怎么校验 JWT token？", user_teams=["backend"])
    print(ans)
```

把 `EMBED_MODEL` 换成 TEI HTTP client 可以进一步提速；加上一个 `bge-reranker-v2-m3` 对 top-50 精排就是完整生产版。

### 10.13 性能 & 成本估算

**规模假设**：50 万代码文件，平均每文件 10 个 chunk → **500 万 chunk**，1024 维 float32。

| 项 | 估算 |
|----|------|
| 原始向量体积 | 500w × 1024 × 4B ≈ 20 GB |
| Qdrant HNSW 实际内存 | × 1.2 ≈ **24 GB**（打开 `on_disk: true` 可降至 8 GB + SSD） |
| BM25 sparse index | ~3 GB |
| 一次全量 embed（A100 80GB, bge-code-v1） | 500w × 平均 800 token → 4-6 小时 |
| 一次全量 embed（CPU, 32 核） | ~60 小时，慢 10 倍 |
| API embed（voyage-code-3） | 500w × 800 token × $0.18/1M ≈ **$720** |
| 在线 query（dense + bm25 + rerank） | P99 ~400ms（本地 GPU） |
| 每月增量 re-index（假设 5% churn） | A100 一晚上 |

**关键优化**：
- **Matryoshka embedding**（Jina-v3 支持）可以把 1024 维存储裁成 256 维，省 75% 内存，精度损失 < 3%。
- **product quantization (PQ)**：Qdrant 支持 PQ，内存再降 4-8x，QPS 不降。
- **on-disk HNSW**：Qdrant `on_disk: true`，向量放 SSD，只缓存索引，内存压力小。

### 10.14 评测 RAG 质量

**没有 golden 问答集 = 在瞎优化**。最小化评测集：
- 50-100 条真实 query（从用户问答记录或自己编）。
- 每条标注"理想答案应引用哪几个 file:line"。

**指标**：
- **Recall@k**：理想 chunk 是否在 top-k 里。Recall@5 > 0.7 算合格。
- **MRR (Mean Reciprocal Rank)**：理想 chunk 排名倒数的平均。> 0.6 较好。
- **NDCG@10**：多个相关 chunk 时的排序质量。
- **端到端**：LLM 答案 vs 人工答案，LLM-as-judge 打分（0-5）。
- **Faithfulness**：答案里每个断言是否真的在引用的 chunk 里（用另一个 LLM 判），防幻觉。

工具：RAGAS、TruLens、LlamaIndex 自带 `RetrieverEvaluator`。

### 10.15 生产化五要点

**(1) 监控检索命中率**
每次 query 记录：返回多少 chunk、用户是否点开引用、是否追问。如果"0 引用被点开"的比例 > 30%，说明检索或回答质量有问题。

**(2) 用户反馈闭环**
回答下面放 👍/👎。负反馈单独入库，周会过一遍，归因到"切块问题 / embedding 问题 / prompt 问题 / 模型问题"。没这个闭环，RAG 系统会慢慢腐化。

**(3) 定期重 embed**
embedding 模型半年会出新版（bge-code-v1 → v1.5 → v2），benchmark 涨 5-10 点时值得全量重建。留出"影子 collection"做 A/B。

**(4) 降级策略**
- Qdrant 挂了 → 回退到本地 BM25-only（Whoosh / Tantivy）。
- Embedding 服务挂了 → 退到纯 BM25 + filter。
- LLM 挂了 → 只返回检索结果（不生成答案），用户也能用。

**(5) Cost 控制**
- Embedding 走本地 bge-code-v1，不要走 OpenAI/voyage（50 万文件全量一次几百刀，还不能复用）。
- Rerank 加缓存（相同 query hash → 相同结果，TTL 1h）。
- LLM 回答走 prompt caching（GLM 和 Claude 都支持），同一 repo context 重复使用命中率能到 80%+。

### 10.16 小结：起步路径

按这个顺序做，两周内上线第一版：

| 周 | 动作 |
|----|------|
| 第 1-3 天 | 选 10 个代表性 repo（~5 万文件），跑 tree-sitter 切块 + bge-code-v1 embed + LanceDB 存。用 50 条手写 query 测 Recall@5。 |
| 第 4-7 天 | 迁到 Qdrant，加 BM25 混合搜索，加 team/lang metadata filter，接 GLM-5.1 生成答案带 citation。 |
| 第 2 周 | 加 bge-reranker-v2、HyDE query 改写、git webhook 增量更新；搭评测集 + 👍👎 反馈闭环；上 IDE 插件或 Slack bot 给小范围用户试用。 |
| 第 3-4 周 | 根据负反馈优化（切块策略、prompt 模板、reranker 阈值），规划全公司铺开 + 权限合规审核。 |

**第一期不要做的事**（YAGNI）：GraphRAG、多模态 embedding、自训 embedding 模型、自研向量库。这些每一项都能单独耗掉 1-2 个月。先让 70 分系统跑起来，再根据 bad case 决定加哪块。

---

*本笔记配套：`phase7_model_serving.md`（上游：把 GLM-5.1 服务化）、`phase9_evaluation.md`（下游：SWE-Bench / LiveCodeBench / HumanEvalPack 系统评测）。*

---

## 📌 章末检查

**带走这 5 条**
- agent 系统 6 层：UI / 控制器 / 规划 / 工具 / 上下文 / 记忆。哪一层薄弱、整体就停在哪一层的上限。
- Cline / Roo / Aider / OpenHands 是 2026 主流开源 baseline，先抄骨架再改，比从零写省 80% 工程量。
- **auto-compact**（超 80% ctx 就摘要）和 **tool line-range 限制** 是不让 context 爆炸的两条命脉。
- Code RAG 五件套：tree-sitter 切块 + bge-code-v1 embedding + Qdrant + BM25 hybrid + bge-reranker-v2。
- 第一期 YAGNI：GraphRAG / 多模态 emb / 自训 emb / 自研向量库——这些每一项都能再耗 1-2 个月。

**自检 3 题**（< 5 分钟）
1. 为什么 agent 在 8K context 下表现可能比 200K 还好？
2. Code RAG 的 reranker 贡献多少？没有 reranker 行不行？
3. tree-sitter 切块为什么比按行 / 按字数切好？

<details><summary>参考答案</summary>

1. 超过 64K context 后 RULER needle 检索准确率开始明显下降，agent 反而更容易"忘"前期信息（lost in the middle）。auto-compact 是必备补救：把已完成的工具回合摘要成一句，腾出 ctx 装新观察。
2. 通常 +10-15pp Recall@5。dense + BM25 hybrid 给的是 "candidate 100"，reranker 才决定"top 5 谁进 prompt"；预算紧时可以省，但代价是相关性肉眼可见地差。
3. tree-sitter 按 AST 切，保留 function / class 完整语法单元；按行 / 字数切会把函数体拦腰斩断，retrieval 出的片段不可读、模型也无法 reason。
</details>

> ⚠️ **常见坑** · agent 调 `read_file` 工具一次性读 5000 行文件——单次 tool call 就吃 30K token，三轮后 context 爆炸。必须给 `read_file` 强制 `start_line` / `end_line` 参数 + 默认 100 行上限；让 agent 自己 `grep` / `outline` 先定位，再针对性 read。

**下一步** → 进入 [phase_lab 实验册](./phase_lab.md) 用 30-60 分钟实验把全栈学到的东西手过一遍。术语速查 → [▣ 索引](./phase_glossary.md)。

---

## 动手练习

1. 浏览 Cline / Roo Code / Aider / OpenHands 四个开源 agent 仓库，对照 §1 的 6 层架构图，回答：每个项目把"规划层"放在 prompt 里还是放在外部 controller 里？谁的 tool schema 最简洁？
   *提示*：直接读各自 README + 主入口源文件即可。
2. 用 LiteLLM proxy 把你 phase7 部署的 GLM-5.1 endpoint 包装成"伪装成 Anthropic Claude"的 API，然后在 Cline 里把它配置成 model，跑通 "在一个 ≤ 500 行的 Python 项目里加一个新功能" 的小任务。
   *提示*：§2 路径 A 章节 + LiteLLM 文档。注意 system prompt 注入和 tool schema 转换。
3. 实现 §3 的 tool 层：用纯 Python 写 6 个 tool（bash, read_file, write_file, search_files, run_tests, git_diff），每个 tool 都用 docker sandbox 隔离执行，给出完整 OpenAI-style schema。要求 tool 互相不冲突、错误能正确返回给模型。
   *提示*：§3.3 sandbox 章节 + §3.5 tool schema 设计。Docker exec 比 E2B 更适合本地。
4. 抄 §3 的 300 行 minimal agent 骨架，搭一个 ReAct loop：模型 → tool_call → observation → 模型 → … 跑一个真实的 SWE-Bench Lite 题，记录 token 消耗、轮次、是否解决。要求引入 §4 的 auto-compact（超过 ctx 80% 就摘要历史）。
   *提示*：§4 上下文管理 + §5 Reflexion；first run 不要追指标，先跑通。
5. **完整 capstone**：为一个 ≥ 50 万文件的真实大型仓库（公司 monorepo 或开源大项目如 `kubernetes/kubernetes`）搭一套完整 Code RAG 系统——tree-sitter AST 切块 + bge-code-v1 embedding + Qdrant 向量库 + BM25 hybrid search + bge-reranker-v2 + GLM-5.1 生成带 citation 的答案。要求：webhook 增量更新、IDE 插件入口、👍/👎 反馈闭环、降级策略。最后用 §10.16 的 50 条手写 query 评测 Recall@5 ≥ 80%。
   *提示*：§10 整章是这个项目的逐步施工图；§10.13 评测集 + §10.15 运维章节是上线前必读。这是 Phase 8 的"毕业项目"，做完你就有了一套可落地的企业级 coding agent。
