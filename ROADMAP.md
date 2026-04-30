# 开源 Coding LLM 全链路研究路线图

> 主线模型：**GLM-5.1** (Z.ai, 2026-04-07, 754B MoE-DSA, MIT)
> 辅线参照：DeepSeek-Coder-V2、Qwen3-Coder、OpenCoder、StarCoder2
> 目标：(1) 理解全栈 → (2) 小规模复现训练 → (3) 搭建自己的 coding agent 应用

> **读者画像** · 准备启动 8 周自学/团队周会推进计划的工程师，需要一份"按周排好班"的目录索引。
> **前置知识** · 已经读过本仓库 README；对 LLM 训练 - 部署 - agent 三个大块名词不陌生。
> **学完能做** · 知道 9 个 phase 的产出物、依赖关系和现实预算，能给老板交一份排期。

---

## 核心资料锚点

| 资料 | 链接 | 用途 |
|---|---|---|
| GLM-4.5 ARC 技术报告 | arxiv 2508.06471 | 训练方法论主源 |
| GLM-5.1 权重 | hf: `zai-org/GLM-5.1` | 推理/微调 |
| GLM 官方代码 | github.com/zai-org/GLM-5 | 推理示例 |
| DeepSeek-Coder-V2 论文 | arxiv 2406.11931 | 数据配比/训练最详尽 |
| OpenCoder 论文 | arxiv 2411.04905 | 数据处理 pipeline 完全开源 |
| StarCoder2 + The Stack v2 | arxiv 2402.19173 | 数据集基准 |
| SGLang / vLLM GLM-5.1 recipes | 官方 cookbook | 部署 |

---

## Phase 0 — 奠基：建立心智地图

**目的**：搞清楚"一个 SOTA 的 coding LLM 到底做了哪些事"，以及各家选择的差异。

**产出物**：一份 `phase0_notes.md`，内容包括：
- GLM-4.5 ARC 的训练三段论 (pretrain → mid-train → post-train) 细节
- GLM-MoE-DSA 是什么、相对普通 MoE 的差异
- DeepSeek-Coder-V2 vs. Qwen3-Coder vs. GLM-5.1 的关键差异对比表
- 自己训一个 coding LLM 的最小闭环清单

---

## Phase 1 — 预训练数据 pipeline

**流程**：
```
原始仓库 → 语言检测 → 许可证过滤 → 近似去重(MinHash-LSH)
       → 精确去重 → 启发式/模型质量打分 → 评测集去污染
       → FIM 改写 → 按语言/质量分桶配比 → tokenize → 存成 mmap
```

**关键开源实现**：
- BigCode 的 `bigcode-dataset` 仓库（The Stack 同款 pipeline）
- OpenCoder 放出的完整数据处理脚本
- `datatrove` (HuggingFace 的大规模数据处理库)

**动手任务**：用 `datatrove` 跑通一个 10GB GitHub Python 子集的端到端 pipeline，最终产出 tokenized `.bin` 文件可直接喂 Megatron。

---

## Phase 2 — 预训练架构与实操

**需要搞明白的**：
- **架构**：MoE 路由机制 (Top-k / Shared-Experts / DeepSeekMoE / DSA)；Attention (MLA / GQA)；位置编码 (RoPE + 外推)
- **Tokenizer**：BBPE 训练、代码 token 处理 (空白/缩进/注释)
- **训练目标**：Next-Token Prediction + Fill-in-the-Middle (FIM) + 仓库级 packing
- **并行**：DP / TP / PP / EP / SP 的组合，以及各自的通信瓶颈
- **框架**：Megatron-LM（主流）/ Megatron-DeepSpeed / torchtitan（新的更简洁）/ NeMo
- **精度**：BF16 主流，FP8 混合精度（H100+）

**动手任务**：
- 用 torchtitan 或 nanotron 训一个 **0.5B-1.5B MoE coding 小模型**
- 数据：Phase 1 产出的 10GB 子集
- 硬件：假设 8×H100 (或租 Lambda Labs 几小时)
- 目标：跑通完整 loop，拿到一个会生成 Python 的小 baseline

---

## Phase 3 — 中期训练 & 长上下文

- **Mid-training**：退火阶段切换到高质量子集 + 数学推理数据 + 代码执行轨迹
- **长上下文**：GLM-5.1 是 200K 级别。方法栈：RoPE base 调大 / YaRN / position interpolation + 长文本 SFT 数据
- **仓库级训练**：按 import 图 / 文件依赖排序的 repo packing（关键！普通随机 shuffle 学不到跨文件推理）

---

## Phase 4 — SFT：指令与 Agent 轨迹

**数据来源**：
- Self-Instruct / Evol-Instruct / OSS-Instruct（从真实 GitHub 代码反向生成指令）
- 多轮代码对话（调试/修改/重构场景）
- **Agent 轨迹**：把 SOTA 模型（Claude、GPT）跑在 sandbox 里完成任务，保留其完整 tool-calling 轨迹作为 SFT 数据

**框架**：LLaMA Factory（最易上手）/ ms-swift（阿里，支持 GLM 全系列）/ Axolotl

**动手任务**：用 LLaMA Factory 对 **GLM-4.5-Air**（较小的 GLM 变体）做 LoRA SFT，数据用 open-instruct 或自己造 1k 条。

---

## Phase 5 — RL：RLHF / RLVR / Agentic RL

**三条技术路线**：
1. **传统 RLHF**：收集偏好对 → 训 RM → PPO
2. **RLVR (Verifiable Reward)**：用单元测试通过率作为奖励，无需 RM。DeepSeek-R1、Qwen3 都在用
3. **Agentic RL**：完整的"调工具 → 执行 → 观察 → 再决策"轨迹上做 RL，需要 sandbox 支撑

**算法**：PPO → DPO → GRPO（DeepSeek-R1 用的） → REINFORCE++ / RLOO

**动手任务**：在 Phase 4 的 SFT 模型上跑 GRPO，奖励函数 = Python 单测通过率（用 HumanEval 训练集）。

---

## Phase 6 — 评测体系

**核心 benchmark**：
- **HumanEval+ / MBPP+**：经典但已饱和
- **LiveCodeBench**：防污染（月度竞赛题）
- **BigCodeBench**：更真实的库调用
- **SWE-Bench Lite / Verified / Pro**：repo 级真实 bug fix
- **MultiPL-E**：多语言版 HumanEval
- **CRUXEval**：代码执行理解

**动手任务**：搭建本地 evalplus + SWE-Bench harness，对 GLM-4.5-Air、GLM-5.1 (API) 跑一遍，建立你自己的 baseline。

---

## Phase 7 — 推理部署优化

**可用的你现有 skill**：`gpu-inference-optimization-skill`

**核心主题**：
- SGLang vs. vLLM：何时选哪个（SGLang 对 MoE + speculative decoding 优化更激进）
- 量化：FP8（H100+）/ AWQ / GPTQ / SmoothQuant
- KV cache：PagedAttention / Prefix Caching / Chunked Prefill
- Speculative Decoding：GLM 自带 MTP（Multi-Token Prediction）层可用作草稿模型
- MoE 专用优化：Expert Parallelism、专家路由缓存

**动手任务**：
- 用 SGLang 部署 GLM-4.5-Air (AWQ/FP8 量化)，测 throughput 和 latency
- 用 KTransformers 在消费级硬件上跑 GLM-5.1 (CPU+GPU 混合)

---

## Phase 8 — Coding Agent 应用

**两条路径**：

### A. 接入现成 agent 外壳
- Claude Code（通过自定义 model endpoint）
- Cline / Roo Code / Kilo Code（VSCode 插件，都支持 OpenAI-compat）
- 把你本地部署的 GLM-5.1 或 GLM-4.5-Air 包装成 OpenAI API 接进去

### B. 自建 agent（学习路径）
关键组件：
1. **Tool schema**：文件读写、bash、搜索、代码执行、浏览器
2. **Sandbox**：Docker / Firejail / E2B 安全执行
3. **Repo 索引**：tree-sitter 解析 + 依赖图 / embedding 检索
4. **规划循环**：ReAct → Plan-and-Execute → 反思 (Reflexion)
5. **长对话管理**：摘要压缩、记忆、工作区状态

**动手任务**：
- 路径 A：把 GLM-5.1 (API) 接进 Cline，跑一个真实小项目
- 路径 B：写一个 <500 行 Python 的 minimal coding agent（参考 smol-developer / mini-devin）

---

## 建议的推进节奏

| 周 | 阶段 | 重点 |
|---|---|---|
| 1 | Phase 0 | 读论文，建心智地图 |
| 2 | Phase 1 | 跑通数据 pipeline |
| 3-4 | Phase 2 | 小模型从头训练 |
| 5 | Phase 3-4 | 长上下文 + SFT |
| 6 | Phase 5 | RLVR 实验 |
| 7 | Phase 6-7 | 评测 + 部署 |
| 8 | Phase 8 | Agent 应用 |

**现实提醒**：Phase 2 从头训练 1.5B 以上就需要多卡 H100 数小时到数天，预算不够可以降到 ~300M 参数纯做 pipeline 验证；主力目标放在"微调 + 部署 + agent"。

---

## 动手练习

1. 把"建议的推进节奏"表里 8 周计划改写成你自己实际可投入的时间预算（每天 N 小时、是否周末），并标记出哪几个 phase 你打算"读懂即可"，哪几个 phase 你打算"动手跑"。
   *提示*：算清楚每周可用 GPU·hour 数，对照 phase2/phase5 的最低预算门槛。
2. 选定一个具体的"产出物"作为你 8 周后的交付物（例：自训 300M 代码模型、把 GLM-4.5-Air 接进 Cline、跑通 SWE-Bench Lite baseline），并把它倒推成每个 phase 的子目标。
   *提示*：参考"路径 A vs 路径 B"的两条主线选一条。
3. 把"核心资料锚点"表里 7 篇论文/文档**全部下载到本地**并通读 abstract，写一份 1 页摘要回答："这 7 份资料在哪些方法论问题上彼此互相印证，又在哪里有分歧？"
   *提示*：DeepSeek-Coder-V2 vs OpenCoder 在数据 pipeline 上是最容易出现分歧的对比组。
4. 用一张 mermaid 图把 9 个 phase 的"上游产物 → 下游消费"关系画清楚（例：Phase 1 产 token .bin → Phase 2 消费）。
   *提示*：参考 README 的全景图 + phase_basics_training 末尾的"本章为后面哪一 phase 服务"映射表。
5. 自己挑一个公司内部真实场景（例：内部仓库代码补全、法务合规代码审查、CI 失败自愈 agent），从 Phase 0 开始为它定制一份"裁剪版" ROADMAP——明确哪些 phase 可以跳过、哪些 phase 必须深做、最终上线指标是什么。完整写出来 ≥ 2000 字。
   *提示*：这是把通用路线图转成"你自己的项目计划"——是整个仓库笔记的 north star 用法。
