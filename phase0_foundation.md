# Phase 0 基础笔记：自训 Coding LLM 与 GLM-5.1 主线调研

> 立项日期：2026-04-22
> 主线模型：GLM-5.1 (Z.ai, 2026-04-07 发布, 754B MoE-DSA, MIT)
> 目标读者：具备 GPU 推理优化背景的独立研究者

本笔记目的是把训练一个 SOTA coding LLM 的"技术栈全貌"铺开，并与主线模型 GLM-5.1 对齐。阅读前置假设：读者熟悉 Transformer、MoE、KV cache、FlashAttention、PagedAttention 等工程概念，因此跳过基础解释。

> **读者画像** · 想 1 小时内对齐"GLM-5.1 在 2026 是什么状态"的资深推理/训练工程师，需要数据来排研究优先级。
> **前置知识** · MoE / MLA / RoPE / FP8 / KV cache 都听过；读过 GLM-4.5 ARC 或 DeepSeek-V3 任一篇技术报告者最佳；序章 [phase_basics_training](./phase_basics_training.md) 里的训练力学。
> **学完能做** · 写一份"GLM-5.1 vs Qwen3-Coder vs DeepSeek-Coder-V2"内部技术评审，并给出现实自训目标。

---

## 1. GLM-5.1 速览

> 本节给的是 GLM-5.1 的**当前快照**（数字、对手、定位）。如果你想理解 "为什么这个架构长成这样、每一项设计在解决什么瓶颈"，先看 Phase 2 §0.5 的架构演进史（Vanilla Transformer → GPT → LLaMA → Mixtral → DeepSeek → GLM-5.1），再回来看这里的具体参数会更顺。

GLM-5.1 是 Z.ai（原智谱 AI）在 2026-04-07 放出的 754B 开源权重模型，定位"agentic engineering"而非"vibe coding"。它的直接对手是 GPT-5.4 与 Claude Opus 4.6，在 SWE-Bench Pro 上以 58.4 登顶。它所对应的方法论论文是 GLM-5 团队在 2026-02-17 挂出的 `arXiv:2602.15763`（"GLM-5: from Vibe Coding to Agentic Engineering"），架构与训练细节大体沿用 `arXiv:2508.06471`（GLM-4.5 ARC 技术报告）。

### 1.1 架构参数

| 维度 | 值 | 备注 |
|---|---|---|
| 总参数 | 754 B | HF `zai-org/GLM-5.1` |
| 激活参数 / token | ~40 B | 8 routed + 1 shared |
| 总层数 | 78 | 首 3 层 dense，其余 75 层是 sparse MoE |
| Hidden dim | 6144 | |
| Attention heads | 64 query heads / 64 KV heads | 不是 GQA，而是 MLA 式的 q_lora/kv_lora 压缩 |
| q_lora_rank | 2048 | MLA 结构里查询压缩秩 |
| kv_lora_rank | 512 | MLA 结构里键值压缩秩 |
| qk_nope_dim | 192 | 不加 RoPE 的维度 |
| qk_rope_dim | 64 | 加 RoPE 的维度 |
| v_dim | 256 | |
| 总专家数 | 257 | 256 routed + 1 shared |
| 每 token 激活专家 | 8 routed + 1 shared = 9 | top-8 routing |
| moe_intermediate_size | 2048 | 单个 routed expert 的 FFN 中间维 |
| intermediate_size | 12288 | dense FFN 中间维 |
| routed_scaling_factor | 2.5 | MoE 门控分数缩放 |
| context | 200 K（原生） | 最大 128 K 输出 token |
| 注意力 | **MoE-DSA** | 见下节 |
| 精度 | BF16 / FP32 参数；FP8 推理可选 | |
| 许可 | MIT | 可商用、无 DAU 限制 |

### 1.2 MoE-DSA 里的 "DSA" 到底是什么

HF 卡片里 model tag 是 `glm_moe_dsa`。这里的 **DSA = DeepSeek Sparse Attention**（不是 "Dense-Sparse-Alternating"，有些博客写错）——即 DeepSeek 在 `V3.2-Exp`（2025-09）首次工程化部署的那套稀疏注意力。GLM-5 直接集成了这套机制。它由两个模块组成：

1. **Lightning Indexer**：一个极小的多头网络（可用 FP8），对 query token 与前文所有 token 计算 index score，决定要保留哪些 KV 位置。
2. **Fine-grained Top-K Selector**：每个 query 只 attend 到 indexer 打分最高的 k 个 KV（DeepSeek-V3.2 的实现里 k = 2048）。

效果：
- 注意力复杂度从 O(L²) 降到 O(L·k)，k ≪ L；
- 长上下文 API 推理成本实测降到 ~50%；
- 在 MLA 之上实例化，所以 KV cache 仍然被 latent 向量压缩，再叠加稀疏 → 长上下文真正"能用"。

系统挑战（GPU 推理优化背景的读者需要关注）：top-k 选的 KV 位置是 token-dependent 的，导致 KV 工作集碎片化、易变、难以预取。这是 GLM-5.1 在 SGLang/vLLM 侧实现稀疏 attention kernel 时要解决的主要工程问题（参见 `vllm` blog 2025-09-29 与 SGLang Day-0 支持文档）。

### 1.3 训练方法概述（来源：GLM-5 论文 + GLM-4.5 ARC）

- **预训练**：沿用 GLM-4.5 的两段 23T 数据 + 7T 代码/推理上采样配方（详见 §2）。
- **Mid-training**：把序列长度从 32K 推到 128K，喂 repo-level 代码、合成推理、agent 轨迹。
- **Post-training**：用自研 **slime** 异步 RL 基础设施 + 新型 **asynchronous agent RL algorithms**（token-level importance sampling + TITO 管线）。
- **Agentic 能力定义**：100+ 轮工具调用 / 连续 8 小时自主执行不掉链子，这是 GLM-5.1 相比 4.5 的最大差异。

---

## 2. GLM-4.5 ARC 方法论（pretrain / mid-train / post-train）

引用：`arXiv:2508.06471`，GLM-4.5 Team，2025-08。

### 2.1 Pretrain（23T tokens，两阶段）

| 阶段 | 数据 | 量级 | 关键做法 |
|---|---|---|---|
| Stage 1（通用） | web / books / papers / social / 多语 | ~15 T | 分质量桶上采样；SemDeDup 语义去重补 MinHash 漏网之鱼 |
| Stage 2（代码+推理） | GitHub 代码 + 数学/科学页面 | ~7 T | 代码分 3 档质量，低档直接剔除；数学/科学用 LLM 打"教育价值"分并上采样 |

优化器采用 **Muon**（不是 AdamW），配 cosine decay lr schedule + batch size warmup 到 64 M tokens。Muon 在 MoE 大模型上被证明比 AdamW 更稳、step 更少就能收敛——这是 2025 下半年开始流行的换法。

### 2.2 Mid-training（上下文扩展 + 专项数据）

三类数据注入：

1. **Repo-level code**：把一个 repo 的相关文件拼一起，加上 issue/PR/commit，教跨文件依赖与工程推理；
2. **Synthetic reasoning**：大量困难数学/科学题 + stepwise 解答，把 CoT 分布往难处推；
3. **Agentic trajectories**：tool use、浏览、function call 的长轨迹，序列长度拉到 **128 K**。

论文未公开这一阶段的具体 token 量（见 §6）。

### 2.3 Post-training（Expert Specialization → Unified Self-Distillation → RL）

分两大阶段：

**Stage A — Expert Model Specialization**（三个垂直 expert）
- Reasoning Expert：长 CoT SFT + RL
- Agent Expert：工具调用 SFT + 可验证环境 RL
- General Chat Expert：混合回复风格 SFT
- 此阶段 SFT 主要作"冷启动"，把 chat / reasoning / tool-use 基本能力点亮，再靠 expert RL 进一步提。

**Stage B — Unified Training（Self-Distillation + 统一 RL）**
- 用三个 expert 的输出做 self-distillation，蒸馏到一个 "hybrid reasoning" 生成器；
- 再一轮 RL 精修，学到"按任务类型选最合适的 CoT 深度"（thinking / direct response 双模）。

**基础设施：slime**
- Docker runtime + 解耦的 training / generation loop；
- FP8 rollout 推理，BF16 训练；
- 同步/异步两套 paradigm，长轨迹 agent 任务默认异步。
- 这套东西在 GLM-5 论文里进一步升级为 **asynchronous group-wise policy optimization + token-level importance sampling + TITO (token-in-token-out) pipeline**，核心是让动作-奖励在完全异步下仍然能无损对齐。

### 2.4 ARC 指标（GLM-4.5）
TAU-Bench 70.1、AIME-24 91.0、SWE-bench Verified 64.2。355 B 总参/32 B 激活就达到 DeepSeek-V3/R1 (671 B) 同档位，说明配方有效。

---

## 3. 训练一个 SOTA Coding LLM 的完整步骤清单

以下是一份从零训练 checklist（按工程顺序，不是按重要性）。目标是"可实际执行"，每条写清产物。

### 数据侧

1. **爬取原始代码语料**。对标 The Stack v2（67.5 TB，900 B 原始 tokens，600+ 语言）或 RefineCode（960 B tokens，607 语言）。工具：GHArchive、SWH、SoftwareHeritage graph 查询。
2. **许可过滤**。只留 permissive license 或 public domain；黑名单 GPL/AGPL（合规与下游商用考量）。
3. **去重**：exact SHA256 → MinHash-LSH（fuzzy）；OpenCoder 额外保留 star 高、commit 新的文件。
4. **清洗**：
   - 文件 >8 MB 直接丢；
   - 版权声明、PII（email、token、AWS key）正则去除；
   - 自然语言过滤（README/doc 专用规则）；
   - 通用代码过滤（空文件、auto-gen、binary blobs）；
   - 语言特定规则（至少覆盖 Python/Java/C++/Go/Rust/JS/TS/SQL 这 8 门）。
5. **采样再平衡**：对 Java/HTML/JSON 这类过剩语言下采样。RefineCode 从 960 B 采到 ~730 B。
6. **代码质量打分**：用一个小 reward model 或 rule-based 评估器把代码分 3 档（GLM-4.5 做法），低档丢、高档上采样。
7. **代码-文档对齐语料**：docstring ↔ 实现、commit message ↔ diff、issue ↔ patch、PR ↔ code review。这类"程序员自然语言"是 coding 能力的关键。
8. **repo-level 打包**：以 repo 为单位拼接相关文件，保留目录结构与 import 关系。这是 mid-training 的核心素材。
9. **合成数据**：
   - 用已有 LLM 把 leetcode/竞赛题改写成自然描述 + 多解；
   - 用 AST 做代码变换（rename、重构、bug 注入/修复配对）；
   - execution trace 合成（输入 → 逐步状态）教 "execution-aware" 推理。
10. **Fill-in-the-Middle (FIM) 转换**：按 StarCoder2 格式把文件做随机 span mask，教补全能力。

### 训练侧

11. **Tokenizer**：BPE 或 tiktoken，词表 ≥ 96 K（OpenCoder-8B = 96 640）。多语言+代码要保留 whitespace token 与 digit-splitting。
12. **预训练阶段 1（通用底座）**：3 T–15 T tokens 的通用 web + 代码 2:8 或 3:7 混合。OpenCoder-8B = 2.5 T，其中 90% 代码 / 10% code-related web。
13. **预训练阶段 2（annealing / 上采样）**：最后 ~10% steps 用高质量代码 + 数学 + 推理数据，学习率线性衰减到 10%。对小模型（≤8 B）这一步经常决定胜负。
14. **Mid-training / 长上下文扩展**：
    - 把 RoPE base 放大（NTK-aware 或 YaRN）；
    - 32 K → 128 K → 256 K 分档；
    - 喂 repo-level + long doc + agent 轨迹。
    - 对标 DeepSeek-Coder-V2（Yarn 到 128 K）和 Qwen3-Coder（原生 262 K）。
15. **优化器选择**：7 B 以下可用 AdamW；≥30 B MoE 建议 Muon（GLM-4.5 已验证），step 数可省 15-30%。
16. **并行策略**：TP + PP + EP（expert parallel）+ ZeRO-3；MoE 必须 EP。FP8 训练（NVIDIA H100/H200 或 B200）能把 throughput 相对 BF16 提 2-3×。
17. **Post-training SFT 冷启动**：100-500 K 高质量 instruction，覆盖 code gen、debug、code review、多轮 agent。参考 DeepSeek-Coder-V2 的 300 M token SFT。
18. **可验证 RL（Code RL）**：
    - reward = 编译通过 ∧ 单测通过 ∧ 格式合规；
    - GRPO（DeepSeek 系）或 asynchronous group-wise PPO（GLM-5）；
    - 大规模 executable environment：Qwen3-Coder 在阿里云起了 20 K 并发环境做 agent RL。
19. **Agent RL（长程任务）**：
    - SWE-bench / Terminal-Bench 风格的真实 repo 环境；
    - 轨迹可能上百步、数千 tool call；
    - 异步架构必不可少（slime / slime-like），否则 rollout 会成为瓶颈。
20. **自蒸馏合并**：若走了 expert specialization 路线，最后需要做 self-distillation 把多个 expert 合回一个统一模型（GLM-4.5 做法）。
21. **安全 / 宪法式对齐**：Constitutional AI 风格，防 prompt injection（agent 场景特别重要）+ 敏感代码拒答策略。Qwen3-Coder 明确跑了这一步。
22. **评估矩阵**：
    - 通用：HumanEval / HumanEval+ / MBPP+ / BigCodeBench
    - Repo 级：SWE-bench Verified / SWE-bench Pro / NL2Repo
    - Agent：Terminal-Bench 2.0 / TAU-Bench
    - 长上下文：RULER / LongCodeArena
    - Math/推理：AIME、GSM8K、MATH
23. **推理优化交付**：quantization (AWQ / GPTQ / FP8)、speculative decoding、prefix cache、PagedAttention；若用 DSA，还要实现稀疏 KV 的 indexed fetch kernel。
24. **红队与污染检查**：用 n-gram / embedding 两种方法查训练集与评测集重叠；特别注意 HumanEval 在 GitHub 上的污染。
25. **持续训练 / 版本管理**：固化数据 snapshot（git-lfs 或 DVC）、checkpoint 按 step 存、训练日志/loss curve 入 wandb/mlflow。

---

## 4. 模型对比表

| 维度 | GLM-5.1 | DeepSeek-Coder-V2 | Qwen3-Coder | OpenCoder | StarCoder2 |
|---|---|---|---|---|---|
| 发布时间 | 2026-04-07 | 2024-06 | 2025-07（480B）/ 2026-03（Coder-Next 80B） | 2024-11 | 2024-02 |
| 架构 | MoE + MLA + DSA，78 L | MoE + MLA (DeepSeekMoE) | MoE (Qwen3)，62 L，96/8 GQA | Dense（Llama-like） | Dense + GQA + 滑窗 |
| 总参数 | **754 B** | 236 B / 16 B（Lite） | 480 B / 80 B (Next) / 系列 | 1.5 B / 8 B | 3 B / 7 B / 15 B |
| 激活参数 | 40 B | 21 B / 2.4 B | 35 B / 3 B | 全激活 | 全激活 |
| 总专家 | 256 routed + 1 shared | DeepSeekMoE（160 routed + 2 shared） | 160 (8 激活) | N/A | N/A |
| 预训练 tokens | 未公开（推测 ≥ 23 T，继承 4.5） | 10.2 T (V2 base 4.2T + 额外 6T) | 36 T (Qwen3 base) + 7.5 T 代码 mid-train | 2.5 T | 3.3-4.3 T |
| 代码占比 | 未公开 | 60% | 70% | 90% | ~100% |
| 后训练 | Expert specialization → self-distill → async agent RL (slime) | SFT 300M + GRPO (compiler/test reward) | SFT + Agent RL（20K 并发 env）+ 长程 RL | SFT (4.5 M + 375 K) | 仅 base（Instruct 另放） |
| 长上下文 | 200 K 原生 | 128 K（YaRN） | 262 K 原生（480B） | 8 K | 16 K（滑窗 4 K） |
| 注意力 | MLA + DSA 稀疏 | MLA | GQA | MHA/GQA | GQA + 滑窗 |
| 许可 | **MIT** | DeepSeek License（商用需申请，较宽松） | Apache 2.0 | Apache 2.0 + 数据全开 | BigCode OpenRAIL-M |
| 数据开源程度 | 否 | 否 | 否 | **全开**（含 pipeline） | **全开**（The Stack v2） |
| 代表 bench | SWE-Bench Pro 58.4 | HumanEval 90.2, MBPP 76.2 | SWE-Bench Verified 69.6 | HumanEval 66.5 (8B) | HumanEval 46 (15B) |

注：DeepSeek-Coder-V2 与 Qwen3-Coder 具体专家数因实现版本而异；上表取各自主力版本的公开值。StarCoder2 用 FIM 训练，"base"即指未对齐的基础模型。

---

## 5. "自己训一个" 的现实目标设定

### 5.1 四档规模的成本估算

按业界经验法则：训练一个 dense transformer，FLOPs ≈ 6·N·D（N=参数数，D=训练 tokens）；MoE 用**激活参数**算 FLOPs。再按 H100 BF16 实际吞吐 ~400 TFLOPS（MFU ~50%）折算 GPU·小时。H100 云价取 2.5-3.5 $/GPU·小时中位 3。

| 规模 | 有效 FLOPs/token | 建议 tokens | 总 FLOPs | H100·小时（MFU 50%）| GPU 数×时长（8 卡 H100） | 云费用估算 | 数据量 |
|---|---|---|---|---|---|---|---|
| 0.3 B dense | 1.8 GF | 30 B | 5.4e19 | ~3.7 万 | 8×~192 h ≈ 1.6 kh | **≈ $11 万** | ~100 GB 代码 |
| 1.5 B dense | 9 GF | 200 B | 1.8e21 | ~12.5 万 | 64×~80 h ≈ 5.1 kh | **≈ $38 万** | ~1 TB |
| 7 B dense | 42 GF | 1 T | 4.2e22 | ~290 万 | 512×~46 天 ≈ 566 kh | **≈ $870 万** | ~6 TB |
| 30 B MoE (A3 B 激活) | 18 GF | 2 T | 3.6e22 | ~250 万 | 256×~80 天 ≈ 490 kh | **≈ $750 万** | ~12 TB |

说明：
- 上表只算 **预训练**；mid-training + post-training（SFT + RL）典型再加 10-30%；
- 数据清洗/存储/带宽成本另计（通常占 15-25%）；
- 实验迭代（ablation、hyperparam search）通常占总 budget 的 30-50%，预算必须翻倍预留；
- 用 FP8（H100/H200）或 FP4（B200）训练，throughput 可再 1.5-2.5×，对应把上表 H100 数除以这个因子即可估 B200 成本；
- MoE 30 B / 激活 3 B 有效 FLOPs 其实比 7 B dense 小，但通信代价高（EP）、并且需要更大 batch 才能稳定，实际 MFU 可能只有 30-35%。

### 5.2 建议：从 1.5 B dense 起步

路线建议（单人 + 租 GPU 形态）：

**Phase 1（验证 pipeline，预算 1-3 万美元）**：训练一个 **0.3 B dense** 模型，30-50 B tokens。
- 目的：跑通"数据清洗 → tokenizer → 训练 → eval → 部署"全链路；
- 4-8 张 H100 一周内可完成；
- HumanEval 预期 20-30 分，不追指标，只追流程正确。

**Phase 2（能拿出 demo，预算 20-40 万美元）**：训练 **1.5 B dense**，200-300 B tokens。
- 目的：做出能跑、能 serve、在 HumanEval/MBPP 有竞争力（50-60 分）的模型；
- 64-128 张 H100 × 1-2 周；
- 重点做好 annealing + 中等规模 SFT，RL 可选；
- 这一档开始可以认真做 mid-training 扩长上下文到 32-64 K。

**Phase 3（差异化产品）**：选一条：
- **A. 继续自训 7 B**：需要 $500 万+ 以及 1-2 个月 512 卡 H100，收益不一定打得过直接微调 Qwen3 或 GLM-4.5-Air；
- **B. 基于 GLM-5.1 做 LoRA / QLoRA / continued pretraining**：投入 1-5 万美元，在垂直领域（某一行业代码、某一 DSL、某一内部框架）做强；这是 GPU 推理优化背景的读者最容易变现的路径。
- **C. 做 MoE 30 B A3 B**：如果有 512+ H100 两个月 + 丰富工程能力，性价比反而好于 7 B dense。

**为什么从 1.5 B 起步而不是 0.3 B 或 7 B**：
- 0.3 B 做完"仅仅跑通"没多少外部价值，不值得单独停留；
- 7 B 的"能力涌现门槛"确实存在（HumanEval 从 30→60 分的跳跃），但直接自训 7 B 对独立研究者是自毁预算；更好的策略是先用 1.5 B 把方法论跑顺，然后在 Phase 3 选 B 或 C。
- 1.5 B 是目前学术/独立研究者**能独立拥有全权**的最大档，也是 ablation 可行的上限。OpenCoder-1.5B 本身就证明这个档次能"认真做科研"。

---

## 6. 关键开放问题 / 黑盒清单

以下是公开资料（论文 / blog / 模型卡）里**没讲清楚**、但决定 SOTA 能否复现的点。做 Phase 0 前要明确这些是"未解风险"，不能假装知道。

1. **GLM-5.1 / GLM-5 还没有对应的 arXiv 技术报告**。`2602.15763` 是方法论纲要，但关于 754 B 具体的预训练 token 量、数据配比、mid-training token 量、post-training 数据规模，**均未公开**。GLM-4.5 ARC 论文是最接近的参考。
2. **GLM-4.5 的 mid-training 具体 token 量未公开**。只知道"progressive 扩到 128 K"和"repo-level + 合成推理 + agent 轨迹"三类，但 repo-level 代码用了多少、合成推理用了多少，论文没给。
3. **DSA 的 Lightning Indexer 训练细节**。DeepSeek-V3.2 把 indexer 当小网络和主干联合训练，但 loss 设计、怎么保证 top-k 可导、indexer 初始化是否用了 distill，细节都只在 DeepSeek-V3.2 paper（`2512.02556`）里部分披露，GLM-5.1 如何在主干上接入 DSA（从 head 零训 vs. 从已有 checkpoint 加装）**没有公开**。
4. **agent RL 的 reward 设计**。GLM-5 宣称"100+ 轮不掉链子"、Qwen3-Coder 起 20 K 并发环境，但**稀疏/过程奖励如何分配、防 reward hacking 的具体手段、训练数据采样策略**全是黑盒。
5. **数据版权与合规**。所有大厂都模糊处理 GitHub 代码的许可过滤。OpenCoder 是唯一全公开数据的，也只放了 permissive 子集。自训要么接受只用 permissive（产品天花板偏低），要么自担法律风险。
6. **Muon 优化器的 MoE 稳定性**。GLM-4.5 报告 Muon 对 355 B MoE 工作得很好，但超参（lr schedule、weight decay、梯度裁剪）未完整公开。Muon 在社区的 MoE 大规模实验数据不多。
7. **self-distillation 的具体损失**。GLM-4.5 说"把三个 expert 蒸回一个"，但是 KL? reverse KL? token-level? sequence-level? 论文没给对比实验。
8. **agent 评测集污染**。SWE-bench Verified 已经被业界喂得很熟，新模型是否用真实 SWE-bench repo 做训练的负面证据几乎拿不出来——这是整个 agent coding 赛道的共同风险。
9. **推理端 DSA 的 kernel 生态**。vLLM / SGLang 对 DSA 的支持是 "day-0" 但仍在演进，长期稳定性、与 speculative decoding 的兼容、FP8 weight + DSA sparse attention 的联合优化，目前都还不成熟。这对 GPU 推理优化背景的读者反而是一个切入机会。
10. **成本估算的不确定性**。§5.1 的价格按云价 3 $/H100·h 估，但 spot、长租、自建差异可达 5×。加上 MoE 真实 MFU 与理论值落差，最终账单浮动 ±40% 属于常态。

---

## 参考资料

- GLM-4.5 ARC 技术报告 (`arXiv:2508.06471`)
- GLM-5 方法论 (`arXiv:2602.15763`)
- GLM-5.1 HF 模型卡 (`zai-org/GLM-5.1`) 与 Z.ai blog (`z.ai/blog/glm-5.1`)
- DeepSeek-Coder-V2 (`arXiv:2406.11931`)
- DeepSeek-V3.2-Exp (`arXiv:2512.02556`)，DSA 的权威来源
- Qwen3 技术报告 (`arXiv:2505.09388`)；Qwen3-Coder blog (`qwenlm.github.io/blog/qwen3-coder`)；Qwen3-Coder-Next (`arXiv:2603.00729`)
- OpenCoder (`arXiv:2411.04905`)
- StarCoder2 & The Stack v2 (`arXiv:2402.19173`)
- vLLM blog "DeepSeek-V3.2-Exp in vLLM" (2025-09-29)；SGLang Day-0 blog (2025-09-29)

*注*：因 WebFetch 在本次环境被拒，arxiv 原文未直接拉取，上文细节综合自 WebSearch 聚合结果（MarkTechPost、Analytics Vidhya、emergentmind、HF papers、官方 blog 等）。个别数值（如 GLM-5.1 层数/专家数）跨源已交叉验证，仍建议在拿到 arxiv PDF 后校对一次。

---

## 动手练习

1. 在 HuggingFace 上打开 `zai-org/GLM-5.1` 与 `deepseek-ai/DeepSeek-V3`，把两份 `config.json` 的字段做并列对比，找出 ≥ 5 处实质差异（不是命名）并指明每一处的设计动机。
   *提示*：重点看 q_lora_rank / kv_lora_rank / num_experts / moe_intermediate_size / topk_method。
2. 凭 §1 速览表，**手写一段 200 字的"模型卡口播稿"**：让一位听完 30 秒就能复述"GLM-5.1 凭什么在 SWE-Bench Pro 拿 58.4"。
   *提示*：抓住 MoE-DSA + agentic post-training 两个差异点。
3. 用 §3 的步骤清单，对你自己最熟悉的一个开源 base model（如 OpenCoder-1.5B）做一次"反查表"——它的每一个步骤是否都做了、做到了什么程度、缺哪一步。
   *提示*：OpenCoder 把数据/SFT/RL 全开源，最适合做对照。
4. 对 §6 黑盒清单的 10 条，每条写一段 100 字的"侦探推断"——基于公开线索（其他公司的论文、社区复现、模型行为），你猜 GLM-5.1 在这一项上具体做了什么？
   *提示*：把 DeepSeek-V3.2 / Qwen3 / Llama-3.1 当作旁证。
5. 完成 §5 的"自己训一个"现实目标设定：基于你能拿到的真实预算（GPU 类型 + 卡时数），写一份 1 页投资回报分析——你打算训多少参数、多少 token、对标哪条基线、达到分数多少算成功、失败时的 rollback 是什么。
   *提示*：交叉看 ROADMAP 的"建议推进节奏"+ phase2 §0 的最小可行 MoE 实验配置。
