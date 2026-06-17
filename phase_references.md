# Phase References · 参考文献页

> 这是本 Coding LLM 研究笔记（README.md / ROADMAP.md / phase0–phase8 / phase_basics_training.md）所引用论文与开源仓库的合订索引。所有 arXiv 编号经去重，按主题分组；每组开头有一段导读，每条按"模型/方法名 · 标题 · arXiv 编号 · 链接"的格式给出。
>
> 维护原则：论文标题保留英文原文；中文笔记里出现过的链接全部回填；项目仓库一句话点评其在工程链路里的作用；最后给出一个分阶段的"必读路径"和若干中文综述/博客补充。

---

## 1. 架构类（Architecture）

> 这一类的核心问题是：在 100B–1T 量级下，如何同时拿到"高质量稠密参数 + 可扩展的稀疏激活 + 长上下文 + 低 KV 显存"。主线沿着 GLM-4.5 / GLM-5、DeepSeek-V2/V3/V3.2、Qwen3、Mixtral、LLaMA 一路演进，关键词是 **MoE（细粒度 + 共享专家）、MLA（隐空间 KV 压缩）、DSA（稀疏注意力 + Lightning Indexer）、MTP（多 token 预测）、YaRN/LongRoPE（长上下文外推）**。读这一组主要是为了搞清楚"为什么 GLM-5.2 长这样"。

- **GLM-4.5 ARC** · *GLM-4.5: Agentic, Reasoning, Coding (ARC) Foundation Models* · arXiv:2508.06471 · [链接](https://arxiv.org/abs/2508.06471)
- **GLM-5 方法论** · *GLM-5: from Vibe Coding to Agentic Engineering* · arXiv:2602.15763 · [链接](https://arxiv.org/abs/2602.15763)
- **DeepSeek-Coder-V2** · *DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence* · arXiv:2406.11931 · [链接](https://arxiv.org/abs/2406.11931)
- **DeepSeek-V3** · *DeepSeek-V3 Technical Report*（含 MLA / DeepSeekMoE / FP8 / MTP） · arXiv:2412.19437 · [链接](https://arxiv.org/abs/2412.19437)
- **DeepSeek-V3.2-Exp** · *DeepSeek-V3.2-Exp: DSA — DeepSeek Sparse Attention with Lightning Indexer* · arXiv:2512.02556 · [链接](https://arxiv.org/abs/2512.02556)
- **DeepSeekMoE** · *DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models* · arXiv:2401.06066 · [链接](https://arxiv.org/abs/2401.06066)
- **DeepSeek-Coder V1** · *DeepSeek-Coder: When the Large Language Model Meets Programming* · arXiv:2401.14196 · [链接](https://arxiv.org/abs/2401.14196)
- **Qwen3** · *Qwen3 Technical Report* · arXiv:2505.09388 · [链接](https://arxiv.org/abs/2505.09388)
- **Qwen3-Coder-Next** · *Qwen3-Coder-Next Technical Report* · arXiv:2603.00729 · [链接](https://arxiv.org/abs/2603.00729)
- **YaRN** · *YaRN: Efficient Context Window Extension of Large Language Models* · arXiv:2309.00071 · [链接](https://arxiv.org/abs/2309.00071)
- **LongRoPE** · *LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens* · arXiv:2402.13753 · [链接](https://arxiv.org/abs/2402.13753)
- **LongLoRA** · *LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models* · arXiv:2309.12307 · [链接](https://arxiv.org/abs/2309.12307)
- **Ring Attention** · *Ring Attention with Blockwise Transformers for Near-Infinite Context* · arXiv:2310.01889 · [链接](https://arxiv.org/abs/2310.01889)
- **FlashAttention-2** · *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* · arXiv:2307.08691 · [链接](https://arxiv.org/abs/2307.08691)
- **FlashAttention-3** · *FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-Precision* · arXiv:2407.08608 · [链接](https://arxiv.org/abs/2407.08608)
- **MiniCPM (WSD)** · *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies* · arXiv:2404.06395 · [链接](https://arxiv.org/abs/2404.06395)
- **In-Context Pretraining** · *In-Context Pretraining: Language Modeling Beyond Document Boundaries* · arXiv:2310.10638 · [链接](https://arxiv.org/abs/2310.10638)

仓库与工具：

- **Megatron-LM** · NVIDIA 大模型训练框架，MoE / TP / PP / SP 主参考实现 · [GitHub](https://github.com/NVIDIA/Megatron-LM)
- **TransformerEngine** · NVIDIA FP8 / Hopper 算子库，配 Megatron 用 · [GitHub](https://github.com/NVIDIA/TransformerEngine)
- **torchtitan** · PyTorch 官方面向千卡训练的最小化教学框架 · [GitHub](https://github.com/pytorch/torchtitan)
- **nanotron** · HuggingFace 出品的轻量 3D 并行训练框架，含 MoE 例子 · [GitHub](https://github.com/huggingface/nanotron)

---

## 2. 数据类（Data）

> 这一类的核心问题是：万亿 token 的 code corpus 怎么收、怎么洗、怎么去重、怎么按语言/质量/许可证打标，以及怎么从 GitHub Issues、Commit、Stack Exchange 里挤出"对话式"代码推理数据。读完应该知道 The Stack v2 / OpenCoder / DeepSeek-Coder 三家在 license 过滤、PII 脱敏、SemDeDup、FIM packing 上的具体路径差异。

- **StarCoder2 + The Stack v2** · *StarCoder 2 and The Stack v2: The Next Generation* · arXiv:2402.19173 · [链接](https://arxiv.org/abs/2402.19173)
- **OpenCoder** · *OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models* · arXiv:2411.04905 · [链接](https://arxiv.org/abs/2411.04905)
- **Dolma** · *Dolma: An Open Corpus of Three Trillion Tokens for Language Model Pretraining Research* · arXiv:2402.00159 · [链接](https://arxiv.org/abs/2402.00159)
- **FIM** · *Efficient Training of Language Models to Fill in the Middle* · arXiv:2207.14255 · [链接](https://arxiv.org/abs/2207.14255)

仓库与数据集：

- **bigcode/the-stack-github-issues** · The Stack 配套 GitHub Issues 子集（HuggingFace Dataset） · [HF](https://huggingface.co/datasets/bigcode/the-stack-github-issues)
- **codeparrot/github-code** · 轻量 GitHub 代码子集，适合 demo 与小规模实验（HuggingFace Dataset） · [HF](https://huggingface.co/datasets/codeparrot/github-code)
- **go-enry/go-enry** · GitHub linguist 的 Go 端口，用于按语言识别文件 · [GitHub](https://github.com/go-enry/go-enry)

---

## 3. 微调与 RL 类（SFT, PEFT, RLHF, RLVR）

> 这一类的核心问题是：在一个 base 模型上"教指令、教偏好、教推理"。SFT 侧关心怎样合成高质量 instruction（Self-Instruct → Evol-Instruct → OSS-Instruct/Magicoder → AgentInstruct），PEFT 侧关心如何低成本微调（LoRA/QLoRA/DoRA/GaLore），RL 侧的主线则是 InstructGPT-PPO → DPO → GRPO → DeepSeek-R1 的 RLVR 路径，最终在 GLM-4.5 / R1 上把 long-CoT、self-verification、agentic tool-use 的能力"长"出来。

- **InstructGPT** · *Training Language Models to Follow Instructions with Human Feedback* · arXiv:2203.02155 · [链接](https://arxiv.org/abs/2203.02155)
- **Self-Instruct** · *Self-Instruct: Aligning Language Models with Self-Generated Instructions* · arXiv:2212.10560 · [链接](https://arxiv.org/abs/2212.10560)
- **WizardCoder / Evol-Instruct** · *WizardCoder: Empowering Code Large Language Models with Evol-Instruct* · arXiv:2306.08568 · [链接](https://arxiv.org/abs/2306.08568)
- **Magicoder / OSS-Instruct** · *Magicoder: Empowering Code Generation with OSS-Instruct* · arXiv:2312.02120 · [链接](https://arxiv.org/abs/2312.02120)
- **ToRA** · *ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving* · arXiv:2309.17452 · [链接](https://arxiv.org/abs/2309.17452)
- **AgentInstruct** · *AgentInstruct: Toward Generative Teaching with Agentic Flows* · arXiv:2407.03502 · [链接](https://arxiv.org/abs/2407.03502)
- **SWE-Gym** · *Training Software Engineering Agents and Verifiers with SWE-Gym* · arXiv:2412.21139 · [链接](https://arxiv.org/abs/2412.21139)
- **LoRA** · *LoRA: Low-Rank Adaptation of Large Language Models* · arXiv:2106.09685 · [链接](https://arxiv.org/abs/2106.09685)
- **QLoRA** · *QLoRA: Efficient Finetuning of Quantized LLMs* · arXiv:2305.14314 · [链接](https://arxiv.org/abs/2305.14314)
- **DoRA** · *DoRA: Weight-Decomposed Low-Rank Adaptation* · arXiv:2402.09353 · [链接](https://arxiv.org/abs/2402.09353)
- **GaLore** · *GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection* · arXiv:2403.03507 · [链接](https://arxiv.org/abs/2403.03507)
- **DPO** · *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* · arXiv:2305.18290 · [链接](https://arxiv.org/abs/2305.18290)
- **DeepSeekMath / GRPO** · *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models* · arXiv:2402.03300 · [链接](https://arxiv.org/abs/2402.03300)
- **DeepSeek-R1** · *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning* · arXiv:2501.12948 · [链接](https://arxiv.org/abs/2501.12948)

仓库与工具：

- **LLaMA-Factory** · 一站式 SFT / DPO / LoRA 实战框架，支持几十种模型 · [GitHub](https://github.com/hiyouga/LLaMA-Factory)

---

## 4. 评测类（Evaluation）

> 这一类的核心问题是：怎么衡量"代码能力"，并且不被污染、不被刷榜。HumanEval / MBPP 是行业基线，HumanEval+ / MBPP+ / BigCodeBench 是它们的"测试增强 + 真实 API"升级版；LiveCodeBench 用 contamination-free 时间窗解决榜单泄漏；SWE-Bench / RepoBench 把战场从函数级搬到仓库级；CRUXEval / MultiPL-E 分别强调"执行推理"和"多语言"。

- **HumanEval** · *Evaluating Large Language Models Trained on Code* · arXiv:2107.03374 · [链接](https://arxiv.org/abs/2107.03374)
- **MBPP** · *Program Synthesis with Large Language Models* · arXiv:2108.07732 · [链接](https://arxiv.org/abs/2108.07732)
- **EvalPlus / HumanEval+ / MBPP+** · *Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation* · arXiv:2305.01210 · [链接](https://arxiv.org/abs/2305.01210)
- **BigCodeBench** · *BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions* · arXiv:2406.15877 · [链接](https://arxiv.org/abs/2406.15877)
- **LiveCodeBench** · *LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code* · arXiv:2403.07974 · [链接](https://arxiv.org/abs/2403.07974)
- **APPS** · *Measuring Coding Challenge Competence With APPS* · arXiv:2105.09938 · [链接](https://arxiv.org/abs/2105.09938)
- **SWE-Bench** · *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* · arXiv:2310.06770 · [链接](https://arxiv.org/abs/2310.06770)
- **RepoBench** · *RepoBench: Benchmarking Repository-Level Code Auto-Completion Systems* · arXiv:2306.03091 · [链接](https://arxiv.org/abs/2306.03091)
- **MultiPL-E** · *MultiPL-E: A Scalable and Polyglot Approach to Benchmarking Neural Code Generation* · arXiv:2208.08227 · [链接](https://arxiv.org/abs/2208.08227)
- **CRUXEval** · *CRUXEval: A Benchmark for Code Reasoning, Understanding and Execution* · arXiv:2401.03065 · [链接](https://arxiv.org/abs/2401.03065)
- **RULER** · *RULER: What's the Real Context Size of Your Long-Context Language Models?* · arXiv:2404.06654 · [链接](https://arxiv.org/abs/2404.06654)
- **LongBench** · *LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding* · arXiv:2308.14508 · [链接](https://arxiv.org/abs/2308.14508)

仓库：

- **LiveCodeBench** · 时间窗滚动收题、可复现的 contamination-free 评测框架 · [GitHub](https://github.com/LiveCodeBench/LiveCodeBench)
- **SWE-bench** · Princeton NLP，仓库级真实 issue 评测套件 · [GitHub](https://github.com/princeton-nlp/SWE-bench)
- **CRUXEval** · Meta 出品，输入/输出预测两类任务的执行推理评测 · [GitHub](https://github.com/facebookresearch/cruxeval)

---

## 5. 推理与部署类（Serving, Quantization, Speculative Decoding）

> 这一类的核心问题是：训完一个 744B MoE-DSA，怎么把它真正"跑起来"——KV 怎么排（PagedAttention / RadixAttention）、权重怎么压（AWQ / GPTQ / SmoothQuant）、token 怎么投机（Medusa / EAGLE / MTP）、kernel 怎么 fuse（DeepGEMM / TransformerEngine）。SGLang、vLLM、KTransformers、xLLM 是当下的主流引擎栈。

- **PagedAttention (vLLM)** · *Efficient Memory Management for Large Language Model Serving with PagedAttention* · arXiv:2309.06180 · [链接](https://arxiv.org/abs/2309.06180)
- **RadixAttention (SGLang)** · *Efficiently Programming Large Language Models using SGLang* · arXiv:2312.07104 · [链接](https://arxiv.org/abs/2312.07104)
- **AWQ** · *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration* · arXiv:2306.00978 · [链接](https://arxiv.org/abs/2306.00978)
- **GPTQ** · *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers* · arXiv:2210.17323 · [链接](https://arxiv.org/abs/2210.17323)
- **SmoothQuant** · *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models* · arXiv:2211.10438 · [链接](https://arxiv.org/abs/2211.10438)
- **Medusa** · *Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads* · arXiv:2401.10774 · [链接](https://arxiv.org/abs/2401.10774)
- **xLLM** · *xLLM: A High-Performance LLM Inference Framework* · arXiv:2510.14686 · [链接](https://arxiv.org/abs/2510.14686)

仓库与工程文档：

- **EAGLE-1/2/3** · 投机解码主流实现，含 EAGLE-2 / EAGLE-3 升级版 · [GitHub](https://github.com/SafeAILab/EAGLE)
- **DeepGEMM** · DeepSeek 自研 FP8 MoE GEMM kernel，配 V3 用 · [GitHub](https://github.com/deepseek-ai/DeepGEMM)
- **SGLang Cookbook (GLM)** · GLM-5 官方 SGLang 部署 cookbook，含 B200 / MI300 变体 · [GitHub](https://github.com/sgl-project/sgl-cookbook)
- **vLLM Recipes (GLM)** · vLLM 官方 GLM 系列部署食谱 · [GitHub](https://github.com/vllm-project/recipes)
- **KTransformers** · CPU + GPU 异构推理框架，含 GLM-5.2 kt-kernel 教程 · [GitHub](https://github.com/kvcache-ai/ktransformers)
- **xLLM (JD)** · 京东开源高性能推理框架 · [GitHub](https://github.com/jd-opensource/xllm)

---

## 6. Agent 与 RAG 类（Agent Frameworks, IDE, Sandbox）

> 这一类的核心问题是：把 LLM 从"补全器"变成"工程师"。早期是 ReAct（思考-行动交错）、Reflexion（自反思）这些 prompting 模板，2024 年起 SWE-Agent 与 OpenHands 把"agent + 文件系统 + shell + 浏览器"整套环境固化下来，并配套 SWE-Bench 这类仓库级评测；IDE 侧 Cline / Roo Code / Kilo Code 把这些能力嵌进 VS Code 的日常工作流。

- **ReAct** · *ReAct: Synergizing Reasoning and Acting in Language Models* · arXiv:2210.03629 · [链接](https://arxiv.org/abs/2210.03629)
- **Reflexion** · *Reflexion: Language Agents with Verbal Reinforcement Learning* · arXiv:2303.11366 · [链接](https://arxiv.org/abs/2303.11366)
- **SWE-Agent** · *SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering* · arXiv:2405.15793 · [链接](https://arxiv.org/abs/2405.15793)
- **OpenHands** · *OpenHands: An Open Platform for AI Software Developers as Generalist Agents* · arXiv:2407.16741 · [链接](https://arxiv.org/abs/2407.16741)
- **SWE-Bench** · *SWE-bench: Can Language Models Resolve Real-World GitHub Issues?* · arXiv:2310.06770 · [链接](https://arxiv.org/abs/2310.06770)（与评测类共享）

仓库：

- **OpenHands** · All-Hands AI，全功能 agent 平台，含 docker sandbox + 多 backend · [GitHub](https://github.com/All-Hands-AI/OpenHands)
- **SWE-agent** · Princeton NLP，最小可复现的 ACI（Agent-Computer Interface）实现 · [GitHub](https://github.com/SWE-agent/SWE-agent)
- **Cline** · 开源 VS Code 编程 agent，原 Claude Dev · [GitHub](https://github.com/cline/cline)
- **Roo Code** · Cline 分叉，主打多模型与多 mode 编排 · [GitHub](https://github.com/RooCodeInc/Roo-Code)
- **Kilo Code** · Roo / Cline 衍生，融合多家 IDE agent 特性 · [GitHub](https://github.com/Kilo-Org/kilocode)
- **GLM-5 (zai-org)** · 智谱 GLM-5 / GLM-5.2 官方仓库 · [GitHub](https://github.com/zai-org/GLM-5)
- **smol-developer** · 单文件 prompt-only 编程 agent 教学项目 · [GitHub](https://github.com/smol-ai/developer)
- **tree-sitter** · 增量解析器框架，AST 级 chunk / RAG 切片必备 · [Web](https://tree-sitter.github.io)
- **Firecracker** · AWS 出品的轻量 microVM，agent sandbox 主流方案 · [Web](https://firecracker-microvm.github.io)

---

## 必读路径推荐

### 如果你只能读 5 篇（按顺序）

1. **GLM-4.5 ARC**（arXiv:2508.06471）—— 全栈视角的"训练方法论主源"，把架构、数据、mid-training、SFT、RL、agentic 全串了一遍。
2. **DeepSeek-V3 技术报告**（arXiv:2412.19437）—— MLA / DeepSeekMoE / FP8 / MTP 的工程权威，所有数学细节最扎实。
3. **OpenCoder**（arXiv:2411.04905）—— 唯一一份把 code corpus 处理 pipeline **完全开源**的论文，配数据复现脚本食用。
4. **DeepSeek-R1**（arXiv:2501.12948）—— RLVR / GRPO 大规模成功样本，"推理能力靠 RL 长出来"的代表作。
5. **SWE-Bench**（arXiv:2310.06770）—— 不读完这篇，就没法理解 2024 年以后的 agentic coding 评测语境。

### 如果你想精读 15 篇（在前 5 篇基础上扩展）

6. **StarCoder2 + The Stack v2**（arXiv:2402.19173）—— 数据基线
7. **DeepSeek-Coder-V2**（arXiv:2406.11931）—— 数据配比与训练最详尽
8. **DeepSeekMath / GRPO**（arXiv:2402.03300）—— GRPO 起点
9. **InstructGPT**（arXiv:2203.02155）—— PPO-for-LLM 起点
10. **DPO**（arXiv:2305.18290）—— 偏好优化的闭式简化
11. **YaRN**（arXiv:2309.00071）—— 长上下文外推主流
12. **PagedAttention / vLLM**（arXiv:2309.06180）—— 推理引擎奠基
13. **AWQ**（arXiv:2306.00978）—— 4-bit 量化主流
14. **OpenHands**（arXiv:2407.16741）—— Agent 平台系统设计
15. **LiveCodeBench**（arXiv:2403.07974）—— 反污染评测的工程代表

### 中文综述/博客推荐（5–8 个）

- **机器之心** · DeepSeek-V3 / R1 / GLM-4.5 / Qwen3 系列长文解读，时效快、配图清楚 · <https://www.jiqizhixin.com>
- **量子位** · "万字长文"风的论文导读，适合作为入门概览 · <https://www.qbitai.com>
- **PaperWeekly** · 学术向论文精读，对 DPO / GRPO / FlashAttention 这类方法讲得最细 · <https://www.paperweekly.site>
- **知乎专栏：大模型炼丹指南** · 训练实操向，MoE / FP8 / 长上下文等工程问题答得很硬核 · <https://zhuanlan.zhihu.com>
- **Lil'Log（OpenAI Lilian Weng 个人博客，非中文但值得双语精读）** · *LLM Agents*、*Reward Hacking* 等系列的中文转译版本极多 · <https://lilianweng.github.io/lil-log>
- **苏剑林 Blog（kexue.fm）** · RoPE / LoRA / GRPO 等数学推导的中文一手资料 · <https://kexue.fm>
- **HuggingFace 中文社区博客** · OpenCoder、StarCoder2 等开源数据集发布同步配中文版 · <https://huggingface.co/blog/zh>
- **DeepSeek 官方公众号 / 智谱 AI 公众号** · 一手发布 + 训练细节披露的官方渠道，常常比 arXiv 更早 · 微信公众号搜"DeepSeek"、"智谱AI"

---

## 如何获取一份本文档原始 markdown

```bash
# 当前文件位置
/Users/huangshengqiu/Public/code/ai_research/codellm/phase_references.md

# 拷贝到桌面
cp /Users/huangshengqiu/Public/code/ai_research/codellm/phase_references.md ~/Desktop/

# 或者用编辑器直接打开
open -a "Visual Studio Code" /Users/huangshengqiu/Public/code/ai_research/codellm/phase_references.md

# 也可以转成 PDF（需要 pandoc + 中文字体）
pandoc /Users/huangshengqiu/Public/code/ai_research/codellm/phase_references.md \
  -o ~/Desktop/phase_references.pdf \
  --pdf-engine=xelatex -V CJKmainfont="PingFang SC"
```

> 本文档与 README.md / ROADMAP.md / phase0–phase8 / phase_basics_training.md 同目录，可直接 `grep -nE 'arxiv|github' phase_references.md` 反查任一引用的出处。
