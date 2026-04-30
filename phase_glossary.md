# 索引 · Glossary

> 高频技术术语速查表，按主题分组。每条带"详见 §X.Y"指向最详细解释。
> 配套笔记：`README.md` / `phase_basics_training.md` / `phase0` ~ `phase8`。
> 选词规则：从 12 份笔记中提取所有 **加粗** 关键词，按出现频次和"理解后续章节是否必备"两个维度筛出 48 条。

---

## 主题分组（推荐阅读顺序）

### 一、架构类（11 条）

#### MoE · Mixture of Experts
**定义**：每 token 只激活一小部分专家网络，参数量大但 FLOPs 小。GLM-5.1 用 256 routed + 1 shared，top-8 激活。
**详见**：`phase2 §0.5.4` `phase2 §1.2`
**相关**：Shared Expert / EP / aux-loss-free

#### DSA · DeepSeek Sparse Attention
**定义**：用 Lightning Indexer 给每个 query 动态挑 top-k KV，把 attention 从 O(L²) 降到 O(L·k)。GLM-5.1 从 DeepSeek-V3.2 集成。
**详见**：`phase0 §1.2` `phase2 §1.4`
**相关**：MLA / Lightning Indexer

#### MLA · Multi-head Latent Attention
**定义**：把 K/V 先压缩到低维 latent (rank≈512)，用时再投回，KV cache 缩约 10×。DeepSeek-V2 起、GLM-5.1 沿用。
**详见**：`phase_basics §11` `phase2 §1.5`
**相关**：DSA / GQA / KV Cache

#### MHA / MQA / GQA
**定义**：Attention 的三档 KV 共享方案——MHA（每头独立，最费）、MQA（K/V 1 头共享，最省精度差）、GQA（分 g 组共享，平衡）。LLaMA-3 用 GQA。
**详见**：`phase_basics §11` `phase2 §1.5`
**相关**：MLA / KV Cache

#### RoPE · Rotary Position Embedding
**定义**：用旋转矩阵给 Q/K 注入位置信息，相对位置由旋转角度差表达。LLaMA 系/GLM/Qwen 全家桶都用。
**详见**：`phase2 §1.6` `phase3 §4.2`
**相关**：YaRN / LongRoPE

#### YaRN · Yet another RoPE extensioN
**定义**：RoPE 长上下文外推算法——低频维度插值、高频维度外推、加 attention temperature scaling。32K→128K 的事实标准。
**详见**：`phase3 §4.5`
**相关**：LongRoPE / NTK-aware / Position Interpolation

#### LongRoPE
**定义**：用进化搜索给 RoPE 每个维度找非均匀缩放因子，扩到 200K+ 仍能保留短上下文性能。Phi-3-mini-128K / GLM-5.1 推测使用。
**详见**：`phase3 §4.6`
**相关**：YaRN / RoPE

#### RMSNorm
**定义**：LayerNorm 简化版，只用均方根归一化、去掉 mean 减法。比 LayerNorm 快 10-20%、效果相当，现代 LLM 标配。
**详见**：`phase2 §0.5.3` `phase_basics §3`
**相关**：pre-norm / SwiGLU

#### SwiGLU
**定义**：FFN 激活函数，用 Swish(xW1) ⊙ (xW3) 做门控，参数比 GeLU FFN 多 1.5×但效果显著好。LLaMA / GLM 默认。
**详见**：`phase2 §0.5.3`
**相关**：RMSNorm / FFN

#### MTP · Multi-Token Prediction
**定义**：训练时让模型一次预测多个未来 token（DeepSeek-V3 / GLM-5.1 自带）。训练信号更密；推理时可直接做投机采样。
**详见**：`phase2 §3.3` `phase7 §MTP`
**相关**：Speculative Decoding / FIM

#### FIM · Fill-In-the-Middle
**定义**：把 `prefix [middle] suffix` 改写成 `<prefix><suffix><middle>` 训练，让模型学会代码补全。StarCoder2 标准做法。
**详见**：`phase1 §3.9` `phase2 §3.2`
**相关**：NTP / Packing

---

### 二、训练流程类（12 条）

#### Pretrain（预训练）
**定义**：在数 T tokens 无监督文本/代码上做 next-token prediction，每 token 都算 loss。lr ≈ 1e-4，0.5-2 epoch。
**详见**：`phase_basics §13` `phase2`
**相关**：SFT / RLHF

#### SFT · Supervised Fine-Tuning
**定义**：在 (instruction, response) 对上继续训，**只对 assistant 部分算 loss**。让 base 学会"听指令"。lr ≈ 1e-5，1-3 epoch。
**详见**：`phase_basics §13` `phase4`
**相关**：Chat Template / Loss Masking / LoRA

#### RLHF · RL from Human Feedback
**定义**：经典三段式（SFT → Reward Model → PPO）让模型对齐人类偏好。Coding 场景已基本被 RLVR 取代。
**详见**：`phase5 §2.1`
**相关**：PPO / DPO / Reward Model

#### RLVR · RL with Verifiable Reward
**定义**：用编译器/单测/格式校验给的二值奖励替代 RM。Coding 与数学因"天生可验证"成为最适合的场景。
**详见**：`phase5 §2.2`
**相关**：GRPO / Verifier / Agentic RL

#### DPO · Direct Preference Optimization
**定义**：把 RLHF 闭式解写成监督 loss，跳过显式 RM 和 PPO 采样，只要偏好对就能训。便宜稳定但能力上限略低。
**详见**：`phase5 §3 / §2`
**相关**：RLHF / PPO

#### PPO · Proximal Policy Optimization
**定义**：经典 actor-critic 强化学习算法，用 clip ratio 限制策略更新幅度。需要 value head + reference model，4 模型并存。
**详见**：`phase5 §3`
**相关**：GRPO / Critic / Reference Model

#### GRPO · Group Relative Policy Optimization
**定义**：DeepSeek-R1 的简化版 PPO——同一 prompt 采 G 条 rollout，组内 z-score 当 baseline，扔掉 critic value head，显存减半。
**详见**：`phase5 §3`
**相关**：PPO / RLVR / Rollout

#### Causal Mask + Teacher Forcing
**定义**：上三角 -∞ 屏蔽未来 token；训练时 forward 看真实 ground-truth 而非自生成 token——支撑 Transformer 可一次并行算所有位置 loss 的两个机制。
**详见**：`phase_basics §5`
**相关**：KV Cache / NTP

#### Chat Template
**定义**：用 `<|im_start|>` `<|user|>` 等特殊 token 把多轮对话拼成一段长文本，让 base 学会角色切换。SFT/Agent 调用的底层协议。
**详见**：`phase_basics §14` `phase4 §10.6`
**相关**：Tool Calling / Loss Masking

#### Loss Masking
**定义**：把不该学的 token labels 设 -100 (PyTorch ignore_index)。SFT 只对 assistant 算、observation 不算、终止 token `<|im_end|>` **必须算**。
**详见**：`phase4 §4.2 / §10.5`
**相关**：Chat Template / SFT

#### Packing
**定义**：把多条短样本首尾拼成一条定长序列减少 padding 浪费。需配合 document attention mask 防跨文档泄漏。
**详见**：`phase_basics §6.2` `phase4 §4.1`
**相关**：Repo-level Packing / Attention Mask

#### Scaling Law / Chinchilla
**定义**：经验公式 N ≈ 12·L·d²、FLOPs ≈ 6·N·D，最优配比 D/N ≈ 20。给定算力可秒算"几张卡几天能训多大"。
**详见**：`phase_basics §12`
**相关**：MFU / FLOPs

---

### 三、推理 & 部署类（10 条）

#### KV Cache
**定义**：自回归解码时缓存历史 K/V 避免重算。每 token 在 7B 32 层模型约 512 KB，是推理显存与带宽的头号矛盾。
**详见**：`phase_basics §16` `phase7 §核心问题`
**相关**：PagedAttention / MLA / Prefill+Decode

#### PagedAttention
**定义**：vLLM 提出的 KV Cache 分页管理（仿 OS 虚拟内存），消除 KV 碎片、支持 continuous batching，吞吐 2-4×。
**详见**：`phase7 §vLLM`
**相关**：vLLM / RadixAttention / Continuous Batching

#### RadixAttention
**定义**：SGLang 的系统级前缀缓存——基数树存共享前缀的 KV，多请求/多轮共享，配合 LRU 驱逐。多轮 agent 场景吞吐显著领先。
**详见**：`phase7 §RadixAttention`
**相关**：vLLM APC / Prefix Cache / SGLang

#### Speculative Decoding
**定义**：用一个小模型/MTP head 一次猜多个 token，主模型并行验证。Memory-bound 场景关键加速器，GLM-5.1 自带 MTP head。
**详见**：`phase7 §MTP` `phase_basics §16.2`
**相关**：MTP / EAGLE / Medusa

#### FP8（E4M3 / E5M2）
**定义**：H100/H200 原生 8-bit 浮点。E4M3（前向）+ E5M2（反向）+ per-block scaling 几乎无损。训练 throughput 比 BF16 提 1.5-2.5×，推理几乎免费 2× 显存。
**详见**：`phase2 §6.1` `phase7 §FP8`
**相关**：BF16 / AWQ

#### AWQ · Activation-aware Weight Quantization
**定义**：W4A16 后训练量化，按 activation magnitude 给重要 channel 留高精度。生产首选 4-bit，几乎无损精度。
**详见**：`phase7 §AWQ`
**相关**：GPTQ / FP8

#### GPTQ
**定义**：基于二阶信息（Hessian 近似）的 W4A16 后训练量化，比 AWQ 早、对长尾 activation 略敏感，校准慢但生态广。
**详见**：`phase7 §GPTQ`
**相关**：AWQ / SmoothQuant

#### TP / PP / EP / DP
**定义**：四种并行——DP（数据切到不同卡，AllReduce 梯度）、TP（单层权重列切，AllReduce 激活）、PP（按层切，P2P 激活）、EP（MoE 专家切到不同卡，AllToAll token）。组合公式：总卡数 = DP × TP × PP × EP。
**详见**：`phase_basics §15` `phase2 §4`
**相关**：FSDP / ZeRO / SP

#### FSDP / ZeRO
**定义**：分片数据并行——把参数/梯度/optimizer state 按 DP rank 切，需要时 AllGather。FSDP 是 PyTorch 原生 ZeRO-3，"懒人版"分布式。
**详见**：`phase_basics §15.3` `phase2 §4.3`
**相关**：DP / TP

#### vLLM / SGLang
**定义**：两大开源推理引擎——vLLM 主打 PagedAttention + 通用稳定，SGLang 主打 RadixAttention + 多轮 agent 吞吐。GLM-5.1 双方都 day-0 支持。
**详见**：`phase7 §推理框架对比`
**相关**：PagedAttention / RadixAttention / TensorRT-LLM

---

### 四、评测 & 数据类（10 条）

#### The Stack v2
**定义**：BigCode 公开的代码预训练语料（67.5 TB / 600+ 语言 / 来自 Software Heritage）。配套 SWHID resolve 工具链，是开源代码训练的事实底座。
**详见**：`phase1 §1`
**相关**：StarCoder2 / OpenCoder

#### MinHash-LSH
**定义**：近似去重算法——用 MinHash 估 Jaccard 相似度、LSH 分桶加速。Code LLM pipeline 的关键步骤，必须配 SemDeDup 兜底。
**详见**：`phase1 §3.4`
**相关**：Decontamination / SemDeDup

#### Decontamination（去污染）
**定义**：用 n-gram (10-gram) overlap 或 embedding 相似度，从训练集剔除评测集（HumanEval / MBPP / SWE-Bench）样本。**最容易被跳过、但最致命**。
**详见**：`phase1 §3.8`
**相关**：MinHash / SWE-Bench

#### HumanEval / HumanEval+
**定义**：OpenAI 164 题函数级 Python 编程基准；EvalPlus 把单测扩了 80×（变成 HumanEval+），抗过拟合。`+` 是揭穿过拟合的第一层滤网。
**详见**：`phase6 §HumanEval`
**相关**：MBPP / LiveCodeBench / EvalPlus

#### LiveCodeBench
**定义**：按题目发布日期分段的滚动榜单（LeetCode/AtCoder/Codeforces），只报告"模型训练截止后"窗口的 pass@1，**抗污染最现实的标尺**。
**详见**：`phase6 §LiveCodeBench`
**相关**：HumanEval+ / Decontamination

#### SWE-Bench / Verified / Lite
**定义**：从真实 GitHub PR 抽出 issue+repo+ground-truth patch 的 agent benchmark。Verified（500 题人工审过）= 唯一可信的非饱和 pass@1 信号；Lite 是 300 道单文件 bug fix 入门版。
**详见**：`phase6 §SWE-Bench`
**相关**：Agent / Docker / SWE-Gym

#### Tree-sitter
**定义**：增量解析器生成器，给 100+ 语言提供 AST。Repo 级切块、import 图构建、签名地图（Aider 风格）的标准工具。
**详见**：`phase8 §RAG切块`
**相关**：Repo-level Packing / Embedding

#### Embedding（代码向量）
**定义**：把代码片段映射到稠密向量做语义检索。中文代码场景推 `bge-code-v1`，必须按 AST 切块、prepend 文件路径。
**详见**：`phase8 §RAG`
**相关**：Reranker / RAG / bge-code-v1

#### Reranker
**定义**：检索第二段——用更慢更强的 cross-encoder（如 `bge-reranker-v2-m3`）重排 embedding 召回的 top-k。命中率 +10-20%，延迟 +50-200ms。
**详见**：`phase8 §RAG`
**相关**：Embedding / Hybrid Search

#### RAG · Retrieval-Augmented Generation
**定义**：把检索到的代码片段塞进 prompt 让 LLM 回答。Coding 场景核心：必须 code-aware embedding + 跨文件 call graph + Hybrid (RRF) 融合。
**详见**：`phase8 §RAG`
**相关**：Embedding / Reranker / Tree-sitter

---

### 五、Agent / 应用类（5 条）

#### Agent
**定义**：由 LLM 驱动的"感知—规划—工具—反思"循环执行体。GLM-5.1 定位"100+ 轮工具调用 / 8 小时连续不掉链子"。
**详见**：`phase8 §架构总览`
**相关**：ReAct / Tool Calling / Reflexion

#### Tool Calling
**定义**：LLM 输出结构化 `<tool_call>{"name":..., "arguments":...}</tool_call>`、外部执行后用 `<observation>` 回填的协议。SFT 必须用 chat_template 学。
**详见**：`phase_basics §14.3` `phase4 §10.6`
**相关**：Chat Template / Agent

#### ReAct · Reason + Act
**定义**：Thought → Action → Observation 循环模板。最常用 single-agent 骨架，工具数 ≤10、单任务 < 50 轮的首选。
**详见**：`phase8 §ReAct`
**相关**：Agent / Reflexion / Plan-Execute

#### Reflexion
**定义**：失败后让 agent 自我反思生成修正策略再重试。SWE-Bench 类长程任务的关键补救机制，配合 self-distill 提分明显。
**详见**：`phase8 §Reflexion`
**相关**：ReAct / Hindsight Relabeling

#### Auto-compact（历史摘要）
**定义**：长对话/长 trace 把早期轮次摘要成短文本节省 KV cache，配合 200K+ 长上下文模型。Coding agent 不做这个跑不过 SWE-Bench Lite。
**详见**：`phase8 §上下文管理`
**相关**：Long Context / KV Cache

---

## A-Z 索引（速查）

| 术语 | 主题 | 详见 |
|---|---|---|
| Agent | Agent | phase8 |
| Auto-compact | Agent | phase8 |
| AWQ | 部署 | phase7 |
| Causal Mask | 训练 | phase_basics §5 |
| Chat Template | 训练 | phase_basics §14 |
| Chinchilla / Scaling Law | 训练 | phase_basics §12 |
| Decontamination | 数据 | phase1 §3.8 |
| DPO | 训练 | phase5 §3 |
| DSA | 架构 | phase0 §1.2 |
| Embedding | 评测/数据 | phase8 |
| FIM | 架构 | phase1 §3.9 |
| FP8 | 部署 | phase7 |
| FSDP / ZeRO | 部署 | phase_basics §15.3 |
| GPTQ | 部署 | phase7 |
| GQA / MQA / MHA | 架构 | phase_basics §11 |
| GRPO | 训练 | phase5 §3 |
| HumanEval / + | 评测 | phase6 |
| KV Cache | 部署 | phase_basics §16 |
| LiveCodeBench | 评测 | phase6 |
| LongRoPE | 架构 | phase3 §4.6 |
| Loss Masking | 训练 | phase4 §4.2 |
| MinHash-LSH | 数据 | phase1 §3.4 |
| MLA | 架构 | phase2 §1.5 |
| MoE | 架构 | phase2 §1.2 |
| MTP | 架构 | phase2 §3.3 |
| Packing | 训练 | phase_basics §6 |
| PagedAttention | 部署 | phase7 |
| PPO | 训练 | phase5 §3 |
| Pretrain | 训练 | phase_basics §13 |
| RadixAttention | 部署 | phase7 |
| RAG | 应用 | phase8 |
| ReAct | Agent | phase8 |
| Reflexion | Agent | phase8 |
| Reranker | 应用 | phase8 |
| RLHF | 训练 | phase5 §2.1 |
| RLVR | 训练 | phase5 §2.2 |
| RMSNorm | 架构 | phase2 §0.5.3 |
| RoPE | 架构 | phase2 §1.6 |
| SFT | 训练 | phase4 |
| SGLang | 部署 | phase7 |
| Speculative Decoding | 部署 | phase7 |
| SwiGLU | 架构 | phase2 §0.5.3 |
| SWE-Bench | 评测 | phase6 |
| Teacher Forcing | 训练 | phase_basics §5 |
| The Stack v2 | 数据 | phase1 §1 |
| Tool Calling | Agent | phase_basics §14.3 |
| TP / PP / EP / DP | 部署 | phase_basics §15 |
| Tree-sitter | 数据 | phase8 |
| vLLM | 部署 | phase7 |
| YaRN | 架构 | phase3 §4.5 |

---

## 给新手的 5 个"先看懂这个再读"优先级

如果你只来得及搞懂 5 个术语再去读后面 30 万字，请按这个顺序：

1. **Pretrain → SFT → RLHF/RLVR 三阶段范式**（`phase_basics §13`）——整套笔记的骨架，每一段方法论都是在回答"这是哪一阶段的事情"。
2. **KV Cache + Prefill/Decode**（`phase_basics §16`）——Phase 7 部署一半的内容（PagedAttention / RadixAttention / Speculative / FP8）都是为了优化它。
3. **MoE + MLA + DSA 三件套**（`phase0 §1` / `phase2 §1.2-1.5`）——GLM-5.1 全部架构创新都在这里，不懂这三个无法看 GLM-5.1 模型卡。
4. **Loss Masking + Chat Template**（`phase4 §4.2 / §10.5`）——SFT 最容易踩雷的地方（`<|im_end|>` 必须算 loss、observation 必须屏蔽），决定 SFT 是否会"全军覆没"。
5. **Decontamination 与 LiveCodeBench**（`phase1 §3.8` / `phase6`）——任何看似漂亮的 HumanEval 分数没去污染就是幻觉，先建立"哪些指标可信"的免疫力比训模型更重要。
