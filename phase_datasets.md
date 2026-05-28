# 🗂 数据卡片库 · 25 张主流数据集 · 一目了然

> 📅 主线快照：2026-05-28 · 上次核对：2026-05-28

> **⚡ 三句话要点**
> 1. 训 LLM 80% 工作量在数据，但"数据该用哪份"散落在 phase1 + phase4 没集中页——本章把 **25 个主流数据集**做成卡片：license / 规模 / 质量 / 推荐用法 / 已知坑。
> 2. **三类用途**分别归档：(A) **预训练** 11 张 / (B) **SFT 指令 + 工具** 9 张 / (C) **评测** 5 张。预训和评测的污染检查规则不一样，混了会翻车。
> 3. 每张卡固定 6 字段——抄就抄全部，**只抄 HuggingFace ID 不看 license 是合规事故**的源头。

---

## 卡片格式

```
### N · 数据集名（HF/源 link）
| 字段 | 值 |
|---|---|
| **规模** | tokens / 样本数 |
| **license** | 重要 — 决定能不能商业用 |
| **质量** | 综合主观分（★ ~ ★★★★★）+ 一句评价 |
| **推荐用法** | pretrain / SFT / RL / eval 各自适合哪段 |
| **已知坑** | 别人踩过你不用踩 |
| **配套论文** | arXiv / 官方报告 |
```

---

## A · 预训练数据（11 张）

### A1 · The Stack v2 (BigCode)

| 字段 | 值 |
|---|---|
| **规模** | 32T tokens · 658M 个 source file · 658 种语言 |
| **license** | Mixed (含 GPL/AGPL/permissive)；**`bigcode/the-stack-v2-dedup` 子集**只含 permissive (MIT/Apache/BSD) |
| **质量** | ★★★★☆ · 当前代码 pretrain 的**默认基线**，OpenCoder/StarCoder2/GLM-4.5 都用 |
| **推荐用法** | pretrain · 必从 `-dedup` 版本开始 |
| **已知坑** | (1) 默认版本含 GPL 系，**用 `-dedup` 或自己 license filter**；(2) 仍有 PII 残留，得跑二次 PII 扫；(3) 评测集污染未清除 |
| **论文** | [arXiv:2402.19173](https://arxiv.org/abs/2402.19173) StarCoder2 + The Stack v2 |

### A2 · OpenCoder (INF-LabAI · 2024)

| 字段 | 值 |
|---|---|
| **规模** | 2.5T 高质量 code tokens + 完整 pipeline 开源 |
| **license** | Apache-2.0 |
| **质量** | ★★★★★ · 全 pipeline 公开（dedup / 过滤 / 去污 / 配比），**复现唯一可参考的工业级 corpus** |
| **推荐用法** | pretrain · 把它当成"数据 pipeline 教科书" |
| **已知坑** | 中文不强（< 5%），自训中文场景要补 CCI-3 |
| **论文** | [arXiv:2411.04905](https://arxiv.org/abs/2411.04905) |

### A3 · CodeParrot Github-Code (HF)

| 字段 | 值 |
|---|---|
| **规模** | 1TB raw · 32 语言 |
| **license** | Mixed · 含 GPL · **学术友好不商用友好** |
| **质量** | ★★★☆☆ · 较 old，The Stack v2 出来后基本被取代 |
| **推荐用法** | 学术对照实验 / 教学；生产用 The Stack v2 |
| **已知坑** | dedup 不彻底，PII 大量残留 |

### A4 · CCI-3 (BAAI · 2024)

| 字段 | 值 |
|---|---|
| **规模** | ~500B 中文 high-quality tokens |
| **license** | 自定（学术 + 商用都允许，但要 attribution） |
| **质量** | ★★★★☆ · 中文 LLM pretrain 当前最干净的开源中文 corpus |
| **推荐用法** | pretrain · 中文配比的主力来源 |
| **已知坑** | 文学/新闻偏多，代码相关 < 1% |

### A5 · Dolma (AI2 · 2024)

| 字段 | 值 |
|---|---|
| **规模** | 3T tokens 通用 |
| **license** | ODC-By 1.0（attribution required） |
| **质量** | ★★★★☆ · OLMo 用的同款，质量好但偏英文通用 |
| **推荐用法** | pretrain · 通用 mix 25-40% |
| **已知坑** | 没特别针对 code 优化 |

### A6 · FineWeb / FineWeb-Edu (HuggingFace · 2024)

| 字段 | 值 |
|---|---|
| **规模** | FineWeb 15T · FineWeb-Edu 1.3T |
| **license** | ODC-By 1.0 |
| **质量** | ★★★★☆ · FineWeb-Edu 子集教育内容质量极高，是 2024 后 SOTA 配比标配 |
| **推荐用法** | pretrain · -Edu 子集作为高质量段（mid-training / annealing） |
| **已知坑** | 评测集污染未清除，自己跑 decontam |

### A7 · OpenMath-2 (NVIDIA · 2024)

| 字段 | 值 |
|---|---|
| **规模** | 14M 数学问题 + 解答 |
| **license** | CC-BY-4.0 |
| **质量** | ★★★★☆ · code-math 任务必加 |
| **推荐用法** | mid-training 数学子集 10-15% |
| **已知坑** | 部分题来自 GSM8K / MATH 训练集，**评测前必去污染** |

### A8 · NaturalProver / proof-pile-2

| 字段 | 值 |
|---|---|
| **规模** | 1B+ proof token，Lean/Coq 等形式化 |
| **license** | MIT-style |
| **质量** | ★★★☆☆ · 偏冷门用，做"推理"维度提升 |
| **推荐用法** | mid-training 0.5-1% 给"硬推理"覆盖 |
| **已知坑** | 太特定，比例高反而降通用代码能力 |

### A9 · CodeAlpaca (2023)

| 字段 | 值 |
|---|---|
| **规模** | 20k Python instruction |
| **license** | CC-BY-NC-4.0 (**非商用！**) |
| **质量** | ★★☆☆☆ · 早期数据，被 OSS-Instruct 取代 |
| **推荐用法** | 不推荐生产用；教学例子可以 |
| **已知坑** | CC-NC license + 严重的 GPT-3.5 distillation 痕迹 |

### A10 · DCLM (Datasets for Curriculum Language Models · 2024)

| 字段 | 值 |
|---|---|
| **规模** | 4T tokens 通用 + benchmark |
| **license** | 自定 (attribution + 学术) |
| **质量** | ★★★★☆ · 提供数据**配方**而不仅数据，是 2024 的最佳实践参考 |
| **推荐用法** | pretrain · 学方法论 |

### A11 · StarCoderData / StarCoderPlusData

| 字段 | 值 |
|---|---|
| **规模** | 850GB · 86 语言 · GitHub 抓取 |
| **license** | BigCode OpenRail-M (类 Apache 但加 ethical use) |
| **质量** | ★★★★☆ · StarCoder/2 训练数据，已被 The Stack v2 覆盖 |
| **推荐用法** | 历史对照；新项目用 The Stack v2 |

---

## B · SFT / 工具调用 / Agent 轨迹（9 张）

### B1 · OSS-Instruct / Magicoder-OSS-Instruct-75K (ise-uiuc · 2023)

| 字段 | 值 |
|---|---|
| **规模** | 75K Python instruction-response |
| **license** | MIT |
| **质量** | ★★★★★ · 用真实代码逆向合成 instruction，**质量比 self-instruct 高一个量级** |
| **推荐用法** | SFT · 默认主力 30k-50k |
| **已知坑** | early 版本含 HumanEval 题面污染，用 v2+ |
| **论文** | [arXiv:2312.02120](https://arxiv.org/abs/2312.02120) Magicoder |

### B2 · OpenCodeInstruct (NVIDIA · 2024)

| 字段 | 值 |
|---|---|
| **规模** | 1.5M instruction · Python/JS/Java/C++ 等 |
| **license** | CC-BY-4.0 |
| **质量** | ★★★★☆ · 多语言、规模大、质量过得去 |
| **推荐用法** | SFT · 多语言场景首选 |
| **已知坑** | 含部分 GPT-4 生成内容，警惕 distillation 政策 |

### B3 · xLAM (Salesforce · 2024)

| 字段 | 值 |
|---|---|
| **规模** | 60k+ tool-use traj，单 / 多 / 并行 / 序列四类 |
| **license** | CC-BY-NC-4.0 (**非商用！**) |
| **质量** | ★★★★★ · function calling 数据最规范的开源集 |
| **推荐用法** | SFT · agent tool-use cold start 主力 |
| **已知坑** | 非商用 license 是硬约束；BFCL 评测污染中等 |
| **论文** | [arXiv:2406.18518](https://arxiv.org/abs/2406.18518) |

### B4 · ToolACE (Huawei Noah · 2024)

| 字段 | 值 |
|---|---|
| **规模** | 26k API + 11k tool 调用 |
| **license** | Apache-2.0 |
| **质量** | ★★★★☆ · 合成 + 多 agent 校验，质量稳；8B 模型用它在 BFCL 一度逼近 GPT-4 |
| **推荐用法** | SFT · tool-use 数据次要补充（混 xLAM 5:3） |

### B5 · Hammer 数据集 / Hammer 模型 (MadeAgents · 2024)

| 字段 | 值 |
|---|---|
| **规模** | 7k 高质量 function call + 1.5B/7B 配套模型 |
| **license** | Apache-2.0 |
| **质量** | ★★★★☆ · "小模型 + 数据质量 > 大模型 + 通用数据"的代表 |
| **推荐用法** | SFT · 7B 量级 function calling 起点 |

### B6 · SWE-Gym (2024)

| 字段 | 值 |
|---|---|
| **规模** | 数千个真实 GitHub issue + docker 镜像 |
| **license** | 见各 repo 原 license（Apache/MIT 居多） |
| **质量** | ★★★★★ · agent RL 训练几乎唯一的"真实环境" |
| **推荐用法** | RL · SWE-style agentic RL 的 env |
| **已知坑** | docker 镜像总体积 ~500GB，本地存储吃紧 |

### B7 · OpenHands trajectories (All-Hands-AI · 2024-2025)

| 字段 | 值 |
|---|---|
| **规模** | 数万条 agent traj（含成功 / 失败） |
| **license** | MIT |
| **质量** | ★★★★☆ · event-stream 格式，包含 reasoning + tool_call + observation 完整结构 |
| **推荐用法** | SFT · agent 轨迹 cold start |
| **已知坑** | 大部分 traj 是 Claude/GPT 生成，分布有偏 |

### B8 · NaturalInstructions / Super-NaturalInstructions (AI2)

| 字段 | 值 |
|---|---|
| **规模** | 1600+ 任务 · 5M+ 样本 |
| **license** | Apache-2.0 |
| **质量** | ★★★☆☆ · 通用 instruction，coding 占 < 5% |
| **推荐用法** | SFT · 通用对话能力的补充 |

### B9 · SmolTalk (HuggingFaceTB · 2024)

| 字段 | 值 |
|---|---|
| **规模** | 1.1M · 多任务多语言 SFT |
| **license** | Apache-2.0 |
| **质量** | ★★★★☆ · SmolLM2 训练用，质量过滤严格 |
| **推荐用法** | SFT · 小模型的通用对话补充 |

---

## C · 评测（5 张）

### C1 · HumanEval / HumanEval+ (OpenAI · EvalPlus)

| 字段 | 值 |
|---|---|
| **规模** | 164 题（HumanEval+ 用同样 prompt，但单测扩 8×） |
| **license** | MIT |
| **质量** | ★★★★★ · 入门基线 · SOTA 已饱和 > 95% |
| **推荐用法** | eval · 永远跑作 sanity check |
| **已知坑** | **题面广泛被 SFT 数据集污染**，分数高不一定有意义；改看 HumanEval+ 鲁棒分 |
| **论文** | OpenAI [arXiv:2107.03374](https://arxiv.org/abs/2107.03374) · EvalPlus [arXiv:2305.01210](https://arxiv.org/abs/2305.01210) |

### C2 · MBPP / MBPP+ (Google · EvalPlus)

| 字段 | 值 |
|---|---|
| **规模** | 1000 题 (MBPP) / 378 题 (MBPP+ sanitized) |
| **license** | CC-BY-4.0 |
| **质量** | ★★★★☆ · 比 HumanEval 难度均匀；MBPP+ 是去噪版 |
| **推荐用法** | eval · 和 HumanEval+ 配对报告 |
| **已知坑** | 同 C1，污染严重 |

### C3 · LiveCodeBench (NUS · 2024+)

| 字段 | 值 |
|---|---|
| **规模** | 月度滚动新增（~1000+ 题，2025-10 起每月加 50-100） |
| **license** | Apache-2.0 |
| **质量** | ★★★★★ · **必须报告时间窗** · 2026 主流模型对比的金标 |
| **推荐用法** | eval · 时间窗外才是有效 OOD 评测 |
| **已知坑** | 仅算法题，不测 repo 级；分 4 难度档差异巨大 |
| **论文** | [arXiv:2403.07974](https://arxiv.org/abs/2403.07974) |

### C4 · SWE-Bench / Lite / Verified / Pro / Multimodal (Princeton)

| 字段 | 值 |
|---|---|
| **规模** | Lite 300 · Verified 500（人工审）· Pro ~700（更难，2026 新版）· Multimodal 600 |
| **license** | Apache-2.0 |
| **质量** | ★★★★★ · **当前 coding agent 评测的天花板** |
| **推荐用法** | eval · agent 必跑；Verified > Lite > Pro 的顺序选 |
| **已知坑** | (1) docker image 总 ~500GB；(2) 跑一遍要 50-100 H100·h；(3) 部分题已被 distill 进开源 SFT 数据，自检污染 |
| **论文** | [arXiv:2310.06770](https://arxiv.org/abs/2310.06770) |

### C5 · BigCodeBench (BigCode · 2024)

| 字段 | 值 |
|---|---|
| **规模** | 1140 题 · 1000+ 真实 library 调用 |
| **license** | Apache-2.0 |
| **质量** | ★★★★☆ · 比 HumanEval 更接近"调真实库"，少污染（题面新） |
| **推荐用法** | eval · HumanEval/MBPP 已饱和后的替代基线 |

---

## 📌 章末检查

**带走这 5 条**
- 默认 pretrain 组合 = **The Stack v2-dedup + OpenCoder pipeline + FineWeb-Edu + CCI-3 + OpenMath-2**（中英文 + code + math 全 covered）。
- 默认 SFT 组合 = **OSS-Instruct 30k + Issue-PR 5k + xLAM/ToolACE 5k + agent 轨迹 1k**（覆盖 phase4 §10.8 配比）。
- **License 必看**：CodeAlpaca / xLAM 是 CC-BY-**NC** 不能商用，OpenCodeInstruct 是 GPT-4 distill 慎入。
- **评测污染检查永远要做**：用 phase1 §C 的 10-gram + MinHash 双扫，从 SFT 数据里剔除 HumanEval/MBPP/LiveCodeBench 题面 + 题解。
- LiveCodeBench / SWE-Bench Verified **报告时一定带时间窗 + 配套版本号**，否则数字没意义。

**自检 3 题**
1. 你公司要做商用 coding LLM，下面哪个数据**不能**直接用？(a) The Stack v2-dedup (b) CodeAlpaca (c) OSS-Instruct (d) FineWeb-Edu
2. 想做"工具调用"专项 SFT，xLAM / ToolACE / Hammer 三选一，怎么选？
3. 你的 SFT 模型 HumanEval+ 跑 91%，但 LiveCodeBench (2025-10 后题) 只有 32%——意味着什么？

<details><summary>参考答案</summary>

1. **(b) CodeAlpaca**，CC-BY-NC 非商用；其它都是商用友好。
2. 商用 → ToolACE / Hammer（都是 Apache-2.0）；最规范但仅学术 → xLAM。**默认 ToolACE 主力 + Hammer 数据补充**。
3. **训练数据含 HumanEval 污染**。两者差距 60pp 不正常；正常该是 20-30pp 差距。先回 phase1 §C 跑去污染。
</details>

> ⚠️ **常见坑** · 看一个数据集就直接 `datasets.load_dataset()`——HF 下了 200GB 才发现 license 是 CC-NC 没法上线。**load_dataset 之前必看 dataset card 的 license 字段**。

**下一步** · 数据 pipeline 实操 → [phase1](./phase1_data_pipeline.md) · 评测污染检查 → [📓 phase_failures §B1](./phase_failures.md) · 术语速查 → [▣ phase_glossary](./phase_glossary.md)。
