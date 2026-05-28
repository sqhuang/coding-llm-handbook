# 📋 TL;DR · 全书 15 分钟速读卡

> 📅 主线快照：2026-05-28 · 上次核对：2026-05-28

> **怎么用这一页**：把全书 13 个 phase 的 ⚡ 三句话要点 + 📌 章末检查关键结论抽到一处。**适合扫一遍判断"我该不该认真读这本"**，或读完后当随身速查卡。每章带跳转链接。

---

## 序 · [phase_basics_training](./phase_basics_training.md) · 训练基础

**⚡ 要点**
1. NTP loss = 标准 `F.cross_entropy(shift_logits, shift_labels)`，与 HF `outputs.loss` 字节级一致。
2. causal mask + packing + position_ids reset 是 LLM 训练的"显存放大器"。
3. 显存账单 = 参数 + 梯度 + 优化器状态 + 激活 + KV cache，五项必须分别算清。

**📌 带走**：chat template 是契约，BF16 训 / FP16 推，KV cache = `2·L·H·d·seq·bytes·B`。

---

## 0 · [phase0_foundation](./phase0_foundation.md) · GLM-5.1 全景

**⚡ 要点**
1. GLM-5.1 = 754B 总 / 40B 激活 / 78 层 (3 dense + 75 MoE) / 256+1 expert / MLA q_lora 2048·kv_lora 512 / 200K 原生 / MIT。
2. DSA = **DeepSeek Sparse Attention**（Lightning Indexer + top-k KV），把 attention 从 O(L²) 降到 O(L·k)。
3. 训练栈沿用 GLM-4.5 ARC：23T 双段 pretrain + 7T 代码/推理上采样 + mid-train 128K + slime 异步 RL。优化器 AdamW → Muon。

**📌 带走**：top-8 routing + aux-loss-free 防 expert collapse。

---

## 1 · [phase1_data_pipeline](./phase1_data_pipeline.md) · 数据 pipeline

**⚡ 要点**
1. The Stack v2 / OpenCoder 是基线；任何自训语料先和这两份比统计量。
2. MinHash 近似去重 + 10-gram exact match 双保险——前者抓段落重复、后者抓评测污染。
3. Issue-PR 是金矿，四形态 (a)(b)(c)(d) 按过滤难度递减。

**📌 带走**：unicode 归一化必做，license filter 必做（GPL 不能进训）。

---

## 2 · [phase2_pretraining](./phase2_pretraining.md) · 预训练架构

**⚡ 要点**
1. DSA 每个 query 动态挑 top-N KV，KV 工作集碎片化是工程根源。
2. 小规模复现首选 **torchtitan**，8×H100 训 1B MoE 目标 MFU ≥ 35%。
3. MoE 新四样：Muon 替代 AdamW · aux-loss-free routing · FP8 混合精度 · repo-level packing。

**📌 带走**：监控 `expert_load_var` 比看 loss 更早发现路由 collapse。

---

## 3 · [phase3_midtraining_longcontext](./phase3_midtraining_longcontext.md) · 中期训练与长上下文

**⚡ 要点**
1. mid-training = pretrain 末尾的"退火"段，上采样代码/数学到 60-70%，配 WSD 二次衰减。
2. 长上下文 = RoPE 调教（YaRN / LongRoPE）+ 数据侧 repo-level packing 双管齐下。
3. YaRN 在 32K→128K 是事实标准；> 200K 才考虑 LongRoPE 非均匀缩放。

**📌 带走**：验真用 **RULER** needle，不要只看 PPL。

---

## 4 · [phase4_sft](./phase4_sft.md) · SFT 指令微调

**⚡ 要点**
1. Agent 能力**必须走轨迹数据**，最小配方：8k OSS-Instruct + 1-3k Claude/GPT sandbox 成功轨迹 + 500 中文通用，LoRA 2 epoch。
2. **Chat template + loss mask** 是最容易翻车的点——必须做 mask 可视化自检。
3. 三大框架选型：LLaMA Factory 易上手 / ms-swift 对 GLM 原生 / Axolotl 配置灵活，三选一。

**📌 带走**：LoRA target_modules 必须 `all-linear`（不止 q/k/v_proj）。

---

## 5 · [phase5_rl](./phase5_rl.md) · 强化学习

**⚡ 要点**
1. **GRPO** 扔掉 critic 用组内 z-score 做 baseline，显存减半且收敛更稳——R1 把它打成 2025-2026 事实标准。
2. **纯 RLVR 在 agent 方向不够**（reward 太稀疏），必须补 subgoal shaping 或 process signal。
3. Reward hacking 四大坑：测试覆盖率不够 / RM 训不到位 / KL 太松 / rollout-training 分布偏移。

**📌 带走**：reward = pytest 通过率，模型会学删测试；anti-hack 加 `-2.0` 惩罚。

---

## 6 · [phase6_evaluation](./phase6_evaluation.md) · 评测

**⚡ 要点**
1. HumanEval / MBPP 已饱和；2026 主线是 **SWE-Bench Lite / Verified / Pro** + LiveCodeBench。
2. `pass@k` 必须用 unbiased estimate `1 - C(n-c,k)/C(n,k)`，c/k 直接近似会高估。
3. LiveCodeBench 要报告**时间窗**否则训练污染让分数虚高。

**📌 带走**：内部 30-40 题 SWE-Bench v1 是公司决策的 north star。

---

## 7 · [phase7_deployment](./phase7_deployment.md) · 部署

**⚡ 要点**
1. SGLang / vLLM / KTransformers 三选一；按"全卡 vs offload"和"agent 多轮 vs 单轮"两轴决策。
2. MoE 推理瓶颈在 expert load + KV 命中，不是算力。
3. 量化层级：FP8 ≈ 无损（< 1pp）/ W8A8 / W4A16（长 coding 任务掉 2-4pp）。

**📌 带走**：prod 5 项不全过不要上线（5min 告警 / 10min 回滚 / dashboard / 评测对齐 / 成本可预测）。

---

## 8 · [phase8_agent_apps](./phase8_agent_apps.md) · Agent 应用

**⚡ 要点**
1. 两条路径：**A · 外壳路径** Roo Code + LiteLLM proxy 即插即用；**B · 自建路径** ReAct + Docker + 6 工具 300 行骨架。
2. 自建 agent 三层栈：**repo map**（tree-sitter）→ **auto-compact**（长对话摘要）→ **Reflexion**（失败反思）。
3. Sandbox 必上：Docker / Firejail / E2B 三选一。

**📌 带走**：`read_file` 强制 line-range，agent 一次读 5000 行 → context 爆炸。

---

## ⚒ · [phase_tooluse](./phase_tooluse.md) · Tool Use 速读

**⚡ 要点**
1. Tool use 是横切 phase4/5/8 的主线——chat template 是契约、SFT 教格式、RL 提任务成功率、sandbox 兜底执行。
2. 7 节点学完顺序：chat template → 轨迹四形态 → tool schema → sandbox → cold start SFT → RL shaping → ReAct。
3. 业界 8 例：Anthropic Claude / OpenAI strict mode + constrained decoding / MCP / Claude Code / SWE-agent / OpenHands / GLM-4.5 ARC / xLAM。

**📌 带走**：schema 合法性用 constrained decoding 解（4.2），比 SFT/RL 都靠谱。

---

## ✪ · [phase_capstone](./phase_capstone.md) · 4 周端到端

**⚡ 要点**
1. 端到端实验册：把 phase0-8 在同一个 4 周项目里走一遍——卡 / 数据 / 模型 / 超参 / 命令 / 验收 / 思考题。
2. 预算：8×H100 · 4 周 · 总成本 ≈ $4K（H100 按需 $2/hr · 总 2000 GPU-hour）。
3. 配套 `tools/track.py` 看板 CLI 跟踪 19 step 状态 / 备注 / 实际花费 → tracker.json。

**📌 带走**：拷贝即可用的 `capstone_runtime/` bundle，4 个 Mac 已 verify + 15 个目标机 verify。

---

## 💻 · [phase_consumer](./phase_consumer.md) · 消费卡实战路径

**⚡ 要点**
1. 1×4090/5090 已覆盖 phase 0/1/6/7/8 + phase 4 QLoRA + 小规模 GRPO，**80% 认知性工作不用等专业卡**。
2. 不能做的就两件事：**大模型从头预训**（≥ 7B dense / 任何 MoE）+ **百卡级 RL**——这本来也不属于个人主线。
3. 4 个 starter：QLoRA 9B / TRL GRPO 1.5B / Code RAG 50k 文件 / 30B-A3B Int4 部署 + mini-agent demo。

**📌 带走**：24/7 训 4090 = 一个小取暖器（450W）+ ¥7/天电费；`nvidia-smi -pl 350` 降功保稳。

---

## 一页表总览

| Phase | 核心动作 | 入门 GPU | 投入 |
|---|---|---|---|
| 序 · 基础 | 读 + 手算 | 无 | 1 day |
| 0 · 全景 | 评测 baseline 5 题 | 0-1 卡 | 4h |
| 1 · 数据 | datatrove pipeline | CPU + 1 卡 | 1-2 weeks |
| 2 · 架构 | 0.5B-1B MoE 从零 | 8×H100 | 2-4 weeks |
| 3 · mid-train + ctx | 5B token 退火 + RoPE 扩 | 8×H100 | 200 GPU-h |
| 4 · SFT | LoRA 36k 数据 2 epoch | 1-8 卡 | 1-3 days |
| 5 · RL | GRPO 100 step on SWE-Gym | 4-8 卡 | 1-5 days |
| 6 · eval | evalplus + SWE-Bench | 1-2 卡 | 1 day |
| 7 · 部署 | SGLang FP8 + bench | 8 卡 | 1-2 days |
| 8 · Agent | mini-agent + RAG + 真任务 demo | 1 卡 | 1-3 days |
| ⚒ tooluse | 跨章节速读 | — | 1h |
| ✪ capstone | 全流程 4 周 | 8×H100 | 4 weeks |
| 💻 consumer | 没专业卡先做这 4 个 | 1×4090 | varies |

---

**继续阅读** · 翻 [README §按角色读法](./README.md#按角色读法-30-万字怎么读) 挑你的 path · 翻 [📓 phase_failures](./phase_failures.md) 看常见坑 · 翻 [▣ phase_glossary](./phase_glossary.md) 查 48 条术语。
