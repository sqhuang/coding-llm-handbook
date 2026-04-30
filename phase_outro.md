# 结语 · 从读完到真正上手

> 上一章你看完了 Agent 应用的全景。这一章不讲新知识，只回答一个问题：
> **合上这本笔记之后，下周一早上你打开电脑，第一件事做什么？**

笔记到这里 30 万字已经压在脑子里了。问题是知识是惰性的，不上手就会以 7 天为周期蒸发。这一章是收束，但更是一份**操作手册** —— 把"我读过"变成"我能做"。

---

## 1. 你已经掌握了什么

不是恭喜，是盘点。下面这份清单是过完 9 个 phase 之后，你**应当**具备的能力。如果某条勾不上，回去翻对应章节，不要往下走。

| # | 能力项 | 自检问题 | 对应章节 |
|---|--------|----------|----------|
| 1 | 数据流水线 | 给你 10TB GitHub raw dump，能不能在一周内产出可训练的 token？ | Phase 1 |
| 2 | MoE 架构权衡 | 说得清 DeepSeek-V3 的 DSA、GLM-5.1 的 754B 路由策略和 Llama dense 的取舍 | Phase 2 |
| 3 | 长上下文工程 | YaRN / NTK / Position Interpolation 三种扩展法的成本与精度差异 | Phase 3 |
| 4 | SFT 数据配方 | 自己设计一份 coding SFT 数据，能列出领域分布、难度分桶、去重策略 | Phase 4 |
| 5 | RLVR 闭环 | 在小模型上跑通过一次完整 GRPO/DAPO，知道 reward hacking 长什么样 | Phase 5 |
| 6 | 评测设计 | 不只会跑 HumanEval，能解释为什么 SWE-Bench 比它更接近真实 | Phase 6 |
| 7 | 推理部署 | vLLM / SGLang / TensorRT-LLM 三选一，能给出 QPS / TTFT / 成本三轴对比 | Phase 7 |
| 8 | Agent 编排 | 区分 Cline / Cursor / Claude Code 的上下文管理与工具调用差异 | Phase 8 |
| 9 | 全栈视角 | 能从一个线上问题（如"幻觉路径"）回溯到上游某个数据/训练阶段 | Phase 0 + 全 |
| 10 | 成本直觉 | 张口就能估"训一个 70B coding 模型大概多少卡天多少美金" | 全 |
| 11 | 论文消化 | 看到一篇新论文，30 分钟内判断是否值得复现、值得集成 | 全 |
| 12 | 工程审美 | 知道哪些是 hype、哪些是真改进，不会被每周新 SOTA 牵着走 | Phase 0 + 6 |

> **一句话总结**：你现在站在的位置，是"能听懂团队里所有人讲什么"，不是"能独自做出 GLM-5.1"。

---

## 2. 12 周深入路径

每周一个里程碑，每个里程碑都有**可交付物**（GitHub repo / blog / 模型 checkpoint）。不交付不算完成。

| Week | 主题 | 里程碑 | 交付物 |
|------|------|--------|--------|
| W1 | 环境搭建 | 跑通 GLM-5.1 推理（vLLM/SGLang）+ 接入 Cline | 一段录屏 |
| W2 | 评测基线 | 在自己机器上跑完 HumanEval+ / MBPP+ / LiveCodeBench-mini | 一份 baseline.json |
| W3 | 数据小试 | 自建 1k 条高质量 coding SFT 数据（去重 + 验证 + 难度分级） | dataset.parquet + datasheet |
| W4 | LoRA 微调 | 在 7B 开源模型上跑通 LoRA SFT，HumanEval+ 至少 +2pp | 训练日志 + 对比表 |
| W5 | 全参 SFT | 在 14B/32B 上跑全参 SFT，比 LoRA 多涨多少 | wandb 链接 + 复盘 |
| W6 | RLVR 入门 | 用 verl/OpenRLHF 跑一次 GRPO，目标域随便选（数学或简单编码） | 训练曲线 + 1 篇 blog |
| W7 | 长上下文 | 把上面 SFT 模型扩到 128K，过 Needle-in-Haystack | 评测报告 |
| W8 | 部署优化 | 把模型 quantize 到 INT8/FP8，TTFT 与精度的 trade-off 表 | benchmark.md |
| W9 | Agent 集成 | 接入一个真实 Agent 框架（Cline / OpenHands / SWE-agent） | demo 视频 |
| W10 | 私域微调 | 在公司或自己代码库上跑一次领域 SFT | 内部模型 + 评估 |
| W11 | 端到端评测 | 用 SWE-Bench Verified 子集评估端到端表现 | 分数 + 失败案例分析 |
| W12 | 总结产出 | 把整个过程写成一份技术博客，公开发布 | 一篇 5000 字以上博客 |

> **一句话总结**：12 周不是"完整复刻 GLM-5.1"，是建立**自己**的"数据 → 训 → 评 → 部署 → 应用"端到端肌肉记忆。

---

## 3. 6 个推荐项目（按难度递增）

每个项目都按这五元组给：**标题 / 时长 / 涉及章节 / 交付 / 评估指标**。

### Project 1 · "GLM-5.1 + Cline 完成一个真实 PR"
- **时长**：1 个周末（8-12h）
- **涉及**：Phase 7 部署 + Phase 8 应用
- **交付**：在自己开源仓库或 fork 的项目里，由 GLM-5.1 驱动的 Cline 完成一次真实 PR（修 bug 或加 feature），合入主干
- **评估**：PR 是否被合并 / 人工干预次数 / 总耗时

### Project 2 · "私域代码库的 RAG-augmented Agent"
- **时长**：1-2 周
- **涉及**：Phase 8 + Phase 1（数据切分）+ Phase 6（评测）
- **交付**：把 5-10w 行代码（公司内部或大型 OSS 项目）建成 code index，让 Agent 在你的代码库上回答问题/做小改动
- **评估**：在自建 50 条问答集上的命中率 / 改动正确率

### Project 3 · "Coding LoRA：把 Qwen3-Coder-7B 在你最熟悉的语言上再涨 5pp"
- **时长**：2-3 周
- **涉及**：Phase 1 + Phase 4 + Phase 6
- **交付**：一个 LoRA adapter（HuggingFace 上传）+ 数据卡片 + 评测报告
- **评估**：HumanEval+ / MBPP+ / 自建私域 benchmark 的提升幅度

### Project 4 · "RLVR 闭环：用 GRPO 训一个 14B coding 模型"
- **时长**：3-4 周
- **涉及**：Phase 4 + Phase 5 + Phase 7
- **交付**：开源 reward function、训练脚本、最终 checkpoint，附 wandb 全过程
- **评估**：相比纯 SFT 的提升 / reward hacking 案例分析 / 训练成本

### Project 5 · "SWE-Bench 私有版：构建你公司的端到端代码 Agent 评测"
- **时长**：4-6 周
- **涉及**：Phase 6 + Phase 8 + Phase 1
- **交付**：一个仿 SWE-Bench 的私域评测集（issue + golden patch + test），并跑通 GLM-5.1 / Claude / GPT 多家对比
- **评估**：评测集本身的稳定性（多次跑分方差）/ 对模型差异的区分度

### Project 6 · "私有代码 Coding LLM 全套体系：SFT + RLVR + 部署 + Cline 集成"（终极项目）
- **时长**：3 个月
- **涉及**：全部 9 个 phase
- **交付**：
  - 一个私域 coding 模型（基于 GLM-5.1 或 Qwen3-Coder）
  - SFT + RLVR 全流程脚本
  - vLLM/SGLang 部署 + 公司内部 Cline fork
  - 端到端评测报告
  - 总结博客或内部技术分享
- **评估**：在公司私域 benchmark 上超过通用 GLM-5.1 / Claude / GPT，且推理成本可控

> **一句话总结**：先做 Project 1 让自己有反馈，再决定走应用线（2、5）还是训练线（3、4、6）。

---

## 4. 必关注的开源仓库（按用途分组）

### 训练框架
- `volcengine/verl` — 字节 RLHF/RLVR，社区最活跃
- `OpenRLHF/OpenRLHF` — 教育友好的 RLHF
- `hiyouga/LLaMA-Factory` — 一站式 SFT/LoRA/DPO
- `NVIDIA/Megatron-LM` — 大规模预训练标杆
- `huggingface/trl` — 入门必备

### 推理部署
- `vllm-project/vllm` — 事实标准
- `sgl-project/sglang` — 在长文本/Agent 场景上 vLLM 的强力对手
- `NVIDIA/TensorRT-LLM` — 企业级压榨硬件
- `ggml-org/llama.cpp` — 端侧

### Agent / 应用
- `cline/cline` — 开源 Cline，可改可学
- `All-Hands-AI/OpenHands` — SWE-Bench SOTA 之一，可作蓝本
- `princeton-nlp/SWE-agent` — 学术起源
- `aider-AI/aider` — 终端形态参考

### 数据与评测
- `bigcode-project/bigcodebench` — 评测必备
- `princeton-nlp/SWE-bench` — 端到端评测黄金标准
- `huggingface/datatrove` — 大规模数据清洗

> **一句话总结**：这 14 个仓库 fork + star 一遍，至少把 verl、vLLM、Cline 三个的源码读到能改的程度。

---

## 5. 必看论文清单（最近 12 个月）

### 必读（5 篇 · 不读没法做事）
| 论文 | 为什么必读 |
|------|----------|
| GLM-5.1 / GLM-4.5 技术报告 | 你的目标模型 |
| DeepSeek-V3 / R1 技术报告 | 当前 MoE + RL 教科书 |
| Qwen3 / Qwen3-Coder 技术报告 | 中文社区开源标杆 |
| DAPO（字节） | RLVR 工程化代表 |
| SWE-Bench Verified | 评测体系基础 |

### 重要（建议 1 个月内读完）
- Kimi-K2 / K1.5 报告（长上下文 + RL）
- DeepSeek-V3.2 / DSA 系列
- Llama 4 / Mistral 最新报告
- "Self-Rewarding LLM" 系列
- "Process Reward Model" / "PRM800K" 后续

### 建议（按兴趣选读）
- Constitutional AI 后续
- 各家 Agent benchmark 论文（Multi-SWE-Bench、TerminalBench 等）
- 推理优化：FlashAttention-3、Speculative Decoding 新工作

> **一句话总结**：必读 5 篇精读，重要 5 篇略读，建议项按需。**不要追新**，追新只会让你 12 个月里读了 200 篇，但一篇都做不出来。

---

## 6. 该加入的中文社区

| 社区 | 形态 | 推荐理由 |
|------|------|----------|
| 机器之心 / 量子位 | 公众号 | 中文资讯密度最高，5 分钟知道发生了什么 |
| PaperWeekly | 公众号 + 知识星球 | 论文解读偏深，作者多是一线研究员 |
| 知乎"大模型"话题 | 知乎 | 工程一线的 trick 和踩坑 |
| Hugging Face 中文社区 | Discord + 微信群 | 开源生态第一手 |
| ModelScope 社区 | 论坛 | 阿里系开源，国内访问最稳 |
| MLNLP / 北航 / 清华 NLP 实验室公众号 | 公众号 | 学术视角 |
| GitHub trending（每周看） | 网页 | 最快的工程风向标 |
| Discord：vLLM / SGLang / verl 官方 | Discord | 直接和 maintainer 对话 |
| Twitter（X）：见下一节 | X | 英文一线，半天延迟 |
| B 站：李沐、跟李沐学 AI、Andrej Karpathy 中字 | 视频 | 系统讲解 |

> **一句话总结**：选 3 个长期跟，不要全订（信息过载比信息不足更糟）。

---

## 7. 10 个新人陷阱

> 列在前面是因为踩进去太常见了。

1. **追 SOTA 不复现**。读 50 篇论文，自己一行代码不写。**对策**：读 1 篇就动手 1 次。
2. **数据当成苦力活**。把 95% 时间花在调架构上，5% 花在数据。**对策**：反过来。
3. **小模型上能 work 就以为大模型也能**。Scaling 不线性。**对策**：至少在 7B 和 32B 各验证一次。
4. **过度依赖 LoRA**。LoRA 是"快速验证"，不是"最终方案"。**对策**：关键节点必须做全参 SFT 对比。
5. **评测全靠 HumanEval**。HumanEval 已经被刷烂了。**对策**：私域 benchmark + LiveCodeBench + SWE-Bench Verified 三轴。
6. **Reward hacking 不警惕**。RL 跑出 99 分，部署上线大跌眼镜。**对策**：每次 RL 必须有 held-out 真实任务集。
7. **不算成本**。训一次 200 万人民币不知道，老板拍桌才发现。**对策**：每个项目动工前先估 GPU·hour 和美金。
8. **Agent 框架自己重写**。OpenHands / Cline 已经够好，先 fork 再说。**对策**：1 个月内不要从零写 Agent。
9. **私域数据当公开数据用**。合规、版权、license 三个雷区。**对策**：建立 data card 制度，每条数据可追溯。
10. **一个人闷头干**。LLM 是团队运动。**对策**：找 2-3 个对手互相 review，或公开 blog 接受拍砖。

> **一句话总结**：90% 的失败不是技术问题，是**没有反馈循环**。让自己每周都能被打脸一次。

---

## 8. 如何持续追踪 SOTA

### Twitter / X 必关注（按权重）
- `@_philschmid` — 工程化新闻聚合
- `@arankomatsuzaki` — 每天 arXiv 摘要
- `@huybery`、`@JustinLin610` — Qwen 团队
- `@deepseek_ai`、`@Zai_org`（GLM）官方账号
- `@vllm_project`、`@sgl_project` 官方
- `@karpathy` — 视角性最强
- `@Tim_Dettmers` — 量化和系统
- `@cwolferesearch` — 综述和教程

### Newsletter / RSS
- **The Batch**（吴恩达） — 周更，宏观
- **Import AI**（Jack Clark） — 周更，深度评论
- **Sebastian Raschka's Magazine** — 月更，技术深度
- **AK 的 Hugging Face Daily Papers** — 每天的论文 top 10
- **LatentSpace 播客** — 工程一线访谈

### 中文 Weekly
- 机器之心 / 量子位每周精选
- PaperWeekly 周报
- 我爱自然语言处理（52nlp）

### 个人节奏建议
| 频率 | 做什么 |
|------|--------|
| 每天 | 刷 Twitter + HF Daily Papers，5-15 分钟 |
| 每周 | 精读 1 篇论文 + 跑 1 次代码 |
| 每月 | 写 1 篇技术 blog（哪怕 500 字也行）|
| 每季度 | 复盘项目进度，调整 12 周路径 |

> **一句话总结**：信息密度足够大的时代，**纪律 > 努力**。每周 1 篇论文 + 1 次动手，比每天刷 8 小时 X 强 10 倍。

---

## 9. 最后的话

LLM 这一行最快的学习路径是：**读得少、做得多、写得多**。

这本笔记给的是地图，地图替代不了脚程。GLM-5.1 这种 754B 模型你大概率不会从头训一遍 —— 但你不需要。你需要的是：在它的基础上做出**别人做不出的东西**，无论是某个垂直领域的微调、某个工程化的 trick，还是某个被忽视的应用场景。

合上文档，打开终端。下周一这个时候，希望看到你 GitHub 上多一个 commit。

---

> **附**：这本笔记的所有章节（含本章）总计约 30 万字，读完只是起点。如果一年后你回头看，发现某些章节"过时了"—— 那是好消息，说明你已经追上来了。

