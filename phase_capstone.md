# Capstone 实验：4 周 · 8×H100 · 把 GLM-4.5-Air-Base 训成「内部 Python coding 助手」

> 📅 主线快照：2026-05-15 · 上次核对：2026-05-15

> **⚡ 三句话要点**
> 1. 这是一份**端到端可执行**的实验册——把 phase0-8 的所有概念在同一个 4 周项目里走一遍，**每一步都给出卡 / 数据 / 模型 / 超参 / 命令 / 验收 / 思考题**，跑完就拥有 "自己跑过全流程" 的经验。
> 2. 预算锚点：**8×H100 80GB · 4 周 · 总成本 ≈ $4K**（按 H100 按需 $2/hr · 总 ~2000 GPU-hour）；预算紧时退到 4×H100 + 跳过 mid-training 也成。
> 3. 配套 `tools/track.py` 看板 CLI 跟踪 todo / doing / done / blocked 四种状态——所有 19 个 step 的状态、备注、实际花费都进 `tracker.json`，**研究日志不再丢**。
>
> **拷到目标机就跑用的脚手架** → [`capstone_runtime/`](./capstone_runtime/)：19 个 step 脚本（4 个已在 Mac 上 verify、其余是 fail-fast 的模板）+ configs + 19 单元测试 + preflight 自检 + 一份 README 红线说明「我没法保证在你的 H100 集群跑通，你必须 verify 的那些事」。`tar` 一份扔过去 `cd && make setup && make preflight && make test` 即可开工。

---

## 0. 实验设计与边界

### 0.1 目标
- **业务目标**：把一个公开 base 模型微调成"会写公司风格 Python 代码"的 coding assistant，可在内部 IDE 插件里替代 GLM-5.1 API（成本下降 70%、私有代码不外发）。
- **技术目标**：跑通 phase0 → phase8 全 pipeline 一遍，每一步都有可量化的 deliverable，跑完能在面试 / 内部技术分享上**逐 step 复述**。
- **非目标**：不追 SOTA、不打 SWE-Bench 公开榜、不重训 base（资源不够）。

### 0.2 资源边界
| 项目 | 配置 | 备注 |
|---|---|---|
| 主硬件 | **8 × H100 80GB SXM** · NVLink + IB 400G | 公司自建或 Lambda / RunPod / 火山按需 |
| 退路硬件 | 4 × A100 80GB | 跳过 mid-training（step 5）、SFT 改 QLoRA 即可继续 |
| **暂时没专业卡？** | 1-2 × RTX 4090/5090 | 走 [💻 phase_consumer](./phase_consumer.md)：先做 step 01/04/11/17/19 + QLoRA 9B + Code RAG + mini-agent，等专业卡到货前 80% 的认知性工作不用等 |
| 存储 | NVMe ≥ 4TB · 网盘 ≥ 10TB | The Stack v2 Python 子集 + ckpt + 评测产物 |
| 网络 | HF mirror（推荐 modelscope 国内镜像 / hf-mirror.com） | 单次 GLM-4.5-Air-Base 下载 ~200GB |
| 预算上限 | **2000 H100-hour（≈ $4K）** + 数据 / API ~$300 | 超 30% 必须重新评审，否则砍 scope |
| 时间窗 | 4 周（4×40h ≈ 160 工时） | 单人；2 人可压缩到 3 周但通信代价上升 |

### 0.3 验收（north star）
- ✅ HumanEval+ pass@1 **vs base +5pp 以上**
- ✅ 公司内部 SWE-Bench v0（10 题）resolved rate **vs base +10pp 以上**
- ✅ 端到端 RAG + agent demo 在公司一个真实 repo 上能完成 ≥ 3 个"修小 bug"任务
- ✅ 全 19 个 step **全部 done**，每个有 `tracker.json` 记录的实际耗时与备注
- ❌ 任何一项不达，**写一份失败复盘比拿 PR 还重要**——这是研究意义所在

### 0.4 为什么是 GLM-4.5-Air-Base 而不是 GLM-5.1
- GLM-5.1 是 754B MoE，单机 8×H100 跑不动训练（只能推理）；Air-Base 是 106B/12B 激活的 MoE-DSA 变体，**全参 LoRA 可塞进 8×H100**。
- Air-Base 同源 GLM-4.5 ARC，post-training 配方可直接借鉴。
- 替代候选：Qwen3-Coder-30B-A3B-Base（更小但非 GLM 系）、DeepSeek-Coder-V2-Lite-Base（16B MoE 但 Coder-V2 没有 DSA）。

---

## 1. 看板系统：怎么用 `tools/track.py`

```bash
# 看全局
python tools/track.py board

# 看某个 step 详细
python tools/track.py show capstone-04-sft

# 开始一个 step
python tools/track.py start capstone-04-sft

# 加备注（任何时候）
python tools/track.py log capstone-04-sft "lr 调到 1e-4 后 loss 平稳"

# 标记完成（提示填实际耗时 + 实际成本）
python tools/track.py done capstone-04-sft --hours 18 --cost 36

# 阻塞（不影响别的 step）
python tools/track.py block capstone-05-rl --reason "等 reward sandbox 镜像"
```

状态机：`todo → doing → done`，或随时 `→ blocked`，blocked 可回到 `doing`。每条 step 自动盖时间戳。

`tracker.json` 是唯一的 source of truth——可以提交到 git 当研究日志、可以 grep 查"我哪一步花了最多时间"、跨多个实验对比"上次同类 step 实际 vs 估算"。

---

## 2. 主线 step 一览（19 步）

| ID | Phase | Step | ETA | 预算 |
|---|---|---|---|---|
| 01 | 0 | 选基座 + 现状评测（base API） | 4h | $0 |
| 02 | 0 | 搭内部 SWE-Bench v0（10 题） | 16h | $0 |
| 03 | 1 | 数据 pipeline：The Stack Python 5GB + 私有 PR 10k | 24h | $50 |
| 04 | 1 | 评测集去污染（10-gram + MinHash） | 4h | $0 |
| 05 | 2 | 加载 Air-Base + MoE 路由健康度观察 | 6h | $12 |
| 06 | 3 | Mid-training 退火 5B token（WSD） | 200h | $400 |
| 07 | 3 | RoPE 扩到 128K + RULER 验真 | 24h | $50 |
| 08 | 4 | SFT 数据合成：OSS-Instruct 30k + Issue-PR 5k | 8h | $50 |
| 09 | 4 | LoRA SFT r=64 · 2 epoch | 36h | $72 |
| 10 | 4 | Agent 轨迹 1k 条（30 题 ×3 尝试） | 16h | $80 |
| 11 | 5 | 搭 SWE-Gym sandbox + reward 设计 | 16h | $0 |
| 12 | 5 | GRPO 100 step + 反 reward hacking | 80h | $160 |
| 13 | 6 | HumanEval+ / MBPP+ / LiveCodeBench | 12h | $24 |
| 14 | 6 | 内部 SWE-Bench v0 跑 base vs all variants | 16h | $32 |
| 15 | 7 | SGLang FP8 部署 + bench | 12h | $24 |
| 16 | 7 | LoRA adapter 合并 + prefix cache 调优 | 8h | $16 |
| 17 | 8 | Code RAG（tree-sitter + bge + Qdrant） | 24h | $30 |
| 18 | 8 | mini_agent.py 接 deploy endpoint + 真实任务 demo | 12h | $24 |
| 19 | ★ | 复盘：失败 case 分析 + 写一篇内部技术分享 | 16h | $0 |
| | | **合计** | **≈ 534 工时（含等待 / 504 GPU-hour 训练）** | **≈ $1024 计算 + $60 数据** |

实际 GPU-hour 远低于预算 2000h，是因为 mid-training 200h × 8 卡 = 1600 GPU-hour 已经吃掉绝大部分。step 09 / 12 / 14 是次大头。

---

## Step 01 · Phase 0 · 选基座 + 现状评测（base API）

**状态字段**：`capstone-01-baseline`

**输入**：无（项目起点）
**输出**：一份 `baseline_report.md`，包含 GLM-5.1 API / GLM-4.5-Air-Base / Qwen3-Coder-30B-A3B-Base 三家在 HumanEval+ / 内部 5 题手测上的分数

| 字段 | 值 |
|---|---|
| 卡 | 0（API 调用 / 单卡本地 inference） |
| 数据 | HumanEval+（`evalplus`）· 内部 5 题手测题面 |
| 模型 | GLM-5.1 API / 本地 Air-Base / Qwen3-Coder（vLLM 启起来） |
| 超参 | `temperature=0.2, top_p=0.95, n=20`（pass@10 需要 ≥ 20 采样） |
| 命令 | `evalplus.evaluate --dataset humaneval --model glm-5.1 --backend openai` |
| 预算 | 4 工时 + $0（已有 API 额度）；如本地推理另 + 2 H100-hour ≈ $4 |
| 验收 | 三家分数都跑出 + 找出 1 个 base 答错但 GLM-5.1 答对的 case |

**思考 3 问**
1. GLM-5.1 比 Air-Base 在 HumanEval+ 上高 8pp，最可能的来源是哪两块？（提示：phase0 + phase5）
2. Qwen3-Coder-30B-A3B 是 30B 总参 / 3B 激活，FLOPs ≈ Air-Base 的 1/4，但分数只差 3pp——为什么？这说明 coding 任务的瓶颈在算力还是数据？
3. 如果你公司当前 API 月费 < $2K，本项目的 ROI 该怎么算？需要做到什么分数 + 多少 QPS 才回本？

---

## Step 02 · Phase 0 · 搭内部 SWE-Bench v0（10 题）

**状态字段**：`capstone-02-internal-bench`

**输入**：公司任意一个有 PR + CI 的 Python repo
**输出**：`internal_bench_v0/` 目录，含 10 个 instance JSON + `run.sh`

| 字段 | 值 |
|---|---|
| 卡 | 0（runner 跑 docker，不需要 GPU） |
| 数据 | 公司 repo 近 12 个月 merged PR 中"修 bug"标签的 10 个 |
| 模型 | 无（这一步只搭题面 + sandbox） |
| 工具 | `examples/phase6/collect.py` + 官方 `swebench` repo 的 docker base |
| 命令 | `python examples/phase6/collect.py --repo internal/foo --since 2025-05-01 --until 2026-05-01 --label bug` |
| 预算 | 16 工时 + $0（本地 docker） |
| 验收 | 10 题都能 `gold_patch` apply 后 F2P 全过 + base 模型至少答错 6 题（题目难度合格） |

**思考 3 问**
1. 为什么选 10 题不是 100 题？（提示：v0 vs v1 的工程成本 + 边际效用曲线）
2. F2P / P2P 测试集为什么必须分开？合并成"总测试"会有什么 silent failure？
3. 如果 10 题 base 全错（你以为难度合格），但训完模型也全错——你怎么 debug 是"题面表达问题"还是"模型确实做不到"？

> ⚠️ **常见坑** · 把 issue title 直接当 problem_statement——很多 issue 标题就剧透了修法（"AttributeError on line 42"），模型在评测里看到等于作弊。必须用 issue body 的前 N 段 + 自动剥离 stack trace。

---

## Step 03 · Phase 1 · 数据 pipeline

**状态字段**：`capstone-03-data`

**输入**：The Stack v2 Python subset + 公司私有 PR
**输出**：`data/final/*.parquet`（去重后 ≈ 3GB） + `data/private_pr.jsonl`（≈ 50MB）

| 字段 | 值 |
|---|---|
| 卡 | 0-1 张（datatrove CPU 为主；MinHash 巨大集群可跑 GPU 加速版） |
| 数据 | `bigcode/the-stack-v2-dedup` Python subset · 公司 GitLab API 拉 12 个月 PR |
| 模型 | 无 |
| 命令 | `python examples/phase1/run_pipeline.py` · `python examples/phase4/extract_pr_sft.py`（其实 step 08 用） |
| 超参 | MinHash `num_perm=128, threshold=0.8` · 启发式过滤 StarCoder2 阈值 |
| 预算 | 24 工时（数据下载 + 跑 pipeline）+ 单机存储 4TB ≈ $50 |
| 验收 | 去重前后比 ≤ 60%（≥ 40% 被去掉是正常的）· license 全部 permissive · 抽样 100 条人工核查 ≥ 95% 是"真 Python 代码" |

**思考 3 问**
1. The Stack v2 已经是 dedup 版了，为什么还要再跑一次 MinHash？（提示：跨 split / 跨语言 / unicode）
2. 公司私有 PR 占总训练 token 的 5%——这个比例够"对齐公司风格"吗？多了会怎样、少了会怎样？
3. 如果你只能保留一项数据处理（在 license / heuristic / dedup / decontam 里），保哪一项？为什么？

---

## Step 04 · Phase 1 · 评测集去污染

**状态字段**：`capstone-04-decontam`

**输入**：step 03 产出的 parquet + 评测集题面（HumanEval+ / MBPP+ / LiveCodeBench / 内部 v0）
**输出**：`data/clean/`（清掉污染样本后的数据）+ `decontam_report.md`

| 字段 | 值 |
|---|---|
| 卡 | 0（CPU） |
| 数据 | 评测集所有 prompt 的 10-gram 索引 |
| 模型 | 无 |
| 命令 | 自写脚本：拉评测题面 → 10-gram exact match + MinHash 双扫 → 命中样本删除 |
| 预算 | 4 工时 + $0 |
| 验收 | 命中率 < 0.1% 样本（如果 > 1% 说明 pipeline 上游漏了，回 step 03） |

**思考 3 问**
1. HumanEval 题面只有 20-50 token，10-gram 几乎覆盖全题——这种情况 MinHash 还有价值吗？
2. LiveCodeBench 是滚动新增题目，你怎么持续跑去污染？（提示：评测题面订阅 + 增量索引）
3. 如果你发现 The Stack v2 里有原版 HumanEval 题面——这是 BigCode 的 bug 还是你的 bug？怎么验证？

> ⚠️ **常见坑** · 只查"题面"不查"题解"——有些数据集（如某些 Codeforces dump）直接收录了 HumanEval 题面 + 解答，10-gram 扫题面会漏题解块。**双向扫**：题面 + 经典解答。

---

## Step 05 · Phase 2 · 加载 Air-Base + MoE 路由健康度观察

**状态字段**：`capstone-05-arch-probe`

**输入**：HF `zai-org/GLM-4.5-Air-Base` 权重
**输出**：`probe_report.md`（含 expert_load 直方图 + DSA top-k 分布）

| 字段 | 值 |
|---|---|
| 卡 | 8 × H100 80GB（FP8 加载需要 ≥ 4 卡 TP） |
| 数据 | 一份 1k 条 Python 代码片段（取 step 03 抽样） |
| 模型 | `zai-org/GLM-4.5-Air-Base`（106B/12B 激活，MoE-DSA） |
| 命令 | `python -m vllm.entrypoints.openai.api_server --model zai-org/GLM-4.5-Air-Base --tensor-parallel-size 8 --quantization fp8` |
| 超参 | TP=8, FP8 KV cache, max_model_len=32768 |
| 预算 | 6 工时 + 6 GPU-hour ≈ $12 |
| 验收 | 8 张卡显存占用 ≈ 50GB / 张 · expert_load_var < 0.2 · 不报 OOM |

**思考 3 问**
1. expert_load_var 怎么测？写一个 hook 抓 router logits 还是看 SGLang/vLLM 暴露的 metric？
2. 如果某些 expert load > 5× 平均值，是 base 训坏了还是你的 prompt 分布太偏？（提示：跑两份 prompt——Python 代码 vs 通用对话——对比 routing 直方图）
3. DSA 的 top-k 在不同 prompt 长度上怎么变化？短 prompt 是否能省 indexer 那一步？

---

## Step 06 · Phase 3 · Mid-training 退火 5B token

**状态字段**：`capstone-06-midtraining`

**输入**：step 04 的 clean 数据 + Air-Base
**输出**：`ckpt/midtrain_final/`（5B token 后的权重）

| 字段 | 值 |
|---|---|
| 卡 | 8 × H100 80GB（FSDP / DeepSpeed ZeRO-3） |
| 数据 | 5B token = 70% 代码（step 03）+ 20% 中文通用（CCI-3）+ 10% 数学（OpenMath-2） |
| 模型 | step 05 加载的 Air-Base |
| 框架 | torchtitan（推荐）或 LLaMA Factory continue-pretrain |
| 超参 | lr 2e-5（peak）· WSD：1B warmup → 3B stable → 1B decay-to-0 · global_bs=512 · seq_len=8192 · BF16 + FP32 master · grad_clip=1.0 |
| 命令 | `torchrun --nproc 8 train.py --config configs/midtrain_air.toml` |
| 预算 | 200 工时（自然时间）+ 8 × 25h GPU = 200 GPU-hour ≈ $400 |
| 验收 | code_loss 比 web_loss 多降 0.5+ · annealing 段 loss 二次下降明显 · expert_load_var 不漂移 |

**思考 3 问**
1. WSD 的 stable 段你设了 3B token，但实测 loss 还在下降——你会延长 stable 还是按计划进入 decay？背后的 trade-off 是什么？
2. 5B token 是个小数（base 训了 23T）——这个量级真能"挪动"权重吗？怎么验证不是"过拟合到这 5B 的偏置"？
3. 跑到 80% 时发现某张卡掉了——重启从 ckpt 续训和重新开始的 cost / 风险对比？

> ⚠️ **常见坑** · 用 base 的 RoPE base（10000）做 5B mid-training，但训练 seq_len=8192 远短于 base 原生 32K——结果模型反而"忘了" base 的长上下文能力。**保持 base 的 RoPE 配置不变**，或预先按 step 07 扩长再训。

---

## Step 07 · Phase 3 · RoPE 扩到 128K + RULER 验真

**状态字段**：`capstone-07-longctx`

**输入**：step 06 ckpt
**输出**：ckpt + `ruler_report.md`（不同 ctx 的 needle 检索准确率）

| 字段 | 值 |
|---|---|
| 卡 | 8 × H100 80GB（推理为主，训练只动 RoPE 不需要全反传） |
| 数据 | RULER 官方 13 task · LongBench-v2 sample |
| 模型 | step 06 ckpt |
| 操作 | 改 `config.json` 的 `rope_scaling`：YaRN factor=4.0, original_max=32768 |
| 命令 | `python tools/patch_rope.py --ckpt midtrain_final --strategy yarn --factor 4.0` 然后 `python -m ruler.eval --model patched_ckpt` |
| 超参 | YaRN beta_fast=32 beta_slow=1 · attention_factor=0.1 |
| 预算 | 24 工时 + 12 GPU-hour ≈ $24 |
| 验收 | RULER@128K needle 检索 ≥ 85% · LongBench-v2 比 base 提升或持平（不掉就行） |

**思考 3 问**
1. YaRN factor=4.0 把 32K 扩到 128K，但其实只算"4×"——为什么不能拍 factor=16 一步到位 512K？
2. RULER 13 个 task 难度不一，哪几个最能体现"真长上下文"？哪几个只是文本召回？
3. 如果你的目标场景永远不会 > 64K，扩到 128K 是浪费吗？（提示：训练成本 vs 推理 KV cache 成本）

---

## Step 08 · Phase 4 · SFT 数据合成

**状态字段**：`capstone-08-sft-data`

**输入**：step 03 私有 PR + OSS-Instruct seed + step 02 agent 轨迹
**输出**：`sft_data.jsonl`（≈ 36k 条，混比按 phase4 §10.8）

| 字段 | 值 |
|---|---|
| 卡 | 1 × H100（调 GLM-4.5-Air 合成 instruction）或 API |
| 数据 | The Stack Python 200 段 seed → OSS-Instruct 30k 条 / `examples/phase4/extract_pr_sft.py` 抽 5k PR / step 10 agent 轨迹 1k |
| 模型 | 合成用 GLM-4.5-Air-Instruct（不是 base）或 GLM-5.1 API |
| 命令 | `python tools/oss_instruct_gen.py --seeds 200 --output sft.jsonl --target 30000` |
| 超参 | temperature=1.0 多样性 · 每条 seed 生成 N=150 个 instruction → quality filter 留 1 个 |
| 预算 | 8 工时 + 合成 API 调用 ≈ $50 |
| 验收 | 36k 条 · 抽样 100 条人工核 ≥ 90% "instruction 不泄漏 response 内容" · `apply_chat_template` 渲染零报错 |

**思考 3 问**
1. OSS-Instruct 30k vs 私有 PR 5k 的混比 6:1——如果你想模型"更像公司风格"，怎么调？为什么不直接 1:1？
2. 合成时用 base 的 instruct 版（Air-Instruct）vs 用更强的 GLM-5.1 API，质量差多少？成本差多少？
3. 用 Air-Instruct 合成训 Air-Base，会不会有"自我蒸馏"的 risk？怎么避免？

---

## Step 09 · Phase 4 · LoRA SFT r=64 · 2 epoch

**状态字段**：`capstone-09-sft-train`

**输入**：step 08 数据 + step 07 ckpt
**输出**：`ckpt/sft_lora_r64/` adapter

| 字段 | 值 |
|---|---|
| 卡 | 8 × H100 80GB（LoRA 单机够） |
| 数据 | step 08 的 36k 条 SFT data |
| 模型 | step 07 ckpt（已退火 + 长上下文） |
| 框架 | LLaMA Factory 0.9+ |
| 超参 | LoRA r=64, alpha=128, dropout=0.05, target=all-linear · lr=1e-4 cosine · global_bs=128 · seq_len=8192 · 2 epoch · packing on · neftune α=5 |
| 命令 | `llamafactory-cli train configs/sft_air_lora.yaml` |
| 预算 | 36 工时 + 8 × 4.5h GPU = 36 GPU-hour ≈ $72 |
| 验收 | train loss 平滑下降 0.6 → 0.4 · 中间 ckpt @ epoch 1 HumanEval+ 比 base +3pp |

**思考 3 问**
1. LoRA r=64 vs 全参 SFT 在 36k 数据上分数差几 pp？（猜，然后跑短 ablation 验证）
2. 为什么 packing 必开？不开会让有效 batch 是多少？
3. neftune α=5 加噪是 free lunch 吗？什么场景下反而有害？

> ⚠️ **常见坑** · LoRA target_modules 只设 q/k/v_proj，会让 SFT 学不到 FFN 层的工程知识；**all-linear**（含 gate/up/down）是 2026 主流默认。

---

## Step 10 · Phase 4 · Agent 轨迹合成

**状态字段**：`capstone-10-agent-traj`

**输入**：step 02 的内部 v0 10 题 + 一个公开仓库 20 题
**输出**：`agent_traj.jsonl`（≈ 1k 条，4 形态混合）

| 字段 | 值 |
|---|---|
| 卡 | 1 × H100 80GB（跑 ReAct loop 的 inference） |
| 数据 | 30 题 × 30-40 次尝试 = 1000+ 条轨迹（保留 ≥ 30 条 final 通过的） |
| 模型 | 收集器：GLM-5.1 API 或本地 Air-Instruct |
| 工具 | `examples/phase8/mini_agent.py` 改造成轨迹采集器 |
| 命令 | `python tools/agent_collector.py --tasks 30 --attempts 35 --out agent_traj.jsonl` |
| 预算 | 16 工时 + 10 GPU-hour（如果用本地 inference）≈ $80 |
| 验收 | ≥ 30 条 final passing · 轨迹平均 6-12 turn · loss mask 通过 chat_template `return_assistant_tokens_mask` 验证 |

**思考 3 问**
1. 30 题 × 35 尝试 → 1050 条轨迹，只留 30 条 final pass——剩下 1020 条 fail 轨迹完全丢吗？（提示：失败轨迹也能给信号）
2. 用 GLM-5.1 当老师会被 distill 警察盯上吗？许可证条款怎么写的？
3. 轨迹长度 6-12 turn 是经验值——更长（30 turn+）的"难题轨迹"对 phase 5 RL 价值是高还是低？

---

## Step 11 · Phase 5 · 搭 SWE-Gym sandbox + reward 设计

**状态字段**：`capstone-11-rl-env`

**输入**：step 02 的 10 题 + step 10 的 30 题（共 40 题做 RL 训练集）
**输出**：`sandbox/` docker 镜像 + `reward.py` 配置

| 字段 | 值 |
|---|---|
| 卡 | 0（sandbox 是 docker） |
| 数据 | 40 题各自的 docker image + F2P/P2P 测试 |
| 模型 | 无 |
| 工具 | SWE-Gym 仓库的 `swegym.envs.Container` + 自写 reward fn |
| Reward | sparse: F2P 通过率 1.0 / fail 0.0 · dense: git_diff 触达正确文件 +0.1 / lint pass +0.05 · anti-hack: 测试文件被改 -2.0 / `@pytest.skip` 出现 -1.0 |
| 预算 | 16 工时 + $0 |
| 验收 | 40 个 task 都能 `env.reset() → env.step(patch) → env.score()` 跑通 · 整套一次 episode < 5min |

**思考 3 问**
1. anti-hack 惩罚 -2.0 比 sparse reward +1.0 还重——会不会导致模型"宁可不修也不动测试"卡死？
2. dense reward 给 0.1 是不是太少？给 0.5 会不会盖过 sparse？怎么选数值（grid search vs 一次定）？
3. 40 题平均一个 episode 4min，100 step RL × group_size 8 = 8 × 100 × 4 = 53h 单纯 rollout——能怎么并行加速？

---

## Step 12 · Phase 5 · GRPO 100 step

**状态字段**：`capstone-12-rl-train`

**输入**：step 09 SFT ckpt + step 11 sandbox + step 11 reward
**输出**：`ckpt/rl_grpo/` adapter

| 字段 | 值 |
|---|---|
| 卡 | 8 × H100 80GB（4 卡 trainer + 4 卡 vLLM rollout，VERL 风格） |
| 数据 | step 11 的 40 题 RL 训练集 |
| 模型 | step 09 SFT 合并后的权重 |
| 框架 | VERL 0.3+ 或 OpenRLHF 0.6+（推荐 VERL，更适合多卡 async） |
| 超参 | LoRA r=16（在 SFT 之上叠）· lr=5e-6 · group_size=8 · KL beta=0.04 · clip=0.2 · max_completion=2048 · 100 step（约 32k samples） |
| 命令 | `bash examples/phase5/grpo_run.sh`（基于 `examples/phase5/grpo_humaneval.py` 改成 SWE-Gym 版本） |
| 预算 | 80 工时 + 8 × 10h GPU = 80 GPU-hour ≈ $160 |
| 验收 | reward 均值从 0.15 涨到 ≥ 0.30 · KL < 5.0 始终 · 验证集 8 题 resolved rate +5pp |

**思考 3 问**
1. group_size=8 你估算 reward 方差能稳住吗？想想 40 题里有多少是"全 0 或全 1"的极端 batch？
2. 跑到 50 step reward 卡在 0.20 不动——你怎么 debug：是 KL 太紧、reward 设计漏洞、还是 SFT base 上限到了？
3. GRPO 训完，KL 从 0 升到 4.5——这个模型还像 SFT 模型吗？怎么定量验证"没走太远"？

---

## Step 13 · Phase 6 · 公开评测套

**状态字段**：`capstone-13-pub-eval`

**输入**：base / step 06 / step 09 / step 12 共 4 个 ckpt
**输出**：`eval_pub.csv`（4 模型 × 3 评测 = 12 个分数）

| 字段 | 值 |
|---|---|
| 卡 | 1-2 × H100（vLLM serve） |
| 数据 | HumanEval+ / MBPP+ / LiveCodeBench（time window 2025-10+） |
| 模型 | 4 个 ckpt 顺序跑 |
| 工具 | `evalplus` + `livecodebench` 官方 runner |
| 超参 | n=20 采样 · temperature=0.2 · 报告 pass@1 / pass@10 |
| 预算 | 12 工时 + 12 GPU-hour ≈ $24 |
| 验收 | step 12 ≥ base +5pp on HumanEval+ pass@1 · LiveCodeBench 不掉点（"通用退化"红线） |

**思考 3 问**
1. 4 个 ckpt 评测顺序：你先跑 base 还是先跑 RL？（提示：发现 evaluator bug 时返工成本）
2. LiveCodeBench 比 HumanEval 难得多，分数从 35 → 38 算"提升"还是"噪声"？怎么算显著性？
3. 如果发现 step 12 在 HumanEval+ 涨了 7pp 但 LiveCodeBench 掉了 4pp——你怎么取舍？

---

## Step 14 · Phase 6 · 内部 SWE-Bench v0 评测

**状态字段**：`capstone-14-internal-eval`

**输入**：step 02 的 10 题 + 4 个 ckpt
**输出**：`eval_internal.csv`

| 字段 | 值 |
|---|---|
| 卡 | 1-2 × H100 + docker | 
| 数据 | step 02 的 10 题 |
| 模型 | 4 个 ckpt |
| 工具 | `examples/phase8/mini_agent.py` 跑每题 ≤ 25 turn |
| 超参 | temperature=0.2 · max_turn=25 · sandbox timeout=600s · 3 次采样取多数 |
| 预算 | 16 工时 + 8 GPU-hour ≈ $32 |
| 验收 | step 12 resolved rate ≥ base +10pp · 每题手工看 1 条失败轨迹做 root-cause |

**思考 3 问**
1. 10 题分数从 1/10 涨到 3/10——这"+20pp"在统计上有意义吗？（提示：Wilson 区间）
2. 你 SFT 数据里用过 step 02 的题吗？如果用过，本次评测算不算自评？
3. 哪一题 base 答对而 step 12 答错——这种"回归"比"没提升"更值得分析，为什么？

> ⚠️ **常见坑** · 用 `temperature=0` greedy 评测 agent task——一条采样卡死或工具调用格式错就整题挂；agentic 任务一定要 ≥ 3 次采样取多数表决。

---

## Step 15 · Phase 7 · SGLang FP8 部署

**状态字段**：`capstone-15-deploy`

**输入**：step 12 ckpt
**输出**：常驻 endpoint `http://0.0.0.0:30000/v1`

| 字段 | 值 |
|---|---|
| 卡 | 8 × H100 80GB（TP=8） |
| 数据 | 无（部署） |
| 模型 | step 12 ckpt merge LoRA 后 |
| 框架 | SGLang 0.4+ |
| 超参 | `--quantization fp8 --tp 8 --enable-mixed-chunk --enable-prefix-caching --max-running-requests 64 --max-total-tokens 1048576` |
| 命令 | `python -m sglang.launch_server --model-path merged_ckpt --quantization fp8 --tp 8 ...` |
| 预算 | 12 工时 + 12 GPU-hour bench ≈ $24 |
| 验收 | TTFT < 500ms @ 4k prompt · throughput ≥ 800 tok/s @ batch 16 · FP8 vs BF16 HumanEval+ 差距 < 1pp |

**思考 3 问**
1. FP8 在 long-ctx（> 64K）上是否仍然 < 1pp 掉点？怎么验？
2. `--enable-prefix-caching` 在 agent 场景命中率多少？怎么测？
3. 8×H100 跑 12B 激活的 MoE 算大材小用还是刚好？4×H100 能不能跑、TTFT 涨多少？

---

## Step 16 · Phase 7 · LoRA 合并 + prefix cache 调优

**状态字段**：`capstone-16-deploy-opt`

**输入**：step 15 endpoint
**输出**：`deploy_report.md`（bench 对比 + 决策记录）

| 字段 | 值 |
|---|---|
| 卡 | 8 × H100 |
| 数据 | 100 个真实 agent prompt 样本 |
| 模型 | step 15 endpoint |
| 工具 | SGLang `bench_serving` + 自写 prefix-hit logger |
| 超参 | 测试 4 组配置：base / +prefix-cache / +chunked-prefill / +speculative |
| 预算 | 8 工时 + 8 GPU-hour ≈ $16 |
| 验收 | 找出最优配置 + 写出 trade-off 表（吞吐 vs 首 token 延迟 vs 显存） |

**思考 3 问**
1. speculative decoding 用 GLM-4.5-Air-Base 自己当 draft 行不行？draft 模型选什么最划算？
2. prefix cache 在 RAG 场景命中率应 > 80%——如果只有 30% 是哪里漏了？
3. chunked prefill 在 200K prompt 上吐 token 顺滑很多但首 token 慢——什么业务接受这种 trade-off？

---

## Step 17 · Phase 8 · Code RAG

**状态字段**：`capstone-17-rag`

**输入**：公司一个真实 ≥ 50k 文件的 monorepo
**输出**：`rag/` 服务（Qdrant + bge + reranker）

| 字段 | 值 |
|---|---|
| 卡 | 1 × A100 40GB（embedding inference） + Qdrant CPU 集群 |
| 数据 | repo 全量代码 + git log |
| 模型 | `BAAI/bge-code-v1`（embedding）· `BAAI/bge-reranker-v2-m3`（rerank） |
| 工具 | `tree-sitter-languages` 切块 · `qdrant-client` · 自写 hybrid search |
| 超参 | 切块 max=1024 tok / overlap=128 · embedding dim=1024 · top_k_dense=50, top_k_bm25=50 → rerank top 5 |
| 命令 | `python tools/rag_index.py --repo /path/to/monorepo --backend qdrant://localhost:6333` |
| 预算 | 24 工时 + 5 GPU-hour ≈ $30 |
| 验收 | 50 条手写 query Recall@5 ≥ 80% · 索引 50k 文件 < 6h · 增量更新（一个 commit）< 30s |

**思考 3 问**
1. tree-sitter 切块在 Python 上 boundary 很自然，但 SQL / proto / yaml 怎么切？退化到行级吗？
2. dense + BM25 hybrid，权重 alpha=0.5 是默认——什么场景下应该偏 BM25（α<0.3）？
3. reranker bge-v2 比 dense 多 10pp Recall@5 但延迟翻 2×——agent 场景接受吗？

---

## Step 18 · Phase 8 · mini_agent 接 endpoint + demo

**状态字段**：`capstone-18-agent-demo`

**输入**：step 15 endpoint + step 17 RAG
**输出**：`demo/` 一个能跑通 3 个真实任务的 mini-agent + 录屏

| 字段 | 值 |
|---|---|
| 卡 | 跑在 step 15 / 17 的卡上 |
| 数据 | 3 个真实公司 task：(a) 修一个 import bug (b) 加一个 CLI 选项 (c) 重构一个 50 行函数 |
| 模型 | step 15 endpoint |
| 工具 | `examples/phase8/mini_agent.py` + step 17 的 `rag.search` tool 注入 |
| 超参 | max_turn=20 · auto_compact_threshold=80% · context 64K |
| 预算 | 12 工时 + 12 GPU-hour ≈ $24 |
| 验收 | 3 个任务至少 2 个一次通过 · 失败的那 1 个能给清晰 error trace · 录屏 ≤ 5min |

**思考 3 问**
1. 3 个任务的难度梯度（修 bug / 加 feature / 重构）你期待 success rate 是不是单调下降？为什么？
2. agent 调 RAG 的频率：每轮都查还是只在不确定时查？怎么让模型自己决定？
3. 录屏给老板看的话，你会展示成功 case 还是展示一个失败 case + 你的复盘？哪个更"研究"？

---

## Step 19 · 复盘 · 失败 case 分析 + 内部技术分享

**状态字段**：`capstone-19-retro`

**输入**：tracker.json 全部记录 + 4 周内所有 ckpt / 评测 / 日志
**输出**：`RETRO.md`（≥ 3k 字）+ 一次 30 分钟内部分享 slides

| 字段 | 值 |
|---|---|
| 卡 | 0 |
| 数据 | 自己 4 周的全部产物 |
| 模型 | 无 |
| 内容要点 | (1) 哪一步严重超预算？为什么？(2) 哪一步收益意外好？(3) 如果重来一遍最大的改动是什么？(4) "用通用 base + 私有数据 SFT" 真的比 "纯 API + RAG" 强吗？算总账。(5) tracker.json 里哪一类 step 的"估算 vs 实际"误差最大？ |
| 预算 | 16 工时 + $0 |
| 验收 | RETRO.md ≥ 3k 字 · 内部分享至少有 5 个同事提问且 ≥ 2 个被采纳到下一期 |

**思考 3 问（也是分享题）**
1. 用一句话总结这 4 周——是"工程练习"还是"研究产出"？
2. 如果给你 16 周而不是 4 周，scope 改怎么扩？（提示：phase2 全参 / phase5 用 SWE-Gym 全量题 / 重训 base 部分层）
3. 这一个实验对你的下一个项目最大的工程影响是什么？（不是模型分数，是"你以后会怎么做"）

---

## 3. Tracker.json 数据模型

```json
{
  "experiment_id": "capstone-air-2026Q2",
  "started": "2026-05-15",
  "budget_usd": 4000,
  "budget_gpu_hours": 2000,
  "steps": [
    {
      "id": "capstone-01-baseline",
      "phase": 0,
      "name": "选基座 + 现状评测",
      "status": "todo",
      "owner": "sq",
      "gpu": "0 (API only)",
      "data": "HumanEval+ · 内部 5 题",
      "model": "GLM-5.1 API / Air-Base / Qwen3-Coder-30B-A3B",
      "hparams": {"temperature": 0.2, "n": 20},
      "eta_hours": 4,
      "eta_cost_usd": 0,
      "actual_hours": null,
      "actual_cost_usd": null,
      "started_at": null,
      "done_at": null,
      "blocked_reason": null,
      "log": []
    }
  ]
}
```

每次 `track.py log` 追加 `{"at": "2026-05-15T14:23:01Z", "msg": "..."}`，永不删除——这是研究日志。

---

## 4. 实验完成后的产物清单

- `tracker.json` 全部 19 step done + 实际数字
- `ckpt/midtrain_final/` / `ckpt/sft_lora_r64/` / `ckpt/rl_grpo/`
- `data/clean/` / `data/private_pr.jsonl` / `sft_data.jsonl` / `agent_traj.jsonl`
- `eval_pub.csv` + `eval_internal.csv` + `decontam_report.md` + `ruler_report.md` + `deploy_report.md`
- `RETRO.md` + 内部分享 slides
- 一段 ≤ 5min 的 demo 录屏

把以上 7 类产物打包一份 release，就有了下一份简历 / 内部晋升答辩 / 给老板要预算的素材。

---

## 📌 章末检查

**带走这 5 条**
- 4 周内跑完 phase0-8 全链路是**可达的**——前提是 8×H100 + 4-5k 美元预算 + 不追 SOTA。
- mid-training 200 GPU-hour 占总训练预算 80%，是单步最大头；预算紧时可跳过（直接 SFT），代价是 base→SFT 提升上限 -3 ~ -5pp。
- LoRA r=64 + SFT + GRPO 是 2026 单机最经济组合；全参 SFT 需要 32×H100 起。
- tracker.json 不是花架子——研究里 50% 的时间损耗源于"忘了上次怎么调的"，看板把它从 50% 压到 5%。
- 评测套必须 step 13（公开）+ step 14（内部）双口径，缺一个都会被反问。

**自检 3 题**（< 5 分钟）
1. 如果只有 4×H100，你会砍掉哪 3 步、怎么调整剩余步骤的参数？
2. step 12 reward 卡死，最快的 root-cause 检查清单 5 条是什么？
3. RETRO.md 里"估算 vs 实际"通常哪一类 step 误差最大？（数据 / 训练 / 评测 / 部署 / 复盘）

<details><summary>参考答案</summary>

1. 砍 step 06 / 07 / 11（mid-training + 长上下文扩展 + RL sandbox）；SFT 改 QLoRA r=32 单机能跑；RL 用 DPO 数据替代 GRPO；step 17 RAG 用 FAISS 替代 Qdrant。
2. (a) reward 是否一直为 0（sandbox 跑挂？）；(b) KL 是否爆（lr 太大）；(c) entropy 是否塌（探索失败 / temperature 太低）；(d) group_size 内方差（全 0/全 1 batch 多）；(e) anti-hack 罚是否压制了正常修复路径。
3. **数据**类（step 03 / 08 / 10 / 17）——下载 / 清洗 / 合成的边界 case 最多；其次是 step 19 复盘（永远比想的久）。训练 / 评测一旦跑通时间很准。
</details>

> ⚠️ **常见坑** · 4 周计划开局两天就因 HF 下载失败 / docker 网络不通 / GPU 调度排队损失 1-2 天——**第 1 周永远把"环境就绪"作为 step 0**，独立排个 8h 时间盒，跑通 `vllm serve base + evalplus + docker hello-world` 三件套再开计时。

**下一步** → 打开 `tools/track.py board` 启动你的 4 周。术语速查 → [▣ 索引](./phase_glossary.md)。
