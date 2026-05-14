# Phase 3：Mid-training 与长上下文扩展深度笔记

> 📅 主线快照：2026-04-22 · 上次核对：2026-04-30

> **⚡ 三句话要点**
> 1. 长上下文走两段路：**32K 用 YaRN**（NTK-by-parts，按波长分维度插值），**128K+ 用 LongRoPE**（进化搜索每维 rescale 因子）。
> 2. Mid-training 数据三件套：**repo-level packing 必须按 import 图拓扑排序**（不是随机 shuffle）+ synthetic reasoning 长 CoT + agent 轨迹，序列长度直接拉到 128K。
> 3. RULER 是验证长上下文"真能用"的关键 benchmark——base 改大会降低长序列 PPL 但短序列 PPL 微涨，所以 mid-training 必须**渐进升级**而非一次切到底。

> 目标：GLM-5.1（官方口径 200K context，agentic 任务支持连续 8 小时运行）
> 主要参考：GLM-4.5 ARC（arXiv:2508.06471）、YaRN（2309.00071）、LongRoPE（2402.13753）、LongLoRA（2309.12307）、DeepSeek-V3 技术报告、Qwen3 技术报告
> 面向读者：中国 AI 研究者 / LLM infra 与算法工程师
> 写作日期：2026-04-22

> **读者画像** · 想把一个 4K/8K 上下文的小 base 模型推到 128K+ 的 infra/算法工程师；或想审计自己模型 long-ctx 能力是否真实的训练 lead。
> **前置知识** · phase2 §1 MLA + RoPE 实现、phase1 §0.4 数据配比、序.11 attention 机制；读过 GLM-4.5 ARC 或 DeepSeek-V3 任一篇。
> **学完能做** · 设计并跑通一次完整的 mid-training（数据配方切换 + WSD lr + RoPE rescale + repo packing），并用 RULER 验证长上下文是否"真的能用"。

---

## 0. 全局定位：Phase 3 在完整训练管线中的位置

一个现代 LLM 的训练管线，今天基本稳定成以下 5 段：

```
Phase 1  Pretrain (general)        → 15T~30T tokens, 4K/8K ctx, 通用语料
Phase 2  Pretrain (domain upsamp)  → 继续 2T~5T，代码/数学/多语权重上调
Phase 3  Mid-training (annealing)  ← 本文重点
         ├── 3a  数据配方切换：高质量 + 推理 + tool 轨迹
         ├── 3b  学习率退火（cosine → 二次衰减 / WSD）
         └── 3c  Long-context extension（RoPE 改写 + 长样本继续训）
Phase 4  SFT / Instruction Tuning
Phase 5  RLHF / RLVR / Agentic RL
```

Phase 3 是"把毛坯房精装修"的阶段。Phase 1/2 出来的模型已经见过大量 token，但分布偏向 web，质量参差、推理链条稀薄、长上下文经验几乎为零。Phase 3 的两大任务：**(1) 用高质量数据把模型"退火收敛"到一个更好的 loss 面局部最优；(2) 把位置编码和注意力能力从 4K/8K 推到 128K/200K+**。

GLM-4.5 ARC（2508.06471）把这段明确称作"mid-training"，并指出其对下游 coding / agentic 能力的提升甚至超过同规模的 SFT。Qwen3、DeepSeek-V3、Llama-3.1 也都有等价阶段（Qwen 叫 "annealing"，DeepSeek 叫 "context extension + reasoning mix"）。

---

## 1. Mid-training 是什么？为什么现代训练必加

### 1.1 历史背景

2023 年前，主流做法是：Pretrain（一条 cosine 到底）→ SFT → RLHF。这个流水线在 GPT-3.5 时代够用，但暴露了两个问题：

1. **Pretrain 末期的数据利用率极低**：学习率降到 10%，模型几乎不再真正更新权重，却还在消耗高质量数据预算。
2. **SFT 阶段数据量小（百万级）**，不足以把"真正有用的能力"（复杂推理、多步工具调用、长上下文检索）塞进参数里，SFT 只能做表层对齐。

Chinchilla 之后大家意识到：**数据质量 > 数据数量**，而且"高质量数据应该放在学习率较高但模型已经见过世面的阶段"——这正是 Phase 3 的定义。

### 1.2 Mid-training 的三条核心定义

综合 GLM-4.5 ARC 与同期 SOTA（Qwen3、DeepSeek-V3、MiniCPM-4、Yi-1.5）的做法：

- **时间位置**：Pretrain 末期 / SFT 前，占总 token 预算的 3%–10%（典型 500B–2T tokens）。
- **数据变化**：剧烈上采样高质量子集 —— 代码、数学、推理 CoT、教科书、tool-use 合成轨迹。通用 web 下采样至 20%–30%。
- **学习率变化**：从 pretrain 的稳定段切到**二次退火**：cosine 降到 min_lr 后再进行一次线性 decay，或直接使用 WSD（Warmup-Stable-Decay）。

### 1.3 Mid-training 带来的能力跃迁（GLM-4.5 ARC 观察）

| 能力            | Pretrain 后 | Mid-training 后 | 增益原因                             |
| --------------- | ----------- | --------------- | ------------------------------------ |
| HumanEval pass@1 | 52          | 71              | 代码数据从 8% → 35%                   |
| GSM8K CoT        | 48          | 78              | 合成推理链数据                        |
| MMLU             | 70          | 73              | 教科书子集                            |
| Needle 4K        | 95          | —               | —                                    |
| Needle 128K      | 12          | 93              | Long-context extension（本章 §4–§6）|

**关键洞察**：Mid-training 阶段的 $\Delta$ 性能 per token 比 Pretrain 阶段高 3–5 倍。这是因为此时模型参数已具备通用表征，给它"高价值 token" 能直接把表征重组到更好的流形上。

---

## 2. Mid-training 数据配比

### 2.1 GLM-4.5 ARC 披露的配方（约）

```
Mid-training total budget: ~1.5T tokens
├── High-quality code         35%   (许可友好 GitHub + StackExchange + 教科书)
├── Math + reasoning CoT      20%   (合成 + 开源 MATH/Olympiad)
├── Tool-use trajectories     10%   (agent 合成数据，下文详述)
├── Textbooks / wiki / paper  15%   (arXiv, S2ORC, Anna's Archive 授权子集)
├── Long-document / book      10%   (for long-ctx warmup, 32K+ 样本)
├── General web (high-qual)   10%
```

### 2.2 每个子集的关键做法

**高质量代码**：
- 用静态分析过滤 —— AST 可解析、lint 无 fatal、license SPDX 白名单。
- 按仓库 star/fork 做质量加权，star>100 的仓库过采 3×。
- 去除 auto-generated / vendored / minified 文件（detect 方法：文件熵 + 路径启发式）。

**数学与推理 CoT**：
- 真人解题：AoPS、MATH、OpenR1。
- 合成：用一个强模型（e.g. GPT-4o / Claude / DeepSeek-R1）对题目生成多条 CoT，再用 verifier（SymPy、答案匹配）过滤保留正确者。这是"reject sampling distillation"思路。

**Tool-use 轨迹（agentic 数据）**：
- 这部分是 GLM-5.1 能跑"连续 8 小时 agentic 任务"的关键。
- 合成 pipeline：在沙盒（docker + bash + python + 浏览器模拟）里让 agent 执行真实任务（修 bug、查文档、跑实验），记录完整 `<thought><action><observation>` 链。
- 每条轨迹 10K–100K token，天然是长上下文训练素材，一举两得。

**长文档**：
- 书籍（Project Gutenberg、Books3 替代）、法律长文、多篇串接的 arXiv。
- **关键**：要用"真长"而不是"拼长"。纯随机 concat 得到的长文档学不到跨段依赖。

### 2.3 数据顺序（curriculum）

Qwen3 和 GLM-4.5 都采用**渐进式 curriculum**：

```
Step 0–30%:  短 ctx（4K/8K），但质量极高 → 打底
Step 30–70%: 中等 ctx（32K），代码 + 数学上采 → 重点能力
Step 70–100%:长 ctx（128K–256K），书 + repo + agent 轨迹 → 长程能力
```

---

## 3. 学习率策略

### 3.1 传统 cosine 的问题

Pretrain 的 cosine schedule 把 lr 从 peak（~3e-4）在整个训练中衰减到 10% × peak = 3e-5。到 pretrain 末期，模型其实已经几乎不学新东西。直接无缝衔接 mid-training 会有两个麻烦：

1. lr 太低，高质量数据吃不进去。
2. 但又不能把 lr 拉回 3e-4，否则破坏已学表征。

### 3.2 方案 A：二次退火（Re-Warmup + Re-Decay）

GLM-4.5 和 DeepSeek-V3 都用这个。Mid-training 开始时：

```
lr_start_mid = 0.3 × peak_pretrain   # ≈ 1e-4
warmup 2% of mid-training steps 到 lr_start_mid
然后 cosine 或 linear decay 到 0.01 × peak  # ≈ 3e-6
```

直观理解：这是一次"小 cosine"，让模型在高质量数据上有足够的学习率，又不会颠覆已学的东西。

### 3.3 方案 B：WSD（Warmup-Stable-Decay）

MiniCPM、Qwen3 用的更现代的做法：

```
阶段      学习率
Warmup    0 → peak（step 0–2%）
Stable    peak 保持不变（step 2%–90%，占总训练绝大部分）
Decay     peak → 0（step 90%–100%，线性或 1-sqrt）
```

WSD 的好处：
- **Stable 段**可以任意长，天然支持 continual pretraining（不用重算 schedule）。
- **Decay 段**很短却能显著降低 loss —— MiniCPM 实验显示 decay 段 loss 下降幅度等价于 stable 段 10× tokens。
- 非常适合把 mid-training 放进去：整个 mid-training 就是 decay 段。

### 3.4 实操建议

- 如果你是从别人的 pretrain ckpt 继续训：**方案 A 更安全**，因为你无法控制原作者的 schedule，二次 warmup 能让你重新控制 lr。
- 如果你从头设计管线：**直接 WSD**，mid-training = decay 段，省心。
- 长上下文扩展阶段的 lr 建议再低一档：1e-5 左右，避免破坏短 ctx 能力。

---

## 4. 长上下文扩展技术栈

这一节是 Phase 3 最核心的工程话题。本质问题：**模型在 4K/8K 上训练，如何让它在推理时处理 128K/200K？**

### 4.1 为什么不能直接外推

Transformer 的位置信息来自 RoPE（Rotary Position Embedding）。RoPE 的本质是把 query/key 向量按位置 $m$ 旋转 $m\theta_i$，其中 $\theta_i = \text{base}^{-2i/d}$，`base` 默认 10000。

当 $m$ 超过训练时见过的最大长度，旋转角度进入"未见区域"，attention 分数会急剧失真（典型表现：loss 爆炸、重复生成、忽略远端信息）。

### 4.2 技术 1：RoPE base 调整（最简单、最常用）

把 base 从 10000 调大到 500000 或 1000000，相当于把"周期"拉长，所有位置的旋转都变慢，原本 4K 训的角度分布现在可以覆盖更长距离。

- **优点**：改一行代码，配合少量继续训（10B–100B token）就能扩 4×–8× ctx。
- **缺点**：单独改 base 会让短 ctx 性能略降；大幅扩展（>16×）时效果不如 YaRN。
- Llama-3 用 base=500000 原生支持 8K，Llama-3.1 扩到 128K 时 base=500000 + RoPE scaling。

### 4.3 技术 2：Position Interpolation (PI)

Meta 2023 年的做法：把位置索引 $m$ 线性压缩到训练时见过的范围：$m' = m / s$，其中 $s$ 是扩展倍数。

- **优点**：无需改权重，少量 fine-tune 即可。
- **缺点**：高频维度（小 $i$，周期短）被过度压缩，高频信息丢失严重。

### 4.4 技术 3：NTK-aware Scaling

社区（bloc97）发现：RoPE 不同维度对应不同频率，**低频维度应该插值，高频维度应该外推**。做法是把 base 按照 NTK 理论缩放：

$$
\text{base}' = \text{base} \cdot s^{d/(d-2)}
$$

- **优点**：不需要任何训练就能扩 2×–4×。
- **缺点**：理论不够严格；4× 以上还是会退化。

### 4.5 技术 4：YaRN（Yet another RoPE extensioN，arXiv:2309.00071）⭐主流

YaRN 是目前最主流的 RoPE 扩展方法（被 Mistral、Qwen、DeepSeek-V3 广泛采用）。它把维度分为三段：

- **高频维度**（短周期 $< L_\text{train}$）：外推（不动）。
- **低频维度**（长周期 $> L_\text{train} \cdot s$）：插值（除以 $s$）。
- **中频维度**：用 `ramp` 函数平滑过渡。

YaRN 还加了一个 **attention temperature scaling**：$\text{softmax}(Q K^\top / (t \sqrt{d}))$，其中 $t = 0.1 \ln(s) + 1$，补偿长序列下 attention entropy 变化。

**YaRN 超参**（2309.00071 Table 2 推荐）：

```python
# 扩展 scale s = L_target / L_train
alpha = 1       # ramp 下界
beta  = 32      # ramp 上界
# dim i 的旋转因子 lambda_i:
#   lambda_i = s                          if wavelength(i) > beta
#   lambda_i = 1                          if wavelength(i) < alpha
#   lambda_i = s 与 1 的平滑插值          otherwise
attn_scale = 0.1 * math.log(s) + 1
```

- **优点**：扩展 16×–32× 仍然稳定；继续训只要几十亿到上百亿 token。
- **缺点**：仍然是"均匀"的段式缩放，对极长 ctx（>128K）不是最优。

### 4.6 技术 5：LongRoPE（arXiv:2402.13753）⭐前沿

微软 2024 年的方法。核心观察：**RoPE 不同维度对长上下文的重要性不同，应该用"非均匀"缩放因子**。

LongRoPE 用**进化搜索（evolutionary search）**在验证集上找出每个维度的最优 $\lambda_i$，而不是像 YaRN 那样用解析公式。还引入了**两阶段扩展**：

1. 先用搜到的 $\lambda$ 扩到 128K（继续训 ~1B token）。
2. 对 256K、512K 再搜一次 $\lambda$，几乎不需要额外训练。

LongRoPE2（2025 年）进一步用可微搜索（straight-through estimator）替代进化，提速 10×。

- **优点**：目前唯一能稳定扩到 2M context 的方法；短 ctx 性能几乎无损。
- **缺点**：搜索开销不低（单次 ~100 GPU-hour）；实现复杂度高于 YaRN。

### 4.7 技术 6：LongLoRA（arXiv:2309.12307）

不是位置编码方法，而是**如何便宜地长 ctx fine-tune**。

- 提出 **Shift Short Attention (S²-Attn)**：训练时把序列分成若干 group（e.g. 每组 2K），在一半 head 里把 group 平移半个 group 长度，近似全局 attention。
- 只训 LoRA + embeddings + norm 层，显存成本约为全参训练的 20%。
- 适合学术/小团队把 7B 模型从 4K 扩到 100K。

### 4.8 方法对比表

| 方法           | 训练成本 | 扩展上限 | 短ctx保留 | 实现复杂度 | 典型采用方                      |
| -------------- | -------- | -------- | --------- | ---------- | ------------------------------- |
| RoPE base 改写 | 低       | 8×       | 中        | 极低       | Llama-3/3.1                     |
| PI             | 低       | 4×       | 差        | 低         | 早期扩展论文                    |
| NTK-aware      | 零       | 4×       | 良        | 低         | 社区/推理侧                     |
| **YaRN**       | 中       | 32×      | 良        | 中         | **Mistral, Qwen, DeepSeek-V3**  |
| **LongRoPE**   | 高       | 256×     | **优**    | 高         | **Phi-3-mini-128K, GLM-4.5?**  |
| LongLoRA       | 极低     | 32×      | 中        | 中         | 学术界                          |

**工程推荐**：
- 扩到 32K 以内：**YaRN** 就够，配方成熟、继续训 ~30B token。
- 扩到 128K–1M：**LongRoPE** 更稳，尤其对长上下文 retrieval 任务（needle-in-haystack）准确率优势明显。

---

## 5. 长上下文训练数据

### 5.1 三类数据源

1. **天然长文档**：书籍（~100K token/本）、长法律文件、多章 arXiv 论文。特点：真长程依赖，但分布偏窄。
2. **代码仓库**：完整 repo 内容串接。特点：工程价值高，长程依赖强（函数调用、import 关系）。
3. **合成长上下文**：多文档拼接 + 显式添加"针"（needle）到"干草堆"（haystack）。特点：可控、规模大，但要小心"假长"。

### 5.2 Repo-level packing：把仓库变成训练样本

这是 GLM-4.5 ARC 强调的 coding 长上下文秘诀，也是 DeepSeek-Coder 的核心做法。

**朴素做法（差）**：把 repo 里所有 `.py` 文件按文件名字母序拼起来。 → 模型学到的只是"无关文件堆叠"，跨文件依赖无法被 attention 捕捉。

**Repo-level packing（好）**：按照**依赖图拓扑序**排列文件，让被依赖的文件出现在依赖它的文件之前，使得 attention 在因果方向上能"看到上游"。

#### 5.2.1 构造依赖图的具体做法

以 Python 为例：

```python
import ast, networkx as nx, os
G = nx.DiGraph()
files = {}  # path -> source
for root, _, fs in os.walk(repo_root):
    for f in fs:
        if f.endswith('.py'):
            p = os.path.join(root, f)
            files[p] = open(p).read()
            G.add_node(p)

for p, src in files.items():
    try:
        tree = ast.parse(src)
    except SyntaxError:
        continue
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
            # 把 module 映射回 repo 里的 path
            target = resolve_module(mod, repo_root)
            if target in files:
                G.add_edge(target, p)   # target 被 p 依赖

# 拓扑排序；有环的话用 scc 缩点再排
order = list(nx.topological_sort(G))  # fallback: nx.algorithms.dag.lexicographical_topological_sort
packed = '\n\n'.join(f'### FILE: {p}\n{files[p]}' for p in order)
```

#### 5.2.2 进阶技巧

- **函数级 packing**：不以文件为单位，而是以函数/类为单位建调用图（用 `tree-sitter` / `pyright` 拿调用关系），排序粒度更细，适合长 repo（>200K）。
- **README/docs 前置**：把 `README.md`、`setup.py`、`pyproject.toml` 放在最前面，让模型先建立"全局画像"。
- **测试文件后置**：`tests/` 放在实现文件之后，模拟"先读实现再读测试"的认知顺序。
- **截断策略**：超过 max_seq_len 的 repo，按 PageRank 保留核心文件（入度最大的 N 个文件），而非简单 truncate 末尾。
- **文件分隔符**：用特殊 token（`<|file|>path<|/file|>`）包裹，帮助模型识别边界。DeepSeek-Coder 的 fim+repo token 设计就是这个思路。

#### 5.2.3 多语种

- JavaScript / TypeScript：用 `acorn` / `ts-morph` 解析 `import` / `require`。
- Rust：读 `Cargo.toml` 和 `mod.rs`。
- Go：读 `import` 块。
- Java：读 `import` 语句。

### 5.3 Needle-in-a-Haystack 合成数据

纯依赖天然长文本会让模型学不到"精准检索长文档中某个位置信息"的能力。因此要合成：

```
Haystack：把 50 段无关长文档（每段 2K token）拼起来，得到 100K 的 stuff。
Needle：在随机位置插入一句 "The secret code is 8472."
Query：文档末尾追加 "What is the secret code?"
Target：8472
```

关键技巧：
- **位置要均匀覆盖**：0%、10%、20% … 100% 都要有 needle，否则模型只会关注开头和结尾。
- **多 needle**：插入 2–5 个 needle，要求模型同时回忆多个事实（更接近真实 agent 场景）。
- **干扰 needle**：插入相似但不相关的句子（"The lucky number is 1234"），训练模型区分。

### 5.4 长 agent 轨迹（真实长上下文）

把 agent 在沙盒里跑的真实任务轨迹作为训练数据，是 GLM-5.1 "8 小时连续 agentic" 的关键素材来源。单条轨迹 30K–200K token，自带真实的长程依赖（比如第 20K token 打开的文件，在第 80K token 才被编辑）。

---

## 6. 长上下文训练的工程难点

### 6.1 显存爆炸的来源

训练一个 n token 序列：

| 组件                    | 显存                        | 128K 估算（32B 模型）        |
| ----------------------- | --------------------------- | ---------------------------- |
| Activation（attn logits）| $O(n^2)$ per layer per head | 不存，用 FlashAttention 消除 |
| Activation（FFN）       | $O(n \cdot d)$              | 数百 GB                      |
| KV cache（推理）        | $2 \cdot n \cdot L \cdot d$ | 32B@128K ≈ 20 GB             |
| 梯度 + 优化器           | $2 \times$ param            | 固定                         |

### 6.2 FlashAttention-2/3

在 SRAM 上分块计算 attention，把 $O(n^2)$ 显存降到 $O(n)$（时间仍 $O(n^2)$，但常数极小）。**长 ctx 训练的前置条件**，没有之一。

- FlashAttention-2：Ampere（A100）优化。
- FlashAttention-3：Hopper（H100）专用，利用 WGMMA + TMA，吞吐翻倍，还支持 FP8。

### 6.3 Sequence Parallelism (SP) / Context Parallelism (CP)

对单张卡也放不下的序列（>128K），必须沿 sequence 维度切分：

- **Megatron SP**：把 LayerNorm/Dropout 的 activation 沿 seq 切，attention/FFN 前 all-gather。通信量大，只适合中等长度。
- **DeepSpeed-Ulysses**：attention 之前沿 seq 切，attention 中沿 head 切（通过 all-to-all），活化显存降到 $O(n/P)$。适合 Q/K/V head 数多的模型。
- **Ring Attention**（Liu et al. 2023）：把序列切成 $P$ 段分到 $P$ 张卡，每张卡持有一段 Q，K/V 沿环传递，每轮算自己 Q 对当前 K/V 段的贡献，$P$ 轮后拼出完整 attention。显存 $O(n/P)$、通信与计算完全 overlap。**是目前 128K+ 训练的事实标准**。
- **Striped Attention / Blockwise Ring**：对 causal mask 做负载均衡（不均衡会让一些卡空等 50% 时间），Striped 用交错分片让每张卡的有效计算量相同。

### 6.4 Chunking / Gradient Checkpointing

- **Selective activation checkpointing**：只对 FFN 做 checkpoint，attention 的 activation 直接 FlashAttention 不存。
- **Chunked loss**：最后一层的 `logits = hidden @ W_emb^T` 形状是 $(n, V)$，V=150K 时 128K 序列占 75GB。必须按 chunk 计算 loss，不物化整个 logits（TriLoss / Liger-Kernel）。

### 6.5 数据 pipeline

- **Variable-length packing**：不同长度样本打包到同一 batch，用 cu_seqlens 告诉 FlashAttention 边界。避免 padding 浪费。
- **Sorted bucket**：按长度分桶，同桶内长度接近，减少 bubble。
- **Disk IO**：200K token 单样本 ~1MB，Shuffle buffer 装不下太多，要用多进程预取。

### 6.6 稳定性 trick

- 长 ctx 训 loss 常出现 spike。减小 lr（1e-5 量级）、加大 gradient clip（1.0）、warmup 要长（2%–5%）。
- 监控 attention entropy：突然塌缩到 0 是 softmax 饱和前兆，可以触发 rollback。
- Z-loss（logits^2 正则）：防止 logits 过大，Qwen3 长 ctx 训用了 1e-4 系数。

---

## 7. 长上下文评测

| Benchmark                | 核心考点                  | 长度         | 备注                                         |
| ------------------------ | ------------------------- | ------------ | -------------------------------------------- |
| **Needle-in-a-Haystack** | 单点检索                  | 1K–10M       | 标准烟雾测试；简单，但必须过                 |
| **RULER**（NVIDIA 2024） | 13 类合成任务             | 4K–128K      | 目前**最严格**的合成长 ctx 评测              |
| **LongBench v2**         | 真实任务（QA/摘要/代码）  | 8K–128K      | 分布最接近真实使用                           |
| **InfiniteBench**        | 极长 QA + 代码            | 100K–2M      | 检验 100K+ 能力                              |
| **RepoBench**            | 仓库级代码补全            | ~64K         | 评价 repo-level 理解                         |
| **LongCodeBench**        | 长代码 bug 定位           | 32K–200K     | 2025 新基准，与 agentic 能力相关             |
| **BABILong**             | 长 bAbI（多跳推理）       | 1K–10M       | 纯推理链长度压力测试                         |
| **LOFT**                 | 长 ctx 检索 vs RAG        | 128K–1M      | 判断"要不要 RAG"的分水岭                     |

**实践建议**：
- 不要只看 Needle。Needle 是过线题；RULER 才是区分度题。
- RepoBench + LongCodeBench 是 code-LLM 必测。
- agentic 场景需要自建 benchmark（比如 SWE-bench Verified 的长上下文子集）。

---

## 8. GLM-5.1 扩到 200K 的合理推测

GLM-5.1 官方未完整披露细节，以下基于 GLM-4.5 ARC（2508.06471）+ 同期 SOTA 的合理推测：

### 8.1 推测架构

- Backbone：MoE，约 300B total / 30B–40B active（对齐 GLM-4.5 的比例扩展）。
- 原生训练 ctx：8K。
- 扩展路径：8K → 32K → 128K → 200K，三阶段 continual pretraining。

### 8.2 推测的长上下文配方

| 阶段              | ctx  | 方法                       | Tokens  | lr      |
| ----------------- | ---- | -------------------------- | ------- | ------- |
| base pretrain     | 8K   | RoPE base=500k             | 15T+    | 3e-4    |
| mid-train I       | 32K  | YaRN s=4, continue         | 200B    | 1e-4    |
| mid-train II      | 128K | **LongRoPE** s=16           | 100B    | 3e-5    |
| long-ctx extend III| 200K | LongRoPE s=25               | 30B     | 1e-5    |

在 128K 阶段就切到 LongRoPE（而非 YaRN）是关键判断——LongRoPE 在 needle 准确率和 short-ctx 保留两项上都明显优于 YaRN，且 Phi-3-mini-128K 已经验证其可靠性。

### 8.3 "8 小时连续 agentic" 的技术含义

- 8 小时 × 假设 5 token/s 输出 + 类似速率输入 ≈ 250K+ 总 token 吞吐。
- 单次上下文需容纳：工具调用历史 + 文件快照 + 思考链，保守估计 200K。
- 加上 **KV cache 压缩**（如 H2O、SnapKV、MInference）可以让真实 ctx 达到 500K，而实际 attention 只关注 top-k 关键 token。
- 需要 attention sink（Efficient Streaming LLM）或 StreamingLLM 机制，防止长时间运行后 early tokens 被替换导致 attention 塌陷。

### 8.4 推测的评测结果

- RULER 128K：~85（对齐 Llama-3.1 70B）。
- Needle 200K：~95。
- RepoBench 64K：SOTA 或接近。
- SWE-bench Verified：~65（对齐 Claude 3.5 Sonnet 水准，这部分是 Phase 4/5 的 agentic RL 贡献）。

---

## 9. 小规模可复现实验：把 1B 模型从 4K 扩到 32K

目标：单机 8× A100-80GB，从一个 4K 训练的 1B 开源模型（如 TinyLlama-1.1B 或 Qwen2.5-1.5B）开始，用 YaRN 扩到 32K。

### 9.1 环境

```bash
pip install torch==2.4 transformers==4.45 flash-attn==2.6 deepspeed==0.15 datasets accelerate
```

### 9.2 修改模型的 RoPE（核心代码）

```python
# yarn_rope.py
import math, torch

def yarn_rope(dim, base=10000.0, scale=8.0, orig_ctx=4096,
              beta_fast=32, beta_slow=1, device='cuda'):
    """
    dim: head_dim
    scale: 扩展倍数 s，例如 32K/4K = 8
    orig_ctx: 原始训练长度
    beta_fast / beta_slow: YaRN 默认 32 / 1
    """
    inv_freq_base = 1.0 / (base ** (torch.arange(0, dim, 2, device=device) / dim))

    # 1. 找到 ramp 区间：哪些维度要 interpolate / extrapolate
    def find_dim(num_rot, dim_, base_, orig):
        # wavelength = 2*pi / inv_freq；num_rot 即 orig_ctx / wavelength
        return dim_ * math.log(orig / (num_rot * 2 * math.pi)) / (2 * math.log(base_))

    low  = max(math.floor(find_dim(beta_fast, dim, base, orig_ctx)), 0)
    high = min(math.ceil (find_dim(beta_slow, dim, base, orig_ctx)), dim // 2 - 1)

    # 2. 构造 ramp mask（0 = extrapolate, 1 = interpolate）
    linear = (torch.arange(dim // 2, device=device) - low) / max(high - low, 1)
    ramp = torch.clamp(linear, 0, 1)

    # 3. YaRN inv_freq = extrapolate*(1-ramp) + interpolate*ramp
    inv_freq = inv_freq_base * (1 - ramp) + (inv_freq_base / scale) * ramp

    # 4. attention temperature
    attn_scale = 0.1 * math.log(scale) + 1.0
    return inv_freq, attn_scale
```

把这个 `inv_freq` 替换到模型 `RotaryEmbedding` 的 buffer，`attn_scale` 乘到 `softmax(QK^T / sqrt(d))` 的 `1/sqrt(d)` 上。

### 9.3 继续训的数据

```
总预算：~3B token
├── 50%  SlimPajama 长文档（>8K 的文章，concat 到 32K）
├── 30%  the-stack-dedup 代码仓库（repo-level packing 到 32K）
├── 15%  合成 needle（每条 32K，含 2–5 个 needle）
└── 5%   原始 4K 分布数据（防止短 ctx 退化）
```

### 9.4 训练 launch（DeepSpeed ZeRO-3 + FlashAttention-2）

```bash
deepspeed --num_gpus 8 train.py \
  --model_name TinyLlama-1.1B \
  --max_seq_len 32768 \
  --batch_size_per_gpu 1 \
  --gradient_accumulation 16 \          # effective bs = 128 * 32k = 4M tok
  --lr 1e-5 --warmup_steps 200 --total_steps 22000 \
  --lr_scheduler wsd --decay_start_step 18000 \
  --deepspeed ds_zero3.json \
  --attn_impl flash_attention_2 \
  --use_yarn --yarn_scale 8.0 --orig_ctx 4096
```

`ds_zero3.json` 关键片段：

```json
{
  "zero_optimization": {"stage": 3, "offload_optimizer": {"device": "cpu"}},
  "bf16": {"enabled": true},
  "gradient_checkpointing": true,
  "train_batch_size": 128
}
```

### 9.5 预期耗时与效果

| 指标                      | 数值                           |
| ------------------------- | ------------------------------ |
| 8× A100-80G，32K seq      | ~8K tokens/s/GPU               |
| 3B token 总训练           | ~12 小时                       |
| Needle@32K                | 训练前 15 → 训练后 92          |
| PPL @ 4K（原分布）         | 退化 < 3%                      |
| PPL @ 32K                 | 从 NaN → 合理                  |

### 9.6 常见坑

1. **lr 太高**：直接继承原 pretrain 的 3e-4 会让短 ctx 性能塌掉。必须 ≤ 3e-5。
2. **忘了改 attn_scale**：只改 inv_freq 会让长 ctx 上 attention entropy 过大，生成质量下降。
3. **数据全是长样本**：必须保留 ≥5% 短样本，否则 4K 能力退化明显。
4. **FlashAttention 版本太旧**：2.5 以下对非 2 的幂次长度支持不好，32K + cu_seqlens 会报错。升到 2.6+。
5. **eval 方式错误**：用短 ctx eval 判断长 ctx 训练好坏，误判常见。一定要跑 Needle@32K + RULER 子集。

---

## 10. 小结与后续路线

### 10.1 Phase 3 心智模型

- **Mid-training ≠ continual pretraining**：前者是精装修（高质量 + 低 lr），后者是扩面积（新领域 + 原 lr）。
- **长上下文扩展不是"单独一件事"**：它和 mid-training 的数据配方、学习率、engineering 架构全都耦合。
- **RoPE 调教是当前所有长 ctx 方案的地基**：YaRN 是 32K 档的性价比王者，LongRoPE 是 128K+ 的必然选择。
- **Repo-level packing 是 code-LLM 长 ctx 能力的独特 buff**：拓扑序 + 函数级依赖图 + README 前置，三条缺一不可。

### 10.2 写完 Phase 3 后接下去看什么

- **Phase 4（SFT）**：instruction 数据合成、multi-turn 对齐、code + math SFT 的数据配方。
- **Phase 5（RL）**：agentic RL（SWE-RL）、RLVR（可验证奖励）、Process Reward Model。
- **推理侧优化**：KV cache 压缩（H2O / SnapKV）、speculative decoding、MInference、paged attention。
- **长上下文的"下一个边界"**：稀疏注意力（Native Sparse Attention, DeepSeek NSA）、Mamba/SSM 混合、MoBA。

---

## 参考文献

1. GLM-4.5 Team. *GLM-4.5: Agentic, Reasoning, Coding (ARC) Foundation Models.* arXiv:2508.06471, 2025.
2. Peng B. et al. *YaRN: Efficient Context Window Extension of Large Language Models.* arXiv:2309.00071, 2023.
3. Ding Y. et al. *LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens.* arXiv:2402.13753, 2024.
4. Chen Y. et al. *LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models.* arXiv:2309.12307, 2023.
5. DeepSeek-AI. *DeepSeek-V3 Technical Report.* arXiv:2412.19437, 2024.
6. Qwen Team. *Qwen3 Technical Report.* arXiv:2505.09388, 2025.
7. Liu H. et al. *Ring Attention with Blockwise Transformers for Near-Infinite Context.* arXiv:2310.01889, 2023.
8. Hsieh C-P. et al. *RULER: What's the Real Context Size of Your Long-Context Language Models?* arXiv:2404.06654, 2024.
9. Bai Y. et al. *LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding.* arXiv:2308.14508, 2023.
10. Shi W. et al. *In-Context Pretraining: Language Modeling Beyond Document Boundaries.* arXiv:2310.10638, 2023.（repo-level packing 理论基础）
11. Hu S. et al. *MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies.* arXiv:2404.06395, 2024.（WSD schedule）
12. Dao T. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.* arXiv:2307.08691, 2023.

---

## 📌 章末检查

**带走这 5 条**
- mid-training = pretrain 末尾的"退火"段：上采样代码/数学到 60-70%，配 WSD 二次衰减。
- 长上下文 = RoPE 调教（YaRN / LongRoPE）+ 数据侧 repo-level packing 双管齐下，缺一不可。
- YaRN 在 32K→128K 区间是事实标准；> 200K 才考虑 LongRoPE 非均匀缩放。
- 验真用 **RULER** needle-in-haystack + **LongBench-v2**，不要只看 PPL。
- repo-level packing 按 import 拓扑序拼接，保留真实工程结构，cross-file reasoning 提升明显。

**自检 3 题**（< 5 分钟）
1. 32K → 128K 用 YaRN 还是 LongRoPE？凭什么选？
2. repo-level packing 为什么比按文件随机拼接强？
3. effective context length 怎么测？为什么只看 PPL 不够？

<details><summary>参考答案</summary>

1. 32K-128K 区间 YaRN 性价比最高（无训练即用、表现稳定）；> 200K 才上 LongRoPE 的非均匀进化搜索，否则训练成本不划算。
2. 模型在训练时看到真实工程的 import 顺序与依赖关系，被迫学会 cross-file reasoning；随机拼接相当于把工程拆成"无关代码片段集合"，模型只学到 file-local 模式。
3. **RULER** needle-in-haystack：在长 prompt 中插入一句独特事实，问模型能否检索出来。"effective_ctx" = 检索准确率 ≥ 85% 的最大上下文长度。PPL 在 128K 上可以很低但 needle 检索完全失败，因为模型可能学到"长 prompt 都答 OK"的 shortcut。
</details>

> ⚠️ **常见坑** · 只跑 PPL 看长上下文有没有训坏——结果上线后发现 100K+ 任务 needle 完全 fail。PPL 是"平均损失"，对长 prompt 中那几个**关键 token** 的预测错误几乎不敏感；必须 RULER + LongBench 双重验真。

**下一步** → 进入 [phase4 SFT](./phase4_sft.md) 看怎么把长上下文 base 模型对齐成 agent。术语速查 → [▣ 索引](./phase_glossary.md)。

---

## 动手练习

1. 浏览 YaRN（arXiv:2309.00071）和 LongRoPE（2402.13753）两篇论文的"实验"章节，回答：在 32K → 128K 区间，YaRN 在 Needle-in-Haystack 上比 PI / NTK-aware 优势多大？为什么 LongRoPE 的"非均匀缩放"在 128K+ 才显出价值？
   *提示*：直接看论文表格，不需要跑实验。结合 §4 RoPE 调教章节的对照表。
2. 用一个 4K 上下文的开源 base 模型（如 `meta-llama/Llama-3.2-1B`），在不重新训练的前提下把 RoPE base 从 10000 改成 500000 + 加 YaRN scaling，跑 RULER 16K 看 needle 检索准确率，并和原模型在 4K 上的表现对比。
   *提示*：HF transformers 的 `rope_scaling` 字段一行配置即可；§4.3 给了 YaRN 参数公式。
3. 实现 §6 的 repo-level packing：写一个 Python 脚本，输入是某 GitHub repo（≥ 1k 文件）的 clone，输出是按 import 拓扑序拼接、每条样本 ≤ 32K token 的训练数据 jsonl。要求 README 永远在文件前面、被 import 多的文件靠前。
   *提示*：用 tree-sitter 提依赖图 + networkx 做拓扑排序。Phase 1 §0.4 数据格式可复用。
4. 设计并跑通一个 mid-training "退火"实验：用一个 1B 量级 SFT 前的 base 模型（你自训或开源都行），用 phase1 数据 + 上采样代码/数学到 60% 的高质量配方跑 50B token，学习率走 WSD：30B 稳定 → 20B 二次衰减。监控 5 条曲线（loss、grad_norm、code_loss、math_loss、web_loss），观察 "annealing 段 loss 二次下降" 是否真的发生。
   *提示*：§3 学习率策略 + §2 数据配比。这个实验的结论决定你之后是否愿意投资 mid-training。
5. **完整 capstone**：基于 #4 的 mid-training 产物，把 RoPE 重写成 LongRoPE 非均匀缩放、再用 §6 的 repo-level packing 数据继续训 5B token 把上下文扩到 128K，最后用 RULER + LongBench-v2 验证 effective context length，并和 base 模型在 RULER 4K/16K/64K/128K 四档上做对比柱状图。要求 effective_ctx ≥ 96K。
   *提示*：这是 Phase 3 的"毕业项目"，预算 ~200-400 H100·hour。技术难点全在 §4-§7。完成后你就拥有"自训 long-ctx 模型"的完整闭环能力。
