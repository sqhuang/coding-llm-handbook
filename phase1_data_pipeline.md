# Phase 1：代码预训练数据 Pipeline 深度笔记

> 📅 主线快照：2026-04-22 · 上次核对：2026-04-30

> **⚡ 三句话要点**
> 1. 11 步清洗里**去污染 + MinHash-LSH 去重**两步错了，loss 再漂亮都是幻觉——必须先验证对 HumanEval / MBPP / LiveCodeBench 零命中。
> 2. 启发式过滤照 StarCoder2 规则（行长、字母比例、autogen 标记、必须含 def/class/import）；PII 先过最致命几类（私钥整文件丢，API key / email 替换为占位符）。
> 3. 最终产出是 mmap `.bin`；喂训练前 **README 摘要 + path tag 必须前置拼到 content 头部**，否则模型学不到跨文件依赖的因果方向。

> 目标读者：准备从零训练一个 Coding LLM 的研究者；主线对齐 GLM-4.5 / GLM-5.1 (arxiv 2508.06471)，交叉参考 StarCoder2 (2402.19173)、OpenCoder (2411.04905)、DeepSeek-Coder-V2 (2406.11931)。
> 定位：这是一份"能动手跑"的工程笔记，不是综述。所有步骤都给出参数、阈值、代码或工具链。

> **读者画像** · 准备亲手搭一条 ≥ 100B token 代码数据 pipeline 的工程师；或想审计已有数据集是否"可以下锅"的训练 lead。
> **前置知识** · 用过 Spark / Ray / datatrove 任一种分布式数据处理框架；序.6 [packing](./phase_basics_training.md) 与 phase0 §1 GLM-5.1 速览。
> **学完能做** · 从原始 GitHub dump 跑通 11 步清洗 → 给每一步选阈值 → 输出可直接 mmap 进 Megatron 的 tokenized `.bin`。

---

## 0. 从 CV 分类训练数据到 Code LLM 数据：观念迁移

> 这一节写给从图像分类背景切过来的读者。如果你已经熟悉 NLP 预训练数据体系，可以直接跳到 §0.1 TL;DR。

如果你过去一直在跟 ImageNet 打交道，切到 Code LLM 数据时会有**四层彻底的概念错位**。先把它们摆清楚，后面所有具体步骤才能看得懂在做什么。

### 0.0.1 单位变了：从"一张图"到"一个 token"

| 维度 | ImageNet 分类 | Code LLM 预训练 |
|---|---|---|
| **基本单位** | 一张图，形状 `(3, 224, 224)` float32 | 一个 token，**标量** `()` int32；真正喂模型的是序列 `(L,)` int32（L=512~8192） |
| **样本数量级** | 120 万张图 | **数万亿 tokens**（GLM-4.5 量级 ≥ 15T） |
| **单样本尺寸** | 定长，约 15 万浮点数 | 变长，通常拼成 2k / 4k / 8k 长度 |
| **每个 epoch 看多少数据** | 120 万 × 224²×3 ≈ 1.8 × 10¹¹ 标量 | 1T-2T tokens，约 1 × 10¹² 整数 |
| **训练多少 epoch** | 90-300 | **0.5-2**（数据太多了根本跑不完） |

**关键直觉**：在 LLM 世界里，数据量的计量单位是 **token**（一个 token 大致是 3-4 个英文字符或一个汉字/一个 Python 关键字片段）。当有人说"训练了 15T tokens"，那就意味着模型见过了 15 万亿个这样的小整数。这个规模比 ImageNet 的像素总数还要大几个量级，而且每个 token 都进了一次前向 + 反向，不像 CV 可以 batch 内高度并行。

**工程推论**：你不会像 CV 那样真的把整个数据集读进内存或者随机访问单张图。代码数据是**流式消费**的——tokenize 后存成一条巨大的 token 流（几十 TB 的 `.bin` 文件 mmap），训练时按 offset 切窗口读。

### 0.0.2 标签消失了：从监督到自监督

CV 分类是典型的监督学习：每张图有人工标注的类别 (image, label)，loss = CrossEntropy(model(image), label)。

Code LLM 训练是**自监督**：数据只是一段文本，没有额外 label。任务是"根据前 N-1 个 token，预测第 N 个 token"（Next-Token Prediction, NTP）：

```
输入：def fibonacci(n):
目标：<next-token>  # 模型要学会预测这里接 "return"
```

每条序列里的每个位置都是一个训练样本，这意味着：
- **标注成本为零**——互联网上所有代码都是现成训练数据
- **"label" 直接从数据本身切出来**：`input_ids = sequence[:-1]`，`labels = sequence[1:]`
- **清洗成了重头戏**——CV 里 ImageNet 的质量有人工标注背书，LLM 里没有人审过数据，垃圾全进模型
- 因此 Phase 1 整个 pipeline 的 95% 工作量都在**去重 + 过滤 + 去污染**，而不是"找数据"

**类比**：CV 老师给学生"带答案的题目"；LLM 让学生做**完形填空**——题目和答案都从同一段文字里切出来。

### 0.0.3 数据形态变了：从"图像文件"到"代码文本 + 元数据"

CV 磁盘上的数据结构通常是：
```
imagenet/
├── train/
│   ├── n01440764/   (class 0: tench)
│   │   ├── n01440764_10026.JPEG
│   │   └── ...
│   └── n01443537/   (class 1: goldfish)
│       └── ...
└── val/
```

每张图是独立的二进制文件，元数据（类别）编码在父目录名里。

Code LLM 数据则是**一条一条的文本记录**，每条带一堆元数据字段。Hugging Face 上的 `bigcode/the-stack-v2-dedup` 一条样本长这样（JSON 形式）：

```json
{
  "content": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n",
  "repo_name": "alice/algorithms",
  "path": "py/recursion/fib.py",
  "language": "Python",
  "license": "MIT",
  "size": 112,
  "star_count": 34,
  "hexsha": "a1b2c3...",
  "max_line_length": 45,
  "avg_line_length": 22.4,
  "alphanum_fraction": 0.78
}
```

**和 CV 对应的心智地图**：

| CV | Code LLM |
|---|---|
| `image.JPEG`（二进制像素矩阵） | `content` 字段（UTF-8 代码文本） |
| 父目录名=类别（`n01440764/`） | `language` 字段（`"Python"`） |
| `labels.txt` 全局标注表 | `license`, `repo_name`, `path` 等元数据，**跟每条样本走**，用来后续过滤/配比 |
| 单张图 3×224×224 定长 | `content` 长度从 1 行到 50 万行都有，**变长** |

**存储格式的三个阶段**（这点非常重要，新手容易混）：

1. **Raw 阶段**：`.jsonl` 或 `.parquet`，每行/每记录是一个上面那样的 JSON。人可读，用来清洗和过滤。一条记录 = 一个代码文件。
2. **Tokenized 阶段**：把 `content` 过 tokenizer 变成 `[int, int, int, ...]`，其他元数据丢掉（或者保留做配比用）。
3. **Packed 阶段**：把所有文件的 token 列表**首尾相接拼成一条巨长流**，中间用 `<|endoftext|>` 分隔，再按固定长度（比如 4096）切片，存成 `.bin` (纯 token 数组) + `.idx` (记录每个 sample 起点的索引)。训练时 mmap 读取。

所以一个代码文件从磁盘到 GPU 的完整旅程是：**文本 → JSON 记录 → 过滤保留 → tokenize → 拼进巨流 → 切 4k 窗口 → batch → GPU**。

### 0.0.4 组织方式变了：从 ImageFolder 到 Packed Sequences

CV 的 DataLoader 逻辑：在全集里**独立均匀随机**抽 batch_size 张图，每张图自成一个样本。

LLM 训练对"样本"的定义不同。假设用 4096 长度训练：

- **单文件 < 4096 tokens**（占大多数）：不能让每个短文件单独一个 sample，否则算力大量浪费在 padding 上。做法是**把多个文件 token 首尾拼接**成一条 4096 长度：
  ```
  [file1 tokens][<|eos|>][file2 tokens][<|eos|>][file3 partial...]
  ```
  这就是 **packing**。每个 4096 长度的窗口可能包含 5-50 个原始文件的片段。
- **单文件 > 4096 tokens**（大型源文件、整库）：切成多段，每段是独立 sample。
- **仓库级 packing**（repo-aware）：把同一仓库的多个文件**按依赖顺序**（比如 `utils.py` 排在 `main.py` 前面）拼进同一条长序列，让模型能看到跨文件依赖。这是 GLM / DeepSeek 类 SOTA 的关键技巧（详见 §7）。

对比到 CV，相当于"把 10 张小图拼成一张大图送进去，但模型要学会在一张大图里同时识别每个子区域"。

```mermaid
flowchart LR
  F1["文件 A<br/>800 tokens"] --> P["首尾拼接 + [EOS] 分隔"]
  F2["文件 B<br/>200 tokens"] --> P
  F3["文件 C<br/>1500 tokens"] --> P
  F4["文件 D (partial)<br/>1596 tokens"] --> P
  P --> S["Packed Sample<br/>(B, 4096) int32<br/>填满 · 0 浪费"]
  S --> M["Document Attention Mask<br/>同文档内互见<br/>跨文档屏蔽"]
  M --> GPU["训练 GPU"]
```

**为什么要 packing 而不是 padding？**
- 算力利用率：padding 的 token 不参与有效训练，对 4k 长度来说可能浪费 50% 算力
- attention 机制无需 pad mask（用 document attention mask 实现"同一 document 内互见，跨 document 不见"）

---

## 0.1 怎么真的把数据搞到手：三条路径

从小白视角讲，"获取数据"比想象中简单——因为有人已经把 GitHub 扒过一遍并做了基础清洗了。

### 路径 A：从 Hugging Face 拉现成数据集（强烈推荐起步）

这是 90% 研究者的选择。操作步骤：

```bash
pip install datasets huggingface_hub
huggingface-cli login   # 需要一个 HF token, 免费账号即可
```

```python
from datasets import load_dataset

# streaming=True 就不会把整个数据集拉到本地（它有几 TB）
ds = load_dataset(
    "bigcode/the-stack-v2-dedup",
    "Python",              # 按语言切分的 subset
    streaming=True,
    split="train",
)

# 看一条样本
sample = next(iter(ds))
print(sample.keys())       # ['blob_id', 'path', 'content', 'src_encoding', 'language', ...]
print(sample["content"][:300])
print("license:", sample["license"])
print("repo:", sample["repo_name"])

# 要"下载"到本地？先起步只拉前 10GB Python 的部分到 parquet
import itertools, pyarrow as pa, pyarrow.parquet as pq
buf = []
for s in itertools.islice(ds, 200_000):  # 大概 ~10GB Python
    buf.append(s)
# ... 写到 parquet
```

**可选数据集清单**（按推荐度）：
- `bigcode/the-stack-v2-dedup` — 默认首选，已做精确去重，可直接用
- `bigcode/the-stack-v2` — 全量未去重，自己想做 MinHash 实验用
- `bigcode/starcoderdata` — StarCoder 原始训练集，单语言可下
- `HuggingFaceFW/fineweb-edu` — 高质量英文自然语言（兜底）
- `open-phi/programming_books_llama` — 编程书籍
- `codeparrot/github-code` — 轻量级 GitHub 子集，适合 demo

**预算感**：10 GB 原始 Python ≈ 2 GB parquet ≈ 600 M tokens。训一个 1.5B dense 模型想看 30B tokens，大概需要 500 GB parquet，~2 TB 原始 JSON。起步阶段 50 GB 完全够玩。

### 路径 B：自爬 GitHub（进阶，有特定需求时）

适用场景：你要补"最近 3 个月新出的 Rust 库"这种数据集里还没有的。

步骤概要：
1. **找仓库列表**：用 [GH Archive](https://www.gharchive.org/)（每小时 dump 所有 GitHub 公共事件）过滤 `CreateEvent` / `PushEvent` + star 数阈值。
2. **取 token**：GitHub personal access token，未认证 rate limit 60 req/hour，认证后 5000。大规模爬要多账号轮询。
3. **clone**：
   ```bash
   git clone --depth=1 --single-branch https://github.com/alice/repo.git
   ```
   `--depth=1` 只取最新 commit，省 90% 磁盘。
4. **抓内容**：遍历 `repo/` 目录，过滤 `.py .js .go ...` 扩展名，每个文件提取 `(path, content, sha, size)`，写 JSONL。
5. **合规**：读 `LICENSE` 文件，只保留 permissive（MIT, Apache-2.0, BSD-*）；其他丢弃或者单独桶存。

**坑**：GitHub ToS 说可以为了"检索与分析"自动化访问，但"大规模训练模型"灰色地带——实际操作上大家都在做，但你不能把训练出来的模型当作完全无风险的产物。真正规范的做法是走 Software Heritage（The Stack 走的就是它）。

### 路径 C：合成数据（高阶补充，不是替代）

用大模型把已有代码**改写 / 扩增**。主流做法：
- **OSS-Instruct**（Magicoder）：从真实 GitHub 代码片段反向生成自然语言指令
- **Evol-Instruct**：逐步加难度
- 这些更多是 SFT 阶段（Phase 4）用的，预训练阶段占比通常 <5%

---

## 0.2 一条数据从下载到喂进模型的"全程追踪"

把一个真实的 Python 文件想象成**游客**，它要通过 pipeline 这条流水线 12 个关口，有一半会被刷掉。用一个具体例子让整件事变得可触摸：

```mermaid
flowchart LR
  R["Raw 文件<br/>The Stack v2"] --> G1["1. 语言检测"]
  G1 --> G2["2. 许可证过滤"]
  G2 --> G3["3. SHA256 精确去重"]
  G3 --> G4["4. MinHash 近似去重"]
  G4 --> G5["5. 启发式过滤"]
  G5 --> G6["6. 质量打分"]
  G6 --> G7["7. PII 脱敏"]
  G7 --> G8["8. 评测集去污染"]
  G8 --> G9["9. FIM 改写"]
  G9 --> G10["10. 配比采样"]
  G10 --> G11["11. Tokenize"]
  G11 --> G12["12. Packing 4k"]
  G12 --> G13["13. 写 .bin / .idx"]
  G13 --> GPU["训练 GPU"]
  G3 -.被刷.-> X(("存活率<br/>30-40%"))
  G4 -.被刷.-> X
  G6 -.被刷.-> X
  G8 -.被刷.-> X
  classDef dropped fill:#f9f3e6,stroke:#a52619,stroke-dasharray:4 3,color:#1c1814
  class X dropped
```


**起点**：从 `the-stack-v2-dedup` 拉下来一条记录：
```json
{
  "content": "# Copyright 2022, contact admin@example.com\nimport numpy as np\n\ndef solve(arr):\n    return np.sort(arr)[::-1]\n\nif __name__ == '__main__':\n    print(solve([3,1,4,1,5]))\n",
  "repo_name": "bob/toy-algo",
  "path": "sort_desc.py",
  "language": "Python",
  "license": "MIT",
  "size": 178
}
```

**流水线**：

| 步骤 | 判定 / 变换 | 这条记录的命运 |
|---|---|---|
| 1. 语言检测 | go-enry 识别 | `Python` ✓ 进 Python bucket |
| 2. 许可证过滤 | 白名单 {MIT, Apache-2.0, BSD-*} | MIT ✓ |
| 3. 文件级 SHA256 | hash content 查重 | 未撞库 ✓ |
| 4. MinHash-LSH | 跟已有池 Jaccard 相似度 | 最大相似度 0.23 < 0.7 阈值 ✓ |
| 5. 启发式过滤 | 平均行长 < 100、字母比例 > 25%、非 autogenerated | ✓ |
| 6. 质量打分 | fastText 分类或小 LM PPL | 分数 0.74 > 0.5 阈值 ✓ |
| 7. PII 脱敏 | email/密钥/IP 替换 | 发现 `admin@example.com` → `<EMAIL>`，content 被改写 |
| 8. 评测集去污染 | 10-gram 匹配 HumanEval/MBPP/LiveCodeBench | 无命中 ✓ |
| 9. FIM 改写 | 50% 概率随机切一个中间 span | 抽中！变成 `<fim_prefix>import numpy as np\n\ndef solve(arr):\n    <fim_suffix>\n\nif __name__...<fim_middle>return np.sort(arr)[::-1]` |
| 10. 配比采样 | Python 占目标 18%，此记录按权重写入 | ✓ |
| 11. Tokenize | GLM tokenizer 压成 int 数组 | `[151643, 318, 4321, ...]` 长度 64 |
| 12. Packing | 跟其他文件拼成 4096 长度窗口 | 落在第 31872 条 packed sample 的 offset 2048-2112 |
| 13. 写 `.bin` + `.idx` | mmap 格式 | 完成，训练时可随机访问 |

这条样本经过了 13 个关口都活着到了 GPU，是幸运儿。大多数从 the-stack-v2 原始拉下来的文件会在 3/4/6/8 这几步被筛掉，最终存活率大约 **30-40%**——跟 CV 里"标注质量差的图被剔"一个道理，只是机制全自动。

---

## 0.3 数据配比 = CV 的 Class Balance

CV 里你会关注"每个类别多少张图"防止长尾；LLM 里对应概念是**数据配比 (data mixing)**，但维度多一层：

**CV 里的 balance 维度**：类别。
**LLM 里的 balance 维度**：
- 语言（Python 18% / C++ 12% / JavaScript 10% / ...）
- 数据类型（纯代码 70% / issue 5% / commit 3% / docs 5% / math 10% / 通用文本 7%）
- 质量分桶（high 50% / medium 35% / low 15%）
- 仓库 star 数分层（热门 / 长尾）

典型做法：给每个数据源一个采样权重，DataLoader 每 step 按权重多项式分布抽一个 shard。训练中期还可以**动态调权重**（curriculum：前期多样性优先，后期精英数据优先）。

这等价于 CV 里的 **weighted sampling**，只是在 LLM 里权重是训练配方的核心超参，直接决定模型擅长什么。

---

## 0.4 CV → Code LLM 概念对照表（速查）

| CV 概念 | Code LLM 对应 | 差异点 |
|---|---|---|
| ImageNet (1.2M images) | The Stack v2 (~460B tokens) | 量级差 6 个数量级 |
| (image, label) 二元组 | (context, next_token)，自监督 | 无人工标注 |
| 类别 (1000 classes) | 语言 + 质量 + 类型三维 mixing | 多维配比 |
| `ImageFolder` | parquet + streaming HF dataset | 元数据随样本走 |
| Train/Val 7:3 切分 | 几乎不 hold-out，用独立 benchmark（HumanEval）代替 | 数据量大，val 浪费 |
| 数据增强 (RandomCrop, Flip) | FIM 改写、代码等价重命名、AST 扰动 | 保留语义 |
| BatchNorm 统计 | 无对应（LLM 用 LayerNorm，不依赖 batch 统计） | — |
| Class Balance | Data Mixing Ratio | LLM 里权重是核心超参 |
| `torch.utils.data.DataLoader` | `megatron.data.gpt_dataset` / `nanotron.data` / mmap dataloader | 需要支持 mmap + packing + repo-aware |
| Mixup / CutMix | Packing 跨文档拼接（不是增强，是效率） | 动机不同 |
| ImageNet 标注错误率 ~5% | 代码数据"脏数据"率 30-60%（需要 pipeline 清） | 清洗是主工作 |
| `pretrained on ImageNet` 后 fine-tune | `pretrained on The Stack` 后 SFT / instruction-tuning | 结构对应，方法不同 |

记住这张对照表，后续所有技术细节（MinHash、FIM、packing、decontamination）本质上都是在回答同一个问题：**如何在没有人工标签的前提下，把一堆无序文本整理成训练信号尽可能强、污染尽可能小的 token 流。**

---

## 0.5 TL;DR 工程视角总结

一个可复现的代码预训练数据 pipeline 在 2025 年的共识大致如下：

- **底座数据**：以 `bigcode/the-stack-v2`（Software Heritage 抓取，600+ 语言，~900B tokens dedup 后）作为原始池；如果硬要自己爬 GitHub，用 GH Archive + gharchive.org 的 event stream 构建仓库列表，然后 `git clone --depth=1`。
- **清洗主干**：语言识别（go-enry）→ 许可证白名单 → 文件级 SHA256 精确去重 → 仓库级 MinHash-LSH 近似去重 → 启发式过滤 → 小 LM / fastText 质量分 → PII 脱敏 → 评测集 decontamination → FIM 改写 → 按语言/难度配比采样 → SentencePiece/BPE tokenize → 2k/4k/8k packing → 写 mmap (`.bin` + `.idx`) 或 Parquet 分片。
- **工具链**：`bigcode-project/bigcode-dataset`（清洗、PII、decontamination 脚本齐全）+ `huggingface/datatrove`（流式分布式执行框架，SLURM / Ray / 本地多进程通吃）+ `tokenizers` / `sentencepiece`。
- **数据配比**：代码 70–80%、代码相关自然语言 10–15%（issue、PR、commit msg、docs、StackExchange）、数学 / 推理 5–10%、通用文本兜底 5%；与 DeepSeek-Coder-V2 基本一致，GLM-4.5 ARC 报告里把 math + reasoning 显式拉到 ≥ 10%。
- **FIM**：50% 文件走 FIM，PSM 和 SPM 按 1:1 混，span 长度服从截断指数分布。

如果资源有限只能先做一件事：**先跑通 decontamination + MinHash-LSH 去重**，这是后续训练信号是否可信的分水岭。

---

## 0.6 企业私有代码资产入模：把"公司的代码"变成训练数据

> 场景：公司有几百个内部 Git 仓库、几千万行业务代码、一堆 wiki 和 issue，想让 LLM 真正"懂我们的代码资产"，然后基于这套资产开发下一代应用（辅助开发、bug 定位、API 推荐、文档问答、自动重构……）。这一节专门讲数据侧怎么做。

### 0.6.1 动手前的 30 分钟：先决定你想要模型具备"哪一种能力"

这是**全链路里最容易跳过、代价最惨**的一步。不同能力对应完全不同的数据准备路线，搞错了前面白忙。把能力拆成四种原子：

| 能力 | 典型提问 | 数据路线 | 代价 |
|---|---|---|---|
| **A. 知道我们的 API 能做什么**（知识注入） | "我们有没有现成的支付风控模块？" | **Continued Pretraining (CPT)** 把代码知识压进权重 | 需 ≥100M-1B tokens + GPU |
| **B. 按我们的风格写新代码**（风格 / 惯例注入） | "按我们的 Service 规范给 `OrderService` 加一个方法" | CPT + **范例 SFT**（in-style examples） | 中等 |
| **C. 基于代码回答开发者问题**（问答 / 检索） | "`PaymentService.refund()` 在什么情况下会抛 Timeout？" | **RAG**（embedding 索引 + 检索增强） | 低，几天可上 |
| **D. 执行特定工程任务**（Agent 能力） | "帮我给 `auth.login` 加一个单测并跑通 CI" | **Agent 轨迹 SFT** + 工具 + sandbox | 高，需要轨迹数据 |

**现实中一个成熟的内部 LLM 落地，ABCD 全要**，但起步路径差异很大：

- **钱少 + 想快见效**：C（RAG）→ D（轻量 SFT）→ B → A。先跑 RAG 把"问答"做起来，同时收集真实问答数据用于后续 SFT。
- **追求终局能力**：A（CPT）→ B（SFT）→ D（Agent）→ C（RAG 作为补充）。
- **混合方案（推荐中大型团队）**：并行推进 C + A。RAG 能立刻上线产生价值并收集数据，CPT 在后台跑准备下一代版本。

**数据准备的差异**：A/B 需要的是**大量、多样、自然分布**的原始代码（接近预训练语料）；C 需要的是**切块 + 向量化 + 元数据完整**的检索库；D 需要的是**(问题, 工具调用序列, 结果)** 的轨迹数据。下面的 §0.6.2 之后主要讲 A/B/C 的公共基础——把代码资产清洗好，三条路都用得上。

### 0.6.2 数据盘点：公司的代码资产其实有 9 种

大多数团队只盘点了 1-2 种就开始训练，导致模型"只懂代码本身、不懂代码背后的意图"。完整清单（按信号密度从高到低）：

| # | 数据源 | 独特价值 | 典型体量（中型公司） |
|---|---|---|---|
| 1 | **Git 仓库主分支代码** | 核心，所有路线的底座 | 500 万 - 5000 万行 |
| 2 | **README / CONTRIBUTING / ADR / RFC** | 教模型"这段代码为什么存在、怎么正确使用" | 几千个文档 |
| 3 | **Issue + PR + Code Review 评论** | **最被低估的金矿**——天然的 (问题, 解决方案) 对 | 数万-数十万条 |
| 4 | **Commit message + diff** | 细粒度的 (意图, 代码变更) 对 | 数十万 commits |
| 5 | **单元测试 + 集成测试** | (规约, 实现) 对；RL 阶段可做**可验证奖励** | 几十万个测例 |
| 6 | **API schema (proto / OpenAPI / GraphQL)** | 跨服务调用关系的结构化真相 | 几百个 proto |
| 7 | **企业内部 Wiki / Confluence / Notion** | 架构决策、运维手册、培训材料 | 数千页 |
| 8 | **Slack / 企业微信工程频道历史** | 真实开发对话，含大量 troubleshooting | 百万级消息 |
| 9 | **CI/CD 运行日志 + 失败恢复记录** | 教模型识别"这种 stack trace 对应什么 bug" | GB 级日志 |

**优先级建议**：第一轮只做 1+2+3+5；2-3 轮再补 4+6+7；8+9 敏感度高、信噪比低，除非特别场景别做。

### 0.6.3 私有代码的"多 5 道关"：比公开代码更复杂的清洗

公开代码 pipeline 那 12 步全都要做，此外**额外加这 5 道关**，任何一步漏了都可能爆炸：

#### (1) 秘密扫描（必做 · 最优先）
工具：`trufflehog`、`gitleaks`、`detect-secrets`。扫目标：AWS/GCP keys、数据库连接串、私钥、token、证书、OAuth secrets。

```bash
# 对每个 repo 跑一次全历史扫描
trufflehog git file://./our-repo/ --json > leaks.json
# 命中的文件整个丢弃（不是脱敏——历史总能恢复）
```

一旦训练进去，就算你脱敏了代码，模型可能根据上下文**猜出来**——这是私有代码训练最大的失败模式。

#### (2) PII / 客户数据脱敏
员工姓名、邮箱、内部 IP 网段、客户 ID、真实业务数据（测试 fixture 里常出现）。用规则 + NER 混合检测，替换为 `<EMAIL>` / `<IP>` / `<CUSTOMER_ID>` 占位符。

#### (3) 访问控制继承（Access-Aware Training）—— 数据治理难题

公司 A 的开发者能看仓库 X，开发者 B 看不到。但你训出来的模型对 A 和 B 的回答是一样的。**这就是数据泄漏**。

实践中有三种处理姿态：

- **保守**：只用**全公司可见**的仓库训练（通常只剩 10-20% 代码），其他走 RAG 按权限过滤
- **激进**：全部喂进去，靠产品层做访问控制（风险：prompt injection 可能绕过）
- **分层**：训 2-3 个模型，按访问级别（全员 / 工程部门 / 特定团队）分层微调
  
GLM-5.1 级别的 MoE 模型还可以尝试"按访问域分专家"，但这是前沿做法。

#### (4) 去重 against 公开数据
你公司 fork 过的开源项目、引入的 npm/pypi 包、从 StackOverflow 抄的代码，本质已经在公开预训练集里。重复训练这些**浪费算力、加剧过拟合、还干扰信号**。做法：拉 The Stack v2 的 file-level SHA256 表，与你的私有代码做 hash diff，命中的剔除。

#### (5) 仓库质量分层与生命周期筛选
私有仓库 50% 是僵尸代码——废弃项目、一次性 PoC、离职员工的实验分支。信号维度：

- 最近 12 个月有 commit：✓
- 有 CI 通过记录：✓
- 有至少 1 个 contributor 非作者本人：✓
- 不在 `archived / deprecated / sunset` 列表：✓
- 有 README 且 > 50 字：✓

通过 3/5 进 main bucket，通过 1-2 个进 low-weight bucket，否则丢弃。

### 0.6.4 把代码"加工"：从裸 content 到高信号训练语料

这一步是私有代码训练**胜负手**——原始代码 token 信号密度不够，需要"增强"。

#### 技术 1：Repo 级元数据前置（Repo Context Injection）
每个文件训练样本前面拼接 **"这是什么项目"** 的摘要：

```text
<|repo:payment-service|>
<|description: 负责所有支付收单、退款、对账的核心服务|>
<|path: src/refund/policy.py|>
<|lang: python|>

class RefundPolicy:
    def can_refund(self, order: Order) -> bool:
        ...
```

让模型学到**路径、项目、文件**的语义关联，而不是看每个文件都孤立。

#### 技术 2：Import-Graph Repo Packing
已在 §0.0.4 提过，私有场景特别有用：把 `pyproject.toml` / `package.json` / `go.mod` 解析出依赖，同一包里的文件按**拓扑序**拼接进同一条 8k 序列。

#### 技术 3：Issue-PR-Diff 三元组
单个 issue 通常对应 1-N 个 PR，每个 PR 是 (title, description, diff, review comments, final merged code)。打包成高价值样本：

```text
### Issue #1234 · 退款超时场景偶发双重退款
**Reporter**: eng
**Description**: 用户在网络抖动时连续点击退款按钮，产生两笔退款记录...

### PR #1251 · Fix double-refund race
**Files changed**:
```
diff --git a/src/refund/policy.py ...
```
**Review**:
- @reviewer: 用分布式锁会不会太重？改用 idempotency key 更合适...
- @author: 好，已改用 `order_id + user_id` 做 idempotency key
```

这种样本一次能教会模型：**业务问题 → 技术权衡 → 最终实现**。价值 10× 于纯代码。

#### 技术 4：Test-Code 配对（(spec, impl) 对）
对每个非测试文件 `foo.py`，找它对应的 `test_foo.py`（或 `FooTest.java` 等），生成两种训练形态：

- **Spec → Impl**：`[测试代码] → [被测代码]`，教模型从规约生成实现
- **Impl → Spec**：`[被测代码] → [测试代码]`，教模型为新代码写测试

这是 D 能力（Agent 写单测）的核心数据。

#### 技术 5：Docstring / 注释扩写
用大模型（GPT-5 / Claude / GLM-5.1 自身）给内部代码**批量生成 docstring**，然后把"代码 → docstring"和"docstring → 代码"双向做成训练样本。这是最廉价的数据增强，通常能涨 5-10 个百分点的内部评测。

#### 技术 6：Commit Message + Diff 指令化
把 git 历史改写成指令样本：

```text
USER: 把 order.py 里的 status 字段从 string 改为枚举，保留向后兼容
ASSISTANT: <think>要同时处理序列化/反序列化/DB 迁移...</think>
```diff
[真实 commit 的 diff]
```
```

十万个 commits = 十万条对话数据。质量取决于你们 commit message 的规范度。

### 0.6.5 能喂多少？—— 数据体量的现实估算

**估算公式**：代码行数 × 每行约 6-10 token × 清洗存活率 30-50% ≈ 可训练 tokens。

| 公司规模 | 代码行数 | 估算 tokens | 推荐路线 |
|---|---|---|---|
| 创业公司 | 10-100 万行 | 3-30M tokens | **只做 SFT + RAG**，CPT 信号太弱 |
| 中型 | 500-5000 万行 | 150M-1.5B tokens | **CPT 1-2 epoch 混 20-40% 公开代码** 防遗忘 |
| 大型科技公司 | 5000 万 - 数亿行 | 1.5B-20B tokens | 可以做**严肃 CPT**，甚至 from scratch 小模型 |
| 超大型（Google / MS 规模） | > 10 亿行 | > 30B tokens | 专门 from-scratch 训练私域基座 |

**关键经验**：内部代码独立训练几乎都**过拟合**（模型记住了特殊 API 但忘了通用推理）。通行做法：

- **混合配比**：内部 30-50% + 公开代码 40-60% + 数学 / 通用 5-10%
- **学习率很小**：CPT 从 `1e-5` 起步（预训练是 `1e-4` 级别）
- **Epoch 控制在 1-2**：防止灾难性遗忘；用 replay buffer 少量混最初的公开数据

### 0.6.6 端到端最小 pipeline 脚本（可直接改着用）

一个真实的"从 git clone 到训练集 parquet"最小可行流水线：

```python
"""
private_code_pipeline.py
依赖: pip install datatrove trufflehog-python gitpython tiktoken fasttext go-enry
"""
from pathlib import Path
import subprocess, json, hashlib, re
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.filters import LanguageFilter, GopherQualityFilter
from datatrove.pipeline.dedup import MinhashDedupSignature, MinhashDedupBuckets
from datatrove.pipeline.writers import ParquetWriter
from datatrove.executor import LocalPipelineExecutor

REPOS = Path("/mnt/data/company-repos.txt").read_text().splitlines()
OUT = Path("/mnt/data/private-corpus")

# ---------- Stage 1: clone + 扫码 + 扁平化 ----------
def clone_and_extract(repo_url: str, out: Path):
    name = repo_url.split("/")[-1].replace(".git", "")
    dst = out / "cloned" / name
    if not dst.exists():
        subprocess.run(["git", "clone", "--depth=1", repo_url, str(dst)], check=True)
    # 秘密扫描：命中的文件整个丢
    leaks = subprocess.run(
        ["trufflehog", "filesystem", str(dst), "--json"],
        capture_output=True, text=True,
    )
    banned = {json.loads(l)["SourceMetadata"]["Data"]["Filesystem"]["file"]
              for l in leaks.stdout.strip().splitlines() if l}
    # 遍历文件写 jsonl
    sink = (out / "raw" / f"{name}.jsonl").open("w")
    for f in dst.rglob("*"):
        if not f.is_file() or str(f) in banned: continue
        if f.suffix not in {".py", ".go", ".java", ".ts", ".tsx", ".rs", ".kt"}: continue
        try: content = f.read_text(encoding="utf-8", errors="ignore")
        except: continue
        if len(content) < 20 or len(content) > 1_000_000: continue
        record = {
            "repo": name,
            "path": str(f.relative_to(dst)),
            "content": content,
            "sha256": hashlib.sha256(content.encode()).hexdigest(),
            "size": len(content),
            "ext": f.suffix,
        }
        sink.write(json.dumps(record, ensure_ascii=False) + "\n")
    sink.close()

for url in REPOS:
    clone_and_extract(url, OUT)

# ---------- Stage 2: PII / Email / IP 脱敏 ----------
PII_PATTERNS = [
    (re.compile(r"[\w._-]+@(ourcorp|internal)\.com"), "<EMAIL>"),
    (re.compile(r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "<INTERNAL_IP>"),
    (re.compile(r"\bemp\d{6}\b"), "<EMP_ID>"),
]
def scrub(text):
    for p, rep in PII_PATTERNS: text = p.sub(rep, text)
    return text

# ---------- Stage 3: datatrove 跑去重 + 过滤 ----------
pipeline = [
    JsonlReader(data_folder=str(OUT / "raw")),
    LanguageFilter(languages=["en"]),          # 注释大多英文，过滤纯非英文
    GopherQualityFilter(),                      # 启发式质量
    MinhashDedupSignature(output_folder=str(OUT / "sig"), config={"num_perm": 128}),
    MinhashDedupBuckets(input_folder=str(OUT / "sig"), output_folder=str(OUT / "buckets")),
    ParquetWriter(output_folder=str(OUT / "clean"), output_filename="shard_${rank}.parquet"),
]
LocalPipelineExecutor(pipeline=pipeline, tasks=16).run()

# ---------- Stage 4: Repo 元数据前置 + Tokenize ----------
# 实际训练前把 README 摘要 + path tag 拼到 content 头部
# 然后用目标模型的 tokenizer（GLM tokenizer）转 token，写 mmap .bin
```

这是骨架，真实 pipeline 还要加：许可证继承检查、CI 成功率筛选、import-graph 拓扑排序。但从 0 到 1 跑通，这脚本够用。

### 0.6.7 评测体系：公开 benchmark 对你没意义

HumanEval 考你模型懂不懂 Python，不考它懂不懂你们的 `PaymentService`。必须自建**公司内部评测集**。三层建议：

**Layer 1 · 知识问答集（50-200 题，人工出）**
- "我们有没有 xxx 功能的模块？在哪个 repo？"
- "`auth.Login()` 的 MFA 参数是哪几个？"
- "退款服务 timeout 设多少？为什么？"

评估：大模型 judge 打分，或者配标准答案查 F1。

**Layer 2 · 代码生成集（20-50 题）**
- "给 `OrderService` 加一个按用户 ID 查订单的接口，遵循我们的 API 规范"
- 评估：生成代码能否在内部 test 环境编译 + 通过我们写的"风格 + 正确性" judge test

**Layer 3 · Agent 任务集（10-30 题，最高价值）**
- 从真实 Jira ticket 里挑已解决的，抹掉最终 PR，让 agent 从零完成
- 评估：完整的 CI pipeline 通过 + code review approval
- 这就是**内部版 SWE-Bench**

第 3 层最贵但最有用。建议先做第 1 层快速迭代，第 3 层在 V2 版本上线。

### 0.6.8 数据治理 · 权限 · 模型"遗忘"

几个企业场景特有的治理问题：

1. **数据血缘**：训练集里每条样本来自哪个 repo 哪个 commit？出了问题能否回溯？建议给每条样本带 `(repo, sha, timestamp)`。
2. **权重也是敏感资产**：训完的 checkpoint 包含了公司代码信息，**不能发 Hugging Face public、不能放公网可访问的存储**；内部部署走 VPC + SSO。
3. **离职员工代码**：员工 A 离职后法务说他的代码要从系统抹除——**这在当前 LLM 技术下基本做不到**（machine unlearning 是开放研究问题）。现实处理：提前在数据合同里约定"训练数据归属公司"，把法律风险前置。
4. **仓库归档 / 业务线砍掉**：被砍的业务代码继续在模型里影响输出，用户以为功能还在。定期（季度）重训 delta 是目前唯一办法。
5. **客户代码混入**：SaaS 公司代码库里可能有客户代码（配置、模板、PoC）。这属于**受托数据**，没授权不能训练——做一次完整审计比事后补救便宜 100 倍。

### 0.6.9 4 周 MVP 路线（从零到能跑的私有 Coding LLM 数据侧）

| 周 | 交付 | 关键动作 |
|---|---|---|
| **Week 1** | 资产地图 + 合规清单 | 盘点 9 类数据源，秘密扫描全仓库，过法务/安全基线 |
| **Week 2** | 清洗 pipeline 跑通 | §0.6.6 脚本本地跑通，产出 parquet + tokenized bin |
| **Week 3** | RAG baseline 上线（拿到真实反馈） | 用 embedding 模型建索引，挂在内部 IM，收集用户 query |
| **Week 4** | CPT 试验 + 内部评测集 v0 | 在 GLM-4.5-Air 上跑 LoRA CPT（10-30B tokens 够），同步出 50 题评测 |

Week 3 的 RAG 不是"凑数"——**它本身就是能力 C 的生产方案**，同时收集到的用户真实 query 是 Week 5+ 做 SFT 的金矿。

---

**一句话带走**：企业私有代码训练的核心不是"我数据少"，而是**"我数据的信号密度要远高于公开代码"**——靠 Issue-PR 对、Test-Code 对、Commit 改写、元数据前置把每个 token 的价值放大 5-10 倍，配上 30-50% 公开代码做 anchoring，中型公司完全有机会做出**比通用 GLM-5.1 更懂自己代码**的专属模型。

---

## 1. 代码预训练数据总览：主流数据源的体量与特点

下面的数字都是"下载时的大致量级"，实际进 tokenizer 前会再缩 3-10 倍。

| 数据源 | 体量（原始） | 体量（dedup 后 tokens） | 典型用途 | 坑点 |
|---|---|---|---|---|
| **The Stack v2 full** (bigcode) | ~32 TB, 600+ 语言 | 约 900B tokens | 主力代码语料 | 要用 SWHID 重新 resolve，直接下全量基本不现实；走 `the-stack-v2-dedup` 子集 |
| **The Stack v2 dedup** | ~6.4 TB | ~460B tokens | 直接能喂的版本 | 许可证过滤已做到 permissive-only，非 permissive 要自己重做 |
| **GitHub raw（自爬）** | 不限，看仓库列表 | - | 补最新 / 特定领域仓库 | 合规风险自担，Rate limit 严重 |
| **CommitPack / CommitPackFT** (bigcode) | 4 TB / 2 GB | ~70B / - | 指令跟随、diff 风格预训练 | commit msg 噪声大，需二次清洗 |
| **GitHub Issues & PRs** (bigcode/the-stack-github-issues) | ~54 GB | ~15B | 对话式代码推理 | 需要按 thread 组装，去掉机器人评论 |
| **Jupyter Notebooks** (bigcode/jupyter-code-text-pairs) | ~50 GB | ~10B | 交错 code/text，天然 CoT | 解析 `.ipynb` JSON，output cell 经常是大 base64 图 |
| **Kaggle Notebooks** | ~10 GB | ~3B | 数据科学风格 | 许可证模糊 |
| **StackExchange / StackOverflow** (HF: HuggingFaceH4/stack-exchange-preferences, RedPajama) | ~75 GB | ~20B | QA、解释性文本 | HTML→Markdown 时代码块边界易漂 |
| **文档 / Docs**（readthedocs, MDN, cppreference 等爬虫） | ~20 GB | ~5B | API 说明，降幻觉 | 版权：MDN CC-BY-SA，需保留署名 |
| **竞赛代码** (CodeContests, APPS, TACO) | 小 | ~1B | 算法题风格 | 与 HumanEval / MBPP 有重合，务必 decontam |
| **数学** (proof-pile-2, OpenWebMath, FineMath) | ~100 GB | ~50B | 强化 reasoning | 代码 LLM 里加 5–10% 明显涨 MBPP / HumanEval-CoT |

**GLM-4.5 ARC 的配方关键点**（2508.06471 第 3 节）：
- 代码主体用自建 GitHub 抓取 + The Stack v2 去重合并，按仓库质量分层。
- 显著提升了 repo-level 长序列样本占比（>32k token 的仓库拼接样本 >20%）。
- 明确加入 code-related math（Lean、Isabelle 证明、竞赛题解）作为单独 bucket。

**DeepSeek-Coder-V2 的配方关键点**（2406.11931 §2）：
- 语料 60% 源代码 + 10% math + 30% natural language。
- 相比 V1 把语言数从 86 拉到 338。
- 在 fill-in-the-middle 上保留 PSM 格式；文件级 dedup 后 token 数 1170B。

**OpenCoder 的配方关键点**（2411.04905）：
- 把整条 pipeline 全部开源（配置、规则、代码），是目前最值得照抄的参考实现。
- 预训练 2.5T tokens；RefineCode 阶段用 PPL + 规则过滤二次清洗。
- 在 annealing 阶段喂高质量合成数据，这是 2024 年后新范式。

---

## 2. 完整 Pipeline 流程图

```mermaid
flowchart TD
    A[Raw: The Stack v2 / GH clone / Archive] --> B[语言检测 go-enry / linguist]
    B --> C[许可证过滤 license whitelist]
    C --> D[精确去重 file-level SHA256]
    D --> E[近似去重 MinHash-LSH repo/file-level]
    E --> F[启发式过滤 heuristic rules]
    F --> G[质量打分 fastText / small LM PPL]
    G --> H[PII 脱敏 email / key / IP]
    H --> I[去污染 n-gram vs HumanEval/MBPP/LCB ...]
    I --> J[FIM 改写 PSM/SPM 50%]
    J --> K[配比采样 per-language / per-bucket]
    K --> L[Tokenize BPE/SentencePiece]
    L --> M[Packing 2k/4k/8k 序列, repo-aware]
    M --> N[写出 mmap .bin/.idx 或 parquet shards]
    N --> O[训练 DataLoader]
```

每一步对应一个独立 pipeline stage；在 `datatrove` 里每个 stage 就是一个 `PipelineStep`，天然支持 checkpointing。工程上务必做到：**每个 stage 都能独立重跑，不依赖前一步在内存里的状态**。中间产物统一落 parquet（列式、快，支持 push-down filter）。

---

## 3. 每一步的技术细节

### 3.1 语言检测

- 工具：`go-enry/go-enry`（GitHub linguist 的 Go 端口，速度快）或 `github-linguist` 本身。
- 不要只看扩展名：`.h` 可能是 C / C++ / Objective-C；`.pl` 可能是 Perl 或 Prolog；`.ts` 有 TypeScript 和 TypoScript 两种。
- enry 先按扩展名筛候选，再用 Bayesian classifier 看内容。
- 实战：对 600 语言做 whitelist，只保留 ~80 个主流语言（Python、C、C++、Java、JS、TS、Go、Rust、Kotlin、Swift、Scala、Ruby、PHP、C#、Shell、SQL、HTML、CSS、Markdown、YAML、Dockerfile、CMake、Make、Lua、R、Julia、Haskell、OCaml、Elixir、Erlang、Clojure、Dart、Objective-C、Perl、Fortran、Assembly、Verilog、VHDL、Solidity、Lean、Isabelle、Coq、Racket、Scheme、TeX、Matlab...）。
- 小语种样本量不足以泛化，喂进去反而拖 loss。

### 3.2 许可证过滤

- The Stack v2 已按 permissive / non-permissive 打过标签，字段 `license`。
- 白名单（permissive）：`MIT`、`Apache-2.0`、`BSD-2-Clause`、`BSD-3-Clause`、`ISC`、`0BSD`、`Unlicense`、`CC0-1.0`、`MPL-2.0`（有 copyleft，保守可剔）、`Zlib`、`WTFPL`。
- 黑名单：`GPL-*`、`AGPL-*`、`LGPL-*`（LGPL 有争议但多数项目规避）、`CC-BY-SA`、`EPL`、`CDDL`、无许可证（`no license` 按 "All rights reserved" 处理，一律剔）。
- 仓库级判定：优先看 `LICENSE` / `COPYING` 文件 + `package.json` / `Cargo.toml` / `pyproject.toml` 里的 license 字段；头部 SPDX 注释作为 fallback。
- GLM / DeepSeek 在报告里没有完全公开，但 StarCoder2 和 OpenCoder 都严格只用 permissive。如果你要对外发布模型，必须 permissive-only，否则只能内部用。
- 额外一层：opt-out（"Am I in The Stack"）。StarCoder2 尊重作者的退出请求列表；合规起见你应该 join 最新一版 opt-out list。

### 3.3 精确去重（exact dedup）

- 最便宜先做：file-level SHA256（或者 xxhash64，更快）。
- 仓库内精确重复（vendored 文件、copy of node_modules 提交进来）会吃大量重复 token，SHA256 一遍能砍掉 20–40%。
- 实现：map stage 输出 `(hash, path, repo)`，reduce 按 hash 分桶，每桶留一个（优先 star 数高的仓库那份）。
- 内存：`hash → repo_id` 表，4 亿文件 × 40B ≈ 16GB，单机内存能扛；再大就 RocksDB 或 bloom filter 预筛。

### 3.4 近似去重 MinHash-LSH（**最关键的一步**）

这是区分一个 pipeline 工业级与否的分水岭。参数来自 StarCoder / StarCoder2 和 OpenCoder 的公开配置：

- **N-gram**: 字符级 5-gram，或 token 级 7-gram。代码里字符级更稳，因为变量名对语义影响大。OpenCoder 用字符级 5-gram；StarCoder 用 word-level 7-gram。
- **Hash 数（permutations / num_perm）**: 256。再少（128）召回掉，再多（512）开销翻倍但提升小。
- **Bands × Rows**: bands=50, rows=5（50×5=250≈256）。对应相似度阈值大约 $s^*=(1/b)^{1/r}=(1/50)^{1/5}\approx 0.55$。
- **Jaccard 阈值**: 0.7–0.85。代码去重比自然语言保守，0.7 足够去掉 refactor / rename 的 near-dup；0.85 会漏。
- **粒度**：
  - 先 **文件级** MinHash，团掉 fork / vendored code。
  - 再 **仓库级** MinHash（整个仓库所有文件拼成一个"文档"）打 fork 仓库。
  - StarCoder2 用 "near-dedup groups" 图划分：MinHash-LSH 建候选对 → union-find 得连通分量 → 每组保留一个代表（按 star、最近提交时间打分）。
- **工具**：
  - `datasketch`（纯 Python，小数据用）。
  - `text-dedup`（chenghao shen 写的，专门做 pretraining 去重，支持 spark / slurm / datatrove hook）。
  - `bigcode-project/bigcode-dataset/near_deduplication`：官方版本，PySpark + MinHash，作为参考实现。
- **内存与规模**：10B 文档做 LSH 需要 shuffle，单机不现实，必须 Spark / Ray。务必做 shard-wise：按 hash 前缀分 1024 个 bucket 分布式 join。
- **踩坑**：
  - 不做大小写归一化 / 空白归一化会高估差异；建议预处理时 lower-case（但代码里保留大小写是有意义的，所以只 strip 连续空白并去注释后做 hash）。
  - 空文件 / 很短文件（<50 字符）直接丢，不要进 MinHash，否则 false positive 爆炸。

### 3.5 启发式过滤（heuristic filters）

这些规则多数直接抄自 StarCoder2 paper Appendix C 与 `bigcode-dataset/preprocessing/filtering.py`：

通用规则（所有语言）：
- **平均行长**：均值 > 100 字符 或 最大行长 > 1000 → 丢。长行多半是压缩 / 自动生成。
- **字母数字比例**：(A-Za-z0-9 字符) / 总字符 < 0.25 → 多半是二进制 / base64 / assets，丢。
- **字母比例**：字母 / 总字符 < 0.15 → 丢。
- **行数**：< 3 行（除非是 config 文件）或 > 100000 → 丢。
- **最长"单词"**：> 1000 字符 → 丢（minified JS 典型特征）。
- **十六进制数比例**：> 50% → 丢（二进制 dump）。
- **decimal digit 比例**：> 60% → 丢（数据文件）。
- **编码**：非 UTF-8 直接丢；含大量 replacement char (�) 丢。
- **最小 token 数**：<10 个标识符的文件基本没信息，丢。

自动生成文件检测（auto-generated detection）：
- 匹配常见标记：`DO NOT EDIT`、`AUTO-GENERATED`、`autogenerated`、`@generated`、`Code generated by`、`This file was automatically generated`、`DO NOT MODIFY`。
- 文件名启发：`*.pb.go`、`*_pb2.py`、`*.min.js`、`*.bundle.js`、`jquery.js`、`*.lock`、`package-lock.json`、`yarn.lock`、`Pipfile.lock`、`poetry.lock`、`Cargo.lock`、`go.sum`、`.ipynb_checkpoints/*`、`__pycache__/*`。
- 用工具：`bigcode-dataset/preprocessing/filtering/utils.py` 里有 `is_autogen_file()` 直接能用。

语言特定规则（示例）：
- **Python**: 含 `#!/usr/bin/env python` 且全文无 `def`/`class` → 脚本，保留；但 `<=` 5 行 + 无函数 → 丢。ratio of `print(...)` lines > 80% → demo / debug 脚本，降权。
- **HTML/CSS/XML**: HTML 里 `<script>` 块 > 50% 行数 → 很可能是代码混入，切出 script 块单独按 JS 处理。
- **JSON / YAML**: 非 config（检测 well-known 文件名）一律丢；大 JSON 数据文件毫无训练价值。
- **Markdown**: 代码块少于 2 个且总长 < 500 字符 → 丢；README 保留但要区分 bucket。
- **SQL**: 仅 INSERT 语句的 dump 文件 → 丢（常见数据泄漏）。
- **Shell**: `base64 -d` / `eval` 比例高 → 很可能是恶意脚本，丢。

XML 解析可用 `bigcode-dataset/decontamination` 里的 `is_xml()` 轻量判别；更复杂的走 `tree-sitter` 能确认能否 parse（parser 失败 → 丢）。

### 3.6 质量打分

两条路线，建议两种都用，投票或加权：

**路线 A：fastText 二分类器（快、糙）**
- 正样本：high-star repo + curated（e.g., Python 标准库、linux kernel、TensorFlow）。
- 负样本：随机抓样 + 已被启发式过滤过的"低质样本"。
- 训练：`fasttext supervised -input train.txt -output code_quality -lr 0.5 -epoch 5 -wordNgrams 2 -bucket 200000 -dim 100`。
- 推理阈值：打分 < 0.3 → 丢；0.3–0.6 → 降权（下采样 0.5x）；> 0.6 → 保留。
- OpenCoder 用类似思路，参见其 RefineCode 文档。

**路线 B：小 LM 的 PPL 打分（准、慢）**
- 拿一个 160M / 410M 的 Pythia / TinyLlama，在一个"干净"语料上短训 5–10B tokens，作为 scorer。
- 对候选文件计算 per-token NLL，按语言归一化（不同语言 entropy 不同）；超过 mean + 2σ 的丢。
- DeepSeek-Coder-V2 和 GLM-4.5 都明确用过 PPL-based filtering，阈值不公开但典型做法是每语言保留 PPL 分布的 5–95 分位。

**组合**：fastText 做第一刀砍掉 30–50%，小 LM 做第二刀再砍 10–20%。

### 3.7 PII 脱敏

- 直接用 `bigcode-project/bigcode-dataset/pii/` 的 starpii 模型 + 正则组合。
- 覆盖类型：email、IPv4/IPv6、SSH / RSA / DSA / EC private key block、AWS access key / secret、GitHub token (`ghp_…`、`gho_…`、`ghs_…`)、Slack token (`xox[baprs]-…`)、JWT、Stripe `sk_…`、Google API key (`AIza…`)、phone number（各国格式）、信用卡（Luhn 校验）。
- 正则能抓 80%，starpii NER 模型补长尾（命名实体：人名、地址）。
- 替换策略：email → `<EMAIL_0>`、`<EMAIL_1>`…（保持同一文档内可指代性）；key → 直接替换为 `<KEY>`；人名 → `<NAME>`。
- **千万别原样保留私钥**：出现带 `-----BEGIN RSA PRIVATE KEY-----` 的文件，整个文件丢弃，不是替换，因为即使 mask 了也证明这份文件的来源仓库一定泄漏过 key，法律 / 伦理风险大。

### 3.8 去污染（decontamination）——最容易被跳过、但最致命

目标：不让评测集的原题进训练集，哪怕是改写过的。

**评测集清单**（训练代码 LLM 至少要防这些）：
- HumanEval / HumanEval+ / HumanEval-X
- MBPP / MBPP+
- LiveCodeBench（动态，要定期更新）
- APPS / CodeContests
- DS-1000
- SWE-bench / SWE-bench Verified / SWE-bench Lite
- BigCodeBench
- CRUXEval
- ClassEval
- MultiPL-E
- 数学：GSM8K、MATH、MATH-500、AIME

**算法**（参考 `bigcode-project/bigcode-dataset/decontamination`）：
1. 对每个评测集样本，抽取 prompt + canonical solution。
2. 做 **10-gram**（word-level）精确匹配：把每个 eval sample 切成 10-gram 集合 $E_i$。
3. 训练样本切 10-gram 集合 $T$。
4. 若 $|E_i \cap T| / |E_i| > 0.5$（或任意 2 个以上 10-gram 命中），判为污染，丢弃该训练样本。
5. 对短样本（<10 tokens）退化成 full-string 匹配。

**变体**：
- StarCoder 最初用的是 substring match + 自定义 tokenizer。
- OpenAI / DeepSeek 用 13-gram。
- LiveCodeBench 要按"release date 之后的数据"切，避免时间泄漏。

**实现要点**：
- 在 tokenize 之前做（基于字符 / 词），而非 tokenize 之后，避免 tokenizer 差异。
- 用 **inverted index**（10-gram → doc_ids）加速，不要两两比。
- Decontam 放在 FIM 改写之前，因为改写后匹配会变难。
- 同时对 issue 数据、commit msg、docs 做（HumanEval 的题目在 blog 里也被抄过）。

**验证方式**：decontam 之后挑 20 个 HumanEval 样本，grep 训练集，必须 0 命中。

### 3.9 FIM 改写

参考 Bavarian et al. 2022 (arxiv 2207.14255) 以及 DeepSeek / StarCoder2 的做法：

- **格式**：
  - PSM (Prefix-Suffix-Middle)：`<PRE>{prefix}<SUF>{suffix}<MID>{middle}<EOT>`
  - SPM (Suffix-Prefix-Middle)：`<SUF>{suffix}<PRE>{prefix}<MID>{middle}<EOT>`
  - special tokens 在 tokenizer 里预留。
- **应用比例**：50% 文件转成 FIM 样本；PSM:SPM = 1:1。
- **span 采样**：
  - 在 **字符级** 先随机切（避免 token 边界偏差再 detokenize）。
  - 切分点：uniform random，两刀切出 prefix / middle / suffix。
  - middle 长度约束：至少 1 个字符，不超过文件的 50%。
- **上下文完整性**：FIM 改写必须在 **文件级**（对单文件做），不要跨文件切，否则模型学不到真实的补全分布。
- **代码块对齐**（可选）：部分实现会在行边界 / 表达式边界切，效果更稳，但实现复杂；初版用纯 random 就够。

### 3.10 配比采样与 bucket 策略

把数据划 bucket，然后按配方采样。典型 bucket：

```
code_main          (70%)  Python/Java/C/C++/Go/Rust/TS/JS 等主流
code_longtail      (10%)  小语种
code_nl            (5%)   issues, commits, PRs
docs               (3%)   README, readthedocs, MDN
stackexchange      (2%)   Q&A
jupyter            (2%)   code/text interleave
math_reasoning     (5%)   OpenWebMath, proof-pile-2
general_text       (3%)   RedPajama-Common-Crawl 代表性子集
```

- **重要**：在 `code_main` 内部按 token 数做 **temperature sampling**（α=0.3 左右），防止 JS/Python 把小语种淹没；OpenCoder / StarCoder2 都这么做。
- **GLM-4.5 / GLM-5.1** 的配比没完全公开，但 2508.06471 表格里显示 math + reasoning 明显 ≥ 10%，这是它 coding 分高的关键差异之一。
- **DeepSeek-Coder-V2**：code 60% + code-related NL 10% + general NL 30%，general NL 比例高是为了保通用能力，适合做"通用 + coding 双强"的模型。
- **annealing 阶段**：训练末期把高质量数据（curated tutorial、教材式代码、精选 PR diff）比例拉到 30–50%，学习率同步 decay。OpenCoder 明确用这个。

### 3.11 Tokenize

- 词表大小：64k（StarCoder）→ 128k（DeepSeek V2 / GLM-4.5）→ 150k+（Qwen2.5-Coder）。代码 tokenizer 比纯文本要大，典型 100k+。
- 算法：BPE（sentencepiece 的 `--model_type=bpe` 或 `tokenizers` 的 `ByteLevelBPETokenizer`）。
- 训练语料：从最终 dedup/filter 后的代码语料抽 5–20GB sample。
- 预 tokenization：**byte-level BPE** 是当前主流，避免 UTF-8 错位；`GPT2TokenizerFast` / tiktoken cl100k 都是参考。
- 特殊 token：`<fim_prefix>` `<fim_middle>` `<fim_suffix>` `<fim_pad>` `<filename>` `<reponame>` `<gh_stars>` `<issue_start>` `<issue_comment>` `<issue_closed>` `<jupyter_start>` `<jupyter_text>` `<jupyter_code>` `<jupyter_output>` 等——这批 meta token 训练时拼在样本头，推理时可以用作 control signal。
- 压缩率评估：训完 tokenizer 后对各语言算 bytes/token；Python 期望 ~3.5–3.8，C/C++ ~3.3，JSON ~2.8。差太多说明词表不平衡。

### 3.12 Packing（repo-aware）

长上下文（32k / 128k）模型需要 repo-level packing，否则样本大部分是 <2k 的小文件。

**朴素 packing**：把短序列 concat 到目标长度（用 EOS 分隔），高效但跨文件无语义。

**Repo-level packing（GLM-4.5、DeepSeek-Coder-V2 的做法）**：
- 按 **仓库** 聚合，同仓库文件放一起。
- 仓库内按 **import / 依赖图** 拓扑排序：被 import 的文件放前面，调用方放后面。
- 具体实现：
  - Python：解析 `import` / `from X import Y`，构 DAG，拓扑序。静态分析工具：`import-linter`、或直接用 `ast` 扫顶层。
  - JS/TS：解析 `import` / `require`，用 `@babel/parser`。
  - Java：按 package + import；同 package 内按文件名字典序。
  - C/C++：按 `#include`，但 include graph 有环，用 SCC + 条件拓扑。
  - Go：按 module 路径 + package。
- fallback：无法解析或跨语言混合时，按文件路径字典序（同目录聚集）。
- 拼接格式：
  ```
  <reponame>owner/repo<gh_stars>123
  <filename>src/utils.py
  {code}
  <filename>src/main.py
  {code}
  ...
  <eos>
  ```
- **窗口**：2k / 4k packing 用于早期阶段，8k/32k 用于长上下文阶段，128k 在 annealing / 长上下文扩展阶段做。
- 参考：DeepSeek-Coder V1 (2401.14196) §3.2 有 "Repository-Level Data Construction" 完整描述，V2 沿用。

### 3.13 写出格式

- **mmap binary**（Megatron-LM 风格）：`.bin`（连续 token id, uint16/uint32）+ `.idx`（doc offsets）。训练时 mmap 零拷贝，最快。
- **Parquet shards**：每个 shard 1–2GB，列 `input_ids`、`length`、`meta`。适合 HF `datasets` + `webdataset` 流式。
- **WebDataset tar**：多 GPU 跨节点流式友好，适合云对象存储。
- **datatrove 原生**：写 `ParquetWriter` 或 `JsonlWriter`，后续再转 mmap。

推荐组合：**parquet 做中间态**（人能 debug，列式查询快）+ **mmap 做最终训练态**（性能最优）。

---

## 4. 数据配比：GLM-4.5 vs. DeepSeek-Coder-V2

两家都没完全公开配方，但从论文和放出的报告能拼：

| 维度 | DeepSeek-Coder-V2 (2406.11931) | GLM-4.5 / GLM-5.1 ARC (2508.06471) |
|---|---|---|
| 总 token | ~10.2T（含 V1 的 2T + 新增 8.2T） | 未完全披露，推测 15T+ |
| 代码占比 | 60% | 略高，~65–70% |
| 代码相关 NL（issue/commit/docs） | 含在 math+NL 40% 内 | 显式 bucket，~10% |
| Math | 10% | ≥ 10%，含 Lean/Isabelle formal |
| 通用 NL | 30% | ~15–20%（更偏 reasoning-heavy） |
| 语言数 | 338 | 未明说，推测类似 |
| Repo-level packing | 是（V1 就有） | 是，且占比显著提升 |
| FIM | 50% PSM | 用，比例未公开 |
| 长上下文阶段 | 128k via YaRN | 128k / 1M，分阶段扩展 |
| 质量过滤 | PPL + 规则 | PPL + 规则 + 多阶段筛选 |

**经验判断（你自己复现时）**：
- 15B 以下的模型，代码 70% + math 10% + NL 20% 是个稳的起点。
- math 从 5% 往 10% 提，HumanEval 不一定涨，但 HumanEval-CoT、APPS、MATH 会涨。
- NL 拉到 30%+ 对 coding 有轻微负影响，适合"通用模型兼顾 coding"；纯 coding 模型不要超过 25%。
- 小语种即使做 temperature 0.3 还是会过拟合；<1% 占比的语言考虑直接合成数据补齐。

---

## 5. FIM（Fill-in-the-Middle）深入

Bavarian et al. 2022 的关键洞察：FIM 训练不伤 left-to-right 能力，前提是比例 ≤ 50% 且格式设计正确。

**PSM 格式**：
```
<fim_prefix>def fibonacci(n):
    if n < 2:
        return n
    return <fim_suffix>

print(fibonacci(10))<fim_middle>fibonacci(n-1) + fibonacci(n-2)
<|endoftext|>
```
训练时所有 token 都参与 loss；推理时给到 `<fim_middle>` 前的部分，模型续写 middle。

**SPM 格式**（suffix 先出）：
```
<fim_suffix>

print(fibonacci(10))<fim_prefix>def fibonacci(n):
    if n < 2:
        return n
    return <fim_middle>fibonacci(n-1) + fibonacci(n-2)
<|endoftext|>
```

**为什么要两种都训**：
- PSM 更自然，便于 IDE 补全场景。
- SPM 在 KV cache 复用上更友好（prefix 末尾可复用），对长上下文 IDE 场景性能更好。
- 两种 1:1 混合对下游都友好，额外代价几乎为 0。

**span 采样的工程实现**：
```python
import numpy as np

def fim_transform(text: str, rng: np.random.Generator,
                  psm_prob=0.5, fim_rate=0.5) -> str:
    if rng.random() > fim_rate:
        return text  # 保持原 LTR 样本
    n = len(text)
    if n < 10:
        return text
    # 字符级随机两刀
    i, j = sorted(rng.integers(0, n, size=2).tolist())
    prefix, middle, suffix = text[:i], text[i:j], text[j:]
    if rng.random() < psm_prob:
        return f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>{middle}"
    else:
        return f"<fim_suffix>{suffix}<fim_prefix>{prefix}<fim_middle>{middle}"
```

- 务必在 packing **之前** 做 FIM 改写，不要在 packing 之后切，会跨文件。
- FIM special token 参与 tokenizer 训练（作为 reserved special tokens），不被 BPE 拆开。

---

## 6. Repo-level Packing 实现思路

核心目标：一个训练样本内的文件要"有语义联系"，而非随机拼接。

### 6.1 数据结构

对每个仓库预先构建 manifest：

```json
{
  "repo": "owner/repo",
  "stars": 1234,
  "license": "MIT",
  "language_mix": {"Python": 0.8, "Markdown": 0.15, "YAML": 0.05},
  "files": [
    {"path": "src/core/utils.py", "size": 1423, "sha": "...", "tokens": 380},
    ...
  ],
  "import_graph": {"src/main.py": ["src/core/utils.py", "src/core/model.py"], ...}
}
```

### 6.2 构建 import 图

**Python（最简版，足够多数场景）**：

```python
import ast, pathlib

def parse_imports(path: pathlib.Path, repo_root: pathlib.Path):
    try:
        tree = ast.parse(path.read_text(encoding='utf-8', errors='ignore'))
    except SyntaxError:
        return []
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
        elif isinstance(node, ast.Import):
            for n in node.names:
                imports.append(n.name)
    # 把 dotted module 名映射到 repo 内文件
    resolved = []
    for mod in imports:
        cand = repo_root / (mod.replace('.', '/') + '.py')
        if cand.exists():
            resolved.append(str(cand.relative_to(repo_root)))
        cand2 = repo_root / (mod.replace('.', '/')) / '__init__.py'
        if cand2.exists():
            resolved.append(str(cand2.relative_to(repo_root)))
    return resolved
```

### 6.3 拓扑排序 + 环处理

```python
from graphlib import TopologicalSorter, CycleError

def order_files(import_graph: dict[str, list[str]]) -> list[str]:
    ts = TopologicalSorter(import_graph)
    try:
        return list(ts.static_order())
    except CycleError:
        # 退化：按路径字典序
        return sorted(import_graph.keys())
```

### 6.4 Pack 成训练样本

```python
def pack_repo(files_in_order, tokenize, max_len=8192):
    buf, buf_len = [], 0
    samples = []
    for f in files_in_order:
        header = f"<filename>{f['path']}\n"
        ids = tokenize(header + f['text']) + [EOS]
        if buf_len + len(ids) > max_len and buf:
            samples.append(buf[:max_len])
            buf, buf_len = [], 0
        buf.extend(ids); buf_len += len(ids)
    if buf:
        samples.append(buf[:max_len])
    return samples
```

- **跨样本连续性**：同一仓库跨多个样本时，后续样本可在头部加 `<reponame>...` 作为 context hint。
- **窗口混配**：80% 样本 8k，20% 样本 32k+，保证长上下文训练信号。
- **按 star 加权采样**：高 star 仓库上采样 2–3 倍（log scale），但别过度，小仓库也有价值。

---

## 7. 可运行的 Mini Pipeline：用 datatrove 处理 10GB Python 子集

前置：`pip install datatrove[all] datasketch fasttext`（Python 3.10+）。

这个脚本串起 reader → 语言过滤 → 启发式 → PII → MinHash 去重 → writer，跑在单机多进程上，10GB 数据 32 核机器大概 30–60 分钟。源文件在 `examples/phase1/run_pipeline.py`，build 时由 include 指令注入：

<!-- include: examples/phase1/run_pipeline.py -->


**跑起来后该 sanity-check 的事**：
1. `stage1` 的过滤率应在 40–60%（太低说明过滤太松，太高说明误伤）。
2. `stage2_dedup` 的过滤率应在 20–40%（The Stack v2 已经做过一轮 dedup，所以少）；对 raw GitHub 数据会到 50%+。
3. 抽 100 条 `final` 人工看一眼，有没有混入大型 JSON / base64 / autogen 文件。
4. grep HumanEval 的函数名（如 `truncate_number`、`separate_paren_groups`），应该为 0 命中。

---

## 8. 常见坑（踩过才懂）

1. **私钥泄漏**：光做正则 mask 不够，**整个包含私钥文件的仓库都要放黑名单**，因为往往一个仓库多个 key、commit history 里还有。出问题时是责任问题。

2. **License 陷阱**：
   - 头部 SPDX 注释可能和 LICENSE 文件冲突，以 LICENSE 为准。
   - `package.json` 里写 `"license": "MIT"` 但根目录无 LICENSE 文件 → 法律上争议，保守剔。
   - `dual-licensed` 比如 `(MIT OR Apache-2.0)`，能用。
   - `SEE LICENSE IN xxx` 需要递归解析，常见但容易漏。
   - GPL / AGPL 的"感染性"延伸到同仓库所有文件，哪怕单文件头部写了 MIT，仓库整体 GPL 时保守剔。

3. **HTML 混入代码**：`.html` 文件里 `<script>` / `<style>` 块经常是大块 JS/CSS。不处理会把 JS/CSS tokenizer 压缩率拉坏。实现：用 `selectolax` / `lxml` 切出 script 块，按 JS 单独处理，剩下的 HTML 走 docs bucket。

4. **Autogenerated 文件**：
   - `*.pb.go` / `*_pb2.py` / `*_generated.go` / `openapi_generated/**` 是 protobuf / codegen 产物，训练用反而会让模型学一堆格式化噪声。
   - Migration 文件（Rails `db/migrate/*.rb`、Django `*/migrations/0001_*.py`）有特定时间戳前缀，可 regex 匹配。
   - Python 的 `__pycache__`、`.pyc` 要过滤（有些仓库误提交）。
   - 经验：启发式 + 文件名 regex + `@generated` 标记三管齐下，漏网率 < 3%。

5. **minified / bundled 文件**：`*.min.js`、`*.bundle.js`、`jquery*.js`、`webpack-*.js`，最长行长度 + 文件名双重判断。minified CSS/JS 往往单行数万字符，一个文件能喂出一个样本，严重 bias 分布。

6. **The Stack v2 的 SWHID resolve**：HF 上的 v2 数据集需要你用 SoftwareHeritage 的 API 把 SWHID → blob，**下载速率限制严重**（默认每 IP ~100/s）。要么用 `bigcode/the-stack-v2-dedup` 预解析版本，要么申请 SWH token。

7. **Unicode 规范化**：NFC / NFKC 要统一，否则标识符 "é" 和 "é"（组合字符）被当成不同 token。tokenizer 训练前统一 NFC。

8. **不同语言的注释 / docstring**：Python docstring 是 string literal，不是注释；中文注释占比高的仓库 tokenizer 如果没覆盖中文字符会变成一堆 `<0xXX>` byte fallback，压缩率烂。训 tokenizer 的 sample 要混 10% 中文代码。

9. **评测集的"改写"污染**：LiveCodeBench 样本被 StackOverflow 讨论过，StackOverflow 进训练 → 泄漏。decontam 要覆盖 **代码语料 + 所有 NL 语料**，不只是代码。

10. **Jupyter `.ipynb` 陷阱**：
    - `outputs` 里常含 base64 PNG、HTML 表格（几 MB 单 cell）。
    - `execution_count`、`id` 等元信息没训练价值。
    - 用 `nbconvert --to script` 只保留 code cell 太粗暴，会失去 text/code 交错结构；用 `bigcode-dataset/preprocessing/notebook` 的做法：保留 markdown + code，丢 output（或只留 stdout 的纯文本）。

11. **Dataset 偏置**：
    - 高 star 仓库被开发者"镜像"、fork 成千上万，MinHash 后仍可能因细节差异留多份。做 **per-repo canonical** 保留一份。
    - 某些框架生态（React、Spring）文件模板化严重，boilerplate 被过采样 → 模型学到生成样板代码的倾向。用 **n-gram entropy** 对每个 repo 打分，太低的降权。

12. **时间泄漏**：LiveCodeBench / SWE-bench Verified 是 "date-sensitive" 基准。训练数据切一个 cutoff（比如 2024-06），晚于 cutoff 的仓库 commit 不进训练，用于 held-out 验证。

13. **并发与存储**：pipeline 会产生大量小文件 + 中间态，建议用 S3 / GCS 对象存储 + parquet，本地盘只做 tokenize 后的 mmap。IO 峰值能到 5–10GB/s，SSD 优先。

14. **Reproducibility**：每个 stage 写 manifest（输入 sha、参数 hash、代码 commit、输出 path），事后能追溯"这一版模型是哪个配方训的"。这一点 OpenCoder 做得最好，强烈建议参考。

---

## 9. 推荐的动手顺序（Phase 1 内部）

把 8–12 周的 Phase 1 拆成：

- **Week 1–2**：跑通 datatrove 单机版本在 10GB 子集上，确认 reader / heuristic / writer 链路闭环。输出：parquet shards + 过滤率报告。
- **Week 3–4**：接入 MinHash-LSH（`bigcode-dataset/near_deduplication` 的 PySpark 实现），在 100GB 规模验证。评估指标：near-dup 对数、保留率、抽样人工验证。
- **Week 5**：decontamination 跑通 + 验证（**最关键，单独一周**）。
- **Week 6–7**：质量模型训练（fastText + small-LM PPL），跑在 1T token 规模。
- **Week 8–9**：repo-level packing、FIM、tokenizer 训练、mmap 输出。
- **Week 10–11**：全量配比采样 + 预训练数据 snapshot v1，冻结。
- **Week 12**：跑一个 1.5B 模型在 50B tokens 上做 ablation，确认 loss 曲线、HumanEval 基线正常（dense 1.5B 在干净 50B tokens 上 HumanEval pass@1 能到 ~20%）。

---

## 10. 关键参考与工具链清单

**论文**：
- StarCoder2 & The Stack v2: arxiv 2402.19173
- OpenCoder: arxiv 2411.04905（**最值得逐页精读**）
- DeepSeek-Coder: arxiv 2401.14196（V1，repo-level packing 细节）
- DeepSeek-Coder-V2: arxiv 2406.11931
- GLM-4.5 / GLM-5.1 ARC: arxiv 2508.06471
- FIM: arxiv 2207.14255
- RefineCode（OpenCoder 数据篇）：随 2411.04905 一起
- SlimPajama / Dolma 去重经验：arxiv 2402.00159 (Dolma)

**代码库**：
- `bigcode-project/bigcode-dataset`：preprocessing / near_deduplication / pii / decontamination 四大子目录
- `huggingface/datatrove`：执行框架
- `ChenghaoMou/text-dedup`：MinHash / SimHash / Exact / Suffix array，功能最全
- `huggingface/tokenizers`、`google/sentencepiece`
- `go-enry/go-enry`：语言识别
- `nvidia/Megatron-LM`：`tools/preprocess_data.py` 输出 mmap

**HF 数据集**：
- `bigcode/the-stack-v2` / `bigcode/the-stack-v2-dedup`
- `bigcode/commitpack` / `bigcode/commitpackft`
- `bigcode/the-stack-github-issues`
- `bigcode/the-stack-jupyter`
- `open-web-math/open-web-math`
- `EleutherAI/proof-pile-2`

**评测**：
- `openai/human-eval`、`google-research/mbpp`
- `LiveCodeBench/LiveCodeBench`
- `bigcode-project/bigcodebench`
- `princeton-nlp/SWE-bench` / `swe-bench/SWE-bench_Verified`

---

## 附录 A：MinHash-LSH 参数速查

给定 `b` 个 band、`r` 行每 band（总 hash 数 `n = b*r`），LSH 的相似度 ≥ s 的候选命中概率：
$$
P(\text{candidate} | s) = 1 - (1 - s^r)^b
$$
常用配置：

| n | b | r | 阈值 s* ≈ (1/b)^(1/r) | 适用 |
|---|---|---|---|---|
| 128 | 16 | 8 | 0.70 | 代码，偏严 |
| 128 | 25 | 5 | 0.52 | 代码，偏松（StarCoder v1） |
| 256 | 50 | 5 | 0.55 | 代码，主流（StarCoder2） |
| 256 | 20 | 13 | 0.84 | 严格精排 |

实战经验：一版 coarse（低阈值、宽召回） + 一版 fine（高阈值、精确）两遍跑，先 coarse 砍掉大量近似重复，再 fine 内部精排保留最优副本。

## 附录 B：启发式规则的默认阈值表

| 维度 | 默认阈值 | 备注 |
|---|---|---|
| file size | 1KB – 1MB | <1KB 过短，>1MB 多半是数据 |
| 行数 | 3 – 100k | - |
| avg line len | ≤ 100 | 低于 5 警惕（可能只有空行） |
| max line len | ≤ 1000 | 检测 minified |
| alpha ratio | ≥ 0.15 | 字母字符比例 |
| alphanum ratio | ≥ 0.25 | - |
| digit ratio | ≤ 0.60 | 过滤 number dump |
| hex ratio | ≤ 0.50 | 过滤 binary dump |
| 最长 word | ≤ 1000 chars | - |
| `#`/`//` 注释行比 | ≤ 0.80 | 过滤大量 license header-only |
| 唯一行比例 | ≥ 0.10 | 重复行多 → log 文件 |

## 附录 C：快速健康检查清单（每次 snapshot 后跑）

- [ ] 每语言 token 数柱状图（log scale），检查是否某语言突然消失或暴涨。
- [ ] 抽样 100 条人工读，记录"一眼假"样本数；> 5 条就要回去查 filter。
- [ ] HumanEval / MBPP 全量 grep，必须 0 命中。
- [ ] 私钥正则 grep，必须 0 命中。
- [ ] License 字段分布，non-permissive 必须 0%。
- [ ] tokenizer 压缩率（bytes/token）按语言列出，和基线比。
- [ ] 序列长度分布，8k packing 下应 > 95% 样本满 8k。
- [ ] 重复度抽样：对 10k 样本两两 MinHash，Jaccard > 0.7 的对数应 < 0.01%。

以上八项过不了，不要开始训练。

---

## 📌 章末检查

**带走这 5 条**
- The Stack v2 / OpenCoder 是公开基线；任何自训语料都要先和这两份比统计量。
- MinHash 近似去重 + 10-gram exact match 双保险——前者抓段落级重复、后者抓评测集污染。
- 启发式过滤阈值**必须按语言分桶**调（Python/Java/Rust 的"短文件"分布差 3×）。
- loss mask 必须和 tokenizer chat template 字节级对齐，不能"目测正确"。
- Issue-PR 是金矿，四形态 (a) Issue→test (b) Issue→diff (c) Diff→title (d) review 多轮，按过滤难度递减。

**自检 3 题**（< 5 分钟）
1. 为什么 MinHash 去重前一定要做 **unicode 归一化**？
2. 评测集去污染的 10-gram exact match 为什么不能省、不能只靠 MinHash？
3. PR 四形态里，哪种最容易快速产大批量样本？哪种最难但对 multi-turn agent 最关键？

<details><summary>参考答案</summary>

1. 否则 NFC/NFD、半角全角、CRLF/LF、不同缩进会让"事实相同"的文本 hash 出不同 shingle，中文/日文/Markdown 文本去重率会虚高。
2. MinHash 默认阈值（Jaccard ≈ 0.7）会漏短文本污染——HumanEval/MBPP 题面只有 20-50 token，hash 噪声盖过信号；必须 10-gram exact match 兜底。
3. (c) Diff→title 最容易（PR title 已是干净一句话，过滤宽，规模可达千万级）；(d) review 多轮最难（要重建对话顺序 + 链式 loss mask），但对训练"模型能吸取人类反馈"这种 multi-turn agent 能力最关键。
</details>

> ⚠️ **常见坑** · 不做 license filter，把 The Stack v2 里的 GPL/AGPL 代码混进训练集。下游用户可能调出一段你**不能合法分发**的代码——这是 2024 GitHub Copilot 集体诉讼里的核心证据，2026 至今仍未结案。

**下一步** → 进入 [phase2 预训练架构](./phase2_pretraining.md) 看数据怎么进 MoE。术语速查 → [▣ 索引](./phase_glossary.md)。

---

## 动手练习

1. 在 HuggingFace 上打开 `bigcode/the-stack-v2-dedup` 数据卡，浏览 statistics 页，回答：The Stack v2 在 Python 上有多少 token？许可证分布如何？相比 v1 增量在哪里？
   *提示*：直接看数据卡 + arxiv 2402.19173，不需要下载数据。
2. 用 `datatrove` 跑一个 1GB GitHub Python 子集（从 `bigcode/the-stack-smol` 取）的最小 pipeline：language filter → MinHash 近似去重 → 启发式过滤 → tokenize 出 `.bin`。要求最终样本数 ≤ 输入的 60%，并报告每一步的过滤率。
   *提示*：参考附录 B 的阈值表 + datatrove 官方 examples。
3. 实现一个**评测集去污染**脚本：输入 HumanEval+ / MBPP+ 题面，对 §C 健康检查清单第 3 条做严格校验——10-gram exact match 全量扫描 1GB tokenized 语料，报告命中数。如果命中 > 0，说明你的数据被污染了。
   *提示*：用 `datasketch` 的 MinHash + 直接字符串 grep 双保险。OpenCoder 论文有参考实现。
4. 复现 §6 (Issue-PR 金矿) 的 (c) 形态：从一个真实 GitHub repo（≥ 1k star）抓 100 个 merged PR，用 `gh api` + 自写脚本生成 `(diff, pr_title)` 单轮 SFT 数据，过滤掉 `quality_ok = False` 的样本，最后产出 jsonl 给 Phase 4 用。
   *提示*：参考 §10.9 抽取脚本，注意时间加权和 §10.10 三个坑。
5. 端到端复刻 OpenCoder 数据 pipeline 的 1/100 缩小版：输入是 100GB 多语言 GitHub raw → 输出是 ~10B token 可训语料 + 完整数据卡（语言分布 / 重复度直方图 / 污染检查报告 / tokenizer 压缩率分语言）。整个过程有完整的 reproducible 脚本和 Makefile。
   *提示*：OpenCoder 把所有脚本 (`OpenCoder-LLM/OpenCoder-DataPipeline`) 都开源了，照搬骨架但缩小数据规模到家用机/小集群可承受。这是本章最重要的 capstone 项目。
