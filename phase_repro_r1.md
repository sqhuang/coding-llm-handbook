# 🧬 复现 case · 在 4090 上做一个最小版的 R1-style reasoning trace

> 📅 主线快照：2026-05-28 · 上次核对：2026-05-28

> **⚡ 三句话要点**
> 1. DeepSeek-R1 的核心证据是「**通过 RLVR 让 small base model 涌现 `<think>...</think>` 长 reasoning trace**」——本章给一份在 **1×RTX 4090 24GB** 上 12-18 小时跑得通的最小复现配方。
> 2. 不追 R1-Distill 的最终分数，目标是看到一个可观测的「**aha moment**」——reward 曲线从随机平台拐头上升 + completion length 自动从 50 token 涨到 300+。
> 3. 走完后你拥有：一份能解 GSM8K 子集的 ckpt + 一份 RL 训练曲线截图 + 对 GRPO 实战参数的直觉。**不是"复现 R1"，是"复现 R1 的关键现象"**。

> ⚠️ **诚实声明** · 本章是 minimum viable starter，**不是 "在 4090 上能复刻 R1-Distill-7B" 的承诺**。R1 真正的能力来自 671B 主模型 + DeepSeek-V3 base + 海量数据 cold start。本配方目标是让你**亲眼看到**那条 reward + length 同步上升的曲线，建立 "RL 涌现 reasoning 是真的" 这个 conviction。

---

## 1. 配方一览

| 字段 | 值 |
|---|---|
| 主硬件 | 1 × RTX 4090 24GB (5090 / 2×4090 更舒服) |
| Base 模型 | `Qwen/Qwen2.5-1.5B-Instruct` （或 1.5B-Math-Instruct，math reasoning 更强基线） |
| 数据 | GSM8K train split 1k 题（共 7473 题，先取 1000 看现象） |
| 框架 | TRL 0.12+ 的 `GRPOTrainer` + 内嵌 vLLM rollout |
| 算法 | GRPO · group_size=4 · KL beta=0.04 · clip=0.2 |
| Reward | 答案匹配（正则提取 `\\boxed{N}` 并 string == ground_truth） + 格式奖励（`<think>` 标签存在 +0.1） |
| 训练步数 | 200-300 step（约 12-18h on 4090） |
| 预期信号 | reward 均值 0.15 → 0.55；completion length 60 → 250+ |
| 预算 | 12-18 GPU-hour · 电费 ≈ ¥5 · 不需要云租 |

---

## 2. 7 步动手

### 步 1 · 环境

```bash
pip install "trl>=0.12" "vllm>=0.6" "transformers>=4.46" peft datasets accelerate
pip install math-verify  # 评估 boxed 答案的库
```

### 步 2 · 数据 prep

```python
# data/gsm8k.py
from datasets import load_dataset

SYSTEM = (
    "Solve the problem step by step inside <think>...</think>, then output "
    "the final numerical answer inside \\boxed{}."
)

def to_chat(ex):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": ex["question"]},
        ],
        "answer": ex["answer"].split("####")[-1].strip(),
    }

ds = load_dataset("gsm8k", "main", split="train").select(range(1000))
ds = ds.map(to_chat, remove_columns=ds.column_names)
ds.save_to_disk("./data/gsm8k_1k")
```

### 步 3 · Reward functions

```python
# rewards.py
import re
from math_verify import parse, verify

BOXED = re.compile(r"\\boxed\{([^}]+)\}")
THINK = re.compile(r"<think>.*?</think>", re.S)

def reward_correctness(completions, answer, **kw):
    rewards = []
    for comp, gt in zip(completions, answer):
        text = comp[0]["content"] if isinstance(comp, list) else comp
        m = BOXED.search(text)
        if not m:
            rewards.append(0.0); continue
        try:
            ok = verify(parse(m.group(1)), parse(gt))
            rewards.append(1.0 if ok else 0.0)
        except Exception:
            rewards.append(0.0)
    return rewards

def reward_format(completions, **kw):
    rewards = []
    for comp in completions:
        text = comp[0]["content"] if isinstance(comp, list) else comp
        rewards.append(0.1 if THINK.search(text) and BOXED.search(text) else 0.0)
    return rewards
```

### 步 4 · GRPO 训练脚本

```python
# train.py
from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from rewards import reward_correctness, reward_format

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
tok = AutoTokenizer.from_pretrained(MODEL)

ds = load_from_disk("./data/gsm8k_1k")
ds = ds.map(lambda ex: {"prompt": tok.apply_chat_template(ex["prompt"], tokenize=False, add_generation_prompt=True),
                        "answer": ex["answer"]})

cfg = GRPOConfig(
    output_dir="./ckpt_r1mini",
    learning_rate=5e-6,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_generations=4,             # 单卡只能给 4，再大会爆
    max_prompt_length=512,
    max_completion_length=1024,
    num_train_epochs=1,
    beta=0.04,
    temperature=0.9,
    use_vllm=True,
    vllm_gpu_memory_utilization=0.35,  # 给 trainer 留 65%
    logging_steps=1,
    save_steps=50,
    bf16=True,
    report_to="wandb",  # 强烈建议看 wandb 曲线
)

lora = LoraConfig(r=8, lora_alpha=16, target_modules="all-linear",
                  task_type="CAUSAL_LM")

trainer = GRPOTrainer(
    model=MODEL,
    reward_funcs=[reward_correctness, reward_format],
    args=cfg,
    train_dataset=ds,
    peft_config=lora,
)
trainer.train()
```

### 步 5 · 启动 + 监控

```bash
WANDB_PROJECT=r1mini python train.py
```

**前 30 step 看什么**：
- `reward/correctness` 均值应该在 0.10-0.20（初始命中率）
- `reward/format` 应该快速上升到 0.10（模型很快学会包 `<think>`）
- `kl` < 1.0（如果 > 5 立刻降 beta 重启）
- `completion_length` 60-90（base 模型短回答习惯）

### 步 6 · 关键观察点（第 80-150 step）

如果配方对，你会看到这个序列：
1. 第 40-60 step：`reward/format` 稳定 0.10（格式学会了）
2. 第 80-100 step：`reward/correctness` 开始拐头上升，从 0.20 → 0.30
3. **第 100-150 step：completion_length 突然涨**，从 90 → 200+，这就是「aha moment」——模型发现"想得更长 → 答得更对"
4. 第 150-200 step：reward 持续上升到 0.45-0.55，length 稳定在 250-400

> ⚠️ 如果第 100 step `reward/correctness` 还在 0.20 不动 → 大概率是 `temperature` 太低（< 0.7）或 reward 正则没匹配上。先 `python -c "from rewards import *; print(reward_correctness([\"\\\\boxed{42}\"], [\"42\"]))"` 验证 reward 函数本身能输出 1.0。

### 步 7 · 评测产出

```python
# eval.py
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="bfloat16", device_map="auto")
model = PeftModel.from_pretrained(base, "./ckpt_r1mini/checkpoint-200")
tok = AutoTokenizer.from_pretrained(MODEL)

test = load_dataset("gsm8k", "main", split="test").select(range(100))
correct = 0
for ex in test:
    prompt = tok.apply_chat_template([
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": ex["question"]}
    ], tokenize=False, add_generation_prompt=True)
    out = model.generate(**tok(prompt, return_tensors="pt").to(model.device),
                         max_new_tokens=512, temperature=0.0)
    text = tok.decode(out[0])
    # ... extract \boxed{} and check
    pass
print(f"GSM8K test (100): {correct}/100")
```

**预期成绩**：
- base Qwen2.5-1.5B-Instruct: GSM8K 100 题 ≈ 55-60 通过
- 训完最小版：≈ 70-75 通过（**+ 15pp** 是合理目标，不要追 +30）

---

## 3. 与真正的 R1 之间还差什么

| 维度 | R1 / R1-Zero | 你这个 minimum viable |
|---|---|---|
| Base | DeepSeek-V3 671B | Qwen2.5-1.5B |
| 数据 | 数十万真实推理 prompt + 多领域 | 1k GSM8K |
| Cold start SFT | 数千条 long reasoning chain demo | 跳过（reward 直接拉） |
| RL 步数 | ~万 step | 200 step |
| 算力 | 千卡级 | 1×4090 |
| 涌现强度 | 100k+ token reasoning + 跨域泛化 | 250 token reasoning + 仅 GSM8K |

**学到什么**：
- 第一次亲眼看到 `completion_length` 曲线被 reward 牵着上升——R1 现象**确实是真的**，不是论文营销。
- GRPO 在小模型 + 简单 reward 上确实能跑，不需要 critic / RM。
- KL beta=0.04 / clip=0.2 / temperature=0.9 这套**默认参数能跑**，没必要无脑调。

**学不到的**：
- 千卡级 rollout 的工程挑战（VERL / slime 的 disaggregate 架构）
- 多领域泛化（GSM8K → MATH → AIME → code 的 transfer）
- Long context reasoning（你只跑了 1024 max_completion）

---

## 4. 故障兜底

按 [📓 phase_failures §E](./phase_failures.md) 排查；最常见的本配方专属坑：

| 症状 | 大概率原因 | 修法 |
|---|---|---|
| 200 step reward 还在 0.20 不动 | temperature 太低 / reward 正则没匹配 / num_generations=1（应 ≥ 4） | 升 temperature 0.9, num_generations=4, `print(reward_fn([sample]))` 手测 |
| OOM @ step 5 | vllm + trainer 同卡 = 显存吃满 | `vllm_gpu_memory_utilization=0.30`, num_generations=2 |
| KL > 10 | lr 太大 / beta 太小 | lr 砍半到 2.5e-6 / beta 升到 0.08 |
| completion_length 一直 < 100 | format reward 权重太低 → 模型没动力包 `<think>` | format reward 升到 0.2 / 在 system prompt 里强调 |
| `math_verify` import 失败 | python 版本不对（要 ≥3.10） | conda env Python 3.10+ |

---

## 5. 升级路径

跑通这个 minimum viable 之后想做更"真"的复现：

| 想做 | 推荐路径 |
|---|---|
| 跨域：不只 GSM8K | 混 MATH + APPS Competition Math，1:1:1 比例 |
| 真 cold start | 跑 phase4 §10 抽 1k 高质量 reasoning trace SFT 后再 RL |
| 更长 reasoning | `max_completion_length=4096` + 2×4090（48GB） |
| 更大 base | 7B Qwen → 2×4090 + QLoRA 或租 1×H100 24h |
| 多机 RL | 不要在消费卡做，直接 ✪ [phase_capstone step 12](./phase_capstone.md) |

---

## 📌 章末检查

**带走这 3 条**
- R1 现象不需要 671B 才能复现——**最小版本就能看到 reward 与 length 同步上升**，1×4090 12h 见证奇迹。
- GRPO 的稳定性主要来自 (a) group_size ≥ 4 (b) KL beta 0.04 (c) temperature 0.9，参数有共识。
- "复现 R1" 和 "复现 R1 关键现象" 是两件事；后者每个独立研究者都该做一次，前者是公司项目。

**自检 3 题**
1. 为什么 reward 函数里要分 correctness + format 两项？合并成一项行不行？
2. `num_generations=4` 是为了什么？降到 1 会怎样？
3. 你的 completion_length 在 200 step 还是 60 没动，但 reward 涨了 20pp——可能性最大的解释？

<details><summary>参考答案</summary>

1. **不行**。format 是 dense 信号（每步都能给），correctness 是 sparse 信号（只在答案正确时给 1.0）。如果只有 sparse，初始 100 step reward 全 0 → advantage 全 0 → 不更新。format 把"先学会包 think 标签"这个子目标显式给了 reward，把 dense 桥搭起来。
2. group_size = 用同一 prompt 采 G 个回答算组内 advantage。G=1 时 advantage 永远为 0（自己减自己），GRPO 无法工作。**G=4 是 GRPO 的下限**，G=8 是默认值，你这受显存限制只能给 4。
3. 模型可能学到了 "更短但更对" 的捷径——比如直接背了 GSM8K 一些数字，不需要长推理。可以查训练数据是否泄漏；也可能 reward 函数被 hack（`\\boxed{42}` 匹配上但答案是 42 的题占很大比例）。建议 holdout 集（test split）跑一下看分数有没有真涨。
</details>

> ⚠️ **常见坑** · 看到 wandb 上 reward 涨就以为成功——务必跑一遍 holdout test split（步 7）。**训练集 reward 涨 + holdout 不动 = 过拟合到 reward 函数**，不是 reasoning 涌现。

**下一步** · 想真正训"会推理的"模型 → 看 [phase5 §RL 完整章节](./phase5_rl.md) + [📓 phase_failures §E RL 排查](./phase_failures.md) · 公式速查 → [🧮 phase_math §C](./phase_math.md)。
