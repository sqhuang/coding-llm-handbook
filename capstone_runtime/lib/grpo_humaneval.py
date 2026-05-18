"""
Minimal TRL GRPO recipe on HumanEval. Phase 5 §7.3 抽取脚本。

输入：Phase 4 LoRA SFT 合并后的模型 + HumanEval test split。
Reward = 单测通过率（sparse, 1.0/0.0）+ format bonus（dense, 0.1）。

Run:
    pip install "trl>=0.12" "vllm>=0.6" "transformers>=4.46" peft datasets
    python grpo_humaneval.py
"""
import os
import re
import subprocess
import tempfile

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

MODEL = "path/to/phase4_sft_lora_merged"   # Phase 4 LoRA 合并后的模型
tok = AutoTokenizer.from_pretrained(MODEL)

# 1) 数据：每条样本 = {"prompt": <提示>, "tests": <单测代码>}
ds = load_dataset("openai_humaneval", split="test")  # 教学用；实际用训练集

SYSTEM = (
    "You are a coding assistant. First think inside <think>...</think>, "
    "then output the final Python solution inside ```python ... ``` block."
)


def build_prompt(ex):
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": ex["prompt"]}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


ds = ds.map(lambda ex: {"prompt": build_prompt(ex), "tests": ex["test"],
                        "entry_point": ex["entry_point"]})

CODE_RE = re.compile(r"```python\n(.*?)```", re.S)
FORMAT_RE = re.compile(r"<think>.*?</think>", re.S)


def run_tests(code: str, tests: str, entry_point: str, timeout: int = 10) -> bool:
    prog = code + "\n\n" + tests + f"\n\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(prog)
        path = f.name
    try:
        r = subprocess.run(["python", path], capture_output=True,
                           timeout=timeout, text=True)
        return r.returncode == 0
    except Exception:
        return False
    finally:
        os.unlink(path)


# 2) Reward functions：TRL 允许多个 reward fn，结果会相加
def reward_correctness(completions, tests, entry_point, **kw):
    rewards = []
    for comp, t, ep in zip(completions, tests, entry_point):
        text = comp[0]["content"] if isinstance(comp, list) else comp
        m = CODE_RE.search(text)
        if not m:
            rewards.append(0.0)
            continue
        ok = run_tests(m.group(1), t, ep)
        rewards.append(1.0 if ok else 0.0)
    return rewards


def reward_format(completions, **kw):
    rewards = []
    for comp in completions:
        text = comp[0]["content"] if isinstance(comp, list) else comp
        ok = bool(FORMAT_RE.search(text)) and bool(CODE_RE.search(text))
        rewards.append(0.1 if ok else 0.0)
    return rewards


# 3) 训练配置
cfg = GRPOConfig(
    output_dir="./grpo_out",
    learning_rate=5e-6,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,          # 有效 batch 16 prompts / GPU
    num_generations=8,                      # G = 8
    max_prompt_length=1024,
    max_completion_length=2048,
    num_train_epochs=1,
    beta=0.01,                              # KL coef
    temperature=1.0,
    use_vllm=True,                          # 打开 vLLM 加速 rollout
    vllm_gpu_memory_utilization=0.5,
    logging_steps=1,
    save_steps=100,
    bf16=True,
    report_to="wandb",
)

lora = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear",
                  task_type="CAUSAL_LM")

trainer = GRPOTrainer(
    model=MODEL,
    reward_funcs=[reward_correctness, reward_format],
    args=cfg,
    train_dataset=ds,
    peft_config=lora,
)
trainer.train()
