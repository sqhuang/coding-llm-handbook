# 实验册 · Lab Notebook

> **定位**：本册是主线 9 个 Phase 的"动手附录"。每个实验都是一个 **A vs B** 的微观对照——不教你训一个 754B 的 GLM-5.1，而是用 **1×3090 / Colab T4 / 甚至纯 CPU**，在 **30 分钟到 2 小时**内复现一个具体命题（例如"GQA 真的能压缩 KV cache 吗""RoPE base 改大 50× 真的能外推吗"）。
>
> **风格**：先给假设（你应当看到 X），再给可拷贝的命令，最后给核对清单。代码尽量短（10-25 行），依赖只用 `transformers / datasets / numpy / torch`，偶尔加一个 `vllm / peft / trl` 这种轻量库。
>
> **怎么用**：
> 1. 读完一个 Phase 后，挑 1-2 个对应实验跑一遍，让"参数量""throughput""KV cache MB"这些数字从抽象变具体。
> 2. **先读"你应当看到的对比"再开跑**——带预期跑实验比盲跑收获大 3 倍。
> 3. 跑完用底部的"✓/✗ 清单"自我核对。
>
> **硬件假设**：默认 1×RTX 3090 (24GB) 或 Colab T4 (16GB)。CPU-only 的实验会单独标注。所有大模型推理都用 ≤ 3B 的代理模型（Qwen2.5-0.5B / LLaMA-3.2-1B / GLM-4-Flash 之类），不依赖 GLM-5.1 本体。

---

## 实验全清单（一句话目录）

| # | 实验 | 比的是 | 耗时 | 关联 Phase |
|---|---|---|---|---|
| 1.1 | 三家 BPE 同句对比 | token 数 / 词表大小 | 15 min | P1 |
| 1.2 | tiktoken vs sentencepiece 速度 | tokens/s | 20 min | P1 |
| 1.3 | embedding lookup vs 计算 | 显存 / 延迟 | 20 min | P0 |
| 2.1 | MHA / GQA / MLA KV cache | MB / token | 30 min | P2 |
| 2.2 | Dense vs MoE 同等效激活的 FLOPs | GFLOPs | 30 min | P2 |
| 2.3 | RoPE base = 1e4 vs 5e5 外推 | PPL @ 8K-32K | 45 min | P3 |
| 2.4 | SwiGLU vs ReLU 小数据 loss | 200 step loss | 60 min | P2 |
| 2.5 | LayerNorm vs RMSNorm 速度 | μs/token | 15 min | P2 |
| 3.1 | MinHash-LSH 去重前后 | 文件数 / 耗时 | 45 min | P1 |
| 3.2 | HumanEval n-gram 命中 The Stack | 命中率 | 30 min | P1, P6 |
| 3.3 | FIM 改写前后 prompt 形态 | 视觉对比 | 15 min | P1 |
| 3.4 | packing vs padding throughput | tokens/s | 30 min | P2 |
| 4.1 | Base vs SFT 同指令输出 | 直观对比 | 20 min | P4 |
| 4.2 | LoRA r=8/16/64 loss 曲线 | 收敛速度 | 90 min | P4 |
| 4.3 | Chat template 三家 token 化 | 视觉 + token 数 | 15 min | P4 |
| 4.4 | PPO vs GRPO 200 step reward | reward 曲线 | 90 min | P5 |
| 4.5 | KL coef 0.001/0.01/0.1 多样性 | distinct-n | 60 min | P5 |
| 5.1 | HumanEval vs HumanEval+ | pass@1 差 | 30 min | P6 |
| 5.2 | LiveCodeBench 按月分段 | 污染 gap | 45 min | P6 |
| 5.3 | greedy vs temp=0.7 | pass@1 / pass@10 | 30 min | P6 |
| 6.1 | BF16 vs FP8 vs AWQ | 延迟 / 显存 | 60 min | P7 |
| 6.2 | vLLM vs SGLang | TTFT / throughput | 60 min | P7 |
| 6.3 | KV cache 命中 vs miss | TTFT | 30 min | P7 |
| 6.4 | chunked prefill 开关 | ITL | 30 min | P7 |
| 6.5 | TP=1 vs TP=2 | latency / throughput | 45 min | P7 |
| 7.1 | Cline vs Roo Code 同任务轨迹 | 步数 / 工具调用 | 60 min | P8 |
| 7.2 | 加 reflection 前后成功率 | SWE-Bench Lite 子集 | 90 min | P8 |
| 7.3 | ReAct prompt 三种写法 | 通过率 | 60 min | P8 |

总计 **27 个实验**。下面分 7 组展开。

---

## 1. Tokenizer & Embedding

### 实验 1.1 · 三家 BPE 同句对比（GPT-2 / LLaMA-3 / GLM）

> 关联章节：Phase 1 §1.2 · 耗时：~15 分钟 · 硬件：CPU 即可
> 比的是什么：同一段中英混合代码在三家 tokenizer 下的 **token 数**和**词表大小**

**🎯 实验目标**
确认"中文 / 代码效率"不是抽象指标——同一句话在不同 tokenizer 下 token 数能差 1.5-2×，直接决定推理成本。

**🤔 你应当看到的对比**
- GPT-2 词表 ~50K，对中文几乎逐字节切，token 数最多
- LLaMA-3 词表 128K，对英文/代码友好，对中文中等
- GLM (chatglm3 / glm-4) 词表 ~150K，含大量中文 token，中文最省

**🛠 操作步骤**
```bash
pip install transformers tiktoken
```
```python
from transformers import AutoTokenizer
text = """请实现一个快速排序：
def quicksort(arr):
    if len(arr) <= 1: return arr
    pivot = arr[len(arr)//2]
    left = [x for x in arr if x < pivot]
    return quicksort(left) + [x for x in arr if x == pivot] + quicksort([x for x in arr if x > pivot])
# 时间复杂度 O(n log n)，最坏 O(n²)。"""

for name in ["gpt2", "meta-llama/Meta-Llama-3-8B", "THUDM/chatglm3-6b"]:
    try:
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        ids = tok.encode(text)
        print(f"{name:40s}  vocab={tok.vocab_size:>7}  tokens={len(ids):>4}")
    except Exception as e:
        print(f"{name}: {e}")
```

**📊 关键观察指标**
- 三家 token 数（预期 GPT-2 ≈ 200，LLaMA-3 ≈ 130，GLM ≈ 100）
- 词表大小
- 单看中文部分（"请实现一个快速排序"+"时间复杂度..."）的差距更明显

**💡 结论确认 & 进阶**
- ✓ GLM 的中文 token 数约为 GPT-2 的一半
- ✓ 三家代码部分（def / return / [x for x in...]）token 数差异较小
- ✗ 不是越大词表越好——embedding 矩阵也线性增长
- 延伸：把 `text` 换成纯英文论文段落，差距还有这么大吗？

---

### 实验 1.2 · tiktoken vs sentencepiece 编码速度

> 关联章节：Phase 1 §1.2 · 耗时：~20 分钟 · 硬件：CPU
> 比的是什么：BPE 实现库的工程效率，tokens/s

**🎯 实验目标**
直观感受"tokenizer 是数据 pipeline 瓶颈之一"——纯 Python `sentencepiece` 和 Rust 实现的 `tiktoken` 速度差 5-20×。

**🤔 你应当看到的对比**
- `tiktoken` (Rust): 1-3M tokens/s 单核
- `transformers` 的 fast (Rust 后端): ~1M tokens/s
- `sentencepiece` Python: ~100-300K tokens/s

**🛠 操作步骤**
```bash
pip install tiktoken sentencepiece transformers
```
```python
import time, tiktoken
from transformers import AutoTokenizer
text = "def fibonacci(n):\n    return n if n<2 else fibonacci(n-1)+fibonacci(n-2)\n" * 5000

# tiktoken
enc = tiktoken.get_encoding("cl100k_base")
t = time.time(); ids = enc.encode(text); dt1 = time.time()-t
print(f"tiktoken     {len(ids)/dt1/1e6:.2f} M tok/s")

# transformers fast
tok_fast = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
t = time.time(); ids = tok_fast.encode(text); dt2 = time.time()-t
print(f"hf fast      {len(ids)/dt2/1e6:.2f} M tok/s")

# transformers slow（python）
tok_slow = AutoTokenizer.from_pretrained("gpt2", use_fast=False)
t = time.time(); ids = tok_slow.encode(text); dt3 = time.time()-t
print(f"hf slow      {len(ids)/dt3/1e6:.2f} M tok/s")
```

**📊 关键观察指标**
- 三种实现的 M tokens/s
- tiktoken / slow 的倍数（应当 ≥ 10×）

**💡 结论确认 & 进阶**
- ✓ tiktoken 比 slow Python 实现快 ≥10×
- ✓ fast 接近 tiktoken（同为 Rust）
- 延伸：在数据 pipeline 中如果用 slow tokenizer 处理 1T tokens 要多花多少天？

---

### 实验 1.3 · Embedding lookup vs 重新计算的延迟

> 关联章节：Phase 0 §3.2 · 耗时：~20 分钟 · 硬件：CPU 或 GPU
> 比的是什么：embedding 表 vs 哈希函数计算的访存差异

**🎯 实验目标**
理解"embedding 是访存 bound"——查表比浮点矩阵乘法 cache 更友好。

**🤔 你应当看到的对比**
- 一次 batch=32, seq=2048 的 embedding lookup: < 1 ms (GPU) / 几 ms (CPU)
- 同等元素数的 random init + matmul: 显著更慢

**🛠 操作步骤**
```python
import torch, time
V, D, B, L = 50000, 4096, 32, 2048
emb = torch.nn.Embedding(V, D).cuda() if torch.cuda.is_available() else torch.nn.Embedding(V, D)
ids = torch.randint(0, V, (B, L), device=emb.weight.device)

# warmup
for _ in range(3): _ = emb(ids)
torch.cuda.synchronize() if torch.cuda.is_available() else None

t = time.time()
for _ in range(20): out = emb(ids)
torch.cuda.synchronize() if torch.cuda.is_available() else None
print(f"embedding lookup: {(time.time()-t)/20*1000:.2f} ms/iter, throughput {B*L*20/(time.time()-t)/1e6:.1f} M tok/s")

# 对比：等量参数的 linear
lin = torch.nn.Linear(V, D, bias=False).to(emb.weight.device)
onehot = torch.nn.functional.one_hot(ids, V).float()
t = time.time()
for _ in range(5): out2 = lin(onehot)
torch.cuda.synchronize() if torch.cuda.is_available() else None
print(f"matmul one-hot:   {(time.time()-t)/5*1000:.2f} ms/iter")
```

**📊 关键观察指标**
- lookup 的 ms/iter
- matmul 的 ms/iter（应当 ≥ 50×）

**💡 结论确认 & 进阶**
- ✓ lookup 远快于等价 matmul
- 延伸：如果词表加大到 1M（多语言），lookup 时间几乎不变，但 embedding 显存爆炸——为什么？

---

## 2. 架构变体对比

### 实验 2.1 · MHA / GQA / MLA 的 KV cache 大小

> 关联章节：Phase 2 §2.3 · 耗时：~30 分钟 · 硬件：CPU 即可
> 比的是什么：三种 attention 在同等 hidden_size 下的 **KV cache 字节数 / token**

**🎯 实验目标**
不需要训模型——直接读 config + 算公式，看 KV cache 怎么从"主导显存"变成"次要项"。

**🤔 你应当看到的对比**
对 hidden=4096, layers=32, head_dim=128, BF16：
- MHA (32 KV heads): ~1 MB / token
- GQA (8 KV heads): ~0.25 MB / token
- MLA (rank=512 latent): ~0.07 MB / token

**🛠 操作步骤**
```python
from transformers import AutoConfig

def kv_per_token(cfg):
    L = cfg.num_hidden_layers
    h = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    d = cfg.hidden_size // cfg.num_attention_heads
    bytes_per = 2  # BF16
    return 2 * L * h * d * bytes_per  # K + V

for name in ["meta-llama/Llama-2-7b-hf",          # MHA
             "meta-llama/Meta-Llama-3-8B",        # GQA (8 kv heads)
             "Qwen/Qwen2.5-7B",                   # GQA
             "deepseek-ai/DeepSeek-V2-Lite"]:     # MLA
    try:
        cfg = AutoConfig.from_pretrained(name, trust_remote_code=True)
        print(f"{name:45s}  KV/tok = {kv_per_token(cfg)/1024:.1f} KB")
    except Exception as e:
        print(f"{name}: {e}")

# MLA 公式（DeepSeek-V2 lite: kv_lora_rank=512, qk_rope_head_dim=64, layers=27）
mla_kv = 2 * 27 * (512 + 64) * 1   # 实际 MLA cache 只存 latent + rope
print(f"DeepSeek-V2-Lite MLA实际cache: {mla_kv/1024:.1f} KB/tok")
```

**📊 关键观察指标**
- 三种架构的 KB/token（按公式列表）
- 32K context, batch=4 时的总 KV cache MB

**💡 结论确认 & 进阶**
- ✓ GQA 把 KV cache 砍到 1/4 左右
- ✓ MLA 进一步砍到 ~1/15
- ✗ MLA 不是免费午餐——计算量略增（rope + 上投影）
- 延伸：GLM-5.1 用 MLA + DSA，128K context 时 KV cache 比纯 MHA 节省多少？

---

### 实验 2.2 · Dense vs MoE 同激活参数的 FLOPs

> 关联章节：Phase 2 §3.4 · 耗时：~30 分钟 · 硬件：CPU
> 比的是什么：同样 ~1.5B 激活参数下，Dense 和 MoE 的总参数量与 forward FLOPs

**🎯 实验目标**
看清"MoE 用更多总参数换更小推理 FLOPs"是工程公平交易。

**🤔 你应当看到的对比**
- Dense 1.5B: 总参 1.5B, forward FLOPs ≈ 3 GFLOPs/token
- MoE 8x300M (top-2 = 600M 激活): 总参 ~2.4B, forward FLOPs 接近 600M dense

**🛠 操作步骤**
```python
def dense_flops(d, ff, L, vocab=128000):
    # 每 token forward 近似：12 * L * d^2 (attn) + 2 * L * d * ff (FFN) + d * vocab (lm head)
    return 12*L*d*d + 2*L*d*ff + d*vocab

def moe_flops(d, ff, L, n_exp, top_k, vocab=128000):
    # FFN 部分只激活 top_k 个 expert
    return 12*L*d*d + top_k * 2*L*d*ff + d*vocab

def params_dense(d, ff, L, vocab=128000):
    return L*(4*d*d + 3*d*ff) + 2*vocab*d

def params_moe(d, ff, L, n_exp, vocab=128000):
    return L*(4*d*d + n_exp*3*d*ff) + 2*vocab*d

# Dense 1.5B
d, ff, L = 1536, 6144, 28
print(f"Dense  params={params_dense(d,ff,L)/1e9:.2f}B  GFLOPs={dense_flops(d,ff,L)/1e9:.2f}")

# MoE 8 expert top-2 大致同等激活
d2, ff2, L2 = 1024, 2816, 24
print(f"MoE 8e top2  total={params_moe(d2,ff2,L2,8)/1e9:.2f}B  active≈{params_moe(d2,ff2,L2,2)/1e9:.2f}B  GFLOPs={moe_flops(d2,ff2,L2,8,2)/1e9:.2f}")

# GLM-5.1 估算（754B 总, ~70B 激活）
d3, ff3, L3 = 8192, 3072, 80
print(f"GLM-5.1-like  total={params_moe(d3,ff3,L3,256)/1e9:.0f}B  active(top8)≈{params_moe(d3,ff3,L3,8)/1e9:.0f}B")
```

**📊 关键观察指标**
- Dense 总参 vs MoE 总参（MoE 显著大）
- forward GFLOPs（MoE 显著小）

**💡 结论确认 & 进阶**
- ✓ MoE 总参大 2-5×，但 FLOPs 与"激活参数"接近
- ✓ 这就是为什么 GLM-5.1 754B 推理只要 ~70B dense 的成本
- 延伸：MoE 的"显存"成本和 FLOPs 不是一回事——为什么 vLLM 跑 MoE 仍需把所有 expert 装进显存？

---

### 实验 2.3 · RoPE base = 1e4 vs 5e5 的外推 PPL

> 关联章节：Phase 3 §1.3 · 耗时：~45 分钟 · 硬件：1×3090 / Colab T4
> 比的是什么：同一个小模型，仅改 RoPE base，外推到训练长度 2-4× 后的 PPL

**🎯 实验目标**
亲手验证"RoPE base 改大能延长有效上下文"。

**🤔 你应当看到的对比**
- base = 10000 (默认): seq_len = 4× train_len 时 PPL 飞涨（指数级）
- base = 500000: 同 seq_len PPL 几乎不变
- 但要付代价：base 改大后短序列 PPL 略升

**🛠 操作步骤**
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
model_id = "Qwen/Qwen2.5-0.5B"  # 训练 32K，但我们伪装它训练 2K 测外推
tok = AutoTokenizer.from_pretrained(model_id)
text = open("/etc/services").read()[:200000]  # 任何长文本即可

def ppl(model, ids):
    with torch.no_grad():
        out = model(ids, labels=ids)
    return torch.exp(out.loss).item()

ids_full = tok(text, return_tensors="pt").input_ids.cuda()

for base in [10000, 100000, 500000]:
    m = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).cuda()
    m.config.rope_theta = base
    # 强制重建 rope cache（不同模型实现略异；Qwen2 会在 forward 时按 config 计算）
    for layer in m.model.layers:
        if hasattr(layer.self_attn, "rotary_emb"):
            layer.self_attn.rotary_emb.base = base
    for L in [1024, 4096, 16384]:
        p = ppl(m, ids_full[:, :L])
        print(f"base={base:>6}  L={L:>5}  PPL={p:.2f}")
    del m; torch.cuda.empty_cache()
```

**📊 关键观察指标**
- (base=1e4, L=16K) PPL（应当 > 100）
- (base=5e5, L=16K) PPL（应当 < 20）
- 短序列 (L=1024) 各 base 的 PPL 对比

**💡 结论确认 & 进阶**
- ✓ base 改大显著降低长序列 PPL
- ✓ 但短序列 PPL 微涨——这就是 mid-training 阶段为什么要"渐进升级"
- 延伸：为什么单纯改 base 不如 YaRN？YaRN 多做了什么？

---

### 实验 2.4 · SwiGLU vs ReLU FFN 的小数据 loss

> 关联章节：Phase 2 §2.5 · 耗时：~60 分钟 · 硬件：1×3090 或 Colab T4
> 比的是什么：相同参数预算下，两种激活的 200 step training loss

**🎯 实验目标**
量化 SwiGLU 的"免费午餐"——同等 FLOPs 下 loss 更低。

**🤔 你应当看到的对比**
- SwiGLU 200 step loss 比 ReLU 低 0.1-0.2 nats
- SwiGLU 用 3 个矩阵（gate / up / down），需要把 ff 维度调到 2/3 以保持等参

**🛠 操作步骤**
```python
import torch, torch.nn as nn
from datasets import load_dataset

class Block(nn.Module):
    def __init__(self, d=256, ff=1024, swiglu=True):
        super().__init__()
        self.swiglu = swiglu
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        if swiglu:
            ff_eff = int(ff * 2/3)
            self.w1, self.w2, self.w3 = nn.Linear(d,ff_eff), nn.Linear(d,ff_eff), nn.Linear(ff_eff,d)
        else:
            self.w1, self.w3 = nn.Linear(d,ff), nn.Linear(ff,d)
    def forward(self, x):
        y, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + y
        h = self.ln2(x)
        if self.swiglu:
            h = self.w3(torch.nn.functional.silu(self.w1(h)) * self.w2(h))
        else:
            h = self.w3(torch.relu(self.w1(h)))
        return x + h

class TinyLM(nn.Module):
    def __init__(self, V=10000, d=256, swiglu=True):
        super().__init__()
        self.emb = nn.Embedding(V, d)
        self.blocks = nn.ModuleList([Block(d, swiglu=swiglu) for _ in range(4)])
        self.head = nn.Linear(d, V)
    def forward(self, x): 
        h = self.emb(x)
        for b in self.blocks: h = b(h)
        return self.head(h)

ds = load_dataset("roneneldan/TinyStories", split="train[:5000]")
text = " ".join(ds["text"])[:500000]
ids = torch.tensor([ord(c)%10000 for c in text], dtype=torch.long).cuda()

for swiglu in [False, True]:
    torch.manual_seed(42)
    m = TinyLM(swiglu=swiglu).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    losses = []
    for step in range(200):
        i = torch.randint(0, len(ids)-129, (32,))
        batch = torch.stack([ids[k:k+128] for k in i])
        logits = m(batch[:,:-1])
        loss = nn.functional.cross_entropy(logits.reshape(-1,10000), batch[:,1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    print(f"{'SwiGLU' if swiglu else 'ReLU':6s}  step200 loss={losses[-1]:.3f}  avg last20={sum(losses[-20:])/20:.3f}")
```

**📊 关键观察指标**
- 200 step 时两者 loss 差
- 收敛速率差异

**💡 结论确认 & 进阶**
- ✓ SwiGLU 终值 loss 更低（≥ 0.05 nats）
- ✗ 但参数量需手动调（ff × 2/3），否则不公平
- 延伸：把 SwiGLU 换成 GeGLU（gelu 替 silu）会更好吗？

---

### 实验 2.5 · LayerNorm vs RMSNorm 速度

> 关联章节：Phase 2 §2.4 · 耗时：~15 分钟 · 硬件：GPU
> 比的是什么：同 hidden 下两种 norm 的 forward μs

**🎯 实验目标**
确认 RMSNorm 的"省一次均值减"在大 hidden 下确实更快。

**🤔 你应当看到的对比**
- RMSNorm 比 LayerNorm 快 10-30%（GPU 上）
- batch×seq 越大差距越小（被 matmul 主导）

**🛠 操作步骤**
```python
import torch, torch.nn as nn, time

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps=eps
    def forward(self,x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

x = torch.randn(32, 2048, 4096, device='cuda', dtype=torch.bfloat16)
ln, rn = nn.LayerNorm(4096).cuda().bfloat16(), RMSNorm(4096).cuda().bfloat16()

for name, m in [('LN', ln), ('RMSNorm', rn)]:
    for _ in range(5): _ = m(x)  # warmup
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(100): _ = m(x)
    torch.cuda.synchronize()
    print(f"{name:8s}  {(time.time()-t)/100*1e6:.1f} μs/iter")
```

**📊 关键观察指标**
- 两者 μs/iter
- RMSNorm / LN 速度比

**💡 结论确认 & 进阶**
- ✓ RMSNorm 快 10-30%
- 延伸：用 fused kernel（如 `flash_attn` 自带的 RMSNorm）能再加速多少？

---

## 3. 数据处理

### 实验 3.1 · MinHash-LSH 去重前后

> 关联章节：Phase 1 §3.4 · 耗时：~45 分钟 · 硬件：CPU
> 比的是什么：在小型代码语料上去重前后的**文件数**和**耗时**

**🎯 实验目标**
真正跑一遍 MinHash-LSH，理解"为什么 The Stack v2 去重前 6T、去重后 0.5T"。

**🤔 你应当看到的对比**
- 输入 ~10K 个 GitHub Python 文件
- 去重后剩 ~6-8K（~20-40% 重复率，因为大量 boilerplate）

**🛠 操作步骤**
```bash
pip install datasketch datasets
```
```python
from datasets import load_dataset
from datasketch import MinHash, MinHashLSH
import re, time

ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
docs = []
for i, ex in enumerate(ds):
    if i >= 10000: break
    docs.append(ex["content"])
print(f"loaded {len(docs)} docs")

def get_mh(text, num_perm=128):
    mh = MinHash(num_perm=num_perm)
    for w in re.findall(r"\w+", text.lower()):
        mh.update(w.encode())
    return mh

t = time.time()
lsh = MinHashLSH(threshold=0.8, num_perm=128)
keep = []
for i, d in enumerate(docs):
    mh = get_mh(d)
    if not lsh.query(mh):
        lsh.insert(str(i), mh)
        keep.append(i)
print(f"dedup time: {time.time()-t:.1f}s")
print(f"before: {len(docs)}  after: {len(keep)}  removed: {len(docs)-len(keep)} ({(1-len(keep)/len(docs))*100:.1f}%)")
```

**📊 关键观察指标**
- 去重前 / 后文件数
- 重复率 %
- 耗时（10K 文件应当 < 5 分钟）

**💡 结论确认 & 进阶**
- ✓ 真实代码语料重复率 20-40%
- ✓ MinHash 是次线性的（threshold 越高越快）
- 延伸：threshold=0.7 vs 0.9 的去重率差多少？激进去重伤害代码完整度吗？

---

### 实验 3.2 · HumanEval n-gram 在 The Stack 上的命中

> 关联章节：Phase 1 §4.2, Phase 6 §3.2 · 耗时：~30 分钟 · 硬件：CPU
> 比的是什么：HumanEval 题目的 13-gram 在公开代码语料中的命中数

**🎯 实验目标**
亲眼看到"评测集污染"是怎么自然发生的——HumanEval 早就在 GitHub 上传遍了。

**🤔 你应当看到的对比**
- 13-gram 命中：~30-50% 的 HumanEval 题目能在 The Stack 子集中找到 ≥1 次重叠
- prompt+canonical_solution 直接匹配：~10-30%

**🛠 操作步骤**
```python
from datasets import load_dataset

he = load_dataset("openai/openai_humaneval", split="test")
stack = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train")

def ngrams(text, n=13):
    toks = text.split()
    return set(" ".join(toks[i:i+n]) for i in range(len(toks)-n+1))

# 把所有 stack 文件的 13-gram 入集合（小子集）
stack_grams = set()
for i, ex in enumerate(stack):
    if i >= 5000: break
    stack_grams |= ngrams(ex["content"])
print(f"stack 13-grams: {len(stack_grams):,}")

hit = 0
for ex in he:
    full = ex["prompt"] + ex["canonical_solution"]
    h = ngrams(full)
    if h & stack_grams:
        hit += 1
print(f"HumanEval hits: {hit}/{len(he)} = {hit/len(he)*100:.1f}%")
```

**📊 关键观察指标**
- 命中题数 / 总数
- 命中比例（应当 ≥ 20%）

**💡 结论确认 & 进阶**
- ✓ 评测污染是默认状态而非例外
- ✓ 这就是为什么必须用 LiveCodeBench 等"按月"的基准
- 延伸：n=8 vs n=20 命中率差多少？n 太小会假阳，太大会漏检

---

### 实验 3.3 · FIM 改写前后 prompt 形态

> 关联章节：Phase 1 §5.3 · 耗时：~15 分钟 · 硬件：CPU
> 比的是什么：原始代码 vs FIM (PSM/SPM) 改写后的训练样本视觉

**🎯 实验目标**
理解 FIM 不是新数据，是把原数据"切三段重排"，让模型学会中间填空。

**🤔 你应当看到的对比**
- 原始：连续代码
- PSM (Prefix-Suffix-Middle): `<PRE>前缀<SUF>后缀<MID>中间<EOT>`
- SPM (Suffix-Prefix-Middle): `<SUF>后缀<PRE>前缀<MID>中间<EOT>`

**🛠 操作步骤**
```python
import random
code = """def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr)//2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)"""

def fim_psm(code):
    n = len(code)
    a, b = sorted(random.sample(range(n), 2))
    return f"<PRE>{code[:a]}<SUF>{code[b:]}<MID>{code[a:b]}<EOT>"

def fim_spm(code):
    n = len(code)
    a, b = sorted(random.sample(range(n), 2))
    return f"<SUF>{code[b:]}<PRE>{code[:a]}<MID>{code[a:b]}<EOT>"

random.seed(42)
print("=== ORIGINAL ===\n" + code)
print("\n=== PSM ===\n" + fim_psm(code))
print("\n=== SPM ===\n" + fim_spm(code))
```

**📊 关键观察指标**
- 改写后整段长度（应当与原始一致 + 4 个特殊 token）
- "缺口"位置随机性

**💡 结论确认 & 进阶**
- ✓ FIM 不增加数据量，只是重排
- ✓ PSM 与 SPM 都需要——推理时不知道用户给的是哪种顺序
- 延伸：FIM 比例多少最佳？SantaCoder 论文给了 0.5，再高会损害左到右能力，为什么？

---

### 实验 3.4 · packing vs padding 的 throughput

> 关联章节：Phase 2 §4.2 · 耗时：~30 分钟 · 硬件：1×3090
> 比的是什么：变长序列在 batch 内 padding vs packing 的 tokens/s

**🎯 实验目标**
看到 packing 把"白费的 padding token 计算"换成"真实 token 计算"的 throughput 提升。

**🤔 你应当看到的对比**
- 平均长度 = 512, max_len = 2048 时
- padding：~25% token 是 pad
- packing：> 95% token 有效
- throughput 提升 ~30-50%

**🛠 操作步骤**
```python
import torch, time, random
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16).cuda()
opt = torch.optim.AdamW(m.parameters(), lr=1e-5)

# 生成变长样本
random.seed(0)
samples = [torch.randint(0, 50000, (random.randint(100, 1500),)) for _ in range(256)]
print(f"avg len = {sum(len(s) for s in samples)/len(samples):.0f}, max = {max(len(s) for s in samples)}")

# Padding 模式
def run_pad(batch_size=8, steps=20):
    t = time.time()
    for step in range(steps):
        batch = random.sample(samples, batch_size)
        max_l = max(len(b) for b in batch)
        ids = torch.stack([torch.cat([b, torch.zeros(max_l-len(b), dtype=torch.long)]) for b in batch]).cuda()
        out = m(ids, labels=ids); out.loss.backward(); opt.step(); opt.zero_grad()
    return batch_size * max_l * steps / (time.time()-t)

# Packing 模式
def run_pack(target=4096, steps=20):
    t = time.time(); total=0
    for step in range(steps):
        packed = []
        while len(packed) < target:
            packed.extend(random.choice(samples).tolist())
        ids = torch.tensor(packed[:target]).unsqueeze(0).cuda()
        out = m(ids, labels=ids); out.loss.backward(); opt.step(); opt.zero_grad()
        total += target
    return total / (time.time()-t)

print(f"padding: {run_pad():.0f} tok/s")
print(f"packing: {run_pack():.0f} tok/s")
```

**📊 关键观察指标**
- 两种模式 tokens/s
- 提升 %

**💡 结论确认 & 进阶**
- ✓ packing 有效 throughput 显著高
- ✗ 简单 packing 让前后样本互相 attend，需要 cross-doc attention mask
- 延伸：用 `flash_attn_varlen` 严格切分 doc 后还有 padding 优势吗？

---

## 4. 微调与 RL

### 实验 4.1 · Base vs SFT 同指令输出

> 关联章节：Phase 4 §1.2 · 耗时：~20 分钟 · 硬件：1×3090
> 比的是什么：base 模型与对应 instruct 版在同一 prompt 下的输出风格

**🎯 实验目标**
直观看到 SFT 把"续写器"改造成"指令跟随器"——base 会"补全"，instruct 会"回答"。

**🤔 你应当看到的对比**
- base: 把"用 Python 写个快排"补全成 "...，思路是..."（继续写下去）
- instruct: 直接给代码块

**🛠 操作步骤**
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

prompt = "用 Python 写一个快速排序，要带注释。"
for name in ["Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-0.5B-Instruct"]:
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).cuda()
    if "Instruct" in name:
        text = tok.apply_chat_template([{"role":"user","content":prompt}], tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    ids = tok(text, return_tensors="pt").to("cuda")
    out = m.generate(**ids, max_new_tokens=200, do_sample=False)
    print(f"\n===== {name} =====")
    print(tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True))
    del m; torch.cuda.empty_cache()
```

**📊 关键观察指标**
- 视觉对比两段输出
- 是否含代码块 / Markdown 结构

**💡 结论确认 & 进阶**
- ✓ instruct 输出更结构化、更直接
- ✓ base 倾向于"续写一段话"
- 延伸：base 加上 few-shot 示例后能模仿 instruct 风格吗？

---

### 实验 4.2 · LoRA r=8/16/64 的 loss 曲线

> 关联章节：Phase 4 §2.4 · 耗时：~90 分钟 · 硬件：1×3090
> 比的是什么：相同数据下，LoRA rank 对收敛 / 显存的影响

**🎯 实验目标**
回答"LoRA rank 该选多大"这个常被问的问题——亲眼看 r=8 够不够。

**🤔 你应当看到的对比**
- r=8: 显存最少，loss 收敛稍慢但终值接近 r=16
- r=16: 性价比最高
- r=64: 显存增加 ~4×，loss 提升 < 5%

**🛠 操作步骤**
```bash
pip install peft trl datasets
```
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
ds = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")
ds = ds.map(lambda x: {"text": f"### Instruction:\n{x['instruction']}\n### Response:\n{x['output']}"})

for r in [8, 16, 64]:
    m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16)
    cfg = LoraConfig(r=r, lora_alpha=2*r, target_modules=["q_proj","v_proj"], task_type="CAUSAL_LM")
    m = get_peft_model(m, cfg)
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    trainer = SFTTrainer(
        model=m, tokenizer=tok, train_dataset=ds,
        args=SFTConfig(output_dir=f"./lora_r{r}", per_device_train_batch_size=4,
                        max_steps=200, logging_steps=20, learning_rate=2e-4,
                        bf16=True, report_to="none", max_seq_length=512))
    trainer.train()
    losses = [l["loss"] for l in trainer.state.log_history if "loss" in l]
    print(f"r={r:>2}  trainable={n_train/1e6:.2f}M  final_loss={losses[-1]:.3f}  avg_last3={sum(losses[-3:])/3:.3f}")
```

**📊 关键观察指标**
- 三个 rank 的可训练参数 M
- 200 step 终 loss
- GPU 显存（用 `nvidia-smi`）

**💡 结论确认 & 进阶**
- ✓ r=16 与 r=64 终 loss 差 < 5%
- ✓ r=8 已经接近上限
- 延伸：把 target_modules 扩到 `["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"]` 后 r=8 能追上 r=64 吗？

---

### 实验 4.3 · Chat template 三家 token 化差异

> 关联章节：Phase 4 §1.4 · 耗时：~15 分钟 · 硬件：CPU
> 比的是什么：ChatML / GLM / Llama-3 三种 chat template 同 prompt 的 token 数 / 视觉

**🎯 实验目标**
理解 chat template 不是装饰——它决定推理时 prompt 的格式正确性。

**🤔 你应当看到的对比**
- ChatML: `<|im_start|>...<|im_end|>` 包裹
- GLM: `[gMASK]<sop><|user|>...<|assistant|>`
- Llama-3: `<|begin_of_text|><|start_header_id|>...<|end_header_id|>`

**🛠 操作步骤**
```python
from transformers import AutoTokenizer

msgs = [
    {"role":"system","content":"你是助手。"},
    {"role":"user","content":"写个快排"},
    {"role":"assistant","content":"def qs(a):..."},
    {"role":"user","content":"加上类型注解"},
]

for name in ["Qwen/Qwen2.5-7B-Instruct", "THUDM/chatglm3-6b", "meta-llama/Meta-Llama-3-8B-Instruct"]:
    try:
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        print(f"\n===== {name} (tokens={len(ids)}) =====\n{text[:400]}...")
    except Exception as e:
        print(f"{name}: {e}")
```

**📊 关键观察指标**
- 每家 token 数
- 视觉特殊 token

**💡 结论确认 & 进阶**
- ✓ 三家完全不兼容——直接用错 template 就是"模型变笨"
- ✓ token 数差异 5-15%
- 延伸：vLLM 部署时如果不传 `--chat-template` 会怎么样？

---

### 实验 4.4 · PPO vs GRPO 200 step 的 reward

> 关联章节：Phase 5 §2.3 · 耗时：~90 分钟 · 硬件：1×3090
> 比的是什么：同任务、同模型，PPO 与 GRPO 的 reward 上升速度

**🎯 实验目标**
看见 GRPO 的"无 critic"优势——少一个网络，且 reward 上升不慢于 PPO。

**🤔 你应当看到的对比**
- 任务：让 0.5B 模型生成"长度刚好 50 字符"的句子（RLVR 玩具任务）
- PPO: reward 100 step 才稳定上升
- GRPO: 50 step 就上升

**🛠 操作步骤**
```bash
pip install trl
```
```python
# 用 trl 的 GRPOTrainer 与 PPOTrainer 跑同一个 reward function
# reward = -|len(text)-50|
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
prompts = ["写一句话"] * 256
ds = Dataset.from_dict({"prompt": prompts})

def reward_fn(completions, **kw):
    return [-abs(len(c)-50) for c in completions]

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=reward_fn,
    args=GRPOConfig(output_dir="./grpo", per_device_train_batch_size=2,
                     num_generations=4, max_completion_length=80,
                     max_steps=200, logging_steps=10, bf16=True, report_to="none"),
    train_dataset=ds)
trainer.train()
rewards = [l["reward"] for l in trainer.state.log_history if "reward" in l]
print(f"GRPO reward: start={rewards[0]:.2f}  end={rewards[-1]:.2f}")
# PPO 同步骤略 — 见 trl 文档 PPOTrainer
```

**📊 关键观察指标**
- step 0 / 100 / 200 的平均 reward
- reward 上升曲线

**💡 结论确认 & 进阶**
- ✓ GRPO 不需要 value head，显存 ~30% 更少
- ✓ reward 趋势相近
- 延伸：把 num_generations 从 4 改 8 / 16 怎么影响 reward 方差？

---

### 实验 4.5 · KL coef 0.001 / 0.01 / 0.1 的多样性

> 关联章节：Phase 5 §3.2 · 耗时：~60 分钟 · 硬件：1×3090
> 比的是什么：同 RL 训练 100 step 后，三种 KL coef 下生成的 distinct-2

**🎯 实验目标**
看到 KL coef 是"探索 vs 模仿原模型"的旋钮——太小模型会塌缩，太大学不动。

**🤔 你应当看到的对比**
- KL=0.001: distinct-2 → 显著下降（模型坍缩到几句话）
- KL=0.01:  适中，reward 涨且 distinct 维持
- KL=0.1:  reward 几乎不动（被 KL 拉回 base）

**🛠 操作步骤**
（基于上面 4.4 的 GRPO 框架，只改 `beta` 参数；注意 trl `GRPOConfig.beta` 即 KL coef）
```python
def distinct_n(texts, n=2):
    from collections import Counter
    grams = []
    for t in texts: 
        toks = t.split()
        grams += [tuple(toks[i:i+n]) for i in range(len(toks)-n+1)]
    return len(set(grams))/max(len(grams),1)

# 对三组 beta=0.001, 0.01, 0.1 各训 100 step，结束后用相同 prompt 各采样 100 条
# 计算 distinct_n + 平均 reward，列表对比
```

**📊 关键观察指标**
- 三种 KL 下的 distinct-2
- 平均 reward
- 是否塌缩到 < 5 种独特句子

**💡 结论确认 & 进阶**
- ✓ KL=0.001 严重塌缩
- ✓ KL=0.01 是常用甜点区
- 延伸：把 KL 从"对 base 模型"改成"对 SFT 模型"，多样性会怎么变？

---

## 5. 评测

### 实验 5.1 · HumanEval vs HumanEval+ 同模型分差

> 关联章节：Phase 6 §1.3 · 耗时：~30 分钟 · 硬件：1×3090
> 比的是什么：同一模型在 HumanEval 与 HumanEval+ 上的 pass@1 差距

**🎯 实验目标**
看到"原 HumanEval 测试用例不足"——模型在 + 版本上常掉 5-15 个百分点。

**🤔 你应当看到的对比**
- 0.5B-1B 小模型：HumanEval ~25%, HumanEval+ ~18-20%
- 7B 模型：HumanEval ~50%, HumanEval+ ~40-45%

**🛠 操作步骤**
```bash
pip install evalplus
```
```bash
python -m evalplus.evaluate --model "Qwen/Qwen2.5-Coder-0.5B-Instruct" \
    --dataset humaneval --backend hf --n-samples 1 --temperature 0
```
（输出会同时给出 base / plus 两个分数）

或者手写最小版：
```python
from datasets import load_dataset
from transformers import pipeline
import subprocess, tempfile, os

he = load_dataset("evalplus/humanevalplus", split="test")
gen = pipeline("text-generation", "Qwen/Qwen2.5-Coder-0.5B-Instruct", device=0)

def run_tests(code, tests):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code + "\n" + tests + "\nprint('OK')")
        p = f.name
    try:
        r = subprocess.run(["python", p], capture_output=True, timeout=10)
        return "OK" in r.stdout.decode()
    except: return False
    finally: os.unlink(p)

base_pass = plus_pass = 0
for ex in list(he)[:30]:
    completion = gen(ex["prompt"], max_new_tokens=300, do_sample=False)[0]["generated_text"]
    code = completion[len(ex["prompt"]):]
    full = ex["prompt"] + code
    if run_tests(full, ex["test"]): base_pass += 1
    if run_tests(full, ex["test"] + "\n" + ex.get("plus_test","")): plus_pass += 1
print(f"base pass: {base_pass}/30  plus pass: {plus_pass}/30")
```

**📊 关键观察指标**
- HumanEval pass@1
- HumanEval+ pass@1
- 差值

**💡 结论确认 & 进阶**
- ✓ + 版本严格更难
- ✓ 差值 5-15 个点是常态
- 延伸：哪些题型 + 版本掉得最多？（边界条件 / 异常输入）

---

### 实验 5.2 · LiveCodeBench 按月分段的污染 gap

> 关联章节：Phase 6 §3.2 · 耗时：~45 分钟 · 硬件：1×3090
> 比的是什么：同模型在 cutoff **之前** vs **之后**月份题目的 pass@1

**🎯 实验目标**
直接看到训练污染的"时间幽灵"——cutoff 后题目分数应当显著低于 cutoff 前。

**🤔 你应当看到的对比**
- 取 Qwen2.5-Coder（cutoff ≈ 2024-08）
- 2024 年 1-7 月题目：pass@1 ~30-40%
- 2024 年 9-12 月题目：pass@1 ~15-25%

**🛠 操作步骤**
```python
from datasets import load_dataset
ds = load_dataset("livecodebench/code_generation_lite", split="test", trust_remote_code=True)
# 按 contest_date 分组
import datetime
before, after = [], []
cutoff = datetime.date(2024, 8, 1)
for ex in ds:
    d = datetime.date.fromisoformat(ex["contest_date"][:10])
    (before if d < cutoff else after).append(ex)
print(f"before: {len(before)}  after: {len(after)}")

# 用相同 generate + 测试逻辑跑两组，输出 pass@1
# （为节省时间，每组取 50 题）
```

**📊 关键观察指标**
- before / after pass@1
- gap

**💡 结论确认 & 进阶**
- ✓ after 分数显著低
- ✓ gap 反映污染量
- 延伸：把 cutoff 改成 2024-12，gap 还在吗？（如果模型新版本训过 2024-12 之前的全部题）

---

### 实验 5.3 · greedy vs temp=0.7 在 HumanEval

> 关联章节：Phase 6 §1.4 · 耗时：~30 分钟 · 硬件：1×3090
> 比的是什么：温度对 pass@1 / pass@10 的影响

**🎯 实验目标**
看到"贪心适合 pass@1，采样适合 pass@k"。

**🤔 你应当看到的对比**
- greedy (T=0): pass@1 ~30%, pass@10 也只能 ~30%（一直同一答案）
- T=0.7: pass@1 ~28%（略低，单次更随机）, pass@10 ~50%

**🛠 操作步骤**
```python
# 复用 5.1 的框架，对每题采样 10 次
# 模式 1: do_sample=False, n=1 → pass@1
# 模式 2: do_sample=True, temperature=0.7, n=10 → pass@1 = 第一次成功率，pass@10 = 任一次成功率
```

**📊 关键观察指标**
- 两种模式 pass@1
- T=0.7 的 pass@10

**💡 结论确认 & 进阶**
- ✓ 采样的 pass@10 显著大于 pass@1
- ✓ 这是为什么 RL 阶段非要采样
- 延伸：T=1.5 时 pass@10 还会涨吗？

---

## 6. 推理与部署

### 实验 6.1 · BF16 vs FP8 vs AWQ 的延迟 / 显存

> 关联章节：Phase 7 §2.3 · 耗时：~60 分钟 · 硬件：1×3090（FP8 需 H100/L40，可在 Colab 替换）
> 比的是什么：同模型同 prompt 在三种精度下的 TTFT / 显存

**🎯 实验目标**
看到量化"用精度换显存换吞吐"的具体数字。

**🤔 你应当看到的对比**
- 7B BF16: ~14 GB
- 7B FP8: ~7-8 GB
- 7B AWQ (4bit): ~4-5 GB
- 延迟：AWQ 最快（小批），FP8 中等，BF16 最慢

**🛠 操作步骤**
```bash
pip install vllm autoawq
```
```python
from vllm import LLM, SamplingParams
import time, torch

prompt = "Write a Python function to compute fibonacci.\n"
sp = SamplingParams(temperature=0, max_tokens=200)

# BF16
llm = LLM("Qwen/Qwen2.5-7B-Instruct", dtype="bfloat16")
t = time.time(); _ = llm.generate([prompt]*4, sp); dt = time.time()-t
print(f"BF16 latency: {dt:.2f}s  GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")
del llm; torch.cuda.empty_cache()

# AWQ
llm = LLM("Qwen/Qwen2.5-7B-Instruct-AWQ", quantization="awq", dtype="float16")
t = time.time(); _ = llm.generate([prompt]*4, sp); dt = time.time()-t
print(f"AWQ  latency: {dt:.2f}s  GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")
```

**📊 关键观察指标**
- 三种精度的 GPU GB
- 同 prompt 总延迟

**💡 结论确认 & 进阶**
- ✓ AWQ 4bit 显存接近 1/3
- ✓ FP8 在 H100 上是甜点区
- 延伸：长上下文（8K+）时各精度差距还这么大吗？（KV cache 不被权重量化影响）

---

### 实验 6.2 · vLLM vs SGLang 的 throughput

> 关联章节：Phase 7 §3.2 · 耗时：~60 分钟 · 硬件：1×3090
> 比的是什么：同模型同请求负载在两个引擎下的 TTFT / output tokens/s

**🎯 实验目标**
看到引擎选择会让同一模型 throughput 差 1.2-2×。

**🤔 你应当看到的对比**
- vLLM: 强项是 paged-attention 高 batch
- SGLang: 强项是 RadixAttention 长 prompt 复用

**🛠 操作步骤**
```bash
# Terminal A
pip install vllm
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-1.5B-Instruct --port 8000

# Terminal B
pip install sglang
python -m sglang.launch_server --model-path Qwen/Qwen2.5-1.5B-Instruct --port 8001
```
```python
import asyncio, aiohttp, time

async def stress(url, n=50):
    async with aiohttp.ClientSession() as s:
        async def call():
            t0 = time.time()
            async with s.post(url, json={"model":"...", "prompt":"hello "*100, "max_tokens":200}) as r:
                d = await r.json()
            return time.time()-t0
        return await asyncio.gather(*[call() for _ in range(n)])

# 分别 stress 两个 endpoint
```

**📊 关键观察指标**
- 平均 TTFT
- 总 throughput (tokens/s)
- p99 延迟

**💡 结论确认 & 进阶**
- ✓ 两者差距 1.2-2×（依负载而定）
- 延伸：长 system prompt 共享时 SGLang 优势更明显——为什么？

---

### 实验 6.3 · KV cache 命中 vs miss 的 TTFT

> 关联章节：Phase 7 §3.3 · 耗时：~30 分钟 · 硬件：1×3090
> 比的是什么：相同 system prompt 第一次请求 vs 第二次请求的 TTFT

**🎯 实验目标**
看到 prefix caching 对长 system prompt 应用的"接近免费"加速。

**🤔 你应当看到的对比**
- 第一次（miss）: prefill 全部 N tokens, TTFT ~ N/throughput
- 第二次（hit）: 只 prefill 新增部分, TTFT 接近常数

**🛠 操作步骤**
```python
from vllm import LLM, SamplingParams
import time
llm = LLM("Qwen/Qwen2.5-1.5B-Instruct", enable_prefix_caching=True)
sys = "你是一个资深 Python 工程师。" * 200  # 长 system prompt
sp = SamplingParams(temperature=0, max_tokens=50)

for i in range(3):
    t = time.time()
    _ = llm.generate([sys + f"\n问题{i}"], sp)
    print(f"call {i}: {time.time()-t:.2f}s")
```

**📊 关键观察指标**
- 三次调用的延迟
- 第二次起应当显著缩短

**💡 结论确认 & 进阶**
- ✓ prefix caching 第二次起 TTFT 减半甚至更多
- 延伸：缓存粒度 = 每 16 token 一个 block，那 system prompt 第 17 个 token 改了会怎样？

---

### 实验 6.4 · chunked prefill 开关对长 prompt ITL

> 关联章节：Phase 7 §3.4 · 耗时：~30 分钟 · 硬件：1×3090
> 比的是什么：长 prompt 在 chunked prefill on/off 下的 ITL（inter-token latency）

**🎯 实验目标**
理解 chunked prefill 是为了"不让长 prompt 阻塞短请求的 decode"。

**🤔 你应当看到的对比**
- off: 长 prompt 来时短请求 decode 卡顿（ITL 飙高）
- on: 长 prompt 被切碎，短请求 ITL 平稳

**🛠 操作步骤**
启动 vLLM 两次，分别加 / 不加 `--enable-chunked-prefill`，发并发请求：1 个 8K prompt + 5 个短 prompt（500 tokens 输出），测每个短请求的 token 间隔。

```python
# 用 streaming 方式调用 OpenAI API endpoint，记录每次 chunk 的到达时间
import time, requests
def stream_itl(prompt, n_tokens=500):
    r = requests.post("http://localhost:8000/v1/completions",
        json={"model":"...", "prompt":prompt, "max_tokens":n_tokens, "stream":True}, stream=True)
    times = []; t0 = time.time()
    for line in r.iter_lines():
        if line: times.append(time.time()-t0)
    itl = [times[i+1]-times[i] for i in range(len(times)-1)]
    return sum(itl)/len(itl) if itl else 0
```

**📊 关键观察指标**
- 短请求 ITL（开 / 关 chunked prefill）

**💡 结论确认 & 进阶**
- ✓ chunked prefill 明显平滑 ITL
- 延伸：chunk size 从 512 改 2048 会怎样？

---

### 实验 6.5 · TP=1 vs TP=2 latency / throughput

> 关联章节：Phase 7 §4.2 · 耗时：~45 分钟 · 硬件：2×3090（如果只有 1 张则跳过）
> 比的是什么：同模型在单卡 vs 张量并行 2 卡的延迟与吞吐

**🎯 实验目标**
看到 TP 不是无脑加速——通信开销在小模型上反而拖慢。

**🤔 你应当看到的对比**
- 7B TP=1: latency = X, throughput = Y
- 7B TP=2: latency 略低（更大 batch 也能装），throughput 1.5-1.8×（亚线性）

**🛠 操作步骤**
```bash
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 1 --port 8000
# 另开
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct --tensor-parallel-size 2 --port 8001
```
然后用同样并发负载打两个端口对比。

**📊 关键观察指标**
- TP=1 / TP=2 throughput
- 加速比（< 2 是常态）

**💡 结论确认 & 进阶**
- ✓ TP 加速比通常 1.5-1.8×（被 all-reduce 开销吃掉）
- 延伸：换成 70B 模型，TP=4 的加速比会更接近线性，为什么？

---

## 7. Agent

### 实验 7.1 · Cline vs Roo Code 同任务轨迹

> 关联章节：Phase 8 §2.2 · 耗时：~60 分钟 · 硬件：本地 + Anthropic / OpenAI API
> 比的是什么：两个 VSCode 插件在同一个 SWE 小任务上的步数 / 工具调用次数

**🎯 实验目标**
不同 agent loop 实现差异的直观感受——同模型、同任务，轨迹长度可能差 30%。

**🤔 你应当看到的对比**
- 任务："给这个 Python 项目加一个 README，描述其用途和 API"
- Cline: 偏读多个文件后再写
- Roo Code: 工具调用更细，步数稍多

**🛠 操作步骤**
1. VSCode 安装 Cline 插件，配置同一个 API key（Claude 或 GPT-4）
2. 在 demo 仓库（取一个 100-200 行的小 repo）执行同一指令
3. 安装 Roo Code 插件，相同仓库（先 git reset 还原），相同指令
4. 记录每次的总轮数 / 工具调用类型

**📊 关键观察指标**
- 两个 agent 的总步数
- 工具调用类型分布
- 任务是否最终成功

**💡 结论确认 & 进阶**
- ✓ 同模型同任务，agent 框架影响显著
- 延伸：如果换成 GLM-4.5 / Qwen3-Coder，哪个 agent 框架更适配？

---

### 实验 7.2 · 加 reflection 步骤前后的 SWE-Bench 子集

> 关联章节：Phase 8 §3.3 · 耗时：~90 分钟 · 硬件：API
> 比的是什么：基础 ReAct vs ReAct+Reflection 在 5 道 SWE-Bench Lite 题上的成功率

**🎯 实验目标**
看到"自我反思"作为一种最简单的 agent 增强带来的提升幅度。

**🤔 你应当看到的对比**
- 基础 ReAct: 5 题中过 1-2 题
- + Reflection（失败后回看自己的 trajectory，重试一次）: 5 题中过 2-3 题

**🛠 操作步骤**
取 SWE-Bench Lite 头 5 道题，用 minimal agent loop 实现两版：
```python
# 版本 A: 单次 ReAct
def agent_react(task, model, max_steps=15):
    history = []
    for _ in range(max_steps):
        action = model(history + [task])
        if action == "DONE": break
        result = execute(action)
        history.append((action, result))
    return verify(task)

# 版本 B: 加 reflection
def agent_with_reflect(task, model):
    success = agent_react(task, model)
    if not success:
        reflection = model("Why did the above fail? List 2 lessons.")
        success = agent_react(task + "\nLessons:\n" + reflection, model)
    return success
```

**📊 关键观察指标**
- 两版在 5 题上的通过数
- 平均工具调用次数

**💡 结论确认 & 进阶**
- ✓ Reflection 通常 +1 题（20% 提升）
- 延伸：加上 unit-test feedback 作为 reflection 输入，提升幅度会更大吗？

---

### 实验 7.3 · ReAct prompt 三种写法的通过率

> 关联章节：Phase 8 §3.2 · 耗时：~60 分钟 · 硬件：API
> 比的是什么：同任务、同模型，ReAct prompt 模板对成功率的敏感度

**🎯 实验目标**
理解"prompt engineering 在 agent 里仍然重要"——同语义不同 prompt 可能差 10-20%。

**🤔 你应当看到的对比**
- 写法 A（最简）: "Think step by step. Use tools when needed."
- 写法 B（明确格式）: "Output: Thought: ... Action: ... Observation: ..."
- 写法 C（带示例）: B + 2 个 few-shot 例子

**🛠 操作步骤**
取 5 道简单代码任务（修 bug、加单测），用同模型同 reward 跑三个 prompt：
```python
PROMPTS = {
  "A": "Solve the task. You have tools: read_file, run_python.",
  "B": "Format: Thought: ...\nAction: tool(...)\nObservation: ...\nFinal: ...",
  "C": "...examples...\n" + "Format as above."
}
# 跑 3 × 5 = 15 次，记录通过数
```

**📊 关键观察指标**
- 三种 prompt 通过题数
- 平均轨迹长度

**💡 结论确认 & 进阶**
- ✓ B 通常比 A 好（格式约束减少 model 走偏）
- ✓ C 比 B 再好一点
- 延伸：把模型换成 0.5B 小模型，三种 prompt 差距会扩大还是缩小？

---

## 实验组合包推荐

### 周末 4 小时 · 入门组合（3 个）

适合："读完笔记一遍，想用 4 小时把核心直觉抓住"。

| # | 实验 | 收获 |
|---|---|---|
| 1.1 | 三家 BPE 同句对比 | 理解 tokenizer 不是中性的 |
| 2.3 | RoPE base 外推 | 看见"长上下文"的物理本质 |
| 3.4 | packing vs padding | 理解工程优化的实际价值 |

> 共耗时 ~ 1.5 小时实操 + 2.5 小时写笔记 / 思考。一台 Colab T4 即可。

---

### 周末 8 小时 · 中阶组合（6 个）

入门组合 + 下面 3 个：

| # | 实验 | 收获 |
|---|---|---|
| 2.1 | MHA / GQA / MLA KV cache | 量化 attention 进化路径 |
| 4.2 | LoRA r=8/16/64 | 决定自己微调时怎么选 rank |
| 4.4 | PPO vs GRPO 200 step | 理解为什么 GLM-5.1 用 GRPO |

> 共耗时 ~ 5 小时实操 + 3 小时分析。1×3090 推荐。

---

### 一周 · 完整组合（15 个核心实验）

适合："决心把整本笔记吃透，做一份自己的 lab notebook"。

| # | 实验 | 类别 |
|---|---|---|
| 1.1, 1.2 | tokenizer 速度 + 三家对比 | 数据 |
| 2.1, 2.2, 2.3, 2.5 | KV cache、MoE FLOPs、RoPE、Norm 速度 | 架构 |
| 3.1, 3.2, 3.4 | 去重、污染、packing | 数据 |
| 4.1, 4.2, 4.4 | base vs SFT、LoRA、GRPO | 训练 |
| 5.1, 5.2 | HumanEval+ vs base、LiveCodeBench 时间 gap | 评测 |
| 6.1, 6.3 | 量化对比、prefix caching | 部署 |
| 7.2 | Reflection agent | Agent |

> 建议节奏：周一-周二做数据 + 架构（5 个），周三-周四做训练 + 评测（5 个），周五做部署 + agent（5 个），周末写复盘。

---

## 实验册使用守则

1. **每个实验跑完写 3 行**：观察 / 是否符合预期 / 一个新问题。这是把"跑过的实验"变成"自己的知识"的关键。
2. **遇到数字不符**先怀疑环境（dtype / batch / library version），再怀疑结论。本册给的"应当看到"是粗略量级，机器差异 ±30% 都算正常。
3. **不要省略对照组**——A vs B 的核心是控制变量。如果你只跑了 A，那只是"摸了一下 A"，没有获得对比知识。
4. **优先复用现成模型与数据集**——本册刻意避免"先训练一个模型再…"。如果某个实验诱惑你"训得更大才好看"，请克制，把那个升级版留到笔记主线 Phase 里去做。

> 实验册到此结束。回到主线 Phase 时，建议把跑过的实验对应的章节边读边对照——你会发现很多原本"知道但没感觉"的结论，瞬间多了三分实感。
