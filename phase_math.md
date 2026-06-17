# 🧮 公式集中页 · LLM Coding 全栈数学速查

> 📅 主线快照：2026-05-28 · 上次核对：2026-05-28

> **⚡ 三句话要点**
> 1. AI 研究者最高频查的不是论文也不是代码，是**那一个公式具体怎么写**——本页把全书 12 个核心公式集中一处，KaTeX 渲染，每条带"哪个 phase 用到 + 推导链接"。
> 2. 公式 ≠ 公式背诵——所有条目都包含**形式 + 一句直觉 + 主要超参的默认值**，遇到工程问题能马上反查"我这步该填什么"。
> 3. 三大类：**架构** (RMSNorm / RoPE / YaRN / MLA / DSA) · **训练目标** (NTP / FIM / MTP / pass@k) · **后训练** (PPO / DPO / GRPO / KL approx)。

---

## §A 架构层公式

### A1 · RMSNorm（替代 LayerNorm，2019）

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \varepsilon}} \odot \gamma
$$

- **直觉**：去掉 LayerNorm 的 `μ` 项，只用均方根归一。比 LayerNorm 快 10-20%、效果相当。
- **超参**：`ε = 1e-6`（GLM/Llama 系）；`γ` 可学。
- **用到**：phase2 §架构演进、所有现代 dense + MoE。

### A2 · RoPE 标准式（Rotary Position Embedding，2021）

对每对维度 $(2i, 2i+1)$，位置 $m$ 处：

$$
\begin{pmatrix} q'_{m,2i} \\ q'_{m,2i+1} \end{pmatrix} =
\begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix}
\begin{pmatrix} q_{m,2i} \\ q_{m,2i+1} \end{pmatrix},
\quad \theta_i = \mathrm{base}^{-2i/d}
$$

- **直觉**：把 query/key 的每对维度旋转一个由位置决定的角度，相对位置 = 旋转角度差，attention `<q, k>` 自然 encode 相对位置。
- **超参**：`base = 10000`（原始）/ `500000`（Llama-3 起增大底数为长上下文做准备）；`d_head` 通常 64-128。
- **用到**：phase2 §1.6、phase3 §4。

### A3 · YaRN 缩放（Yet another RoPE extensioN，2023）

把每个维度按其 wavelength 分成三档：

$$
\theta'_i = \begin{cases}
\theta_i / s & \lambda_i < L_{\text{train}} \cdot \alpha \quad \text{(高频外推)} \\
\theta_i / (s \cdot \text{ramp}(...))& \alpha \le \lambda_i / L_{\text{train}} \le \beta \quad \text{(混合)} \\
\theta_i & \lambda_i > L_{\text{train}} \cdot \beta \quad \text{(低频插值)}
\end{cases}
$$

并对 attention 温度做缩放：$\text{attn-temp} = 0.1 \ln s + 1$，其中 $s = L_{\text{new}} / L_{\text{train}}$。

- **直觉**：长波长（低频）维度插值不外推、短波长（高频）维度外推不插值——分别保留绝对位置和局部模式。attention temperature 补偿"长 prompt softmax 熵变化"。
- **超参**：`s = 4` (32K → 128K) · `α = 1` · `β = 32` · `factor = s`。
- **用到**：phase3 §4.5、phase_consumer §3.4。

### A4 · MLA · Multi-head Latent Attention（DeepSeek-V2，2024）

把 K/V 压缩到一个低维 latent：

$$
c_t^{KV} = W^{DKV} h_t, \quad k_t^C = W^{UK} c_t^{KV}, \quad v_t^C = W^{UV} c_t^{KV}
$$

KV cache 只存 $c_t^{KV} \in \mathbb{R}^{r_{kv}}$（GLM-5.2: $r_{kv}=512$），推理时再 up-project。Q 也单独压一次到 $r_q$（GLM-5.2: $r_q=2048$）。

- **直觉**：把 H 个头共享的低秩信息显式分解出来，KV cache 缩 ~10×（vs MHA），同时不像 MQA 那样掉精度。
- **超参**：q_lora_rank 通常 d/2-d/4；kv_lora_rank 通常 d/8-d/16；RoPE 在 latent 还是 expanded 上施加是一个工程选择（GLM/DeepSeek 在 latent + extra rotary dim 上）。
- **用到**：phase2 §1.5、phase0 §1。

### A5 · DSA · DeepSeek Sparse Attention（V3.2 / GLM-5.2，2025-2026）

两阶段：

$$
\text{idx}(q_t) = \text{LightningIndexer}(q_t, k_{1:t}) \in [t], \quad |\text{idx}| = k_{\text{top}}
$$
$$
\text{Attn}(q_t, K, V) = \sum_{j \in \text{idx}(q_t)} \alpha_{tj} v_j, \quad \alpha = \mathrm{softmax}(q_t k_j^\top / \sqrt{d})
$$

- **直觉**：Lightning Indexer（轻量小网络）先给每个 query 挑 top-k 个 KV 位置，主 attention 只在 top-k 上算。把 attention 从 O(L²) 降到 O(L·k_top + L·indexer)。
- **超参**：`k_top = 64-256`（GLM-5.2 推测） · indexer 是 4-8 head 的 mini-attention。
- **用到**：phase2 §1.4、phase0 §1.2。

---

## §B 训练目标公式

### B1 · NTP · Next-Token Prediction Loss

$$
\mathcal{L}_{\text{NTP}} = -\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})
$$

PyTorch 实现：

```python
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[...,  1:    ].contiguous()
loss = F.cross_entropy(shift_logits.view(-1, V), shift_labels.view(-1))
```

- **直觉**：用前 $T-1$ token 预测后 $T-1$ token；末尾 token 没"下一个"，丢掉。
- **用到**：phase_basics §4、phase2 §3.1。

### B2 · FIM · Fill-in-the-Middle（OpenAI，2022）

输入重排为 `<PRE> prefix <SUF> suffix <MID>`，loss 只在 mid 段算：

$$
\mathcal{L}_{\text{FIM}} = -\sum_{t \in \text{mid}} \log p_\theta(x_t \mid x_{<t}^{\text{rearr}})
$$

- **直觉**：训练模型"补中间"而不是只"续写右边"。代码补全场景必备（编辑光标常在文件中间）。
- **超参**：FIM 占比 50%（OpenAI / StarCoder），split 点随机 1/3 ~ 2/3。
- **用到**：phase2 §3.2。

### B3 · MTP · Multi-Token Prediction（DeepSeek-V3 / GLM-5.2）

主头预测 $x_{t+1}$，多个 MTP head 预测 $x_{t+2}, x_{t+3}, ...$：

$$
\mathcal{L}_{\text{MTP}} = \mathcal{L}_{\text{NTP}} + \lambda \sum_{k=1}^{K} \mathcal{L}_{\text{NTP}}^{(k)}
$$

每个 MTP head 是"主干倒数第二层 hidden + 真实 $x_{t+k}$ embedding → 小 Transformer block → predict $x_{t+k+1}$"。

- **直觉**：训练时多任务（同时预测 +1/+2/+3 token），推理时把 MTP head 当 draft 做 speculative decoding，加速 1.5-2×。
- **超参**：`K = 2` · `λ = 0.3` (DeepSeek-V3)。
- **用到**：phase2 §3.3、phase7 §spec decoding。

### B4 · pass@k unbiased estimate（Codex 论文，2021）

$$
\text{pass@}k = \mathbb{E}_\text{problems}\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right]
$$

其中 $n$ = 总采样数、$c$ = 通过单测的样本数、$k$ = 报告的 k。

- **直觉**：从 $n$ 个采样中随机抽 $k$ 个，至少 1 个通过的概率。直接用 $c/k$ 在 $k$ 接近 $n$ 时会高估。
- **超参**：常用 `n=20, k ∈ {1, 5, 10}`。
- **用到**：phase6 §3.1。

---

## §C 后训练 / RL 公式

### C1 · PPO Clipped Surrogate（InstructGPT，2022）

$$
\mathcal{J}_{\text{PPO}}(\theta) = \mathbb{E}_t\Big[
\min\big(\rho_t \hat A_t,\ \mathrm{clip}(\rho_t, 1-\varepsilon, 1+\varepsilon) \hat A_t \big)
\Big] - \beta\, \mathbb{E}_t \big[\mathrm{KL}(\pi_\theta \| \pi_{\text{ref}}) \big]
$$

其中 $\rho_t = \pi_\theta(a_t|s_t) / \pi_{\text{old}}(a_t|s_t)$，$\hat A_t$ 是 GAE 估计的 advantage。

- **直觉**：朝高 advantage 方向更新策略，但单 step 走得太远（$\rho > 1+\varepsilon$）被 clip 截住；KL 项额外拉回 reference。
- **超参**：`clip ε = 0.2`，`KL β = 0.01-0.05`。
- **用到**：phase5 §3.1。

### C2 · DPO 闭式损失（Direct Preference Optimization，2023）

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma\Big(
\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} -
\beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}
\Big)
$$

其中 $(y_w, y_l)$ 是人类偏好对（winner / loser）。

- **直觉**：用 Bradley-Terry 模型 + KL 正则的最优策略闭式解，**不需要 RM、不需要 rollout**，但只能学 SFT 已覆盖的分布。
- **超参**：`β = 0.1-0.5`（比 PPO 的 KL β 大）。
- **用到**：phase5 §3.2。

### C3 · GRPO Group-Relative Advantage（DeepSeekMath，2024 + R1，2025）

对同一 prompt $x$ 采样 $G$ 个回答 $\{y_1, ..., y_G\}$，得到 $G$ 个 reward $\{r_1, ..., r_G\}$。advantage：

$$
\hat A_i = \frac{r_i - \mathrm{mean}(r_{1:G})}{\mathrm{std}(r_{1:G}) + \varepsilon}
$$

策略损失（per token）：

$$
\mathcal{J}_{\text{GRPO}} = \mathbb{E}_x \frac{1}{G} \sum_i \mathbb{E}_t \big[ \min(\rho_t^{(i)} \hat A_i, \mathrm{clip}(\rho_t^{(i)}, 1-\varepsilon, 1+\varepsilon) \hat A_i) \big] - \beta \mathrm{KL}(\pi_\theta \| \pi_{\text{ref}})
$$

- **直觉**：扔掉 critic（不学 value head），用"组内 z-score"做 advantage 的免学习 baseline。显存减半，收敛更稳。
- **超参**：`G = 8-16` · `clip ε = 0.2` · `β = 0.04` · KL approx 用下式。
- **用到**：phase5 §3.3、capstone step 12。

### C4 · 低偏 KL 散度估计（Schulman approx，GRPO 用法）

精确 KL 在 token-level 难算，用：

$$
\widehat{\mathrm{KL}}(\pi_\theta \| \pi_{\text{ref}}) \approx \mathbb{E}_t \big[ r_t - \log r_t - 1 \big], \quad r_t = \frac{\pi_{\text{ref}}(a_t|s_t)}{\pi_\theta(a_t|s_t)}
$$

- **直觉**：$r - \log r - 1 \ge 0$ 且无偏方差小，比 `log π_θ - log π_ref` 直接估更稳。
- **用到**：phase5 §3.3、所有 GRPO 实现。

### C5 · RLVR Reward 一般形式（DeepSeek-R1）

$$
r(x, y) = r_{\text{verifier}}(y) + \sum_k \lambda_k\, r_k^{\text{shape}}(y)
$$

其中 $r_{\text{verifier}} \in \{0, 1\}$ 来自单测 / 编译 / 形式化验证，$r_k^{\text{shape}}$ 是格式 / 长度 / 子目标等 shaping 信号（小权重）。

- **直觉**：sparse 主信号（单测通过）+ dense 辅信号（diff touch / lint 通过）+ anti-hack 负信号（-2.0 罚改测试）。
- **用到**：phase5 §4、capstone step 11。

### C6 · Muon 优化器思路（2024）

对每个矩阵参数 $W$，先算梯度 $G$ 的 Newton-Schulz 近似正交化 $\tilde G$，再更新：

$$
W_{t+1} = W_t - \eta \cdot \tilde G_t, \quad \tilde G \approx \text{closest-orthogonal}(G)
$$

- **直觉**：Adam 是"每个标量参数学到自己的 lr"，Muon 是"每个矩阵的更新方向被正交化"。理论上对矩阵参数更合理，实测 MoE 上 step 数省 15-30%。
- **超参**：lr 通常砍到 AdamW 的 1/3（如 AdamW 用 3e-4 → Muon 用 1e-4）；embedding / LayerNorm 仍用 AdamW。
- **用到**：phase0 §1、phase2 §训练目标。

---

## 📎 一页对照表

| 公式 | 哪里用 | 关键超参 |
|---|---|---|
| RMSNorm | phase2 全员 | ε=1e-6 |
| RoPE | phase2/3 | base=10k / 500k |
| YaRN | phase3 §4.5 | s=4, α=1, β=32 |
| MLA | phase2 §1.5 | q_lora=2048 / kv_lora=512 |
| DSA | phase2 §1.4 | k_top=64-256 |
| NTP | phase2 §3.1 | — |
| FIM | phase2 §3.2 | 占比 50%，1/3-2/3 split |
| MTP | phase2 §3.3 | K=2, λ=0.3 |
| pass@k | phase6 §3.1 | n=20, k∈{1,5,10} |
| PPO | phase5 §3.1 | clip=0.2, β_KL=0.01-0.05 |
| DPO | phase5 §3.2 | β=0.1-0.5 |
| GRPO | phase5 §3.3 | G=8-16, clip=0.2, β=0.04 |
| KL approx | phase5 §3.3 | — |
| RLVR reward | phase5 §4 | anti-hack=-2.0 |
| Muon | phase2 / 0 | lr ≈ AdamW × 1/3 |

---

## 📌 章末检查

**带走这 5 条**
- 这些公式不需要"会推导"，但需要**会写代码**——每条都有等价 PyTorch 几行实现。
- 公式 + 超参缺一不可——`clip=0.2` / `KL β=0.04` / `G=8` 是 GRPO 默认值，**直接抄不会错**。
- KaTeX 渲染依赖网络——离线 PDF 会渲染失败，要用 print stylesheet 的备用方案。
- **Muon vs AdamW** 在 lr 量级上差 ~3×，换优化器忘换 lr 是常见崩溃源。
- **MLA 的 RoPE 应用位置**（latent vs expanded）是论文不会写得很清楚的工程细节——参考 DeepSeek-V3 / GLM-5.2 的开源 ref impl。

**自检 3 题**
1. 默写 GRPO 的 advantage 公式 + 说出 `G=8` vs `G=4` 的区别。
2. RoPE base 从 10000 调到 500000，物理意义是什么？什么时候要这样调？
3. PPO 和 GRPO 在算 KL 时都用 Schulman approx，**它的形式**是什么、为什么不直接用 `log π_θ - log π_ref`？

<details><summary>参考答案</summary>

1. $\hat A_i = (r_i - \mu) / (\sigma + \varepsilon)$，$\mu, \sigma$ 在同 prompt 的 G 个 sample 内统计。`G=8` reward 方差降到 ~ $\sigma^2/8$ 但显存翻 ~8×；`G=4` 显存友好但 advantage 噪声大。**默认 G=8 起步**，单卡 RL 用 G=4。
2. RoPE base 决定 $\theta_i$ 的指数衰减速度，base 越大、低频维度的波长越长，**模型在外推时不易混淆远距离位置**。base 10k → 500k 是 Llama-3 用来支持 8K → 32K+ 长上下文的"零成本预扩展"。一般在**训练前**改，**不在训练后**改（会破坏已学位置感）。
3. $\widehat{\mathrm{KL}} \approx r - \log r - 1$，其中 $r = \pi_{\text{ref}} / \pi_\theta$。直接用 `log π_θ - log π_ref` 是无偏但**方差大且可能为负**（KL ≥ 0 应该满足）；Schulman approx **永远非负 + 方差更小**，工程上更稳。
</details>

> ⚠️ **常见坑** · 看到论文公式直接 copy 实现忘了 ε 项——`std(reward) + 1e-8` 不加在 RL early stage（reward 全 0 时 std=0）直接 NaN。每个除法分母都加 ε 是工程习惯。

**下一步** · 看 [phase5 §3](./phase5_rl.md) 完整算法推导 · [phase_failures §E](./phase_failures.md) 看 RL 出问题怎么排查 · [▣ phase_glossary](./phase_glossary.md) 术语速查。
