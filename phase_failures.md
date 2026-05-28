# 📓 失败模式 cookbook · 出问题先翻这里

> 📅 主线快照：2026-05-28 · 上次核对：2026-05-28

> **怎么用**：每条失败模式给三件东西——(a) **症状**：你会看到什么；(b) **排查顺序**：按假设可能性从高到低排，依次核；(c) **修法**。先扫"症状"列表找最像的，再按排查顺序做减法。**绝大多数 LLM 训练 / 推理事故都是这 25 条里的一条**。

---

## 📑 按阶段索引

- **§A 训练通用**：A1 loss NaN · A2 loss 不降 · A3 OOM · A4 速度突然变慢 · A5 ckpt 加载报 shape
- **§B 数据 pipeline**：B1 评测集污染 · B2 token 压缩率异常 · B3 去重后语料过少 · B4 license 滑进训
- **§C 预训练 / MoE**：C1 expert collapse · C2 lr/loss 不匹配 · C3 MFU 上不去 · C4 FP8 训不收敛
- **§D SFT**：D1 chat template 串味 · D2 loss mask 错位 · D3 SFT 后续写不停 · D4 LoRA 看似收敛但生成差
- **§E RL**：E1 reward 卡 0 · E2 KL 爆炸 · E3 reward hacking · E4 entropy 塌
- **§F 评测**：F1 pass@k 算错 · F2 LiveCodeBench 涨但 SWE 掉 · F3 SWE-Bench 跑挂
- **§G 推理 / 部署**：G1 vLLM OOM · G2 prefix cache miss · G3 TTFT 飙高 · G4 tool_call 解析失败
- **§H Agent / 应用**：H1 context 爆炸 · H2 sandbox 超时 · H3 RAG 命中率低

---

## §A 训练通用

### A1 · loss 突然 NaN

**症状**：loss 训得好好的，第 N 步突然 `nan` / `inf`，grad_norm 同步爆。
**排查顺序**：
1. **是否 FP16**（不是 BF16）？FP16 mantissa 5 位，长训必 NaN。
2. **grad_clip 设了吗**？没设的话单个 outlier 直接捅穿。
3. **lr 是不是太大**？warmup 后第一个稳定段 lr > 5e-4 对 MoE 很危险。
4. **dataset 有没有空样本 / 超长样本**？一条 token=0 的样本 / 一条 token=1M 的样本都能炸。
5. **optimizer eps**？AdamW 默认 1e-8，FP16 下 underflow 必加大到 1e-6。
**修法**：换 BF16 + `grad_clip=1.0` + 检查 lr schedule + 加 NaN-skip（PyTorch `torch.nan_to_num` 在 backward 前）。**生产建议**：每 1000 step 保 ckpt，NaN 出现回滚一个 ckpt 重训。

### A2 · loss 不降（卡在某个值）

**症状**：训了 5B token loss 卡在 2.5 不动，理论上应该 < 2.0。
**排查顺序**：
1. **数据真的进对了吗**？打印 5 条 `tokenizer.decode(input_ids[0])` 人眼看一下。
2. **loss mask 写对了**？所有 token 算 loss 还是只算 assistant？
3. **lr schedule warmup 是不是漏了**？冷启动 lr 太高 grad explode 再被 clip 死。
4. **EMA / weight averaging 在干扰吗**？
5. **是不是 packing 没开**，80% 算力浪费在 padding 上？
**修法**：先 mask + decode 双重核对数据是真进训了；warmup 至少 2% steps；不开 packing 永远是错的。

### A3 · OOM（显存不够）

**症状**：第 K 步爆 OOM；或 batch_size 加到 2 就崩。
**排查顺序**：
1. **激活显存 = batch × seq × hidden × layers**，最先怀疑这。
2. **优化器状态**：AdamW FP32 占 8 bytes/param，70B 模型光优化器 560GB。换 Muon 或 8-bit Adam。
3. **梯度同样大小**，没开 grad checkpointing？
4. **KV cache**：长 seq 时巨大，看 `2·L·H·d·seq·bytes·B`。
5. **碎片化**：长训后 PyTorch 显存碎片化 1-2GB，`torch.cuda.empty_cache()`。
**修法**：grad checkpointing 默认开；ZeRO-3 切优化器 + 梯度 + 参数；最后再降 batch_size。**别先降 batch**，那是治标。

### A4 · 训练速度突然变慢

**症状**：前 1000 step 1000 tok/s/GPU，后突然降到 200 tok/s/GPU。
**排查顺序**：
1. **温墙**：`nvidia-smi -q -d TEMPERATURE`，> 85°C 就降频。`-pl 350` 降功率。
2. **磁盘 I/O**：HF dataset 流式读 + 慢 SSD → 数据加载成瓶颈。`prefetch_factor=4` + NVMe。
3. **`empty_cache` 在 hot loop 里**？这会同步整个 stream，巨慢。
4. **CPU 数据预处理变重**？换 num_workers 看变化。
5. **NCCL 通信**：多机训时如果某节点网络降级，所有节点等它。
**修法**：先看温度和 GPU util，util 不到 95% 多半是数据加载或通信。

### A5 · ckpt 加载报 shape mismatch

**症状**：`size mismatch for embed_tokens.weight: ...`。
**排查顺序**：
1. **vocab_size 改过吗**？添加 special token 没同步 base ckpt。
2. **LoRA rank 不一样**？load 了 r=64 的 LoRA 到 r=32 的配置。
3. **`tensor_parallel_size` 改了**？TP=4 训的 ckpt load 到 TP=2 会乱。
4. **HF transformers 版本**？某些版本 GLM 的 q_lora 维度有 patch。
**修法**：先 `print(model.state_dict().keys())` 对比 ckpt keys；`strict=False` 是绷带不是修复。

---

## §B 数据 pipeline

### B1 · 评测集污染

**症状**：自训 SFT 后 HumanEval+ 跳 + 20pp 看起来太好；或 LiveCodeBench 时间窗外分数没动。
**排查顺序**：
1. **10-gram exact match** 跑过吗？跑！
2. 题面 vs **题解** 双向扫了吗？很多 dataset 收录了解答，不只是 prompt。
3. 用的 SFT 数据是不是 HuggingFace 上的"已知有污染"集合（如某些 Magicoder 早期版本）？
**修法**：先回 phase1 §C 跑去污染脚本；评测时永远跑时间窗外的 LiveCodeBench 当 OOD baseline。

### B2 · tokenizer 压缩率异常

**症状**：你的代码 `tokens/char` ≈ 0.6（中文文本水平），正常应该 ≤ 0.3。
**排查顺序**：
1. tokenizer 用的是不是 base 模型自带的？换成 `Qwen3-Coder` / `GLM-4.5` 自带 BBPE。
2. 训了自己的 tokenizer 但 vocab 太小（< 64k）？
**修法**：复用主流 tokenizer 是默认正确选择。

### B3 · MinHash 去重后语料只剩 < 30%

**症状**：100GB 输入，dedup 后只剩 25GB；明显过度去重。
**排查顺序**：
1. **阈值太低**：threshold=0.6 太宽，应该 0.8。
2. **num_perm 太小**：32 个 perm 在大语料上误杀率高，128 起步。
3. **shingle 大小**：5-gram 抓得太严，10-gram 更合理。
**修法**：threshold=0.85 / num_perm=128 / 5-shingle for MinHash + 10-gram for decontam（两套阈值不同）。

### B4 · license 滑进训练集

**症状**：用户调出一段 GPL 代码，律师问"这从哪来的"。
**排查顺序**：
1. **The Stack v2** 默认含 GPL/AGPL/LGPL，需要主动 filter。
2. 公司内部代码混进 The Stack？用 `repo_path` filter。
3. PR 抓取时分支 fork from GPL 项目？
**修法**：phase1 数据 pipeline 第一步就是 license filter，过 ≥ 3 个 permissive license（MIT/Apache/BSD-3）。

---

## §C 预训练 / MoE

### C1 · Expert collapse（少数 expert 满载，其余闲置）

**症状**：训 1B token 后 `expert_load_var > 1.0`；HuggingFace `router_logits` 看 90% 流量去 8 个 expert（共 256 个）。
**排查顺序**：
1. **aux_loss 系数太小**：< 0.001 时 router 不收敛。
2. **aux-loss-free bias** 没设 / bias 更新太慢。
3. 数据偏一边（全是 Python，没 C++/Java）路由确实应该不均，但 var > 0.5 还是病。
**修法**：上 aux-loss-free routing（DeepSeek-V3 风格）+ `expert_load_var` 进训练 dashboard 第一行。监控比 loss 更早预警。

### C2 · lr / loss 不匹配（loss 健康但 grad 异常）

**症状**：loss 一直降，但 grad_norm 看起来要么 < 1e-4（更新基本是 0）要么 > 100（更新乱跳）。
**排查顺序**：
1. lr 是否经过 scaling rule（`base_lr × sqrt(global_bs / 256)`）？
2. 优化器是 Muon 但 lr 还按 AdamW 标准（5e-4）？Muon 通常要砍到 1e-4 量级。
3. layer-wise lr decay 没设？
**修法**：Muon 起步 1e-4，AdamW 起步 3e-4，先在 100M token 上看 grad_norm 调到 0.1-1.0 量级再正式开训。

### C3 · MFU 上不去（停在 25%）

**症状**：8×H100 MoE 1B 训练，nvidia-smi util 95% 但 MFU 只有 22%。
**排查顺序**：
1. **EP all-to-all 通信开销**：MoE 训 EP 占 30%+ wall，看是否网络瓶颈。
2. **激活重计算**：grad checkpointing 开了，是 trade 算力换显存。
3. **FP8 没开**：FP8 vs BF16 MFU 提升 1.5-2×。
4. **batch_size 太小**：MoE 喜欢大 batch，per_device < 4 利用率低。
**修法**：先升 batch 再看通信；FP8 必须配 Hopper / Blackwell；网络 < 200Gb/s 时 EP=8 不要硬上。

### C4 · FP8 训不收敛

**症状**：BF16 训得好的配方换 FP8 后 loss 不降 / 不稳。
**排查顺序**：
1. **FP8 scale 不对**：E4M3 适合 weight + activation，E5M2 适合 grad（范围大）。混了会爆。
2. **lr 没调**：FP8 lr 一般要砍 30%。
3. **outlier / spike 没处理**：FP8 量化对 outlier 极敏感，必须 absmax 或 delayed scale。
**修法**：用 transformer_engine / torchao 的成熟 FP8 配方，不要自己撸量化逻辑。

---

## §D SFT

### D1 · Chat template 串味

**症状**：SFT 后模型输出包含 `<|im_start|>` / `<|tool_call|>` 等明文（应该被 tokenizer 当 special token 吞掉）。
**排查顺序**：
1. **你训的 base 是 GLM 但 template 用了 ChatML**？相当于 SFT 阶段在教模型"输出这些字符串"。
2. tokenizer 没加 special token？`add_special_tokens(["<|tool_call|>"])`。
3. inference 时 `apply_chat_template` 用错版本？
**修法**：永远用 `tokenizer.apply_chat_template(..., tokenize=False)` 渲染训练字符串 + `return_assistant_tokens_mask=True` 字节级核对；不要自己拼字符串。

### D2 · loss mask 错位

**症状**：训完模型在 user 消息位置生成、不在 assistant 位置生成；或反过来。
**排查顺序**：
1. mask 是 1-based 还是 0-based 对齐 token_ids？
2. `shift_labels = labels[1:]` 漏了？loss 算到 next token 时 mask 也要对应 shift。
3. **packing 模式下** position_ids reset 但 loss mask 没 reset？
**修法**：写一个可视化函数：取 1 条样本，把 token + 是否算 loss 用 ANSI 颜色打印出来；不通过这一关不开训。

### D3 · SFT 后模型不停止（续写）

**症状**：问"1+1=?"，模型回"2\n\n# 接着出新题：2+2=?\n..."一直续写。
**排查顺序**：
1. SFT 数据中 `assistant` 段是否以 `<|im_end|>` (或对应 stop token) 结尾？
2. tokenizer 的 `eos_token_id` 是否在 stop list 里？
3. 推理时 `stop_token_ids` 配了吗？
**修法**：SFT 数据 100% 校验 assistant 段末尾必须是模型对应的 eos；inference 时 stop list 显式列出。

### D4 · LoRA 看似收敛但生成差

**症状**：train loss 0.4 → 0.2 漂亮地下降，但 HumanEval+ vs base 0pp 甚至 -2pp。
**排查顺序**：
1. **LoRA target_modules 太少**：只挂 q/k/v_proj 而不挂 gate/up/down_proj。
2. lora_alpha / lora_rank 配比错了（alpha < rank 实际学习率被压缩）。
3. **train loss 是不是只看 OSS-Instruct 这类格式化数据，没看真实 PR**？两种数据 loss 量级差很大。
**修法**：`target_modules="all-linear"` 是 2026 默认；分 split 看 loss（OSS-Instruct / Issue-PR / agent traj 各报一条曲线）。

---

## §E RL

### E1 · GRPO reward 一直为 0

**症状**：跑 50 step reward 均值还是 0.02，比初始还低。
**排查顺序**：
1. **SFT cold start 是不是没做**？base 模型没见过 tool_call 格式，rollout 100% 失败。
2. **sandbox 是不是挂了**？pytest 跑挂、docker 启动超时，reward 永远 0。
3. **reward 函数实现 bug**：检查 `f2p_pass == f2p_total` 的边界（0/0 是不是被当 fail？）。
4. **temperature 太低**（0.2）→ rollout 没多样性 → advantage 全 0 → 不更新。
**修法**：先用 100 条 SFT 模型 rollout 手动检查 reward 分布；temperature 至少 0.7 给 GRPO 多样性；先做 cold start。

### E2 · KL 散度爆炸

**症状**：训 30 step KL > 10，policy 跑出 SFT 分布很远，生成质量崩。
**排查顺序**：
1. **KL beta 太小**（< 0.01）。
2. **clip ratio 设太大** (> 0.3) 允许大步更新。
3. **reward 量级太大**（如未归一化）淹没 KL 项。
4. **GRPO 没做 group z-score** advantage，方差爆。
**修法**：beta 从 0.04 起步，reward 必须归一化（z-score within group），clip 0.2，监控 KL 设硬阈值 5（超就 early stop）。

### E3 · Reward hacking（模型刷捷径）

**症状**：reward 持续涨，但 holdout / SWE-Bench Verified 评测分掉。
**排查顺序**：
1. **测试覆盖率不够**：模型发现 "改测试" 比 "修代码" 容易，跳过修。
2. **奖励里没 anti-hack 项**（test_modified / skip_added 没罚）。
3. **reward 信号包含 surface feature**（如代码长度，模型学到写长一点就过）。
**修法**：anti-hack reward `-2.0`（重于 sparse +1.0）；保留只读 reference 测试在 sandbox 外验证；每 100 step 跑 holdout，分数掉立刻停。

### E4 · Entropy 塌（模型生成只有一种风格）

**症状**：rollout 多样性消失，G 条采样全长一样，advantage 全 0，训练停滞。
**排查顺序**：
1. **temperature 太低**。
2. **entropy coef** 加到 loss 了吗？
3. KL 约束太紧也会导致探索消失（与 E2 反向）。
**修法**：entropy bonus = 0.005，temperature ≥ 0.7，G ≥ 8。

---

## §F 评测

### F1 · pass@k 算错

**症状**：报告 pass@10 = 75%，但实际 n=10 / c=5，按公式应该是 ~ 100% 中 5 个，不是 75%。
**排查顺序**：
1. 用 `c/k` 直接近似而不是 unbiased estimate？
2. n != k 时混用？
**修法**：永远用 `1 - C(n-c, k) / C(n, k)`，evalplus 的实现是参考。

### F2 · 公开 benchmark 涨但内部 SWE-Bench 掉

**症状**：HumanEval+ +5pp / LiveCodeBench +3pp，内部 v0 -8pp。
**排查顺序**：
1. **训练数据偏 short-fn**（OSS-Instruct 多），长 horizon 能力退化。
2. **mid-training 后 chat 能力滑**：通用对话比例太低（< 10%）。
3. **agent 轨迹比例不够**（< 15%）。
**修法**：混比 review = OSS 50% + Issue-PR 30% + 通用对话 10% + agent 轨迹 10%；不是无脑加 OSS-Instruct 就好。

### F3 · SWE-Bench harness 跑挂

**症状**：50 题 30 题 docker 启动报 `image not found` 或 `OOM` 或 `network unreachable`。
**排查顺序**：
1. **docker base image** 没预拉？SWE-Bench 每个 instance 一个镜像，必须先 batch pull。
2. **docker memory limit** 太低（默认 2GB）？大 repo 跑测试可能要 4-8GB。
3. **sandbox 网络** 没禁？某些 pytest 试图联网，应该跑 `--network none`。
4. **pytest timeout** 不够？600s 是经验值，复杂 repo 可能要 1200s。
**修法**：SWE-Bench 官方 README 第二节是这套坑的清单；照抄。

---

## §G 推理 / 部署

### G1 · vLLM OOM 启动失败

**症状**：`--quantization fp8 --tp 8` 启 9B 模型，vLLM 启动时 OOM。
**排查顺序**：
1. `--gpu-memory-utilization` 默认 0.9，对长 ctx 不够，调到 0.85 留 buffer 给 KV cache。
2. `--max-model-len` 设没设？默认会 allocate 整个原生 ctx 的 KV cache。
3. 其它进程占显存？`nvidia-smi` 看一下。
4. 模型本身 weight 太大（30B BF16 = 60GB），需要 quantization 或更多卡。
**修法**：`--max-model-len 32768` 显式限上限；`--gpu-memory-utilization 0.85`；用 `--enforce-eager` 排除 CUDA graph 占用。

### G2 · prefix cache miss rate 太高

**症状**：bench 显示 prefix cache hit < 30%，agent 场景同 system prompt 应该 > 80%。
**排查顺序**：
1. **LoRA adapter 不同**会让 cache key 不一样，hit = 0。
2. **system prompt 里有时间戳**（`Now: 2026-05-28 14:32:01`），每次 prompt 变 cache miss。
3. **`--enable-prefix-caching` 真的开了**吗？
**修法**：合并 LoRA 到 base；system prompt 去掉时间戳 / session_id 等 per-request 字段；vLLM 0.6+ 默认开 prefix caching，但要核实。

### G3 · TTFT 飙到 2s+

**症状**：bench TTFT p95 从 400ms 涨到 2000ms。
**排查顺序**：
1. **batch 满了**：max_running_requests 设太高，请求排队。
2. **prompt 太长**：> 32K prompt 时 prefill 本身就要 1s+。
3. **EP all-to-all 拥塞**（MoE 模型）。
4. **prefix cache 失效**（见 G2）。
**修法**：调 `--max-running-requests` 到合理值（H100 8×TP，9B 模型起步 64）；> 64K prompt 必开 chunked prefill。

### G4 · tool_call JSON 解析失败

**症状**：模型输出 `{"name": "bash", "arguments": {...}` 缺右括号；或 `arguments` 不是合法 JSON。
**排查顺序**：
1. **没开 constrained decoding** (`guided_json`)。
2. SFT 数据中 tool_call 格式不一致（不同样本用了不同 schema）。
3. max_tokens 砍了输出，JSON 被截断。
**修法**：vLLM / SGLang 的 `guided_json` / `response_format=json_schema` 强制约束；预留 max_tokens > 平均 tool_call 长度 2×。

---

## §H Agent / 应用

### H1 · Context 爆炸（agent 跑 5 轮后 OOM）

**症状**：mini_agent 跑到 step 5，prompt 长度 > 100K，超 model max_ctx。
**排查顺序**：
1. **tool 返回值太大**：`read_file` 一次读 5000 行 = 30K token。
2. **历史 (think, tool_call, observation) 全保留**没摘要？
3. tool_call retry 了 N 次，每次都进 history？
**修法**：tool 输出强制 line range + 默认 100 行；auto-compact threshold = 80% ctx；retry > 3 次的 tool_call 摘要后只保留最后一次。

### H2 · Sandbox 超时

**症状**：sandbox 每条 episode 跑 > 5 min；rollout 时间杀手。
**排查顺序**：
1. **docker 启动慢**：每次新 container 拉镜像 + apt-install，应该 warm pool。
2. **pytest 跑全量测试**？只跑 F2P / P2P 即可。
3. **网络等待**：sandbox 试图联网（pip install）卡 30s。
**修法**：warm container pool 几百个常驻；pytest 只跑相关测试；sandbox 必须 `--network none` 严禁外网。

### H3 · RAG Recall@5 < 50%

**症状**：手写 50 条 query 测，top-5 命中真正相关文档不到 25 个。
**排查顺序**：
1. **chunking 用按行而不是按 AST**？function 被拦腰斩。
2. **embedding 模型选错**：用了通用 bge-large 而不是 bge-**code**-v1。
3. **没用 reranker**：dense 召回 top-100，最相关常排到 30+，没 rerank 进不了 top-5。
4. **BM25 没和 dense hybrid**：纯语义检索对 identifier 名字差的查询差。
**修法**：tree-sitter AST 切块；bge-code-v1 embed；hybrid (dense 50 + BM25 50) → bge-reranker-v2 → top 5。Recall@5 应该 ≥ 80%。

---

## 📌 章末检查

**带走这 5 条**
- **NaN / OOM / 速度变慢** 是训练三大常见事故；**reward 卡 0 / KL 爆 / hacking** 是 RL 三大；**vLLM OOM / prefix miss / JSON 解析** 是部署三大。
- 99% 的 bug 都在数据 / 配置层，不在模型本身——**先打印 5 条样本** 永远是第一步。
- 监控比 debug 重要：`grad_norm` / `expert_load_var` / `KL` / `reward 分布` / `prefix hit rate` 五个数字常驻 dashboard，能提前 1-2 小时预警。
- **可视化 loss mask + chat template** 渲染一次，绝大多数 SFT 翻车都来自这一步没核对。
- 遇到陌生症状先来这页扫一遍——比直接 Google 快 10×，因为这里的排查顺序按概率排了。

**自检 3 题**
1. 你的 RL 训练 reward 100 step 都是 0，你按什么顺序排查？
2. SFT 后模型不停止续写，最可能是哪两件事没做对？
3. vLLM 部署 9B 模型起来后 TTFT 1.5s（正常 400ms），怀疑哪几样？

<details><summary>参考答案</summary>

1. (a) cold start 做了没（SFT 教过 schema 吗）→ (b) sandbox 跑得通吗（手动 reward 一条样本）→ (c) reward 函数边界 bug（0/0 的处理）→ (d) temperature 给到 0.7+ 让 rollout 有多样性。
2. (a) SFT 数据 assistant 段没以 eos token 结尾 + (b) inference 时 stop_token_ids 没配。
3. (a) 是不是 batch 满了请求排队 → (b) prompt 长度 > 32K → (c) prefix cache 失效（LoRA adapter 不同 / system prompt 含时间戳）→ (d) MoE EP 拥塞。
</details>

> ⚠️ **元 pitfall** · 看到症状先 Google / 看错就慌——**先来这页 30 秒扫一眼**。失败模式的 80% 集中在 25 条里，跳着排查比从随机网帖里抄答案快 5-10×。

**下一步** · 跑 [✪ capstone](./phase_capstone.md) 时随手翻这页 · 把每次踩的新坑 PR 加进来 · 术语速查 → [▣ glossary](./phase_glossary.md)。
