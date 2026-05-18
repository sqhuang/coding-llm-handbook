# `capstone_runtime/` — 拷贝即可用的实验骨架

> 配套实验册（含每步思考题）：[`../phase_capstone.md`](../phase_capstone.md)。
> 把整个 `capstone_runtime/` 目录 `tar` 起来传到目标机就是一份自洽的工程脚手架——
> 不要再去找散在仓库其他地方的脚本，必要文件全在这里（`lib/` 里有从 `examples/` 拷过来的副本）。

## 红线：什么"已 verify"，什么"需要你在目标机 verify"

我只在一台 **macOS arm64 / 36GB RAM / 无 GPU** 上做过端到端验证。**我不能保证它在你的 H100 集群上原样跑通**——你的 CUDA / 驱动版本、网络拓扑、HF 镜像、私有 repo 结构都可能有差异。下表说清状态：

| 文件 / 命令 | 在我这边 | 在你的 H100 box | 需要你 verify 的事 |
|---|---|---|---|
| `make preflight` | ✅ 跑通 | 应该跑通 | torch + GPU 探测是否准确 |
| `make test` (`tests/`，19 case) | ✅ 全过 | 应该全过 | nothing；CPU only |
| `make step-01` (mini 3-task baseline) | ✅ graceful exit no-key | ✅ 应能跑（任意 OpenAI 兼容 endpoint） | API key 设置 + 你的 endpoint 真存在 |
| `make step-04` (decontam) | ✅ 已跑通 ~10MB 合成数据 | ✅ 应能跑大数据 | 大 dataset 的 RAM / 时间 |
| `make step-11` (reward self-check) | ✅ 4 case 全 pass | ✅ 应能跑 | nothing；纯 Python |
| `make step-19` (retro 渲染) | ✅ RETRO.md 已生成 | ✅ 应能跑 | nothing；纯 Python |
| `make step-02` (内部 bench scaffold) | ✅ sample 写出 | ✅ 应能跑 | git/docker 已装；你的 repo 的 GH_TOKEN |
| `make step-03` (datatrove pipeline) | ⚠️ **没** 跑（要下 50GB） | ❓ 必须验 | datatrove 装得上；HF token 可用 |
| `make step-05` (vLLM 起 Air-Base) | ⚠️ Mac 无 GPU | ❓ 必须验 | vllm + FP8 + 8×H100 + 200GB 模型下载完 |
| `make step-06` (torchtitan midtrain) | ⚠️ launcher 模板 | ❓ 必须验 | torchtitan 安装 / `configs/midtrain_air.toml` 与你的 cluster 对齐 |
| `make step-07` (RoPE patch) | ✅ patch 逻辑跑通 | ⚠️ RULER 部分 | clone RULER + 跑 13 task |
| `make step-08` (SFT 数据合成) | ⚠️ 仅打印命令 | ❓ 必须验 | API key + seed 数据准备好 |
| `make step-09` (LLaMA Factory SFT) | ⚠️ launcher 模板 | ❓ 必须验 | LLaMA Factory 装得上；模型 path 对 |
| `make step-10` (agent 轨迹采集) | ⚠️ 仅打印命令 | ❓ 必须验 | mini_agent 你已扩了 `--collect-mode` |
| `make step-12` (VERL GRPO) | ⚠️ launcher 模板 | ❓ 必须验 | VERL 0.3+ 装得上；reward fn 接入 |
| `make step-13` (evalplus / LCB) | ⚠️ 仅打印命令 | ❓ 必须验 | evalplus 装得上；endpoint 起着 |
| `make step-14` (内部 bench eval) | ⚠️ 仅打印命令 | ❓ 必须验 | step 02 真实例 + step 15 endpoint |
| `make step-15` (SGLang FP8 部署) | ⚠️ launcher 模板 | ❓ 必须验 | SGLang 装得上；FP8 ckpt 路径对 |
| `make step-16` (bench grid) | ⚠️ 仅打印命令 | ❓ 必须验 | step 15 endpoint 起着 |
| `make step-17` (Code RAG) | ⚠️ 仅打印命令 | ❓ 必须验 | Qdrant 装得起；REPO_PATH 设了 |
| `make step-18` (agent demo) | ⚠️ 仅打印命令 | ❓ 必须验 | step 15 + 17 同时 ready |

**「⚠️ 仅打印命令」**：脚本会失败-fast 报出缺什么（缺 binary / 缺 env / 缺 data），并打印**你需要在目标机手敲的那一串命令**。这是有意为之——我不想假装某个 200 GPU-hour 的 torchtitan 训练能"一键启动"，那是骗你。

简单说：preflight / 4 个 Tier-1 step / 全部 tests **能在 Mac 上证明工具链没坏**；GPU 步骤的脚本是**可读、可改、不假装能跑**的骨架。

## 目录结构

```
capstone_runtime/
├── README.md             # ← 这份
├── Makefile              # 所有命令入口
├── pytest.ini            # 测试配置
├── requirements.txt      # Tier-1 deps; Tier-2 注释掉了，目标机解注释
├── env.example           # 复制成 .env 改自己的 key
├── tracker.json          # 19-step kanban 初始状态（status=todo）
├── tools/
│   ├── track.py          # kanban CLI
│   └── preflight.py      # HW/SW/网络 自检
├── steps/
│   ├── 01_baseline.py    ✅
│   ├── 02_internal_bench.py
│   ├── 03_data.py
│   ├── 04_decontam.py    ✅
│   ├── 05_arch_probe.py
│   ├── 06_midtraining.sh
│   ├── 07_longctx.py     ✅ (patch 部分)
│   ├── 08_sft_data.py
│   ├── 09_sft_train.sh
│   ├── 10_agent_traj.py
│   ├── 11_rl_env.py      ✅ (reward 部分)
│   ├── 12_rl_train.sh
│   ├── 13_pub_eval.py
│   ├── 14_internal_eval.py
│   ├── 15_deploy.sh
│   ├── 16_deploy_opt.py
│   ├── 17_rag.py
│   ├── 18_agent_demo.py
│   ├── 19_retro.py       ✅
│   └── _template.py      # 抄这个写新 step
├── configs/
│   ├── midtrain_air.toml   # torchtitan
│   ├── sft_air_lora.yaml   # LLaMA Factory
│   ├── ds_z3_config.json   # DeepSpeed ZeRO-3
│   ├── grpo_air.yaml       # VERL GRPO
│   └── swegym_tasks.yaml   # 40-task RL 训练集索引
├── lib/                  # 从仓库 examples/ 拷过来的 5 份脚本，让 bundle 自洽
│   ├── run_pipeline.py     # datatrove pipeline (step 03 调用)
│   ├── extract_pr_sft.py   # 私有 PR → SFT (step 08 调用)
│   ├── grpo_humaneval.py   # TRL GRPO 单机回退 (step 12 备选)
│   ├── collect.py          # 内部 SWE-Bench 收集器 (step 02 主力)
│   └── mini_agent.py       # 300 行 agent (step 10 / 18 复用)
├── tests/                # 19 个单元 / 端到端 test，纯 CPU
│   ├── test_preflight.py
│   ├── test_track.py
│   ├── test_decontam.py
│   ├── test_reward.py
│   └── test_retro.py
├── data/  ckpt/  logs/   # 占位空目录；实验产物落这里（.gitignore 已加）
```

## 拷到目标机的最少 6 步

```bash
# 0) 把整个 capstone_runtime/ 拷过去（保留 lib/ 和 tests/）
scp -r capstone_runtime/  user@h100-box:/data/experiments/
ssh user@h100-box
cd /data/experiments/capstone_runtime

# 1) Python venv（强烈建议；别污染系统 Python）
python3 -m venv .venv
source .venv/bin/activate
make setup                       # 装 Tier-1 deps（requests / datasketch / pytest）

# 2) Tier-2 deps（GPU 步骤需要的，按 requirements.txt 注释里的清单解注释）
pip install torch transformers accelerate datasets peft trl
pip install vllm sglang evalplus datatrove
# ... 看一下错，逐条 fix

# 3) 配 env
cp env.example .env
$EDITOR .env                     # 填 HF_TOKEN / GH_TOKEN / CUDA_VISIBLE_DEVICES 等
set -a; source .env; set +a

# 4) preflight + tests
make preflight                   # 应当看到 8 张卡 + 网络 OK
make test                        # 19 case 应当全过

# 5) 按 phase_capstone.md 的顺序跑：每一步先 read 脚本，再
make step-01
python tools/track.py log capstone-01-baseline "..."
python tools/track.py done capstone-01-baseline --hours 5 --cost 3
```

## 把状态文件 commit 起来（强烈推荐）

```bash
git init                          # 如果还没
git add tracker.json              # 看板 = 研究日志
git commit -m "exp: log step NN done"
```

每天结束 commit 一次 `tracker.json`，等于自动有了一份带时间戳的研究日志。`git log -p tracker.json` 是审计自己实验过程的最便宜方式。

## 不打算做的事

- **不打算把每个 step 都写成"一键启动 + 一键报告"**。原因：每个真实 cluster 的差异太大，假装通用反而误导。脚本骨架 + 思考题才是这个 bundle 真正的价值。
- **不打算追 SOTA**。详见 [`phase_capstone.md`](../phase_capstone.md) §0.3 验收标准。
- **不打算自动算账**。`tracker.json` 的 `actual_hours / actual_cost_usd` 字段是**你**填的；我不假装能从系统监控自动 attribute 成本到 step。

## 反馈 / 报错

跑挂了？某个 step 在你的环境怎么改都不行？欢迎在主仓库提 issue（label `[capstone]`），把 `make preflight` + 出错那一步的 stderr 贴上即可。
