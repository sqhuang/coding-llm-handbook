# `tools/track.py` — capstone 看板 CLI

> 配套实验册：[`../phase_capstone.md`](../phase_capstone.md)

把 4 周 capstone 实验的 19 个 step 当 kanban 卡片管理：状态、备注、估算 vs 实际成本都进同一份 `tracker.json`。纯 stdlib，没有外部依赖，跑 Python 3.9+ 即可。

## 状态机

```
todo ──start──▶ doing ──done──▶ done
                 │ ▲
                 │ │ unblock
                 ▼ │
              blocked
```

`reopen` 把已完成的 step 退回 `doing`（少用，但比误标 done 后无法回退强）。

## 常用命令

| 命令 | 用途 |
|---|---|
| `python tools/track.py board` | 全局看板（4 列） |
| `python tools/track.py show capstone-04-decontam` | 单步详情 + 最近 10 条 log |
| `python tools/track.py start <id>` | 切到 doing，自动盖时间戳 |
| `python tools/track.py log <id> "loss 平稳后改 lr"` | 加备注（永不丢） |
| `python tools/track.py done <id> --hours 18 --cost 36` | 切到 done，必须填实际工时 + 美元 |
| `python tools/track.py block <id> --reason "..."` | 阻塞（不挡 board 其它列） |
| `python tools/track.py unblock <id>` | 解阻塞，回到 doing 或 todo |
| `python tools/track.py reopen <id>` | 把误标 done 的退回 |
| `python tools/track.py budget` | 实际 vs 预算燃烧率 |
| `python tools/track.py export-md` | 把看板渲染到 `tracker_view.md`（commit 友好） |

`<id>` 支持模糊匹配——`python tools/track.py show 04-decontam` 会自动找到唯一前缀。

## tracker.json 数据格式

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

把 `tracker.json` 提交到 git，你就拥有了**完整的研究日志**：跨季度对比"上次同类 step 估算 vs 实际"误差，找出自己最爱错估的环节（数据 / 训练 / 评测 / 部署 / 复盘），下次估算更准。

## 自定义实验

`tracker.json` 不是 capstone 专属，任何"多 step 串起来的实验"都能用——比如下次做 phase5 RL ablation，可以 `cp tracker.json tracker_rl_ablation.json` 然后清空 steps 字段，重写 5-10 个 step 即可。

为了简单，CLI 默认读 `./tracker.json`；要切实验把当前实验 json `mv` 进/出根目录就行。
