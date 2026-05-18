#!/usr/bin/env python3
"""
Step 19 — 复盘：tracker.json → RETRO.md skeleton.

Reads `tracker.json` and writes `logs/RETRO.md` with:
  - cover table (steps done / total, hours actual vs eta, $ actual vs eta)
  - per-step section pre-filled with eta vs actual + recent log entries
  - "estimate accuracy" bucket: which step types you most often under/over-shoot
  - blank prompts at the bottom for the 5 retro questions

Self-contained: stdlib only. Runs in seconds on any box.
"""
from __future__ import annotations

import datetime as dt
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACKER = ROOT / "tracker.json"
OUT = ROOT / "logs" / "RETRO.md"


def fmt_hours(h: float | int | None) -> str:
    if h is None:
        return "—"
    return f"{h:g}h"


def fmt_cost(c: float | int | None) -> str:
    if c is None:
        return "—"
    return f"${c:g}"


def estimate_accuracy(steps: list[dict]) -> dict[str, list[float]]:
    """Group steps by phase, return ratio actual/eta per group (done only)."""
    buckets: dict[str, list[float]] = defaultdict(list)
    for s in steps:
        if s["status"] != "done":
            continue
        eta = s.get("eta_hours") or 0
        actual = s.get("actual_hours") or 0
        if eta == 0:
            continue
        buckets[f"phase {s['phase']}"].append(actual / eta)
    return buckets


def build_md(state: dict) -> str:
    steps = state["steps"]
    done = [s for s in steps if s["status"] == "done"]

    eta_h = sum((s.get("eta_hours") or 0) for s in steps)
    eta_c = sum((s.get("eta_cost_usd") or 0) for s in steps)
    act_h = sum((s.get("actual_hours") or 0) for s in done)
    act_c = sum((s.get("actual_cost_usd") or 0) for s in done)

    out: list[str] = []
    out.append(f"# 📓 复盘：{state['experiment_id']}")
    out.append("")
    out.append(f"> 起始 {state['started']} · 生成于 {dt.date.today().isoformat()}")
    out.append("")

    # ---- summary ----
    out.append("## 1. 总览")
    out.append("")
    out.append(f"- 步骤完成：**{len(done)}/{len(steps)}**")
    out.append(f"- 工时：实际 **{fmt_hours(act_h)}** / 估算 **{fmt_hours(eta_h)}** "
               f"({act_h / eta_h:.1%} 燃烧率，仅 done 计入实际)" if eta_h else
               f"- 工时：实际 **{fmt_hours(act_h)}**")
    out.append(f"- 成本：实际 **{fmt_cost(act_c)}** / 估算 **{fmt_cost(eta_c)}** "
               f"({act_c / eta_c:.1%} 燃烧率)" if eta_c else
               f"- 成本：实际 **{fmt_cost(act_c)}**")
    out.append("")

    # ---- per-step ----
    out.append("## 2. 步骤复盘")
    out.append("")
    out.append("| ID | Phase | 状态 | ETA | 实际 | 误差 |")
    out.append("|---|---|---|---|---|---|")
    for s in steps:
        eta_h_s = s.get("eta_hours")
        act_h_s = s.get("actual_hours")
        if eta_h_s and act_h_s:
            ratio = act_h_s / eta_h_s
            err = f"{ratio:.2f}× {'⚠️' if ratio > 1.3 or ratio < 0.7 else '✅'}"
        else:
            err = "—"
        out.append(
            f"| `{s['id']}` | {s['phase']} | {s['status']} | "
            f"{fmt_hours(eta_h_s)} / {fmt_cost(s.get('eta_cost_usd'))} | "
            f"{fmt_hours(act_h_s)} / {fmt_cost(s.get('actual_cost_usd'))} | {err} |"
        )
    out.append("")

    # ---- estimate accuracy ----
    out.append("## 3. 估算准确度（actual / eta，越接近 1.0 越准）")
    out.append("")
    buckets = estimate_accuracy(steps)
    if not buckets:
        out.append("_(没有已完成的 step；跑完几个再回来看。)_")
    else:
        out.append("| Phase | n | 平均 ratio | 最大 | 最小 |")
        out.append("|---|---|---|---|---|")
        for k, vals in sorted(buckets.items()):
            avg = sum(vals) / len(vals)
            out.append(f"| {k} | {len(vals)} | {avg:.2f} | {max(vals):.2f} | {min(vals):.2f} |")
    out.append("")

    # ---- log digest ----
    out.append("## 4. 关键日志（每步取最后 3 条）")
    out.append("")
    for s in steps:
        log = s.get("log") or []
        if not log:
            continue
        out.append(f"### `{s['id']}` — {s['name']}")
        for entry in log[-3:]:
            out.append(f"- `{entry['at']}` {entry['msg']}")
        out.append("")

    # ---- prompts ----
    out.append("## 5. 复盘 5 问（请手填）")
    out.append("")
    for q in [
        "哪一步严重超预算？根因是什么？（卡 / 数据 / 框架 / 自己估算偏）",
        "哪一步收益意外好？为什么？下次能否复用？",
        "如果重来一遍，最大的改动是什么？为什么之前没想到？",
        "「通用 base + 私有数据 SFT」是否真的优于「纯 API + RAG」？算一次总账（含 ROI / TCO）。",
        "tracker.json 的「估算 vs 实际」误差最大的是哪一类 step？下次怎么校准？",
    ]:
        out.append(f"- **Q:** {q}")
        out.append(f"  **A:** _(待填)_")
        out.append("")

    return "\n".join(out) + "\n"


def main() -> int:
    state = json.loads(TRACKER.read_text(encoding="utf-8"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build_md(state), encoding="utf-8")
    print(f"✓ wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
