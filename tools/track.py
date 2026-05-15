#!/usr/bin/env python3
"""
track.py — Kanban CLI for the capstone experiment (phase_capstone.md).

State lives in ./tracker.json. The whole point is to be cheap and grep-friendly:
status, owner, hparams, eta vs actual, and a free-form log per step. Plain
stdlib only — runs anywhere Python 3.9+ runs.

Quick start:
    python tools/track.py board
    python tools/track.py show capstone-04-sft
    python tools/track.py start capstone-04-sft
    python tools/track.py log capstone-04-sft "lr 1e-4 后 loss 平稳"
    python tools/track.py done capstone-04-sft --hours 18 --cost 36
    python tools/track.py block capstone-05-rl --reason "等 sandbox 镜像"
    python tools/track.py unblock capstone-05-rl
    python tools/track.py budget          # 看 GPU-hour / $ 实际 vs 预算
    python tools/track.py export-md       # 把看板写到 tracker_view.md
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
TRACKER = ROOT / "tracker.json"
VIEW_MD = ROOT / "tracker_view.md"

STATUSES = ("todo", "doing", "blocked", "done")
COLOR = {
    "todo":    "\x1b[90m",   # grey
    "doing":   "\x1b[33m",   # yellow
    "blocked": "\x1b[31m",   # red
    "done":    "\x1b[32m",   # green
}
RESET = "\x1b[0m"


# -------------------- IO --------------------
def load() -> dict[str, Any]:
    if not TRACKER.exists():
        sys.exit(f"tracker.json not found at {TRACKER}. Run `init` first or copy the template.")
    return json.loads(TRACKER.read_text(encoding="utf-8"))


def save(state: dict[str, Any]) -> None:
    TRACKER.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")


def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def find_step(state: dict[str, Any], step_id: str) -> dict[str, Any]:
    for s in state["steps"]:
        if s["id"] == step_id:
            return s
    matches = [s for s in state["steps"] if step_id in s["id"]]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        sys.exit(f"ambiguous id '{step_id}': {[m['id'] for m in matches]}")
    sys.exit(f"no step matches '{step_id}'. Try `track.py board` for ids.")


# -------------------- commands --------------------
def cmd_board(args: argparse.Namespace) -> None:
    state = load()
    cols: dict[str, list[dict[str, Any]]] = {s: [] for s in STATUSES}
    for s in state["steps"]:
        cols[s["status"]].append(s)

    header = f"📋 {state['experiment_id']}   started {state['started']}"
    print(header)
    print("─" * len(header))
    for st in STATUSES:
        c = COLOR[st]
        title = f"{c}{st.upper():<7}{RESET} ({len(cols[st])})"
        print(f"\n{title}")
        for s in cols[st]:
            extra = ""
            if s["status"] == "blocked":
                extra = f"  ⚠️  {s.get('blocked_reason') or ''}"
            elif s["status"] == "done":
                actual_h = s.get("actual_hours")
                eta_h = s.get("eta_hours")
                if actual_h is not None and eta_h:
                    ratio = actual_h / eta_h
                    flag = "✅" if 0.7 <= ratio <= 1.3 else "⚠️"
                    extra = f"  {flag} {actual_h}h / {eta_h}h"
            print(f"  · {s['id']:<32} P{s['phase']}  {s['name']}{extra}")
    print()


def cmd_show(args: argparse.Namespace) -> None:
    state = load()
    s = find_step(state, args.step_id)
    print(f"{COLOR[s['status']]}[{s['status'].upper()}]{RESET} {s['id']}  ·  Phase {s['phase']}")
    print(f"  name      : {s['name']}")
    print(f"  owner     : {s.get('owner') or '-'}")
    print(f"  gpu       : {s.get('gpu') or '-'}")
    print(f"  data      : {s.get('data') or '-'}")
    print(f"  model     : {s.get('model') or '-'}")
    hp = s.get("hparams") or {}
    if hp:
        print(f"  hparams   : {json.dumps(hp, ensure_ascii=False)}")
    print(f"  eta       : {s.get('eta_hours')}h / ${s.get('eta_cost_usd')}")
    if s.get("started_at"):
        print(f"  started   : {s['started_at']}")
    if s.get("done_at"):
        print(f"  done      : {s['done_at']}  (actual {s.get('actual_hours')}h / ${s.get('actual_cost_usd')})")
    if s.get("blocked_reason"):
        print(f"  blocked   : {s['blocked_reason']}")
    log = s.get("log") or []
    if log:
        print(f"  log ({len(log)}):")
        for entry in log[-10:]:
            print(f"    · {entry['at']}  {entry['msg']}")
        if len(log) > 10:
            print(f"    … ({len(log) - 10} earlier entries hidden)")
    else:
        print("  log       : (empty)")


def _transition(state: dict[str, Any], step: dict[str, Any], to: str) -> None:
    if step["status"] == to:
        return
    step["status"] = to


def cmd_start(args: argparse.Namespace) -> None:
    state = load()
    s = find_step(state, args.step_id)
    if s["status"] == "done":
        sys.exit(f"{s['id']} is already done; use `reopen` if you really mean to.")
    s["started_at"] = s.get("started_at") or now_iso()
    _transition(state, s, "doing")
    s.setdefault("log", []).append({"at": now_iso(), "msg": "→ doing"})
    save(state)
    print(f"▶ {s['id']} → doing")


def cmd_done(args: argparse.Namespace) -> None:
    state = load()
    s = find_step(state, args.step_id)
    s["done_at"] = now_iso()
    s["actual_hours"] = args.hours
    s["actual_cost_usd"] = args.cost
    _transition(state, s, "done")
    note = args.note or "→ done"
    s.setdefault("log", []).append({"at": now_iso(), "msg": note})
    save(state)
    print(f"✓ {s['id']} → done ({args.hours}h / ${args.cost})")


def cmd_log(args: argparse.Namespace) -> None:
    state = load()
    s = find_step(state, args.step_id)
    s.setdefault("log", []).append({"at": now_iso(), "msg": args.message})
    save(state)
    print(f"📝 logged on {s['id']}")


def cmd_block(args: argparse.Namespace) -> None:
    state = load()
    s = find_step(state, args.step_id)
    s["blocked_reason"] = args.reason
    _transition(state, s, "blocked")
    s.setdefault("log", []).append({"at": now_iso(), "msg": f"BLOCKED: {args.reason}"})
    save(state)
    print(f"⛔ {s['id']} → blocked  ({args.reason})")


def cmd_unblock(args: argparse.Namespace) -> None:
    state = load()
    s = find_step(state, args.step_id)
    s["blocked_reason"] = None
    target = "doing" if s.get("started_at") else "todo"
    _transition(state, s, target)
    s.setdefault("log", []).append({"at": now_iso(), "msg": f"unblocked → {target}"})
    save(state)
    print(f"▶ {s['id']} → {target}")


def cmd_reopen(args: argparse.Namespace) -> None:
    state = load()
    s = find_step(state, args.step_id)
    s["done_at"] = None
    _transition(state, s, "doing")
    s.setdefault("log", []).append({"at": now_iso(), "msg": "reopened"})
    save(state)
    print(f"↩ {s['id']} → doing (reopened)")


def cmd_budget(args: argparse.Namespace) -> None:
    state = load()
    eta_h = sum((s.get("eta_hours") or 0) for s in state["steps"])
    eta_c = sum((s.get("eta_cost_usd") or 0) for s in state["steps"])
    act_h = sum((s.get("actual_hours") or 0) for s in state["steps"] if s["status"] == "done")
    act_c = sum((s.get("actual_cost_usd") or 0) for s in state["steps"] if s["status"] == "done")
    done = sum(1 for s in state["steps"] if s["status"] == "done")
    total = len(state["steps"])
    print(f"📊 progress     : {done}/{total} steps done")
    print(f"   eta total    : {eta_h:>5}h / ${eta_c}")
    print(f"   actual sofar : {act_h:>5}h / ${act_c}")
    print(f"   budget cap   : {state.get('budget_gpu_hours')}h / ${state.get('budget_usd')}")
    if act_c and eta_c:
        ratio = act_c / (eta_c * done / total) if done else 0
        if ratio > 1.3:
            print(f"   ⚠️  burn rate {ratio:.2f}× over plan — review scope")
        else:
            print(f"   ✅ burn rate {ratio:.2f}× of pro-rated plan")


def cmd_export_md(args: argparse.Namespace) -> None:
    state = load()
    out: list[str] = []
    out.append(f"# 📋 {state['experiment_id']} · 看板\n")
    out.append(f"> 起始 {state['started']} · 预算 {state.get('budget_gpu_hours')} GPU-hr / "
               f"${state.get('budget_usd')}\n")
    for st in STATUSES:
        rows = [s for s in state["steps"] if s["status"] == st]
        out.append(f"\n## {st.upper()} ({len(rows)})\n")
        if not rows:
            out.append("_(空)_\n")
            continue
        out.append("| ID | Phase | Name | ETA | 实际 |")
        out.append("|---|---|---|---|---|")
        for s in rows:
            eta = f"{s.get('eta_hours')}h / ${s.get('eta_cost_usd')}"
            actual = (f"{s.get('actual_hours')}h / ${s.get('actual_cost_usd')}"
                      if s["status"] == "done" else "—")
            out.append(f"| `{s['id']}` | {s['phase']} | {s['name']} | {eta} | {actual} |")
    VIEW_MD.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"✓ wrote {VIEW_MD.relative_to(ROOT)}")


# -------------------- entrypoint --------------------
def main() -> None:
    p = argparse.ArgumentParser(prog="track.py", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("board", help="Show the kanban board").set_defaults(func=cmd_board)
    sub.add_parser("budget", help="Show budget burn vs plan").set_defaults(func=cmd_budget)
    sub.add_parser("export-md", help="Write tracker_view.md").set_defaults(func=cmd_export_md)

    sp = sub.add_parser("show", help="Show details for one step")
    sp.add_argument("step_id")
    sp.set_defaults(func=cmd_show)

    sp = sub.add_parser("start", help="todo → doing")
    sp.add_argument("step_id")
    sp.set_defaults(func=cmd_start)

    sp = sub.add_parser("done", help="doing → done")
    sp.add_argument("step_id")
    sp.add_argument("--hours", type=float, required=True, help="actual wall-clock hours")
    sp.add_argument("--cost", type=float, required=True, help="actual USD spent")
    sp.add_argument("--note", default=None)
    sp.set_defaults(func=cmd_done)

    sp = sub.add_parser("log", help="append a free-form note")
    sp.add_argument("step_id")
    sp.add_argument("message")
    sp.set_defaults(func=cmd_log)

    sp = sub.add_parser("block", help="mark blocked")
    sp.add_argument("step_id")
    sp.add_argument("--reason", required=True)
    sp.set_defaults(func=cmd_block)

    sp = sub.add_parser("unblock", help="clear blocked")
    sp.add_argument("step_id")
    sp.set_defaults(func=cmd_unblock)

    sp = sub.add_parser("reopen", help="done → doing")
    sp.add_argument("step_id")
    sp.set_defaults(func=cmd_reopen)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
