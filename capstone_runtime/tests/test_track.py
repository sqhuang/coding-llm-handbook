"""End-to-end test of tools/track.py against a temp tracker.json."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACK = ROOT / "tools" / "track.py"


def make_tracker(tmp_path: Path) -> Path:
    """Write a minimal 2-step tracker.json in tmp_path."""
    state = {
        "experiment_id": "test-exp",
        "started": "2026-05-15",
        "budget_usd": 100,
        "budget_gpu_hours": 50,
        "steps": [
            {"id": "test-01-first", "phase": 0, "name": "first", "status": "todo",
             "owner": "tester", "gpu": "0", "data": "—", "model": "—",
             "hparams": {}, "eta_hours": 1, "eta_cost_usd": 0,
             "actual_hours": None, "actual_cost_usd": None,
             "started_at": None, "done_at": None, "blocked_reason": None, "log": []},
            {"id": "test-02-second", "phase": 1, "name": "second", "status": "todo",
             "owner": "tester", "gpu": "0", "data": "—", "model": "—",
             "hparams": {}, "eta_hours": 2, "eta_cost_usd": 5,
             "actual_hours": None, "actual_cost_usd": None,
             "started_at": None, "done_at": None, "blocked_reason": None, "log": []},
        ],
    }
    p = tmp_path / "tracker.json"
    p.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    return p


def run(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(TRACK), *args],
        cwd=tmp_path, capture_output=True, text=True, check=False,
    )


def test_board_lists_all(tmp_path):
    make_tracker(tmp_path)
    r = run(tmp_path, "board")
    assert r.returncode == 0, r.stderr
    assert "test-01-first" in r.stdout
    assert "test-02-second" in r.stdout
    assert "TODO" in r.stdout


def test_start_done_flow(tmp_path):
    p = make_tracker(tmp_path)
    assert run(tmp_path, "start", "01-first").returncode == 0
    assert run(tmp_path, "log", "01-first", "hello").returncode == 0
    assert run(tmp_path, "done", "01-first", "--hours", "0.5", "--cost", "0").returncode == 0
    state = json.loads(p.read_text())
    s = state["steps"][0]
    assert s["status"] == "done"
    assert s["actual_hours"] == 0.5
    assert any("hello" in e["msg"] for e in s["log"])


def test_block_unblock(tmp_path):
    p = make_tracker(tmp_path)
    run(tmp_path, "start", "02-second")
    run(tmp_path, "block", "02-second", "--reason", "waiting on GPU")
    s = json.loads(p.read_text())["steps"][1]
    assert s["status"] == "blocked"
    assert "GPU" in s["blocked_reason"]
    run(tmp_path, "unblock", "02-second")
    s = json.loads(p.read_text())["steps"][1]
    assert s["status"] == "doing"
    assert s["blocked_reason"] is None


def test_ambiguous_id_errors(tmp_path):
    make_tracker(tmp_path)
    r = run(tmp_path, "show", "test")
    assert r.returncode != 0
    assert "ambiguous" in r.stderr.lower() or "ambiguous" in r.stdout.lower()


def test_budget_output(tmp_path):
    make_tracker(tmp_path)
    r = run(tmp_path, "budget")
    assert r.returncode == 0
    assert "534h" in r.stdout or "3h" in r.stdout or "eta" in r.stdout.lower()


def test_export_md(tmp_path):
    make_tracker(tmp_path)
    r = run(tmp_path, "export-md")
    assert r.returncode == 0
    md = (tmp_path / "tracker_view.md").read_text()
    assert "test-01-first" in md
    assert "TODO" in md
