"""Preflight should run on any machine without crashing."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_preflight_runs_without_crash():
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "preflight.py")],
        capture_output=True, text=True,
    )
    # exit 0 (full pass) or 1 (hard fail). Anything else (e.g. python crash) is a regression.
    assert r.returncode in (0, 1), f"stdout: {r.stdout}\nstderr: {r.stderr}"
    assert "preflight" in r.stdout.lower()
    assert "python" in r.stdout.lower()
