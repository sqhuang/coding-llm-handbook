"""Test that 19_retro.py renders a markdown file with key sections."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_retro_renders(tmp_path):
    # We can't easily move ROOT, so instead overwrite logs/RETRO.md in-place
    # and assert against the actual tracker.json that ships in the bundle.
    out = ROOT / "logs" / "RETRO.md"
    r = subprocess.run([sys.executable, str(ROOT / "steps" / "19_retro.py")],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    text = out.read_text(encoding="utf-8")
    assert "复盘" in text
    assert "步骤复盘" in text
    assert "估算准确度" in text
    assert "复盘 5 问" in text


def test_retro_handles_partial_done_steps(tmp_path):
    """Backup tracker.json, drop in a fake one with one done step, render, restore."""
    tracker = ROOT / "tracker.json"
    backup = tracker.read_text(encoding="utf-8")
    try:
        fake = json.loads(backup)
        fake["steps"][0]["status"] = "done"
        fake["steps"][0]["actual_hours"] = 5.0
        fake["steps"][0]["actual_cost_usd"] = 10.0
        fake["steps"][0]["eta_hours"] = 4
        fake["steps"][0]["eta_cost_usd"] = 0
        fake["steps"][0]["log"] = [
            {"at": "2026-05-15T10:00:00+00:00", "msg": "done early"},
        ]
        tracker.write_text(json.dumps(fake, ensure_ascii=False), encoding="utf-8")
        r = subprocess.run([sys.executable, str(ROOT / "steps" / "19_retro.py")],
                           capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        text = (ROOT / "logs" / "RETRO.md").read_text(encoding="utf-8")
        assert "1/19" in text   # one step done
        assert "phase 0" in text  # phase bucket present
        assert "done early" in text  # log digest works
    finally:
        tracker.write_text(backup, encoding="utf-8")
