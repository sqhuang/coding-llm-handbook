"""Unit tests for the n-gram + minhash decontaminator."""
import json
import subprocess
import sys
from pathlib import Path

from .conftest import load_step

ROOT = Path(__file__).resolve().parent.parent


def test_normalize_collapses_whitespace():
    mod = load_step("04_decontam.py")
    assert mod.normalize("Hello   World\t\nfoo") == "hello world foo"


def test_normalize_unicode_nfkc():
    mod = load_step("04_decontam.py")
    # NFKC folds full-width chars into ASCII
    assert mod.normalize("ＡＢＣ") == "abc"


def test_ngrams_short_text_yields_whole():
    mod = load_step("04_decontam.py")
    grams = list(mod.ngrams("hello world", n=10))
    assert grams == ["hello world"]


def test_ngrams_window():
    mod = load_step("04_decontam.py")
    words = "a b c d e f g h i j k l".split()
    grams = list(mod.ngrams(" ".join(words), n=10))
    assert grams[0] == "a b c d e f g h i j"
    assert grams[-1] == "c d e f g h i j k l"


def test_e2e_catches_verbatim_dup(tmp_path):
    """Run 04_decontam.py against synthetic data; verbatim copy must be filtered."""
    eval_path = tmp_path / "eval.jsonl"
    eval_path.write_text(
        json.dumps({"id": "p1", "prompt": "the quick brown fox jumps over the lazy dog "
                                          "before the moon rises slowly tonight"}) + "\n",
        encoding="utf-8",
    )

    train_path = tmp_path / "train.jsonl"
    train_path.write_text(
        json.dumps({"text": "the quick brown fox jumps over the lazy dog "
                            "before the moon rises slowly tonight and "
                            "then goes home"}) + "\n"
        + json.dumps({"text": "def add(a, b):\n    return a + b"}) + "\n",
        encoding="utf-8",
    )

    out_dir = tmp_path / "clean"
    report = tmp_path / "report.md"

    r = subprocess.run(
        [sys.executable, str(ROOT / "steps" / "04_decontam.py"),
         "--train", str(train_path),
         "--eval-prompts", str(eval_path),
         "--out-dir", str(out_dir),
         "--report", str(report)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr

    # The verbatim sample was caught; clean output should contain only `def add`.
    out_files = list(out_dir.glob("*.jsonl"))
    assert len(out_files) == 1
    rows = [json.loads(l) for l in out_files[0].read_text().splitlines()]
    assert len(rows) == 1
    assert "def add" in rows[0]["text"]
