#!/usr/bin/env python3
"""
Step 04 — 评测集去污染：10-gram exact match + (optional) MinHash.

读 step 03 输出的 jsonl/parquet 训练数据，对照评测集题面建索引，扫出污染样本，
把通过的样本写到 `data/clean/`。

Usage:
    python steps/04_decontam.py \
        --train data/raw/*.jsonl \
        --eval-prompts data/eval/humaneval.jsonl data/eval/mbpp.jsonl \
        --out-dir data/clean/ \
        --report logs/decontam_report.md

Self-contained：只用 stdlib（10-gram）+ 可选 datasketch（MinHash）。
没有 datasketch 也能跑——会跳过 minhash，只做 10-gram。

Pure Python；任何 CPU 都能跑；对 ~5GB 训练数据预期 ~10-30 分钟（单进程）。
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Iterator

# Try optional dep; gracefully skip MinHash if missing.
try:
    from datasketch import MinHash, MinHashLSH  # type: ignore
    HAS_DATASKETCH = True
except ImportError:
    HAS_DATASKETCH = False


WHITESPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """NFKC unicode + lowercase + collapse whitespace.

    Critical: skipping this is the #1 root cause of decontam false-negatives
    on Chinese / Japanese / Markdown source.
    """
    t = unicodedata.normalize("NFKC", text).lower()
    return WHITESPACE_RE.sub(" ", t).strip()


def ngrams(text: str, n: int = 10) -> Iterator[str]:
    """Word-level n-grams. We normalize first, then split on whitespace.

    Word level (not char level) keeps memory bounded; n=10 is the SWE / OpenCoder
    convention. For very short prompts (< 20 words), the index also catches them.
    """
    words = normalize(text).split()
    if len(words) < n:
        # Whole prompt as one shingle (short HumanEval-style problems).
        if words:
            yield " ".join(words)
        return
    for i in range(len(words) - n + 1):
        yield " ".join(words[i:i + n])


def hash_shingle(shingle: str) -> int:
    """Stable 64-bit hash for set membership."""
    return int(hashlib.blake2b(shingle.encode(), digest_size=8).hexdigest(), 16)


def load_eval_prompts(paths: list[str]) -> list[dict]:
    """Each path is jsonl; each row needs a 'prompt' (and optionally 'id')."""
    rows: list[dict] = []
    for p in paths:
        for fp in glob.glob(p):
            with open(fp, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    prompt = rec.get("prompt") or rec.get("text") or ""
                    if not prompt:
                        continue
                    rows.append({
                        "src": fp,
                        "id": rec.get("id", f"{Path(fp).stem}::{i}"),
                        "prompt": prompt,
                    })
    return rows


def build_ngram_index(eval_rows: list[dict], n: int) -> set[int]:
    """Hash-set of every n-gram from every eval prompt."""
    index: set[int] = set()
    for row in eval_rows:
        for sh in ngrams(row["prompt"], n):
            index.add(hash_shingle(sh))
    return index


def build_minhash_index(eval_rows: list[dict], num_perm: int, threshold: float) -> "MinHashLSH | None":
    if not HAS_DATASKETCH:
        return None
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for row in eval_rows:
        m = MinHash(num_perm=num_perm)
        for sh in ngrams(row["prompt"], n=5):  # 5-shingle for minhash, coarser
            m.update(sh.encode())
        lsh.insert(row["id"], m)
    return lsh


def text_of(rec: dict) -> str:
    for key in ("text", "content", "code", "prompt"):
        if key in rec and isinstance(rec[key], str):
            return rec[key]
    return ""


def iter_train(paths: list[str]) -> Iterator[tuple[str, int, dict]]:
    for p in paths:
        for fp in glob.glob(p):
            with open(fp, encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield fp, i, json.loads(line)
                    except json.JSONDecodeError:
                        continue


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", nargs="+", required=True,
                   help="jsonl glob(s) of training data")
    p.add_argument("--eval-prompts", nargs="+", required=True,
                   help="jsonl glob(s) of eval prompts to protect against")
    p.add_argument("--out-dir", required=True,
                   help="dir to write clean jsonl files")
    p.add_argument("--report", default="logs/decontam_report.md")
    p.add_argument("--ngram", type=int, default=10)
    p.add_argument("--minhash", action="store_true",
                   help="also run MinHash LSH (slower, catches near-dupes)")
    p.add_argument("--minhash-threshold", type=float, default=0.85)
    p.add_argument("--num-perm", type=int, default=128)
    args = p.parse_args()

    eval_rows = load_eval_prompts(args.eval_prompts)
    if not eval_rows:
        print("ERROR: no eval prompts loaded — check --eval-prompts paths",
              file=sys.stderr)
        return 2
    print(f"loaded {len(eval_rows)} eval prompts from {args.eval_prompts}")

    ngram_index = build_ngram_index(eval_rows, args.ngram)
    print(f"built {args.ngram}-gram index with {len(ngram_index):,} shingles")

    minhash_index = build_minhash_index(eval_rows, args.num_perm, args.minhash_threshold) \
        if args.minhash else None
    if args.minhash and not HAS_DATASKETCH:
        print("WARN: --minhash requested but datasketch not installed; skipped",
              file=sys.stderr)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "ngram_hits": 0, "minhash_hits": 0, "kept": 0}
    contam_by_eval: dict[str, int] = {}

    out_handles: dict[str, "object"] = {}
    try:
        for src, i, rec in iter_train(args.train):
            stats["total"] += 1
            text = text_of(rec)
            if not text:
                continue

            ngram_hit = False
            for sh in ngrams(text, args.ngram):
                if hash_shingle(sh) in ngram_index:
                    ngram_hit = True
                    break
            if ngram_hit:
                stats["ngram_hits"] += 1

            mh_hit = False
            if minhash_index is not None:
                m = MinHash(num_perm=args.num_perm)
                for sh in ngrams(text, n=5):
                    m.update(sh.encode())
                if minhash_index.query(m):
                    mh_hit = True
                    stats["minhash_hits"] += 1

            if ngram_hit or mh_hit:
                continue

            stats["kept"] += 1
            out_path = out_dir / Path(src).name
            if str(out_path) not in out_handles:
                out_handles[str(out_path)] = open(out_path, "w", encoding="utf-8")
            out_handles[str(out_path)].write(json.dumps(rec, ensure_ascii=False) + "\n")
    finally:
        for h in out_handles.values():
            h.close()

    # ---- report ----
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write("# 📊 Decontam report\n\n")
        f.write(f"- 训练数据：{args.train}\n")
        f.write(f"- 评测题面：{args.eval_prompts} ({len(eval_rows)} prompts)\n")
        f.write(f"- N-gram size：{args.ngram}\n")
        f.write(f"- MinHash：{'on' if args.minhash and HAS_DATASKETCH else 'off'}\n\n")
        f.write("| 指标 | 数值 | 占比 |\n|---|---|---|\n")
        total = max(stats["total"], 1)
        f.write(f"| 总样本 | {stats['total']:,} | 100% |\n")
        f.write(f"| 10-gram 命中（污染） | {stats['ngram_hits']:,} | "
                f"{stats['ngram_hits'] / total:.3%} |\n")
        f.write(f"| MinHash 命中（污染） | {stats['minhash_hits']:,} | "
                f"{stats['minhash_hits'] / total:.3%} |\n")
        f.write(f"| 保留 | {stats['kept']:,} | {stats['kept'] / total:.1%} |\n\n")
        f.write("> 验收：污染率应 < 0.1%。如果 > 1%，回 step 03 检查。\n")

    print()
    print(f"=== summary ===")
    print(f"  total samples : {stats['total']:,}")
    print(f"  ngram hits    : {stats['ngram_hits']:,}")
    print(f"  minhash hits  : {stats['minhash_hits']:,}")
    print(f"  kept          : {stats['kept']:,}")
    print(f"  report        : {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
