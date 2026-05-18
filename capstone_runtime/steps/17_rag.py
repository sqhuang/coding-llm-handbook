#!/usr/bin/env python3
"""
Step 17 — Code RAG: tree-sitter chunks → bge-code-v1 → Qdrant + BM25 hybrid + reranker.

REQUIRES:
  - hw:       1 × A100 40GB (embedding) + Qdrant CPU
  - software: qdrant-client, tree-sitter-languages, FlagEmbedding, rank-bm25
  - data:     一个真实 repo（绝对路径）
  - env:      QDRANT_URL（默认 http://localhost:6333）
"""
import os
import sys
from pathlib import Path


def main() -> int:
    fails = []
    for mod in ("qdrant_client", "tree_sitter_languages", "FlagEmbedding"):
        try:
            __import__(mod)
        except ImportError:
            fails.append(f"pip install {mod.replace('_', '-')}")

    if fails:
        print("missing deps:")
        for f in fails:
            print(f"  - {f}")
        return 2

    repo = os.environ.get("REPO_PATH")
    if not repo or not Path(repo).is_dir():
        print("ERROR: set REPO_PATH=/abs/path/to/repo")
        return 2

    print(f"""
target: {repo}
qdrant: {os.environ.get('QDRANT_URL', 'http://localhost:6333')}

pseudocode for lib/rag_index.py (TODO impl):

  1. walk REPO_PATH, skip .git / vendor / node_modules
  2. for each .py/.ts/.go/...:
       parse with tree-sitter
       extract function / class nodes
       split into chunks (max 1024 tokens, overlap 128)
       embed with BAAI/bge-code-v1 (1024-d)
       upsert into Qdrant collection "code"
  3. build BM25 index in parallel (rank-bm25)
  4. expose lib/rag_search.py with:
       search(query, k=50_dense + 50_bm25) → 100 candidates → bge-reranker → top 5

validation: 50 hand-written queries, Recall@5 ≥ 80%.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
