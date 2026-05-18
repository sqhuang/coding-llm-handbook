#!/usr/bin/env python3
"""
Step NN — <short title>

WHAT THIS DOES: <one paragraph>

REQUIRES:
  - hw:       <e.g. 8 × H100 80GB, NVLink>
  - software: <e.g. torch 2.5+, vllm 0.6+, sglang 0.4+>
  - data:     <HF dataset id, size>
  - env:      <env vars from env.example>

HOW TO USE THIS FILE:
  1. Read REQUIRES, install / set up.
  2. Edit the CONFIG block below for your cluster.
  3. `make step-NN`  — wraps this in track.py start + log tee.
  4. When done:      `python tools/track.py done capstone-NN-... --hours .. --cost ..`

This is a TEMPLATE: it must run far enough to fail with a clear "X is missing"
message on any box, so you know which line to edit, but it WILL NOT fully run
without the cluster + data + tokens.
"""
import os
import shutil
import sys


def need(var: str) -> str:
    v = os.environ.get(var)
    if not v:
        sys.exit(f"missing env var: {var}  (see env.example)")
    return v


def need_bin(name: str) -> str:
    p = shutil.which(name)
    if not p:
        sys.exit(f"missing binary: {name}  (install per REQUIRES above)")
    return p


# ---------------------------- CONFIG ----------------------------
CONFIG = {
    # edit me
}
# ---------------------------------------------------------------


def main() -> int:
    print("template — edit me. Exit 0 just to keep make targets green in CI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
