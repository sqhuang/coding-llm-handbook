#!/usr/bin/env python3
"""
Step 07 — RoPE 扩到 128K + RULER 验真。

WHAT: patch config.json 的 rope_scaling 字段（YaRN factor=4.0），跑 RULER 13-task。

REQUIRES:
  - hw:       8 × H100 (推理为主)
  - software: transformers >= 4.46, RULER repo cloned
  - data:     ckpt/midtrain_final/ from step 06
  - env:      no special token needed (local model)
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT = ROOT / "ckpt" / "midtrain_final"


def patch_rope(ckpt_dir: Path, factor: float = 4.0, original_max: int = 32768) -> int:
    cfg_path = ckpt_dir / "config.json"
    if not cfg_path.exists():
        print(f"ERROR: {cfg_path} not found")
        return 2
    cfg = json.loads(cfg_path.read_text())
    cfg["rope_scaling"] = {
        "type": "yarn",
        "factor": factor,
        "original_max_position_embeddings": original_max,
        "beta_fast": 32,
        "beta_slow": 1,
        "attention_factor": 0.1,
    }
    cfg["max_position_embeddings"] = int(original_max * factor)
    backup = cfg_path.with_suffix(".json.bak")
    if not backup.exists():
        shutil.copy2(cfg_path, backup)
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"✓ patched {cfg_path}")
    print(f"  YaRN factor={factor}, max_pos={cfg['max_position_embeddings']}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--factor", type=float, default=4.0)
    p.add_argument("--original-max", type=int, default=32768)
    args = p.parse_args()

    rc = patch_rope(args.ckpt, args.factor, args.original_max)
    if rc != 0:
        return rc

    print("""
next: run RULER eval

  git clone https://github.com/hsiehjackson/RULER && cd RULER
  bash scripts/run.sh \\
      --model_name_or_path "$(realpath ../""" + str(args.ckpt.relative_to(ROOT)) + """)" \\
      --output_dir ../logs/ruler/

target: needle accuracy @ 128K ≥ 85%.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
