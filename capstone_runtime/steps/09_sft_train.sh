#!/usr/bin/env bash
# Step 09 — LoRA SFT r=64 · 2 epoch.
#
# REQUIRES:
#   hw:       8 × H100 80GB
#   software: LLaMA Factory 0.9+
#   data:     data/sft_combined.jsonl (output of step 08+10)
#   ckpt:     ckpt/midtrain_final_long/ (post-RoPE patch)
#   env:      CUDA_VISIBLE_DEVICES, HF_TOKEN
#
# CONFIG: configs/sft_air_lora.yaml — edit dataset paths / ckpt path / output_dir.

set -euo pipefail
cd "$(dirname "$0")/.."

: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES}"

CFG="configs/sft_air_lora.yaml"
[[ -f "$CFG" ]] || { echo "ERROR: $CFG missing" >&2; exit 2; }
[[ -f data/sft_combined.jsonl ]] || {
  echo "ERROR: data/sft_combined.jsonl missing — finish step 08 + 10 first." >&2
  exit 2
}

if ! command -v llamafactory-cli >/dev/null; then
  echo "ERROR: llamafactory-cli not on PATH." >&2
  echo "Install: pip install llamafactory[deepspeed,metrics] && pip install bitsandbytes" >&2
  exit 2
fi

echo "→ launching LLaMA Factory SFT with $CFG"
llamafactory-cli train "$CFG" 2>&1 | tee -a logs/09_sft_train.log

echo
echo "✓ SFT done. Adapter at ckpt/sft_lora_r64/."
echo "  next: python steps/10_agent_traj.py  (or step 12 RL if you already have traj)"
