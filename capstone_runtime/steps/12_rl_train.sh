#!/usr/bin/env bash
# Step 12 — GRPO 100 step.
#
# REQUIRES:
#   hw:       8 × H100 80GB (4 trainer + 4 vLLM rollout)
#   software: VERL 0.3+ (preferred) or TRL >= 0.12
#   data:     40-task SWE-Gym sandbox (step 11) + ckpt/sft_lora_merged/
#   env:      CUDA_VISIBLE_DEVICES, HF_TOKEN
#
# CONFIG: configs/grpo_air.yaml — group_size / KL beta / lr.
#
# Lightweight alternative: TRL GRPO single-node (lib/grpo_humaneval.py),
# but for SWE-Gym RL you want VERL's rollout-trainer split.

set -euo pipefail
cd "$(dirname "$0")/.."

: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES}"

CFG="configs/grpo_air.yaml"
[[ -f "$CFG" ]] || { echo "ERROR: $CFG missing"; exit 2; }
[[ -d ckpt/sft_lora_merged ]] || {
  echo "ERROR: ckpt/sft_lora_merged/ missing — merge step 09's LoRA first:"
  echo "       llamafactory-cli export configs/merge_sft.yaml"
  exit 2
}

if [[ -d verl ]]; then
  echo "→ launching VERL GRPO"
  cd verl
  bash examples/grpo_coder/run_grpo.sh --config "../$CFG" 2>&1 | tee -a ../logs/12_rl_train.log
else
  echo "VERL not found; falling back to TRL single-node (smaller scale):"
  echo "  python lib/grpo_humaneval.py  # edit MODEL / dataset first"
  exit 3
fi

echo "✓ RL done. Adapter at ckpt/rl_grpo/."
