#!/usr/bin/env bash
# Step 06 — Mid-training 退火 5B token（WSD）.
#
# REQUIRES:
#   hw:       8 × H100 80GB, NVLink + IB ≥ 200G
#   software: torchtitan (preferred) or LLaMA Factory continue-pretrain
#   data:     data/clean/  (output of step 03/04)
#   env:      HF_TOKEN, HF_HOME, CUDA_VISIBLE_DEVICES
#
# THIS LAUNCHER: validates env + paths, then dispatches to torchtitan.
# Edit `configs/midtrain_air.toml` for hparams.

set -euo pipefail

cd "$(dirname "$0")/.."   # always run from capstone_runtime/

# ---- preflight ----
: "${HF_TOKEN:?missing HF_TOKEN — see env.example}"
: "${CUDA_VISIBLE_DEVICES:?CUDA_VISIBLE_DEVICES not set}"

if [[ ! -d data/clean ]]; then
  echo "ERROR: data/clean/ not found. Run step 03 + step 04 first." >&2
  exit 2
fi

if ! command -v torchrun >/dev/null; then
  echo "ERROR: torchrun not found. Install torch on the H100 box." >&2
  exit 2
fi

CFG="configs/midtrain_air.toml"
[[ -f "$CFG" ]] || { echo "ERROR: $CFG missing" >&2; exit 2; }

NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l | tr -d ' ')
echo "→ launching torchtitan on $NGPU GPUs with $CFG"

# Replace this with the torchtitan train.py path on your box.
# torchtitan repo: https://github.com/pytorch/torchtitan
TORCHTITAN_TRAIN=${TORCHTITAN_TRAIN:-torchtitan/train.py}

torchrun --nproc-per-node="$NGPU" --master-port=29500 \
  "$TORCHTITAN_TRAIN" --config "$CFG" \
  2>&1 | tee -a logs/06_midtraining.log

echo
echo "✓ midtraining done. Inspect ckpt/midtrain_final/."
echo "  next: python steps/07_longctx.py"
