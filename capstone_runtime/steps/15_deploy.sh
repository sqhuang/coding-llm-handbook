#!/usr/bin/env bash
# Step 15 — SGLang FP8 部署.
#
# REQUIRES:
#   hw:       8 × H100 80GB (TP=8)
#   software: sglang >= 0.4.0
#   data:     ckpt/sft_lora_merged 或 ckpt/rl_grpo_merged
#   env:      CUDA_VISIBLE_DEVICES
#
# 输出常驻 endpoint：http://0.0.0.0:30000/v1
# 这是 step 13/14/16/18 都会用的 endpoint。

set -euo pipefail
cd "$(dirname "$0")/.."

: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES}"

MODEL_PATH=${MODEL_PATH:-ckpt/rl_grpo_merged}
[[ -d "$MODEL_PATH" ]] || { echo "ERROR: $MODEL_PATH missing — merge LoRA first."; exit 2; }

if ! python -c "import sglang" 2>/dev/null; then
  echo "ERROR: sglang not installed. `pip install 'sglang>=0.4.0'` on the H100 box." >&2
  exit 2
fi

PORT=${PORT:-30000}
NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l | tr -d ' ')

echo "→ launching SGLang on $NGPU GPUs · port $PORT"
echo "  (Ctrl-C to stop; for daemon use systemd / tmux)"

exec python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --quantization fp8 \
  --tp "$NGPU" \
  --port "$PORT" \
  --host 0.0.0.0 \
  --max-running-requests 64 \
  --max-total-tokens 1048576 \
  --enable-mixed-chunk \
  --enable-prefix-caching \
  2>&1 | tee -a logs/15_deploy.log
