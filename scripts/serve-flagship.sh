#!/usr/bin/env bash
# Run AEON's flagship DFlash recipe EXACTLY: their serve_qwen36_dflash.sh inside
# aeon-vllm-ultimate:latest, XS body (modelopt) + z-lab DFlash drafter, n=10.
set -euo pipefail
IMG="${IMG:-ghcr.io/aeon-7/aeon-vllm-ultimate:latest}"
ROOT="${ROOT:-$HOME/aeon27b}"
PORT="${PORT:-8000}"
NSPEC="${NSPEC:-10}"          # README-validated winner (n>=12 crashes)
PROFILE="${PROFILE:-production}"

docker rm -f aeon-flagship >/dev/null 2>&1 || true
mkdir -p "$ROOT/vllm-cache-xs"
exec docker run --rm --name aeon-flagship --gpus all --ipc host --network host \
  -e HF_HUB_OFFLINE=1 \
  -e MODEL_DIR=/models/aeon-xs \
  -e DFLASH_DIR=/models/dflash-drafter \
  -e NUM_SPECULATIVE_TOKENS="$NSPEC" \
  -e PORT="$PORT" \
  -e PROFILE="$PROFILE" \
  -e TORCHINDUCTOR_COMPILE_THREADS="${COMPILE_THREADS:-4}" \
  -e VLLM_LOGGING_LEVEL="${LOGLEVEL:-INFO}" \
  -v "$ROOT/models/aeon-xs:/models/aeon-xs:ro" \
  -v "$ROOT/models/dflash-draft:/models/dflash-drafter:ro" \
  -v "$ROOT/aeon-flagship/scripts:/aeon-scripts:ro" \
  -v "$ROOT/vllm-cache-xs:/root/.cache/vllm" \
  --entrypoint bash "$IMG" /aeon-scripts/serve_qwen36_dflash.sh
