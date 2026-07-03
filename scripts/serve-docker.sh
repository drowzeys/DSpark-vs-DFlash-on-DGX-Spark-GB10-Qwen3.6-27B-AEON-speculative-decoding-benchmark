#!/usr/bin/env bash
# Serve the AEON base NVFP4 target in the 0.24 container with a chosen draft,
# DSpark Markov patches bind-mounted in (no-op for the DFlash draft).
#   ./serve-docker.sh <name> <draft_dir> [extra vllm args...]
set -euo pipefail
NAME="${1:?name}"; DRAFT_DIR="${2:?draft dir}"; shift 2 || true
IMG="${IMG:-ghcr.io/aeon-7/aeon-vllm-ultimate:2026-07-01-v0.24.0}"
ROOT="${ROOT:-$HOME/aeon27b}"
SP=/usr/local/lib/python3.12/site-packages/vllm
PP="$ROOT/patch-port/v024_patched"
PORT="${PORT:-8010}"
K="${K:-8}"
UTIL="${UTIL:-0.85}"
MAXLEN="${MAXLEN:-8192}"
SEQS="${SEQS:-16}"
SPEC="{\"method\":\"dflash\",\"model\":\"/draft\",\"num_speculative_tokens\":$K,\"draft_sample_method\":\"probabilistic\"}"

docker rm -f "$NAME" >/dev/null 2>&1 || true
exec docker run --rm --name "$NAME" --gpus all --ipc=host --shm-size=16g --net=host \
  -e HF_HUB_OFFLINE=1 -e TORCHINDUCTOR_COMPILE_THREADS="${COMPILE_THREADS:-4}" \
  -e VLLM_LOGGING_LEVEL="${LOGLEVEL:-INFO}" \
  -v "$ROOT/vllm-cache:/root/.cache/vllm" \
  -v "$ROOT/models/target-nvfp4:/model:ro" \
  -v "$DRAFT_DIR:/draft:ro" \
  -v "$PP/qwen3_dflash.py:$SP/model_executor/models/qwen3_dflash.py:ro" \
  -v "$PP/llm_base_proposer.py:$SP/v1/spec_decode/llm_base_proposer.py:ro" \
  --entrypoint vllm "$IMG" serve /model \
    --quantization compressed-tensors --dtype bfloat16 --trust-remote-code \
    --port "$PORT" --max-model-len "$MAXLEN" --max-num-seqs "$SEQS" \
    --gpu-memory-utilization "$UTIL" --mamba-cache-dtype float32 \
    --served-model-name aeon27b --chat-template /model/chat_template.jinja \
    --speculative-config "$SPEC" --attention-backend flash_attn "$@"
