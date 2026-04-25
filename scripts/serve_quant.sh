#!/usr/bin/env bash
# Serve the quantized model. Same flags as serve_base.sh PLUS:
#   --quantization compressed-tensors   tells vLLM the checkpoint is LLM-Compressor format
#
# Set QUANT_KV_CACHE_DTYPE=fp8 to also quantize KV cache. Keep it unset for a
# cleaner weight-quantization-only comparison against the base model.
set -euo pipefail

args=(
    "${QUANT_MODEL:-./models/qwen2.5-14b-w4a16}"
    --port "${QUANT_PORT:-8001}"
    --quantization compressed-tensors
    --enable-prefix-caching
    --enable-chunked-prefill
    --max-num-seqs "${MAX_NUM_SEQS:-64}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}"
    --max-model-len "${MAX_MODEL_LEN:-4096}"
    --dtype "${QUANT_DTYPE:-auto}"
)

if [[ -n "${QUANT_KV_CACHE_DTYPE:-}" ]]; then
    args+=(--kv-cache-dtype "${QUANT_KV_CACHE_DTYPE}")
fi

exec vllm serve "${args[@]}"
