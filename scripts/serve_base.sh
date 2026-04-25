#!/usr/bin/env bash
# Serve the FP16 base model with the same vLLM optimizations as the quantized
# instance, so the comparison is fair.
#
# Flag rationale:
#   --enable-prefix-caching       reuse KV cache across requests sharing a prefix.
#                                 Big win for Adventure mode (shared system prompt).
#   --enable-chunked-prefill      interleave prefill + decode for better TTFT
#                                 under concurrent load.
#   --max-num-seqs 64             cap concurrent sequences for stable latency on 14B.
#   --gpu-memory-utilization 0.9  leave headroom for KV cache growth.
#   --max-model-len 4096          keep seq length tractable on a single GPU.
set -euo pipefail

args=(
    "${BASE_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
    --port "${BASE_PORT:-8000}"
    --enable-prefix-caching
    --enable-chunked-prefill
    --max-num-seqs "${MAX_NUM_SEQS:-64}"
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.9}"
    --max-model-len "${MAX_MODEL_LEN:-4096}"
    --dtype "${BASE_DTYPE:-auto}"
)

if [[ -n "${BASE_KV_CACHE_DTYPE:-}" ]]; then
    args+=(--kv-cache-dtype "${BASE_KV_CACHE_DTYPE}")
fi

exec vllm serve "${args[@]}"
