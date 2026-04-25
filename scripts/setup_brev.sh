#!/usr/bin/env bash
# One-shot setup for a fresh Brev / NVIDIA cloud GPU instance.
#
# Walks through the boilerplate: install deps, point HF cache at persistent
# storage, optionally log into HF, run a small smoke-test quantization to
# confirm the toolchain works, then run the real Qwen 14B quantization.
#
# Usage (defaults are sane for a Brev A100/H100 launchable):
#
#     ./scripts/setup_brev.sh                       # full path: deps + smoke + real
#     ./scripts/setup_brev.sh --skip-smoke          # straight to the real quant
#     ./scripts/setup_brev.sh --skip-deps --skip-smoke
#     MODEL=Qwen/Qwen2.5-7B-Instruct ./scripts/setup_brev.sh   # smaller target
#
# Env overrides:
#   PERSISTENT_DIR   where to put HF cache + models (auto-detected if unset)
#   MODEL            base model HF id (default Qwen/Qwen2.5-14B-Instruct)
#   RECIPE           recipe path (default recipes/w4a16_gptq.yaml)
#   OUTPUT           quantized output dir (default ./models/qwen2.5-14b-w4a16)
set -euo pipefail

SKIP_DEPS=false
SKIP_SMOKE=false
SKIP_REAL=false
for arg in "$@"; do
    case "$arg" in
        --skip-deps)  SKIP_DEPS=true  ;;
        --skip-smoke) SKIP_SMOKE=true ;;
        --skip-real)  SKIP_REAL=true  ;;
        -h|--help)
            sed -n '2,18p' "$0"; exit 0 ;;
        *) echo "unknown arg: $arg" >&2; exit 1 ;;
    esac
done

MODEL="${MODEL:-Qwen/Qwen2.5-14B-Instruct}"
RECIPE="${RECIPE:-recipes/w4a16_gptq.yaml}"
OUTPUT="${OUTPUT:-./models/qwen2.5-14b-w4a16}"

# --- 1. Detect persistent storage ---------------------------------------------
if [[ -z "${PERSISTENT_DIR:-}" ]]; then
    for cand in /persistent /workspace /mnt/data; do
        if [[ -d "$cand" && -w "$cand" ]]; then
            PERSISTENT_DIR="$cand"
            break
        fi
    done
    PERSISTENT_DIR="${PERSISTENT_DIR:-$HOME}"
fi
echo ">> Persistent dir: $PERSISTENT_DIR"

export HF_HOME="$PERSISTENT_DIR/.cache/huggingface"
mkdir -p "$HF_HOME"
echo ">> HF_HOME=$HF_HOME"

# --- 2. Install deps ----------------------------------------------------------
if ! $SKIP_DEPS; then
    echo ">> Installing requirements.txt"
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo ">> Skipping deps"
fi

# --- 3. HF auth ---------------------------------------------------------------
if ! huggingface-cli whoami > /dev/null 2>&1; then
    echo ">> Not logged into HuggingFace."
    echo "   Run:  huggingface-cli login"
    echo "   (paste a read token from https://huggingface.co/settings/tokens)"
    echo "   Then re-run this script."
    exit 1
fi
echo ">> HF user: $(huggingface-cli whoami)"

# --- 4. GPU sanity ------------------------------------------------------------
if command -v nvidia-smi > /dev/null 2>&1; then
    echo ">> GPU:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo "!! nvidia-smi not found — quantization will fail without a GPU." >&2
    exit 1
fi

# --- 5. Smoke test on a 1.5B model -------------------------------------------
if ! $SKIP_SMOKE; then
    echo ""
    echo ">> Smoke test: quantizing Qwen2.5-1.5B-Instruct (10 min)"
    python scripts/quantize.py \
        --model Qwen/Qwen2.5-1.5B-Instruct \
        --recipe recipes/w4a16_gptq.yaml \
        --output ./models/qwen-smoke \
        --num-samples 64
    echo ">> Smoke test passed."
fi

# --- 6. Real quantization -----------------------------------------------------
if ! $SKIP_REAL; then
    echo ""
    echo ">> Real quant: $MODEL --recipe $RECIPE --output $OUTPUT (30-60 min)"
    echo ">> Run inside tmux/screen if you're worried about SSH disconnects."
    python scripts/quantize.py \
        --model "$MODEL" \
        --recipe "$RECIPE" \
        --output "$OUTPUT"
    echo ""
    echo ">> Done. Quantized model at $OUTPUT"
    echo ">> Next:  ./scripts/serve_quant.sh"
fi
