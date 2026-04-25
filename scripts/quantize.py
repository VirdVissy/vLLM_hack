"""
Run LLM Compressor against a base model with a given recipe.

Defaults reproduce the W4A16 GPTQ run we describe in the README:

    python scripts/quantize.py \
        --model Qwen/Qwen2.5-14B-Instruct \
        --recipe recipes/w4a16_gptq.yaml \
        --output ./models/qwen2.5-14b-w4a16
"""

import argparse

from llmcompressor.transformers import oneshot


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF id or local path of base model")
    p.add_argument("--recipe", required=True, help="Path to LLM Compressor YAML recipe")
    p.add_argument("--output", required=True, help="Where to write the quantized model")
    p.add_argument("--dataset", default="open_platypus", help="Calibration dataset name")
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--max-seq-length", type=int, default=2048)
    args = p.parse_args()

    oneshot(
        model=args.model,
        dataset=args.dataset,
        recipe=args.recipe,
        output_dir=args.output,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
