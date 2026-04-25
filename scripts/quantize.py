"""
Run LLM Compressor against a base model with a given recipe.

Defaults reproduce the W4A16 GPTQ run we describe in the README:

    python scripts/quantize.py \
        --model Qwen/Qwen2.5-14B-Instruct \
        --recipe recipes/w4a16_gptq.yaml \
        --output ./models/qwen2.5-14b-w4a16
"""

import argparse
from pathlib import Path

try:
    from llmcompressor import oneshot
except ImportError:  # llmcompressor<0.5 compatibility
    from llmcompressor.transformers import oneshot


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="HF id or local path of base model")
    p.add_argument("--recipe", required=True, help="Path to LLM Compressor YAML recipe")
    p.add_argument("--output", required=True, help="Where to write the quantized model")
    p.add_argument("--dataset", default="open_platypus", help="Calibration dataset name")
    p.add_argument("--tokenizer", default=None, help="Optional tokenizer id/path")
    p.add_argument("--model-revision", default="main")
    p.add_argument("--precision", default="auto")
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--preprocessing-workers", type=int, default=1)
    p.add_argument("--trust-remote-code", action="store_true")
    args = p.parse_args()

    recipe = Path(args.recipe)
    if not recipe.exists():
        raise SystemExit(f"Recipe not found: {recipe}")

    oneshot(
        model=args.model,
        tokenizer=args.tokenizer,
        model_revision=args.model_revision,
        precision=args.precision,
        trust_remote_code_model=args.trust_remote_code,
        save_compressed=True,
        dataset=args.dataset,
        recipe=str(recipe),
        output_dir=args.output,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=args.num_samples,
        shuffle_calibration_samples=False,
        preprocessing_num_workers=args.preprocessing_workers,
    )


if __name__ == "__main__":
    main()
