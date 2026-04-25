"""
Run lm-evaluation-harness against the two vLLM endpoints and dump a unified
summary to results/bench.json.

Both endpoints expose the OpenAI-compatible API, so we use lm-eval's
`local-completions` model wrapper.

Example (smoke test, 20 examples per task):
    python -m gauntlet.bench.run --tasks gsm8k --limit 20

Full eval:
    python -m gauntlet.bench.run --tasks mmlu,gsm8k,humaneval
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

DEFAULT_TASKS = ["mmlu", "gsm8k", "humaneval"]


def run_lm_eval(
    model_label: str,
    base_url: str,
    model_id: str,
    tasks: list[str],
    output_dir: Path,
    num_fewshot: int = 0,
    limit: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "lm_eval",
        "--model", "local-completions",
        "--model_args",
        f"base_url={base_url}/completions,model={model_id},num_concurrent=4,tokenized_requests=False",
        "--tasks", ",".join(tasks),
        "--output_path", str(output_dir / model_label),
    ]
    if num_fewshot:
        cmd += ["--num_fewshot", str(num_fewshot)]
    if limit:
        cmd += ["--limit", str(limit)]
    print(f"\n=== Running {model_label} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def collect(output_dir: Path) -> dict:
    results: dict = {}
    if not output_dir.exists():
        return results
    for model_dir in output_dir.iterdir():
        if not model_dir.is_dir():
            continue
        json_files = sorted(model_dir.rglob("results_*.json"))
        if not json_files:
            continue
        data = json.loads(json_files[-1].read_text())
        results[model_dir.name] = {
            task: _headline(metrics) for task, metrics in data.get("results", {}).items()
        }
    return results


def _headline(metrics: dict) -> float | None:
    for k in ("acc,none", "acc_norm,none", "exact_match,none", "pass@1,none", "pass@1"):
        if k in metrics:
            return metrics[k]
    for k, v in metrics.items():
        if isinstance(v, float) and "stderr" not in k:
            return v
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--quant-url", default="http://localhost:8001/v1")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-14B-Instruct")
    p.add_argument("--quant-model", default="./models/qwen2.5-14b-w4a16")
    p.add_argument("--output", default="results/lm_eval")
    p.add_argument("--summary", default="results/bench.json")
    p.add_argument("--num-fewshot", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="Examples per task (smoke test)")
    args = p.parse_args()

    tasks = args.tasks.split(",")
    out = Path(args.output)
    started = time.time()

    run_lm_eval("base", args.base_url, args.base_model, tasks, out, args.num_fewshot, args.limit)
    run_lm_eval("quant", args.quant_url, args.quant_model, tasks, out, args.num_fewshot, args.limit)

    summary = {
        "tasks": tasks,
        "models": {"base": args.base_model, "quant": args.quant_model},
        "results": collect(out),
        "elapsed_seconds": time.time() - started,
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {args.summary}")


if __name__ == "__main__":
    main()
