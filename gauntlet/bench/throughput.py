"""
Concurrent load test against the two vLLM endpoints.

Measures TTFT, inter-token latency, per-request throughput, and aggregate
tokens/sec under a fixed concurrency budget. Writes:

  results/throughput.json           — flat {label: tok/s} for the dashboard
  results/throughput_detail.json    — full per-model summary (p50, p95, etc.)

Example:
    python -m gauntlet.bench.throughput --concurrency 8 --requests 32
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

PROMPT = (
    "Write a 200-word continuation to the following story. Be vivid and specific.\n\n"
    "The lighthouse keeper had not seen the boat in three days, but tonight "
    "something was different. The fog itself seemed to be listening."
)


async def one_request(client: AsyncOpenAI, model: str, max_tokens: int) -> dict:
    started = time.perf_counter()
    first_token_at: float | None = None
    last_token_at: float | None = None
    chunks = 0
    completion_tokens: int | None = None

    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.7,
        "stream_options": {"include_usage": True},
    }
    try:
        stream = await client.chat.completions.create(**request_kwargs)
    except Exception:
        request_kwargs.pop("stream_options", None)
        stream = await client.chat.completions.create(**request_kwargs)

    async for chunk in stream:
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            completion_tokens = getattr(usage, "completion_tokens", None)
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        text = getattr(delta, "content", None)
        if not text:
            continue
        now = time.perf_counter()
        if first_token_at is None:
            first_token_at = now
        last_token_at = now
        chunks += 1

    end = time.perf_counter()
    tokens = completion_tokens or chunks
    return {
        "ttft": (first_token_at - started) if first_token_at else None,
        "itl": (
            (last_token_at - first_token_at) / max(1, chunks - 1)
            if (first_token_at and last_token_at and chunks > 1)
            else None
        ),
        "elapsed": end - started,
        "tokens": tokens,
        "chunks": chunks,
        "token_source": "usage" if completion_tokens is not None else "stream_chunks",
        "throughput": tokens / (end - started) if end > started else 0.0,
    }


async def run_benchmark(
    label: str,
    base_url: str,
    api_key: str,
    model: str,
    concurrency: int,
    num_requests: int,
    max_tokens: int,
) -> dict:
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    sem = asyncio.Semaphore(concurrency)

    async def worker() -> dict:
        async with sem:
            try:
                return await one_request(client, model, max_tokens)
            except Exception as e:
                return {"error": str(e), "tokens": 0, "throughput": 0.0}

    wall_start = time.perf_counter()
    try:
        results = await asyncio.gather(*[worker() for _ in range(num_requests)])
        wall_elapsed = time.perf_counter() - wall_start
        return summarize(label, results, concurrency, wall_elapsed)
    finally:
        await client.close()


def summarize(label: str, results: list[dict], concurrency: int, wall_elapsed: float) -> dict:
    successes = [r for r in results if "error" not in r]
    errors = [r["error"] for r in results if "error" in r]
    ttfts = [r["ttft"] for r in successes if r["ttft"] is not None]
    itls = [r["itl"] for r in successes if r["itl"] is not None]
    throughputs = [r["throughput"] for r in successes]
    total_tokens = sum(r["tokens"] for r in successes)
    token_sources = sorted({r.get("token_source", "unknown") for r in successes})
    return {
        "label": label,
        "concurrency": concurrency,
        "requests": len(results),
        "successful_requests": len(successes),
        "failed_requests": len(errors),
        "sample_errors": errors[:3],
        "token_sources": token_sources,
        "ttft_p50": statistics.median(ttfts) if ttfts else None,
        "ttft_p95": _p95(ttfts),
        "itl_p50": statistics.median(itls) if itls else None,
        "itl_p95": _p95(itls),
        "per_request_throughput_p50": statistics.median(throughputs) if throughputs else None,
        "aggregate_tokens_per_sec": (total_tokens / wall_elapsed) if wall_elapsed > 0 else 0.0,
        "wall_seconds": wall_elapsed,
        "total_tokens": total_tokens,
    }


def _p95(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[int(0.95 * (len(s) - 1))]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--quant-url", default="http://localhost:8001/v1")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-14B-Instruct")
    p.add_argument("--quant-model", default="./models/qwen2.5-14b-w4a16")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--requests", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--only", choices=["base", "quant", "both"], default="both")
    p.add_argument("--output", default="results/throughput.json")
    p.add_argument("--detailed", default="results/throughput_detail.json")
    args = p.parse_args()

    summary: dict = {}
    if args.only in ("base", "both"):
        print(f"[base] {args.base_model} on {args.base_url}")
        summary["base"] = asyncio.run(
            run_benchmark(
                "base", args.base_url, args.api_key, args.base_model,
                args.concurrency, args.requests, args.max_tokens,
            )
        )
        print(json.dumps(summary["base"], indent=2))
    if args.only in ("quant", "both"):
        print(f"\n[quant] {args.quant_model} on {args.quant_url}")
        summary["quant"] = asyncio.run(
            run_benchmark(
                "quant", args.quant_url, args.api_key, args.quant_model,
                args.concurrency, args.requests, args.max_tokens,
            )
        )
        print(json.dumps(summary["quant"], indent=2))

    Path(args.detailed).parent.mkdir(parents=True, exist_ok=True)
    Path(args.detailed).write_text(json.dumps(summary, indent=2))

    flat = {label: data["aggregate_tokens_per_sec"] for label, data in summary.items()}
    if Path(args.output).exists():
        existing = json.loads(Path(args.output).read_text())
        existing.update(flat)
        flat = existing
    Path(args.output).write_text(json.dumps(flat, indent=2))
    print(f"\nWrote {args.output} and {args.detailed}")


if __name__ == "__main__":
    main()
