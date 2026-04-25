"""
Pull together results/bench.json + throughput_detail.json + memory.json into
a single one-glance "quantization tax" table that prints to stdout.

Run after the three benchmark steps have populated results/.
"""

from __future__ import annotations

import json
from pathlib import Path


def fmt_pct(base: float | None, quant: float | None, higher_is_better: bool) -> str:
    if base in (None, 0) or quant is None:
        return "—"
    delta = (quant - base) / base * 100.0
    sign = "+" if delta >= 0 else ""
    arrow = ""
    if higher_is_better:
        arrow = "↓" if delta < 0 else "↑"
    else:
        arrow = "↑" if delta < 0 else "↓"  # lower-is-better: drop is good
    return f"{sign}{delta:.1f}% {arrow}"


def _load(path: str) -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def _fmt(v, spec: str) -> str:
    if v is None:
        return "—"
    try:
        return format(v, spec)
    except (TypeError, ValueError):
        return str(v)


def main() -> None:
    bench = _load("results/bench.json")
    tp = _load("results/throughput_detail.json")
    mem = _load("results/memory.json")

    rows: list[tuple[str, str, str, str]] = []  # (label, base, quant, delta)

    # --- Accuracy ---
    base_results = bench.get("results", {}).get("base", {})
    quant_results = bench.get("results", {}).get("quant", {})
    for task in sorted(set(base_results) | set(quant_results)):
        b = base_results.get(task)
        q = quant_results.get(task)
        rows.append((f"{task} (acc)", _fmt(b, ".4f"), _fmt(q, ".4f"), fmt_pct(b, q, higher_is_better=True)))

    # --- Throughput / latency ---
    if tp:
        b_tp = tp.get("base", {}).get("aggregate_tokens_per_sec")
        q_tp = tp.get("quant", {}).get("aggregate_tokens_per_sec")
        rows.append(("Throughput (tok/s)", _fmt(b_tp, ".1f"), _fmt(q_tp, ".1f"), fmt_pct(b_tp, q_tp, higher_is_better=True)))

        b_ttft = tp.get("base", {}).get("ttft_p50")
        q_ttft = tp.get("quant", {}).get("ttft_p50")
        rows.append(("TTFT p50 (s)", _fmt(b_ttft, ".3f"), _fmt(q_ttft, ".3f"), fmt_pct(b_ttft, q_ttft, higher_is_better=False)))

        b_itl = tp.get("base", {}).get("itl_p50")
        q_itl = tp.get("quant", {}).get("itl_p50")
        rows.append(("ITL p50 (s)", _fmt(b_itl, ".4f"), _fmt(q_itl, ".4f"), fmt_pct(b_itl, q_itl, higher_is_better=False)))

    # --- Memory ---
    if mem:
        b_mem = mem.get("base", {}).get("used_mib")
        q_mem = mem.get("quant", {}).get("used_mib")
        rows.append(("GPU memory (MiB)", _fmt(b_mem, ""), _fmt(q_mem, ""), fmt_pct(b_mem, q_mem, higher_is_better=False)))

    if not rows:
        print("No results yet. Run benchmarks first:")
        print("  python -m gauntlet.bench.run        # accuracy")
        print("  python -m gauntlet.bench.throughput # speed")
        print("  python -m gauntlet.bench.memory     # memory")
        return

    print(f"\n{'Metric':<22} {'Base (FP16)':>14} {'Quant':>14}  {'Δ':<14}")
    print("-" * 70)
    for label, b_str, q_str, delta in rows:
        print(f"{label:<22} {b_str:>14} {q_str:>14}  {delta:<14}")
    print()


if __name__ == "__main__":
    main()
