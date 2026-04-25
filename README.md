# The Gauntlet

> Stress-testing quantized models through play.

A side-by-side arena that pits a **base FP16 model** against its **INT4 quantized counterpart** — same prompts, two vLLM instances, live metrics. Two modes: a text-RPG dungeon master ("Adventure") and a judge-driven challenge gauntlet ("Break It"). Backed by a benchmark dashboard that maps the full accuracy / speed / memory tradeoff.

> **Benchmarks tell you a number. The Gauntlet lets you *feel* the difference.**

## The narrative arc

1. **"Look how good the quantized model is."** Adventure mode — judges play a text RPG. The quantized model responds 2–3× faster; both feel coherent. Fun.
2. **"Now let's find where it breaks."** Break It mode — judges throw logic puzzles, math traps, JSON formatting, consistency attacks. The cracks appear systematically.
3. **"Here's the real tradeoff."** Dashboard with Pareto frontier, per-task radar, cost-per-token, and a *quantization tax* summary.

## Modes

### Adventure
A shared text-RPG scenario rendered in two panels — same seed, same world state, same player input, different model. Streaming responses make the speed gap visible in real time. A consistency tracker watches whether the model remembers earlier context (the sword you picked up three turns ago).

This is the stress test nobody runs: **does quantization break long-horizon creative coherence?**

### Break It (adversarial)
Categorized challenges with auto-scoring where possible:

| Category            | Auto-scored?  | Example                                            |
| ------------------- | ------------- | -------------------------------------------------- |
| Logic puzzles       | judge vote    | "Two doors, one guard always lies..."              |
| Math                | yes           | "3 apples − 5 = ?" trap variants                   |
| Instruction follow  | partial       | "Explain quantum computing only in haiku"          |
| Consistency         | judge vote    | "You said X earlier — defend Y" (it never said X)  |
| Context stress      | yes           | progressively longer prompts                       |
| Structured output   | yes           | strict JSON schemas                                |
| Multilingual        | yes           | "Answer in French, reason in English"              |

Live scoreboard updates each round.

## Architecture

```
                    ┌──────────────────────────────────────┐
                    │           The Gauntlet UI            │
                    │         (Gradio — two panels)        │
                    └──────────┬───────────────┬───────────┘
                               │  same prompt  │
                               ▼               ▼
                    ┌────────────────┐   ┌────────────────┐
                    │  vLLM :8000    │   │  vLLM :8001    │
                    │  Qwen2.5-14B   │   │  Qwen2.5-14B   │
                    │   FP16 (base)  │   │   INT4 (quant) │
                    └────────┬───────┘   └────────┬───────┘
                             │                    │
                             └─────────┬──────────┘
                                       ▼
                          ┌──────────────────────┐
                          │   Metrics + scoring  │
                          │  · tok/s, TTFT, mem  │
                          │  · accuracy per task │
                          │  · Pareto + radar    │
                          └──────────────────────┘
```

Both models are served via vLLM's OpenAI-compatible API on different ports; the app calls them with the standard `openai` Python client.

## Stack

| Layer          | Tool                                                                          |
| -------------- | ----------------------------------------------------------------------------- |
| Serving        | vLLM, two instances, OpenAI-compatible API                                    |
| Model          | `Qwen/Qwen2.5-14B-Instruct` (base) + INT4 quantized                           |
| Quantization   | [LLM Compressor](https://github.com/vllm-project/llm-compressor) (W4A16 GPTQ) |
| Benchmarks     | `lm-evaluation-harness` (MMLU, GSM8K, HumanEval)                              |
| UI             | Gradio                                                                        |
| Dashboard      | Plotly inside Gradio                                                          |

## Quick start

```bash
# 1. quantize the base model
python scripts/quantize.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --recipe recipes/w4a16_gptq.yaml \
    --output ./models/qwen2.5-14b-w4a16

# 2. serve both models (two GPUs, or one GPU sequentially)
vllm serve Qwen/Qwen2.5-14B-Instruct --port 8000
vllm serve ./models/qwen2.5-14b-w4a16 --port 8001 \
    --quantization compressed-tensors

# 3. run the offline benchmark suite (populates the dashboard)
python -m gauntlet.bench --tasks mmlu,gsm8k,humaneval

# 4. launch the arena
python -m gauntlet.app
```

## Repo layout

```
gauntlet/
  app.py              # Gradio UI: Adventure + Break It + Dashboard tabs
  clients.py          # two OpenAI clients pointed at vLLM :8000 / :8001
  adventure/
    dm.py             # dungeon-master prompt + game state
    consistency.py    # context-recall tracker
  break_it/
    challenges.py     # prompt bank by category
    scoring.py        # auto-graders (JSON, math, schema, etc.)
  bench/
    run.py            # lm-eval-harness wrapper
    plots.py          # Pareto frontier, radar, cost charts
scripts/
  quantize.py         # LLM Compressor recipe runner
recipes/
  w4a16_gptq.yaml     # quantization config
results/
  bench.json          # benchmark output consumed by the dashboard
```

## Quantization documentation

What we did, why, and how to reproduce it:

- **Method:** GPTQ W4A16 via LLM Compressor.
- **Calibration data:** 512 samples from `open-platypus` (mixed reasoning + instruction-following).
- **Why W4A16 over W8A8 or W4A8:** target was max memory savings while keeping the sm89/sm90 fast paths; W4A16 hits the throughput sweet spot on a single H100 / 4090-class GPU.
- **Memory:** ~28 GB → ~8 GB.
- **Configs tested:** see `recipes/`. We benchmarked W8A8, W4A16-GPTQ, and W4A16-AWQ — GPTQ won the gauntlet.
- **Reproducibility:** `scripts/quantize.py` is deterministic given the recipe file and a fixed calibration seed.

## What we measure

| Metric                   | Tool                          | Why                                  |
| ------------------------ | ----------------------------- | ------------------------------------ |
| MMLU / GSM8K / HumanEval | lm-eval-harness               | Standardized capability snapshot     |
| Throughput (tok/s)       | vLLM `/metrics` + load test   | Real serving cost                    |
| TTFT / inter-token lat.  | client-side timestamps        | UX impact                            |
| GPU memory               | `nvidia-smi` snapshot         | Deployability                        |
| Per-category accuracy    | Break It auto-scoring         | Where the quantization tax lands     |
| Cost / 1M tokens         | derived from throughput       | Bottom-line tradeoff                 |

## Demo script (5 minutes)

1. **Open the app.** "Qwen 2.5 14B, quantized 28 GB → 8 GB with LLM Compressor."
2. **Adventure mode.** "You wake up in a dungeon..." — 3–4 turns. Both coherent. Quantized is visibly faster. Judges smile.
3. **"Now let's break it."** Switch to Break It.
4. Judge picks a category. Scoreboard updates live.
5. Run 2–3 more challenges. A pattern emerges — math and logic degrade, instruction-following and creativity survive.
6. **Dashboard.** "Here's what 3.5× speedup and 70% memory reduction actually cost — and it's not what MMLU would have told you."

## License

MIT — see [LICENSE](./LICENSE).
