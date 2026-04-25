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
# 1. quantize the base model (try any recipe in recipes/)
python scripts/quantize.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --recipe recipes/w4a16_gptq.yaml \
    --output ./models/qwen2.5-14b-w4a16

# 2. serve both models with optimization flags pre-baked
./scripts/serve_base.sh    # FP16, port 8000
./scripts/serve_quant.sh   # INT4, port 8001
# optional extra KV-cache compression:
# QUANT_KV_CACHE_DTYPE=fp8 ./scripts/serve_quant.sh

# 3. measure everything
python -m gauntlet.bench.run         --tasks mmlu,gsm8k,humaneval   # accuracy
python -m gauntlet.bench.throughput  --concurrency 8 --requests 32  # speed + latency
python -m gauntlet.bench.memory      --base-pid <pid> --quant-pid <pid>

# 4. one-glance summary table
python scripts/summarize.py

# 5. launch the arena
python -m gauntlet.app
```

## Repo layout

```
gauntlet/
  app.py              # Gradio UI: Adventure + Break It + Dashboard tabs
  clients.py          # two OpenAI clients (vLLM :8000 / :8001) + parallel_stream
  adventure/
    dm.py             # dungeon-master prompt + game state
    consistency.py    # inventory + death tracker, contradiction detection
  break_it/
    challenges.py     # 11 challenges across 7 categories
    scoring.py        # auto-graders (JSON, math, schema, multilingual, etc.)
  bench/
    run.py            # lm-eval-harness wrapper (MMLU/GSM8K/HumanEval)
    throughput.py     # async load test (TTFT, ITL, throughput)
    memory.py         # nvidia-smi snapshot per process
    plots.py          # Pareto, radar, cost charts
scripts/
  quantize.py         # LLM Compressor one-shot runner
  serve_base.sh       # vLLM launch: base FP16 + optimization flags
  serve_quant.sh      # vLLM launch: quantized + extra flags (FP8 KV cache)
  summarize.py        # prints the headline "quantization tax" table
recipes/
  w4a16_gptq.yaml     # 4-bit weights, 16-bit acts, GPTQ
  w8a8_int8.yaml      # 8-bit weights + acts, SmoothQuant + GPTQ
  fp8_dynamic.yaml    # FP8 weights + acts (H100 native)
results/
  bench.json                # accuracy summary
  throughput_detail.json    # latency + throughput
  memory.json               # GPU memory per process
```

## Quantization

What we tried and why. Each recipe is a one-line swap in `scripts/quantize.py`:

| Recipe                | Bits (W/A) | Method            | Best for                                   |
| --------------------- | ---------- | ----------------- | ------------------------------------------ |
| `w4a16_gptq.yaml`     | 4 / 16     | GPTQ              | Max memory savings; T4/4090-class GPUs     |
| `w8a8_int8.yaml`      | 8 / 8      | SmoothQuant + GPTQ| Better math/reasoning preservation         |
| `fp8_dynamic.yaml`    | 8 / 8 (FP) | FP8 dynamic       | H100 / Ada Lovelace — native fast path     |

- **Calibration data:** 512 samples from `open_platypus` (mixed reasoning + instruction).
- **Memory:** Qwen 2.5 14B baseline ~28 GB → W4A16 ~8 GB → W8A8 ~14 GB → FP8 ~14 GB (KV cache shrinks separately, see below).
- **Reproducibility:** `scripts/quantize.py` is deterministic given the recipe + calibration seed.
- **The Gauntlet runs all three** so the demo has data to argue *which* quantization actually pays off — not just "quant beats base."

## Serving optimization

Quantization is half the story; vLLM flags are the other half. `scripts/serve_*.sh` apply both layers consistently across the comparison:

| Flag                              | Purpose                                                  |
| --------------------------------- | -------------------------------------------------------- |
| `--enable-prefix-caching`         | Reuse KV cache across requests sharing a prefix. Big win for the Adventure system prompt that repeats every turn. |
| `--enable-chunked-prefill`        | Interleave prefill + decode → better TTFT under load.    |
| `--max-num-seqs 64`               | Cap concurrent sequences for predictable tail latency.   |
| `--gpu-memory-utilization 0.9`    | Headroom for KV cache growth.                            |
| `--max-model-len 4096`            | Tractable seq length on a single GPU.                    |
| `--quantization compressed-tensors` *(quant only)* | Tells vLLM the weights are LLM-Compressor format. |
| `QUANT_KV_CACHE_DTYPE=fp8` *(optional quant serving-stack test)* | Shrinks the KV cache too, on top of weight quantization; leave unset for a cleaner weight-only comparison. |

Both servers run with identical flags except `--quantization compressed-tensors` by default, so the base comparison isolates weight quantization. If you enable FP8 KV cache for the quant server only, report it as an additional serving-stack optimization rather than pure weight-quantization gain.

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
