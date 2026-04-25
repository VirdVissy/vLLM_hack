"""
Two OpenAI-compatible clients pointed at two vLLM endpoints.

The base FP16 model and the INT4 quantized model are served on different ports.
The rest of the app calls them through a uniform interface so we can stream both
at once and time each side identically.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Iterator

from openai import OpenAI


@dataclass
class ModelClient:
    name: str
    label: str
    model_id: str
    client: OpenAI


def make_clients() -> tuple[ModelClient, ModelClient]:
    base = ModelClient(
        name="base",
        label=os.getenv("BASE_LABEL", "FP16 (base)"),
        model_id=os.getenv("BASE_MODEL", "Qwen/Qwen2.5-14B-Instruct"),
        client=OpenAI(
            base_url=os.getenv("BASE_URL", "http://localhost:8000/v1"),
            api_key=os.getenv("BASE_API_KEY", "EMPTY"),
        ),
    )
    quant = ModelClient(
        name="quant",
        label=os.getenv("QUANT_LABEL", "INT4 (quantized)"),
        model_id=os.getenv("QUANT_MODEL", "./models/qwen2.5-14b-w4a16"),
        client=OpenAI(
            base_url=os.getenv("QUANT_URL", "http://localhost:8001/v1"),
            api_key=os.getenv("QUANT_API_KEY", "EMPTY"),
        ),
    )
    return base, quant


def stream_chat(mc: ModelClient, messages: list[dict], **kwargs) -> Iterator[dict]:
    """Stream a chat completion. Yields delta events and a final done event with metrics."""
    start = time.perf_counter()
    first_token_at: float | None = None
    tokens = 0
    accumulated = ""

    stream = mc.client.chat.completions.create(
        model=mc.model_id,
        messages=messages,
        stream=True,
        **kwargs,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        if not content:
            continue
        if first_token_at is None:
            first_token_at = time.perf_counter()
        tokens += 1
        accumulated += content
        yield {
            "type": "delta",
            "content": content,
            "accumulated": accumulated,
            "tokens": tokens,
            "elapsed": time.perf_counter() - start,
            "ttft": (first_token_at - start) if first_token_at else None,
        }

    elapsed = time.perf_counter() - start
    yield {
        "type": "done",
        "accumulated": accumulated,
        "tokens": tokens,
        "elapsed": elapsed,
        "ttft": (first_token_at - start) if first_token_at else None,
        "tokens_per_sec": (tokens / elapsed) if elapsed > 0 else 0.0,
    }


def parallel_stream(
    base: ModelClient,
    quant: ModelClient,
    messages: list[dict],
    **kwargs,
) -> Iterator[tuple[dict, dict]]:
    """
    Run base + quantized streams concurrently in threads.

    Yields (base_state, quant_state) tuples each time either side advances.
    Each state dict has keys: text, tokens, elapsed, ttft, done (None until finished),
    and tokens_per_sec (None until finished).
    """
    q: queue.Queue = queue.Queue()

    def worker(label: str, mc: ModelClient) -> None:
        try:
            for ev in stream_chat(mc, messages, **kwargs):
                q.put((label, ev))
        except Exception as e:
            q.put((label, {"type": "error", "error": str(e)}))
        finally:
            q.put((label, None))

    threading.Thread(target=worker, args=("base", base), daemon=True).start()
    threading.Thread(target=worker, args=("quant", quant), daemon=True).start()

    state = {
        "base": {"text": "", "tokens": 0, "elapsed": 0.0, "ttft": None, "done": None, "tokens_per_sec": None, "error": None},
        "quant": {"text": "", "tokens": 0, "elapsed": 0.0, "ttft": None, "done": None, "tokens_per_sec": None, "error": None},
    }
    finished = {"base": False, "quant": False}

    while not (finished["base"] and finished["quant"]):
        label, ev = q.get()
        if ev is None:
            finished[label] = True
            continue
        s = state[label]
        if ev["type"] == "delta":
            s["text"] = ev["accumulated"]
            s["tokens"] = ev["tokens"]
            s["elapsed"] = ev["elapsed"]
            s["ttft"] = ev["ttft"]
        elif ev["type"] == "done":
            s["text"] = ev["accumulated"]
            s["tokens"] = ev["tokens"]
            s["elapsed"] = ev["elapsed"]
            s["ttft"] = ev["ttft"]
            s["tokens_per_sec"] = ev["tokens_per_sec"]
            s["done"] = ev
        elif ev["type"] == "error":
            s["error"] = ev["error"]
        yield state["base"].copy(), state["quant"].copy()
