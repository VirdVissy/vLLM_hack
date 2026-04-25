"""Dashboard plots — radar (per-task accuracy), Pareto (speed vs. quality), cost."""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go


def load_summary(path: str = "results/bench.json") -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def radar_chart(summary: dict) -> go.Figure:
    results = summary.get("results", {})
    if not results:
        return _placeholder("No benchmark results yet — run `python -m gauntlet.bench.run`")

    tasks = sorted({t for model in results.values() for t in model.keys()})
    if not tasks:
        return _placeholder("No benchmark task scores found in results/bench.json")
    fig = go.Figure()
    for model_label, scores in results.items():
        values = [(scores.get(t) or 0.0) for t in tasks]
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=tasks + [tasks[0]],
                fill="toself",
                name=model_label,
            )
        )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Per-task accuracy",
        showlegend=True,
    )
    return fig


def pareto_chart(points: list[dict]) -> go.Figure:
    if not points:
        return _placeholder("Pareto — populate results/throughput.json")
    fig = go.Figure()
    for p in points:
        fig.add_trace(
            go.Scatter(
                x=[p["throughput"]],
                y=[p["accuracy"]],
                mode="markers+text",
                marker=dict(size=18),
                text=[p["label"]],
                textposition="top center",
                name=p["label"],
            )
        )
    fig.update_layout(
        xaxis_title="Throughput (tokens / sec)",
        yaxis_title="Accuracy (mean across tasks)",
        title="Pareto frontier — speed vs. quality",
    )
    return fig


def cost_chart(throughputs: dict[str, float], gpu_cost_per_hour: float = 2.0) -> go.Figure:
    if not throughputs:
        return _placeholder("Cost — populate results/throughput.json")
    labels: list[str] = []
    costs: list[float] = []
    for label, tps in throughputs.items():
        cost = (gpu_cost_per_hour / 3600.0) * (1_000_000 / tps) if tps else 0.0
        labels.append(label)
        costs.append(cost)
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=costs,
                text=[f"${c:.3f}" for c in costs],
                textposition="auto",
            )
        ]
    )
    fig.update_layout(
        title=f"Cost per 1M tokens (at ${gpu_cost_per_hour:.2f}/GPU-hour)",
        yaxis_title="USD",
    )
    return fig


def _placeholder(msg: str) -> go.Figure:
    return go.Figure().update_layout(title=msg)
