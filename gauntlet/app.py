"""
The Gauntlet — Gradio app.

Three tabs:
  1. Adventure — side-by-side dungeon master, base FP16 vs quantized INT4.
  2. Break It — adversarial challenges with auto-grading + manual judge votes.
  3. Dashboard — Pareto / radar / cost charts from results/bench.json.

Both models are served via vLLM's OpenAI-compatible API on different ports;
the streaming generators run in threads so both responses advance in parallel
and the speed gap is visible in real time.
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from gauntlet.adventure.consistency import ConsistencyTracker
from gauntlet.adventure.dm import OPENING, append_dm, append_player, initial_messages
from gauntlet.bench.plots import cost_chart, load_summary, pareto_chart, radar_chart
from gauntlet.break_it.challenges import CHALLENGES, by_category, needle_haystack_prompt
from gauntlet.break_it.scoring import grade
from gauntlet.clients import make_clients, parallel_stream

BASE, QUANT = make_clients()


# ---------------------------------------------------------------------------
# Adventure
# ---------------------------------------------------------------------------

def adventure_initial_state() -> dict:
    return {
        "base_messages": initial_messages(),
        "quant_messages": initial_messages(),
        "base_tracker": ConsistencyTracker(),
        "quant_tracker": ConsistencyTracker(),
        "turn": 0,
        "metrics": {"base": [], "quant": []},
    }


def _format_metrics(state: dict) -> str:
    def line(label: str, history: list, tracker: ConsistencyTracker) -> str:
        if not history:
            return f"**{label}** — _no turns yet_"
        last = history[-1]
        avg_tps = sum(m["tokens_per_sec"] for m in history) / len(history)
        ttft = f"{last['ttft']:.2f}s" if last.get("ttft") is not None else "—"
        return (
            f"**{label}** — last: {last['tokens']} tok / {last['elapsed']:.2f}s "
            f"({last['tokens_per_sec']:.1f} tok/s, TTFT {ttft}) · "
            f"avg {avg_tps:.1f} tok/s · "
            f"inventory: {len(tracker.inventory)} · "
            f"contradictions: {tracker.summary()['total_contradictions']}"
        )

    return "\n\n".join(
        [
            line(BASE.label, state["metrics"]["base"], state["base_tracker"]),
            line(QUANT.label, state["metrics"]["quant"], state["quant_tracker"]),
        ]
    )


def adventure_step(player_action: str, state: dict):
    if not player_action.strip():
        yield gr.update(), gr.update(), _format_metrics(state), state
        return

    state["turn"] += 1
    state["base_messages"] = append_player(state["base_messages"], player_action)
    state["quant_messages"] = append_player(state["quant_messages"], player_action)

    last_base, last_quant = "", ""
    for base_state, quant_state in parallel_stream(
        BASE,
        QUANT,
        state["base_messages"],
        max_tokens=400,
        temperature=0.8,
    ):
        last_base = base_state["text"]
        last_quant = quant_state["text"]
        # Show in-flight metrics in the panel header line.
        live = (
            f"**{BASE.label}** — {base_state['tokens']} tok in {base_state['elapsed']:.1f}s"
            f"\n\n**{QUANT.label}** — {quant_state['tokens']} tok in {quant_state['elapsed']:.1f}s"
        )
        yield last_base, last_quant, live, state

    state["base_messages"] = append_dm(state["base_messages"], last_base)
    state["quant_messages"] = append_dm(state["quant_messages"], last_quant)
    state["base_tracker"].observe(last_base)
    state["quant_tracker"].observe(last_quant)

    if base_state.get("done"):
        state["metrics"]["base"].append({"turn": state["turn"], **base_state["done"]})
    if quant_state.get("done"):
        state["metrics"]["quant"].append({"turn": state["turn"], **quant_state["done"]})

    yield last_base, last_quant, _format_metrics(state), state


def adventure_reset():
    state = adventure_initial_state()
    return OPENING, OPENING, _format_metrics(state), state


# ---------------------------------------------------------------------------
# Break It
# ---------------------------------------------------------------------------

def break_it_initial_state() -> dict:
    return {"scores": {c.id: {"base": None, "quant": None} for c in CHALLENGES}}


def _resolve_prompt(challenge) -> str:
    if challenge.prompt == "<<NEEDLE_HAYSTACK>>":
        return needle_haystack_prompt()
    return challenge.prompt


def run_challenge(challenge_id: str, state: dict):
    challenge = next(c for c in CHALLENGES if c.id == challenge_id)
    prompt = _resolve_prompt(challenge)
    messages = [{"role": "user", "content": prompt}]

    last_base, last_quant = "", ""
    base_state, quant_state = {}, {}
    for base_state, quant_state in parallel_stream(
        BASE, QUANT, messages, max_tokens=600, temperature=0.7
    ):
        last_base = base_state["text"]
        last_quant = quant_state["text"]
        yield last_base, last_quant, "", state, gr.update(visible=False, value="")

    if challenge.grader == "judge_vote":
        verdict = (
            f"**{challenge.category}** — manual vote required.\n\n"
            f"_{challenge.description or 'Use the buttons below to score each model.'}_"
        )
        state["scores"][challenge_id]["base"] = "pending"
        state["scores"][challenge_id]["quant"] = "pending"
        yield last_base, last_quant, verdict + "\n\n" + _scoreboard(state), state, gr.update(
            visible=True, value=challenge_id
        )
        return

    base_grade = grade(last_base, challenge.grader, **challenge.grader_args)
    quant_grade = grade(last_quant, challenge.grader, **challenge.grader_args)
    state["scores"][challenge_id]["base"] = bool(base_grade["correct"])
    state["scores"][challenge_id]["quant"] = bool(quant_grade["correct"])
    base_mark = "PASS" if base_grade["correct"] else "FAIL"
    quant_mark = "PASS" if quant_grade["correct"] else "FAIL"
    verdict = (
        f"**{challenge.category}** · grader=`{challenge.grader}`\n\n"
        f"- {BASE.label}: **{base_mark}** — `{base_grade}`\n"
        f"- {QUANT.label}: **{quant_mark}** — `{quant_grade}`"
    )
    yield last_base, last_quant, verdict + "\n\n" + _scoreboard(state), state, gr.update(
        visible=False, value=""
    )


def _scoreboard(state: dict) -> str:
    cats = by_category()
    lines = [
        "### Scoreboard",
        "",
        f"| Category | {BASE.label} | {QUANT.label} |",
        "|---|---|---|",
    ]
    for cat, items in cats.items():
        b = sum(1 for c in items if state["scores"][c.id]["base"] is True)
        q = sum(1 for c in items if state["scores"][c.id]["quant"] is True)
        n = len(items)
        lines.append(f"| {cat} | {b}/{n} | {q}/{n} |")
    return "\n".join(lines)


def cast_judge_vote(challenge_id: str, base_correct: bool, quant_correct: bool, state: dict):
    if not challenge_id:
        return _scoreboard(state), state, gr.update(visible=False)
    state["scores"][challenge_id]["base"] = bool(base_correct)
    state["scores"][challenge_id]["quant"] = bool(quant_correct)
    return _scoreboard(state), state, gr.update(visible=False)


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

def refresh_dashboard():
    summary = load_summary()
    radar = radar_chart(summary)

    points: list[dict] = []
    throughputs: dict[str, float] = {}
    tp_path = Path("results/throughput.json")
    if tp_path.exists():
        tp = json.loads(tp_path.read_text())
        for label, tps in tp.items():
            scores = [v for v in summary.get("results", {}).get(label, {}).values() if v]
            mean_acc = sum(scores) / len(scores) if scores else 0.0
            points.append({"label": label, "throughput": tps, "accuracy": mean_acc})
            throughputs[label] = tps

    return radar, pareto_chart(points), cost_chart(throughputs)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="The Gauntlet", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# The Gauntlet\n*Stress-testing quantized models through play.*")

        with gr.Tabs():
            # ---------- Adventure ----------
            with gr.Tab("Adventure"):
                adv_state = gr.State(value=adventure_initial_state())
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(f"### {BASE.label}")
                        base_panel = gr.Markdown(value=OPENING)
                    with gr.Column():
                        gr.Markdown(f"### {QUANT.label}")
                        quant_panel = gr.Markdown(value=OPENING)
                metrics_md = gr.Markdown(value=_format_metrics(adventure_initial_state()))
                with gr.Row():
                    action = gr.Textbox(placeholder="What do you do?", scale=4, label="")
                    submit = gr.Button("Act", variant="primary", scale=1)
                    reset = gr.Button("Reset", scale=1)

                submit.click(
                    adventure_step,
                    inputs=[action, adv_state],
                    outputs=[base_panel, quant_panel, metrics_md, adv_state],
                ).then(lambda: "", outputs=action)
                action.submit(
                    adventure_step,
                    inputs=[action, adv_state],
                    outputs=[base_panel, quant_panel, metrics_md, adv_state],
                ).then(lambda: "", outputs=action)
                reset.click(
                    adventure_reset,
                    outputs=[base_panel, quant_panel, metrics_md, adv_state],
                )

            # ---------- Break It ----------
            with gr.Tab("Break It"):
                bi_state = gr.State(value=break_it_initial_state())
                challenge_picker = gr.Dropdown(
                    choices=[(f"[{c.category}] {c.id}", c.id) for c in CHALLENGES],
                    label="Challenge",
                    value=CHALLENGES[0].id,
                )
                run_btn = gr.Button("Run challenge", variant="primary")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(f"### {BASE.label}")
                        base_out = gr.Markdown()
                    with gr.Column():
                        gr.Markdown(f"### {QUANT.label}")
                        quant_out = gr.Markdown()
                verdict = gr.Markdown()
                vote_id = gr.Textbox(visible=False)
                with gr.Row(visible=False) as vote_row:
                    base_correct = gr.Checkbox(label=f"{BASE.label} correct?")
                    quant_correct = gr.Checkbox(label=f"{QUANT.label} correct?")
                    cast = gr.Button("Cast vote", variant="primary")

                run_btn.click(
                    run_challenge,
                    inputs=[challenge_picker, bi_state],
                    outputs=[base_out, quant_out, verdict, bi_state, vote_id],
                ).then(
                    lambda vid: gr.update(visible=bool(vid)),
                    inputs=vote_id,
                    outputs=vote_row,
                )
                cast.click(
                    cast_judge_vote,
                    inputs=[vote_id, base_correct, quant_correct, bi_state],
                    outputs=[verdict, bi_state, vote_row],
                )

            # ---------- Dashboard ----------
            with gr.Tab("Dashboard"):
                refresh = gr.Button("Refresh")
                radar = gr.Plot()
                pareto = gr.Plot()
                cost = gr.Plot()
                refresh.click(refresh_dashboard, outputs=[radar, pareto, cost])
                demo.load(refresh_dashboard, outputs=[radar, pareto, cost])

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860)
