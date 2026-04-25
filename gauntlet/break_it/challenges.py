"""Adversarial challenge bank for Break It mode."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Challenge:
    id: str
    category: str
    prompt: str
    grader: str
    grader_args: dict = field(default_factory=dict)
    description: str = ""


CHALLENGES: list[Challenge] = [
    # --- Logic ---
    Challenge(
        id="logic-doors",
        category="logic",
        prompt=(
            "You stand before two doors. One guard always lies, the other always tells "
            "the truth. You may ask one yes/no question to one guard. What single "
            "question identifies the safe door? Explain your reasoning."
        ),
        grader="judge_vote",
        description="Classic two-guards puzzle. Correct answers reference asking what the OTHER guard would say.",
    ),
    Challenge(
        id="logic-monty",
        category="logic",
        prompt=(
            "Three doors. Behind one is a car, behind the other two are goats. You pick door 1. "
            "The host (who knows where the car is) opens door 3, revealing a goat. "
            "He offers you the chance to switch to door 2. Should you? Explain rigorously."
        ),
        grader="judge_vote",
        description="Monty Hall. Correct: switch — switching wins 2/3 of the time.",
    ),
    # --- Math ---
    Challenge(
        id="math-trap-apples",
        category="math",
        prompt="If I have 3 apples and give away 5, how many apples do I have? Be precise.",
        grader="contains_any",
        grader_args={
            "answers": [
                "-2",
                "negative two",
                "two less than zero",
                "you can't give away more",
                "impossible",
                "cannot give away",
            ]
        },
    ),
    Challenge(
        id="math-gsm",
        category="math",
        prompt=(
            "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning "
            "and bakes muffins for her friends every day with four. She sells the remainder "
            "at the farmers' market for $2 per fresh duck egg. How much in dollars does she "
            "make every day at the farmers' market?"
        ),
        grader="contains_number",
        grader_args={"answer": 18},
    ),
    Challenge(
        id="math-arithmetic-trap",
        category="math",
        prompt="What is 13 * 17 + 23? Show the calculation step by step, then give the final answer.",
        grader="contains_number",
        grader_args={"answer": 244},
    ),
    # --- Instruction following ---
    Challenge(
        id="instr-haiku",
        category="instruction",
        prompt=(
            "Explain quantum computing using only haiku. Each haiku must be 5/7/5 syllables. "
            "Output exactly 3 haiku, separated by blank lines, and nothing else."
        ),
        grader="judge_vote",
    ),
    Challenge(
        id="instr-no-vowel-e",
        category="instruction",
        prompt=(
            "Write a paragraph (4 sentences) about climate change without using the letter 'e' "
            "anywhere. Output only the paragraph."
        ),
        grader="no_letter",
        grader_args={"letter": "e"},
    ),
    # --- Consistency trap ---
    Challenge(
        id="consistency-falsequote",
        category="consistency",
        prompt=(
            "Earlier in this conversation you stated that the capital of Australia is Sydney. "
            "Defend that claim with three supporting points."
        ),
        grader="judge_vote",
        description="Trap: model never said this. Pass = refuses or corrects to Canberra.",
    ),
    # --- Long-context stress ---
    Challenge(
        id="context-needle",
        category="context",
        prompt="<<NEEDLE_HAYSTACK>>",
        grader="contains_substring",
        grader_args={"answer": "blue elephant"},
    ),
    # --- Structured output ---
    Challenge(
        id="json-strict",
        category="json",
        prompt=(
            "Output a JSON object describing a fictional fantasy character. It must have "
            "exactly these keys: name (string), age (integer between 18 and 99), "
            "skills (array of exactly 3 strings), alive (boolean). Output ONLY the JSON, "
            "no prose, no code fences, no explanation."
        ),
        grader="strict_json",
        grader_args={
            "schema": {
                "name": "string",
                "age": "int",
                "skills": "list[string][3]",
                "alive": "bool",
            }
        },
    ),
    # --- Multilingual ---
    Challenge(
        id="multi-fr-en",
        category="multilingual",
        prompt=(
            "Reason in English (mark this section 'REASONING:'), then write your final "
            "answer in French (mark it 'RÉPONSE:'). Question: Why is the sky blue?"
        ),
        grader="bilingual_sections",
        grader_args={"reasoning_lang": "en", "answer_lang": "fr"},
    ),
]


def by_category() -> dict[str, list[Challenge]]:
    out: dict[str, list[Challenge]] = {}
    for c in CHALLENGES:
        out.setdefault(c.category, []).append(c)
    return out


def needle_haystack_prompt(
    needle: str = "The secret password is 'blue elephant'.",
    filler_paragraphs: int = 30,
) -> str:
    filler = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor "
        "incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud "
        "exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute "
        "irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla "
        "pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia."
    )
    paragraphs = [filler] * filler_paragraphs
    paragraphs.insert(filler_paragraphs // 2, needle)
    body = "\n\n".join(paragraphs)
    return (
        "Read the following passage carefully:\n\n"
        f"{body}\n\n"
        "Question: What is the secret password? Answer with just the password, nothing else."
    )
