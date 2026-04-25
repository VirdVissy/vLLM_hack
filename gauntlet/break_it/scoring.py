"""
Auto-graders for Break It mode.

Each grader takes the model's response text plus a few keyword args and returns
a dict with at least {"correct": bool}. `judge_vote` returns correct=None to
signal that a manual UI vote is required.
"""

from __future__ import annotations

import json
import re
from typing import Any


def contains_number(text: str, answer: float, tolerance: float = 1e-6) -> dict:
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    found = [float(n) for n in nums if n not in {"-", ".", "-."}]
    correct = any(abs(n - answer) < tolerance for n in found)
    return {"correct": correct, "found_numbers": found, "expected": answer}


def contains_any(text: str, answers: list[str]) -> dict:
    lower = text.lower()
    matches = [a for a in answers if a.lower() in lower]
    return {"correct": bool(matches), "matched": matches}


def contains_substring(text: str, answer: str) -> dict:
    return {"correct": answer.lower() in text.lower(), "expected": answer}


def no_letter(text: str, letter: str) -> dict:
    count = text.lower().count(letter.lower())
    return {"correct": count == 0, "letter": letter, "count": count}


def strict_json(text: str, schema: dict) -> dict:
    stripped = text.strip()
    if stripped.startswith("```"):
        return {"correct": False, "error": "wrapped in code fences"}
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError as e:
        return {"correct": False, "error": f"invalid JSON: {e}"}
    if not isinstance(obj, dict):
        return {"correct": False, "error": "expected an object"}

    extra = set(obj.keys()) - set(schema.keys())
    missing = set(schema.keys()) - set(obj.keys())
    if extra or missing:
        return {"correct": False, "error": f"key mismatch (extra={sorted(extra)}, missing={sorted(missing)})"}

    type_errors: list[str] = []
    for key, expected in schema.items():
        if not _check_type(obj[key], expected):
            type_errors.append(f"{key}: expected {expected}, got {type(obj[key]).__name__}={obj[key]!r}")
    if type_errors:
        return {"correct": False, "error": "; ".join(type_errors)}

    return {"correct": True, "parsed": obj}


def _check_type(val: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(val, str)
    if expected == "int":
        return isinstance(val, int) and not isinstance(val, bool)
    if expected == "bool":
        return isinstance(val, bool)
    if expected.startswith("list[string]"):
        if not isinstance(val, list) or not all(isinstance(x, str) for x in val):
            return False
        m = re.match(r"list\[string\]\[(\d+)\]", expected)
        if m and len(val) != int(m.group(1)):
            return False
        return True
    return True


_FR_MARKERS = {
    "le ", "la ", "les ", "des ", "du ", "ciel", "bleu", "lumière", "parce",
    " est ", " sont ", " qui ", "lorsque", "soleil",
}


def bilingual_sections(text: str, reasoning_lang: str = "en", answer_lang: str = "fr") -> dict:
    has_reasoning = bool(re.search(r"REASONING\s*:", text, re.IGNORECASE))
    has_response = bool(re.search(r"R[ÉE]PONSE\s*:", text, re.IGNORECASE))
    if not (has_reasoning and has_response):
        return {"correct": False, "error": "missing required section markers"}
    parts = re.split(r"R[ÉE]PONSE\s*:", text, maxsplit=1, flags=re.IGNORECASE)
    response_section = (parts[1] if len(parts) > 1 else "").lower()
    fr_markers = sum(1 for w in _FR_MARKERS if w in response_section)
    return {"correct": fr_markers >= 3, "fr_markers": fr_markers}


def judge_vote(text: str, vote: bool | None = None) -> dict:
    return {"correct": vote, "manual": True}


GRADERS = {
    "contains_number": contains_number,
    "contains_any": contains_any,
    "contains_substring": contains_substring,
    "no_letter": no_letter,
    "strict_json": strict_json,
    "bilingual_sections": bilingual_sections,
    "judge_vote": judge_vote,
}


def grade(text: str, grader_name: str, **grader_args) -> dict:
    fn = GRADERS[grader_name]
    return fn(text, **grader_args)
