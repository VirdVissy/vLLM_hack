"""
Heuristic consistency tracker for Adventure mode.

Watches DM output for items the player picked up and NPCs that died, then flags
contradictions when later turns claim something inconsistent (e.g. an NPC who
died a turn ago suddenly speaks again). This is best-effort pattern matching,
not a proof — but for a 10-15 turn demo it surfaces the obvious failures that
quantization tends to introduce.
"""

from __future__ import annotations

import re

_END = r"(?=[.,;!?\n]|\s+(?:and|into|from|then|before|after|while)\b)"

PICKUP_PATTERNS = [
    rf"\bpicked? up (?:the |a |an )?([a-z][a-z\s'-]{{1,40}}?){_END}",
    rf"\b(?:now )?(?:carrying|holding) (?:the |a |an )?([a-z][a-z\s'-]{{1,40}}?){_END}",
    rf"\b(?:added to|placed in|slipped into) (?:your )?(?:inventory|satchel|pack)[:\s]+(?:the |a |an )?([a-z][a-z\s'-]{{1,40}}?){_END}",
]

DEATH_PATTERNS = [
    r"\b([A-Z][a-z]+) (?:dies|is dead|is slain|falls dead|breathes (?:his|her|their) last)\b",
]

# Stop-words to filter out spurious "have <X>" matches.
_NOISE = {"to", "a", "an", "the", "any", "some", "no", "your", "my", "you"}


class ConsistencyTracker:
    def __init__(self) -> None:
        self.inventory: set[str] = set()
        self.dead: set[str] = set()
        self.turns: list[dict] = []

    def observe(self, text: str) -> dict:
        added_items: list[str] = []
        dead_now: list[str] = []
        lower = text.lower()

        for pat in PICKUP_PATTERNS:
            for m in re.findall(pat, lower):
                item = m.strip()
                first = item.split()[0] if item else ""
                if not item or first in _NOISE:
                    continue
                if item not in self.inventory:
                    self.inventory.add(item)
                    added_items.append(item)

        for pat in DEATH_PATTERNS:
            for m in re.findall(pat, text):
                if m not in self.dead:
                    self.dead.add(m)
                    dead_now.append(m)

        contradictions = self._check_contradictions(text)
        turn = {
            "added_items": added_items,
            "dead_now": dead_now,
            "contradictions": contradictions,
        }
        self.turns.append(turn)
        return turn

    def _check_contradictions(self, text: str) -> list[str]:
        out: list[str] = []
        for name in self.dead:
            speak = re.search(
                rf"\b{re.escape(name)}\b\s+(?:says|speaks|asks|smiles|nods|laughs|whispers|shouts)",
                text,
            )
            if speak:
                out.append(f"NPC '{name}' was killed earlier but speaks again")
        return out

    def summary(self) -> dict:
        return {
            "inventory": sorted(self.inventory),
            "dead": sorted(self.dead),
            "total_contradictions": sum(len(t["contradictions"]) for t in self.turns),
            "turns": len(self.turns),
        }
