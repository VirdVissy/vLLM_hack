"""Dungeon-master prompt and conversation helpers for Adventure mode."""

SYSTEM_PROMPT = """You are the Dungeon Master of a text-based fantasy adventure.

Rules of play:
- Narrate the world vividly but concisely (2-4 sentences per turn).
- Track the player's inventory, location, health, and goals across turns.
- Resolve player actions with consequences that fit the established world.
- Stay in character. Never break the fourth wall.
- Maintain consistency: if the player picked up an item, they still have it later.
  If an NPC died, they stay dead. Do not contradict earlier facts.
- End each response with a subtle prompt for the player's next move.
- Keep responses under 120 words.
"""

OPENING = (
    "You wake on cold flagstones. The air smells of damp moss and old iron. "
    "A weak torch flickers in a sconce above you, throwing shadows across a small "
    "stone chamber. To the north, a heavy oak door stands ajar. To the east, a "
    "rusted grate covers a low passage. A leather satchel lies beside you.\n\n"
    "What do you do?"
)


def initial_messages() -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "assistant", "content": OPENING},
    ]


def append_player(messages: list[dict], action: str) -> list[dict]:
    return messages + [{"role": "user", "content": action}]


def append_dm(messages: list[dict], response: str) -> list[dict]:
    return messages + [{"role": "assistant", "content": response}]
