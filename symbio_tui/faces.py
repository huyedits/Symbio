"""A face, drawn from the mood the session actually emits.

Not decoration. `chat_turn` surfaces one `[Mood: tag]` line per turn, drawn
from the eleven tags in chat_text._VALID_MOODS, and this is that tag rendered
as a face instead of as the word. A face that were merely chosen at random
would be a lie told in a friendly way; this one changes because the session
changed.

Anything unrecognised falls back to the neutral face rather than being guessed
at — a new mood tag added in chat_text should show up as "no strong feeling",
not as whichever emotion happens to sort first.
"""
from __future__ import annotations

# The eleven tags chat_text._VALID_MOODS can produce, plus the two states the
# UI knows about that the model does not emit: working, and not connected.
FACES: dict[str, str] = {
    "happy":      ":)",
    "excited":    "\\(^o^)/",
    "grateful":   "(^_^)",
    "curious":    "(o_O)?",
    "neutral":    ":|",
    "confused":   "(@_@)",
    "anxious":    ">_<",
    "sad":        ":(",
    "frustrated": "-_-;",
    "impatient":  "(._.)",
    "angry":      ">:(",
    # shy is not in the model's vocabulary; it is here because the store can
    # carry it and because you asked for it.
    "shy":        ">////<",
    # UI states, kept out of the model's namespace on purpose.
    "working":    "(~_~)",
    "offline":    "(x_x)",
}

DEFAULT = FACES["neutral"]


def face(mood: str | None) -> str:
    """The face for a mood tag, or the neutral one for anything unknown."""
    if not mood:
        return DEFAULT
    return FACES.get(str(mood).strip().lower(), DEFAULT)


def banner(mood: str | None, assistant: str = "Symbio", width: int = 80) -> str:
    """The face and its label, or just the face when there is no room."""
    drawn = face(mood)
    label = (mood or "neutral").strip().lower()
    if width < 28:
        return drawn
    return f"{drawn}   {assistant} — {label}"
