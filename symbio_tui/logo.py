"""A stick figure, sitting, and what to show when there is no room for it.

Three sizes rather than one, because a banner that does not fit is a banner
that wraps into noise. The widest form is drawn only when the terminal can
hold it; below that it degrades to a single line and then to nothing, and the
session is unaffected either way.
"""
from __future__ import annotations

SITTING = r"""
        o
       /|\
      / | \___
        |     \
       / \_____\
      /__/
"""

SMALL = r"""
   o
  /|\_
 /_| \_
"""


def logo(width: int, height: int = 24) -> str:
    """The largest figure this terminal can hold, or "" when none fits."""
    if width < 24 or height < 12:
        return ""
    if width < 44 or height < 18:
        return SMALL.strip("\n")
    return SITTING.strip("\n")


def tagline(assistant: str = "Symbio") -> str:
    return f"{assistant} — sitting with the machine, learning what it does"
