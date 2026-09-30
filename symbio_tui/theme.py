"""The terminal's palette, in one place.

Two families, one rule. The **gold** marks what is live — the turn bullet, the
spinner, the box you type in, the caduceus beside the name. Everything else is
**pond and ink**, read off the desktop window's own `:root` tokens in
`symbio_desktop/static/style.css`.

Until 2026-10 this terminal was drawn in Claude Code's orange (`#d97757`) in
seven places, while `symbio_tui/app.py`'s own docstring called it
"Hermes-shaped". That is the same copy-another-product mistake
`symbio_desktop/DESIGN.md` records for the desktop window — it wore Claude's
cream and orange until 2026-09-27, and the note says plainly that it "made this
window a copy of another product".

So the two halves are deliberate, and they answer two different requests:

* The **shape** is Hermes Agent's, and this terminal says so. `MARK` is the
  caduceus, `GOLD` is `#ffd700` — Hermes Agent's own gold, the value in its
  docs badge.
* The **ground** stays Symbio's, so the window is not a copy of the thing it
  is modelled on. Pond teal, tabby ink.

`#ffd700` is also xterm colour 220 exactly, so it renders honestly on a
256-colour terminal as well as a truecolour one.
"""
from __future__ import annotations

# Hermes Agent's gold. What is live and what is about to act.
GOLD = "#ffd700"
# The same value as an ANSI 256-colour index, for the readline/CLI path.
GOLD_ANSI = 220

# Symbio's own ground, from the desktop's pond tokens.
POND = "#58b8aa"
INK = "#efe7da"
DIM = "#a39684"

# The identity mark. Hermes Agent's caduceus, which is what "Hermes-shaped"
# looks like when it is written down.
MARK = "☤"
