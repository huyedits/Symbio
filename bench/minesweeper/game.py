"""Minesweeper as an environment the headmaster plays in text.

Rows are letters, columns are numbers, so a move reads the way a person says
it: "reveal C4", "flag B7". The first reveal is always safe (the mines are
laid after it, avoiding that cell and its neighbours), as in every real
Minesweeper — a loss on move one teaches nothing.

The game also knows which hidden cells are PROVABLY safe or provably mines
from what is showing (`sound_moves`). That is not a hint to the player; it is
how a move gets graded afterwards: a reveal that survived only because it was
lucky is not the same lesson as one that followed from the numbers.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from itertools import combinations

ROWS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass
class Game:
    height: int = 9
    width: int = 9
    mines: int = 10
    seed: int = 0
    mine_at: set = field(default_factory=set)
    shown: set = field(default_factory=set)
    flags: set = field(default_factory=set)
    state: str = "playing"          # playing | won | lost
    moves: int = 0

    # ---------------------------------------------------------------- setup

    def _lay_mines(self, safe: tuple[int, int]) -> None:
        rng = random.Random(self.seed)
        keep_clear = {safe, *self.neighbours(*safe)}
        cells = [(r, c) for r in range(self.height) for c in range(self.width)
                 if (r, c) not in keep_clear]
        self.mine_at = set(rng.sample(cells, self.mines))

    def neighbours(self, r: int, c: int):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if (dr or dc) and 0 <= r + dr < self.height and 0 <= c + dc < self.width:
                    yield r + dr, c + dc

    def count(self, r: int, c: int) -> int:
        return sum(n in self.mine_at for n in self.neighbours(r, c))

    # ---------------------------------------------------------------- moves

    def name(self, r: int, c: int) -> str:
        return f"{ROWS[r]}{c + 1}"

    def parse(self, text: str):
        """("reveal"|"flag"|"unflag", (r, c)) from a reply, or None."""
        m = re.search(r"\b(reveal|flag|unflag|open|click)\s+(?:on\s+|cell\s+)?([A-Za-z])\s*(\d{1,2})\b",
                      text, re.I)
        if not m:
            return None
        verb = {"open": "reveal", "click": "reveal"}.get(m.group(1).lower(), m.group(1).lower())
        r, c = ROWS.index(m.group(2).upper()), int(m.group(3)) - 1
        if not (0 <= r < self.height and 0 <= c < self.width):
            return None
        return verb, (r, c)

    def play(self, verb: str, cell: tuple[int, int]) -> str:
        """Apply a move; return what happened, in words the player sees."""
        if self.state != "playing":
            return f"The game is over ({self.state})."
        r, c = cell
        self.moves += 1
        if verb == "flag":
            if cell in self.shown:
                return f"{self.name(r, c)} is already revealed; there is nothing to flag."
            self.flags.add(cell)
            return f"Flagged {self.name(r, c)}."
        if verb == "unflag":
            self.flags.discard(cell)
            return f"Removed the flag from {self.name(r, c)}."
        if cell in self.shown:
            return f"{self.name(r, c)} is already revealed. Pick a hidden cell (#)."
        if cell in self.flags:
            return f"{self.name(r, c)} is flagged. Unflag it first if you mean to reveal it."
        if not self.mine_at:
            self._lay_mines(cell)
        if cell in self.mine_at:
            self.state = "lost"
            return f"BOOM. {self.name(r, c)} was a mine. You lose."
        self._flood(cell)
        if len(self.shown) == self.height * self.width - self.mines:
            self.state = "won"
            return "Every safe cell is revealed. You win."
        return f"Revealed {self.name(r, c)}."

    def _flood(self, start):
        stack = [start]
        while stack:
            cell = stack.pop()
            if cell in self.shown:
                continue
            self.shown.add(cell)
            self.flags.discard(cell)
            if self.count(*cell) == 0:
                stack.extend(n for n in self.neighbours(*cell) if n not in self.shown)

    # ---------------------------------------------------------------- view

    def render(self) -> str:
        head = "   " + " ".join(f"{c + 1}" for c in range(self.width))
        lines = [head]
        for r in range(self.height):
            row = []
            for c in range(self.width):
                if (r, c) in self.shown:
                    n = self.count(r, c)
                    row.append(str(n) if n else ".")
                elif (r, c) in self.flags:
                    row.append("F")
                elif self.state == "lost" and (r, c) in self.mine_at:
                    row.append("*")
                else:
                    row.append("#")
            lines.append(f"{ROWS[r]}  " + " ".join(row))
        return "\n".join(lines)

    # ---------------------------------------------------------------- logic

    def sound_moves(self) -> tuple[set, set]:
        """(provably safe hidden cells, provably mined hidden cells) from what is SHOWN.

        Two rules, applied until nothing changes, which is what a careful human
        does: a number whose hidden neighbours all must be mines (or none can
        be), and the subset rule between two numbers' hidden-neighbour sets.
        Cells neither rule decides need a guess; they appear in neither set.
        """
        if not self.shown:
            return set(), set()
        safe, mines = set(), set()
        changed = True
        while changed:
            changed = False
            constraints = []
            for cell in self.shown:
                hidden = {n for n in self.neighbours(*cell)
                          if n not in self.shown and n not in safe}
                need = self.count(*cell) - len(hidden & mines)
                unknown = hidden - mines
                if not unknown:
                    continue
                if need == 0:
                    if not unknown <= safe:
                        safe |= unknown
                        changed = True
                elif need == len(unknown):
                    if not unknown <= mines:
                        mines |= unknown
                        changed = True
                else:
                    constraints.append((frozenset(unknown), need))
            for (a, na), (b, nb) in combinations(set(constraints), 2):
                for small, ns, big, nbig in ((a, na, b, nb), (b, nb, a, na)):
                    if small < big:
                        rest, left = big - small, nbig - ns
                        if left == 0 and not rest <= safe:
                            safe |= rest
                            changed = True
                        elif left == len(rest) and not rest <= mines:
                            mines |= rest
                            changed = True
        return safe, mines

    def grade_move(self, verb: str, cell) -> str:
        """Judge a move against the logic BEFORE it is played.

        sound   the numbers prove it (safe reveal, or flag on a proven mine)
        guess   the numbers do not decide it, and the board offered no sound move
        reckless the numbers do not decide it, but a sound move existed
        wrong   the numbers prove the opposite
        first   the opening reveal, which is always safe
        """
        if not self.shown:
            return "first"
        safe, mines = self.sound_moves()
        if verb == "reveal":
            if cell in safe:
                return "sound"
            if cell in mines:
                return "wrong"
        elif verb == "flag":
            if cell in mines:
                return "sound"
            if cell in safe:
                return "wrong"
        return "guess" if not (safe or mines - self.flags) else "reckless"
