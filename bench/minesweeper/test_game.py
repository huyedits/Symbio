"""The Minesweeper environment, checked before any model plays in it.

A benchmark that is never tested measures its own bugs, so: the first click
never loses, flood fill opens zeros, the win condition fires, moves parse the
way a model writes them, and the logic grader agrees with hand-worked boards.
Plus a ceiling: a player that only makes sound moves and guesses otherwise
has to win a sensible share of beginner boards, or the grader is wrong.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from game import Game  # noqa: E402


def test_first_reveal_is_always_safe():
    for seed in range(200):
        g = Game(seed=seed)
        g.play("reveal", (4, 4))
        assert g.state != "lost"


def test_flood_fill_opens_the_zero_region():
    g = Game(seed=1)
    g.play("reveal", (0, 0))
    # The opening clears the cell and its neighbours of mines, so (0,0) is a 0
    # and must have opened more than itself.
    assert len(g.shown) > 1


def test_win_when_every_safe_cell_is_shown():
    g = Game(height=3, width=3, mines=1, seed=0)
    g.mine_at = {(0, 0)}
    for r in range(3):
        for c in range(3):
            if (r, c) != (0, 0):
                g.play("reveal", (r, c))
    assert g.state == "won"


def test_moves_parse_as_models_write_them():
    g = Game()
    assert g.parse("reveal C4") == ("reveal", (2, 3))
    assert g.parse("I'll flag b7 because the 1 at A7 is satisfied") == ("flag", (1, 6))
    assert g.parse("Click on E 5") == ("reveal", (4, 4))
    assert g.parse("reveal Z9") is None, "row off the board"
    assert g.parse("no idea") is None


def test_logic_finds_the_forced_mine_and_the_safe_cell():
    # A 1x3 strip:  [1][#][#]  with the mine at column 2 of 3? Use a 2x2 board.
    #   1 #        mine at (1,1). Shown: (0,0)=1, (0,1)=1, (1,0)=1.
    #   1 #  ->    the only hidden cell must be the mine.
    g = Game(height=2, width=2, mines=1, seed=0)
    g.mine_at = {(1, 1)}
    g.shown = {(0, 0), (0, 1), (1, 0)}
    safe, mines = g.sound_moves()
    assert mines == {(1, 1)} and not safe
    assert g.grade_move("flag", (1, 1)) == "sound"


def test_the_subset_rule():
    # Row:   1 1 .   over hidden   # # #   (3 wide, 2 tall), one mine at (1,0).
    # (0,0)=1 sees {(1,0),(1,1)}; (0,1)=1 sees {(1,0),(1,1),(1,2)}.
    # Subset: (1,2) must be safe.
    g = Game(height=2, width=3, mines=1, seed=0)
    g.mine_at = {(1, 0)}
    g.shown = {(0, 0), (0, 1)}
    safe, _ = g.sound_moves()
    assert (1, 2) in safe
    assert g.grade_move("reveal", (1, 2)) == "sound"
    assert g.grade_move("reveal", (1, 0)) == "reckless", "a sound move existed"


def logic_player(seed: int) -> str:
    g = Game(seed=seed)
    g.play("reveal", (4, 4))
    rng = random.Random(seed)
    while g.state == "playing":
        safe, mines = g.sound_moves()
        for m in mines - g.flags:
            g.flags.add(m)
        if safe:
            g.play("reveal", sorted(safe)[0])
            continue
        hidden = [(r, c) for r in range(g.height) for c in range(g.width)
                  if (r, c) not in g.shown and (r, c) not in g.flags]
        g.play("reveal", rng.choice(hidden))
    return g.state


def test_the_logic_player_wins_most_beginner_boards():
    """Beginner Minesweeper (9x9, 10 mines) is won by pure logic plus forced
    guesses most of the time; a grader that makes a logic player lose most
    games is not computing the logic."""
    wins = sum(logic_player(s) == "won" for s in range(200))
    assert wins >= 160, wins    # measured 451/500 = 90% on seeds 0-499
