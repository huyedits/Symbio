"""The headmaster plays Minesweeper; every move is graded by the rules.

    venv/bin/python bench/minesweeper/play.py [games] [--adapter DIR] [--seed0 N]

Beginner boards (9x9, 10 mines), the opening reveal made for it at the centre
so every game starts from a board with numbers on it. Each move is graded
BEFORE it is played (see Game.grade_move) and the outcome after, so a game
reports not only win/loss but how it played: a win built on reckless guesses
and a loss on a forced 50/50 are different results.

Every move is appended to attempts.jsonl with the board it was made on — that
log is what the self-teaching loop mines.
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from game import Game  # noqa: E402

MODEL = "mlx-community/Qwen3-14B-3bit"
MAX_MOVES = 60
MAX_INVALID = 3

SYSTEM = (
    "You are playing Minesweeper on a 9x9 board with 10 mines. Rows are letters "
    "A-I, columns are numbers 1-9. '#' is hidden, 'F' is your flag, '.' is an "
    "empty revealed cell, and a digit is how many of that cell's 8 neighbours "
    "are mines. Reveal every cell that is not a mine to win.\n"
    "Reply with exactly one move and nothing else: 'reveal <cell>' or "
    "'flag <cell>', for example: reveal C4"
)


def user_turn(game: Game, last: str, order: int = 0) -> str:
    """The board, plus the list of cells a move can name.

    The grid alone was unreadable to the 14B: on three boards, 23 of its 24
    proposed moves named a cell already revealed (it reveals E1 on a board
    where row E is all '.'). Listing the hidden cells is the same information
    the grid holds, stated in the form a move is written in; it says nothing
    about which of them are safe.

    The list is SHUFFLED, by `order`. In reading order the model named the
    first cell listed twelve times out of twelve at temperature 1.0 — A6, a
    mine — so it never proposed anything else and nothing could be learned.
    A different order per sample is the exploration; a fixed order per game
    position (the eval) keeps a run reproducible.
    """
    import random

    note = f"Last move: {last}\n" if last else ""
    hidden = [game.name(r, c) for r in range(game.height) for c in range(game.width)
              if (r, c) not in game.shown and (r, c) not in game.flags]
    random.Random(order).shuffle(hidden)
    return (f"{note}Board:\n{game.render()}\n"
            f"Hidden cells: {' '.join(hidden)}\nYour move?")


def ask(model, tok, messages, sampler, max_tokens=16) -> str:
    from mlx_lm import generate

    try:
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True)
    return generate(model, tok, prompt=prompt, max_tokens=max_tokens, sampler=sampler, verbose=False)


def play_game(model, tok, seed: int, sampler, log_path: Path | None = None,
              tag: str = "") -> dict:
    game = Game(seed=seed)
    game.play("reveal", (4, 4))
    last, invalid = "", 0
    grades = Counter()
    while game.state == "playing" and game.moves < MAX_MOVES:
        board = game.render()
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user_turn(game, last, order=seed * 1000 + game.moves)}]
        reply = ask(model, tok, messages, sampler).strip()
        move = game.parse(reply)
        if move is None:
            invalid += 1
            grades["unparsed"] += 1
            last = f"'{reply[:40]}' is not a move. Reply like: reveal C4"
            if invalid >= MAX_INVALID:
                break
            continue
        verb, cell = move
        grade = game.grade_move(verb, cell)
        outcome = game.play(verb, cell)
        if "already" in outcome or "flagged" in outcome:
            grade = "wasted"
        grades[grade] += 1
        last = f"{verb} {game.name(*cell)} -> {outcome}"
        if log_path:
            with log_path.open("a") as fh:
                fh.write(json.dumps({"tag": tag, "seed": seed, "board": board, "last": messages[1]["content"],
                                     "reply": reply, "verb": verb, "cell": game.name(*cell),
                                     "grade": grade, "outcome": outcome}) + "\n")
    total = game.height * game.width - game.mines
    return {"seed": seed, "state": game.state if game.state != "playing" else "stuck",
            "revealed": f"{len(game.shown)}/{total}", "moves": game.moves, "grades": dict(grades)}


def main(argv):
    from mlx_lm import load
    from mlx_lm.sample_utils import make_sampler

    games = int(next((a for a in argv if a.isdigit()), 10))
    adapter = argv[argv.index("--adapter") + 1] if "--adapter" in argv else None
    seed0 = int(argv[argv.index("--seed0") + 1]) if "--seed0" in argv else 1000
    log = Path(argv[argv.index("--log") + 1]) if "--log" in argv else None
    model, tok = load(MODEL, adapter_path=adapter)
    sampler = make_sampler(temp=0.0)
    results, totals = [], Counter()
    t0 = time.time()
    for s in range(seed0, seed0 + games):
        r = play_game(model, tok, s, sampler, log, tag="eval")
        results.append(r)
        totals.update(r["grades"])
        print(f"  seed {s}: {r['state']:5} revealed {r['revealed']:6} moves {r['moves']:3} {r['grades']}",
              flush=True)
    wins = sum(r["state"] == "won" for r in results)
    graded = sum(totals.values())
    print(f"\n{'adapter ' + adapter if adapter else 'base'}: won {wins}/{games}; "
          f"moves: " + ", ".join(f"{k} {v} ({100 * v // max(1, graded)}%)" for k, v in totals.most_common())
          + f"; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
