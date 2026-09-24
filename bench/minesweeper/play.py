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


# For the thinking mode: the rules a player reasons with, stated as a
# procedure, and the answer format pinned to the last line. The bare SYSTEM
# prompt says what the digits mean but not how to USE them, and the untrained
# model's first open-ended think ran past 1,500 tokens without an answer.
SYSTEM_THINK = (
    "You are an expert Minesweeper player. The board is 9x9 with 10 mines. Rows are "
    "letters A-I, columns are numbers 1-9; a cell's neighbours are the up to 8 cells "
    "touching it, diagonals included. '#' is hidden, 'F' is a flag, '.' is a revealed "
    "cell with no neighbouring mines, and a digit N is a revealed cell with exactly N "
    "mines among its neighbours.\n\n"
    "How to find a move you can PROVE:\n"
    "1. Pick a digit that touches hidden cells. List its hidden neighbours and count "
    "its flagged neighbours.\n"
    "2. If N minus the flags equals the number of hidden neighbours, every one of "
    "those hidden neighbours is a mine: flag one.\n"
    "3. If the flags already equal N, every other hidden neighbour is safe: reveal one.\n"
    "4. If one digit's hidden neighbours are a subset of another's, subtract: the "
    "difference in their counts is how many mines are in the extra cells.\n"
    "Only reveal a cell you proved safe, and only flag a cell you proved is a mine. "
    "Never reveal a cell next to a number unless you proved it safe. If nothing can be "
    "proved, reveal a hidden cell far from all numbers.\n\n"
    "Example: '1' at B2 whose only hidden neighbour is C3 -> C3 is a mine -> flag C3. "
    "Then a '1' at B3 touching the flag at C3 and hidden C4 -> C4 is safe -> reveal C4.\n\n"
    "Think briefly, checking one or two digits. Your final answer must be one line: "
    "'reveal <cell>' or 'flag <cell>', naming a cell from the hidden list."
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
    board = labeled(game) if LABELED else game.render()
    extra = f"Numbers touching hidden cells:\n{constraints(game, order)}\n" if CONSTRAINTS else ""
    return (f"{note}Board:\n{board}\n{extra}"
            f"Hidden cells: {' '.join(hidden)}\nYour move?")


# Set by --constraints (option 1). Every revealed number that touches hidden
# cells, with those cells and its flags listed. This is what reading the grid
# produces; what it does NOT do is the arithmetic — which cells are mines or
# safe is still the model's to work out. Lines are shuffled by `order`, like
# the hidden list, so "take the first line" is not a strategy.
CONSTRAINTS = False

SYSTEM_CONSTRAINTS = (
    "You are playing Minesweeper on a 9x9 board with 10 mines. Rows are letters A-I, "
    "columns 1-9. You are given each revealed number that touches hidden cells, which "
    "hidden cells touch it, and which flags touch it.\n"
    "For one number: mines left = the number minus its flags.\n"
    "- If mines left equals how many hidden cells it touches, every one of them is a mine: flag one.\n"
    "- If mines left is 0, every hidden cell it touches is safe: reveal one.\n"
    "Only make a move one of the numbers proves. Reply with exactly one move and nothing "
    "else: 'reveal <cell>' or 'flag <cell>', for example: reveal C4"
)


def constraints(game: Game, order: int = 0) -> str:
    import random

    lines = []
    for r in range(game.height):
        for c in range(game.width):
            if (r, c) not in game.shown or not game.count(r, c):
                continue
            hidden = [game.name(*n) for n in game.neighbours(r, c)
                      if n not in game.shown and n not in game.flags]
            if not hidden:
                continue
            flags = [game.name(*n) for n in game.neighbours(r, c) if n in game.flags]
            lines.append(f"{game.name(r, c)} shows {game.count(r, c)}: hidden {' '.join(hidden)}; "
                         f"flags {' '.join(flags) if flags else 'none'}")
    random.Random(order + 7).shuffle(lines)
    return "\n".join(lines)


def system_prompt(think: int = 0) -> str:
    if think:
        return SYSTEM_THINK
    return SYSTEM_CONSTRAINTS if CONSTRAINTS else SYSTEM


# Set by --labeled. Reading its own think, the model spent most of a 900-token
# budget working out which character of "C  . . 2 # 3 1 2 1 1" is C4 — and
# re-reading the grid to check — before it reached any deduction. Naming
# every cell in place removes the column counting and nothing else: it says
# what each cell shows, not what follows from it.
LABELED = False


def labeled(game: Game) -> str:
    rows = []
    for r in range(game.height):
        cells = []
        for c in range(game.width):
            name = game.name(r, c)
            if (r, c) in game.shown:
                n = game.count(r, c)
                cells.append(f"{name}={n}" if n else f"{name}=.")
            elif (r, c) in game.flags:
                cells.append(f"{name}=F")
            else:
                cells.append(f"{name}=#")
        rows.append(" ".join(cells))
    return "\n".join(rows)


def ask(model, tok, messages, sampler, max_tokens=16) -> str:
    from mlx_lm import generate

    try:
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = tok.apply_chat_template(messages, add_generation_prompt=True)
    return generate(model, tok, prompt=prompt, max_tokens=max_tokens, sampler=sampler, verbose=False)


def ask_think(model, tok, messages, sampler, budget: int) -> tuple[str, int, bool]:
    """Think for at most `budget` tokens, then answer. (reply, think tokens, forced).

    Budget forcing: if the think has not closed by the budget, close it
    ourselves and ask for the answer line. Without the cap the model thought
    for 1,500+ tokens on one position and never answered.

    Measured 2026-09-25, untrained 14B, 600-token budget, same 10 test boards:
    plain grid 0/10, sound on hidden 4/18 (22%), proven mine 8/18; labeled
    grid 0/10, sound 4/15 (27%), proven mine 10/15. No thinking was 42%
    sound on hidden. The think was cut off on 47 of 55 moves, and a forced
    answer names whatever cell it was examining — a frontier cell, where the
    mines are. A capped think made it worse, not better.
    """
    from mlx_lm import generate

    text = tok.apply_chat_template(messages, add_generation_prompt=True,
                                   enable_thinking=True, tokenize=False)
    thought = generate(model, tok, prompt=text, max_tokens=budget, sampler=sampler, verbose=False)
    forced = "</think>" not in thought
    if forced:
        thought = thought.rstrip() + "\n\nTime is up; I must answer now.\n</think>\n\n"
        answer = generate(model, tok, prompt=text + thought, max_tokens=16,
                          sampler=sampler, verbose=False)
    else:
        answer = thought.split("</think>", 1)[1]
    used = len(tok.encode(thought.split("</think>")[0]))
    return answer.strip(), used, forced


def play_game(model, tok, seed: int, sampler, log_path: Path | None = None,
              tag: str = "", think: int = 0) -> dict:
    game = Game(seed=seed)
    game.play("reveal", (4, 4))
    last, invalid = "", 0
    grades = Counter()
    while game.state == "playing" and game.moves < MAX_MOVES:
        board = game.render()
        messages = [{"role": "system", "content": system_prompt(think)},
                    {"role": "user", "content": user_turn(game, last, order=seed * 1000 + game.moves)}]
        if think:
            reply, used, forced = ask_think(model, tok, messages, sampler, think)
            grades["think_tokens"] += used
            grades["forced_close"] += forced
        else:
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
    think = int(argv[argv.index("--think") + 1]) if "--think" in argv else 0
    global LABELED, CONSTRAINTS
    LABELED = "--labeled" in argv
    CONSTRAINTS = "--constraints" in argv
    import mlx.core as mx

    mx.random.seed(0)
    model, tok = load(MODEL, adapter_path=adapter)
    # Greedy decoding loops inside a think; Qwen's own advice is 0.6 there.
    sampler = make_sampler(temp=0.6 if think else 0.0)
    results, totals = [], Counter()
    t0 = time.time()
    for s in range(seed0, seed0 + games):
        r = play_game(model, tok, s, sampler, log, tag="eval", think=think)
        results.append(r)
        totals.update(r["grades"])
        print(f"  seed {s}: {r['state']:5} revealed {r['revealed']:6} moves {r['moves']:3} {r['grades']}",
              flush=True)
    wins = sum(r["state"] == "won" for r in results)
    think_tokens = totals.pop("think_tokens", 0)
    forced = totals.pop("forced_close", 0)
    graded = sum(totals.values())
    label = (("adapter " + adapter if adapter else "base") + (f", think {think}" if think else "")
             + (", labeled" if LABELED else "") + (", constraints" if CONSTRAINTS else ""))
    print(f"\n{label}: won {wins}/{games}; "
          f"moves: " + ", ".join(f"{k} {v} ({100 * v // max(1, graded)}%)" for k, v in totals.most_common())
          + (f"; avg think {think_tokens // max(1, graded)} tok, forced close {forced}/{graded}" if think else "")
          + f"; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
