"""Minesweeper through the eyes: the board is a picture, not text.

    venv/bin/python bench/minesweeper/vision_play.py read [boards]
    venv/bin/python bench/minesweeper/vision_play.py play [games]

`read`: can the vision worker (Qwen3-VL-4B) see the board at all? It is
shown a rendered position and asked to transcribe it, and every cell is
checked against the truth. If reading fails, nothing downstream can work,
and a play score would only measure how it fails.

`play`: the vision worker plays from the picture alone, each move graded by
the same rules as the text runs (sound / reckless / wrong / wasted), on the
same unseen boards (seeds 1000+).
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from game import Game  # noqa: E402
from render import render_png  # noqa: E402

VLM = "lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit"
SHOT = Path("/tmp/minesweeper_board.png")

READ_PROMPT = (
    "This is a Minesweeper board, 9 rows (A-I, left margin) by 9 columns (1-9, top margin). "
    "Raised grey tiles are hidden; flat light cells are revealed; a red flag marks a flag. "
    "Transcribe it: 9 lines, one per row A to I, each exactly 9 symbols separated by spaces. "
    "Use # for hidden, F for flagged, . for a revealed empty cell, and the digit for a "
    "revealed number. Output only the 9 lines."
)

PLAY_PROMPT = (
    "This is a Minesweeper board with 10 mines, rows A-I (left margin), columns 1-9 (top "
    "margin). Raised grey tiles are hidden, flat light cells are revealed, and a number "
    "counts the mines among that cell's 8 neighbours. A number whose hidden neighbours "
    "equal it are all mines; a number whose flags already equal it makes its other hidden "
    "neighbours safe. Make a move the numbers prove. Reply with exactly one move and "
    "nothing else: 'reveal <cell>' or 'flag <cell>', for example: reveal C4"
)


def ask(question: str, max_tokens: int) -> str:
    from symbio import vision

    return vision._generate(str(SHOT), question, max_tokens, VLM)


def truth_rows(game: Game) -> list[list[str]]:
    rows = []
    for r in range(game.height):
        row = []
        for c in range(game.width):
            if (r, c) in game.shown:
                n = game.count(r, c)
                row.append(str(n) if n else ".")
            else:
                row.append("F" if (r, c) in game.flags else "#")
        rows.append(row)
    return rows


def parse_rows(text: str) -> list[list[str]]:
    rows = []
    for line in text.strip().splitlines():
        toks = [t for t in line.replace(",", " ").split()
                if len(t) == 1 and t in "#F.12345678"]
        if len(toks) >= 9:
            rows.append(toks[-9:])
    return rows[:9]


def positions(n: int):
    """n mid-game positions: the opening, plus a few sound moves in."""
    for seed in range(1000, 1000 + n):
        g = Game(seed=seed)
        g.play("reveal", (4, 4))
        for _ in range(seed % 3):
            safe, mines = g.sound_moves()
            if mines:
                g.flags.add(sorted(mines)[0])
            if safe:
                g.play("reveal", sorted(safe)[0])
        yield g


def cmd_read(n: int) -> None:
    cells = right = hidden_right = hidden_total = number_right = number_total = 0
    shape_bad = 0
    for g in positions(n):
        render_png(g, SHOT)
        got = parse_rows(ask(READ_PROMPT, 220))
        want = truth_rows(g)
        if len(got) != 9:
            shape_bad += 1
        for r in range(9):
            for c in range(9):
                w = want[r][c]
                seen = got[r][c] if r < len(got) else None
                cells += 1
                right += seen == w
                if w in "#F":
                    hidden_total += 1
                    hidden_right += seen == w
                elif w.isdigit():
                    number_total += 1
                    number_right += seen == w
        print(f"  seed {g.seed}: {sum(got[r][c] == want[r][c] for r in range(min(9, len(got))) for c in range(9))}/81 cells"
              + ("" if len(got) == 9 else f"  (only {len(got)} rows parsed)"), flush=True)
    print(f"\nvision read {n} boards: cells {right}/{cells} ({100 * right // cells}%); "
          f"hidden/flag {hidden_right}/{hidden_total}; numbers {number_right}/{number_total}; "
          f"malformed transcriptions {shape_bad}")


def ane_read(path: Path) -> list[list[str]]:
    """Read the board with Apple's Vision framework (Neural Engine) + pixels, no model.

    Digits come from VNRecognizeTextRequest, placed into the cell their box
    centre falls in. Hidden vs revealed comes from the tile colour at the
    cell's centre; a flag is the red of its pennant. This is what the OS can
    do on its own, with the grid geometry known from the renderer.
    """
    import Vision
    from Foundation import NSURL
    from PIL import Image

    from render import CELL, MARGIN

    im = Image.open(path).convert("RGB")
    W, H = im.size
    rows = [["#"] * 9 for _ in range(9)]
    for r in range(9):
        for c in range(9):
            x0, y0 = MARGIN + c * CELL, MARGIN + r * CELL
            centre = im.getpixel((x0 + CELL // 2, y0 + CELL // 2 + 20))
            pennant = im.getpixel((x0 + CELL // 2 - 5, y0 + CELL // 2 - 9))
            if pennant[0] > 180 and pennant[1] < 60:
                rows[r][c] = "F"
            elif sum(centre) / 3 > 205:
                rows[r][c] = "."
    # Whole-board OCR missed most lone digits (88% of cells, every miss a
    # number read as empty: the recogniser wants text, and an isolated "1" in
    # a grid is not much of one). So each revealed cell that is not plain
    # background is cropped, enlarged and read on its own.
    tmp = Path("/tmp/ane_cell.png")
    for r in range(9):
        for c in range(9):
            if rows[r][c] != ".":
                continue
            x0, y0 = MARGIN + c * CELL, MARGIN + r * CELL
            crop = im.crop((x0 + 4, y0 + 4, x0 + CELL - 4, y0 + CELL - 4))
            if max(abs(p[0] - p[2]) + abs(p[1] - p[2]) for p in crop.getdata()) < 60:
                continue    # no coloured ink: an empty cell
            canvas = Image.new("RGB", (CELL * 3, CELL * 3), (255, 255, 255))
            canvas.paste(crop.resize((CELL * 2, CELL * 2)), (CELL // 2, CELL // 2))
            canvas.save(tmp)
            handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(
                NSURL.fileURLWithPath_(str(tmp)), None)
            req = Vision.VNRecognizeTextRequest.alloc().init()
            req.setRecognitionLevel_(0)
            req.setUsesLanguageCorrection_(False)
            handler.performRequests_error_([req], None)
            text = "".join(str(o.topCandidates_(1)[0].string()) for o in (req.results() or []))
            digit = next((ch for ch in text if ch in "12345678"), None)
            rows[r][c] = digit or "?"
    _ = W, H
    return rows


def cmd_ane(n: int) -> None:
    cells = right = 0
    t = 0.0
    for g in positions(n):
        render_png(g, SHOT)
        t0 = time.time()
        got = ane_read(SHOT)
        t += time.time() - t0
        want = truth_rows(g)
        ok = sum(got[r][c] == want[r][c] for r in range(9) for c in range(9))
        cells += 81
        right += ok
        wrong = [f"{'ABCDEFGHI'[r]}{c + 1}:{got[r][c]}!={want[r][c]}" for r in range(9) for c in range(9)
                 if got[r][c] != want[r][c]]
        print(f"  seed {g.seed}: {ok}/81 cells {wrong[:4]}", flush=True)
    print(f"\nANE read {n} boards: cells {right}/{cells} ({100 * right // cells}%); "
          f"{1000 * t / n:.0f}ms per board")


def cmd_ane_play(games: int) -> None:
    """Eyes and brain split: the Neural Engine reads the picture, the 14B moves.

    The 14B never sees the game's own state — only what the ANE read off the
    rendered image, written back out as a grid and a shuffled hidden list,
    the same prompt that scored best in text (42% sound on hidden cells).
    """
    import random

    from mlx_lm import generate, load
    from mlx_lm.sample_utils import make_sampler

    import play

    model, tok = load(play.MODEL)
    sampler = make_sampler(temp=0.0)
    totals, wins, misread = Counter(), 0, 0
    t0 = time.time()
    for seed in range(1000, 1000 + games):
        g = Game(seed=seed)
        g.play("reveal", (4, 4))
        grades, invalid, last = Counter(), 0, ""
        while g.state == "playing" and g.moves < 60:
            render_png(g, SHOT)
            seen = ane_read(SHOT)
            misread += sum(seen[r][c] != w for r, row in enumerate(truth_rows(g)) for c, w in enumerate(row))
            grid = "   " + " ".join(str(c + 1) for c in range(9)) + "\n" + "\n".join(
                f"{'ABCDEFGHI'[r]}  " + " ".join(seen[r]) for r in range(9))
            hidden = [f"{'ABCDEFGHI'[r]}{c + 1}" for r in range(9) for c in range(9) if seen[r][c] == "#"]
            random.Random(seed * 1000 + g.moves).shuffle(hidden)
            note = f"Last move: {last}\n" if last else ""
            user = f"{note}Board:\n{grid}\nHidden cells: {' '.join(hidden)}\nYour move?"
            msgs = [{"role": "system", "content": play.SYSTEM}, {"role": "user", "content": user}]
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
            reply = generate(model, tok, prompt=prompt, max_tokens=16, sampler=sampler, verbose=False)
            move = g.parse(reply)
            if move is None:
                grades["unparsed"] += 1
                invalid += 1
                if invalid >= 3:
                    break
                continue
            verb, cell = move
            grade = g.grade_move(verb, cell)
            outcome = g.play(verb, cell)
            if "already" in outcome or "flagged" in outcome:
                grade = "wasted"
            grades[grade] += 1
            last = f"{verb} {g.name(*cell)} -> {outcome}"
        wins += g.state == "won"
        totals.update(grades)
        print(f"  seed {seed}: {g.state:7} revealed {len(g.shown)}/71 {dict(grades)}", flush=True)
    graded = sum(totals.values())
    print(f"\nANE eyes + 14B: won {wins}/{games}; moves: "
          + ", ".join(f"{k} {v} ({100 * v // max(1, graded)}%)" for k, v in totals.most_common())
          + f"; misread cells across all looks {misread}; {time.time() - t0:.0f}s")


def cmd_play(games: int) -> None:
    totals, wins = Counter(), 0
    t0 = time.time()
    for seed in range(1000, 1000 + games):
        g = Game(seed=seed)
        g.play("reveal", (4, 4))
        grades, invalid = Counter(), 0
        while g.state == "playing" and g.moves < 60:
            render_png(g, SHOT)
            reply = ask(PLAY_PROMPT, 16)
            move = g.parse(reply)
            if move is None:
                grades["unparsed"] += 1
                invalid += 1
                if invalid >= 3:
                    break
                continue
            verb, cell = move
            grade = g.grade_move(verb, cell)
            outcome = g.play(verb, cell)
            if "already" in outcome or "flagged" in outcome:
                grade = "wasted"
            grades[grade] += 1
            if grade == "wasted" and grades["wasted"] >= 5:
                break
        wins += g.state == "won"
        totals.update(grades)
        print(f"  seed {seed}: {g.state:7} revealed {len(g.shown)}/71 {dict(grades)}", flush=True)
    graded = sum(totals.values())
    print(f"\nvision play: won {wins}/{games}; moves: "
          + ", ".join(f"{k} {v} ({100 * v // max(1, graded)}%)" for k, v in totals.most_common())
          + f"; {time.time() - t0:.0f}s")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "read"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    {"read": cmd_read, "play": cmd_play, "ane": cmd_ane, "ane_play": cmd_ane_play}[mode](n)
