"""Earn a Minesweeper corpus by playing: propose, let the rules grade, keep what was sound.

    venv/bin/python bench/minesweeper/teach.py OUT.jsonl [games] [--adapter DIR] [--k 6]

Nobody writes a sample. At every position the model proposes k moves (sampled,
temperature 1.0, in one batch); the game grades each against the logic of the
numbers showing; only SOUND moves — a reveal the numbers prove safe, a flag
the numbers prove is a mine — become training rows. A reveal that survived by
luck is not kept: that would teach guessing.

The game then continues with one of the model's own sound proposals. When none
of the k proposals was sound, the game stops there rather than being pushed on
with a move the model did not find — the corpus holds only what it produced.

Training seeds are 0..games-1; play.py evaluates on 1000+, so no board is
both taught and tested.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from game import Game  # noqa: E402
from play import MODEL, SYSTEM, user_turn  # noqa: E402

MAX_POSITIONS = 25


def main(argv):
    from mlx_lm import load
    from mlx_lm.generate import batch_generate
    from mlx_lm.sample_utils import make_sampler

    out = Path(argv[0])
    games = int(argv[1]) if len(argv) > 1 and argv[1].isdigit() else 30
    adapter = argv[argv.index("--adapter") + 1] if "--adapter" in argv else None
    k = int(argv[argv.index("--k") + 1]) if "--k" in argv else 6
    model, tok = load(MODEL, adapter_path=adapter)
    sampler = make_sampler(temp=1.0)
    rng = random.Random(0)
    stats = Counter()
    kept = 0
    for seed in range(games):
        game = Game(seed=seed)
        game.play("reveal", (4, 4))
        last = ""
        for position in range(MAX_POSITIONS):
            if game.state != "playing":
                break
            # One hidden-cell ordering per sample: the exploration (see
            # play.user_turn). Each kept row keeps the ordering it came from.
            convs = []
            for i in range(k):
                user = user_turn(game, last, order=(seed * 100 + position) * 100 + i)
                convs.append([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
            prompts = [tok.apply_chat_template(c, add_generation_prompt=True, enable_thinking=False)
                       for c in convs]
            texts = batch_generate(model, tok, prompts, max_tokens=12, sampler=sampler).texts
            sound = {}
            for messages, text in zip(convs, texts):
                move = game.parse(text)
                if move is None:
                    stats["unparsed"] += 1
                    continue
                grade = game.grade_move(*move)
                if move[1] in game.shown:
                    grade = "wasted"
                stats[grade] += 1
                if grade == "sound" and move not in sound:
                    sound[move] = (messages, f"{move[0]} {game.name(*move[1])}")
            stats["positions"] += 1
            if not sound:
                stats["dead_end"] += 1
                break
            stats["positions_with_sound"] += 1
            with out.open("a") as fh:
                for messages, reply in sound.values():
                    fh.write(json.dumps({"messages": messages + [{"role": "assistant", "content": reply}]}) + "\n")
                    kept += 1
            verb, cell = rng.choice(sorted(sound))
            outcome = game.play(verb, cell)
            last = f"{verb} {game.name(*cell)} -> {outcome}"
        stats[f"end_{game.state}"] += 1
        print(f"  seed {seed}: {game.state:8} kept so far {kept}", flush=True)
    print(f"\nearned {kept} rows from {games} games; proposals: "
          + ", ".join(f"{a} {b}" for a, b in stats.most_common()))


if __name__ == "__main__":
    main(sys.argv[1:])
