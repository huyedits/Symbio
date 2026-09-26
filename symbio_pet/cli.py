"""Symbio Pet — launch the cat.

    symbio-pet                 watch this install's model and its fine-tunes
    symbio-pet --demo          play a scripted run, no model or training needed
    symbio-pet --snapshot DIR  render the demo's key frames to PNGs and exit

The model is not in here. The pet reads what the daemon, the trainer and the
golden gate leave on disk, so it costs a window and a timer, not a copy of the
weights.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# The demo's key frames: (seconds into the demo, file name).
FRAMES = (
    (3.0, "01-asleep"), (6.6, "02-waking"), (11.0, "03-idle"),
    (17.0, "04-thinking"), (24.0, "05-training-early"), (41.0, "06-training-late"),
    (45.5, "07-judging"), (48.4, "08-kept"), (49.0, "09-kept-hop"),
    (62.0, "10-skill-training"), (74.6, "11-spit"), (75.4, "12-spat"),
    (91.0, "13-failed"),
)


def available() -> tuple[bool, str]:
    """Can the pet be drawn here, and if not, why not."""
    try:
        import AppKit  # noqa: F401
    except ImportError as e:
        return False, (
            f"The pet draws with AppKit through PyObjC, and {e.name} is not "
            "installed here. `pip install pyobjc-framework-Cocoa` (or "
            "`pip install \"symbio-cli[pet]\"`) adds it. macOS only.")
    return True, ""


def snapshot(directory: Path) -> list[Path]:
    """Step the demo at 30 fps and write its key frames as PNGs."""
    from symbio_pet.cat import Cat
    from symbio_pet.demo import DemoFeed
    from symbio_pet.view import Painter, render_png

    directory.mkdir(parents=True, exist_ok=True)
    now = [0.0]
    feed = DemoFeed(clock=lambda: now[0])
    cat = Cat(seed=7)
    painter = Painter()
    frame, next_poll, written = 1.0 / 30.0, 0.0, []
    pending = list(FRAMES)
    while pending:
        now[0] += frame
        snap = None
        if now[0] >= next_poll:
            next_poll += feed.interval
            snap = feed.poll()
        cat.update(frame, snap)
        if now[0] >= pending[0][0]:
            _, name = pending.pop(0)
            path = directory / f"{name}.png"
            render_png(painter, cat, path)
            written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="symbio-pet",
        description="A desktop cat that shows Symbio fine-tuning itself.")
    parser.add_argument("--demo", action="store_true",
                        help="Play a scripted run instead of watching Symbio")
    parser.add_argument("--port", type=int, default=8742,
                        help="Port of the chat window a click opens (default 8742)")
    parser.add_argument("--snapshot", metavar="DIR",
                        help="Render the demo's key frames to PNGs in DIR and exit")
    args = parser.parse_args(argv)

    ok, why = available()
    if not ok:
        print(f"  {why}")
        return 1

    if args.snapshot:
        for path in snapshot(Path(args.snapshot)):
            print(f"  {path}")
        return 0

    from symbio_pet.feed import Feed, load_constants
    from symbio_pet.view import Pet

    constants = load_constants()
    if args.demo:
        from symbio_pet.demo import DemoFeed

        feed = DemoFeed()
        print("\n  Symbio Pet  →  demo (a scripted run, nothing is trained)")
    else:
        feed = Feed(constants)
        print(f"\n  Symbio Pet  →  watching {constants.PROJECT_DIR}")
    print("  Double-click the cat to chat, drag it anywhere, right-click for more.")
    # Flushed: AppKit ends the process with exit(), which skips Python's own
    # buffers, so a piped banner would otherwise never appear.
    print("  Ctrl+C to stop\n", flush=True)
    pet = Pet(feed, position_file=constants.LOG_DIR / "pet_position.json",
              log_dir=constants.LOG_DIR, port=args.port)
    return pet.run()


if __name__ == "__main__":
    sys.exit(main())
