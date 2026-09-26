"""Draw a Minesweeper board the way the game looks, for the vision runs.

Classic look: raised grey tiles for hidden cells, flat light cells when
revealed, the traditional number colours, a red flag, and row letters and
column numbers in the margins so a cell can be named from the picture alone.

Cells are 56px: the vision worker cannot ground anything under its ~32px
patch (see the project's vision notes), so nothing here is drawn smaller.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from game import ROWS, Game

CELL = 56
MARGIN = 44
NUMBER_COLOURS = {1: (0, 0, 255), 2: (0, 128, 0), 3: (255, 0, 0), 4: (0, 0, 128),
                  5: (128, 0, 0), 6: (0, 128, 128), 7: (0, 0, 0), 8: (128, 128, 128)}


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def render_png(game: Game, path: str | Path) -> Path:
    w = MARGIN + game.width * CELL + 8
    h = MARGIN + game.height * CELL + 8
    im = Image.new("RGB", (w, h), (245, 245, 245))
    d = ImageDraw.Draw(im)
    label, digit = _font(24), _font(34)

    for c in range(game.width):
        d.text((MARGIN + c * CELL + CELL // 2, MARGIN // 2), str(c + 1),
               fill=(40, 40, 40), font=label, anchor="mm")
    for r in range(game.height):
        d.text((MARGIN // 2, MARGIN + r * CELL + CELL // 2), ROWS[r],
               fill=(40, 40, 40), font=label, anchor="mm")

    for r in range(game.height):
        for c in range(game.width):
            x0, y0 = MARGIN + c * CELL, MARGIN + r * CELL
            x1, y1 = x0 + CELL - 1, y0 + CELL - 1
            if (r, c) in game.shown:
                d.rectangle([x0, y0, x1, y1], fill=(222, 222, 222), outline=(160, 160, 160))
                n = game.count(r, c)
                if n:
                    d.text(((x0 + x1) // 2, (y0 + y1) // 2), str(n),
                           fill=NUMBER_COLOURS[n], font=digit, anchor="mm")
            else:
                # Raised tile: light top-left edge, dark bottom-right edge.
                d.rectangle([x0, y0, x1, y1], fill=(189, 189, 189))
                d.line([(x0, y0), (x1, y0)], fill=(255, 255, 255), width=4)
                d.line([(x0, y0), (x0, y1)], fill=(255, 255, 255), width=4)
                d.line([(x0, y1), (x1, y1)], fill=(123, 123, 123), width=4)
                d.line([(x1, y0), (x1, y1)], fill=(123, 123, 123), width=4)
                if (r, c) in game.flags:
                    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
                    d.line([(cx, cy - 16), (cx, cy + 16)], fill=(0, 0, 0), width=3)
                    d.polygon([(cx, cy - 16), (cx - 14, cy - 8), (cx, cy)], fill=(220, 0, 0))
    path = Path(path)
    im.save(path)
    return path
