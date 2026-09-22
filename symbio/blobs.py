"""Where the *things* are on a screen, before anything is asked to name them.

Why this exists
---------------
`vision.locate` asks the VLM to ground boxes over a whole screenshot, and on a
dense real desktop the generic "find every interactive element" sweep comes
back with nothing at all (measured 2026-09-07, 1920x1080, VS Code plus a
terminal plus a full dock). The model is not blind; the ask is wrong. A
screenshot handed over whole is a wall of wallpaper with a few small controls
in it, and a single pass over uniform image patches spends its attention
evenly across a picture that is mostly nothing.

So look at the pixels first and find the blobs — the regions that actually
carry structure — then ask about those. A button has edges: a border, a label,
a fill that differs from what is behind it. Wallpaper, an empty panel and a
blurred photo do not, at any size. That difference is measurable with
arithmetic, needs no model, and costs milliseconds.

This module is deliberately model-free and dependency-free beyond Pillow,
which is already required. It returns candidate regions in ORIGINAL-image
pixels; `vision.read_blobs` is what turns them into labelled elements.

What a blob is
--------------
A connected clump of edge energy, dilated so the strokes of a word join into
one region rather than eleven. Scored by how densely that clump fills its own
bounding box and how much local contrast it carries, so a crisp button beats a
smear of JPEG noise of the same size. Regions that cover most of the screen are
dropped: a desktop background is not a control, and neither is a whole window.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# The work resolution. Everything below is computed on a downscaled copy and
# the boxes are scaled back at the end: a 1920x1080 Retina grab is 8M pixels,
# and nothing here needs more than a coarse map of where the structure is.
# 900 keeps a 28px composer (the control that started all of this — see
# symbio/computer.py) about 13px tall at work scale, well above the floor.
_WORK_MAX = 900

# Edge energy below this is texture, not an element boundary. Measured against
# the screenshots in screenshots/: a macOS wallpaper gradient peaks in the low
# teens, real control borders and text sit far above it.
_EDGE_THRESHOLD = 32

# How far apart two pieces of edge can be and still count as one thing. Five
# work-scale pixels joins the letters of a word and the label to the border
# around it, without swallowing the control next to it.
_DILATE = 5

# A blob has to be at least this many work-scale pixels on its longest side.
# Below it there is nothing a VLM could name anyway.
_MIN_SIDE = 6

# And it must not be the whole screen. A region past this fraction of the
# image is a window, a background or a page — context, not a target.
_MAX_AREA_FRACTION = 0.5


def _grayscale(path: str) -> tuple[Any, float]:
    """The image at work scale, in grey, plus the factor back to original."""
    from PIL import Image

    im = Image.open(path)
    im = im.convert("RGB")
    width, height = im.size
    longest = max(width, height)
    scale = 1.0
    if longest > _WORK_MAX:
        scale = longest / _WORK_MAX
        im = im.resize((max(1, round(width / scale)), max(1, round(height / scale))),
                       Image.BILINEAR)
    return im.convert("L"), scale


def _edge_mask(gray: Any) -> tuple[bytes, int, int]:
    """A binary mask of where the structure is, dilated into blobs."""
    from PIL import ImageFilter

    # Smooth first: a one-pixel line of sensor or compression noise produces
    # as strong an edge response as a real border, and the whole point is to
    # ignore the parts of the screen that are only texture.
    edges = gray.filter(ImageFilter.SMOOTH).filter(ImageFilter.FIND_EDGES)
    binary = edges.point(lambda v: 255 if v >= _EDGE_THRESHOLD else 0)
    grown = binary.filter(ImageFilter.MaxFilter(_DILATE))
    # FIND_EDGES has no neighbours to work with at the image border, so it
    # draws a bright frame right around the picture. Left in, that frame is one
    # connected component the size of the whole screen, and it touches
    # everything: every real blob merges into it and the answer is one region
    # covering the desktop. Blank it.
    from PIL import ImageDraw

    ImageDraw.Draw(grown).rectangle(
        [0, 0, grown.width - 1, grown.height - 1], outline=0, width=_DILATE)
    return grown.tobytes(), grown.width, grown.height


class _Union:
    """Union-find over run labels. Small, flat, and the only bookkeeping the
    component pass needs."""

    def __init__(self) -> None:
        self.parent: list[int] = []

    def add(self) -> int:
        self.parent.append(len(self.parent))
        return len(self.parent) - 1

    def find(self, a: int) -> int:
        root = a
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[a] != root:  # path compression, iterative
            self.parent[a], a = root, self.parent[a]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _runs(row: bytes, width: int) -> list[tuple[int, int]]:
    """Foreground spans in one row, found with bytes.find rather than a Python
    loop over pixels — the same scan at C speed, which is the difference
    between milliseconds and most of a second on a full screen."""
    spans: list[tuple[int, int]] = []
    i = 0
    while i < width:
        start = row.find(255, i)
        if start < 0:
            break
        stop = row.find(0, start)
        stop = width if stop < 0 else stop
        spans.append((start, stop))
        i = stop + 1
    return spans


def _components(mask: bytes, width: int, height: int) -> list[dict[str, int]]:
    """Connected components of the mask, as boxes with a filled-pixel count.

    Scanline runs joined row to row: a component is described by at most a few
    hundred runs instead of by every pixel in it, which is what keeps this
    cheap enough to run before every look.
    """
    union = _Union()
    boxes: list[list[int]] = []   # [minx, miny, maxx, maxy, filled]
    previous: list[tuple[int, int, int]] = []  # (start, stop, label)

    for y in range(height):
        row = mask[y * width:(y + 1) * width]
        current: list[tuple[int, int, int]] = []
        for start, stop in _runs(row, width):
            label = -1
            for pstart, pstop, plabel in previous:
                if pstop <= start:
                    continue
                if pstart >= stop:
                    break
                # Overlapping run on the row above: same blob.
                if label < 0:
                    label = union.find(plabel)
                else:
                    union.union(label, plabel)
            if label < 0:
                label = union.add()
                boxes.append([start, y, stop - 1, y, 0])
            box = boxes[union.find(label)]
            box[0] = min(box[0], start)
            box[1] = min(box[1], y)
            box[2] = max(box[2], stop - 1)
            box[3] = max(box[3], y)
            box[4] += stop - start
            current.append((start, stop, label))
        previous = current

    merged: dict[int, list[int]] = {}
    for index, box in enumerate(boxes):
        root = union.find(index)
        held = merged.get(root)
        if held is None:
            merged[root] = list(box)
            continue
        held[0] = min(held[0], box[0])
        held[1] = min(held[1], box[1])
        held[2] = max(held[2], box[2])
        held[3] = max(held[3], box[3])
        held[4] += box[4]
    return [{"x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], "filled": b[4]}
            for b in merged.values()]


def _overlap(a: dict[str, int], b: dict[str, int], gap: int) -> bool:
    return not (a["x2"] + gap < b["x1"] or b["x2"] + gap < a["x1"]
                or a["y2"] + gap < b["y1"] or b["y2"] + gap < a["y1"])


def _merge(regions: list[dict[str, int]], gap: int, max_area: float) -> list[dict[str, int]]:
    """Join regions that touch or sit within `gap` of each other.

    An icon and the label under it arrive as two components; a person pointing
    at the screen would call them one thing, and a crop of one without the
    other is harder to name, not easier.

    A merge that would grow the result past `max_area` is refused. Without
    that, one wide element — a menu bar, a full-width toolbar — reaches both
    sides of the screen, and everything within a few pixels of it chains into
    a single box covering the desktop. One box covering the desktop is the
    answer this module exists to stop giving.
    """
    result: list[dict[str, int]] = []
    for region in sorted(regions, key=lambda r: (r["y1"], r["x1"])):
        for held in result:
            if not _overlap(held, region, gap):
                continue
            x1, y1 = min(held["x1"], region["x1"]), min(held["y1"], region["y1"])
            x2, y2 = max(held["x2"], region["x2"]), max(held["y2"], region["y2"])
            if float((x2 - x1 + 1) * (y2 - y1 + 1)) > max_area:
                continue
            held["x1"], held["y1"], held["x2"], held["y2"] = x1, y1, x2, y2
            held["filled"] += region["filled"]
            break
        else:
            result.append(dict(region))
    return result


def _contrast(gray: Any, box: tuple[int, int, int, int]) -> float:
    """Spread of brightness inside a region. A flat patch of wallpaper scores
    near zero however large it is; a button against a panel does not."""
    from PIL import ImageStat

    crop = gray.crop(box)
    if not crop.width or not crop.height:
        return 0.0
    try:
        return float(ImageStat.Stat(crop).stddev[0])
    except Exception:
        return 0.0


def salient_regions(
    image_path: str | Path,
    limit: int = 12,
    min_side: int = _MIN_SIDE,
    merge_gap: int = 4,
) -> list[dict[str, Any]]:
    """The regions of a screenshot worth looking at, best first.

    Each is {"box": (x1, y1, x2, y2), "center": (x, y), "score": float,
    "density": float, "contrast": float} in ORIGINAL-image pixels.

    `score` is density × contrast × a mild size term: what fraction of its own
    box the blob fills, how much brightness varies inside it, and a preference
    for the bigger of two otherwise equal regions. Nothing here knows what a
    button is — it knows what *structure* is, which is what separates a control
    from a wallpaper.
    """
    gray, scale = _grayscale(str(image_path))
    mask, width, height = _edge_mask(gray)
    area = float(width * height) or 1.0

    # Size filters run BEFORE the merge, not after: a component the size of
    # the screen dropped at the end has already pulled every real blob into
    # itself on the way there.
    ceiling = area * _MAX_AREA_FRACTION
    candidates = [
        r for r in _components(mask, width, height)
        if max(r["x2"] - r["x1"] + 1, r["y2"] - r["y1"] + 1) >= min_side
        and float((r["x2"] - r["x1"] + 1) * (r["y2"] - r["y1"] + 1)) <= ceiling
    ]
    regions = _merge(candidates, merge_gap, ceiling)
    scored: list[dict[str, Any]] = []
    for region in regions:
        w = region["x2"] - region["x1"] + 1
        h = region["y2"] - region["y1"] + 1
        if max(w, h) < min_side:
            continue
        box_area = float(w * h) or 1.0
        density = min(1.0, region["filled"] / box_area)
        contrast = _contrast(gray, (region["x1"], region["y1"],
                                    region["x2"] + 1, region["y2"] + 1))
        if contrast <= 1.0:
            # Uniform inside: a dilation artefact around a soft gradient, not
            # a thing on the screen.
            continue
        # Size counts as the square root of the area fraction, not the fourth
        # root. The weaker term ranked x.com's post composer — a big, sparse,
        # deliberately quiet box — below every small dense nav label on the
        # page, at rank 33 of 69, so a search that checked the top 32 never saw
        # it. What is being looked for on a screen is usually a control, and a
        # control is bigger than a word.
        size_term = (box_area / area) ** 0.5
        scored.append({
            "_box": (region["x1"], region["y1"], region["x2"], region["y2"]),
            "density": round(density, 3),
            "contrast": round(contrast, 2),
            "score": round(density * contrast * size_term, 3),
        })

    scored.sort(key=lambda r: r["score"], reverse=True)
    out: list[dict[str, Any]] = []
    for region in _spread(scored, max(0, limit), width, height):
        x1, y1, x2, y2 = region.pop("_box")
        box = (round(x1 * scale), round(y1 * scale),
               round(x2 * scale), round(y2 * scale))
        region["box"] = box
        region["center"] = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
        out.append(region)
    return out


# How the budget is spread over the screen. Score alone is not enough: the
# highest-scoring regions on a real page are the dense colourful ones — an ad
# card, a photo, a block of headlines — and they cluster in the same corner,
# so a straight top-N is a list of one sidebar. Measured on x.com's home
# timeline, the post composer (a mostly-empty box, which is what a text field
# looks like to an edge detector) was ranked below sixteen of them and never
# reached the model at all.
#
# So cap how many come from any one part of the screen. This is a coarse grid,
# and it is NOT the unit of reading — the blobs are, and they keep their own
# measured boxes. The grid only decides whose turn it is, which is what stops
# one busy corner from spending the whole budget.
_SPREAD_COLUMNS = 4
_SPREAD_ROWS = 3


def _spread(scored: list[dict[str, Any]], limit: int,
            width: int, height: int) -> list[dict[str, Any]]:
    """Best first, but no more than a few from any one part of the screen."""
    if limit <= 0 or not scored:
        return []
    # The cap scales with the budget. A fixed cap of three cost the x.com
    # composer its place in a list of thirty-two: three tab labels above it
    # filled its cell, and a region scoring 4 from an emptier part of the
    # screen was taken while the composer at 9 was held back. Twice the even
    # share per cell leaves room for a busy cell to be busy without letting
    # one of them have the lot.
    per_cell = max(2, 2 * -(-limit // (_SPREAD_COLUMNS * _SPREAD_ROWS)))
    cell_w = max(1, width // _SPREAD_COLUMNS)
    cell_h = max(1, height // _SPREAD_ROWS)
    taken: dict[tuple[int, int], int] = {}
    picked: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    for region in scored:
        x1, y1, x2, y2 = region["_box"]
        cell = (min((x1 + x2) // 2 // cell_w, _SPREAD_COLUMNS - 1),
                min((y1 + y2) // 2 // cell_h, _SPREAD_ROWS - 1))
        if taken.get(cell, 0) >= per_cell:
            deferred.append(region)
            continue
        taken[cell] = taken.get(cell, 0) + 1
        picked.append(region)
        if len(picked) >= limit:
            return picked
    # The budget outlived the spread: fill the rest by score, so asking for
    # twenty on a screen with four busy corners still returns twenty.
    return (picked + deferred)[:limit]


def format_regions(regions: list[dict[str, Any]]) -> str:
    """One line per region — the shape a click needs, before the names."""
    if not regions:
        return "No distinct regions stood out on this screen."
    return "\n".join(
        f"  ({r['center'][0]}, {r['center'][1]})  "
        f"{r['box'][2] - r['box'][0]}x{r['box'][3] - r['box'][1]}  "
        f"score {r['score']}"
        for r in regions
    )
