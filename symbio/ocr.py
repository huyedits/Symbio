"""ocr.py — find a named control by reading the screen's text, with no model.

A look at the screen used to mean a VLM pass: the headmaster (~10 GB) put to
sleep, a 4B vision model loaded, a generation per question, then the reverse.
Most of what gets asked for has a label, though — "the Submit Query button",
"the Terminal menu", "the Software Update banner" — and macOS reads text off a
screenshot itself, through the Vision framework, on the Neural Engine, with an
exact pixel box for every run.

Measured 2026-09-23 on the 4-surface / 29-case truth set (x.com home and
composer, a VS Code desktop, the test page), against the 4B VLM:

    OCR locator    11/17 found   12/12 absent-says-absent   ~850ms, 4 screens
    VLM locate     12/17 found   11/12 absent-says-absent   ~6s per question

That was the first matcher, which also returned a WRONG box for 4 of the 6
it missed: two "Post" runs on x.com and it picked one, and "the PROBLEMS tab
above the terminal panel" landed on the Terminal menu because "terminal" and
"problems" scored the same. A locator whose hits cannot be trusted is not a
first pass, it is a second source of fabricated coordinates. So matching now
weighs each word by how rare it is ON THIS SCREEN — "problems" appears once,
"terminal" three times — breaks a tie on the query's own casing, and when the
best two runs still tie, it declines rather than guessing. Same set: 12/17
found, 0 wrong, 3 declined as ambiguous (x.com has two "Post" buttons), 2
icon-only controls with no text at all. Those rules were fitted on this set,
so 0 wrong is an in-sample number; a held-out surface is the next check.

Declining is the contract. None of this answers "is it absent": an icon-only
control has no text to read, so an empty result means "ask the vision model",
never "not there". The caller falls through to vision.locate on any miss.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Words that describe the KIND of thing or where it sits, not which one. They
# are what every query has in common, so matching on them matches everything.
_STOP = frozenset("""the a an of to for in on at is are this that it and or
with where what which button field box input link icon menu tab item control
element screen page window sidebar top bottom left right corner navigation bar
panel holding being written text words reading containing new one single first
""".split())


def available() -> bool:
    """Vision framework bindings importable (macOS with pyobjc-framework-Vision)."""
    if sys.platform != "darwin":
        return False
    try:
        import Vision  # noqa: F401
        from Foundation import NSURL  # noqa: F401
    except Exception:
        return False
    return True


def read_text(image_path: str | Path) -> list[dict[str, Any]]:
    """Every text run on the image as {"text", "box"} in top-left pixels.

    Vision reports boxes normalised with the origin at the BOTTOM-left; they
    are converted here so they share a space with vision.locate's elements.
    """
    import Vision
    from Foundation import NSURL
    from PIL import Image

    path = str(image_path)
    with Image.open(path) as im:
        width, height = im.size
    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(
        NSURL.fileURLWithPath_(path), None)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(0)          # 0 = accurate
    # Correction "fixes" UI labels into dictionary words (a filename, a
    # hashtag); the label as drawn is what a query quotes.
    request.setUsesLanguageCorrection_(False)
    handler.performRequests_error_([request], None)

    runs = []
    for obs in (request.results() or []):
        best = obs.topCandidates_(1)
        if not best:
            continue
        rect = obs.boundingBox()
        x, y = rect.origin.x, rect.origin.y
        w, h = rect.size.width, rect.size.height
        runs.append({
            "text": str(best[0].string()),
            "box": (round(x * width), round((1 - y - h) * height),
                    round((x + w) * width), round((1 - y) * height)),
        })
    return runs


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if w not in _STOP and len(w) > 1}


def match(runs: list[dict[str, Any]], query: str) -> dict[str, Any] | None:
    """The one text run the query names, or None when none or several do.

    Each shared word counts 1/(number of runs containing it), so the word
    that singles out one run outweighs the word half the screen carries.
    Equal scores go to the run whose words match the query's own casing —
    "the Terminal menu" is "Terminal", not the panel's "TERMINAL" — then to
    the strictly shorter run: "Post" as a button beats "Post" inside a
    sentence. Two runs that still tie are ambiguous, and the answer is None,
    not the first one found.
    """
    want = _words(query)
    if not want:
        return None
    as_written = set(re.findall(r"[A-Za-z0-9]+", query))
    run_words = [_words(r["text"]) for r in runs]
    frequency = {w: sum(w in have for have in run_words) for w in want}

    scored = []
    for run, have in zip(runs, run_words):
        shared = want & have
        if shared:
            weight = sum(1.0 / frequency[w] for w in shared)
            same_case = len(as_written & set(re.findall(r"[A-Za-z0-9]+", run["text"])))
            scored.append((round(weight, 6), same_case, -len(run["text"]), run))
    if not scored:
        return None
    scored.sort(key=lambda s: s[:3], reverse=True)
    if len(scored) > 1 and scored[0][:3] == scored[1][:3]:
        return None
    return scored[0][3]


def find(image_path: str | Path, query: str) -> list[dict[str, Any]]:
    """Locate `query` by its text. Same shape as vision.locate, or [].

    [] covers every way this can fail — no bindings, nothing readable, no
    match, an ambiguous match — because every one of them means the same
    thing to the caller: ask the vision model instead.
    """
    if not query.strip() or not available():
        return []
    try:
        runs = read_text(image_path)
    except Exception as e:
        logger.debug("OCR failed on %s: %s", image_path, e)
        return []
    run = match(runs, query)
    if run is None:
        return []
    left, top, right, bottom = run["box"]
    return [{
        "label": run["text"],
        "box": run["box"],
        "center": ((left + right) // 2, (top + bottom) // 2),
    }]
