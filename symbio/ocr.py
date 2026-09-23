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
best two runs still tie, it declines rather than guessing. That scored 12/17
found and 0 wrong on the set it was fitted to — and then, held out on live
pages (DOM boxes as truth, 1280x800 at 2x), returned a wrong box for 49 of
167 labelled controls. Two causes: OCR reads a nav bar or a byline as one
line, and a sentence that mentions a word is not the control named by it.

So a line is now cut at separator characters, a segment only counts when
the query accounts for most of its words, and the box returned is the
matched words', not the segment's. Held out, same frozen captures:

                        found  wrong  declined  absent-says-absent
    8 sites (tuned on)    68      1      44        63/64
    6 unseen sites        26      4      38        47/48
    earlier matcher       99     56      26       107/112   (all 14 sites)

"found" includes picking one of two controls with the same label. Of the
four unseen wrong answers, two are a query naming more than the matched text
("Search GOV.UK" -> "Search") and two are the label's word drawn somewhere
that is not the control (a "CLOUDFLARE" wordmark). The matched text is in
the reply, so the model can see the mismatch. The truth set above is now
11/17 found, 0 wrong.

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


# Characters a UI draws BETWEEN controls. OCR reads a nav bar as one line —
# "new | past | comments | ask | show | jobs" — and the line's centre is not
# where any of those links is.
_SEPARATORS = frozenset("|•·—–/")


def read_text(image_path: str | Path) -> list[dict[str, Any]]:
    """Every text segment on the image as {"text", "box"} in top-left pixels.

    A segment is not an OCR line. Vision groups a whole toolbar into one
    observation, so each line is split at separator characters, and every
    piece gets its own box. On HN's header the line's centre is 500px from
    "jobs". Not at wide gaps: the per-word boxes Vision reports are spaced
    evenly by character (every gap measured 0.12-0.13 of the line height,
    toolbar or prose), so a gap in them carries no information.

    Each segment keeps its words with their boxes, so a match can point at
    the words it matched instead of the middle of the segment.

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

    def to_pixels(rect):
        x, y = rect.origin.x, rect.origin.y
        w, h = rect.size.width, rect.size.height
        return (round(x * width), round((1 - y - h) * height),
                round((x + w) * width), round((1 - y) * height))

    segments: list[dict[str, Any]] = []
    for obs in (request.results() or []):
        best = obs.topCandidates_(1)
        if not best:
            continue
        candidate = best[0]
        line = str(candidate.string())
        words = []
        for m in re.finditer(r"\S+", line):
            try:
                found = candidate.boundingBoxForRange_error_(
                    (m.start(), m.end() - m.start()), None)
                found = found[0] if isinstance(found, tuple) else found
                words.append((m.group(), to_pixels(found.boundingBox())))
            except Exception:
                words = []
                break
        if not words:
            # No per-word boxes: keep the line whole rather than guess where
            # its words are. match() still declines it if it reads as prose.
            segments.append({"text": line, "box": to_pixels(obs.boundingBox())})
            continue
        segments.extend(_split_line(words))
    return segments


def _union(boxes):
    boxes = list(boxes)
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _split_line(words):
    """Cut one OCR line into segments at separator characters."""
    out, current = [], []

    def flush():
        if current:
            out.append({"text": " ".join(w for w, _ in current),
                        "box": _union(b for _, b in current),
                        "words": list(current)})
            current.clear()

    for word, box in words:
        if word in _SEPARATORS:
            flush()
        else:
            current.append((word, box))
    flush()
    return out


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if w not in _STOP and len(w) > 1}


def match(runs: list[dict[str, Any]], query: str) -> dict[str, Any] | None:
    """The one text segment the query names, or None when none or several do.

    A segment qualifies only when the query accounts for MOST of its words.
    A label is short and the query quotes it; a sentence that happens to
    contain the word is a mention, not the control. Held out on live sites,
    "the Wikipedia link" had landed on "Wikipedia is free to use, but not
    free to provide" and "the Change language button" on "programming
    language." Inline links inside prose are declined too, which is the
    price: a decline goes to the vision model, a wrong box goes to a click.

    Each shared word counts 1/(number of segments containing it), so the
    word that singles out one segment outweighs the word half the screen
    carries. Equal scores go to the segment whose words match the query's
    own casing — "the Terminal menu" is "Terminal", not the panel's
    "TERMINAL" — then to the strictly shorter one. Two that still tie are
    ambiguous, and the answer is None, not the first one found.
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
        if not shared or len(shared) * 2 <= len(have):
            continue
        weight = sum(1.0 / frequency[w] for w in shared)
        same_case = len(as_written & set(re.findall(r"[A-Za-z0-9]+", run["text"])))
        scored.append((round(weight, 6), same_case, -len(run["text"]), run))
    if not scored:
        return None
    scored.sort(key=lambda s: s[:3], reverse=True)
    if len(scored) > 1 and scored[0][:3] == scored[1][:3]:
        return None
    best = scored[0][3]
    # Point at the words that matched. "the anyone can edit link" is three
    # words at the end of "the free encyclopedia that anyone can edit", and
    # the segment's centre is on "encyclopedia".
    hit = [box for word, box in best.get("words", ()) if _words(word) & want]
    if hit:
        best = {**best, "box": _union(hit)}
    return best


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
