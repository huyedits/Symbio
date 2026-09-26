"""Eyes. A vision-language model that looks at a screenshot and says what is
actually there — including where, in pixels you can click.

Why this exists
---------------
Every browser and desktop tool in this project reported what it DID, never
what was on screen, and the headmaster is text-only: it saw page text scraped
from the DOM and a filename for every screenshot. So it worked blind, and the
failure mode was not "it makes a mistake", it was "it cannot tell that it made
one". Live 2026-09-06, asked to post to @grok, it typed into a page with
nothing focused (the keystrokes became X's single-key shortcuts, which opened
the composer and dropped half the message into it), then clicked the only
control it could name — "Post", the submit button — and shipped the fragment.
Nothing in that chain could see a composer, a caret, or a disabled button.
The transcript ends with the model reasoning in circles about what the page
"might" look like until it ran out of tokens with no output at all.

A description alone would not have fixed it. Knowing "there is a text box"
does not tell you where to click, and the whole accident was a click aimed by
guessing at words. So this module's contract is coordinates: `locate()`
returns pixel centres that `click_at` can use directly.

The coordinate space (measured, not assumed)
--------------------------------------------
Qwen3-VL emits grounding boxes normalised to 0-1000 on each axis, relative to
the ORIGINAL image, not to the processor's resized tensor. Verified 2026-09-07
against screenshots/screenshot_20260809_192055.png (1265x1917): the model's
boxes for the text input, the Submit button and the Open Details link rescaled
by w/1000 and h/1000 landed within ~2px of the real elements measured off the
pixels. Nothing here should "helpfully" also try the resized-pixel or token-
grid interpretations — they were both checked and both wrong, and a click
aimed by a wrong scale is worse than no click at all.

Whole screens and blobs
-----------------------
`locate` reads the screen whole: one pass, attention spread evenly over
uniform patches of a picture that is mostly wallpaper. That is why the generic
sweep returns nothing on a dense desktop. `read_blobs` reads it the other way
— `symbio.blobs` measures where the structure actually is (edges, contrast, a
fill inside a border) with no model at all, and each of those regions is then
shown to the VLM on its own to be named. Wallpaper produces no regions, so it
costs nothing to skip; the coordinates come from the pixels rather than from
generated numbers, which is the one part of grounding this model gets wrong.

Memory
------
This is a second model on a 16 GB machine that already hard-freezes when the
14B headmaster and Chrome are both resident. It is therefore a *swapped*
worker, never a resident one: the caller puts the headmaster to sleep, we
load (~2.5 GB, measured 4.14 GB peak while generating), answer, and
`release()` frees the weights BEFORE the headmaster wakes. That ordering is
the whole discipline — freeing after the reload would just move the
double-residency to the other end of the call. Callers own the bracket;
see ToolsMixin._run_vision, which borrows dispatch's deep-sleep hooks.
"""

from __future__ import annotations

import contextlib
import gc
import io
import json
import logging
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Qwen3-VL 4B, 4-bit. Picked over the 8B because 5 GB of VLM plus Chrome sits
# on the wrong side of this machine's freeze line even with the headmaster
# asleep, and over SmolVLM-500M because a resident 500M model cannot ground
# coordinates — and coordinates are the entire point (see the module docstring).
DEFAULT_VISION_MODEL = "lmstudio-community/Qwen3-VL-4B-Instruct-MLX-4bit"

# Qwen-VL grounding is normalised to this range on both axes.
_COORD_SCALE = 1000.0


@contextlib.contextmanager
def _quiet():
    """Keep the libraries' chatter off the user's terminal.

    transformers warns about processor kwargs on every single call, and
    huggingface_hub draws a "Fetching 14 files" progress bar even when every
    file is already cached. Both go to stdout, which in this project is the
    chat transcript — so a look at the screen would print two lines of library
    noise into the middle of a conversation. Captured rather than silenced:
    it goes to the log, where it is still there if a load starts misbehaving.
    """
    buffer = io.StringIO()
    try:
        from huggingface_hub.utils import disable_progress_bars

        disable_progress_bars()
    except Exception:
        pass
    try:
        import transformers.utils.logging as tf_logging

        tf_logging.set_verbosity_error()
    except Exception:
        pass
    try:
        with contextlib.redirect_stdout(buffer):
            yield
    finally:
        noise = buffer.getvalue().strip()
        if noise:
            logger.debug("vision backend output: %s", noise)


_LOCK = threading.Lock()
_MODEL: Any = None
_PROCESSOR: Any = None
_LOADED_NAME: str = ""


class VisionUnavailable(RuntimeError):
    """mlx-vlm is missing, or the model could not be loaded."""


def model_name(config: dict[str, Any] | None = None) -> str:
    cfg = (config or {}).get("vision", {})
    return str(cfg.get("model_name") or DEFAULT_VISION_MODEL)


def is_enabled(config: dict[str, Any] | None = None) -> bool:
    return bool((config or {}).get("vision", {}).get("enabled", True))


def is_loaded() -> bool:
    return _MODEL is not None


def available() -> bool:
    """Is the vision stack importable at all? Cheap; does not load weights."""
    try:
        from symbio.mlx_gate import require as _require
        _require("mlx_vlm")
    except Exception:
        return False
    return True


def load(name: str = "") -> tuple[Any, Any]:
    """Load (or return the already-loaded) VLM. Not thread-safe by accident:
    the module lock is held so two tool calls cannot race two copies of a
    2.5 GB model into memory at the same time."""
    global _MODEL, _PROCESSOR, _LOADED_NAME
    target = name or DEFAULT_VISION_MODEL
    with _LOCK:
        if _MODEL is not None and _LOADED_NAME == target:
            return _MODEL, _PROCESSOR
        # A different model than the one resident: free first, then load.
        # Loading first would hold both, which is the whole thing we are
        # avoiding.
        if _MODEL is not None:
            _release_locked()
        try:
            from symbio.mlx_gate import attr as _vlm
            vlm_load = _vlm("mlx_vlm.load")
        except ModuleNotFoundError as e:
            # The gate already folded a missing engine into the install hint;
            # keep the tailored vision message, which adds what to install.
            raise VisionUnavailable(
                "mlx-vlm is not installed. Install it with "
                "`pip install mlx-vlm` to give the assistant vision."
            ) from e
        except Exception as e:  # pragma: no cover - import-time environment
            raise VisionUnavailable(
                f"mlx-vlm is not installed ({e}). Install it with "
                f"`pip install mlx-vlm` to give the assistant vision."
            ) from e
        shortfall = _memory_shortfall()
        if shortfall:
            raise VisionUnavailable(shortfall)
        try:
            with _quiet():
                _MODEL, _PROCESSOR = vlm_load(target)
        except Exception as e:
            _MODEL = _PROCESSOR = None
            raise VisionUnavailable(f"Could not load vision model '{target}': {e}") from e
        _LOADED_NAME = target
        return _MODEL, _PROCESSOR


# What the VLM peaks at while generating, measured on this machine, plus a
# little room to land in.
_NEEDED_GB = 5.0


def _memory_shortfall() -> str:
    """A refusal message when there is not enough RAM, or "" to go ahead.

    Not every caller can put the headmaster to sleep first — AIAgent holds its
    model for its whole lifetime and has no unload path — so the check belongs
    here, at the load, rather than in the one caller that happens to bracket
    it. A refusal costs a look; getting this wrong costs the machine, which on
    this project has meant a hard freeze with the 14B and Chrome both resident.
    """
    try:
        from symbio.app.training import free_ram_bytes
    except Exception:
        return ""
    try:
        free = free_ram_bytes()
    except Exception:
        return ""
    if free is None:
        return ""
    free_gb = free / 1e9
    if free_gb >= _NEEDED_GB:
        return ""
    return (
        f"Not enough free memory to look: the vision model needs about "
        f"{_NEEDED_GB:.0f} GB while generating and only {free_gb:.1f} GB is "
        f"free. Something else is holding the RAM — close the browser, or "
        f"unload the main model first. Refusing rather than loading a second "
        f"model on top of the first."
    )


def _release_locked():
    """Free the weights. Caller holds _LOCK."""
    global _MODEL, _PROCESSOR, _LOADED_NAME
    _MODEL = None
    _PROCESSOR = None
    _LOADED_NAME = ""
    gc.collect()
    try:
        from symbio.mlx_gate import attr as _mlx
        mx = _mlx("mlx.core")

        mx.clear_cache()
    except Exception:
        pass


def release() -> bool:
    """Unload the VLM and hand the memory back. Returns whether anything was
    resident. Dropping the reference is not enough on MLX — the allocator
    holds the buffers until clear_cache(), and "still resident when the
    headmaster reloads" is exactly the double-residency that OOMs this box."""
    with _LOCK:
        if _MODEL is None:
            return False
        _release_locked()
        return True


# ---------------------------------------------------------------- generation


def _generate(image_path: str, question: str, max_tokens: int, name: str) -> str:
    model, processor = load(name)
    from symbio.mlx_gate import attr as _vlm
    vlm_generate = _vlm("mlx_vlm.generate")
    apply_chat_template = _vlm("mlx_vlm.prompt_utils.apply_chat_template")

    with _quiet():
        prompt = apply_chat_template(processor, model.config, question, num_images=1)
        out = vlm_generate(
            model, processor, prompt, [str(image_path)],
            max_tokens=max_tokens, verbose=False,
        )
    # mlx-vlm returns a result object on current versions and a bare string on
    # older ones. Both are in the wild; neither is worth pinning a version for.
    return (out.text if hasattr(out, "text") else str(out)).strip()


def describe(
    image_path: str | Path,
    question: str = "",
    config: dict[str, Any] | None = None,
    max_tokens: int = 400,
) -> str:
    """Say what is on screen, in prose."""
    q = question.strip() or (
        "Describe this screen for someone who cannot see it and has to operate "
        "it. Say what application or page it is, what state it is in, what text "
        "is currently entered in any field, and which buttons are enabled or "
        "disabled. Be concrete and brief. Do not guess at anything you cannot "
        "actually see."
    )
    # Describe positions in words; coordinates come from locate(), never from
    # here. Asked to describe a screen, the model volunteers pixel figures of
    # its own — and they are guesses, prefixed "approximately" and derived
    # from nothing. Live 2026-09-07 it offered a composer at (200, 110)-(625,
    # 250) in prose while the grounding pass put the same box at (262, 90)-
    # (806, 204); the headmaster then read out the wrong pair first, because
    # they came first. Two sets of coordinates in one observation is worse
    # than one, and the accurate set is not the one being invented mid-
    # sentence.
    q += (" Describe WHERE things are in words (left, below, top-right of the "
          "sidebar). Do not state any pixel coordinates or numeric positions.")
    return _generate(str(image_path), q, max_tokens, model_name(config))


# Phrased to accept either a noun phrase ("every interactive element") or the
# user's own question ("where is the composer and the Post button?"), because
# what you ask for decides whether grounding works at all — see locate().
_LOCATE_PROMPT = (
    "Find the following in this screenshot and output their coordinates as "
    'JSON: {what}\n'
    'Each entry must have "bbox_2d": [x1, y1, x2, y2] and a "label" naming the '
    "element and its visible text. Report only elements you can actually see."
)

_GENERIC_TARGETS = (
    "every interactive element (text input, button, link, checkbox, menu item)"
)


def locate(
    image_path: str | Path,
    what: str = "",
    config: dict[str, Any] | None = None,
    max_tokens: int = 600,
    verify: bool = True,
) -> list[dict[str, Any]]:
    """Find things on screen and return them with clickable pixel centres.

    Each element is {"label", "box": (x1,y1,x2,y2), "center": (x,y)} in
    ORIGINAL-image pixels, already rescaled out of the model's 0-1000 space.

    ASK FOR WHAT YOU WANT. Measured 2026-09-07 on a real 1920x1080 macOS
    desktop (VS Code, a terminal, a notification, a full dock):

      - the generic "every interactive element" sweep returned NOTHING at all;
      - naming the target grounded it within a few pixels — the Software
        Update banner at (1734, 83) against (1730, 82) measured off the image,
        the editor tab at (424, 75), the Apple/Code/File/Edit menus each
        within ~10px.

    The generic sweep still works on a simple page (it is how the browser
    composer and its submit button were found, pixel-accurate), so it stays the
    default when the caller has nothing specific to ask, and the targeted query
    is tried first whenever there is one.

    ASK FOR ONE THING. Generation is greedy, so this is reproducible — three
    identical calls returned identical coordinates. But accuracy is not
    uniform, and the failure is silent:

      - "the config.json editor tab"            -> (406, 71), true within 5px
      - the same tab asked alongside a second
        target in one call                      -> (1131, 299), ~700px wrong

    Rows of near-identical repeated controls are the other weak spot: asked for
    the dock icons it returned evenly-spaced positions at the correct y with
    fabricated x, drifting by more than a full icon across the row. It does not
    say it is unsure — it returns plausible numbers — so a caller that needs a
    dependable coordinate should name a single distinctive target.
    """
    path = str(image_path)
    width, height = _image_size(path)
    targeted = bool(what.strip())
    targets = what.strip() or _GENERIC_TARGETS
    if targeted and verify and not _on_screen(path, targets, config):
        # Not there. Say so and stop: no grounding pass, no blob sweep, no
        # fallback. A model asked "where is X" answers even when X is absent,
        # and every fallback from here produces another plausible coordinate
        # for something else. An empty list is the honest answer and it is
        # what the caller has to be able to act on.
        return []
    raw = _generate(path, _LOCATE_PROMPT.format(what=targets),
                    max_tokens, model_name(config))
    found = _parse_elements(raw, width, height)
    if found:
        return found
    if not targeted:
        # A generic sweep that came back empty. Do NOT run it again — the
        # fallback IS the generic sweep, and a second 15-second generation
        # gives the same answer. Read the screen as blobs instead: a different
        # instrument, and the one that works on the dense desktop where this
        # sweep is known to return nothing.
        return read_blobs(path, config=config)
    # The targeted phrasing produced no boxes at all — which can just mean it
    # was worded awkwardly. Sweep generically so the caller still learns the
    # layout.
    raw = _generate(path, _LOCATE_PROMPT.format(what=_GENERIC_TARGETS),
                    max_tokens, model_name(config))
    swept = _parse_elements(raw, width, height)
    if swept:
        return swept
    # Still nothing. A whole-screen ask is the wrong instrument here — see
    # read_blobs — so find the regions that carry structure and name those.
    return read_blobs(path, config=config)


# ------------------------------------------------------------------- blobs

# How much of its surroundings a blob crop carries. A control cropped exactly
# to its own edges loses the thing that says what it is: "Post" in a box is a
# button, the same word bare is a word.
_BLOB_PAD = 12

# What a crop comes back as when there is nothing nameable in it. Kept as a
# prefix check rather than an equality one because the model answers "nothing
# — this is part of the wallpaper" as readily as "nothing".
_BLOB_NOTHING = ("nothing", "background", "wallpaper", "empty", "none",
                 "blank", "no ", "n/a")

_BLOB_PROMPT = (
    "This is a small crop of a computer screen. Name the single interface "
    "element at the centre of it in at most eight words, including its "
    "visible text — for example 'Post button', 'search field reading london', "
    "'Wi-Fi menu bar icon'. If the crop is only background, wallpaper or "
    "decoration with no control or text in it, answer exactly: nothing."
)


def read_blobs(
    image_path: str | Path,
    config: dict[str, Any] | None = None,
    limit: int = 8,
    max_tokens: int = 40,
) -> list[dict[str, Any]]:
    """Read a screen as blobs: find what stands out, then name each one.

    `locate` hands the model a whole screenshot and asks it to ground every
    control in one pass. On a dense desktop that returns nothing at all
    (measured 2026-09-07), and the reason is the ask: a screen is mostly
    wallpaper, and one pass spread evenly over uniform patches of a picture
    that is 95% nothing has no reason to land on the 28px composer in it.

    So split the work. `symbio.blobs` finds the regions that carry structure —
    edges, contrast, a border around a fill — with arithmetic, no model, in
    about 30ms; a flat wallpaper produces none of them. Each of those regions
    is then cropped and shown to the VLM on its own, which is the ask it is
    good at: one small picture, one thing in the middle, name it.

    The coordinates come from the pixels, not from the model. That is the
    other half of the point: `locate`'s known failure is fabricating plausible
    positions for rows of near-identical controls, and a box measured off the
    edge map cannot drift across a dock the way a generated number can. The
    model only ever supplies the label.

    Returns the same shape as `locate` — {"label", "box", "center"} in
    original-image pixels — plus "score", so a caller can tell how much the
    region stood out. Empty when nothing on screen has any structure in it,
    which for a blank or solid-colour screen is the true answer.
    """
    from symbio import blobs as _blobs

    if not (config or {}).get("vision", {}).get("blobs", True):
        return []
    path = str(image_path)
    try:
        regions = _blobs.salient_regions(path, limit=max(0, limit))
    except Exception as e:  # pragma: no cover - Pillow failure on a bad file
        logger.debug("blob segmentation failed on %s: %s", path, e)
        return []
    if not regions:
        return []

    name = model_name(config)
    found: list[dict[str, Any]] = []
    for region in regions:
        label = _name_crop(path, region["box"], name, max_tokens, config)
        if not label:
            continue
        found.append({
            "label": label,
            "box": region["box"],
            "center": region["center"],
            "score": region.get("score", 0.0),
        })
    return found


def _name_crop(path: str, box: tuple[int, int, int, int], name: str,
               max_tokens: int, config: dict[str, Any] | None = None) -> str:
    """Ask what one region is, or "" for background and for any failure.

    A region with words in it is named by its words, read on the Neural
    Engine (~0.1 s) — the VLM generation it replaces is several seconds on the
    GPU the headmaster needs back. Only a region with no text (an icon, an
    image) is shown to the VLM.
    """
    try:
        from symbio.app import ane

        if ane.enabled(config) and (config or {}).get("vision", {}).get("ocr_labels", True):
            text = ane.read_crop(path, box)
            if len(text) >= 2:
                return f'"{text[:80]}" (text)'
    except Exception as e:  # pragma: no cover - the helper failing is a fallback, not an error
        logger.debug("ANE crop read failed on %s: %s", box, e)
    answer = _ask_crop(path, box, _BLOB_PROMPT, name, max_tokens)
    lowered = answer.lower().lstrip("\"'*- ")
    if not answer or lowered.startswith(_BLOB_NOTHING):
        return ""
    # One line, whatever it wrapped the name in.
    return answer.splitlines()[0].strip().strip('"').strip()


def _ask_crop(path: str, box: tuple[int, int, int, int], question: str,
              name: str, max_tokens: int) -> str:
    """Put one region in front of the model on its own. "" if anything fails.

    The padding matters: a control cropped exactly to its own edges loses what
    says what it is. "Post" inside a rounded blue box is a button; the same
    word bare is a word.
    """
    from PIL import Image

    temp = ""
    try:
        with Image.open(path) as im:
            left, top, right, bottom = box
            crop = im.crop((max(0, left - _BLOB_PAD), max(0, top - _BLOB_PAD),
                            min(im.width, right + _BLOB_PAD),
                            min(im.height, bottom + _BLOB_PAD)))
            if not crop.width or not crop.height:
                return ""
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
                temp = fh.name
            crop.save(temp)
        return _generate(temp, question, max_tokens, name).strip()
    except Exception as e:
        logger.debug("could not read blob %s: %s", box, e)
        return ""
    finally:
        if temp:
            with contextlib.suppress(OSError):
                os.unlink(temp)


def find_in_blobs(
    image_path: str | Path,
    what: str,
    config: dict[str, Any] | None = None,
    limit: int = 32,
    max_tokens: int = 8,
    stop_after: int = 3,
) -> list[dict[str, Any]]:
    """Find one named thing by checking the regions the pixels found, one by one.

    This is `locate` turned inside out. `locate` asks the model where something
    is and takes the numbers it writes down — and the recorded failure of that
    is a confident coordinate for something else entirely (a dock icon offered
    as a dismissed notification, 2026-09-07), because a model asked "where is
    X" will answer even when X is not there. Here the model is never asked for
    a position at all. The candidate positions come from the edge map, and the
    model is shown each candidate and asked one closed question about it: is
    this the thing? A wrong answer costs one wrong crop, not a click on the
    other side of the screen.

    It is also what reaches the controls `locate` cannot ground: x.com's
    composer at 28px is below the VLM's patch floor as part of a full screen,
    but as a padded crop of its own it fills the picture.

    Best-scoring region first, so a caller taking [0] gets the most prominent
    match rather than an arbitrary one. Every hit is in ORIGINAL-image pixels.

    `limit` is deliberately generous. What is being looked for is usually NOT
    the most eye-catching thing on the screen — measured on x.com's home
    timeline, sixteen ad cards, headlines and photos outscored the post
    composer, because an empty text box is exactly what an edge detector finds
    least interesting. `stop_after` is what keeps that affordable: checking
    stops at the first few matches rather than at the end of the list.
    """
    from symbio import blobs as _blobs

    if not (config or {}).get("vision", {}).get("blobs", True):
        return []
    path = str(image_path)
    target = what.strip()
    if not target:
        return []
    try:
        regions = _blobs.salient_regions(path, limit=max(0, limit))
    except Exception as e:  # pragma: no cover - Pillow failure on a bad file
        logger.debug("blob segmentation failed on %s: %s", path, e)
        return []

    name = model_name(config)
    question = (f"Does this crop of a screen show {target}? "
                f"Answer only yes or no.")
    hits: list[dict[str, Any]] = []
    for region in regions:
        answer = _ask_crop(path, region["box"], question, name, max_tokens)
        if not answer.strip().lower().lstrip("\"'*- ").startswith("yes"):
            continue
        hits.append({
            "label": target,
            "box": region["box"],
            "center": region["center"],
            "score": region.get("score", 0.0),
        })
        if stop_after and len(hits) >= stop_after:
            # Enough to act on, and a hit costs a generation. The list is in
            # score order, so the first few are the prominent ones; going on
            # to check thirty regions for a fourth match is twenty seconds
            # spent on candidates nobody will click.
            break
    return hits


# Asked whether a CROP shows the target, this model is close to useless: over
# four surfaces (x.com signed in, x.com mid-compose, a macOS desktop with VS
# Code and a full dock, a plain HTML page) the crop check rejected 3 of 13
# correctly grounded elements and caught only 3 of 12 targets that were not on
# screen at all. Asked the same question about the WHOLE SCREEN it answers
# well: 13/17 yes on things that are there, 1/12 yes on things that are not.
#
# The difference is not the model's eyesight, it is the question. Absence is a
# property of the screen, and a crop of a control cannot show that the control
# is missing — whatever pixels the crop contains, something is in it. So ask
# before grounding, not after: one 6-token generation for the whole look,
# replacing one generation per located element, and the answer arrives in time
# to skip the grounding pass entirely when the thing is not there.
#
# Measured 2026-09-23 over 29 present/absent cases:
#   crop check (as shipped)      13 correct
#   no check at all              13 correct
#   crop check, repaired prompt  14 correct
#   this, asked of the screen    23 correct
_PRESENCE_PROMPT = "Is {target} visible anywhere on this screen? Answer only yes or no."


def on_screen(image_path: str | Path, target: str,
              config: dict[str, Any] | None = None) -> bool:
    """Public form of the screen-level presence question.

    Callers that want the answer WITHOUT a grounding pass use this: a look that
    reports "not present" and a look that failed to ground are different facts
    and have to be told apart before either is reported to the model.
    """
    return _on_screen(str(image_path), target, config)


def _on_screen(path: str, target: str, config: dict[str, Any] | None) -> bool:
    """Is the thing being asked for on this screen at all?

    Called BEFORE grounding. A "no" is the honest answer to "where is the
    update banner" when the banner was dismissed minutes ago — the incident
    this check exists for, where the model answered with a dock icon at
    (624, 1029) and `desktop_click` on that launches an application.

    Best-effort like everything else here: anything that goes wrong counts as
    present, so a broken check never turns into a screen reported as empty.
    """
    if not target.strip():
        return True
    try:
        answer = _generate(path, _PRESENCE_PROMPT.format(target=target.strip()),
                           6, model_name(config))
    except Exception:
        return True
    return not answer.strip().lower().lstrip("\"'*- ").startswith("no")


def _image_size(path: str) -> tuple[int, int]:
    from PIL import Image

    with Image.open(path) as im:
        return im.size


def _to_pixels(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    """0-1000 normalised -> original-image pixels. See the module docstring for
    why this scale and not the processor's resized dimensions."""
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    px = lambda v: max(0, min(width - 1, round(v / _COORD_SCALE * width)))    # noqa: E731
    py = lambda v: max(0, min(height - 1, round(v / _COORD_SCALE * height)))  # noqa: E731
    left, right = sorted((px(x1), px(x2)))
    top, bottom = sorted((py(y1), py(y2)))
    return left, top, right, bottom


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _parse_elements(raw: str, width: int, height: int) -> list[dict[str, Any]]:
    """Pull grounding boxes out of whatever shape the model wrapped them in.

    Observed in one afternoon: a bare list, a ```json fence, and an object
    keyed "interactive_elements" whose value is the list. A strict parser
    would have thrown away two of those three, so walk the decoded structure
    for anything carrying a bbox instead of insisting on a top-level shape.
    """
    text = raw.strip()
    fenced = _JSON_BLOCK_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    data: Any = None
    try:
        data = json.loads(text)
    except Exception:
        # A truncated or chatty reply: take the outermost bracketed span.
        for opener, closer in (("[", "]"), ("{", "}")):
            start, end = text.find(opener), text.rfind(closer)
            if start != -1 and end > start:
                try:
                    data = json.loads(text[start:end + 1])
                    break
                except Exception:
                    continue
    if data is None:
        return []

    found: list[dict[str, Any]] = []

    def walk(node: Any):
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        box = node.get("bbox_2d") or node.get("bbox") or node.get("box")
        if isinstance(box, (list, tuple)) and len(box) >= 4:
            try:
                left, top, right, bottom = _to_pixels(list(box), width, height)
            except Exception:
                return
            found.append({
                "label": str(node.get("label") or node.get("text") or "element"),
                "box": (left, top, right, bottom),
                "center": ((left + right) // 2, (top + bottom) // 2),
            })
            return
        for value in node.values():
            walk(value)

    walk(data)
    return found


def format_elements(elements: list[dict[str, Any]]) -> str:
    """One line per element, coordinates first — the form a click needs."""
    if not elements:
        return "No interactive elements were located on screen."
    return "\n".join(
        f"  ({e['center'][0]}, {e['center'][1]})  {e['label']}"
        for e in elements
    )
