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
        import mlx_vlm  # noqa: F401
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
            from mlx_vlm import load as vlm_load
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
        import mlx.core as mx

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
    from mlx_vlm import generate as vlm_generate
    from mlx_vlm.prompt_utils import apply_chat_template

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
    raw = _generate(path, _LOCATE_PROMPT.format(what=targets),
                    max_tokens, model_name(config))
    found = _parse_elements(raw, width, height)
    if found and targeted:
        if verify:
            found = [e for e in found if _verify_element(path, e, targets, config)]
        # Everything it pointed at turned out to be something else, which is
        # the answer: the thing asked about is not on screen. Do NOT sweep
        # generically here. Observed 2026-09-07 when that fallback was
        # unconditional: the check correctly rejected a fabricated location for
        # a dismissed notification, the sweep then returned a single box
        # labelled "interactive element" at the dead centre of the screen, and
        # it was handed to the model under "clickable elements". A meaningless
        # coordinate offered as a click target is worse than an empty list.
        return found
    if found or not targeted:
        return found
    # The targeted phrasing produced no boxes at all — which can just mean it
    # was worded awkwardly. Sweep generically so the caller still learns the
    # layout.
    raw = _generate(path, _LOCATE_PROMPT.format(what=_GENERIC_TARGETS),
                    max_tokens, model_name(config))
    return _parse_elements(raw, width, height)


# How much context to include around a located box when checking it. A box
# cropped exactly to its own bounds is hard to recognise out of context.
_VERIFY_PAD = 40


def _verify_element(path: str, element: dict[str, Any], target: str,
                    config: dict[str, Any] | None) -> bool:
    """Crop what was located and ask whether it really shows the target.

    Asked to find something that is NOT on screen, the model does not say so —
    it returns a confident coordinate for something else. Measured 2026-09-07:
    asked for a macOS update banner that had been dismissed minutes earlier, it
    described a sidebar badge and returned (624, 1029), which is a dock icon.
    `desktop_click` on that launches an application. On the desktop the wrong
    coordinate is not a missed click, it is a different action entirely.

    So look again at just the box it pointed to. Cheap (0.5-2.3s on a small
    crop) and it separates the two cases cleanly: the dock crop came back "no",
    the genuinely-located editor tab came back "yes".

    Best-effort in the same direction as everything else here: anything that
    goes wrong counts as verified, so a broken check never discards a good
    coordinate.
    """
    box = element.get("box")
    if not box:
        return True
    try:
        from PIL import Image

        with Image.open(path) as im:
            left, top, right, bottom = box
            crop = im.crop((max(0, left - _VERIFY_PAD), max(0, top - _VERIFY_PAD),
                            min(im.width, right + _VERIFY_PAD),
                            min(im.height, bottom + _VERIFY_PAD)))
            if not crop.width or not crop.height:
                return True
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
                temp = fh.name
            crop.save(temp)
        try:
            answer = _generate(
                temp, f"Does this image show {target}? Answer only yes or no.",
                12, model_name(config))
        finally:
            with contextlib.suppress(OSError):
                os.unlink(temp)
    except Exception:
        return True
    return not answer.strip().lower().startswith("no")


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
