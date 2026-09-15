"""Vision: the coordinate space, the parser, and the memory bracket.

Nothing here loads a VLM. The parts worth testing are the ones that turn a
model's answer into a click — a wrong scale factor or a shape the parser
silently drops both end as a click on empty space, which is exactly the
failure mode vision was added to remove.
"""

import pytest

from symbio import vision


# ---- the coordinate space, measured rather than assumed ----

def test_grounding_boxes_rescale_from_the_normalised_space():
    """Qwen3-VL emits boxes normalised 0-1000 on each axis, relative to the
    ORIGINAL image. Verified 2026-09-07 against a real 1265x1917 screenshot:
    the model's box for the text input, [32, 68, 152, 80], rescaled by w/1000
    and h/1000 lands on (40, 130)-(192, 153), and the element measured off the
    pixels sits at (40, 135)-(193, 154).

    The two rival readings were both checked and both wrong — the processor's
    resized tensor (1280x1920) and the token grid (40x60) — so this is pinned
    deliberately. A click aimed by the wrong scale is worse than no click."""
    assert vision._to_pixels([32, 68, 152, 80], 1265, 1917) == (40, 130, 192, 153)


def test_a_box_is_clamped_to_the_image():
    """A model that overshoots 1000 must not produce a coordinate off-screen."""
    left, top, right, bottom = vision._to_pixels([0, 0, 1200, 1200], 800, 600)
    assert (left, top) == (0, 0)
    assert (right, bottom) == (799, 599)


def test_a_reversed_box_is_normalised_rather_than_dropped():
    """x2 < x1 happens; it describes the same rectangle."""
    assert (vision._to_pixels([152, 80, 32, 68], 1000, 1000)
            == vision._to_pixels([32, 68, 152, 80], 1000, 1000))


def test_the_centre_is_what_a_click_gets():
    els = vision._parse_elements(
        '[{"bbox_2d": [0, 0, 100, 100], "label": "button"}]', 1000, 1000)
    assert els[0]["center"] == (50, 50)


# ---- the parser has to survive the shapes the model actually emits ----

def test_a_bare_list_of_boxes_parses():
    els = vision._parse_elements(
        '[{"bbox_2d": [10, 20, 30, 40], "label": "Post"}]', 1000, 1000)
    assert [e["label"] for e in els] == ["Post"]


def test_a_json_fence_parses():
    els = vision._parse_elements(
        '```json\n[{"bbox_2d": [10, 20, 30, 40], "label": "Post"}]\n```',
        1000, 1000)
    assert [e["label"] for e in els] == ["Post"]


def test_a_wrapper_object_parses():
    """Observed live: the model answered with the list under a key it invented
    for itself, "interactive_elements". A parser that insisted on a top-level
    list would have thrown away a perfectly good look at the screen."""
    els = vision._parse_elements(
        '{"interactive_elements": [{"bbox_2d": [10, 20, 30, 40], '
        '"label": "text input"}]}', 1000, 1000)
    assert [e["label"] for e in els] == ["text input"]


def test_prose_around_the_json_is_tolerated():
    els = vision._parse_elements(
        'Here is what I found:\n[{"bbox_2d": [1, 2, 3, 4], "label": "link"}]\n'
        'Let me know if you need more.', 1000, 1000)
    assert len(els) == 1


def test_an_answer_with_no_boxes_is_empty_not_an_error():
    assert vision._parse_elements("I could not see any buttons.", 800, 600) == []


def test_an_entry_without_a_box_is_skipped_and_the_rest_kept():
    """One malformed entry must not cost the others."""
    els = vision._parse_elements(
        '[{"label": "no box here"}, {"bbox_2d": [1, 2, 3, 4], "label": "ok"}]',
        1000, 1000)
    assert [e["label"] for e in els] == ["ok"]


def test_formatting_puts_the_coordinates_where_a_click_can_read_them():
    els = vision._parse_elements(
        '[{"bbox_2d": [0, 0, 100, 100], "label": "Post button"}]', 1000, 1000)
    assert "(50, 50)" in vision.format_elements(els)


def test_no_elements_says_so_rather_than_returning_nothing():
    assert "No interactive elements" in vision.format_elements([])


# ---- memory: one model at a time on this machine ----

def test_release_reports_whether_anything_was_resident():
    """release() is called in a finally, including on paths where the model
    never loaded. It must be safe there and must not claim it freed a model
    it never held."""
    assert vision.release() is False


def test_config_selects_the_model_and_defaults_without_one():
    assert vision.model_name({}) == vision.DEFAULT_VISION_MODEL
    assert vision.model_name({"vision": {"model_name": "other/model"}}) == "other/model"


def test_vision_can_be_turned_off():
    assert vision.is_enabled({}) is True
    assert vision.is_enabled({"vision": {"enabled": False}}) is False


def test_a_missing_backend_raises_rather_than_returning_a_plausible_answer():
    """If the VLM cannot load, the caller has to be able to tell. Returning an
    empty description would read like a screen with nothing on it, and the
    model would go back to guessing — which is the state this replaces."""
    with pytest.raises(vision.VisionUnavailable):
        vision.load("definitely/not-a-real-model-name")


# ---- what you ask for decides whether grounding works ----

def test_an_empty_target_falls_back_to_the_generic_sweep(monkeypatch):
    """No question to ground means enumerate; that is what works on a simple
    page and it must stay the default."""
    asked = []

    def fake_generate(path, question, max_tokens, name):
        asked.append(question)
        return '[{"bbox_2d": [0, 0, 10, 10], "label": "x"}]'

    monkeypatch.setattr(vision, "_generate", fake_generate)
    monkeypatch.setattr(vision, "_image_size", lambda p: (100, 100))
    vision.locate("shot.png")

    assert len(asked) == 1
    assert vision._GENERIC_TARGETS in asked[0]


def test_a_question_is_grounded_directly(monkeypatch):
    """Measured on a real 1920x1080 desktop: the generic sweep returned
    nothing at all, while naming the target landed within a few pixels. The
    caller's question therefore has to reach the grounding pass."""
    asked = []

    def fake_generate(path, question, max_tokens, name):
        asked.append(question)
        return '[{"bbox_2d": [0, 0, 10, 10], "label": "x"}]'

    monkeypatch.setattr(vision, "_generate", fake_generate)
    monkeypatch.setattr(vision, "_image_size", lambda p: (100, 100))
    vision.locate("shot.png", "where is the Post button?")

    assert len(asked) == 1, "a targeted query that works needs no second pass"
    assert "where is the Post button?" in asked[0]


def test_a_targeted_query_that_finds_nothing_falls_back(monkeypatch):
    """Rather than reporting a screen with nothing on it."""
    asked = []

    def fake_generate(path, question, max_tokens, name):
        asked.append(question)
        return "I cannot see that." if len(asked) == 1 else \
            '[{"bbox_2d": [0, 0, 10, 10], "label": "found"}]'

    monkeypatch.setattr(vision, "_generate", fake_generate)
    monkeypatch.setattr(vision, "_image_size", lambda p: (100, 100))
    found = vision.locate("shot.png", "where is the Post button?")

    assert len(asked) == 2
    assert vision._GENERIC_TARGETS in asked[1]
    assert [e["label"] for e in found] == ["found"]


def test_a_generic_sweep_that_finds_nothing_does_not_retry_itself(monkeypatch):
    """The fallback IS the generic sweep, so repeating it would just cost a
    second 15-second generation for the same answer."""
    asked = []

    def fake_generate(path, question, max_tokens, name):
        asked.append(question)
        return "Nothing here."

    monkeypatch.setattr(vision, "_generate", fake_generate)
    monkeypatch.setattr(vision, "_image_size", lambda p: (100, 100))
    assert vision.locate("shot.png") == []
    assert len(asked) == 1


# ---- a coordinate for something that is not there ----

def test_a_located_box_that_fails_checking_is_dropped(monkeypatch):
    """Asked for something absent, the model returns a confident coordinate
    for something else. Measured 2026-09-07: asked for a macOS update banner
    dismissed minutes earlier, it pointed at (624, 1029) — a dock icon.
    desktop_click there launches an application, so on the desktop a wrong
    coordinate is not a missed click but a different action."""
    monkeypatch.setattr(vision, "_image_size", lambda p: (100, 100))
    monkeypatch.setattr(vision, "_generate",
                        lambda *a, **k: '[{"bbox_2d": [1, 2, 3, 4], "label": "x"}]')
    monkeypatch.setattr(vision, "_verify_element", lambda *a, **k: False)

    assert vision.locate("shot.png", "the update banner") == []


def test_a_rejected_target_does_not_fall_back_to_a_generic_sweep(monkeypatch):
    """The sweep answered a rejected target with a box labelled "interactive
    element" at the dead centre of the screen, handed to the model as
    something to click. An empty list is the honest answer."""
    calls = []

    def fake_generate(path, question, max_tokens, name):
        calls.append(question)
        return '[{"bbox_2d": [1, 2, 3, 4], "label": "x"}]'

    monkeypatch.setattr(vision, "_image_size", lambda p: (100, 100))
    monkeypatch.setattr(vision, "_generate", fake_generate)
    monkeypatch.setattr(vision, "_verify_element", lambda *a, **k: False)
    vision.locate("shot.png", "the update banner")

    assert len(calls) == 1, f"no second sweep after a rejection, got {calls}"


def test_checking_is_skipped_for_the_generic_sweep(monkeypatch):
    """There is no specific target to check a crop against."""
    monkeypatch.setattr(vision, "_image_size", lambda p: (100, 100))
    monkeypatch.setattr(vision, "_generate",
                        lambda *a, **k: '[{"bbox_2d": [1, 2, 3, 4], "label": "x"}]')
    monkeypatch.setattr(vision, "_verify_element",
                        lambda *a, **k: pytest.fail("should not verify a sweep"))

    assert len(vision.locate("shot.png")) == 1


def test_a_check_that_cannot_run_keeps_the_coordinate():
    """Best-effort, like every other check here: a broken verification must
    never discard a good coordinate."""
    # A path that cannot be opened is the simplest form of "the check could
    # not run".
    assert vision._verify_element("nope.png", {"box": (1, 2, 3, 4)}, "x", {}) is True


def test_an_element_with_no_box_is_not_checked():
    assert vision._verify_element("nope.png", {}, "x", {}) is True
