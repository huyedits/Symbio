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


# ---- reading a screen as blobs rather than as one picture ----

def _shot(tmp_path):
    """A gradient wallpaper with two controls on it, saved."""
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (1200, 800))
    draw = ImageDraw.Draw(im)
    for y in range(800):
        draw.line([(0, y), (1200, y)], fill=(40 + y // 12, 60 + y // 20, 90))
    draw.rectangle([100, 100, 260, 140], fill=(240, 240, 240), outline=(20, 20, 20))
    draw.text((120, 115), "Submit", fill=(0, 0, 0))
    draw.rectangle([400, 300, 760, 340], fill=(255, 255, 255), outline=(120, 120, 120))
    draw.text((410, 315), "type here", fill=(30, 30, 30))
    path = tmp_path / "shot.png"
    im.save(path)
    return str(path)


def test_blobs_name_the_regions_the_pixels_found(monkeypatch, tmp_path):
    """One crop, one question, one name. The crop is chosen by arithmetic; the
    model only supplies the label."""
    seen = []

    def fake_generate(path, question, max_tokens, name):
        seen.append(question)
        return "Submit button" if len(seen) == 1 else "text field reading type here"

    monkeypatch.setattr(vision, "_generate", fake_generate)
    found = vision.read_blobs(_shot(tmp_path))

    assert [e["label"] for e in found] == ["Submit button",
                                           "text field reading type here"]
    assert all("crop" in q for q in seen)


def test_blob_coordinates_come_from_the_pixels_not_the_model(monkeypatch, tmp_path):
    """locate's known failure is fabricating plausible positions for rows of
    near-identical controls. A box measured off the edge map cannot drift."""
    monkeypatch.setattr(vision, "_generate",
                        lambda *a, **k: "a button at (999, 999)")
    found = vision.read_blobs(_shot(tmp_path))

    boxes = {e["box"] for e in found}
    assert boxes, "the controls must be found"
    for left, top, right, bottom in boxes:
        assert 90 <= left <= 420 and 90 <= top <= 350


def test_a_crop_that_is_only_background_is_dropped(monkeypatch, tmp_path):
    """Wallpaper offered as something to click is worse than a short list."""
    monkeypatch.setattr(vision, "_generate",
                        lambda *a, **k: "nothing — this is just the wallpaper")
    assert vision.read_blobs(_shot(tmp_path)) == []


def test_a_blank_screen_costs_no_generations(monkeypatch, tmp_path):
    """The segmentation runs first and finds nothing, so the VLM is never
    asked. A screen with no structure in it is answered in 30ms."""
    from PIL import Image

    path = tmp_path / "blank.png"
    Image.new("RGB", (1280, 800), (252, 252, 252)).save(path)
    monkeypatch.setattr(vision, "_generate",
                        lambda *a, **k: pytest.fail("nothing to name"))

    assert vision.read_blobs(str(path)) == []


def test_the_number_of_crops_is_capped(monkeypatch, tmp_path):
    """Each crop is a generation. Twelve of them is a look that takes half a
    minute, on a machine where the headmaster is asleep for the whole of it."""
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (1200, 800), (30, 60, 90))
    draw = ImageDraw.Draw(im)
    for index in range(12):
        x, y = 60 + (index % 4) * 280, 80 + (index // 4) * 220
        draw.rectangle([x, y, x + 180, y + 60], fill=(255, 255, 255), outline=(0, 0, 0))
        draw.text((x + 10, y + 20), f"button {index}", fill=(0, 0, 0))
    path = tmp_path / "many.png"
    im.save(path)

    calls = []
    monkeypatch.setattr(vision, "_generate",
                        lambda *a, **k: calls.append(1) or "a button")

    assert len(vision.read_blobs(str(path), limit=3)) == 3
    assert len(calls) == 3


def test_blob_reading_can_be_turned_off(monkeypatch, tmp_path):
    monkeypatch.setattr(vision, "_generate",
                        lambda *a, **k: pytest.fail("disabled"))
    assert vision.read_blobs(_shot(tmp_path), config={"vision": {"blobs": False}}) == []


def test_a_sweep_that_finds_nothing_falls_through_to_blobs(monkeypatch, tmp_path):
    """The measured failure this exists for: on a dense desktop the generic
    "find every interactive element" sweep returned nothing at all, and the
    screen was reported as empty while a full dock was on it."""
    path = _shot(tmp_path)
    answers = ["Nothing here.", "Submit button", "text field"]

    def fake_generate(image, question, max_tokens, name):
        return answers.pop(0) if answers else "a control"

    monkeypatch.setattr(vision, "_generate", fake_generate)
    found = vision.locate(path)

    assert [e["label"] for e in found] == ["Submit button", "text field"]


# ---- finding one named thing among the blobs ----

def test_the_crop_the_model_says_yes_to_is_the_one_returned(monkeypatch, tmp_path):
    """The model is never asked for a position — only whether the crop it is
    shown is the thing. Positions come from the edge map."""
    path = _shot(tmp_path)
    answers = []

    def fake_generate(image, question, max_tokens, name):
        answers.append(question)
        return "yes" if len(answers) == 2 else "no"

    monkeypatch.setattr(vision, "_generate", fake_generate)
    hits = vision.find_in_blobs(path, "the text field")

    assert len(hits) == 1
    assert hits[0]["label"] == "the text field"
    assert all("Answer only yes or no" in q for q in answers)


def test_a_thing_that_is_not_there_finds_nothing(monkeypatch, tmp_path):
    """locate's recorded failure is a confident coordinate for something else
    — a dock icon offered as a dismissed banner. A closed question about a
    crop cannot produce that."""
    monkeypatch.setattr(vision, "_generate", lambda *a, **k: "no")
    assert vision.find_in_blobs(_shot(tmp_path), "a video player") == []


def test_hits_come_back_best_scoring_first(monkeypatch, tmp_path):
    """A caller taking [0] should get the most prominent match, not an
    arbitrary one."""
    monkeypatch.setattr(vision, "_generate", lambda *a, **k: "Yes.")
    hits = vision.find_in_blobs(_shot(tmp_path), "a control")

    assert len(hits) == 2
    assert hits[0]["score"] >= hits[1]["score"]


def test_an_empty_target_asks_nothing(monkeypatch, tmp_path):
    monkeypatch.setattr(vision, "_generate",
                        lambda *a, **k: pytest.fail("nothing was asked for"))
    assert vision.find_in_blobs(_shot(tmp_path), "   ") == []


# ---- the hybrid: the model picks, the pixels measure ----

def test_a_drifted_box_snaps_onto_the_region_it_overlaps(tmp_path):
    """locate's recorded drift is evenly-spaced positions with fabricated x,
    off by more than a full icon across a row. A box that lands ON something
    structural takes that thing's measured bounds."""
    path = _shot(tmp_path)
    drifted = [{"label": "Submit", "box": (120, 118, 280, 158),
                "center": (200, 138)}]

    snapped = vision.snap_to_blobs(path, drifted)

    assert snapped[0]["snapped"] is True
    # The measured bounds of the drawn button, within the dilation's pixel.
    assert snapped[0]["box"] == (99, 99, 261, 141)
    assert snapped[0]["center"] == (180, 120)
    assert snapped[0]["label"] == "Submit", "the model still says what it is"


def test_a_box_over_empty_space_keeps_its_numbers_and_is_marked(tmp_path):
    """A coordinate over nothing structural is what a fabricated one looks
    like. Say so rather than silently moving it to the nearest real thing —
    the nearest real thing may be a dock icon, and clicking it launches an
    application."""
    path = _shot(tmp_path)
    invented = [{"label": "update banner", "box": (900, 600, 1000, 650),
                 "center": (950, 625)}]

    snapped = vision.snap_to_blobs(path, invented)

    assert snapped[0]["snapped"] is False
    assert snapped[0]["center"] == (950, 625)


def test_a_box_drawn_around_a_control_and_its_label_still_agrees(tmp_path):
    """The model's box and the measured region are the same thing described
    at two sizes; either centre inside the other is agreement."""
    path = _shot(tmp_path)
    loose = [{"label": "the form", "box": (380, 280, 800, 380),
              "center": (590, 330)}]

    assert vision.snap_to_blobs(path, loose)[0]["snapped"] is True


def test_snapping_survives_an_image_it_cannot_read():
    """Best-effort like every other check here: a broken correction must not
    discard a usable coordinate."""
    elements = [{"label": "x", "box": (1, 2, 3, 4), "center": (2, 3)}]
    assert vision.snap_to_blobs("nope.png", elements) == elements


def test_an_element_without_a_box_passes_through(tmp_path):
    elements = [{"label": "x"}]
    assert vision.snap_to_blobs(_shot(tmp_path), elements) == elements


def test_grounded_boxes_come_back_corrected(monkeypatch, tmp_path):
    """The whole point of the hybrid, end to end: one generation says what is
    there, the edge map says where."""
    path = _shot(tmp_path)
    monkeypatch.setattr(
        vision, "_generate",
        lambda *a, **k: '[{"bbox_2d": [100, 148, 233, 197], "label": "Submit"}]')
    monkeypatch.setattr(vision, "_verify_element", lambda *a, **k: True)

    found = vision.locate(path, "the Submit button")

    assert found[0]["box"] == (99, 99, 261, 141)
    assert found[0]["center"] == (180, 120)
    assert found[0]["snapped"] is True
