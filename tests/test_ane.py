"""The side models on the Neural Engine: the decision model, OCR for
see_screen, and the Swift helper behind them.

Most of these run against a fake helper. The last ones build and run the real
symbio_ane helper and are marked `uses_ane` (macOS with the command-line
tools); they are what proves the text is read on the Neural Engine rather
than said to be.
"""
import sys

import pytest

from symbio.app import ane, decider


def _fake_embed(monkeypatch, vectors_for):
    calls = []

    def request(payload, timeout=None):
        calls.append(payload)
        if payload["op"] == "embed":
            return {"ok": True, "vectors": [vectors_for(t) for t in payload["texts"]], "ms": 7}
        return {"ok": False, "status": {"available": False}}

    monkeypatch.setattr(ane, "request", request)
    monkeypatch.setattr(ane, "enabled", lambda config=None: True)
    monkeypatch.setattr(ane, "_decide_state", {"unavailable_until": 0.0})
    monkeypatch.setattr(decider, "_index", {})
    return calls


def _one_hot_by_label(text):
    """A toy embedding: each seed points along its label's axis."""
    labels = list(decider.LABELS)
    for label, items in decider.SEEDS.items():
        if text in items:
            return [1.0 if name == label else 0.0 for name in labels]
    # Unseen messages: point at "code" if they mention a traceback, else chat.
    target = "code" if "traceback" in text else "chat"
    return [1.0 if name == target else 0.0 for name in labels]


def test_the_vote_decides_think_from_the_nearest_examples(monkeypatch, tmp_path):
    monkeypatch.setattr(decider.constants, "PROJECT_DIR", tmp_path)
    _fake_embed(monkeypatch, _one_hot_by_label)
    assert decider.decide("yo what's up")["think"] is False
    decision = decider.decide("here is my traceback, help")
    assert decision == {**decision, "label": "code", "think": True, "source": "nl-contextual"}


def test_the_neural_engine_encoder_goes_first_and_apple_is_not_asked(monkeypatch, tmp_path):
    """Measured: the MiniLM vote on the ANE got 30/31 in under a millisecond;
    Apple's on-device model 27/31 in ~0.7 s. The fast one decides first."""
    monkeypatch.setattr(decider.constants, "PROJECT_DIR", tmp_path)
    calls = _fake_embed(monkeypatch, _one_hot_by_label)
    monkeypatch.setattr(ane, "encoder_dir", lambda: tmp_path)
    monkeypatch.setattr(ane, "encode", lambda texts: {
        "ok": True, "vectors": [_one_hot_by_label(t) for t in texts], "ms": 1})
    decision = decider.decide("here is my traceback, help")
    assert decision["source"] == "minilm-ane" and decision["think"] is True
    assert not [c for c in calls if c["op"] == "decide"], "a confident vote needs no second opinion"


def test_the_seed_vectors_are_embedded_once_and_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(decider.constants, "PROJECT_DIR", tmp_path)
    calls = _fake_embed(monkeypatch, _one_hot_by_label)
    decider.decide("hello")
    decider._index.clear()                 # a new process: reads the cache file
    decider.decide("hello again")
    seed_batches = [c for c in calls if c["op"] == "embed" and len(c["texts"]) > 1]
    assert len(seed_batches) == 1


def test_no_helper_falls_back_to_the_regex(monkeypatch):
    monkeypatch.setattr(ane, "enabled", lambda config=None: False)
    assert decider.decide("fix this bug in main.py") == {
        "label": None, "think": True, "source": "regex"}


def test_ocr_lines_become_click_elements():
    result = {"ok": True, "lines": [{"text": "Post", "conf": 0.9, "box": [100, 40, 60, 20]},
                                    {"text": " ", "box": [0, 0, 1, 1]}]}
    [element] = ane.ocr_elements(result)
    assert element["label"] == "Post"
    assert element["box"] == (100, 40, 160, 60) and element["center"] == (130, 50)


@pytest.mark.parametrize("question,reading", [
    ("", True), ("what does the error say?", True), ("read the dialog", True),
    ("what's on my screen?", True), ("what colour is the logo?", False),
    ("describe the chart", False), ("click the blue icon", False),
])
def test_questions_about_words_are_answered_from_the_text(question, reading):
    assert ane.is_reading_question(question) is reading


def test_a_region_with_words_is_named_by_its_words_not_the_vlm(monkeypatch):
    from symbio import vision

    asked = []
    monkeypatch.setattr(ane, "enabled", lambda config=None: True)
    monkeypatch.setattr(ane, "read_crop", lambda path, box: "Post")
    monkeypatch.setattr(vision, "_ask_crop", lambda *a: asked.append(a) or "a button")
    assert vision._name_crop("shot.png", (0, 0, 60, 20), "vlm", 40, {}) == '"Post" (text)'
    assert asked == []

    monkeypatch.setattr(ane, "read_crop", lambda path, box: "")      # an icon: no words
    assert vision._name_crop("shot.png", (0, 0, 60, 20), "vlm", 40, {}) == "a button"
    assert len(asked) == 1


# ---- the real helper -------------------------------------------------------

needs_mac = pytest.mark.skipif(sys.platform != "darwin", reason="the Neural Engine helper is macOS-only")


@pytest.fixture
def real_helper(tmp_path, monkeypatch):
    if ane.helper_binary() is None:
        pytest.skip(f"cannot build the helper: {ane._build_error}")
    yield
    ane.close()


def _text_image(path, text):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (900, 200), "white")
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 48)
    ImageDraw.Draw(image).text((40, 70), text, fill="black", font=font)
    image.save(path)
    return path


@needs_mac
@pytest.mark.uses_ane
def test_the_helper_reads_text_on_the_neural_engine(real_helper, tmp_path):
    shot = _text_image(tmp_path / "shot.png", "Submit your order now")
    result = ane.ocr(shot)
    assert result["ok"], result
    assert "Submit your order now" in [line["text"] for line in result["lines"]]
    assert result["devices"] == {"VNComputeStageMain": "neuralEngine"}


@needs_mac
@pytest.mark.uses_ane
def test_see_screen_reads_the_screen_without_the_vlm(real_helper, tmp_path, monkeypatch):
    """A question about words is answered on the Neural Engine: no mlx-vlm,
    no headmaster unload — the ~10 GB swap a VLM look costs."""
    from symbio import computer, vision
    from tests.test_thinking_truncation import _session

    shot = _text_image(tmp_path / "desk.png", "Invoice overdue: pay now")
    session = _session(monkeypatch, tmp_path, [])
    monkeypatch.setattr(session, "_desktop_enabled", lambda: True)
    monkeypatch.setattr(session, "_ax_look", lambda question="": "")
    monkeypatch.setattr(computer, "desktop_screenshot_path", lambda: shot)
    monkeypatch.setattr(computer, "screenshot_is_blank", lambda path: False)
    monkeypatch.setattr(vision, "available", lambda: False)     # no VLM at all
    woke = []
    monkeypatch.setattr(session, "_run_vision", lambda *a: woke.append(a))

    look = session._see_screen({"target": "desktop", "question": "what does the banner say?"})

    assert "Invoice overdue: pay now" in look
    assert "read on the Neural Engine" in look
    assert woke == [], "the VLM must not be woken for a question about words"


@needs_mac
@pytest.mark.uses_ane
def test_the_helper_embeds_sentences(real_helper):
    result = ane.request({"op": "embed", "texts": ["hello there", "fix this traceback"]})
    assert result["ok"] and len(result["vectors"]) == 2 and len(result["vectors"][0]) == result["dims"]
    norm = sum(v * v for v in result["vectors"][0]) ** 0.5
    assert abs(norm - 1.0) < 1e-6
