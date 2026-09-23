"""The text-reading first pass of see_screen: symbio/ocr.py and its wiring.

The matcher's contract is that it declines rather than guesses. Every case
below is one the first version of it got WRONG on the 2026-09-23 truth set —
not missed, wrong: it returned a confident box for a different control.
"""

from pathlib import Path

import pytest

from symbio import ocr


def _run(text, box=(0, 0, 10, 10)):
    return {"text": text, "box": box}


def test_the_word_that_singles_a_run_out_outweighs_the_common_one():
    """"the PROBLEMS tab above the terminal panel" landed on the Terminal
    menu: "terminal" and "problems" each matched one word and scored alike.
    "terminal" is on three runs of that screen; "problems" on one."""
    runs = [_run("Terminal", (396, 8, 458, 23)),
            _run("PROBLEMS", (363, 491, 427, 506)),
            _run("V TERMINAL", (374, 519, 449, 533)),
            _run("TERMINAL", (636, 491, 695, 505))]

    got = ocr.match(runs, "the PROBLEMS tab above the terminal panel")

    assert got["text"] == "PROBLEMS"


def test_a_tie_goes_to_the_casing_the_query_used():
    runs = [_run("TERMINAL", (636, 491, 695, 505)),
            _run("Terminal", (396, 8, 458, 23))]

    assert ocr.match(runs, "the Terminal menu in the menu bar")["text"] == "Terminal"


def test_two_identical_labels_are_declined_not_picked():
    """x.com has a sidebar Post button and a composer Post button. The first
    matcher took whichever came first, for three different questions."""
    runs = [_run("Post", (104, 638, 146, 654)), _run("Post", (806, 248, 844, 266))]

    assert ocr.match(runs, "the Post button inside the box holding the text") is None
    assert ocr.match(runs, "the Post button in the left sidebar") is None


def test_the_shorter_run_wins_when_it_is_the_label():
    runs = [_run("this post was written by a local model"), _run("Post")]

    assert ocr.match(runs, "the Post button")["text"] == "Post"


def test_nothing_is_returned_for_text_that_is_not_on_screen():
    runs = [_run("Submit Query"), _run("Open Details")]

    assert ocr.match(runs, "a video player with playback controls") is None
    assert ocr.match(runs, "the button") is None, "kind-of-thing words alone match nothing"


def test_find_fails_closed_on_an_unreadable_file(tmp_path):
    assert ocr.find(tmp_path / "missing.png", "the Submit Query button") == []


@pytest.mark.skipif(not ocr.available(), reason="needs macOS Vision bindings")
def test_find_reads_a_rendered_label(tmp_path):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (800, 200), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=40)
    draw.text((40, 30), "Cancel", fill="black", font=font)
    draw.text((420, 120), "Submit Query", fill="black", font=font)
    path = tmp_path / "shot.png"
    image.save(path)

    hits = ocr.find(path, "the Submit Query button")

    assert len(hits) == 1
    x, y = hits[0]["center"]
    assert 420 <= x <= 700 and 110 <= y <= 180, hits


# ---- see_screen: a text hit answers without waking the vision model ----

def _session(monkeypatch, ocr_hits):
    from symbio.app import chat_tools
    from symbio.app.config import DEFAULT_CONFIG

    monkeypatch.setattr(ocr, "find", lambda path, query: ocr_hits)
    calls = []

    class _Browser:
        is_open = True

        def controls_read(self, limit=25):
            return [], True

        def screenshot_path(self, full_page=False):
            return Path("shot.png")

    class S(chat_tools.ToolsMixin):
        config = {**DEFAULT_CONFIG, "browser": {"enabled": True}, "dispatch": {}}
        browser = _Browser()

        def _status(self, m):
            pass

        def _run_vision(self, shot, question):
            calls.append(question)
            return "a page", []

    from symbio import vision
    monkeypatch.setattr(vision, "is_enabled", lambda config=None: True)
    monkeypatch.setattr(vision, "available", lambda: True)
    return S(), calls


def test_a_text_hit_answers_without_the_vision_model(monkeypatch):
    hit = [{"label": "Submit Query", "box": (197, 133, 297, 156), "center": (247, 144)}]
    session, calls = _session(monkeypatch, hit)

    out = session._see_screen({"question": "the Submit Query button"})

    assert calls == [], "the VLM must not run when the text answered"
    assert "(247, 144)" in out and "browser_click_at" in out


def test_a_text_miss_falls_through_to_vision_and_is_not_absence(monkeypatch):
    session, calls = _session(monkeypatch, [])

    session._see_screen({"question": "the Finder icon in the dock"})

    assert calls == ["the Finder icon in the dock"]


def test_the_text_pass_can_be_switched_off(monkeypatch):
    hit = [{"label": "Submit Query", "box": (0, 0, 1, 1), "center": (0, 0)}]
    session, calls = _session(monkeypatch, hit)
    session.config = {**session.config, "vision": {**session.config.get("vision", {}),
                                                   "ocr": False}}

    session._see_screen({"question": "the Submit Query button"})

    assert calls, "vision.ocr=false must go straight to the vision model"
