"""Screenshot pixels are not mouse points, and on the desktop the difference
is not a missed click — it is a click somewhere else.

A Retina capture is 2880x1800 while the mouse moves in 1440x900 points, so an
unconverted coordinate for the middle of the screen lands off the bottom-right
edge, and one aimed at a menu lands in the middle of a document. Nothing here
calls pyautogui: the tests record what WOULD have been clicked.
"""
import pytest

from symbio import computer
from symbio.app import chat_tools


@pytest.fixture
def mouse(monkeypatch):
    """Records the point handed to the mouse, and pins a 2x display."""
    clicks = []
    monkeypatch.setattr(computer, "desktop_click",
                        lambda x, y, clicks_=1, button="left": clicks.append((x, y)) or "ok")
    monkeypatch.setattr(computer, "_screen_pixel_size", lambda: (2880, 1800))
    monkeypatch.setattr(computer, "desktop_pixel_scale", lambda size: 2.0)
    return clicks


def _session():
    class S(chat_tools.ToolsMixin):
        def __init__(self):
            self.config = {}
    return S()


def test_a_look_first_gives_the_mouse_the_converted_point(mouse):
    session = _session()
    session._last_desktop_shot_size = (2880, 1800)

    session._desktop_action("desktop_click", {"x": 1440, "y": 900})

    assert mouse == [(720, 450)]


def test_clicking_without_a_cached_size_still_converts(mouse):
    """The regression the helper was rewritten to close, reopened by its
    caller. desktop_click_in_image derives the scale itself when given no
    image size — its docstring says the cached size "defaults to (0, 0) ...
    at which point the conversion was skipped and the raw physical pixels went
    straight to the mouse". The caller still branched on that cached size and
    fell through to the unscaled call, so the fix was bypassed in exactly the
    case it was written for: PIL missing, an unreadable capture, or a
    desktop_click issued before any see_screen."""
    session = _session()
    session._last_desktop_shot_size = (0, 0)

    session._desktop_action("desktop_click", {"x": 1440, "y": 900})

    assert mouse == [(720, 450)], "raw physical pixels reached the mouse"


def test_the_registry_front_end_converts_too(mouse, monkeypatch):
    """AIAgent runs the same tools through a different dispatcher, and its
    desktop_click passed the coordinate through untouched — while the same
    diff rewrote desktop_screenshot's description to promise coordinates
    "which desktop_click can use directly"."""
    from symbio import tools

    # tools.py binds the names at import time, so the patch has to land there
    # too — which is itself the finding: it holds a direct reference to the
    # raw mouse function.
    monkeypatch.setattr(tools, "desktop_click_in_image",
                        computer.desktop_click_in_image)
    agent = type("A", (), {})()
    tools._tool_desktop_click(agent, {"x": 1440, "y": 900})

    assert mouse == [(720, 450)]
