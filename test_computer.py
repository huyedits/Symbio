"""Tests for symbio.computer.BrowserSession: every action taken before the
browser has ever been opened must return a graceful error string, not raise
— a real crash (RuntimeError from _ensure_open escaping click()/get_text()/
etc. because they were called outside their own try/except) took down a
live session; see the git history for the exact traceback."""

from symbio.computer import BrowserSession


def test_click_before_open_does_not_raise():
    session = BrowserSession()
    result = session.click(text="first video")
    assert isinstance(result, str)
    assert "not open" in result.lower()


def test_get_text_before_open_does_not_raise():
    session = BrowserSession()
    result = session.get_text()
    assert isinstance(result, str)
    assert "not open" in result.lower()


def test_type_text_before_open_does_not_raise():
    session = BrowserSession()
    result = session.type_text("hello")
    assert isinstance(result, str)
    assert "not open" in result.lower()


def test_scroll_before_open_does_not_raise():
    session = BrowserSession()
    result = session.scroll("down")
    assert isinstance(result, str)
    assert "not open" in result.lower()


def test_press_before_open_does_not_raise():
    session = BrowserSession()
    result = session.press("Enter")
    assert isinstance(result, str)
    assert "not open" in result.lower()


def test_get_html_before_open_does_not_raise():
    session = BrowserSession()
    result = session.get_html()
    assert isinstance(result, str)
    assert "not open" in result.lower()


def test_evaluate_before_open_does_not_raise():
    session = BrowserSession()
    result = session.evaluate("1+1")
    assert isinstance(result, str)
    assert "not open" in result.lower()


def test_screenshot_before_open_does_not_raise():
    session = BrowserSession()
    result = session.screenshot()
    assert isinstance(result, str)
    assert "not open" in result.lower()
