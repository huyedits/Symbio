"""Posting under the user's own name, and proving it went out.

This project has already made a post and reported that it had not — "clicked"
says what the mouse did, never what it achieved — and has already failed to
ground x.com's composer with vision at all, because the composer is 28px tall
and the vision worker cannot see anything thinner than one patch.

So the path is: fill by selector, send by the button x's own code labels, and
then read the timeline back. The verdict is a string the model cannot shape,
and "CONFIRMED" is only ever written when this code found the exact text
rendered in an article afterwards.

Nothing here touches the network. The page is a stub that answers the same
queries a real one would.
"""
import pytest

from symbio import computer, safety


class _Element:
    def __init__(self, page, kind, text="", disabled=False):
        self.page, self.kind, self.text, self.disabled = page, kind, text, disabled

    def click(self):
        self.page.clicked.append(self.kind)
        if self.kind == "post" and not self.disabled:
            self.page.send()

    def inner_text(self):
        return self.page.composer_text if self.kind == "composer" else self.text

    def get_attribute(self, name):
        if name == "aria-disabled":
            return "true" if self.disabled else "false"
        return None


class _Page:
    """Enough of a Playwright page for this one path."""

    def __init__(self, url="https://x.com/home", logged_in=True,
                 post_disabled=False, delivers=True, clears=True):
        self.url = url
        self.logged_in = logged_in
        self.post_disabled = post_disabled
        self.delivers = delivers
        self.clears = clears
        self.composer_text = ""
        self.timeline = []
        self.clicked = []

    # -- what post_to_x calls --
    def query_selector(self, selector):
        if selector == computer.X_COMPOSER:
            return _Element(self, "composer")
        if selector in computer.X_POST_BUTTONS:
            return _Element(self, "post", disabled=self.post_disabled)
        if selector in computer.X_LOGGED_OUT:
            return None if self.logged_in else _Element(self, "login")
        return None

    @property
    def keyboard(self):
        page = self

        class _Keyboard:
            @staticmethod
            def type(text):
                page.composer_text += text

            @staticmethod
            def press(combo):
                if combo == "Meta+Enter":
                    page.send()
        return _Keyboard()

    def evaluate(self, script, arg=None):
        return any(arg in entry for entry in self.timeline) if arg else False

    def wait_for_timeout(self, _ms):
        return None

    def send(self):
        if self.delivers:
            self.timeline.append(" ".join(self.composer_text.split()))
        if self.clears:
            self.composer_text = ""


def _session(page):
    session = computer.BrowserSession.__new__(computer.BrowserSession)
    session._ensure_open = lambda: page
    return session


def test_a_delivered_post_is_confirmed_from_the_timeline():
    page = _Page()

    verdict = _session(page).post_to_x("shipping the desktop window today")

    assert verdict.startswith(computer.X_CONFIRMED)
    assert page.timeline == ["shipping the desktop window today"]


def test_a_cleared_composer_alone_is_not_confirmation():
    """A discarded draft clears the box too. Necessary, not sufficient."""
    page = _Page(delivers=False, clears=True)

    verdict = _session(page).post_to_x("hello")

    assert verdict.startswith(computer.X_NOT_CONFIRMED)
    assert "may have gone out" in verdict and "check x.com" in verdict


def test_text_left_in_the_composer_reads_as_not_sent():
    page = _Page(delivers=False, clears=False)

    verdict = _session(page).post_to_x("hello")

    assert verdict.startswith(computer.X_NOT_CONFIRMED)
    assert "still holds the text" in verdict


def test_a_disabled_post_button_stops_before_sending():
    page = _Page(post_disabled=True)

    verdict = _session(page).post_to_x("hello")

    assert verdict.startswith(computer.X_NOT_CONFIRMED)
    assert page.timeline == []


def test_being_signed_out_is_said_plainly():
    page = _Page(logged_in=False)

    verdict = _session(page).post_to_x("hello")

    assert "Not signed in" in verdict
    assert page.timeline == []


def test_it_refuses_to_post_from_a_page_that_is_not_x():
    """It does not navigate on its own: posting is not something to do on a
    page nobody asked for."""
    page = _Page(url="https://news.ycombinator.com")

    verdict = _session(page).post_to_x("hello")

    assert verdict.startswith(computer.X_UNVERIFIABLE)
    assert page.timeline == []


@pytest.mark.parametrize("text,expected", [
    ("", "Nothing to post"),
    ("x" * 281, "281 characters"),
])
def test_an_unpostable_text_never_reaches_the_page(text, expected):
    page = _Page()

    verdict = _session(page).post_to_x(text)

    assert expected in verdict
    assert page.clicked == []


def test_posting_is_gated_at_the_top_of_the_risk_scale():
    """Published under the user's name, to an audience, with no undo this
    code controls."""
    risk = safety.assess_tool_risk("post_to_x", {"text": "hello"}, {})

    assert risk["risk_score"] == 3
    assert "publishes_publicly" in risk["flags"]


def test_the_confirmation_prompt_shows_what_would_be_posted():
    allowed, prompt = safety.maybe_confirm(
        "post_to_x", {"text": "shipping today"},
        safety.assess_tool_risk("post_to_x", {"text": "shipping today"}, {}),
        {"safety": {"enabled": True}}, lambda _p: False)

    assert allowed is False
    assert "shipping today" in prompt and "publicly" in prompt


@pytest.mark.parametrize("spelling", ["tweet", "post_tweet", "send_tweet", "post_to_x"])
def test_the_names_a_model_reaches_for_all_resolve(spelling):
    from symbio.app import tooling

    groups = {"browser", "memory", "terminal", "code", "web_search"}
    call = '<tool_call>{"name": "%s", "arguments": {"text": "hi"}}</tool_call>' % spelling

    assert tooling.parse_tools(call, groups) == [("post_to_x", {"text": "hi"})]
