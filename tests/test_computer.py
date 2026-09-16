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


# ---- typing must not report success when the keystrokes went nowhere ----

class _StubPage:
    """Minimal stand-in for a Playwright page. `landed` is what the focus
    check evaluates to; `raises` makes evaluate blow up."""

    def __init__(self, landed=True, raises=False):
        self._landed = landed
        self._raises = raises
        self.typed = []
        self.pressed = []

    def evaluate(self, script, arg=None):
        if self._raises:
            raise RuntimeError("page closed")
        return self._landed

    @property
    def keyboard(self):
        return self

    def type(self, text, delay=0):
        self.typed.append(text)

    def press(self, key):
        self.pressed.append(key)


def _session_on(page):
    session = BrowserSession()
    session._page = page
    return session


def test_typing_into_nothing_is_reported_as_a_failure():
    """Live 2026-09-02: asked to tweet, the agent typed with nothing focused,
    pressed Enter, and announced "your tweet has been successfully posted."
    keyboard.type sends keystrokes wherever focus is — the page body on a
    freshly loaded page — and the tool asserted success regardless, so nothing
    downstream could tell it had failed."""
    session = _session_on(_StubPage(landed=False))
    result = session.type_text("posted by symbio")
    assert "Type failed" in result
    assert "Click the field first" in result


def test_typing_into_a_focused_field_still_succeeds():
    session = _session_on(_StubPage(landed=True))
    assert session.type_text("hello").startswith("Typed 'hello'")


def test_an_explicit_selector_is_filled_and_not_focus_checked():
    """With a selector the text is put in with fill(), so the focus check does
    not apply — and must not be able to fail a working call."""
    session = _session_on(_StubPage(landed=False))
    result = session.type_text("hi", selector="#composer")
    assert "Type failed" not in result or "no visible element" in result


def test_a_page_that_will_not_evaluate_counts_as_landed():
    """Best-effort: a verification that cannot run must not turn a working
    type into a reported failure."""
    session = _session_on(_StubPage(raises=True))
    assert session.type_text("hello").startswith("Typed 'hello'")


def test_the_failure_reads_as_a_tool_error_so_the_retry_path_sees_it():
    from symbio.app import learn

    session = _session_on(_StubPage(landed=False))
    assert learn.sounds_like_tool_error(session.type_text("posted by symbio"))


# ---- modifier combinations, the difference between posting and not ----

def test_a_modifier_combination_normalizes_to_playwrights_spelling():
    """Live 2026-09-02: the agent pressed Enter to send a tweet. On X that
    inserts a newline and posts nothing — the composer wants cmd+enter. The
    key path already handled combinations; nothing told the model they
    existed, so the tool description now says so."""
    from symbio.computer import _normalize_key

    assert _normalize_key("cmd+enter") == "Meta+Enter"
    assert _normalize_key("ctrl+enter") == "Control+Enter"
    assert _normalize_key("command+return") == "Meta+Enter"
    assert _normalize_key("shift+tab") == "Shift+Tab"


def test_the_catalog_tells_the_model_combinations_exist():
    from symbio.app import tooling

    block = tooling.build_tools_block()
    assert "cmd+enter" in block


def test_a_combination_survives_the_tag_parser():
    from symbio.app import tooling

    assert tooling.parse_tools("<press>cmd+enter</press>") == [
        ("browser_press", {"key": "cmd+enter"})]


# ---- an action that changed nothing must say so ----

class _TextPage:
    """A browser stub whose page text the test controls."""

    def __init__(self, text):
        self.text = text

    def get_text(self):
        return self.text




def _mixin_session(browser):
    from symbio.app import chat_tools

    class S(chat_tools.ToolsMixin):
        def __init__(self, b):
            self.browser = b
            self.config = {"browser": {"enabled": True}}

    return S(browser)


def test_an_action_that_leaves_the_page_identical_is_flagged():
    """Live 2026-09-02: 'Post' matched three elements, the click took the
    first (the nav button, not the composer's submit), and the observation
    read 'Clicked element containing text Post' either way. The model
    announced the tweet published, twice, with the text still in the box."""
    s = _mixin_session(_TextPage("same"))
    note = s._no_effect_note("browser_click", "same", "Clicked element containing text 'Post'.")
    assert "did not change" in note
    assert "NOT happened" in note


def test_an_action_that_changes_the_page_says_nothing():
    s = _mixin_session(_TextPage("after"))
    assert s._no_effect_note("browser_click", "before", "Clicked.") == ""


def test_an_already_failed_action_is_not_piled_on():
    s = _mixin_session(_TextPage("same"))
    assert s._no_effect_note(
        "browser_click", "same", "Click failed: nothing matched.") == ""


def test_typing_is_judged_by_the_field_not_by_the_page_text():
    """get_text() reads rendered text, and a textarea's value is not rendered
    text — so a message typed into a composer leaves the page snapshot
    identical. This note used to fire on every successful type and tell the
    model the text "has NOT happened yet"; acting on that means typing the
    whole message again on top of itself. type_text verifies against the
    focused element's own value, which is better evidence than a text diff."""
    s = _mixin_session(_TextPage("same"))
    assert s._no_effect_note("browser_type", "same", "Typed 'hello'.") == ""


def test_scrolling_and_closing_are_not_judged():
    """A scroll at the bottom of a page legitimately changes nothing, and a
    closed browser has no 'after' to read."""
    s = _mixin_session(_TextPage("same"))
    assert s._no_effect_note("browser_scroll", "same", "Scrolled down.") == ""
    assert s._no_effect_note("browser_close", "same", "Closed.") == ""


def test_no_before_snapshot_means_no_claim():
    s = _mixin_session(_TextPage("same"))
    assert s._no_effect_note("browser_click", "", "Clicked.") == ""


# ---- Enter is only a submit on some pages ----

class _KeyPage(_StubPage):
    """Adds an answer for the Enter-submitted check, which _StubPage's single
    `landed` flag cannot express: whether the text arrived and whether the
    form went are separate questions about the same field."""

    def __init__(self, landed=True, submitted=True, focused=True, raises=False):
        super().__init__(landed=landed, raises=raises)
        self.submitted = submitted
        self.focused = focused
        self.restored = []

    def evaluate(self, script, arg=None):
        if self._raises:
            raise RuntimeError("page closed")
        # Four scripts reach this now and two of them take no argument, so
        # dispatching on `arg is not None` alone silently answered the
        # pre-type focus check with the Enter-submitted flag. Read the script.
        if arg is not None:
            if "dispatchEvent" in script:
                self.restored.append(arg)  # _restore_field_text
                return None
            return self._landed            # _text_landed
        if "includes" not in script and "trim" not in script:
            return self.focused            # _editable_focused
        return self.submitted              # _enter_submitted


def test_enter_that_only_inserted_a_newline_is_reported_as_a_failure():
    """Live 2026-09-02: Enter in X's composer inserts a newline and posts
    nothing. The key was pressed, so the tool said "Pressed 'Enter'." and the
    agent read that as the tweet going out."""
    page = _KeyPage(submitted=False)
    result = _session_on(page).press("enter")
    assert "Press failed" in result
    assert page.pressed == ["Enter"]


def test_enter_that_submitted_the_form_is_reported_as_pressed():
    assert _session_on(_KeyPage(submitted=True)).press("enter") == "Pressed 'Enter'."


def test_only_enter_is_judged_by_whether_the_field_cleared():
    """cmd+enter is the combination that posts, and a page that leaves the
    text in place after some other key has not necessarily failed at anything."""
    assert _session_on(_KeyPage(submitted=False)).press("cmd+enter") == (
        "Pressed 'Meta+Enter'.")


def test_a_page_that_will_not_evaluate_counts_as_submitted():
    """Best-effort, like the type check: a verification that cannot run must
    not turn a working press into a reported failure."""
    assert _session_on(_KeyPage(raises=True)).press("enter") == "Pressed 'Enter'."


def test_the_press_failure_reads_as_a_tool_error():
    from symbio.app import learn

    assert learn.sounds_like_tool_error(_session_on(_KeyPage(submitted=False)).press("enter"))


def test_type_and_submit_says_when_only_the_typing_happened():
    """browser_type with enter:true reaches the same Enter through the other
    tool, and claimed "and pressed Enter." whatever that did."""
    from symbio.app import learn

    result = _session_on(_KeyPage(landed=True, submitted=False)).type_text(
        "hello", press_enter=True)

    assert "submit failed" in result
    assert learn.sounds_like_tool_error(result)


def test_type_and_submit_that_worked_still_reads_as_success():
    result = _session_on(_KeyPage(landed=True, submitted=True)).type_text(
        "hello", press_enter=True)
    assert result == "Typed 'hello' and pressed Enter."


# ---- which "Post" gets clicked ----

class _Target:
    """One clickable element. `clickable=False` is an element that matches but
    cannot be clicked: covered by an overlay, or inert under an open modal."""

    def __init__(self, label, clicks, clickable=True):
        self.label, self.clicks, self.clickable = label, clicks, clickable

    def click(self, timeout=None):
        if not self.clickable:
            raise RuntimeError(f"'{self.label}' intercepts pointer events")
        self.clicks.append(self.label)


class _Matches:
    """A locator over `count` copies of one target, or over nothing."""

    def __init__(self, target=None, count=1):
        self.target = target
        self._count = 0 if target is None else count

    def count(self):
        return self._count

    def filter(self, visible=True):
        return self

    @property
    def first(self):
        return self.target


class _Scope:
    """A page, or a dialog on one. `table` maps (kind, exact) to the label of
    the element that matches there; kind is 'button', 'link' or 'text'."""

    def __init__(self, clicks, table, unclickable=(), modal=None, counts=None):
        self.clicks, self.table = clicks, table
        self.unclickable, self.modal = set(unclickable), modal
        self.counts = counts or {}

    def _match(self, key):
        label = self.table.get(key)
        if label is None:
            return _Matches()
        return _Matches(
            _Target(label, self.clicks, label not in self.unclickable),
            self.counts.get(key, 1))

    def get_by_role(self, role, name, exact):
        return self._match((role, exact))

    def get_by_text(self, text, exact):
        return self._match(("text", exact))

    def locator(self, selector):
        return _Matches(self.modal) if self.modal is not None else _Matches()


def _click(page, text="Post"):
    return _session_on(page)._try_click(page, text=text)


def test_the_open_dialog_wins_over_the_page_behind_it():
    """Live 2026-09-02: with the composer open, "Post" named both its submit
    button and the sidebar link that had opened it. The sidebar one comes
    first in the document and is inert under the modal, so the click did
    nothing — and the observation said an element had been clicked."""
    clicks: list[str] = []
    dialog = _Scope(clicks, {("button", True): "dialog Post"})
    page = _Scope(clicks, {("button", True): "sidebar Post"}, modal=dialog)

    result = _click(page)

    assert clicks == ["dialog Post"]
    assert "in the open dialog" in result


def test_without_a_dialog_the_page_is_the_only_scope():
    clicks: list[str] = []
    result = _click(_Scope(clicks, {("button", True): "Post"}))

    assert clicks == ["Post"]
    assert "in the open dialog" not in result


def test_a_control_is_preferred_over_prose_naming_it():
    clicks: list[str] = []
    _click(_Scope(clicks, {("button", True): "the submit button",
                           ("text", True): "a paragraph saying Post"}))

    assert clicks == ["the submit button"]


def test_an_exact_label_is_preferred_over_a_substring():
    clicks: list[str] = []
    _click(_Scope(clicks, {("button", True): "Post",
                           ("button", False): "Post to everyone"}))

    assert clicks == ["Post"]


def test_a_match_that_cannot_be_clicked_falls_through_to_the_next():
    """Before, the first visible match was clicked and the exception ended the
    whole attempt — so one covered element hid every candidate behind it."""
    clicks: list[str] = []
    _click(_Scope(clicks, {("button", True): "covered", ("link", True): "the real one"},
                 unclickable=["covered"]))

    assert clicks == ["the real one"]


def test_an_ambiguous_click_says_how_many_it_matched():
    result = _click(_Scope([], {("button", True): "Post"},
                           counts={("button", True): 3}))

    assert "first visible of 3 matches" in result


# ---- filling by selector is not a promise the text is there ----

class _FillPage:
    """A page with one field the selector reaches. `keeps` is whether the
    field still holds what was put in it a moment later — a React or Lexical
    composer that rebuilds its DOM from its own state does not."""

    def __init__(self, keeps=True, raises=False, present=True):
        self.keeps, self.raises, self.present = keeps, raises, present
        self.filled = []

    def locator(self, selector):
        page = self

        class _Match:
            def count(self): return 1 if page.present else 0
            def filter(self, visible=True): return self

            @property
            def first(self):
                class _El:
                    def fill(_s, text, timeout=None):
                        page.filled.append(text)
                return _El()

        return _Match()

    def evaluate(self, script, arg=None):
        if self.raises:
            raise RuntimeError("page closed")
        return self.keeps


def test_a_selector_fill_that_lands_is_a_success():
    page = _FillPage(keeps=True)

    assert _session_on(page).type_text("a post", selector="#composer") == (
        "Typed 'a post'.")
    assert page.filled == ["a post"]


def test_a_field_that_reverts_the_fill_is_not_reported_as_typed():
    """Live against a composer replica that rebuilds itself from its own
    state: fill() reports success, the field is empty a tick later, and the
    tool said "Typed 'gave my agent eyes today'." This is the same false
    success the focus check exists to remove, in the path the tool
    description tells the model to PREFER because filling by selector
    "cannot miss"."""
    page = _FillPage(keeps=False)

    result = _session_on(page).type_text("a post", selector="#composer")

    assert "Type failed" in result
    assert "put them back" in result


def test_the_selector_failure_reads_as_a_tool_error():
    from symbio.app import learn

    assert learn.sounds_like_tool_error(
        _session_on(_FillPage(keeps=False)).type_text("a post", selector="#c"))


def test_a_page_that_will_not_evaluate_counts_as_filled():
    """Best-effort, like every other check here: a verification that cannot
    run must never fail a type that worked. An XPath target lands here too —
    querySelector does not speak XPath."""
    page = _FillPage(raises=True)

    assert _session_on(page).type_text("a post", selector="//div[1]") == (
        "Typed 'a post'.")


# ---- did the click actually send what was sitting in the composer? ----

class _ComposerPage(_Scope):
    """A page with one composer in it. `holds_after` is what the field still
    contains once the click has been dispatched — "" for a composer that
    submitted and cleared."""

    def __init__(self, clicks, table, holds="a tweet", holds_after=None,
                 raises_after=False, url_after=None):
        super().__init__(clicks, table)
        self.holds = holds
        self.holds_after = holds if holds_after is None else holds_after
        self.raises_after = raises_after
        self.waited = 0
        self.pressed = []
        self.url = "https://x.com/home"
        self._url_after = url_after

    def wait_for_timeout(self, ms):
        self.waited += ms

    @property
    def keyboard(self):
        return self

    def press(self, key):
        self.pressed.append(key)

    def evaluate(self, script, arg=None):
        if arg is None:                      # _PENDING_FIELD_JS, before
            return self.holds
        if self.raises_after:                # page will not evaluate at all
            raise RuntimeError("execution context was destroyed")
        if self._url_after:                  # the click navigated
            self.url = self._url_after
        return arg in self.holds_after       # _FIELD_STILL_HOLDS_JS, after


def _clicked(page, text="Post"):
    return _session_on(page).click(text=text)


def test_a_click_that_emptied_the_composer_says_the_text_went():
    """Live 2026-09-07: the tweet posted, and the observation was "Clicked
    button 'Post'." — which says what the tool DID, never what it achieved.
    With nothing else to go on the agent read the page dump wrong and told the
    user the post "may not have been submitted successfully", offering to press
    cmd+enter: the recovery for a failed post, run against a successful one,
    posts it twice."""
    page = _ComposerPage([], {("button", True): "Post"}, holds_after="")

    result = _clicked(page)

    assert result.startswith("Clicked button 'Post'.")
    assert "is now empty" in result
    assert "sends it twice" in result


def test_a_click_that_left_the_composer_full_says_nothing_was_sent():
    page = _ComposerPage([], {("button", True): "Post"})

    result = _clicked(page)

    assert "still holds it" in result
    assert "wrong control" in result


def test_neither_verdict_reads_as_a_tool_error():
    """A click does not say what it was aiming at, so neither sentence may
    fail the call: "Bold" pressed over a full composer has failed at nothing,
    and the cleared case is an outright success. Both only add information —
    the same contract _no_effect_note keeps."""
    from symbio.app import learn

    for after in ("", None):
        page = _ComposerPage([], {("button", True): "Post"}, holds_after=after)
        result = _clicked(page)
        assert not learn.sounds_like_tool_error(result)
        assert "failed" not in result.lower() and "error" not in result.lower()


def test_an_empty_composer_is_not_reported_on_at_all():
    """Most clicks have no pending text behind them, and every one of those
    must read exactly as it did before."""
    page = _ComposerPage([], {("button", True): "Post"}, holds="")

    assert _clicked(page) == "Clicked button 'Post'."


def test_a_click_that_could_not_be_re_read_claims_nothing():
    """A click that navigates leaves no field to check. "The composer is gone,
    so it submitted" would invent the confirmation the model cannot get."""
    page = _ComposerPage([], {("button", True): "Post"}, raises_after=True)

    assert _clicked(page) == "Clicked button 'Post'."


def test_a_failed_click_is_not_given_a_verdict_about_the_composer():
    page = _ComposerPage([], {}, holds_after="")

    result = _clicked(page)

    assert "Click failed" in result
    assert "is now empty" not in result


def test_the_page_is_given_a_moment_to_handle_the_submit():
    """Read straight back and a React composer has not cleared yet, which
    reports every successful post as a failed one."""
    page = _ComposerPage([], {("button", True): "Post"}, holds_after="")

    _clicked(page)

    assert page.waited >= 300


def test_a_click_that_navigated_claims_neither_way():
    """An empty field on a NEW page proves nothing — the document that owned
    the draft went away and took it with it. Live against the replica, a stray
    link click over a filled composer read as "submitted ... do not retype it",
    which is the exact opposite of the recovery for a draft that was lost."""
    page = _ComposerPage([], {("link", True): "Go elsewhere"}, holds_after="",
                         url_after="https://x.com/elsewhere")

    result = _clicked(page, text="Go elsewhere")

    assert "navigated" in result
    assert "Read the new page" in result
    assert "sends it twice" not in result


def test_a_coordinate_click_gets_the_same_verdict():
    """Grounding from a screenshot is the other way the submit button gets
    hit, and it returned a bare "Clicked at (x, y)."."""
    page = _ComposerPage([], {}, holds_after="")
    page.viewport_size = {"width": 1200, "height": 800}
    page.mouse = type("M", (), {"click": lambda self, x, y: None})()

    result = _session_on(page).click_at(600, 400)

    assert result.startswith("Clicked at (600, 400).")
    assert "is now empty" in result


def test_cmd_enter_is_now_judged_too():
    """The combination that posts on X reached neither check: _enter_submitted
    is spelled for bare Enter, so "Pressed 'Meta+Enter'." came back whether or
    not anything went."""
    page = _ComposerPage([], {})

    result = _session_on(page).press("cmd+enter")

    assert result.startswith("Pressed 'Meta+Enter'.")
    assert "still holds it" in result


def test_an_ordinary_keystroke_is_left_alone():
    """Every non-Enter key legitimately leaves a composer holding its text."""
    page = _ComposerPage([], {})

    assert _session_on(page).press("down") == "Pressed 'ArrowDown'."


# ---- typing at an unfocused page is not inert ----

def test_no_keystrokes_are_sent_when_nothing_is_focused():
    """Live 2026-09-06: asked to post to @grok, the agent typed at a page with
    nothing focused. The old order typed FIRST and checked afterwards, and told
    the model the keystrokes "went to the page and were discarded" — which is
    false on every site that binds bare letters. On X each character fired the
    shortcut it is bound to; one of them opened the composer partway through
    the sentence, so the tail of the message landed in a composer the tool had
    opened by accident, and the next click submitted the fragment.

    The check therefore has to happen BEFORE the keys are sent, and the proof
    is that the keyboard was never touched."""
    page = _KeyPage(focused=False)
    result = _session_on(page).type_text("@grok you think a human posted this?")

    assert page.typed == []
    assert "Type failed" in result
    assert "no keys were sent" in result


def test_the_unfocused_failure_does_not_claim_the_keys_vanished():
    """The old message asserted something the tool cannot know and that was
    observably untrue. Whatever this says, it must not say that."""
    result = _session_on(_KeyPage(focused=False)).type_text("hello")
    assert "discarded" not in result.lower()


def test_a_focused_field_that_rejects_the_text_says_it_may_hold_part_of_it():
    """Focused and editable is not the same as landed: the field may have
    reformatted or truncated the text. Retyping on top of a half-filled field
    is how a fragment becomes a post, so the message has to say so."""
    result = _session_on(_KeyPage(landed=False, focused=True)).type_text("hello")
    assert "Type failed" in result
    assert "PART" in result


def test_the_unfocused_failure_still_reads_as_a_tool_error():
    from symbio.app import learn

    assert learn.sounds_like_tool_error(
        _session_on(_KeyPage(focused=False)).type_text("hello"))


# ---- a screen you cannot capture is not a screen with nothing on it ----

def test_a_black_capture_is_recognised_as_a_missing_permission(tmp_path):
    """Observed 2026-09-07: a full desktop capture came back 1920x1080 with
    every pixel at zero, because macOS returns a blank frame — not an error —
    when the process was never granted Screen Recording permission. The vision
    model then described it accurately ("the image is entirely black with no
    text, buttons or links"), which reads as a fact about the screen rather
    than a missing permission, and the model would act on it."""
    from PIL import Image
    from symbio import computer

    black = tmp_path / "black.png"
    Image.new("RGB", (400, 300), (0, 0, 0)).save(black)
    assert computer.screenshot_is_blank(black) is True


def test_a_real_screen_is_not_mistaken_for_a_failed_capture(tmp_path):
    from PIL import Image, ImageDraw
    from symbio import computer

    page = tmp_path / "page.png"
    img = Image.new("RGB", (400, 300), (255, 255, 255))
    ImageDraw.Draw(img).rectangle([20, 20, 200, 60], fill=(30, 120, 220))
    img.save(page)
    assert computer.screenshot_is_blank(page) is False


def test_a_genuinely_dark_screen_with_content_still_counts_as_a_screen(tmp_path):
    """Dark mode is not a failed capture. The test is featurelessness, not
    darkness — a black frame has zero variance, a dark UI does not."""
    from PIL import Image, ImageDraw
    from symbio import computer

    dark = tmp_path / "dark.png"
    img = Image.new("RGB", (400, 300), (10, 10, 12))
    ImageDraw.Draw(img).rectangle([20, 20, 380, 120], fill=(90, 90, 110))
    img.save(dark)
    assert computer.screenshot_is_blank(dark) is False


def test_an_unreadable_file_is_left_to_the_normal_error_path(tmp_path):
    from symbio import computer

    missing = tmp_path / "nope.png"
    assert computer.screenshot_is_blank(missing) is False


def test_browser_type_passes_a_selector_through_to_the_browser():
    """BrowserSession has always supported filling by selector and the chat
    dispatch never passed one, so the reliable path was unreachable from a
    tool call. It matters for controls too small to ground: X's composer is
    28px tall, under one 32px vision patch, so a coordinate either misses it
    by ~36px or is not returned at all, while the selector fills it first
    time."""
    from symbio.app import chat_tools
    from symbio.app.config import DEFAULT_CONFIG

    seen = {}

    class _Browser:
        is_open = True

        def type_text(self, text, selector="", press_enter=False):
            seen.update(text=text, selector=selector, enter=press_enter)
            return f"Typed '{text}'."

        def get_text(self):
            return "page"

    class S(chat_tools.ToolsMixin):
        config = {**DEFAULT_CONFIG, "browser": {"enabled": True}}
        browser = _Browser()
        _last_browsed_url = ""

        def _status(self, m):
            pass

    S()._dispatch_tool("browser_type", {
        "text": "hello", "selector": '[data-testid="tweetTextarea_0"]'})

    assert seen["selector"] == '[data-testid="tweetTextarea_0"]'
    assert seen["text"] == "hello"


def test_browser_type_without_a_selector_still_types_into_the_focused_field():
    from symbio.app import chat_tools
    from symbio.app.config import DEFAULT_CONFIG

    seen = {}

    class _Browser:
        is_open = True

        def type_text(self, text, selector="", press_enter=False):
            seen.update(selector=selector)
            return f"Typed '{text}'."

        def get_text(self):
            return "page"

    class S(chat_tools.ToolsMixin):
        config = {**DEFAULT_CONFIG, "browser": {"enabled": True}}
        browser = _Browser()
        _last_browsed_url = ""

        def _status(self, m):
            pass

    S()._dispatch_tool("browser_type", {"text": "hello"})
    assert seen["selector"] == ""


# ---- a look must not assert absence while holding proof of presence ----

def _look_session(controls, elements, description="a page"):
    from symbio.app import chat_tools
    from symbio.app.config import DEFAULT_CONFIG

    class _Browser:
        is_open = True

        def controls(self, limit=25):
            return controls

        def controls_read(self, limit=25):
            # (controls, the DOM was read). The stub's page always reads;
            # the read-failed case has its own test below.
            return controls, True

        def screenshot_path(self, full_page=False):
            from pathlib import Path
            return Path("shot.png")

    class S(chat_tools.ToolsMixin):
        config = {**DEFAULT_CONFIG, "browser": {"enabled": True}, "dispatch": {}}
        browser = _Browser()

        def _status(self, m):
            pass

        def _run_vision(self, shot, question):
            return description, elements

    return S()


_COMPOSER = {"kind": "field", "selector": '[data-testid="tweetTextarea_0"]',
             "label": "Post text", "value": "", "disabled": False,
             "x": 600, "y": 90}


def test_a_vision_miss_does_not_claim_the_control_is_absent():
    """Live 2026-09-07 on a 28px composer, one observation carried all three
    of: prose saying "no visible text area for composing a post", a controls
    list containing the composer, and a line telling the model to treat it as
    absent. Two of the three said "not there", so it clicked around hunting
    for a composer it had just been handed the handle for."""
    out = _look_session([_COMPOSER], [])._see_screen({"question": "where do I type?"})

    assert "Treat it as NOT present" not in out
    assert '[data-testid="tweetTextarea_0"]' in out
    assert "NOT proof the thing is missing" in out


def test_absence_is_still_reported_when_nothing_contradicts_it():
    """With no controls either, "not there" is the honest answer and has to be
    delivered as one — otherwise an empty result reads as a failed look and
    the model goes back to guessing."""
    out = _look_session([], [])._see_screen({"question": "where is the Slack window?"})

    assert "Treat it as NOT present" in out


def test_a_successful_grounding_adds_no_disclaimer():
    located = [{"label": "Post button", "box": (0, 0, 10, 10), "center": (5, 5)}]
    out = _look_session([_COMPOSER], located)._see_screen({"question": "where?"})

    assert "NOT proof the thing is missing" not in out
    assert "Treat it as NOT present" not in out
    assert "(5, 5)" in out


def test_fields_are_offered_with_their_current_contents():
    """Whether the box already holds text decides between typing and clearing
    first, and it is the thing page text cannot show for a textarea."""
    filled = {**_COMPOSER, "value": "half a message"}
    out = _look_session([filled], [])._see_screen({})

    assert "currently: 'half a message'" in out


def test_an_enter_that_did_not_submit_restores_the_typed_text():
    """Enter that does not submit still inserts a character, and the standard
    recovery is to click the submit button next — so the newline ships with
    the message. Observed 2026-09-07: a successful type-then-click posted the
    message with two trailing newlines.

    Restored, not backspaced: a bare Backspace is a speculative keystroke in
    the one function that just grew a focus check because keystrokes are not
    inert, and one Backspace is the wrong shape for a contenteditable, where
    Enter can insert a whole element."""
    page = _KeyPage(landed=True, submitted=False, focused=True)
    result = _session_on(page).type_text("hello", press_enter=True)

    assert page.pressed == ["Enter"], "no speculative extra keystroke"
    assert page.restored == ["hello"]
    assert "submit failed" in result


def test_the_undo_is_skipped_when_focus_moved_after_enter():
    """If Enter navigated or opened a dialog, the field that would be edited is
    not the one that was typed into."""
    page = _KeyPage(landed=True, submitted=False, focused=False)
    _session_on(page).type_text("hello", press_enter=True)

    assert page.restored == []



def test_an_enter_that_submitted_leaves_the_field_alone():
    page = _KeyPage(landed=True, submitted=True)
    _session_on(page).type_text("hello", press_enter=True)

    assert page.pressed == ["Enter"], "nothing to undo when the form went"


# ---- a form submission gets a machine verdict, never a click's word ----

class _SubmitPage(_Scope):
    """A form page whose URL transition models the POST→redirect settle: it
    lands during wait_for_timeout, the way submit_form reads page.url after
    its one-second wait. `holds` is what _PENDING_FIELD_JS answers ("" when
    two fields hold content, this being the HN prefill path); `holds_after`
    is what _FIELD_STILL_HOLDS_JS still finds after the click."""

    def __init__(self, clicks, table, holds="", holds_after="",
                 url_before="https://news.ycombinator.com/submitlink?u=&t=",
                 url_after="https://news.ycombinator.com/item?id=1001",
                 raises_after=False):
        super().__init__(clicks, table)
        self.holds = holds
        self.holds_after = holds_after
        self.raises_after = raises_after
        self.waited = 0
        self.url = url_before
        self._url_after = url_after

    def wait_for_timeout(self, ms):
        self.waited += ms
        if self._url_after is not None:
            self.url = self._url_after

    def evaluate(self, script, arg=None):
        # Only the AFTER read (FIELD_STILL_HOLDS, arg not None) can fail:
        # _pending_state's BEFORE read wraps its own evaluate and would
        # swallow the raise into a blank field, skipping the check entirely.
        if arg is not None and self.raises_after:
            raise RuntimeError("execution context was destroyed")
        if arg is None:                 # _PENDING_FIELD_JS, before
            return self.holds
        return arg in self.holds_after  # _FIELD_STILL_HOLDS_JS, after


_SUBMIT = {("button", True): "submit"}


def _submitted(page, **kw):
    return _session_on(page).submit_form(target="submit", **kw)


def test_a_live_hn_item_url_confirms_the_submission():
    """The URL that only exists once a story is live is the whole verdict:
    with the prefill path two fields hold content, so _pending_state answers
    "" and the page.url after the click is all the code has to go on."""
    page = _SubmitPage([], _SUBMIT)

    result = _submitted(page)

    assert result.startswith("[Submit CONFIRMED:")
    assert "https://news.ycombinator.com/item?id=1001" in result
    assert "you may report the submission as done" in result


def test_the_submission_is_given_a_second_to_settle():
    page = _SubmitPage([], _SUBMIT)
    _submitted(page)
    assert page.waited >= 1000


def test_a_non_item_page_is_never_confirmed():
    """A cleared form is also a discarded draft; only the live-story URL
    proves the post went through. Landing on a login page is exactly the case
    a logged-out profile produces, and must read as not-confirmed."""
    page = _SubmitPage([], _SUBMIT,
                       url_after="https://news.ycombinator.com/login")
    result = _submitted(page)
    assert result.startswith("[Submit NOT confirmed:")
    assert "https://news.ycombinator.com/login" in result
    assert "do NOT report it as posted" in result


def test_an_unchanged_url_is_evidence_the_form_did_not_go():
    page = _SubmitPage([], _SUBMIT, url_after=None)
    result = _submitted(page)
    assert result.startswith("[Submit NOT confirmed:")
    assert "the URL did not change" in result


def test_a_prefix_expected_url_confirms_on_a_genuine_landing():
    """expected_url confirms on the prefix when the URL actually moved; the
    two signals are read independently, so prove the prefix branch fires on
    a landing it was given."""
    page = _SubmitPage([], _SUBMIT, url_before="https://example.com/form",
                       url_after="https://example.com/form/result?ok=1")
    result = _submitted(page, expected_url="https://example.com/form/result")
    assert result.startswith("[Submit CONFIRMED:")
    assert "starts with the expected" in result


def test_a_failed_click_orders_us_not_to_report_the_submission():
    """The click is the one thing that had to happen; without it the form is
    untouched, so any post-claim after this turn is a fabrication."""
    page = _SubmitPage([], {}, url_after="https://news.ycombinator.com/item?id=9")
    result = _submitted(page)
    assert "Click failed" in result
    assert "Do NOT report this form as submitted" in result


def test_a_page_that_cannot_be_read_back_refuses_a_verdict():
    """A page that will not evaluate after the click leaves the field-state
    check unanswerable — that is 'verification could not run', never 'NOT
    confirmed' (which asserts the form did not go) and never silence."""
    page = _SubmitPage([], _SUBMIT, holds="a story", raises_after=True)
    result = _submitted(page)
    assert result.startswith("[Submit verification could not run")
    assert "Do NOT report the form as submitted" in result


# ---- a page that has loaded is not a page that has rendered ----

class _SpaPage:
    """A client-side app: the document is ready, the body is empty, and the
    text arrives a few polls later. Live on 2026-09-16, claude.ai/pricing
    answered domcontentloaded with a title and zero characters of text."""

    def __init__(self, texts, title="Claude"):
        self.texts = list(texts)
        self._title = title
        self.waited = 0

    def goto(self, url, **kw):
        return None

    def title(self):
        return self._title

    def evaluate(self, script, arg=None):
        return len(self.texts[0].strip()) if self.texts else 0

    def wait_for_timeout(self, ms):
        self.waited += ms
        if len(self.texts) > 1:
            self.texts.pop(0)


def _opened(page, url="https://example.com/"):
    session = BrowserSession()
    session._init = lambda channel="": (None, page)
    session._page = page
    return session.open(url)


def test_an_empty_body_is_waited_out_not_reported_as_the_page():
    page = _SpaPage(["", "", "Pro $17 per month with annual billing"])
    result = _opened(page)
    assert "rendered no text yet" not in result, result
    assert page.waited > 0, "it did not wait for the app to paint"


def test_a_page_that_never_renders_says_so_instead_of_staying_silent():
    """The failure mode this replaces is worse than a slow answer: an empty
    read is handed over as if it were the page, and what fills the gap is the
    model's own guess about what a pricing page says."""
    page = _SpaPage([""])
    result = _opened(page)
    assert "rendered no text yet" in result
    assert "do NOT describe a page you have not read" in result


def test_a_short_page_is_not_mistaken_for_an_empty_one():
    """A local form page is eight characters of heading and entirely real."""
    page = _SpaPage(["The wall"])
    result = _opened(page)
    assert "rendered no text yet" not in result


# ---- posting on any site, with no per-site tool ----

class _PublishPage(_Scope):
    """A page where the submit control publishes: after the click the words
    move OUT of the composer and INTO a rendered element.

    `published` is what _TEXT_RENDERED_JS answers — the text as PUBLISHED
    content, outside every field. `holds_after` is what is still sitting in a
    field. Keeping them apart is the point of the test: a composer holding
    the words is a draft, and reading the page body would call it a post.
    """

    def __init__(self, clicks, table, holds="", holds_after="",
                 published="", url="https://x.com/home"):
        super().__init__(clicks, table)
        self.holds = holds
        self.holds_after = holds_after
        self.published = published
        self.url = url
        self.waited = 0

    def wait_for_timeout(self, ms):
        self.waited += ms

    def evaluate(self, script, arg=None):
        if arg is None:                        # _PENDING_FIELD_JS, before
            return self.holds
        if "inEditable" in script:             # _TEXT_RENDERED_JS
            return bool(self.published) and arg in self.published
        return arg in self.holds_after         # _FIELD_STILL_HOLDS_JS


_POST = {("button", True): "Post"}


def test_text_rendered_outside_a_field_confirms_any_submission():
    """The site-independent proof, and the reason posting needs no tool of
    its own: the words came back as published content, on a URL that never
    changed and a site this code knows nothing about."""
    page = _PublishPage([], _POST, published="shipping the desktop window today")

    result = _session_on(page).submit_form(
        target="Post", expect_text="shipping the desktop window today")

    assert result.startswith("[Submit CONFIRMED:")
    assert "outside every input field" in result
    assert "you may report the submission as done" in result


def test_a_draft_still_in_the_composer_is_never_a_post():
    """The failure this exists to stop. The words ARE on the page — in the
    box the model typed them into. Reading the body text would confirm every
    unsent draft as published."""
    page = _PublishPage([], _POST, holds="a draft nobody sent",
                        holds_after="a draft nobody sent", published="")

    result = _session_on(page).submit_form(
        target="Post", expect_text="a draft nobody sent")

    assert result.startswith("[Submit NOT confirmed:")
    assert "not rendered on the page as published content" in result
    assert "do NOT report it as posted" in result


def test_the_expected_text_is_polled_not_read_once():
    """A feed re-renders after the network call, not during it. One read at
    the moment of the click is the wrong read."""
    page = _PublishPage([], _POST, published="")

    def land_later(ms):
        page.waited += ms
        if page.waited >= 1400:
            page.published = "late but live"

    page.wait_for_timeout = land_later
    result = _session_on(page).submit_form(target="Post",
                                           expect_text="late but live")

    assert result.startswith("[Submit CONFIRMED:")


def test_a_hopeless_wait_still_ends():
    page = _PublishPage([], _POST, published="")
    result = _session_on(page).submit_form(target="Post", expect_text="nope",
                                           timeout_ms=1000)
    assert result.startswith("[Submit NOT confirmed:")


# ---- filling a whole form in one call ----

class _FormPage:
    """Fields addressed by selector, with a read-back that only answers for
    what was actually written."""

    def __init__(self, refuse=()):
        self.values: dict[str, str] = {}
        self.refuse = set(refuse)

    def fill(self, selector, text):
        if selector in self.refuse:
            return
        self.values[selector] = text

    def query_selector(self, selector):
        return object() if selector not in self.refuse else None

    def evaluate(self, script, arg=None):
        # _SELECTOR_HOLDS_JS is given (selector, text).
        if isinstance(arg, (list, tuple)) and len(arg) == 2:
            selector, text = arg
            return text in self.values.get(selector, "")
        return False


def _filled(page, fields):
    session = BrowserSession()
    session._page = page
    session.type_text = lambda text, selector="", press_enter=False: (
        page.fill(selector, text) or "typed")
    return session.fill_form(fields)


def test_a_form_is_filled_and_every_field_read_back():
    page = _FormPage()
    result = _filled(page, {"#title": "My story", "#url": "https://example.com"})

    assert page.values == {"#title": "My story", "#url": "https://example.com"}
    assert "Filled 2/2" in result
    assert "read back and holds its value" in result


def test_a_field_that_did_not_take_its_value_is_named():
    """Half a form submitted is a wrong post, not a partial one."""
    page = _FormPage(refuse={"#url"})
    result = _filled(page, {"#title": "My story", "#url": "https://example.com"})

    assert "NOT filled" in result and "#url" in result
    assert "Do not submit this form yet" in result


def test_fill_form_accepts_the_list_shape_a_model_emits():
    page = _FormPage()
    result = _filled(page, [{"selector": "#title", "text": "one"},
                            {"selector": "#url", "value": "two"}])
    assert page.values == {"#title": "one", "#url": "two"}
    assert "Filled 2/2" in result


def test_fill_form_with_no_fields_says_what_to_pass():
    result = BrowserSession().fill_form({})
    assert "Fill failed" in result and "see_screen" in result


def test_submit_before_open_does_not_raise():
    session = BrowserSession()
    result = session.submit_form(target="submit")
    assert isinstance(result, str)
    assert result.startswith("Submit error")


# ---- the allowlist seeds which domains can skip the prompt ----

def test_allowed_domains_skip_the_domain_confirm():
    session = BrowserSession(allowed_domains=["news.ycombinator.com"])
    ok, _ = session._check_url("https://news.ycombinator.com/story")
    assert ok is True


def test_an_unseeded_domain_consults_the_confirm_fn():
    confirmed = []
    session = BrowserSession(confirm_fn=lambda p: confirmed.append(p) or True)
    ok, _ = session._check_url("https://example.org/x")
    assert ok is True
    assert confirmed == ["Allow browser to access 'example.org'?"]
    # A deny is final: the domain never enters _confirmed.
    said_no = BrowserSession(confirm_fn=lambda p: False)
    ok, msg = said_no._check_url("https://denied.example/x")
    assert ok is False
    assert "denied" in msg
