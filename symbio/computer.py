"""Browser and desktop automation helpers for Symbio.

Uses Playwright for browser control and PyAutoGUI for desktop mouse/keyboard.
Screenshots are saved to screenshots/ so the user can view them, and — since
the headmaster is text-only — symbio/vision.py reads them back through a
vision model, so a screenshot now returns what is on screen and where, rather
than a filename. That is why the capture helpers here come in pairs: the
`screenshot`/`desktop_screenshot` functions report a saved path to the user,
while `screenshot_path`/`desktop_screenshot_path` hand the file to something
that is going to look at it.
"""

from __future__ import annotations

import atexit
import json
import re
import sys
import time
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Any

from symbio.constants import SCREENSHOTS_DIR

SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# Domains that do not require per-session confirmation.
_DEFAULT_ALLOWLIST: frozenset[str] = frozenset({
    "localhost",
    "127.0.0.1",
    "example.com",
})


def _domain(url: str) -> str | None:
    try:
        parsed = urllib.parse.urlparse(url)
        if not parsed.netloc:
            return None
        host = parsed.netloc.split(":")[0].lower()
        return host
    except Exception:
        return None


def _scheme_ok(url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(url)
        return parsed.scheme in ("http", "https")
    except Exception:
        return False


def _short_error(e: Exception) -> str:
    """First line of an exception message; Playwright appends huge call logs."""
    lines = str(e).strip().splitlines()
    return lines[0] if lines else e.__class__.__name__


# Loose key names small models emit, mapped to Playwright's canonical names —
# same idea as the Hermes tool-call normalization in parse_tools.
_KEY_ALIASES = {
    "down": "ArrowDown", "up": "ArrowUp", "left": "ArrowLeft", "right": "ArrowRight",
    "arrowdown": "ArrowDown", "arrowup": "ArrowUp", "arrowleft": "ArrowLeft", "arrowright": "ArrowRight",
    "enter": "Enter", "return": "Enter",
    "esc": "Escape", "escape": "Escape",
    "space": "Space", "spacebar": "Space",
    "tab": "Tab", "backspace": "Backspace",
    "delete": "Delete", "del": "Delete",
    "home": "Home", "end": "End",
    "pageup": "PageUp", "page up": "PageUp", "pgup": "PageUp",
    "pagedown": "PageDown", "page down": "PageDown", "pgdn": "PageDown",
    "shift": "Shift", "ctrl": "Control", "control": "Control",
    "alt": "Alt", "option": "Alt",
    "cmd": "Meta", "command": "Meta", "meta": "Meta", "win": "Meta",
}


def _normalize_key(key: str) -> str:
    """Map a loosely-named key ('down', 'page down', 'ctrl+a') to Playwright's names."""
    key = key.strip()
    if "+" in key:
        parts = [p.strip() for p in key.split("+") if p.strip()]
        return "+".join(_normalize_key(p) for p in parts)
    low = key.lower()
    if low in _KEY_ALIASES:
        return _KEY_ALIASES[low]
    if len(key) == 1:
        return key
    if re.fullmatch(r"f\d{1,2}", low):
        return low.upper()
    return key[:1].upper() + key[1:]


def _first_visible(locator: Any) -> tuple[Any | None, int]:
    """Return (first visible element, total match count) for a locator."""
    count = locator.count()
    visible = locator.filter(visible=True)
    if visible.count() > 0:
        return visible.first, count
    return None, count


def _confirm_domain(domain: str, ask_fn=None) -> bool:
    """Prompt the user before opening a new domain.
    `ask_fn(prompt) -> bool` may be supplied by non-terminal front-ends."""
    prompt = f"Allow browser to access '{domain}'?"
    if ask_fn is not None:
        try:
            return ask_fn(prompt)
        except Exception:
            return False
    print(f"  [Computer] {prompt} [y/N]:", end=" ", flush=True)
    try:
        answer = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"
    return answer in ("y", "yes")


class BrowserSession:
    """Manages a single Playwright browser/page session."""

    def __init__(self, confirm_fn=None, profile_dir=None, chrome_profile=None):
        self._playwright: Any | None = None
        self._browser: Any | None = None
        self._page: Any | None = None
        self._confirmed: set[str] = set(_DEFAULT_ALLOWLIST)
        self._confirm_fn = confirm_fn
        self._channel: str = ""
        self._last_url: str = ""
        # None (the default) launches a fresh, logged-out browser every time,
        # which is what every caller got before this existed. A path here keeps
        # cookies between sessions, so a site the user logs into once by hand
        # stays logged in for later turns — the only way the agent can act on
        # an account rather than just read public pages.
        #
        # Opt-in on purpose: a profile is standing access to whatever it holds,
        # for every future turn, and that is a bigger change than it looks.
        # The per-action confirmations still fire either way.
        self._profile_dir = profile_dir
        # Which profile inside that directory. Playwright has no option for
        # this and always opens "Default" — the identity signed into
        # everything. Passing --profile-directory is the only way to aim at a
        # dedicated one, which is what keeps the agent to the accounts it was
        # deliberately signed into.
        self._chrome_profile = chrome_profile

    @property
    def is_open(self) -> bool:
        """True if a browser page is currently open and ready for actions."""
        return self._page is not None

    def status(self) -> str:
        """Human-readable browser state for the model's context block."""
        if self._page is None:
            return "No browser page is currently open."
        try:
            title = self._page.title()
        except Exception:
            title = "unknown"
        # Spell the tags out rather than naming the tools. Naming them is what
        # taught the model to invent <browser_open>; and this note exists
        # precisely for the moment it has forgotten how to act on the page.
        return (
            f"Browser is open at {self._last_url} "
            f"(page title: \"{title}\"). "
            f"Act on THIS page directly — do not reopen it and do not use a "
            f"shell command. Emit exactly one of: "
            f"<click>visible text</click>, <type>words</type>, "
            f"<press>Enter</press>, <scroll />, <browser_close />."
        )

    def _init(self, channel: str = "") -> tuple[Any, Any]:
        if self._page is not None:
            return self._browser, self._page

        from playwright.sync_api import sync_playwright

        if self._playwright is None:
            self._playwright = sync_playwright().start()
            self._register_exit_cleanup()
        # Prefer Google Chrome when available; fall back to bundled Chromium.
        # A specific channel request overrides the stored default.
        preferred = channel or self._channel or "chrome"
        view = {"viewport": {"width": 1280, "height": 800},
                "accept_downloads": False}

        # Playwright launches Chrome advertising that it is automated:
        # --enable-automation, plus navigator.webdriver = true. Google's sign-in
        # refuses any browser carrying those ("this browser or app may not be
        # secure"), so the user cannot complete the one manual login the
        # persistent profile exists to capture. Dropping the flag is the
        # documented way to run a real, user-driven browser session; the agent
        # is not thereby hidden from anything -- every action it takes still
        # goes through the same domain confirmations.
        stealth = {
            "ignore_default_args": ["--enable-automation"],
            "args": ["--disable-blink-features=AutomationControlled"],
        }
        if self._chrome_profile:
            stealth["args"] = stealth["args"] + [
                f"--profile-directory={self._chrome_profile}"]

        if self._profile_dir is not None:
            # launch_persistent_context returns a CONTEXT, not a Browser. It
            # has .close() -- the only thing this class calls on _browser
            # besides new_context -- so it can be stored in the same slot, and
            # the context IS the browser for teardown purposes.
            from pathlib import Path

            Path(self._profile_dir).mkdir(parents=True, exist_ok=True)
            launch = self._playwright.chromium.launch_persistent_context
            try:
                context = launch(str(self._profile_dir), headless=False,
                                 channel=preferred, **stealth, **view)
                self._channel = preferred
            except Exception:
                context = launch(str(self._profile_dir), headless=False,
                                 **stealth, **view)
                self._channel = ""
            self._browser = context
            # A persistent context opens with a page already; taking it rather
            # than adding a second one keeps the window the user sees the one
            # being driven.
            self._page = context.pages[0] if context.pages else context.new_page()
            return self._browser, self._page

        try:
            self._browser = self._playwright.chromium.launch(
                headless=False, channel=preferred
            )
            self._channel = preferred
        except Exception:
            # Chrome not installed or channel unknown — use bundled Chromium.
            self._browser = self._playwright.chromium.launch(headless=False)
            self._channel = ""
        context = self._browser.new_context(**view)
        self._page = context.new_page()
        return self._browser, self._page

    def _ensure_open(self) -> Any:
        if self._page is None:
            # Deliberately names no specific call syntax: this layer has no
            # opinion on tags, and naming the tool ("use browser_open") taught
            # the model to invent a <browser_open> tag. The app layer appends
            # the exact tag to emit.
            raise RuntimeError(
                "Browser is not open. Load the target URL first, then retry the action."
            )
        return self._page

    # Errors that mean the Playwright connection itself is wedged (dead
    # greenlet/thread, closed pipe or loop) — no further call can succeed.
    # These are matched against the *lowercased* exception message.
    # NOTE: "has been closed" is deliberately NOT here — it matches benign
    # errors like "Target page has been closed" where only the page died.
    # Instead we handle page/browser-level closures separately in _fail.
    _FATAL_MARKERS = (
        "cannot switch to a different thread",
        "event loop is closed",
        "pipe closed",
    )

    def _reset(self):
        """Tear down a broken *browser* so the next browser_open starts clean.

        Only closes the browser (not the Playwright driver) — restarting
        Playwright is slow and almost never necessary. The driver is only
        stopped at process exit by _stop_playwright.
        """
        if self._browser:
            try:
                self._browser.close()
            except Exception:
                pass
        self._browser = None
        self._page = None

    def _fail(self, op: str, e: Exception) -> str:
        msg = str(e).lower()
        # Truly fatal (thread/loop/pipe dead): full teardown.
        if any(marker in msg for marker in self._FATAL_MARKERS):
            self._reset()
            if self._playwright:
                try:
                    self._playwright.stop()
                except Exception:
                    pass
                self._playwright = None
            return (
                f"Browser {op} error: the browser session broke and was reset. "
                "Use browser_open to reopen the page."
            )
        # Page or browser was closed (user closed window, page crashed, etc.):
        # tear down the browser so the next browser_open starts a fresh one,
        # but keep the Playwright driver alive for a fast restart.
        if "has been closed" in msg or "connection closed" in msg:
            self._reset()
            last_url_hint = f" The last opened URL was {self._last_url}." if self._last_url else ""
            return (
                f"Browser {op} error: the browser was closed.{last_url_hint} "
                "Use browser_open to reopen the page."
            )
        return f"Browser {op} error: {_short_error(e)}"

    def _check_url(self, url: str) -> tuple[bool, str]:
        if not url:
            return False, "URL is empty."
        if not _scheme_ok(url):
            return False, f"Only http/https URLs are allowed. Got: {url}"
        domain = _domain(url)
        if not domain:
            return False, f"Could not extract domain from URL: {url}"
        if domain not in self._confirmed:
            if _confirm_domain(domain, ask_fn=self._confirm_fn):
                self._confirmed.add(domain)
            else:
                return False, f"User denied access to '{domain}'."
        return True, ""

    def open(self, url: str, channel: str = "") -> str:
        ok, msg = self._check_url(url)
        if not ok:
            return f"Browser open blocked: {msg}"
        try:
            _, page = self._init(channel=channel)
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            title = page.title()
            self._last_url = url
            return f"Opened browser at {url}. Page title: {title}"
        except Exception as e:
            return self._fail("open", e)

    def navigate(self, url: str) -> str:
        return self.open(url)

    def get_text(self) -> str:
        try:
            page = self._ensure_open()
            text = page.inner_text("body", timeout=10000)
            # Collapse whitespace.
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r"[ \t]+", " ", text)
            return _truncated(text.strip(), 4000)
        except Exception as e:
            return self._fail("get_text", e)

    def get_html(self) -> str:
        try:
            page = self._ensure_open()
            html = page.content()
            return _truncated(html.strip(), 4000)
        except Exception as e:
            return self._fail("get_html", e)

    _TIMEOUT_MS = 4000

    def click(self, selector: str = "", text: str = "") -> str:
        try:
            page = self._ensure_open()
            pending = self._pending_state(page)
            result = self._try_click(page, selector, text)
            if pending[0] and result.startswith("Clicked"):
                result += self._submit_note(page, pending)
            return result
        except Exception as e:
            return self._fail("click", e)

    # A modal dialog makes everything it covers inert: while X's composer is
    # open the "Post" in the sidebar cannot be clicked at all. It is still the
    # first thing a page-wide search for that word finds, though, which is how
    # a click aimed at the submit button landed on the navigation item that
    # had opened the composer in the first place.
    _MODAL_SELECTOR = '[role="dialog"][aria-modal="true"], dialog[open]'

    def _click_scopes(self, page) -> list[tuple[Any, str]]:
        """Where to look for a click target, most likely first."""
        try:
            modal, _ = _first_visible(page.locator(self._MODAL_SELECTOR))
        except Exception:
            modal = None
        if modal is None:
            return [(page, "")]
        return [(modal, " in the open dialog"), (page, "")]

    @staticmethod
    def _click_candidates(scope: Any, text: str):
        """(locator, description) pairs for `text`, best match first.

        Controls before prose, exact labels before substrings. One word names
        both the button that submits a form and the link that opened it, and
        only one of the two does the job; a bare text search finds whichever
        happens to come first in the document.
        """
        for exact in (True, False):
            for role in ("button", "link"):
                yield scope.get_by_role(role, name=text, exact=exact), f"{role} '{text}'"
            yield (
                scope.get_by_text(text, exact=exact),
                f"element containing text '{text}'",
            )

    def _try_click(self, page, selector: str = "", text: str = "", attempt: int = 1) -> str:
        """Single click attempt. On the first failure due to a timeout or
        missing visible element, wait a moment and try once more — small
        models often issue a click before the page has fully settled."""
        if selector:
            # Generic selectors often match dozens of elements, many
            # hidden; click the first *visible* match instead of the
            # first match (which times out on hidden elements).
            target, count = _first_visible(page.locator(selector))
            if target is not None:
                target.click(timeout=self._TIMEOUT_MS)
                which = f" (first visible of {count} matches)" if count > 1 else ""
                return f"Clicked element matching '{selector}'{which}."
            if not text:
                if count == 0:
                    return f"Click failed: nothing matches selector '{selector}'. Try clicking by visible text instead."
                return (
                    f"Click failed: '{selector}' matched {count} element(s) but none are visible. "
                    "Use a more specific selector or click by visible text."
                )
        if text:
            for scope, where in self._click_scopes(page):
                for locator, described in self._click_candidates(scope, text):
                    target, count = _first_visible(locator)
                    if target is None:
                        continue
                    try:
                        target.click(timeout=self._TIMEOUT_MS)
                    except Exception:
                        # Matched but unclickable — covered by an overlay,
                        # inert under a modal, detached mid-click. Another
                        # candidate may still be the one that works.
                        continue
                    ambiguous = f" (first visible of {count} matches)" if count > 1 else ""
                    return f"Clicked {described}{where}{ambiguous}."
            if attempt == 1:
                # Page may still be settling; one automatic retry.
                time.sleep(0.5)
                return self._try_click(page, selector, text, attempt=2)
            return (
                f"Click failed: no visible element with text '{text}'. "
                "Look at the screen to find where the element actually is and "
                "click its coordinates; page text alone cannot tell you where "
                "something is, or which of two controls with the same label is "
                "the one you want."
            )
        return "Error: provide selector or text to click."

    def type_text(self, text: str, selector: str = "", press_enter: bool = False) -> str:
        try:
            page = self._ensure_open()
            if selector:
                target, count = _first_visible(page.locator(selector))
                if target is None:
                    return (
                        f"Type failed: no visible element matches '{selector}' "
                        f"({count} hidden match(es))."
                    )
                target.fill(text, timeout=self._TIMEOUT_MS)
                # fill() is not a promise that the text is there. It was
                # treated as one: the keyboard branch below checks
                # _text_landed and this branch returned "Typed 'x'."
                # unconditionally, which is the same false success the focus
                # check was added to remove — in the path the tool description
                # tells the model to PREFER, on the grounds that filling by
                # selector "cannot miss". It can. A React or Lexical composer
                # that rebuilds its DOM from its own state reverts a
                # programmatic fill, and browser_type was taken out of
                # _MUST_CHANGE_THE_PAGE on the reasoning that type_text
                # verifies itself — true of the branch below, and of nothing
                # here.
                if not self._selector_holds(page, selector, text):
                    return (
                        f"Type failed: '{selector}' does not contain the text "
                        f"after filling it. The field took the keystrokes and "
                        f"put them back — a composer that rebuilds itself from "
                        f"its own state does this. Click the field and type "
                        f"into it instead of filling it."
                    )
            else:
                # Check focus BEFORE sending anything. The old order typed
                # first and complained afterwards, and "the keystrokes went to
                # the page and were discarded" turned out to be false on every
                # site that binds bare letters: live 2026-09-06 on X, typing
                # "@grok you think a human posted this?" at an unfocused page
                # fired the single-key shortcuts those characters are bound to
                # — 'n' opened the composer partway through the sentence — so
                # the tail of the message ("osted this?") landed in a composer
                # the tool had itself opened by accident, the tool reported
                # that nothing had happened, and the next click submitted the
                # fragment. Keystrokes at a page are not inert. Do not send
                # them speculatively.
                if not self._editable_focused(page):
                    return (
                        "Type failed: nothing editable is focused, so no keys "
                        "were sent. Typing at an unfocused page does not "
                        "vanish — on sites that bind single keys (X, Gmail, "
                        "GitHub) each character fires whatever shortcut it is "
                        "bound to. Click the field first, then type, or pass "
                        "a selector to fill it directly."
                    )
                page.keyboard.type(text, delay=10)
                # Focused and editable is still not landed: a field can reject,
                # reformat or truncate what it is given. Reporting "Typed 'x'."
                # regardless is how the agent came to tell the user it had
                # posted a tweet while the composer still showed its
                # placeholder — the tool asserted success, so nothing
                # downstream could tell it had failed.
                if not self._text_landed(page, text):
                    where = self.focused_description()
                    return (
                        f"Type failed: '{text}' is not in the focused field "
                        f"after typing" + (f" (focus is on {where})" if where else "")
                        + ". The field may have rejected or reformatted it, and "
                        "it may now hold PART of the text — check the page and "
                        "clear it before retrying, rather than typing again on "
                        "top of what is there."
                    )
            if press_enter:
                page.keyboard.press("Enter")
                # Same false success press() now catches, reachable through
                # the other tool: browser_type with enter:true is how the
                # agent "submitted" a composer that only took a newline. The
                # text did land, so this is not a type failure -- say what
                # happened and what to do about it.
                if not self._enter_submitted(page):
                    # Take the newline back out. Enter that does not submit
                    # still inserted a character, and the usual recovery is to
                    # click the submit button next — so the newline ships with
                    # the message. Observed 2026-09-07 on a composer replica:
                    # a successful type-then-click posted
                    # "gave my agent eyes today. it looks first now.\n\n".
                    # Nothing else will remove it, because from here on the
                    # field looks exactly like one the user typed that way.
                    # Focus check first, for the same reason the type above
                    # has one: keystrokes at a page are not inert, and between
                    # the Enter and here the page may have navigated, opened a
                    # dialog, or moved focus. Restore the known-good text
                    # rather than trusting one Backspace to undo whatever
                    # Enter did — in a contenteditable it can insert a whole
                    # element, and a search box that submits while keeping its
                    # query reads as "not submitted" and would lose its last
                    # character.
                    try:
                        if self._editable_focused(page):
                            self._restore_field_text(page, text)
                    except Exception:
                        pass
                    # Worded to read as a failure to sounds_like_tool_error:
                    # the text landed, but the call the model asked for was
                    # type-and-submit, and the submit half did not happen.
                    return (
                        f"Typed '{text}' but the submit failed: the field "
                        f"still contains the text, so Enter inserted a "
                        f"newline rather than submitting. Click the submit "
                        f"button, or press cmd+enter, to submit."
                    )
            return f"Typed '{text}'" + (" and pressed Enter." if press_enter else ".")
        except Exception as e:
            return self._fail("type", e)

    @staticmethod
    def _restore_field_text(page: Any, text: str) -> None:
        """Put the focused field back to exactly `text`.

        Only ever narrows what is there back to what was typed, and only when
        the field currently starts with it — so a page that moved focus, or one
        whose Enter did something more interesting than insert a newline, is
        left alone rather than edited blind.
        """
        page.evaluate(
            """(t) => {
                const el = document.activeElement;
                if (!el) return;
                const cur = el.value !== undefined && el.value !== null
                    ? el.value : el.innerText;
                if (typeof cur !== 'string' || !cur.startsWith(t) || cur === t) return;
                if (el.value !== undefined && el.value !== null) {
                    el.value = t;
                } else {
                    el.innerText = t;
                }
                el.dispatchEvent(new Event('input', {bubbles: true}));
            }""",
            text)

    @staticmethod
    def _editable_focused(page: Any) -> bool:
        """Is an editable element focused right now, before any key is sent?

        Best-effort in the same direction as the checks below: a page that
        will not evaluate script counts as focused, so a verification that
        cannot run never blocks a type that would have worked.
        """
        try:
            return bool(page.evaluate(
                """() => {
                    const el = document.activeElement;
                    if (!el || el === document.body) return false;
                    return !!(el.isContentEditable
                        || el.tagName === 'INPUT' || el.tagName === 'TEXTAREA');
                }"""))
        except Exception:
            return True

    @staticmethod
    def _selector_holds(page: Any, selector: str, text: str) -> bool:
        """Does the element `selector` names actually contain `text` now?

        Best-effort in the same direction as every other check here: anything
        it cannot answer counts as landed, so a verification that will not run
        never fails a type that worked. querySelector does not speak XPath, so
        an XPath target raises and passes — which is the honest outcome, since
        this cannot see it either way.
        """
        try:
            return bool(page.evaluate(
                """([sel, t]) => {
                    const el = document.querySelector(sel);
                    if (!el) return true;
                    const v = el.value !== undefined && el.value !== null
                        ? el.value : el.innerText;
                    return typeof v === 'string' ? v.includes(t) : true;
                }""",
                [selector, text]))
        except Exception:
            return True

    @staticmethod
    def _text_landed(page: Any, text: str) -> bool:
        """Did `text` actually reach the focused editable element?

        Best-effort: a page that will not evaluate script, or a field that
        normalises what it stores, must not turn a working type into a
        reported failure — so anything unexpected counts as landed. This is
        here to catch the unambiguous case, keystrokes sent at document.body.
        """
        try:
            return bool(page.evaluate(
                """(t) => {
                    const el = document.activeElement;
                    if (!el || el === document.body) return false;
                    const editable = el.isContentEditable
                        || el.tagName === 'INPUT' || el.tagName === 'TEXTAREA';
                    if (!editable) return false;
                    const v = el.value !== undefined ? el.value : el.innerText;
                    return (v || '').includes(t);
                }""",
                text))
        except Exception:
            return True

    @staticmethod
    def _enter_submitted(page: Any) -> bool:
        """Did pressing Enter actually submit the focused form?

        Enter in a text field submits on some sites (a search box) and
        inserts a newline on others (X.com's composer). The difference is
        observable: a submit clears the field or moves focus, while a newline
        leaves the field focused and still holding its text. Best-effort like
        _text_landed: anything unexpected counts as submitted, so a working
        submit is never reported as a failure.
        """
        try:
            return bool(page.evaluate(
                """() => {
                    const el = document.activeElement;
                    if (!el || el === document.body) return true;
                    const editable = el.isContentEditable
                        || el.tagName === 'INPUT' || el.tagName === 'TEXTAREA';
                    if (!editable) return true;
                    const v = el.value !== undefined ? el.value : el.innerText;
                    return !(v || '').trim();
                }"""))
        except Exception:
            return True

    # A field that already holds text is a pending action: a composer with a
    # message in it, a search box with a query. Whether a click sent it is the
    # one thing a click result never said. Live 2026-09-07 the agent clicked
    # "Post", the tweet went out, and — reading nothing but "Clicked button
    # 'Post'." and a page dump it could not interpret — it told the user the
    # post "may not have been submitted successfully" and offered to press
    # cmd+enter, which is how the same message gets posted twice. _enter_
    # submitted already reads the difference for the keyboard; the field
    # clearing is the same observable evidence, and a click is the path X
    # actually submits on.
    #
    # The text itself is only ever compared, never quoted back into the
    # observation: the field this snapshots can be a password box, and the
    # observation is written to the transcript and to the training corpus.
    _PENDING_FIELD_JS = """() => {
        const typed = (el) => {
            if (!el) return false;
            if (el.isContentEditable) return true;
            if (el.tagName === 'TEXTAREA') return true;
            if (el.tagName !== 'INPUT') return false;
            return ['text', 'search', 'email', 'url', 'tel']
                .includes((el.type || 'text').toLowerCase());
        };
        const val = (el) => {
            const v = el.value !== undefined && el.value !== null
                ? el.value : el.innerText;
            return typeof v === 'string' ? v.trim() : '';
        };
        const active = document.activeElement;
        if (typed(active) && val(active)) return val(active);
        const filled = [...document.querySelectorAll(
            'input, textarea, [contenteditable]')]
            .filter(el => typed(el) && el.getClientRects().length && val(el));
        return filled.length === 1 ? val(filled[0]) : '';
    }"""

    _FIELD_STILL_HOLDS_JS = """(t) => [...document.querySelectorAll(
        'input, textarea, [contenteditable]')].some(el => {
            if (!(el.isContentEditable || el.tagName === 'INPUT'
                  || el.tagName === 'TEXTAREA')) return false;
            if (!el.getClientRects().length) return false;
            const v = el.value !== undefined && el.value !== null
                ? el.value : el.innerText;
            return typeof v === 'string' && v.includes(t);
        })"""

    def _pending_state(self, page: Any) -> tuple[str, str]:
        """(text sitting in an editable field, current url) before an action.

        The text prefers the focused field, and otherwise answers only when
        exactly one visible field on the page holds anything — with a filled
        search box AND a filled composer, "the pending field" is a guess, and
        a guess here produces a confident wrong sentence about what a click
        did, which is the failure this whole area exists to remove.

        The url comes along because "the field is gone" has two causes and
        opposite advice attached to each.
        """
        try:
            text = (page.evaluate(self._PENDING_FIELD_JS) or "").strip()
        except Exception:
            return "", ""
        try:
            return text, page.url or ""
        except Exception:
            return text, ""

    def _submit_note(self, page: Any, pending: tuple[str, str]) -> str:
        """What an action did to the field that held text before it.

        Returns "" when the answer is not knowable, which is a real outcome
        rather than a fallback: a page that will not evaluate leaves nothing to
        read, and calling that "submitted" would invent the very confirmation
        the model has no way to check.

        None of these sentences is worded as a failure. Unlike type(enter=True)
        the caller never said what it was aiming at, and "Bold" clicked over a
        full composer has failed at nothing — so this only adds information and
        lets the model account for it, the same contract as _no_effect_note.
        """
        text, url_before = pending
        if not text:
            return ""
        try:
            page.wait_for_timeout(400)  # let the submit handler run
        except Exception:
            pass
        try:
            still = bool(page.evaluate(self._FIELD_STILL_HOLDS_JS, text))
        except Exception:
            return ""
        if still:
            return (
                " The field that had text in it still holds it, so nothing "
                "was submitted — if this was meant to submit, it hit the "
                "wrong control."
            )
        # An empty field on a NEW page proves nothing: the old document went
        # away and took the draft with it. Both readings are wrong there — a
        # form post navigates to its result page, and so does a stray link
        # click that threw the draft away, and telling the model either "it
        # sent" or "retype it" picks one at random. Checked against the url
        # rather than skipped, because the same-page case is the one that
        # matters and it is the common one.
        try:
            navigated = bool(url_before) and (page.url or "") != url_before
        except Exception:
            navigated = False
        if navigated:
            return (
                " The page navigated, so the field that held text is gone with "
                "the document that owned it — that is not evidence either way. "
                "Read the new page to find out whether it went through."
            )
        return (
            " The field that held text before this is now empty, so this "
            "action submitted or discarded it — it is no longer pending. Do "
            "not retype it or repeat this to make sure: that sends it twice."
        )

    def press(self, key: str) -> str:
        try:
            page = self._ensure_open()
            normalized = _normalize_key(key)
            # Only Enter-family keys are judged. Every other key legitimately
            # leaves a composer holding its text, and the note would then fire
            # on every arrow key with a message half-written.
            submits = normalized.split("+")[-1] == "Enter"
            pending = self._pending_state(page) if submits else ("", "")
            page.keyboard.press(normalized)
            if normalized == "Enter" and not self._enter_submitted(page):
                return (
                    f"Press failed: Enter did not submit the form — the "
                    f"focused field still contains text, so it likely "
                    f"inserted a newline. Click the submit button to submit."
                )
            # cmd+enter, the combination that actually posts on X, reached
            # neither check: _enter_submitted is spelled for bare Enter, so
            # "Pressed 'Meta+Enter'." was returned whether or not anything
            # went. The before/after read does not care how the key is spelled.
            return f"Pressed '{normalized}'." + self._submit_note(page, pending)
        except Exception as e:
            return self._fail("press", e)

    def scroll(self, direction: str = "down", amount: int = 0) -> str:
        try:
            page = self._ensure_open()
            if direction not in ("down", "up"):
                return "Error: direction must be 'down' or 'up'."
            dy = amount if amount > 0 else 800
            if direction == "up":
                dy = -dy
            page.mouse.wheel(0, dy)
            page.wait_for_timeout(300)
            return f"Scrolled {direction} {abs(dy)}px."
        except Exception as e:
            return self._fail("scroll", e)

    def evaluate(self, script: str) -> str:
        try:
            page = self._ensure_open()
            result = page.evaluate(script)
            return json.dumps(result, ensure_ascii=False, default=str)[:4000]
        except Exception as e:
            return self._fail("evaluate", e)

    def screenshot(self, full_page: bool = True) -> str:
        try:
            page = self._ensure_open()
            path = _screenshot_path()
            page.screenshot(path=str(path), full_page=full_page)
            return f"Saved browser screenshot: {path.name}"
        except Exception as e:
            return self._fail("screenshot", e)

    def screenshot_path(self, full_page: bool = False) -> Path:
        """Capture and return the file, for a caller that wants to look at the
        image rather than tell the user where it went.

        Defaults to the VIEWPORT, not the full page, because the only consumer
        is coordinate grounding: a full-page capture of a long feed puts an
        element at y=4200 on an 800px-tall window, and clicking (x, 4200)
        clicks nothing. Viewport pixels are the coordinate space the mouse
        actually lives in.
        """
        page = self._ensure_open()
        path = _screenshot_path()
        page.screenshot(path=str(path), full_page=full_page)
        return path

    def click_at(self, x: int, y: int) -> str:
        """Click viewport coordinates, the way a person points at what they see.

        Everything else in this class aims by text or selector, which is why a
        click meant for a composer landed on the navigation item with the same
        word on it. A coordinate from an actual look at the screen has no
        homonym problem.
        """
        try:
            page = self._ensure_open()
            size = page.viewport_size or {}
            width = int(size.get("width") or 0)
            height = int(size.get("height") or 0)
            if width and height and not (0 <= x < width and 0 <= y < height):
                return (
                    f"Click failed: ({x}, {y}) is outside the {width}x{height} "
                    f"viewport, so it would land on nothing. Coordinates must "
                    f"come from a screenshot of the CURRENT viewport — scroll "
                    f"to bring the target into view and look again."
                )
            pending = self._pending_state(page)
            page.mouse.click(x, y)
            return f"Clicked at ({x}, {y})." + self._submit_note(page, pending)
        except Exception as e:
            return self._fail("click_at", e)

    # Everything the model needs to address a control exactly, read from the
    # DOM rather than from the pixels. Vision says what is on screen and what
    # state it is in; this says how to reach it. They fail in opposite places:
    # a control thinner than one 32px vision patch cannot be grounded at all
    # (x.com's composer is 28px, and grounding it missed by ~36px or returned
    # nothing), while the DOM has its exact handle and no opinion about which
    # of five buttons is the one a person would press.
    _CONTROLS_JS = r"""(limit) => {
        const uniq = (s) => {
            try { return s && document.querySelectorAll(s).length === 1; }
            catch (e) { return false; }
        };
        const sel = (el) => {
            const tag = el.tagName.toLowerCase();
            const tid = el.getAttribute('data-testid');
            if (tid) {
                const s = '[data-testid="' + tid + '"]';
                if (uniq(s)) return s;
            }
            if (el.id) {
                const s = '#' + CSS.escape(el.id);
                if (uniq(s)) return s;
            }
            for (const attr of ['aria-label', 'name', 'placeholder']) {
                const v = el.getAttribute(attr);
                if (v && !v.includes('"')) {
                    const s = tag + '[' + attr + '="' + v + '"]';
                    if (uniq(s)) return s;
                }
            }
            // Last resort: a positional path. Stable enough for the next
            // action, and not worth showing off about.
            const parts = [];
            let node = el;
            while (node && node.nodeType === 1 && parts.length < 5) {
                const parent = node.parentElement;
                if (!parent) break;
                const same = Array.from(parent.children)
                    .filter(c => c.tagName === node.tagName);
                parts.unshift(same.length > 1
                    ? node.tagName.toLowerCase() +
                      ':nth-of-type(' + (same.indexOf(node) + 1) + ')'
                    : node.tagName.toLowerCase());
                node = parent;
            }
            const s = parts.join(' > ');
            return uniq(s) ? s : '';
        };
        const firstLine = (t) => (t || '').trim().split(String.fromCharCode(10))[0];
        const label = (el) => (
            el.getAttribute('aria-label') || el.getAttribute('placeholder')
            || firstLine(el.innerText) || el.value || ''
        ).slice(0, 60);
        const out = [];
        const seen = new Set();
        const q = 'input, textarea, [contenteditable="true"], button, a[href], ' +
                  '[role="button"], [role="textbox"], [role="link"]';
        for (const el of document.querySelectorAll(q)) {
            if (out.length >= 200) break;
            const r = el.getBoundingClientRect();
            if (r.width < 4 || r.height < 4) continue;
            if (r.bottom < 0 || r.top > innerHeight) continue;
            if (r.right < 0 || r.left > innerWidth) continue;
            const style = getComputedStyle(el);
            if (style.visibility === 'hidden' || style.display === 'none') continue;
            const s = sel(el);
            if (!s || seen.has(s)) continue;
            seen.add(s);
            const tag = el.tagName.toLowerCase();
            const editable = el.isContentEditable || tag === 'input'
                || tag === 'textarea' || el.getAttribute('role') === 'textbox';
            let value = '';
            if (editable) {
                value = (el.value !== undefined && el.value !== null)
                    ? String(el.value) : (el.innerText || '');
            }
            out.push({
                kind: editable ? 'field' : 'button',
                selector: s,
                label: label(el),
                value: value.slice(0, 80),
                disabled: !!(el.disabled || el.getAttribute('aria-disabled') === 'true'),
                x: Math.round(r.x + r.width / 2),
                y: Math.round(r.y + r.height / 2),
            });
        }
        return out;
    }"""

    def controls(self, limit: int = 25) -> list[dict]:
        """Visible fields and buttons, each with a selector that addresses it.

        Fields come first, then buttons. Document order alone is the wrong
        order for a budget: on x.com the sidebar nav is eighteen links before
        the composer, so a straight DOM-order cut listed every navigation item
        and none of the two controls anyone needs. What you fill matters more
        than what you can click past.

        Returns [] on any failure — this enriches a look, it must never be the
        reason one fails.
        """
        return self.controls_read(limit)[0]

    def controls_read(self, limit: int = 25) -> tuple[list[dict], bool]:
        """(controls, whether the page could be read at all).

        [] means two different things and a caller acting on it has to know
        which: no control matched, or the read never happened — the browser is
        not open, the page navigated mid-call, a CSP blocked the evaluate. The
        difference decides whether "I could not find it" may be reported to
        the model as "it is NOT there", and see_screen was reporting a failed
        read as proof of absence about controls it had simply never looked at.
        """
        try:
            page = self._ensure_open()
            found = list(page.evaluate(self._CONTROLS_JS, limit) or [])
        except Exception:
            return [], False
        fields = [c for c in found if c.get("kind") == "field"]
        buttons = [c for c in found if c.get("kind") != "field"]
        return (fields + buttons)[:limit], True

    def focused_description(self) -> str:
        """What has keyboard focus right now, in words. '' if nothing does.

        The type path can already tell that focus is wrong; this says what it
        is instead, which is the difference between "that failed" and "you are
        focused on the search box, not the composer".
        """
        try:
            page = self._ensure_open()
            return str(page.evaluate(
                """() => {
                    const el = document.activeElement;
                    if (!el || el === document.body) return '';
                    const label = el.getAttribute('aria-label')
                        || el.getAttribute('placeholder')
                        || el.getAttribute('name') || '';
                    const editable = el.isContentEditable
                        || el.tagName === 'INPUT' || el.tagName === 'TEXTAREA';
                    return `${el.tagName.toLowerCase()}${label ? ' "' + label + '"' : ''}`
                        + (editable ? ' (editable)' : ' (not editable)');
                }""") or "")
        except Exception:
            return ""

    def close(self) -> str:
        """Close the visible browser page/window and return immediately.

        We keep the Playwright driver alive so reopening is fast; the driver
        is stopped lazily on process exit via the atexit hook registered on
        first use. This avoids a multi-second Playwright/asyncio teardown
        blocking every chat turn."""
        try:
            if self._browser:
                self._browser.close()
                self._browser = None
            self._page = None
            self._last_url = ""
            return "Browser closed."
        except Exception as e:
            return f"Browser close error: {e}"

    def _stop_playwright(self):
        """Final cleanup: stop the Playwright driver. Called at process exit."""
        try:
            if self._browser:
                self._browser.close()
                self._browser = None
            if self._playwright:
                self._playwright.stop()
                self._playwright = None
            self._page = None
        except BaseException:
            # Catch KeyboardInterrupt and SystemExit too — this is an atexit
            # handler; if the user is already trying to quit, a cascading
            # interrupt inside browser.close() must not print a traceback.
            pass

    def _register_exit_cleanup(self):
        """Register a one-time atexit hook so Playwright doesn't outlive us."""
        atexit.register(self._stop_playwright)


def _screenshot_path() -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return SCREENSHOTS_DIR / f"screenshot_{ts}.png"


def _truncated(text: str, max_len: int) -> str:
    text = text.strip()
    if len(text) > max_len:
        text = text[:max_len] + "\n... (truncated)"
    return text


# ---------- Desktop automation (PyAutoGUI) ----------

def _init_pyautogui():
    import pyautogui
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.05
    return pyautogui


def desktop_screenshot() -> str:
    try:
        pyautogui = _init_pyautogui()
        path = _screenshot_path()
        img = pyautogui.screenshot()
        img.save(path)
        return f"Saved desktop screenshot: {path.name}"
    except Exception as e:
        return f"Desktop screenshot error: {e}"


def desktop_screenshot_path() -> Path:
    """Capture the screen and return the file, for looking at rather than
    telling the user about."""
    pyautogui = _init_pyautogui()
    path = _screenshot_path()
    pyautogui.screenshot().save(path)
    return path


# macOS hands back a black frame, not an error, when the process capturing the
# screen was never granted Screen Recording permission. Observed 2026-09-07:
# a full desktop capture came back 1920x1080 with every pixel at 0 and the
# vision model dutifully reported "the image is entirely black with no text,
# buttons or links" — which reads like a description of the screen rather than
# a description of a missing permission. Catch it here and say what to do.
_BLANK_MEAN = 8.0
_BLANK_STDDEV = 1.0


def screenshot_is_blank(path) -> bool:
    """Is this capture a featureless black frame rather than a screen?"""
    try:
        from PIL import Image, ImageStat

        with Image.open(path) as im:
            stat = ImageStat.Stat(im.convert("L"))
        return stat.stddev[0] < _BLANK_STDDEV and stat.mean[0] < _BLANK_MEAN
    except Exception:
        # Unreadable for some other reason; let the normal path report that.
        return False


SCREEN_PERMISSION_HINT = (
    "The screen capture came back entirely black. macOS returns a blank frame "
    "instead of an error when the application running me has not been granted "
    "Screen Recording permission. Grant it in System Settings > Privacy & "
    "Security > Screen Recording for the terminal (or app) Symbio runs in, "
    "restart it, and look again. Until then I cannot see the desktop — the "
    "browser page is still visible to me with target='browser'."
)


def desktop_pixel_scale(image_size: tuple[int, int]) -> float:
    """How many image pixels per mouse point on this display.

    On a Retina Mac the screenshot comes back in physical pixels (2880x1800)
    while pyautogui clicks in logical points (1440x900). A coordinate read off
    the image and passed straight to click() therefore lands at twice the
    intended offset — bottom-right of the screen for anything below the middle
    of it. Deriving the factor from the two sizes handles Retina, non-Retina
    and mixed-DPI setups without hardcoding 2.
    """
    pyautogui = _init_pyautogui()
    logical_w, _ = pyautogui.size()
    if not logical_w or not image_size[0]:
        return 1.0
    return image_size[0] / float(logical_w)


def desktop_click_in_image(x: int, y: int, image_size: tuple[int, int] | None = None,
                           clicks: int = 1, button: str = "left") -> str:
    """Click a point expressed in SCREENSHOT pixel coordinates.

    Derives the scale itself when the caller has no image size to offer. The
    caller's cached size used to be the only source, and it defaults to (0, 0)
    — before the session's first look, or whenever PIL could not read the file
    — at which point the conversion was skipped and the raw physical pixels
    went straight to the mouse. That is the 2x miss desktop_pixel_scale exists
    to prevent, and on the desktop a wrong coordinate is a different action,
    not a missed one.
    """
    try:
        if not (image_size and image_size[0]):
            image_size = _screen_pixel_size()
        if not (image_size and image_size[0]):
            return ("Click failed: I cannot tell this display's pixel scale, "
                    "so the coordinate might land at twice its intended "
                    "offset. Take a fresh look with see_screen first.")
        scale = desktop_pixel_scale(image_size)
        return desktop_click(round(x / scale), round(y / scale), clicks, button)
    except Exception as e:
        return f"Desktop click error: {e}"


def _screen_pixel_size() -> tuple[int, int]:
    """The screen's size in captured pixels, or (0, 0)."""
    try:
        pyautogui = _init_pyautogui()
        shot = pyautogui.screenshot()
        return (shot.width, shot.height)
    except Exception:
        return (0, 0)


def desktop_click(x: int, y: int, clicks: int = 1, button: str = "left") -> str:
    try:
        pyautogui = _init_pyautogui()
        pyautogui.click(x, y, clicks=clicks, button=button)
        return f"Clicked at ({x}, {y}) with {button} button ({clicks} click(s))."
    except Exception as e:
        return f"Desktop click error: {e}"


def desktop_move(x: int, y: int) -> str:
    try:
        pyautogui = _init_pyautogui()
        pyautogui.moveTo(x, y)
        return f"Moved mouse to ({x}, {y})."
    except Exception as e:
        return f"Desktop move error: {e}"


def desktop_type(text: str, interval: float = 0.01) -> str:
    try:
        pyautogui = _init_pyautogui()
        pyautogui.typewrite(text, interval=interval)
        return f"Typed '{text}' on the desktop."
    except Exception as e:
        return f"Desktop type error: {e}"


def desktop_press(key: str) -> str:
    try:
        pyautogui = _init_pyautogui()
        # pyautogui wants lowercase names ('down', 'pagedown', 'command').
        k = key.strip().lower().replace(" ", "")
        k = {"arrowdown": "down", "arrowup": "up", "arrowleft": "left", "arrowright": "right",
             "control": "ctrl", "meta": "command", "cmd": "command", "return": "enter"}.get(k, k)
        pyautogui.press(k)
        return f"Pressed '{k}' on the desktop."
    except Exception as e:
        return f"Desktop press error: {e}"
