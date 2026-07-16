"""Browser and desktop automation helpers for Symbio.

Uses Playwright for browser control and PyAutoGUI for desktop mouse/keyboard.
All screenshots are saved to screenshots/ so the user can view them; the text-only
model receives text/HTML representations of the page instead of images.
"""

from __future__ import annotations

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


def _first_visible(locator: Any) -> tuple[Any | None, int]:
    """Return (first visible element, total match count) for a locator."""
    count = locator.count()
    visible = locator.filter(visible=True)
    if visible.count() > 0:
        return visible.first, count
    return None, count


def _confirm_domain(domain: str) -> bool:
    """Prompt the user before opening a new domain."""
    print(f"  [Computer] Allow browser to access '{domain}'? [y/N]:", end=" ", flush=True)
    try:
        answer = input().strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"
    return answer in ("y", "yes")


class BrowserSession:
    """Manages a single Playwright browser/page session."""

    def __init__(self):
        self._playwright: Any | None = None
        self._browser: Any | None = None
        self._page: Any | None = None
        self._confirmed: set[str] = set(_DEFAULT_ALLOWLIST)

    def _init(self) -> tuple[Any, Any]:
        if self._page is not None:
            return self._browser, self._page

        from playwright.sync_api import sync_playwright

        self._playwright = sync_playwright().start()
        # Default to bundled Chromium; user can override via browser channel.
        self._browser = self._playwright.chromium.launch(headless=False)
        context = self._browser.new_context(
            viewport={"width": 1280, "height": 800},
            accept_downloads=False,
        )
        self._page = context.new_page()
        return self._browser, self._page

    def _ensure_open(self) -> Any:
        if self._page is None:
            raise RuntimeError("Browser is not open. Use browser_open first.")
        return self._page

    # Errors that mean the Playwright connection itself is wedged (dead
    # greenlet/thread, closed pipe or loop) — no further call can succeed.
    _FATAL_MARKERS = (
        "cannot switch to a different thread",
        "has been closed",
        "connection closed",
        "event loop is closed",
        "pipe closed",
    )

    def _reset(self):
        """Tear down a broken session so the next browser_open starts clean."""
        for closer in (
            lambda: self._browser.close() if self._browser else None,
            lambda: self._playwright.stop() if self._playwright else None,
        ):
            try:
                closer()
            except Exception:
                pass
        self._browser = None
        self._playwright = None
        self._page = None

    def _fail(self, op: str, e: Exception) -> str:
        msg = str(e).lower()
        if any(marker in msg for marker in self._FATAL_MARKERS):
            self._reset()
            return (
                f"Browser {op} error: the browser session broke and was reset. "
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
            if _confirm_domain(domain):
                self._confirmed.add(domain)
            else:
                return False, f"User denied access to '{domain}'."
        return True, ""

    def open(self, url: str, channel: str = "") -> str:
        ok, msg = self._check_url(url)
        if not ok:
            return f"Browser open blocked: {msg}"
        try:
            _, page = self._init()
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            title = page.title()
            return f"Opened browser at {url}. Page title: {title}"
        except Exception as e:
            return self._fail("open", e)

    def navigate(self, url: str) -> str:
        return self.open(url)

    def get_text(self) -> str:
        page = self._ensure_open()
        try:
            text = page.inner_text("body", timeout=10000)
            # Collapse whitespace.
            text = re.sub(r"\n{3,}", "\n\n", text)
            text = re.sub(r"[ \t]+", " ", text)
            return _truncated(text.strip(), 4000)
        except Exception as e:
            return self._fail("get_text", e)

    def get_html(self) -> str:
        page = self._ensure_open()
        try:
            html = page.content()
            return _truncated(html.strip(), 4000)
        except Exception as e:
            return self._fail("get_html", e)

    _TIMEOUT_MS = 4000

    def click(self, selector: str = "", text: str = "") -> str:
        page = self._ensure_open()
        try:
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
                target, count = _first_visible(page.get_by_text(text, exact=False))
                if target is None:
                    for role in ("button", "link"):
                        target, count = _first_visible(page.get_by_role(role, name=text, exact=False))
                        if target is not None:
                            break
                if target is not None:
                    target.click(timeout=self._TIMEOUT_MS)
                    return f"Clicked element containing text '{text}'."
                return (
                    f"Click failed: no visible element with text '{text}'. "
                    "Use browser_get_text to see what is on the page."
                )
            return "Error: provide selector or text to click."
        except Exception as e:
            return self._fail("click", e)

    def type_text(self, text: str, selector: str = "", press_enter: bool = False) -> str:
        page = self._ensure_open()
        try:
            if selector:
                target, count = _first_visible(page.locator(selector))
                if target is None:
                    return (
                        f"Type failed: no visible element matches '{selector}' "
                        f"({count} hidden match(es))."
                    )
                target.fill(text, timeout=self._TIMEOUT_MS)
            else:
                page.keyboard.type(text, delay=10)
            if press_enter:
                page.keyboard.press("Enter")
            return f"Typed '{text}'" + (" and pressed Enter." if press_enter else ".")
        except Exception as e:
            return self._fail("type", e)

    def press(self, key: str) -> str:
        page = self._ensure_open()
        try:
            page.keyboard.press(key)
            return f"Pressed '{key}'."
        except Exception as e:
            return self._fail("press", e)

    def evaluate(self, script: str) -> str:
        page = self._ensure_open()
        try:
            result = page.evaluate(script)
            return json.dumps(result, ensure_ascii=False, default=str)[:4000]
        except Exception as e:
            return self._fail("evaluate", e)

    def screenshot(self) -> str:
        page = self._ensure_open()
        try:
            path = _screenshot_path()
            page.screenshot(path=str(path), full_page=True)
            return f"Saved browser screenshot: {path.name}"
        except Exception as e:
            return self._fail("screenshot", e)

    def close(self) -> str:
        try:
            if self._browser:
                self._browser.close()
                self._browser = None
            if self._playwright:
                self._playwright.stop()
                self._playwright = None
            self._page = None
            return "Browser closed."
        except Exception as e:
            return f"Browser close error: {e}"


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
        pyautogui.press(key)
        return f"Pressed '{key}' on the desktop."
    except Exception as e:
        return f"Desktop press error: {e}"
