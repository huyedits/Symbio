"""The four pieces of the screen, each owning one job.

Logo on top, conversation in the middle, the command strip under it, the input
box docked to the bottom. History is a RichLog because the daemon's frames
already carry ANSI — the skin chat_style puts on a status line is escape codes,
and re-parsing them into markup would be a second representation that can
drift from the real one.

Suggestions are a separate widget rather than a dropdown inside the input,
because a slash-command list that steals focus is a list you cannot type past.
"""
from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Input, OptionList, RichLog, Static
from textual.widgets.option_list import Option

from symbio_tui.logo import logo, tagline


class Logo(Static):
    """The stick figure, redrawn on resize and dropped when it will not fit."""

    def __init__(self, assistant: str = "Symbio") -> None:
        super().__init__("", id="logo")
        self._assistant = assistant

    def on_mount(self) -> None:
        self.redraw()

    def on_resize(self) -> None:
        self.redraw()

    def redraw(self) -> None:
        size = self.app.size
        art = logo(size.width, size.height)
        self.display = bool(art)
        if art:
            self.update(f"{art}\n{tagline(self._assistant)}")


class History(RichLog):
    """Everything the session has said. Scrollable, ANSI kept as ANSI."""

    def __init__(self) -> None:
        super().__init__(highlight=False, markup=False, wrap=True,
                         auto_scroll=True, id="history")
        # Scrolling is the point of the middle pane: a conversation you cannot
        # go back through is a conversation you have to re-ask.
        self.can_focus = True

    def say(self, text: str) -> None:
        for line in (text or "").splitlines() or [""]:
            self.write(line)

    def say_inline(self, text: str) -> None:
        """A streamed fragment: appended, not given a line of its own."""
        if text:
            self.write(text.rstrip("\n"))


class CommandStrip(Static):
    """What is typeable, in one line, so the commands are on the screen."""

    def __init__(self, names: list[str]) -> None:
        super().__init__("", id="commands")
        self._names = list(names)

    def on_mount(self) -> None:
        self.redraw()

    def on_resize(self) -> None:
        self.redraw()

    def redraw(self) -> None:
        width = max(20, self.app.size.width - 4)
        shown, used = [], 0
        for name in self._names:
            item = f"/{name}"
            if used + len(item) + 2 > width:
                shown.append("…")
                break
            shown.append(item)
            used += len(item) + 2
        self.update("  ".join(shown))


class Suggestions(OptionList):
    """Slash commands matching what has been typed, or hidden."""

    def __init__(self, names: list[str]) -> None:
        super().__init__(id="suggestions")
        self._names = list(names)
        self.display = False
        self.can_focus = False

    def offer(self, typed: str) -> None:
        if not typed.startswith("/"):
            self.display = False
            return
        prefix = typed[1:].lower()
        hits = [n for n in self._names if n.lower().startswith(prefix)][:8]
        self.clear_options()
        if not hits:
            self.display = False
            return
        self.add_options([Option(f"/{n}", id=n) for n in hits])
        self.highlighted = 0
        self.display = True

    @property
    def current(self) -> str | None:
        if not self.display or self.highlighted is None:
            return None
        option = self.get_option_at_index(self.highlighted)
        return option.id if option else None

    def step(self, delta: int) -> None:
        if not self.display or not self.option_count:
            return
        self.highlighted = ((self.highlighted or 0) + delta) % self.option_count


class Composer(Vertical):
    """Suggestions, the status line, and the input box that never moves."""

    def __init__(self, names: list[str]) -> None:
        super().__init__(id="composer")
        self._names = names

    def compose(self) -> ComposeResult:
        yield Suggestions(self._names)
        yield Static("", id="status")
        yield Input(placeholder="Ask, or / for a command", id="prompt")
