"""The four pieces of the screen, each owning one job.

Face on top, conversation in the middle, the command strip under it, the input
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

from symbio_tui.faces import banner


class Face(Static):
    """The session's current mood, as a face.

    Driven by the `[Mood: tag]` line chat_turn emits once a turn, so it moves
    because the session moved. A face picked at random would be a lie told in
    a friendly way.
    """

    def __init__(self, assistant: str = "Symbio") -> None:
        super().__init__("", id="face")
        self._assistant = assistant
        self._mood: str | None = None

    def on_mount(self) -> None:
        self.redraw()

    def on_resize(self) -> None:
        self.redraw()

    def set_mood(self, mood: str | None) -> None:
        self._mood = mood
        self.redraw()

    @property
    def mood(self) -> str | None:
        return self._mood

    def redraw(self) -> None:
        self.update(banner(self._mood, self._assistant, self.app.size.width))


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


class PromptInput(Input):
    """The input box, with the keys the suggestion menu needs taken first.

    Handled on the WIDGET, not on the app. A focused widget sees a key before
    any app-level binding does, so tab reached Input's own focus traversal and
    the completion never ran — visible in a pty as "/sta" sitting in the box
    with "/status" listed right above it, while the headless test passed
    because it called the action instead of pressing the key.
    """

    def on_key(self, event) -> None:
        menu = self.app.query_one(Suggestions)
        if not menu.display:
            return
        if event.key == "tab":
            event.prevent_default()
            event.stop()
            self.app.action_complete()
        elif event.key in ("down", "up"):
            event.prevent_default()
            event.stop()
            menu.step(1 if event.key == "down" else -1)
        elif event.key == "escape":
            event.prevent_default()
            event.stop()
            menu.display = False


class Composer(Vertical):
    """Suggestions, the status line, and the input box that never moves."""

    def __init__(self, names: list[str]) -> None:
        super().__init__(id="composer")
        self._names = names

    def compose(self) -> ComposeResult:
        yield Suggestions(self._names)
        yield Static("", id="status")
        yield PromptInput(placeholder="Ask, or / for a command", id="prompt")
