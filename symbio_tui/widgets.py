"""The pieces of the screen, each owning one job.

The shape is Claude Code's, because that shape has been through more hours of
real use than anything invented here: a quiet header, a transcript that reads
as a list of events rather than as a wall of text, a rounded composer docked to
the bottom with a chevron in it, and one line of hints under it. What the
transcript gains from that is legibility at a glance — an accent dot for a turn
the assistant took, a hooked line for what came back out of it, a dim chevron
for what you typed — so a screenful of scrollback can be skimmed for the one
event you are looking for instead of read.

Face on top, conversation in the middle, the composer docked to the bottom.
History is a RichLog because the daemon's frames already carry ANSI — the skin
chat_style puts on a status line is escape codes, and re-parsing them into
markup would be a second representation that can drift from the real one.
Styled lines are written as Rich renderables next to them, which RichLog
handles without either representation touching the other.

Suggestions are a separate widget rather than a dropdown inside the input,
because a slash-command list that steals focus is a list you cannot type past.
"""
from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Input, OptionList, RichLog, Static
from textual.widgets.option_list import Option

from symbio_tui.faces import banner

# The marks Claude Code uses, and what each one means here.
#   ● a turn the assistant took            ⎿ what came back out of it
#   > what you typed                       ✻ the session working
BULLET = "●"
HOOK = "⎿"
CHEVRON = ">"

# The spinner, in the order it cycles. Four frames at ~8/second reads as
# movement without becoming a strobe in the corner of your eye.
SPINNER = ("✳", "✴", "✵", "✶")


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
    """Everything the session has said. Scrollable, ANSI kept as ANSI.

    Four ways in, because four kinds of line want to look different: what you
    typed, what the assistant said, what a tool gave back, and raw frames from
    the daemon that already carry their own colour.
    """

    def __init__(self) -> None:
        super().__init__(highlight=False, markup=False, wrap=True,
                         auto_scroll=True, id="history")
        # Scrolling is the point of the middle pane: a conversation you cannot
        # go back through is a conversation you have to re-ask.
        self.can_focus = True

    def say(self, text: str) -> None:
        """A raw frame from the daemon. Passed through untouched — it may
        already be carrying ANSI from chat_style."""
        for line in (text or "").splitlines() or [""]:
            self.write(line)

    def say_inline(self, text: str) -> None:
        """A streamed fragment: appended, not given a line of its own."""
        if text:
            self.write(text.rstrip("\n"))

    def say_user(self, text: str) -> None:
        """What you typed, echoed the way the composer showed it."""
        self.write(Text(f"{CHEVRON} {text}", style="dim"))

    def say_agent(self, text: str) -> None:
        """A turn the assistant took. The dot marks the first line; the rest
        of the paragraph is indented under it so a long answer stays one
        visible block rather than merging with what follows."""
        lines = (text or "").splitlines() or [""]
        first = Text(f"{BULLET} ", style="bold #d97757")
        first.append(lines[0], style="")
        self.write(first)
        for line in lines[1:]:
            self.write(Text(f"  {line}"))

    def say_tool(self, text: str) -> None:
        """What came back out of a turn: a tool result, a confirmation, a
        note from the daemon. Hooked and dimmed, so it reads as a consequence
        of the line above rather than as something said."""
        lines = [line.strip() for line in (text or "").splitlines()] or [""]
        self.write(Text(f"  {HOOK}  {lines[0]}", style="dim"))
        for line in lines[1:]:
            self.write(Text(f"     {line}", style="dim"))


class StatusLine(Static):
    """One line above the composer: what the session is doing right now.

    Busy is animated, because a status that cannot be told apart from a frozen
    one is the thing people reach for ctrl-c over. Idle is the same line
    without the spinner, so nothing moves when nothing is happening.
    """

    def __init__(self) -> None:
        super().__init__("", id="status")
        self._text = ""
        self._busy = False
        self._frame = 0
        self._timer = None

    def on_mount(self) -> None:
        self._timer = self.set_interval(0.12, self._tick, pause=True)

    def show(self, text: str, busy: bool = False) -> None:
        self._text = text or ""
        self._busy = busy and bool(self._text)
        if self._timer is not None:
            self._timer.resume() if self._busy else self._timer.pause()
        self._draw()

    def _tick(self) -> None:
        self._frame = (self._frame + 1) % len(SPINNER)
        self._draw()

    def _draw(self) -> None:
        if not self._text:
            self.update("")
            return
        if not self._busy:
            self.update(Text(f"  {self._text}", style="dim"))
            return
        line = Text(f"  {SPINNER[self._frame]} ", style="#d97757")
        line.append(self._text, style="dim")
        line.append("  (esc to interrupt)", style="dim")
        self.update(line)


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


class Hints(Static):
    """The one line under the box. Keys you would otherwise have to be told."""

    def __init__(self) -> None:
        super().__init__("", id="hints")

    def on_mount(self) -> None:
        self.update(Text(
            "  / for commands · tab to complete · ctrl+l clears · ctrl+c quits",
            style="dim"))


class Composer(Vertical):
    """Suggestions, the status line, the input box, and the hints.

    The box is a bordered row rather than a bordered Input: the chevron has to
    sit INSIDE the border to read as a prompt, and an Input that draws its own
    border cannot have anything inside it but text.
    """

    def __init__(self, names: list[str]) -> None:
        super().__init__(id="composer")
        self._names = names

    def compose(self) -> ComposeResult:
        yield Suggestions(self._names)
        yield StatusLine()
        with Horizontal(id="promptrow"):
            yield Static(CHEVRON, id="chevron")
            yield PromptInput(placeholder="Ask anything, or / for a command",
                              id="prompt")
        yield Hints()
