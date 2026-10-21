"""The pieces of the screen, each owning one job.

The shape is Hermes Agent's: a quiet header, a transcript that reads as a list
of events rather than as a wall of text, a rounded composer docked to the
bottom with a chevron in it, and one line of hints under it. What the
transcript gains from that is legibility at a glance — a gold dot for a turn
the assistant took, a hooked line for what came back out of it, a dim chevron
for what you typed, a dim ellipsis for what the model thought out loud before
answering — so a screenful of scrollback can be skimmed for the one
event you are looking for instead of read.

The palette is not borrowed: gold marks what is live, and the ground stays
Symbio's own pond and ink. See symbio_tui/theme.py for why that split is the
whole point.

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

from rich.ansi import AnsiDecoder
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Input, OptionList, RichLog, Static
from textual.widgets.option_list import Option

# Two constants the daemon itself owns, imported rather than repeated: a
# second written copy of either is a representation that can drift from the
# real one — this codebase's bug history is largely two representations
# disagreeing.
from symbio.app.chat_style import BULLET as DAEMON_BULLET  # noqa: E402
from symbio.app.tooling import REASONING_MARKER  # noqa: E402

from symbio_tui.faces import banner
from symbio_tui.theme import DIM, GOLD, MARK

# The marks Hermes Agent uses, and what each one means here.
#   ● a turn the assistant took            ⎿ what came back out of it
#   > what you typed                       ⋯ the model thinking out loud
BULLET = "●"
HOOK = "⎿"
CHEVRON = ">"
THINK = "⋯"

# The answer prefix the daemon streams in front of the reply, painted or
# plain (chat_style.assistant_prefix): the wire chunk that opens the answer
# part of a turn. The TUI's own bullet stays DIFFERENT on purpose — reading
# "● ⏺ …" mid-transcript is how a double mark got printed, 2026-10-14.
DAEMON_PREFIX = f"{DAEMON_BULLET} "

_ANSI = AnsiDecoder()


def _plain(text: str) -> str:
    """The line with SGR paint decoded away. Only for looking at what a
    frame says; the styled form is what reaches the screen."""
    if "\x1b" not in text:
        return text
    return _ANSI.decode_line(text).plain

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
        # min_width: RichLog's default is 78, so every line was wrapped at 78
        # columns whatever the window was: a 160-column window left half the
        # pane empty, and a 60-column one cut the right of every line off
        # (measured 2026-10-08 with run_test at 120 and 60 columns).
        super().__init__(highlight=False, markup=False, wrap=True,
                         auto_scroll=True, min_width=20, id="history")
        # Scrolling is the point of the middle pane: a conversation you cannot
        # go back through is a conversation you have to re-ask.
        self.can_focus = True
        # RichLog wraps a line once, at the width it had when written, and
        # never again. Keeping what was said lets a resize re-wrap all of it.
        self._said: list = []
        self._wrapped_at = 0

    def write(self, content, *args, **kwargs):
        self._said.append(content)
        return self._wrap(content, *args, **kwargs)

    def _wrap(self, content, *args, **kwargs):
        # Wrap at the pane, not at what Rich measures: Rich measures against
        # its console, whose width is not the pane's (80 in a headless run,
        # which capped a 200-column window at 80).
        pane = self.scrollable_content_region.width
        if pane and not args and "width" not in kwargs:
            kwargs["width"] = pane
        return super().write(content, *args, **kwargs)

    def clear(self):
        self._said = []
        return super().clear()

    def on_resize(self) -> None:
        width = self.scrollable_content_region.width
        if not width or width == self._wrapped_at:
            return
        self._wrapped_at = width
        said, at_end = self._said, self.is_vertical_scroll_end
        super().clear()
        for content in said:
            self._wrap(content, scroll_end=False)
        if at_end:
            self.scroll_end(animate=False)

    def say(self, text: str) -> None:
        """A raw frame from the daemon. It arrives with its own chat_style
        paint as SGR escape codes, and RichLog does NOT parse codes embedded
        in a plain string — `write` used to store them as literal characters
        with no style: the colour was dropped and the escapes went into the
        cell grid as text (measured 2026-10-14: a styled daemon line came
        through with style=None, `\x1b[38;2…m` sitting in the segment). So
        the raw form is decoded into Rich styles instead — the same paint,
        one representation."""
        if not text:
            return
        if "\x1b" not in text:
            for line in text.splitlines() or [""]:
                self.write(line)
            return
        for line in text.splitlines() or [""]:
            self.write(_ANSI.decode_line(line))

    def say_inline(self, text: str) -> None:
        """A streamed fragment: appended onto ONE growing line, not one
        RichLog line per token.

        RichLog.write() starts a new line every call, so feeding it token
        fragments put the reply one-token-per-row — "12" rendered as two
        rows, "1" and "2" (measured 2026-10-05, the 6+6 turn through the
        TUI). Instead the app-level LiveLine widget paints the growing
        reply; the app routes stream frames THERE, and flushes it into this
        log as a finished turn when prose ends.
        """
        # Kept for raw passthrough (non-streaming clients); the app routes
        # stream frames to the LiveLine widget instead of here.
        if text and text.endswith("\n"):
            self.write(text.rstrip("\n"))

    def say_user(self, text: str) -> None:
        """What you typed, echoed the way the composer showed it."""
        self.write(Text(f"{CHEVRON} {text}", style="dim"))

    def say_agent(self, text: str) -> None:
        """A turn the assistant took. The dot marks the first line; the rest
        of the paragraph is indented under it so a long answer stays one
        visible block rather than merging with what follows.

        The bullet's style is a span over the dot alone: a style set on the
        Text itself covers everything appended to it, and the first version
        read `Text(f"{BULLET} ", style=f"bold {GOLD}")` then appended the
        prose — which drew every reply bold gold (measured 2026-10-14: one
        stored segment, style "bold #ffd700", over the whole line). Gold
        marks what is live; the reply is ink. A "⏺ " the daemon already
        streamed in front of the text is dropped, so the turn never shows
        two marks."""
        lines = (text or "").splitlines() or [""]
        head = _plain(lines[0])
        if head.startswith(DAEMON_PREFIX):
            head = head[len(DAEMON_PREFIX):]
        first = Text()
        first.append(f"{BULLET} ", style=f"bold {GOLD}")
        if head:
            first.append(head, style="")
        self.write(first)
        for line in lines[1:]:
            self.write(Text(f"  {line}"))

    def say_thinking(self, text: str) -> None:
        """The model reasoning out loud, streamed before it answers.

        Its own voice, not the answer's: dim, an ellipsis for a mark, so the
        transcript separates what the model thought from what it commits to.
        The thinking is part of the raw stream and part of the turn — hiding
        it would make the transcript a lie about how the answer was come by.
        """
        lines = (text or "").splitlines() or [""]
        self.write(Text(f"{THINK} {lines[0]}", style="dim"))
        for line in lines[1:]:
            self.write(Text(f"  {line}", style="dim"))

    def say_tool(self, text: str) -> None:
        """What came back out of a turn: a tool result, a confirmation, a
        note from the daemon. Hooked and dimmed, so it reads as a consequence
        of the line above rather than as something said."""
        lines = [line.strip() for line in (text or "").splitlines()] or [""]
        self.write(Text(f"  {HOOK}  {lines[0]}", style="dim"))
        for line in lines[1:]:
            self.write(Text(f"     {line}", style="dim"))


class LiveLine(Static):
    """The reply being written right now, with the thinking shown as it
    happens — in its own voice, not pasted into the answer.

    The daemon streams a turn over the wire in phases (chat.py `_emit`
    feeding StreamingStripper), and the phase is readable off the text:

    * reasoning opens with REASONING_MARKER and streams a chunk at a time;
    * the answer opens as chat_style's assistant prefix chunk — `⏺ `,
      painted or plain;
    * each generation round closes with a lone `\n` chunk.

    All of that used to land in ONE growing gold line — thinking, answer and
    the daemon's own bullet concatenated — so the reply read as one unmarked
    paste and, on stream, showed a double mark when it reached the
    transcript. Now thinking feeds this line dim and is sealed into the
    transcript the moment the answer opens; the answer feeds this line in
    plain ink and is sealed wherever the app flushes. Nothing is dropped or
    rewritten: the raw stream is what arrived, split where the daemon itself
    marked the edges.

    Streamed replies used to be fed to the RichLog one fragment per write —
    and every write starts a new line, so the answer "12" rendered as two
    rows, "1" then "2" (measured through the TUI on the 6+6 turn,
    2026-10-05). A Static.update() rewrites its own content in place, which
    is the primitive a growing line needs.
    """

    def __init__(self) -> None:
        super().__init__("", id="live")
        self._phase = "wait"          # wait | thinking | answer
        # Plain-text accumulators. Lists of fragments were joined on every
        # draw; a single growing string is the same accumulation at O(1) per
        # fragment, and the redraw (expensive) is coalesced to one per
        # animation frame by _schedule_draw.
        self._thinking: str = ""
        self._answer: str = ""
        self._render_queued: bool = False

    @property
    def pending(self) -> bool:
        """True while anything of this turn is still on the line."""
        return bool(self._thinking or self._answer)

    def feed(self, text: str) -> None:
        """One stream chunk, read for its phase."""
        if not text:
            return
        if REASONING_MARKER in text:
            # A thinking block opens. An answer still on this line is the
            # finished prose of the round before — seal it, then start the
            # thought.
            self._seal_answer()
            self._phase = "thinking"
            after = text.split(REASONING_MARKER, 1)[1]
            if after:
                self._thinking += after
            self._schedule_draw()
            return
        plain = _plain(text)
        if self._phase != "answer" and plain.startswith(DAEMON_PREFIX):
            # The daemon's prefix chunk opens the answer. A thought pending
            # is over: it seals into the transcript first.
            if self._phase == "thinking":
                self._seal_thinking()
            self._phase = "answer"
            tail = plain[len(DAEMON_PREFIX):]
            if tail:
                self._answer += tail
            self._schedule_draw()
            return
        if self._phase == "thinking":
            self._thinking += plain
            self._schedule_draw()
            return
        if plain == "\n" and self._answer:
            # The closing newline of a generation round: the answer is
            # finished prose, and whatever streams next opens a new round.
            self._seal_answer()
            self._phase = "wait"
            return
        self._answer += plain
        self._schedule_draw()

    def flush_into(self, history: "History") -> str:
        """Everything still on this line, in the order the stream produced
        it — thinking first, answer second — sealed into the transcript and
        the line cleared for the next turn. Kept: the answer text."""
        thinking = self._seal_thinking(history)
        answer = self._seal_answer(history)
        self._phase = "wait"
        self._draw()
        return answer

    def _seal_thinking(self, history: "History | None" = None) -> str:
        if not self._thinking:
            return ""
        whole = self._thinking.rstrip("\n")
        self._thinking = ""
        if whole.strip():
            history = history or self._history()
            if history is not None:
                history.say_thinking(whole)
        self._draw()
        return whole

    def _seal_answer(self, history: "History | None" = None) -> str:
        if not self._answer:
            return ""
        whole = self._answer
        self._answer = ""
        if whole.strip():
            history = history or self._history()
            if history is not None:
                history.say_agent(whole)
        self._draw()
        return whole

    def _history(self) -> "History | None":
        """The transcript to seal into, when the app owns one. Nothing here
        drops what arrived: with no transcript to seal into, the text stays
        on the line."""
        try:
            return self.app.query_one(History)
        except Exception:
            return None

    def _draw(self) -> None:
        # A draw (whether scheduled or forced) consumes any pending render.
        self._render_queued = False
        if self._thinking:
            self.update(Text(self._thinking.rstrip("\n"),
                             style="dim"))
        elif self._answer:
            self.update(Text(self._answer.rstrip("\n")))
        else:
            self.update("")

    def _schedule_draw(self) -> None:
        # Coalesce many streamed fragments into a single redraw, taken on the
        # next animation frame instead of once per fragment.
        if self._render_queued:
            return
        self._render_queued = True
        try:
            self.call_after_refresh(self._flush_draw)
        except Exception:
            self._render_queued = False
            self._draw()

    def _flush_draw(self) -> None:
        self._render_queued = False
        self._draw()


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
        line = Text(f"  {SPINNER[self._frame]} ", style=GOLD)
        line.append(self._text, style="dim")
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
