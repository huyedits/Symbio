"""A Hermes-shaped terminal for the resident model.

One screen: history above, a status line, a persistent input box at the
bottom, and slash-command suggestions that appear as you type. It is a CLIENT
of the daemon — the model, the tools and the history all live there — so
closing this window does not end the session and two of them can watch the
same one.

Resizing is Textual's to handle and is deliberately not second-guessed here:
every width in the layout is a fraction or a dock, so a narrow window reflows
instead of truncating. The one thing that must not move is the input box,
which is docked to the bottom.
"""
from __future__ import annotations

import re
import sys

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Input, Static

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from symbio_tui.protocol import DaemonLink  # noqa: E402
from symbio_tui.widgets import (  # noqa: E402
    HOOK, CommandStrip, Composer, Face, History, StatusLine, Suggestions)

# The tag chat_turn prints once a turn: "  [Mood: curious]".
_MOOD_RE = re.compile(r"\[Mood:\s*([^\]]+)\]")

CSS = """
/* Claude Code's proportions: a quiet header, an elastic transcript, and a
   composer docked to the bottom that never moves at any size. The accent is
   one colour (#d97757) and it is used for exactly three things — the turn
   bullet, the spinner and the box you type in — so what is accented is what
   is live. */
Screen { layout: vertical; background: $surface; }
#face { height: auto; color: #d97757; text-align: left; padding: 1 0 0 2; }
#history { height: 1fr; min-height: 3; border: none; padding: 0 2;
           scrollbar-size-vertical: 1; }
#commands { height: auto; color: $text-muted; padding: 0 2; }
#composer { dock: bottom; height: auto; padding: 0 1; }
#suggestions { max-height: 8; border: round #d97757; display: none; }
#status { height: auto; padding: 0 1; }
/* The box: the border belongs to the ROW so the chevron sits inside it, the
   way a prompt does. An Input that draws its own border can hold nothing but
   text. */
#promptrow { height: auto; border: round #d97757; padding: 0 1; }
#chevron { width: 2; height: 1; color: #d97757; }
#prompt { border: none; background: transparent; padding: 0; height: 1;
          width: 1fr; }
#prompt:focus { border: none; background: transparent; }
#hints { height: auto; color: $text-muted; padding: 0 1; }
"""


class SymbioTUI(App):
    CSS = CSS
    TITLE = "Symbio"
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear", "Clear"),
        Binding("pageup", "scroll_back", "Scroll back"),
        Binding("pagedown", "scroll_forward", "Scroll on"),
    ]

    def __init__(self, command_names: list[str] | None = None,
                 link_factory=DaemonLink, inline_lines: int = 14):
        super().__init__()
        self._inline_lines = inline_lines
        if command_names is None:
            from symbio.app.daemon import DaemonClient

            command_names = DaemonClient._command_names()
        self._names = command_names
        self._link_factory = link_factory
        self.link: DaemonLink | None = None
        self._awaiting_input = False
        self._pending_confirm = False

    # ---- layout ----------------------------------------------------------
    def compose(self) -> ComposeResult:
        yield Face()
        yield History()
        yield CommandStrip(self._names)
        yield Composer(self._names)

    def on_mount(self) -> None:
        if self.is_inline:
            # The panel is a fixed slice of the terminal; everything above it
            # stays in the scrollback the terminal already owns.
            self.screen.styles.height = self._inline_lines
        self.query_one("#prompt", Input).focus()
        self.link = self._link_factory(self._on_frame, self._on_close)
        problem = self.link.connect()
        history = self.query_one(History)
        if problem:
            history.say_tool(problem)
            self._set_status("not connected")
        else:
            self._set_status("connected to the resident model")

    # ---- frames from the daemon ------------------------------------------
    def _on_frame(self, msg: dict) -> None:
        self.call_from_thread(self._handle, msg)

    def _on_close(self, why: str) -> None:
        self.call_from_thread(self._set_status, f"disconnected — {why}")

    def _handle(self, msg: dict) -> None:
        history = self.query_one(History)
        kind = msg.get("type")
        if kind == "output":
            text = msg.get("text", "")
            mood = _MOOD_RE.search(text or "")
            if mood:
                # The one line a turn emits about how it read the exchange.
                self.query_one(Face).set_mood(mood.group(1))
            # The mood tag is what the face is drawn from; printing it as
            # well would say the same thing twice, once as a face and once as
            # a machine tag in the middle of the prose.
            # rstrip only: the LEADING whitespace is what marks a line as a
            # tool result rather than as the assistant speaking, and stripping
            # it here would send every indented frame through as prose.
            self._show_output(history, _MOOD_RE.sub("", text).rstrip())
            self._set_status("")
        elif kind == "stream":
            history.say_inline(msg.get("text", ""))
        elif kind == "status":
            # Status frames are the daemon working: spinner on.
            self._set_status(msg.get("text") or "", busy=True)
        elif kind == "input_prompt":
            self._awaiting_input = True
            self._set_status("waiting for you")
        elif kind == "confirm":
            self._pending_confirm = True
            history.say_tool(msg.get("prompt", "Allow this?"))
            self._set_status("approve? y / n")
        elif kind == "done":
            self._set_status("session ended")
            self.query_one(Face).set_mood("offline")

    def _show_output(self, history: History, text: str) -> None:
        """Route one frame to the mark that describes it.

        The daemon sends a turn's prose and its tool output down the same
        channel, so the shape of the line is what tells them apart: anything
        chat_style has already coloured keeps its own ANSI untouched, an
        indented or hooked line is a consequence, and the rest is the
        assistant speaking and gets the bullet. Guessing wrong costs a mark,
        not a message — nothing here drops or rewrites what arrived.
        """
        if not text:
            return
        if "\x1b[" in text:
            history.say(text)
        elif text.startswith((" ", "\t", HOOK)) or text.lstrip().startswith(HOOK):
            history.say_tool(text.strip())
        else:
            history.say_agent(text)

    def _set_status(self, text: str, busy: bool = False) -> None:
        self.query_one(StatusLine).show(text, busy=busy)

    # ---- input -----------------------------------------------------------
    def on_input_changed(self, event: Input.Changed) -> None:
        self.query_one(Suggestions).offer(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        self.query_one(Suggestions).display = False
        if not text:
            return
        if self._pending_confirm:
            self._pending_confirm = False
            answer = text.lower() in ("y", "yes")
            self._send({"type": "confirm", "answer": answer})
            self._set_status("approved" if answer else "declined")
            return
        self.query_one(History).say_user(text)
        self._send({"type": "input", "text": text})
        self._awaiting_input = False
        self._set_status("Thinking…", busy=True)
        self.query_one(Face).set_mood("working")

    def _send(self, message: dict) -> None:
        if self.link is not None and self.link.connected:
            self.link.send(message)
        else:
            self.query_one(History).say_tool("Not connected to a daemon.")

    # ---- actions ---------------------------------------------------------
    def action_complete(self) -> None:
        chosen = self.query_one(Suggestions).current
        if chosen:
            box = self.query_one("#prompt", Input)
            box.value = f"/{chosen} "
            box.cursor_position = len(box.value)
            self.query_one(Suggestions).display = False

    def action_suggest_next(self) -> None:
        self.query_one(Suggestions).step(1)

    def action_suggest_prev(self) -> None:
        self.query_one(Suggestions).step(-1)

    def action_scroll_back(self) -> None:
        """Page the transcript without taking focus off the input box."""
        self.query_one(History).scroll_page_up()

    def action_scroll_forward(self) -> None:
        self.query_one(History).scroll_page_down()

    def action_clear(self) -> None:
        self.query_one(History).clear()

    def action_quit(self) -> None:
        if self.link is not None:
            self.link.close()
        self.exit()


def main() -> int:
    """Inline by default, which is the Claude Code shape rather than a
    full-screen app.

    An alternate-buffer TUI takes the whole terminal and with it your
    scrollback and your native selection: copying a stack trace out of it goes
    through the app's own selection instead of the terminal's. Claude Code does
    not do that, and neither does the chat_style path in this repo, which says
    so in as many words. Inline mode keeps the composer live at the bottom and
    leaves everything above it in ordinary scrollback.

    --fullscreen is still there for a dedicated window, where owning the screen
    is the point.
    """
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--fullscreen", action="store_true",
                    help="take over the terminal instead of running inline")
    ap.add_argument("--lines", type=int, default=14,
                    help="how many lines the inline panel occupies")
    args = ap.parse_args()
    app = SymbioTUI(inline_lines=args.lines)
    app.run(inline=not args.fullscreen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
