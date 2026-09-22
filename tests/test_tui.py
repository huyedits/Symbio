#!/usr/bin/env python3
"""The terminal UI, driven headless.

It is a CLIENT of the daemon — the model, the tools and the history live there
— so everything here runs against a fake link and never needs a model, a
socket or a terminal.
"""
import pytest

pytest.importorskip("textual", reason="the TUI is an optional extra")

from symbio_tui.app import SymbioTUI  # noqa: E402
from symbio_tui.widgets import History, Suggestions  # noqa: E402

NAMES = ["status", "train", "tools", "quit", "skill"]


class FakeLink:
    """Stands in for the Unix socket. Records what the UI sends."""

    def __init__(self, on_frame, on_close):
        self.on_frame, self.on_close = on_frame, on_close
        self.sent: list[dict] = []
        self.connected = True
        self.closed = False

    def connect(self) -> str:
        return ""

    def send(self, message):
        self.sent.append(message)

    def close(self):
        self.closed = True


def _app():
    return SymbioTUI(command_names=NAMES, link_factory=FakeLink)


@pytest.mark.asyncio
async def test_the_input_box_has_focus_the_moment_it_opens():
    """A persistent input box you have to click into is not persistent."""
    app = _app()
    async with app.run_test():
        assert app.focused is app.query_one("#prompt")


@pytest.mark.asyncio
async def test_typing_a_slash_offers_matching_commands():
    app = _app()
    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "/st"
        await pilot.pause()

        suggestions = app.query_one(Suggestions)
        assert suggestions.display is True
        assert suggestions.current == "status"


@pytest.mark.asyncio
async def test_ordinary_text_offers_nothing():
    app = _app()
    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "how much disk is free"
        await pilot.pause()

        assert app.query_one(Suggestions).display is False


@pytest.mark.asyncio
async def test_tab_completes_the_highlighted_command():
    """PRESSING tab, not calling the action. The first version of this test
    called action_complete() directly and passed while the real keyboard did
    nothing: Input handles tab itself as focus traversal, so the app-level
    binding never fired. Found by driving it in a pty."""
    app = _app()
    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "/sk"
        await pilot.pause()
        await pilot.press("tab")

        assert app.query_one("#prompt").value == "/skill "
        assert app.query_one(Suggestions).display is False


@pytest.mark.asyncio
async def test_down_and_up_walk_the_suggestions_from_the_keyboard():
    app = _app()
    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "/s"
        await pilot.pause()
        first = app.query_one(Suggestions).current

        await pilot.press("down")
        second = app.query_one(Suggestions).current
        await pilot.press("up")

        assert second != first
        assert app.query_one(Suggestions).current == first


@pytest.mark.asyncio
async def test_escape_dismisses_the_suggestions():
    app = _app()
    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "/s"
        await pilot.pause()

        await pilot.press("escape")

        assert app.query_one(Suggestions).display is False


@pytest.mark.asyncio
async def test_tab_is_left_alone_when_nothing_is_suggested():
    """With no menu open, tab belongs to the terminal's own focus traversal."""
    app = _app()
    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "ordinary text"
        await pilot.pause()

        await pilot.press("tab")

        assert app.query_one("#prompt").value == "ordinary text"


@pytest.mark.asyncio
async def test_submitting_sends_input_to_the_daemon():
    app = _app()
    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "what is 2+2"
        await pilot.press("enter")

        assert app.link.sent == [{"type": "input", "text": "what is 2+2"}]
        assert app.query_one("#prompt").value == ""


@pytest.mark.asyncio
async def test_a_confirm_frame_turns_the_next_line_into_an_answer():
    """The gate asks on the daemon side; the reply has to be an answer, not a
    new turn, or the approval prompt is silently answered with a question."""
    app = _app()
    async with app.run_test() as pilot:
        app._handle({"type": "confirm", "prompt": "Run `rm -rf /`?"})
        app.query_one("#prompt").value = "n"
        await pilot.press("enter")

        assert app.link.sent == [{"type": "confirm", "answer": False}]


@pytest.mark.asyncio
async def test_a_yes_is_an_approval():
    app = _app()
    async with app.run_test() as pilot:
        app._handle({"type": "confirm", "prompt": "Allow?"})
        app.query_one("#prompt").value = "y"
        await pilot.press("enter")

        assert app.link.sent == [{"type": "confirm", "answer": True}]


@pytest.mark.asyncio
async def test_status_frames_land_on_the_status_line_not_in_the_history():
    """A spinner drawn into the scrollback is a spinner that scrolls."""
    app = _app()
    async with app.run_test() as pilot:
        app._handle({"type": "status", "text": "thinking… (6s)"})
        await pilot.pause()

        assert "thinking" in str(app.query_one("#status").content)


@pytest.mark.asyncio
async def test_output_frames_land_in_the_history():
    app = _app()
    async with app.run_test() as pilot:
        app._handle({"type": "output", "text": "line one\nline two"})
        await pilot.pause()

        assert app.query_one(History).lines


@pytest.mark.asyncio
async def test_a_narrow_window_still_has_its_input_box():
    """Resizing is Textual's job; what this checks is that the layout does not
    depend on width to keep the one widget that must never move."""
    app = _app()
    async with app.run_test(size=(40, 12)) as pilot:
        await pilot.pause()

        box = app.query_one("#prompt")
        assert box.region.width > 0 and box.region.y > 0


@pytest.mark.asyncio
async def test_quitting_closes_the_socket():
    app = _app()
    async with app.run_test():
        link = app.link
        app.action_quit()

    assert link.closed is True


# ---- the shape of the screen ----

@pytest.mark.asyncio
async def test_the_face_sits_on_top_and_the_input_at_the_bottom():
    app = _app()
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()

        face = app.query_one("#face")
        history = app.query_one("#history")
        box = app.query_one("#prompt")
        assert face.region.y < history.region.y < box.region.y


@pytest.mark.asyncio
async def test_a_narrow_terminal_keeps_the_face_and_drops_the_label():
    """The face is the part worth the room; the word is not."""
    app = _app()
    async with app.run_test(size=(24, 12)) as pilot:
        await pilot.pause()

        drawn = str(app.query_one("#face").content)
        assert ":|" in drawn and "Symbio" not in drawn


@pytest.mark.asyncio
async def test_the_face_follows_the_mood_the_session_emits():
    """chat_turn prints one "[Mood: tag]" a turn; the face is that tag, not a
    decoration chosen at random."""
    from symbio_tui.widgets import Face

    app = _app()
    async with app.run_test(size=(80, 30)) as pilot:
        app._handle({"type": "output", "text": "  [Mood: angry]"})
        await pilot.pause()
        assert ">:(" in str(app.query_one(Face).content)

        app._handle({"type": "output", "text": "  [Mood: happy]"})
        await pilot.pause()
        assert ":)" in str(app.query_one(Face).content)


@pytest.mark.asyncio
async def test_an_unknown_mood_is_neutral_rather_than_a_guess():
    """A tag added to chat_text later must read as no strong feeling, not as
    whichever emotion happens to sort first."""
    from symbio_tui.widgets import Face

    app = _app()
    async with app.run_test(size=(80, 30)) as pilot:
        app._handle({"type": "output", "text": "  [Mood: electrified]"})
        await pilot.pause()

        assert ":|" in str(app.query_one(Face).content)


@pytest.mark.asyncio
async def test_sending_a_turn_shows_the_working_face():
    from symbio_tui.widgets import Face

    app = _app()
    async with app.run_test(size=(80, 30)) as pilot:
        app.query_one("#prompt").value = "hello"
        await pilot.press("enter")
        await pilot.pause()

        assert app.query_one(Face).mood == "working"


@pytest.mark.asyncio
async def test_the_commands_are_on_the_screen_not_only_in_a_menu():
    app = _app()
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()

        assert "/status" in str(app.query_one("#commands").content)


@pytest.mark.asyncio
async def test_the_transcript_scrolls_without_losing_the_input_box():
    """A conversation you cannot go back through is one you have to re-ask."""
    app = _app()
    async with app.run_test(size=(80, 24)) as pilot:
        for i in range(200):
            app._handle({"type": "output", "text": f"line {i}"})
        await pilot.pause()
        history = app.query_one(History)
        history.scroll_end(animate=False)
        await pilot.pause()
        bottom = history.scroll_offset.y

        app.action_scroll_back()
        await pilot.pause()

        assert history.scroll_offset.y < bottom
        assert app.focused is app.query_one("#prompt")


@pytest.mark.asyncio
async def test_the_face_sits_top_left():
    app = _app()
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()

        face = app.query_one("#face")
        assert face.region.x <= 1 and face.region.y <= 1


# ---- reachable from the command line ----

def _args(**kw):
    from argparse import Namespace

    kw.setdefault("no_tui", False)
    kw.setdefault("tui", False)
    return Namespace(**kw)


def test_a_terminal_gets_the_panelled_interface(monkeypatch):
    from symbio.app import cli

    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True, raising=False)

    assert cli._wants_tui(_args(), "chat", {}) is True


def test_a_pipe_keeps_the_plain_chat(monkeypatch):
    """A new interface must not be the reason a scripted run stops working."""
    from symbio.app import cli

    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: False, raising=False)

    assert cli._wants_tui(_args(), "chat", {}) is False


def test_no_color_and_dumb_terminals_keep_the_plain_chat(monkeypatch):
    from symbio.app import cli

    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True, raising=False)
    monkeypatch.setenv("TERM", "dumb")
    assert cli._wants_tui(_args(), "chat", {}) is False
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("NO_COLOR", "1")
    assert cli._wants_tui(_args(), "chat", {}) is False


def test_the_flag_and_the_config_key_both_turn_it_off(monkeypatch):
    from symbio.app import cli

    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True, raising=False)
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True, raising=False)

    assert cli._wants_tui(_args(no_tui=True), "chat", {}) is False
    assert cli._wants_tui(_args(), "chat", {"agent": {"tui": False}}) is False


def test_symb_tui_asks_for_it_even_off_a_terminal(monkeypatch):
    """`symb tui` is a request, not a guess — the only thing that overrules it
    is the library being absent."""
    from symbio.app import cli

    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: False, raising=False)
    monkeypatch.setenv("TERM", "dumb")

    assert cli._wants_tui(_args(), "tui", {"agent": {"tui": False}}) is True


# ---- the shape Claude Code reads in ----

def _lines(app) -> list[str]:
    """The transcript as plain text, however each line was styled."""
    return [line.text if hasattr(line, "text") else str(line)
            for line in app.query_one(History).lines]


@pytest.mark.asyncio
async def test_what_you_typed_is_echoed_under_a_chevron():
    app = _app()
    async with app.run_test(size=(80, 30)) as pilot:
        app.query_one("#prompt").value = "post this for me"
        await pilot.press("enter")
        await pilot.pause()

        assert any(line.startswith("> post this for me") for line in _lines(app))


@pytest.mark.asyncio
async def test_a_turn_is_marked_with_the_bullet():
    """One mark per event, so a screenful of scrollback can be skimmed for the
    one you are looking for instead of read."""
    from symbio_tui.widgets import BULLET

    app = _app()
    async with app.run_test(size=(80, 30)) as pilot:
        app._handle({"type": "output", "text": "I posted it."})
        await pilot.pause()

        assert any(line.startswith(f"{BULLET} I posted it.") for line in _lines(app))


@pytest.mark.asyncio
async def test_an_indented_frame_reads_as_a_result_not_as_speech():
    """The daemon sends prose and tool output down one channel; the shape of
    the line is what tells them apart."""
    from symbio_tui.widgets import HOOK

    app = _app()
    async with app.run_test(size=(80, 30)) as pilot:
        app._handle({"type": "output", "text": "  (180, 120)  Submit button"})
        await pilot.pause()

        assert any(HOOK in line for line in _lines(app))


@pytest.mark.asyncio
async def test_a_coloured_frame_keeps_its_own_ansi():
    """chat_style's skin is escape codes. Re-marking it would be a second
    representation of the same line that can drift from the real one."""
    app = _app()
    async with app.run_test(size=(80, 30)) as pilot:
        app._handle({"type": "output", "text": "\x1b[36mthinking… (6s)\x1b[0m"})
        await pilot.pause()

        assert any("thinking" in line for line in _lines(app))


@pytest.mark.asyncio
async def test_the_mood_tag_is_the_face_and_not_also_a_line():
    """Saying it twice — once as a face, once as a machine tag mid-prose — is
    the tag leaking into the conversation."""
    app = _app()
    async with app.run_test(size=(80, 30)) as pilot:
        app._handle({"type": "output", "text": "Done.\n  [Mood: happy]"})
        await pilot.pause()

        assert app.query_one("#face").mood == "happy"
        assert not any("[Mood:" in line for line in _lines(app))


@pytest.mark.asyncio
async def test_a_status_frame_spins_and_an_idle_one_does_not():
    """A status that cannot be told apart from a frozen one is the thing
    people reach for ctrl-c over."""
    from symbio_tui.widgets import SPINNER, StatusLine

    app = _app()
    async with app.run_test(size=(80, 30)) as pilot:
        app._handle({"type": "status", "text": "looking (4s)"})
        await pilot.pause()
        busy = str(app.query_one(StatusLine).content)

        app._handle({"type": "output", "text": "I looked."})
        await pilot.pause()

        assert any(frame in busy for frame in SPINNER)
        assert "looking (4s)" in busy
        assert str(app.query_one(StatusLine).content).strip() == ""


@pytest.mark.asyncio
async def test_the_chevron_sits_inside_the_box_with_the_input():
    """A prompt drawn outside the border is a decoration; inside it, it is a
    prompt. An Input that draws its own border cannot hold one."""
    app = _app()
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()

        row = app.query_one("#promptrow")
        chevron = app.query_one("#chevron")
        box = app.query_one("#prompt")

        assert chevron.region.x < box.region.x
        assert row.region.y <= chevron.region.y
        assert row.region.height >= 3, "the row is what carries the border"


@pytest.mark.asyncio
async def test_the_keys_are_on_the_screen_under_the_box():
    app = _app()
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()

        hints = app.query_one("#hints")
        assert "ctrl+c" in str(hints.content)
        assert hints.region.y > app.query_one("#prompt").region.y
