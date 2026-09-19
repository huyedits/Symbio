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
    app = _app()
    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "/sk"
        await pilot.pause()
        app.action_complete()

        assert app.query_one("#prompt").value == "/skill "
        assert app.query_one(Suggestions).display is False


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
async def test_the_figure_sits_on_top_and_the_input_at_the_bottom():
    app = _app()
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()

        logo = app.query_one("#logo")
        history = app.query_one("#history")
        box = app.query_one("#prompt")
        assert logo.region.y < history.region.y < box.region.y


@pytest.mark.asyncio
async def test_a_short_terminal_drops_the_figure_rather_than_wrapping_it():
    """A banner that does not fit is a banner that turns into noise."""
    app = _app()
    async with app.run_test(size=(30, 11)) as pilot:
        await pilot.pause()

        assert app.query_one("#logo").display is False
        assert app.query_one("#prompt").region.width > 0


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
