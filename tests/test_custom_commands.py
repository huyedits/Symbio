"""User-defined slash commands, and the terminal affordances around them.

A command is a file in commands/ whose body is a prompt. The point of the
feature is recall — a name you can type instead of a paragraph you have to
remember — so the tests cover the three things that make a name usable: it
runs, it completes, and mistyping it suggests the right one.
"""

import json
import os
from pathlib import Path

import pytest

from symbio import constants
from symbio.app import commands
from symbio.app.chat_constants import BUILTIN_COMMAND_NAMES


@pytest.fixture
def scratch_commands(tmp_path, monkeypatch):
    """Point COMMANDS_DIR at a temp directory for the duration of a test."""
    directory = tmp_path / "commands"
    monkeypatch.setattr(constants, "COMMANDS_DIR", directory)
    return directory


def test_a_saved_command_comes_back_with_its_body(scratch_commands):
    commands.save_command("standup", "Tell me what moved and what is blocked.",
                          description="Daily standup")
    cmd = commands.get_command("standup")
    assert cmd is not None
    assert cmd.description == "Daily standup"
    assert "what is blocked" in cmd.body
    assert cmd.author == "user"


def test_arguments_land_where_the_body_says(scratch_commands):
    commands.save_command("review", "Review $ARGUMENTS and tell me what breaks.")
    cmd = commands.get_command("review")
    assert commands.render(cmd, "chat_turn.py") == \
        "Review chat_turn.py and tell me what breaks."


def test_positional_arguments_work_and_missing_ones_vanish(scratch_commands):
    commands.save_command("diff", "Compare $1 with $2.")
    cmd = commands.get_command("diff")
    assert commands.render(cmd, "a.py b.py") == "Compare a.py with b.py."
    assert commands.render(cmd, "a.py") == "Compare a.py with ."


def test_arguments_are_never_silently_dropped(scratch_commands):
    """A body with no placeholder still has to carry what the user typed —
    losing it is the one outcome they cannot debug from the output."""
    commands.save_command("notes", "Summarize my notes.")
    cmd = commands.get_command("notes")
    assert "from last week" in commands.render(cmd, "from last week")


def test_a_command_name_cannot_be_a_path(scratch_commands):
    for bad in ("../escape", "has space", "", "a/b", "sub/../../etc"):
        with pytest.raises(ValueError):
            commands.save_command(bad, "body")


def test_a_name_is_normalized_rather_than_rejected(scratch_commands):
    """The leading slash and the capital are what a person types, not a
    mistake to punish them for."""
    path = commands.save_command("/Standup", "body")
    assert path.stem == "standup"
    assert commands.get_command("/STANDUP") is not None


def test_a_command_needs_a_body(scratch_commands):
    with pytest.raises(ValueError):
        commands.save_command("empty", "   ")


def test_deleting_a_command_removes_the_file(scratch_commands):
    commands.save_command("temp", "body")
    assert commands.delete_command("temp") is True
    assert commands.get_command("temp") is None
    assert commands.delete_command("temp") is False


def test_the_author_is_recorded_so_a_listing_can_say_who_wrote_it(scratch_commands):
    """The assistant can write commands through save_command. The user should
    never find a command in their own list without being able to see that they
    did not write it."""
    commands.save_command("triage", "Do the thing.", author="assistant")
    assert commands.get_command("triage").author == "assistant"


def test_a_malformed_file_costs_only_itself(scratch_commands):
    scratch_commands.mkdir(parents=True)
    (scratch_commands / "good.md").write_text("---\nname: good\n---\n\nbody\n")
    (scratch_commands / "Bad Name.md").write_text("no frontmatter, bad stem")
    names = [c.name for c in commands.load_commands()]
    assert "good" in names


def test_suggestions_catch_the_half_typed_and_the_mistyped():
    known = list(BUILTIN_COMMAND_NAMES)
    assert "status" in commands.suggest("stat", known)
    assert "status" in commands.suggest("staus", known)
    assert commands.suggest("zzzzzz", known) == []


def test_builtin_names_are_all_actually_handled():
    """The table drives the menu, the completer and the suggestions, so a name
    in it that the handler does not know is a command the terminal offers and
    then refuses."""
    source = Path("symbio/app/chat_commands.py").read_text(encoding="utf-8")
    missing = [n for n in BUILTIN_COMMAND_NAMES if f'"/{n}' not in source]
    # /quit's aliases and /help's are written as tuples; check those by hand.
    missing = [n for n in missing if n not in ("quit",)]
    assert not missing, f"listed but not handled: {missing}"


def test_save_command_is_a_real_tool_the_model_can_reach():
    from symbio.app import tooling

    assert any(t["name"] == "save_command" for t in tooling._TOOLS)
    assert tooling.tool_family("save_command") == "memory"


# ------------------------------------------------------------ terminal width

def test_listings_fit_the_window_they_are_given():
    """Every listing is read in a terminal the user chose the size of. A line
    wider than that wraps at an arbitrary column and the column layout is gone."""
    from symbio.app.chat_ui import two_column, wrapped_list

    for width in (110, 80, 60, 45, 40):
        lines = two_column("/commands [new|rm|show] ...",
                           "Your own slash commands, saved in commands/",
                           width=width)
        lines += wrapped_list(
            ["browser_click", "browser_click_at", "browser_close",
             "browser_get_text", "browser_open", "browser_press"], width=width)
        for line in lines:
            assert len(line) <= width, (width, len(line), line)


def test_a_long_unbreakable_token_is_left_whole():
    """A model path or a URL split across lines cannot be copied, and a command
    name split on its hyphen reads as two commands that do not exist."""
    from symbio.app.chat_ui import two_column, wrapped_list

    path = "/Users/x/models/hub/models--Qwen--Qwen3-14B-MLX-4bit/snapshots/abc123"
    assert path in "\n".join(two_column("Model  :", path, width=50))
    listed = "\n".join(wrapped_list(["/skill-adapters", "/index-notes"], width=24))
    assert "/skill-adapters" in listed and "/index-notes" in listed


def test_width_is_clamped_to_something_readable(monkeypatch):
    import shutil as _shutil

    from symbio.app import chat_ui

    monkeypatch.setattr(_shutil, "get_terminal_size",
                        lambda fallback=(80, 24): os.terminal_size((12, 24)))
    assert chat_ui.term_width() >= 40
    monkeypatch.setattr(_shutil, "get_terminal_size",
                        lambda fallback=(80, 24): os.terminal_size((400, 24)))
    assert chat_ui.term_width() <= 110


def test_the_banner_lists_every_command_and_stays_in_the_window():
    """The banner's command list used to be four hand-maintained strings; it is
    generated from the table now, so a new command appears in it for free."""
    from symbio.app import chat_ui
    from symbio.app import config as app_config

    lines = []
    original = chat_ui.term_width
    chat_ui.term_width = lambda default=80: 58
    try:
        chat_ui.print_banner(app_config.load_config(), True, 0,
                             output_fn=lines.append)
    finally:
        chat_ui.term_width = original
    joined = "\n".join(lines)
    for name in ("status", "commands", "tools", "new-skill"):
        assert f"/{name}" in joined, name
    for line in joined.splitlines():
        # The model path is one deliberate exception: unbreakable, and more
        # useful overhanging than chopped.
        if "models" in line or "Model" in line:
            continue
        assert len(line) <= 58, (len(line), line)
