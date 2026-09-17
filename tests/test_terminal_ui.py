"""The interactive skin must fit the terminal, not leak into other clients."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
import unicodedata

import pytest

from symbio import constants
from symbio.app import chat_style as cs, chat_ui, daemon


CONFIG = {"assistant_name": "Caine", "user_name": "Huy",
          "model_name": "Qwen/Qwen3-8B", "learn": {"enabled": False}}
ANSI = re.compile(r"\033\[[0-?]*[ -/]*[@-~]")


def plain(text):
    return ANSI.sub("", text)


def cells(text):
    return sum(0 if unicodedata.category(c) in ("Mn", "Me") else
               2 if unicodedata.east_asian_width(c) in ("W", "F") else 1
               for c in plain(text))


@pytest.mark.parametrize("width", [1, 12, 23, 24, 32, 40, 58, 80, 96, 200])
@pytest.mark.parametrize("color", [False, True])
def test_panel_fits_real_cells_and_closes_both_edges(width, color):
    config = {**CONFIG, "assistant_name": "開発 Caine\u0301",
              "model_name": "組織/" + "長いモデル" * 25}
    lines = cs.welcome_panel(config, width=width, color=color,
                             workspace="/Users/開発/" + "workspace/" * 30,
                             detail="LoRA on · 12 notes")
    limit = min(width, 96)
    assert len(lines) <= 13
    assert all(cells(line) <= limit for line in lines)
    if width >= 24:
        assert all(cells(line) == limit for line in lines if line)
        assert all(plain(line).endswith("│") for line in lines[2:-1])
    if not color:
        assert "\033" not in "\n".join(lines)


@pytest.mark.parametrize("text,width,expected", [
    ("abcdef", 4, "abc…"), ("ab界z", 4, "ab…"),
    ("界界界", 4, "界…"), ("e\u0301clair", 2, "e\u0301…"),
    ("same", 4, "same"), ("long", 1, "…"),
    ("anything", 0, ""), ("anything", -3, ""),
])
def test_clipping_counts_cells(text, width, expected):
    assert cs._fit(text, width) == expected


def test_config_values_cannot_inject_terminal_controls():
    hostile = "\033]0;bad title\007A\033[31mB\033[0m\n\007\u202eC\u2028"
    assert cs._fit(hostile, 80) == "ABC"


def test_snapshot_paths_show_the_model_not_the_hash():
    path = "/cache/hub/models--mlx-community--Qwen3-8B-4bit/snapshots/deadbeef"
    assert cs.model_label(path) == "mlx-community/Qwen3-8B-4bit"
    assert cs.model_label("/models/local-model/") == "local-model"
    assert cs.model_label("openrouter/provider/model") == "openrouter/provider/model"


def test_workspace_shortens_only_the_actual_home(monkeypatch):
    monkeypatch.setattr(cs.os.path, "expanduser", lambda _: "/home/huy")
    assert cs.workspace_label("/home/huy/project") == "~/project"
    assert cs.workspace_label("/home/huy") == "~"
    assert cs.workspace_label("/home/huy-other/project") == "/home/huy-other/project"


def test_layout_remeasures_after_resize(monkeypatch):
    width = 80
    monkeypatch.setattr(cs.shutil, "get_terminal_size",
                        lambda fallback: os.terminal_size((width, 24)))
    assert cells(cs.prompt_context(CONFIG, color=True)) == 80
    width = 32
    assert cells(cs.prompt_context(CONFIG, color=True)) == 32
    assert cells(cs.welcome_panel(CONFIG, color=True)[1]) == 32


@pytest.mark.parametrize("width", [1, 12, 32, 80, 200])
def test_spinner_never_wraps_into_a_second_row(width):
    text = cs.status_frame("⠋ thinking… " + "token " * 60, width=width, color=True)
    assert cells(text) < min(width, 96)
    assert "\n" not in text and "\r" not in text


def test_terminal_banner_is_compact_and_links_to_full_menu(monkeypatch, tmp_path):
    monkeypatch.setattr(constants, "NOTES_DIR", tmp_path)
    monkeypatch.setattr(cs, "colors_enabled", lambda: True)
    monkeypatch.setattr(cs.shutil, "get_terminal_size",
                        lambda fallback: os.terminal_size((58, 24)))
    lines = []
    chat_ui.print_banner(CONFIG, True, 2048, output_fn=lines.append)
    joined = "\n".join(lines)
    assert "Symbio" in joined and "Qwen/Qwen3-8B" in joined
    assert "/ commands" in joined and "/status" in joined and "/tools" in joined
    assert "/new-skill" not in joined, "the full menu should not swamp startup"
    assert "LoRA on" in joined and "2 KiB training" in joined
    assert len(lines) <= 14
    assert all(cells(line) <= 58 for line in lines)


@pytest.mark.parametrize("name,value", [("NO_COLOR", "1"), ("SYMBIO_NO_COLOR", "1"),
                                        ("TERM", "dumb")])
def test_plain_terminal_policy_retains_legacy_banner(monkeypatch, name, value):
    monkeypatch.setattr(cs.sys.stdout, "isatty", lambda: True)
    monkeypatch.setenv(name, value)
    lines = []
    chat_ui.print_banner(CONFIG, False, 0, output_fn=lines.append)
    joined = "\n".join(lines)
    assert "PERSONAL CHAT-FINETUNE CLI" in joined
    assert "/new-skill" in joined
    assert "\033" not in joined and "╭" not in joined


def test_banner_metadata_is_an_allowlist_not_a_config_dump(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "NOTES_DIR", tmp_path)
    data = chat_ui.banner_data({**CONFIG, "api_key": "SECRET-SENTINEL",
                               "provider": {"token": "SECRET-SENTINEL"}}, True, 0)
    assert set(data) == {"kind", "assistant_name", "user_name", "model_name",
                         "workspace", "detail"}
    assert "SECRET-SENTINEL" not in json.dumps(data)
    assert "\033" not in json.dumps(data)


def test_daemon_panel_uses_client_width_and_running_session_identity(monkeypatch, capsys):
    monkeypatch.setattr(cs, "colors_enabled", lambda: True)
    monkeypatch.setattr(cs.shutil, "get_terminal_size",
                        lambda fallback: os.terminal_size((42, 24)))
    client = daemon.DaemonClient({"assistant_name": "Old", "user_name": "Old user"})
    client._print_output({"type": "output", "text": "plain fallback", "presentation": {
        "kind": "welcome", **CONFIG, "workspace": "~/project", "detail": "base model"}})
    out = capsys.readouterr().out
    assert "plain fallback" not in out and "Symbio" in out
    assert "~/project" in out and "Qwen/Qwen3-8B" in out
    assert all(cells(line) <= 42 for line in out.splitlines())
    assert client._skin_prompt("Huy     : ") == cs.user_prompt()
    assert client.config["assistant_name"] == "Caine"


def test_daemon_plain_client_uses_fallback_unchanged(monkeypatch, capsys):
    monkeypatch.setattr(cs, "colors_enabled", lambda: False)
    msg = {"type": "output", "text": "original\nbanner",
           "presentation": {"kind": "welcome", **CONFIG}}
    daemon.DaemonClient({})._print_output(msg)
    assert capsys.readouterr().out == "original\nbanner\n"


def test_readline_markers_are_only_used_for_input(monkeypatch):
    monkeypatch.setattr(cs.sys.stdin, "isatty", lambda: True)
    monkeypatch.setitem(sys.modules, "readline", SimpleNamespace())
    prompt = cs.user_prompt(color=True)
    marked = cs.readline_prompt(prompt)
    assert "\001\033" in marked and "\002" in marked
    assert marked.replace("\001", "").replace("\002", "") == prompt
    assert "\001" not in prompt and "\002" not in prompt
    monkeypatch.setattr(cs.sys.stdin, "isatty", lambda: False)
    assert cs.readline_prompt(prompt) == prompt
    assert cs.readline_prompt("Huy     : ") == "Huy     : "


@pytest.mark.parametrize("backend", ["GNU readline", "libedit"])
def test_completion_is_live_and_never_completes_arguments(monkeypatch, backend):
    state = SimpleNamespace(buffer="/new-", binds=[], names=["new-skill", "status"])
    fake = SimpleNamespace(
        __doc__=backend, get_line_buffer=lambda: state.buffer,
        set_completer=lambda fn: setattr(state, "complete", fn),
        set_completer_delims=lambda delimiters: setattr(state, "delimiters", delimiters),
        parse_and_bind=state.binds.append,
    )
    monkeypatch.setitem(sys.modules, "readline", fake)
    monkeypatch.setattr(cs.sys.stdin, "isatty", lambda: True)
    assert cs.install_command_completion(lambda: state.names)
    assert state.complete("/new-", 0) == "/new-skill"
    assert state.complete("/new-", 1) is None
    state.names.append("new-ui")
    assert state.complete("/new-", 1) == "/new-ui"
    state.buffer = "/run /new-"
    assert state.complete("/new-", 0) is None
    state.buffer = "ordinary prose"
    assert state.complete("", 0) is None
    assert state.delimiters == " \t\n"
    assert ("bind ^I rl_complete" if backend == "libedit" else "tab: complete") in state.binds


def test_completion_is_optional_without_readline(monkeypatch):
    monkeypatch.setattr(cs.sys.stdin, "isatty", lambda: True)
    monkeypatch.setitem(sys.modules, "readline", None)
    assert cs.install_command_completion(lambda: ["status"]) is False
    assert cs.readline_prompt(cs.user_prompt(color=True)) == cs.user_prompt(color=True)


def test_daemon_completion_includes_commands_created_during_session(tmp_path, monkeypatch):
    from symbio.app import commands

    monkeypatch.setattr(constants, "COMMANDS_DIR", tmp_path / "commands")
    assert "review-ui" not in daemon.DaemonClient._command_names()
    commands.save_command("review-ui", "Review the TUI")
    names = daemon.DaemonClient._command_names()
    assert "review-ui" in names and "status" in names and "new-skill" in names


def test_injected_frontends_never_receive_terminal_prompts(monkeypatch):
    from symbio.app.chat import ChatSession

    session = object.__new__(ChatSession)
    session.config = CONFIG
    session.input_fn = lambda prompt: ""
    monkeypatch.setattr(cs, "colors_enabled", lambda: True)
    assert session.user_prompt() == "Huy     : "
    assert session.assistant_prefix() == "Caine   : "


def test_preview_runs_standalone_without_site_packages_or_config(tmp_path):
    preview = Path(cs.__file__).resolve()
    result = subprocess.run(
        [sys.executable, "-S", str(preview), "--demo", "--width", "48", "--no-color"],
        cwd=tmp_path, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert "Demo · no model or tools are running" in result.stdout
    assert "Symbio" in result.stdout and "\033" not in result.stdout
    assert list(tmp_path.iterdir()) == [], "a UI preview must not create runtime state"
