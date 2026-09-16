#!/usr/bin/env python3
"""The call has to match the contract the prompt handed out.

Every tool in the catalog advertises a JSON schema, and until 2026-09-16
nothing checked a call against it. The dispatcher reads what it wants with
`params.get("cmd", "")`, so a call with the argument misspelled was not an
error — it was a call with an empty command, and what came back was whatever
an empty argument produces. Live, `save_command` with `nmae` reached the
implementation and answered "Invalid command name ''", which tells the model
nothing about the name it actually got wrong.

That is the `reflex` mistake kind in learn.py — a wrong SHAPE rather than a
wrong fact — and the cure for a wrong shape is being shown the right one.
Borrowed from Atomic Agents, whose discipline is that every component declares
its input and output schema and nothing is wired by name and hope.
"""
import json
import os
import re

import pytest

from symbio.app import chat_tools, tooling


def _session(config=None):
    class S(chat_tools.ToolsMixin):
        def __init__(self):
            self.config = config if config is not None else {}
            self.enabled_groups = None
            self.output_fn = lambda *a, **k: None
    return S()


# ---- what the check refuses, and what it says ----

def test_a_missing_required_argument_is_refused_with_the_schema():
    """Naming the fault without the schema buys one round and spends the next
    on tool_docs."""
    ok, why = tooling.validate_arguments("run_command", {})

    assert ok is False
    assert "`cmd`" in why
    assert '"required":["cmd"]' in why.replace(" ", "")


def test_a_misspelled_argument_names_the_ones_that_exist():
    ok, why = tooling.validate_arguments("run_command", {"command": "ls"})

    assert ok is False
    assert "no argument `command`" in why and "it takes `cmd`" in why


def test_a_correct_call_passes_silently():
    assert tooling.validate_arguments("run_command", {"cmd": "ls"}) == (True, "")


def test_the_internal_name_is_what_the_dispatcher_actually_passes():
    """The catalog is written in advertised names (`terminal`) and the
    dispatcher works in internal ones (`run_command`). Looking up only one of
    them leaves the check running on nothing at all — a guard that is dead in
    the shipped config while its tests pass."""
    assert tooling.validate_arguments("terminal", {})[0] is False
    assert tooling.validate_arguments("run_command", {})[0] is False


def test_a_tool_outside_the_catalog_is_not_this_checks_business():
    """Unknown names are answered by nearest_tools, which has more to say."""
    assert tooling.validate_arguments("no_such_tool", {"x": 1}) == (True, "")


def test_arguments_that_are_not_an_object_are_refused():
    ok, why = tooling.validate_arguments("run_command", "ls")

    assert ok is False and "JSON object" in why


# ---- what it deliberately tolerates ----

def test_a_number_where_a_string_belongs_is_not_worth_a_round_trip():
    """The dispatcher's own str() does exactly what was meant."""
    assert tooling.validate_arguments("run_command", {"cmd": 5})[0] is True


def test_an_object_where_a_command_line_belongs_is_a_different_mistake():
    ok, why = tooling.validate_arguments("run_command", {"cmd": {"run": "ls"}})

    assert ok is False and "string is expected" in why


@pytest.mark.parametrize("tool,params", [
    ("delete_note", {"query": "the proxy note"}),
    ("delete_note", {"name": "Proxy Info"}),
    ("fill_form", {"values": {"#title": "a"}}),
    ("fill_form", {"form": {"#title": "a"}}),
    ("submit_form", {"selector": "#go"}),
])
def test_the_second_spellings_the_dispatcher_honours_are_accepted(tool, params):
    """These work today. A check that refused them would be a regression
    dressed as rigour."""
    assert tooling.validate_arguments(tool, params) == (True, "")


def test_an_alias_does_not_make_the_argument_optional():
    assert tooling.validate_arguments("delete_note", {})[0] is False
    assert tooling.validate_arguments("submit_form", {})[0] is False


# ---- the three modes, each proven to fire ----

def test_the_dispatcher_refuses_the_call_and_hands_back_the_shape():
    out = _session()._dispatch_tool("run_command", {"command": "ls"})

    assert "no argument `command`" in out
    assert "Schema:" in out


def test_switching_it_off_restores_the_old_silent_coercion(tmp_path, monkeypatch):
    """What the old behaviour actually was: the typo reached the tool, which
    answered about the empty string it had been handed."""
    monkeypatch.delenv("SYMBIO_TOOL_ARGS", raising=False)
    out = _session({"agent": {"validate_tool_arguments": "off"}})._dispatch_tool(
        "save_command", {"nmae": "x"})

    assert "no argument" not in out


def test_audit_mode_records_what_it_would_have_refused(tmp_path, monkeypatch):
    """A guard switched on without this measurement is how three of them ended
    up dead in the shipped config with their tests passing."""
    from symbio import constants

    monkeypatch.setattr(constants, "LOG_DIR", tmp_path)
    monkeypatch.setenv("SYMBIO_TOOL_ARGS", "audit")

    out = _session()._dispatch_tool("save_command", {"nmae": "x"})

    logged = (tmp_path / "tool_argument_audit.jsonl").read_text(encoding="utf-8")
    assert json.loads(logged)["tool"] == "save_command"
    assert "no argument" not in out, "audit must let the call through"


def test_the_mode_reads_from_config_and_from_the_environment(monkeypatch):
    monkeypatch.delenv("SYMBIO_TOOL_ARGS", raising=False)
    assert chat_tools._argument_check_mode(None) == "on"
    assert chat_tools._argument_check_mode(
        {"agent": {"validate_tool_arguments": "off"}}) == "off"
    monkeypatch.setenv("SYMBIO_TOOL_ARGS", "audit")
    assert chat_tools._argument_check_mode(
        {"agent": {"validate_tool_arguments": "off"}}) == "audit", (
        "the env override exists to measure a live install without editing it")


# ---- the contract cannot drift without a failure ----

def _dispatcher_source() -> str:
    from pathlib import Path

    src = Path(chat_tools.__file__).read_text(encoding="utf-8")
    start = src.index("def _dispatch_tool")
    return src[start:src.index("\n    def ", start + 10)]


def test_the_dispatcher_reads_no_argument_the_catalog_never_advertises():
    """The scan that found the aliases, kept as the thing that catches the
    next one. An argument the dispatcher honours and the catalog does not name
    is a capability no model can reach — or, once the check is on, a working
    call refused."""
    read = set(re.findall(r'params\.get\(\s*"([a-z_0-9]+)"', _dispatcher_source()))
    declared = set()
    for spec in tooling.tool_schemas():
        declared |= set((spec.get("parameters") or {}).get("properties") or {})
    for spellings in tooling._ARGUMENT_ALIASES.values():
        for names in spellings.values():
            declared.update(names)

    assert read - declared == set(), (
        "add these to tooling._ARGUMENT_ALIASES, or advertise them")


def test_every_alias_points_at_an_argument_that_really_exists():
    """A typo here silently re-opens the hole it was written to close."""
    by_name = {t["name"]: t for t in tooling.tool_schemas()}
    for tool, spellings in tooling._ARGUMENT_ALIASES.items():
        assert tool in by_name, tool
        properties = (by_name[tool].get("parameters") or {}).get("properties") or {}
        for declared in spellings:
            assert declared in properties, (tool, declared)


def test_every_catalog_schema_is_one_the_check_can_read():
    for spec in tooling.tool_schemas():
        params = spec.get("parameters") or {}
        assert params.get("type") == "object", spec["name"]
        properties = params.get("properties") or {}
        for key in params.get("required") or []:
            assert key in properties, (spec["name"], key)
        for key, prop in properties.items():
            assert isinstance(prop, dict), (spec["name"], key)


# ---- the other loop reads the same contract ----
#
# There are two dispatchers over one advertised catalog: ToolsMixin (chat, the
# daemon, Telegram) and symbio/tools.py (the AIAgent loop). A tool added to the
# catalog is advertised to BOTH and runnable in whichever got the branch, and
# where both have a branch they did not have to agree about the arguments. They
# did not: the catalog told the model `browser_click` takes `target`, and this
# runner read `selector` and `text`. The call then succeeded against the empty
# string, which is the worst of the three possible outcomes.

class _RecordingBrowser:
    def __init__(self):
        self.calls = []

    def click(self, selector="", text=""):
        self.calls.append(("click", selector, text))
        return "clicked"

    def type_text(self, text="", selector="", press_enter=False):
        self.calls.append(("type", text, selector, press_enter))
        return "typed"


def _agent_with_browser():
    from types import SimpleNamespace

    return SimpleNamespace(_browser_session=_RecordingBrowser(), config={},
                           history=[], browser=None)


def _run(agent, name, args):
    from symbio import tools as registry

    entry = next(t for t in registry.build_tool_registry(agent) if t["name"] == name)
    return entry["run"](args)


@pytest.mark.parametrize("target,expected", [
    ("Post", ("click", "", "Post")),
    ("#tweet-button", ("click", "#tweet-button", "")),
])
def test_the_agent_loop_clicks_by_the_name_the_catalog_advertises(target, expected):
    agent = _agent_with_browser()

    _run(agent, "browser_click", {"target": target})

    assert agent._browser_session.calls == [expected]


def test_the_older_spellings_still_work():
    agent = _agent_with_browser()

    _run(agent, "browser_click", {"selector": "#go"})

    assert agent._browser_session.calls == [("click", "#go", "")]


def test_the_agent_loop_sends_what_the_catalog_calls_enter():
    """Advertised as `enter`, read as `press_enter`: a model following the
    prompt typed the text and never sent it."""
    agent = _agent_with_browser()

    _run(agent, "browser_type", {"text": "hello", "enter": True})

    assert agent._browser_session.calls == [("type", "hello", "", True)]
