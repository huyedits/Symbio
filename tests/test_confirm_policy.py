#!/usr/bin/env python3
"""Who is asked, and for what.

The blanket confirm list was written for the Telegram gateway: "the window
that receives this is not in front of whoever is sending it", so a click at
400,300 cannot be judged by the sender and has to be confirmed. It then fired
for every front-end that supplies a confirm_fn — which the daemon always does,
and the daemon is what the desktop window and `symb chat` talk to.

The effect was a harness that asked permission to run `ls`. Measured on this
box: `df -h` scores 0/3, `ls notes` scores 0/3, and both stopped for approval
anyway, because the name gate never consulted the score. Every session on
2026-09-16 that asked something ordinary — free disk space, the Python
version, how many notes there are — died on that prompt, and what the user saw
was an assistant that would not act.

So the gate now asks what it was always really asking: is the person here?
"""
import pytest

from symbio.app import chat_tools
from symbio.app.chat_constants import (
    _ALWAYS_CONFIRM_TOOLS, _LOCAL_TRUSTED_TOOLS, _TELEGRAM_CONFIRM_TOOLS)


def _full_config(**overrides):
    """The real defaults: the dispatcher reads sandbox settings on the way to
    running anything, and a bare {} fails there for reasons unrelated to who
    is being asked."""
    from symbio.app import config as app_config

    config = app_config.load_config()
    for key, value in overrides.items():
        config.setdefault(key, {}).update(value)
    return config


def _session(policy="risk", config=None, answer=True):
    class S(chat_tools.ToolsMixin):
        def __init__(self):
            self.config = config if config is not None else {}
            self.enabled_groups = None
            self.output_fn = lambda *a, **k: None
            self.confirm_fn = lambda _p: answer
            self._confirm_policy = policy
    return S()


# ---- the split ----

def test_the_two_lists_are_disjoint_and_cover_the_old_one():
    """The old name is the union, so a remote front-end gates exactly what it
    gated before."""
    assert not (_ALWAYS_CONFIRM_TOOLS & _LOCAL_TRUSTED_TOOLS)
    assert _ALWAYS_CONFIRM_TOOLS | _LOCAL_TRUSTED_TOOLS == _TELEGRAM_CONFIRM_TOOLS


@pytest.mark.parametrize("tool", ["post_to_x", "submit_form", "config_set",
                                  "train_adapter", "realign", "delete_note",
                                  "schedule_job"])
def test_what_asks_wherever_you_are(tool):
    """Their cost does not depend on their arguments. There is no version of
    post_to_x that is fine unasked."""
    assert _session("risk")._asks_by_name(tool) is True
    assert _session("name")._asks_by_name(tool) is True


@pytest.mark.parametrize("tool", ["run_command", "execute_code", "edit_file",
                                  "write_file", "save_command",
                                  "desktop_click", "open_app"])
def test_what_only_asks_when_you_are_somewhere_else(tool):
    assert _session("risk")._asks_by_name(tool) is False
    assert _session("name")._asks_by_name(tool) is True


def test_a_tool_on_neither_list_is_never_gated_by_name():
    for tool in ("recall", "web_search", "read_file", "write_note", "browser_open"):
        assert _session("name")._asks_by_name(tool) is False


# ---- the gate itself ----

def test_ls_does_not_stop_for_approval_in_front_of_the_machine():
    """The whole complaint, as a test."""
    asked = []
    session = _session("risk", config=_full_config())
    session.confirm_fn = lambda p: asked.append(p) or True

    session._execute_tool("run_command", {"cmd": "ls"})

    assert asked == []


def test_the_same_call_from_a_phone_is_asked():
    asked = []
    session = _session("name", config=_full_config())
    session.confirm_fn = lambda p: asked.append(p) or True

    session._execute_tool("run_command", {"cmd": "ls"})

    assert len(asked) == 1


def test_dropping_the_name_gate_does_not_drop_the_risk_gate():
    """`rm -rf /` scores 3/3 and is refused locally, by the scorer rather than
    by the list. Removing a tool from the name gate removes the question it
    asks about the NAME, never the one it asks about the call."""
    session = _session("risk",
                       config=_full_config(safety={"require_confirm_score": 3}),
                       answer=False)

    out = session._execute_tool("run_command", {"cmd": "rm -rf /"})

    assert "was not approved" in out


def test_a_refusal_by_name_still_refuses():
    session = _session("name", config=_full_config(), answer=False)

    out = session._execute_tool("run_command", {"cmd": "ls"})

    assert out == "Tool 'run_command' was not approved."


# ---- the override ----

def test_config_can_restore_the_old_behaviour_everywhere():
    session = _session("risk", config={"safety": {"confirm_policy": "name"}})

    assert session.confirm_policy() == "name"
    assert session._asks_by_name("run_command") is True


def test_config_can_force_risk_even_on_a_remote_front_end():
    session = _session("name", config={"safety": {"confirm_policy": "risk"}})

    assert session.confirm_policy() == "risk"
    assert session._asks_by_name("run_command") is False


@pytest.mark.parametrize("value", ["", "sometimes", None, 7])
def test_an_unreadable_setting_falls_back_to_the_front_end(value):
    session = _session("name", config={"safety": {"confirm_policy": value}})

    assert session.confirm_policy() == "name"


def test_a_session_with_no_policy_attribute_is_local():
    """Every ChatSession sets it, but the mixin is driven by stand-ins too."""
    class Bare(chat_tools.ToolsMixin):
        config = {}
        enabled_groups = None
        confirm_fn = None

    assert Bare().confirm_policy() == "risk"
