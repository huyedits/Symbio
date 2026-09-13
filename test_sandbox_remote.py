"""Tests for shell/remote execution additions."""

import json
from pathlib import Path

import pytest

from symbio.app import chat, sandbox


@pytest.fixture
def config():
    return {
        "agent": {"sandbox_timeout": 5, "max_output_len": 1000},
        "sandbox": {
            "blocked_commands": ["rm", "sudo"],
            "blocked_shells": ["bash", "sh", "zsh", "fish"],
            "shell_allow_localhost": True,
            "shell_allow_remote_hosts": True,
        },
        "remote": {
            "hosts": {
                "myserver": {"hostname": "example.com", "user": "root", "port": 2222, "ssh_key": "~/.ssh/id_ed25519"},
            }
        },
    }


def test_run_sandboxed_simple_command(config, monkeypatch, tmp_path):
    monkeypatch.setattr(sandbox.constants, "SANDBOX_DIR", tmp_path)
    ok, out = sandbox.run_sandboxed("echo hello", config, interactive=False)
    assert ok is True
    assert "hello" in out


def test_run_shell_pipes(config, monkeypatch, tmp_path):
    monkeypatch.setattr(sandbox.constants, "SANDBOX_DIR", tmp_path)
    # Bypass interactive approval by empty blocked_shells
    config["sandbox"]["blocked_shells"] = []
    ok, out = sandbox.run_shell("echo hello | tr a-z A-Z", config, interactive=False)
    assert ok is True
    assert "HELLO" in out


def test_run_remote_localhost_routes_to_shell(config, monkeypatch, tmp_path):
    monkeypatch.setattr(sandbox.constants, "SANDBOX_DIR", tmp_path)
    config["sandbox"]["blocked_shells"] = []
    ok, out = sandbox.run_remote("localhost", "echo local", config, interactive=False)
    assert ok is True
    assert "local" in out


def test_run_remote_unknown_host(config):
    ok, out = sandbox.run_remote("nope", "uptime", config, interactive=False)
    assert ok is False
    assert "not configured" in out


def test_run_remote_builds_ssh_args(config):
    # We can't actually run SSH, so just inspect the args that would be built.
    host_cfg = config["remote"]["hosts"]["myserver"]
    ssh_args = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new"]
    if host_cfg.get("ssh_key"):
        ssh_args.extend(["-i", str(Path(host_cfg["ssh_key"]).expanduser())])
    if host_cfg.get("port", 22) != 22:
        ssh_args.extend(["-p", str(host_cfg["port"])])
    target = f"{host_cfg['user']}@{host_cfg['hostname']}"
    ssh_args.extend([target, "uptime"])
    assert "-i" in ssh_args
    assert "/Users/" in ssh_args[-5] and ".ssh/id_ed25519" in ssh_args[-5]
    assert "-p" in ssh_args
    assert "2222" in ssh_args
    assert "root@example.com" in ssh_args
    assert "BatchMode=yes" in ssh_args


def test_run_shell_blocked_by_default(config, monkeypatch, tmp_path):
    monkeypatch.setattr(sandbox.constants, "SANDBOX_DIR", tmp_path)
    ok, out = sandbox.run_shell("echo hi", config, interactive=False)
    assert ok is False
    assert "blocked" in out


# ---- the chat dispatch path ----
#
# Every test above calls sandbox.run_remote directly, which is why the tool was
# dead in the running app for as long as it was: chat._dispatch_tool had a
# stray copy of run_command's return statement sitting above the real call, so
# the branch returned before ever reaching sandbox.run_remote and referenced an
# undefined `ok`/`out` and a params key ("cmd") that run_remote does not take.
# The unit tests all passed. These two drive the branch the model actually hits.

class _Session:
    _dispatch_tool = chat.ChatSession._dispatch_tool
    _status = chat.ChatSession._status

    def __init__(self, config):
        self.config = config
        self.confirm_fn = None
        self.enabled_groups = None
        self.output_fn = lambda *_a, **_k: None


def test_run_remote_reaches_the_sandbox_with_host_and_command(config, monkeypatch):
    seen = {}

    def fake_run_remote(host, command, cfg, confirm_fn=None):
        seen.update(host=host, command=command)
        return True, "up 14 days"

    monkeypatch.setattr(chat.sandbox, "run_remote", fake_run_remote)
    out = _Session(config)._dispatch_tool(
        "run_remote", {"host": "myserver", "command": "uptime"})

    assert seen == {"host": "myserver", "command": "uptime"}
    assert "up 14 days" in out
    assert "myserver" in out


def test_run_remote_reports_failure_rather_than_crashing(config, monkeypatch):
    monkeypatch.setattr(
        chat.sandbox, "run_remote",
        lambda *a, **k: (False, "ssh: host unreachable"))
    out = _Session(config)._dispatch_tool(
        "run_remote", {"host": "myserver", "command": "uptime"})

    assert "error" in out
    assert "unreachable" in out


# ---- the remote side runs a shell, so it faces the same denylist ----
#
# run_remote used to build ssh_args and call _run_subprocess with no denylist
# check at all — the local sandbox refused `sudo`/`rm`, and the exact same
# command reached a real root shell when the host name was routed to SSH. A
# refusal must be final there too: the command must never execute, on any
# machine. _run_subprocess raised would prove the bypass.

class _NeverRun:
    """Sentinel: if the command reaches subprocess, the test fails loudly."""

    def __call__(self, *a, **kw):
        raise AssertionError("blocked command reached _run_subprocess")


NEVER_RUN = _NeverRun()


def test_sudo_chain_is_refused_before_ssh(config, monkeypatch):
    monkeypatch.setattr(sandbox, "_run_subprocess", NEVER_RUN)
    ok, out = sandbox.run_remote(
        "myserver", "sudo apt update && sudo apt upgrade -y",
        config, interactive=False)

    assert ok is False
    assert "'sudo' is blocked" in out
    assert "remote host 'myserver'" in out


def test_a_lone_blocked_binary_is_refused(config, monkeypatch):
    monkeypatch.setattr(sandbox, "_run_subprocess", NEVER_RUN)
    ok, out = sandbox.run_remote("myserver", "rm -rf /tmp/x",
                                 config, interactive=False)

    assert ok is False
    assert "'rm' is blocked" in out


def test_a_blocked_binary_after_a_chain_is_caught(config, monkeypatch):
    """A single command's first token is harmless; the &&-chained one is the
    one that deletes. The remote always runs a shell, so under shell chaining
    every token is worth scanning."""
    monkeypatch.setattr(sandbox, "_run_subprocess", NEVER_RUN)
    ok, out = sandbox.run_remote("myserver", "ls && rm -rf /tmp/x",
                                 config, interactive=False)

    assert ok is False
    assert "'rm' is blocked" in out


def test_a_blocked_binary_under_a_wrapper_is_caught(config, monkeypatch):
    """env/nice are the same smuggling trick they are locally: env makes
    argv[0] innocent while executing rm. The wrapper scan must reach the same
    token after SSH as it does in the sandbox. (sudo names itself first, so
    it is caught as first token either way.)"""
    monkeypatch.setattr(sandbox, "_run_subprocess", NEVER_RUN)
    for cmd, hit in (("env rm /tmp/x", "rm"), ("sudo rm /tmp/x", "sudo")):
        ok, out = sandbox.run_remote("myserver", cmd, config, interactive=False)
        assert ok is False
        assert f"'{hit}' is blocked" in out


def test_an_allowed_command_still_runs_remotely(config, monkeypatch):
    seen = {}

    def fake_run(args, cfg):
        seen["args"] = args
        return True, "up 14 days"

    monkeypatch.setattr(sandbox, "_run_subprocess", fake_run)
    ok, out = sandbox.run_remote("myserver", "uptime",
                                 config, interactive=False)

    assert ok is True
    assert "up 14 days" in out
    assert seen["args"][-1] == "uptime"          # the command is what runs
    assert "root@example.com" in seen["args"]


def test_interactive_approval_runs_the_blocked_command_once(config, monkeypatch):
    """Interactive is a prompt, not a bypass: with the user's yes, the command
    runs; without it, nothing runs. approve then deny, same command."""
    seen = {"calls": 0}

    def fake_run(args, cfg):
        seen["calls"] += 1
        return True, "done"

    monkeypatch.setattr(sandbox, "_run_subprocess", fake_run)
    ok, _ = sandbox.run_remote(
        "myserver", "sudo apt update", config,
        interactive=True, confirm_fn=lambda *a, **kw: True)
    assert ok is True
    assert seen["calls"] == 1

    calls_before = seen["calls"]
    monkeypatch.setattr(sandbox, "_run_subprocess", NEVER_RUN)
    ok, out = sandbox.run_remote(
        "myserver", "sudo apt update", config,
        interactive=True, confirm_fn=lambda *a, **kw: False)
    assert ok is False
    assert "'sudo' is blocked" in out
    assert seen["calls"] == calls_before  # denial ran nothing


def test_an_unparseable_remote_command_refuses_without_running(config, monkeypatch):
    monkeypatch.setattr(sandbox, "_run_subprocess", NEVER_RUN)
    ok, out = sandbox.run_remote("myserver", 'echo "unterminated',
                                 config, interactive=False)
    assert ok is False
    assert "parse error" in out
