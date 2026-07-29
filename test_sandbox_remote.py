"""Tests for shell/remote execution additions."""

import json
from pathlib import Path

import pytest

from symbio.app import sandbox


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
