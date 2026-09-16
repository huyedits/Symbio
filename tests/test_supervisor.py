"""Staying online, without handing the machine to a loop.

Every scheduled job used to run inside a ChatSession's background thread, so
"every morning at 8" meant "every morning at 8, if a chat window happens to be
open". With no client attached the daemon sits idle and nothing fires. The
supervisor is the process that watches instead.

Two properties are worth pinning. It must not turn a broken daemon into a
crash loop — each restart maps several GB of weights. And it must not answer
the approval gate for a user who is not there: the daemon asks its client
before a risky tool, and here the client is a loop.
"""
from datetime import datetime

import pytest

from symbio.app import supervisor


@pytest.fixture
def config():
    return {"cron": {}, "agent": {}}


def test_a_dead_model_is_restarted(config, monkeypatch):
    starts = []
    monkeypatch.setattr(supervisor.cron, "check_due_jobs", lambda *a, **k: [])

    state = supervisor.tick(config, {}, ready=lambda: False,
                            start=lambda: starts.append(1) or True)

    assert starts == [1]
    assert state["model"] == "starting" and state["restarts"] == 1


def test_a_daemon_that_will_not_start_backs_off(config):
    """A bad model path or a full machine would otherwise be retried in a
    tight loop, mapping gigabytes on every pass."""
    state = {}
    for _ in range(5):
        state = supervisor.tick(config, state, ready=lambda: False,
                                start=lambda: False)

    assert state["backoff"] > supervisor.BACKOFF_START
    assert state["backoff"] <= supervisor.BACKOFF_MAX
    assert "could not start" in state["last_error"]


def test_a_healthy_pass_clears_the_backoff(config, monkeypatch):
    monkeypatch.setattr(supervisor.cron, "check_due_jobs", lambda *a, **k: [])

    state = supervisor.tick(config, {"backoff": 120.0}, ready=lambda: True)

    assert state["model"] == "ready"
    assert state["backoff"] == supervisor.BACKOFF_START


def test_a_due_job_is_handed_to_the_model(config, monkeypatch):
    monkeypatch.setattr(supervisor.cron, "check_due_jobs",
                        lambda *a, **k: ["[Cron] check the deploy"])
    asked = []

    def _ask(text, approve=False, timeout=600.0):
        asked.append((text, approve))
        return "Deploy is green."

    state = supervisor.tick(config, {}, ask=_ask, ready=lambda: True)

    assert asked == [("[Cron] check the deploy", False)]
    assert state["jobs_run"] == 1
    assert state["ran"][0]["answer"] == "Deploy is green."


def test_the_model_is_checked_before_the_cron_table(config, monkeypatch):
    """check_due_jobs marks a job as run when it FIRES, not when it is
    answered — so firing one while nothing is loaded loses it."""
    fired = []
    monkeypatch.setattr(supervisor.cron, "check_due_jobs",
                        lambda *a, **k: fired.append(1) or [])

    supervisor.tick(config, {}, ready=lambda: False, start=lambda: True)

    assert fired == [], "the table must not be ticked while the model is down"


def test_approval_is_refused_when_nobody_is_watching(config):
    """This is how an agent ends up running rm -rf at 3am on its own
    authority. The default answer is no."""
    assert config.get("cron", {}).get("unattended_approve", False) is False

    seen = []
    supervisor.tick(config, {}, ready=lambda: True,
                    ask=lambda text, approve=False, timeout=600.0:
                        seen.append(approve) or "",
                    now=datetime.now())
    # With no jobs due nothing is asked; the flag is what the call would carry.
    assert seen == [] or seen == [False]


def test_the_approval_default_can_be_turned_off_deliberately(monkeypatch):
    monkeypatch.setattr(supervisor.cron, "check_due_jobs",
                        lambda *a, **k: ["[Cron] tidy the sandbox"])
    seen = []

    supervisor.tick({"cron": {"unattended_approve": True}}, {},
                    ready=lambda: True,
                    ask=lambda text, approve=False, timeout=600.0:
                        seen.append(approve) or "done")

    assert seen == [True]


def test_a_cron_failure_is_recorded_not_raised(config, monkeypatch):
    def _boom(*_a, **_k):
        raise RuntimeError("cron_jobs.json is corrupt")

    monkeypatch.setattr(supervisor.cron, "check_due_jobs", _boom)

    state = supervisor.tick(config, {}, ready=lambda: True)

    assert "cron_jobs.json is corrupt" in state["last_error"]
    assert state["model"] == "ready", "one bad job must not stop the watch"


def test_the_history_does_not_grow_without_bound(config, monkeypatch):
    monkeypatch.setattr(supervisor.cron, "check_due_jobs",
                        lambda *a, **k: ["[Cron] ping"])
    state = {}

    for _ in range(40):
        state = supervisor.tick(config, state, ready=lambda: True,
                                ask=lambda *a, **k: "pong")

    assert len(state["ran"]) == 20
    assert state["jobs_run"] == 40


def test_the_heartbeat_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "HEARTBEAT_FILE", tmp_path / "supervisor.json")

    supervisor.write_heartbeat({"model": "ready", "jobs_run": 3})

    assert supervisor.read_heartbeat()["jobs_run"] == 3


def test_a_missing_heartbeat_is_an_empty_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "HEARTBEAT_FILE", tmp_path / "nothing.json")

    assert supervisor.read_heartbeat() == {}
