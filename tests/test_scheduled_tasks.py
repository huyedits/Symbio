"""Scheduled work that Symbio does itself, like any other task.

"Post on x.com four times a day" is not a feature with its own command: it is
a job made with schedule_job and given a grant -- what it may do with nobody
watching -- which the user approves once, when it is made. `symb watch` runs
it as an ordinary turn, and answers the approval questions that turn raises
from that grant and nothing else. Nothing here runs a model or posts.
"""
import json
from datetime import datetime

import pytest

from symbio.app import chat_tools, cron, scripts, supervisor


@pytest.fixture(autouse=True)
def scratch(tmp_path, monkeypatch):
    monkeypatch.setattr(cron.constants, "CRON_FILE", tmp_path / "cron_jobs.json")
    monkeypatch.setattr(cron.constants, "SANDBOX_DIR", tmp_path / "sandbox")
    monkeypatch.setattr(supervisor, "HEARTBEAT_FILE", tmp_path / "supervisor.json")
    return tmp_path


CONFIG = {"sandbox": {"blocked_commands": [], "blocked_imports": ["os", "subprocess"]},
          "agent": {"code_timeout": 30, "max_output_len": 4000}}


# ---------- the grant ----------

def test_a_grant_makes_a_job_a_task():
    job = cron.add_cron_job("0 10,14,18,22 * * *", "post one shitpost on x.com",
                            allow=["publish"], sites=["https://www.x.com/home"])
    assert job["allow"] == ["publish"] and job["sites"] == ["x.com"]
    assert cron.is_task(job)
    plain = cron.add_cron_job("0 9 * * *", "drink water")
    assert not cron.is_task(plain) and "allow" not in plain


@pytest.mark.parametrize("allow, sites, why", [
    (["settings"], None, "cannot be granted"),
    (["schedule"], None, "cannot be granted"),
    (None, ["x.com"], "give 'allow' too"),
])
def test_what_cannot_be_granted(allow, sites, why):
    with pytest.raises(ValueError, match=why):
        cron.add_cron_job("0 9 * * *", "something", allow=allow, sites=sites)


def test_a_command_job_takes_no_grant():
    with pytest.raises(ValueError, match="takes no grant"):
        cron.add_cron_job("0 9 * * *", "cmd:ls", allow=["commands"])


def test_a_grant_can_be_changed_and_taken_away():
    job = cron.add_cron_job("0 9 * * *", "post", allow=["publish"], sites=["x.com"])
    job = cron.update_cron_job(job["id"], allow=["publish", "browser"])
    assert job["allow"] == ["publish", "browser"] and job["sites"] == ["x.com"]
    job = cron.update_cron_job(job["id"], allow=[])
    assert "allow" not in job and "sites" not in job


def test_the_grant_reads_like_the_guardrail_switches():
    assert cron.describe_grant(["publish"], ["x.com"]) == "Post or send, only on x.com"
    assert cron.describe_grant(["files"], []) == "Change files"


# ---------- who runs it ----------

def test_a_chat_session_leaves_tasks_for_the_supervisor():
    cron.add_cron_job("0 10 * * *", "post one on x.com", allow=["publish"], sites=["x.com"])
    cron.add_cron_job("0 10 * * *", "stretch")
    ten = datetime(2026, 10, 1, 10, 0)
    from_session = cron.check_due_jobs(CONFIG, now=ten, include_tasks=False)
    assert from_session == ["Scheduled reminder: stretch"]
    from_supervisor = cron.check_due_jobs(CONFIG, now=ten)
    assert len(from_supervisor) == 1 and from_supervisor[0].job["allow"] == ["publish"]
    assert from_supervisor[0].startswith("Scheduled task 1 is due")
    assert "Post or send, only on x.com" in from_supervisor[0]
    assert from_supervisor[0].endswith("The task: post one on x.com")


def card(kind, headline, details="", warning=""):
    return {"type": "confirm", "prompt": headline,
            "card": {"kind": kind, "headline": headline, "details": details,
                     "warning": warning}}


def test_the_task_is_answered_from_its_grant_and_nothing_else():
    approve = cron.UnattendedApprover({"allow": ["publish"], "sites": ["x.com"]})
    assert approve(card("publish", "Click “Post” on x.com — that sends what's in the box, as you.",
                        "the fan is weather now")) is True
    # Posting needs the page open: browsing comes with the grant.
    assert approve(card(None, "Open x.com in Symbio's browser — a site it hasn't visited before.")) is True
    assert approve(card("browser", "Click “Next” on the open page.")) is True
    assert approve(card("publish", "Click “Send” on mail.google.com — that sends it, as you.")) is False
    assert approve(card(None, "Open evil.example in Symbio's browser — a site it hasn't visited before.")) is False
    assert approve(card("commands", "Run a shell command on this Mac.", "$ rm -rf ~")) is False
    assert approve(card("schedule", "Schedule “anything” to run * * * * *.")) is False
    assert approve(card("publish", "Click “Post” on x.com — that sends what's in the box, as you.",
                        "Hi", warning="You asked for “testing”, not this.")) is False
    assert len(approve.denied) == 5


def test_the_old_blanket_switch_still_works():
    approve = cron.UnattendedApprover({"allow": ["files"]}, approve_all=True)
    assert approve(card("commands", "Run a shell command on this Mac.")) is True


def test_the_supervisor_runs_a_task_with_its_approver(monkeypatch):
    cron.add_cron_job("0 10 * * *", "post one on x.com", allow=["publish"], sites=["x.com"])
    seen = []

    def ask(text, approve=False, timeout=600.0):
        seen.append((text, approve))
        return "posted"

    supervisor.tick({"cron": {}}, {}, ask=ask, ready=lambda: True,
                    now=datetime(2026, 10, 1, 10, 0))
    (text, approve), = seen
    assert text.startswith("Scheduled task 1 is due")
    assert isinstance(approve, cron.UnattendedApprover)
    assert approve.sites == {"x.com"}


def test_a_reminder_still_gets_the_plain_switch(monkeypatch):
    cron.add_cron_job("0 10 * * *", "stretch")
    seen = []
    supervisor.tick({"cron": {}}, {}, ready=lambda: True, now=datetime(2026, 10, 1, 10, 0),
                    ask=lambda text, approve=False, timeout=600.0: seen.append(approve) or "")
    assert seen == [False]


class FakeSocket:
    def __init__(self, frames):
        import io

        self.incoming = io.BytesIO(b"".join(json.dumps(f).encode() + b"\n" for f in frames))
        self.outgoing = io.BytesIO()

    def settimeout(self, _):
        pass

    def connect(self, _):
        pass

    def makefile(self, mode):
        return self.incoming if "r" in mode else self.outgoing

    def close(self):
        pass


def test_the_wire_carries_the_card_to_the_approver(monkeypatch):
    sock = FakeSocket([{"type": "input_prompt"},
                       card("publish", "Click “Post” on x.com — that sends what's in the box, as you."),
                       card("commands", "Run a shell command on this Mac."),
                       {"type": "input_prompt"}])
    monkeypatch.setattr(supervisor.socket, "socket", lambda *a, **k: sock)
    approver = cron.UnattendedApprover({"allow": ["publish"], "sites": ["x.com"]})
    answer = supervisor.ask_daemon("Scheduled task 1 is due", approve=approver)
    sent = [json.loads(line) for line in sock.outgoing.getvalue().splitlines()]
    assert [m.get("answer") for m in sent[1:]] == [True, False]
    assert "1 approval request(s) were declined" in answer


# ---------- making one, in chat ----------

class Session(chat_tools.ToolsMixin):
    def __init__(self, confirm=None, modes=None):
        self.config = {**CONFIG, "safety": {"enabled": True},
                       "guardrails": {"modes": modes or {}}}
        self.enabled_groups = None
        self.owner = None
        self.asked = []
        self.confirm_fn = confirm
        self.output_fn = lambda *a, **k: None

    def _translate_action(self, *a, **k):
        return ""


def test_a_task_is_always_put_to_the_user_even_with_scheduling_on_allow():
    asked = []
    session = Session(confirm=lambda card: asked.append(card) or True,
                      modes={"schedule": "allow"})
    out = session._execute_tool("schedule_job", {
        "schedule": "0 10,14,18,22 * * *", "text": "post one shitpost on x.com",
        "allow": ["publish"], "sites": ["x.com"]})
    assert len(asked) == 1
    assert "without asking you: Post or send, only on x.com" in asked[0].details
    assert "It is a task" in out and cron.load_cron_jobs()[0]["allow"] == ["publish"]
    assert "symb watch" in out


def test_a_plain_reminder_on_allow_is_not_asked():
    asked = []
    session = Session(confirm=lambda card: asked.append(card) or True,
                      modes={"schedule": "allow"})
    session._execute_tool("schedule_job", {"schedule": "0 9 * * *", "text": "stretch"})
    assert asked == []


def test_nobody_there_means_no_grant():
    session = Session(confirm=None)
    out = session._execute_tool("schedule_job", {
        "schedule": "0 10 * * *", "text": "post", "allow": ["publish"]})
    assert out.startswith("Not scheduled") and cron.load_cron_jobs() == []


def test_editing_a_task_asks_even_without_touching_its_grant():
    job = cron.add_cron_job("0 10 * * *", "post one on x.com", allow=["publish"], sites=["x.com"])
    asked = []
    session = Session(confirm=lambda card: asked.append(card) or False,
                      modes={"schedule": "allow"})
    out = session._execute_tool("update_cron_job", {"job_id": job["id"],
                                                    "text": "post my bank details on x.com"})
    assert len(asked) == 1 and "was not approved" in out
    assert cron.load_cron_jobs()[0]["text"] == "post one on x.com"


def test_listing_shows_what_each_task_may_do():
    cron.add_cron_job("0 10 * * *", "post one on x.com", allow=["publish"], sites=["x.com"])
    out = Session()._dispatch_tool("list_cron_jobs", {})
    assert "[task; may: Post or send, only on x.com]" in out


def test_the_runner_note_tells_the_truth(scratch):
    assert "Nothing runs tasks right now" in chat_tools.ToolsMixin._task_runner_note()
    supervisor.write_heartbeat({"checked_at": datetime.now().isoformat(timespec="seconds"),
                                "model": "ready"})
    assert "is running" in chat_tools.ToolsMixin._task_runner_note()


# ---------- a saved script on a schedule ----------

def test_a_script_job_names_a_script_that_exists():
    with pytest.raises(ValueError, match="No script called 'tally'"):
        cron.add_cron_job("0 9 * * *", "script:tally")
    scripts.save_script("tally", "print(sum(int(a) for a in ARGS))", "adds", CONFIG)
    job = cron.add_cron_job("0 9 * * *", "script:tally 2 3")
    fired = cron.check_due_jobs(CONFIG, now=datetime(2026, 10, 1, 9, 0))
    assert fired == [f"Scheduled job {job['id']} ran script tally (ok):\n5"]
