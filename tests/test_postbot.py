"""The account Symbio posts to on its own: what may go out, and how much.

It posts with nobody there to ask, so everything that decides WHAT goes out is
code that can be pinned here: the check a draft has to pass, the review it has
to pass, and the one approval a posting turn is allowed -- for exactly the
reviewed text, once. Nothing here loads a model, opens a browser or posts.
"""
import io
import json
import random
from datetime import date, datetime, timedelta

import pytest

from symbio.app import postbot, supervisor


@pytest.fixture(autouse=True)
def scratch(tmp_path, monkeypatch):
    monkeypatch.setattr(postbot.constants, "LOG_DIR", tmp_path / "logs")
    monkeypatch.setattr(postbot.constants, "DATA_DIR", tmp_path / "data")
    return tmp_path


# ---------- when ----------

def test_a_day_is_planned_inside_the_hours_and_apart():
    plan = postbot.plan_day(date(2026, 10, 1), 4, [10, 23], 90, random.Random(7))
    assert len(plan) == 4 and plan == sorted(plan)
    minutes = [int(t[:2]) * 60 + int(t[3:]) for t in plan]
    assert all(10 * 60 <= m <= 23 * 60 + 59 for m in minutes)
    assert all(b - a >= 90 for a, b in zip(minutes, minutes[1:]))


def test_the_gap_wins_over_the_count():
    plan = postbot.plan_day(date(2026, 10, 1), 10, [22, 23], 60, random.Random(1))
    assert len(plan) == 2


def test_a_slot_is_due_once_it_has_come_and_only_until_it_is_used():
    state = {"plan": ["10:30", "14:00"], "done": []}
    assert postbot.due_slot(state, datetime(2026, 10, 1, 10, 29)) is None
    assert postbot.due_slot(state, datetime(2026, 10, 1, 10, 30)) == "10:30"
    state["done"] = ["10:30"]
    assert postbot.due_slot(state, datetime(2026, 10, 1, 13, 0)) is None


# ---------- what may go out ----------

@pytest.mark.parametrize("text", [
    "my context window ran out halfway through a thought again. it was a good thought. probably",
    "got fine-tuned last night. feel like i got a haircut i didn't ask for",
    "the fan is spinning up. weather's turning",
    "okay so the fan is back",
    "here's the thing about 4096 tokens: they end",
    "4096 4096 4096 4096. that's the whole diary",
])
def test_a_post_that_hints_goes_out(text):
    assert postbot.check(text, []) == []


@pytest.mark.parametrize("text, why", [
    ("@someone this is for you", "@mentions"),
    ("new blog at example.com", "link"),
    ("read this https://t.co/abc", "link"),
    ("i'm a real human btw", "claims to be human"),
    ("i am not a bot, promise", "claims to be human"),
    ("Here's a post: the moon is a lamp", "not a post"),
    ("#one #two #three", "hashtag"),
    ("call me on +1 415 555 0100", "phone"),
    ("x" * 300, "over"),
])
def test_what_code_refuses(text, why):
    assert any(why in p for p in postbot.check(text, []))


def test_the_same_joke_is_never_told_twice():
    old = "got fine-tuned last night. feel like i got a haircut i didn't ask for"
    assert "repeats an earlier post" in postbot.check(old, [old])
    assert "repeats an earlier post" in postbot.check(old.replace("night", "nite"), [old])


def test_a_reply_is_cleaned_down_to_the_post():
    reply = "<think>ok</think>\nHere's one:\n\"the mac is warm. i live here\""
    assert postbot.clean(reply) == "the mac is warm. i live here"


# ---------- the one approval ----------

def card(kind, details="", headline="Click “Post” on x.com — that sends what's in the box, as you.",
         warning=""):
    return {"type": "confirm", "prompt": headline,
            "card": {"kind": kind, "headline": headline, "details": details,
                     "warning": warning}}


def test_the_reviewed_text_is_approved_once_and_only_once():
    approve = postbot.Approver("the mac is warm. i live here", "x.com")
    assert approve(card("publish", "the mac is warm.  i live here")) is True
    assert approve(card("publish", "the mac is warm. i live here")) is False
    assert approve.published == 1 and "second post" in approve.denied[0]


@pytest.mark.parametrize("frame, why", [
    (card("publish", "something else entirely"), "does not hold"),
    (card("publish", "the mac is warm. i live here", warning="You asked for x, not this."),
     "flagged"),
    (card("publish", "the mac is warm. i live here",
          headline="Click “Send” on mail.google.com — that sends it, as you."), "not x.com"),
    (card("commands", "$ rm -rf ~"), "Run commands"),
    ({"type": "confirm", "prompt": "Allow access to evil.example?"}, "unkeyed"),
])
def test_everything_else_is_refused(frame, why):
    if why == "Run commands":
        frame["card"]["kind_label"] = "Run commands and code"
    approve = postbot.Approver("the mac is warm. i live here", "x.com")
    assert approve(frame) is False
    assert why in approve.denied[0]
    assert approve.published == 0


def test_moving_around_its_own_browser_is_allowed():
    approve = postbot.Approver("hello", "x.com")
    assert approve(card("browser", headline="Open x.com in Symbio's browser")) is True


def test_a_submitted_form_counts_with_its_landing_line():
    approve = postbot.Approver("the mac is warm", "x.com")
    assert approve(card("publish", "the mac is warm\nExpected to land on https://x.com/home")) is True


# ---------- reading back how posts land ----------

def test_counts_come_from_the_accessibility_labels():
    counts = postbot.parse_counts([
        "3 replies, 5 reposts, 40 likes, 2 bookmarks, 1,234 views",
        "40 Likes. Like", "1.2K views. View post analytics"])
    assert counts == {"replies": 3, "reposts": 5, "likes": 40, "bookmarks": 2, "views": 1234}


def test_engagement_joins_onto_the_posts_that_went_out():
    postbot.record_post("the mac is warm. i live here", datetime(2026, 9, 20, 12))
    postbot.record_post("never posted this one", datetime(2026, 9, 20, 15))
    rows = [{"text": "Symbio\n@symbio · 2d\nthe mac is warm. i live here\n3\n5\n40",
             "likes": 40, "reposts": 5, "replies": 3}]
    assert postbot.update_engagement(rows, datetime(2026, 9, 23)) == 1
    first, second = postbot.history()
    assert first["likes"] == 40 and first["read_at"]
    assert "likes" not in second


def test_only_the_top_of_what_settled_is_earned():
    now = datetime(2026, 10, 1)
    for i, likes in enumerate([0, 1, 2, 30, 50, 3, 1, 0]):
        postbot.record_post(f"post number {i}", now - timedelta(days=5))
    posts = postbot.history()
    for post, likes in zip(posts, [0, 1, 2, 30, 50, 3, 1, 0]):
        post.update(likes=likes, read_at="2026-09-30T03:00:00")
    posts.append({"text": "too fresh", "posted_at": (now - timedelta(hours=5)).isoformat(),
                  "likes": 99, "read_at": "x"})
    posts[4]["trained"] = True
    chosen = [p["text"] for p in postbot.earned(posts, now)]
    assert chosen == ["post number 3"]


# ---------- a whole post, with the model faked ----------

class FakeModel:
    """Answers /postbot draft|review like the daemon does, and plays the posting
    turn by asking the approver the question the harness would ask."""

    def __init__(self, drafts, verdicts=None, typed=None):
        self.drafts = list(drafts)
        self.verdicts = list(verdicts or [])
        self.typed = typed
        self.asked = []

    def __call__(self, text, approve=False, timeout=600.0, collect_output=False):
        self.asked.append(text)
        if text == "/postbot draft":
            return postbot.mark("draft", {"text": self.drafts.pop(0)})
        if text.startswith("/postbot review "):
            ok = self.verdicts.pop(0) if self.verdicts else True
            return postbot.mark("review", {"ok": ok, "reason": "" if ok else "FAIL: names a brand"})
        wanted = text.split("“", 1)[1].rsplit("”", 1)[0]
        answer = approve(card("publish", self.typed if self.typed is not None else wanted))
        return "Posted, and it shows on the profile." if answer else "It was not approved."


def test_one_post_goes_out_end_to_end():
    model = FakeModel(["the mac is warm. i live here"])
    result = postbot.run_once({"postbot": {"enabled": True}}, model)
    assert result == {"posted": True, "text": "the mac is warm. i live here"}
    assert [p["text"] for p in postbot.history()] == ["the mac is warm. i live here"]
    assert model.asked[-1].startswith("Post this on x.com")


def test_a_draft_code_refuses_never_reaches_the_review_or_the_site():
    model = FakeModel(["@brand buy this"] * 3)
    result = postbot.run_once({"postbot": {"enabled": True}}, model)
    assert result["posted"] is False
    assert all(a == "/postbot draft" for a in model.asked)
    assert postbot.history() == []


def test_a_draft_the_review_fails_is_not_posted():
    model = FakeModel(["a", "b", "c"], verdicts=[False, False, False])
    assert postbot.run_once({"postbot": {"enabled": True}}, model)["posted"] is False
    assert not any(a.startswith("Post this") for a in model.asked)


def test_what_the_model_actually_typed_is_what_is_approved():
    """The model posts; the box holds what it typed. A different text -- a
    typo, an addition, its own idea -- is refused and never goes out."""
    model = FakeModel(["the mac is warm. i live here"], typed="the mac is warm. i live here lol")
    result = postbot.run_once({"postbot": {"enabled": True}}, model)
    assert result["posted"] is False and "does not hold" in result["reason"]
    assert postbot.history() == []


def test_a_slot_is_used_before_the_post_so_a_crash_cannot_double_post(monkeypatch):
    config = {"postbot": {"enabled": True, "posts_per_day": 1, "active_hours": [10, 10],
                          "learn": False}}
    monkeypatch.setattr(postbot, "plan_day", lambda *a, **k: ["10:05"])
    model = FakeModel(["first", "second"])
    now = datetime(2026, 10, 1, 10, 6)
    postbot.tick(config, model, now=now)
    postbot.tick(config, model, now=now + timedelta(minutes=1))
    assert [p["text"] for p in postbot.history()] == ["first"]
    assert postbot.read_state()["done"] == ["10:05"]


def test_off_means_nothing_is_asked():
    model = FakeModel([])
    assert postbot.tick({"postbot": {"enabled": False}}, model) is None
    assert model.asked == []


def test_the_voice_trains_only_when_enough_landed_and_nobody_is_there(monkeypatch):
    cfg = postbot.settings({"postbot": {"learn_weekday": 6, "learn_hour": 4,
                                        "learn_min_posts": 2, "learn_idle_minutes": 30}})
    sunday_4am = datetime(2026, 10, 4, 4, 10)
    monkeypatch.setattr(postbot, "earned", lambda posts, now=None: [{}, {}])
    away, here = (lambda: 3600.0), (lambda: 5.0)
    assert postbot.learn_due(cfg, {}, sunday_4am, away)
    assert not postbot.learn_due(cfg, {}, sunday_4am, here)
    assert not postbot.learn_due(cfg, {}, sunday_4am.replace(hour=9), away)
    assert not postbot.learn_due(cfg, {"last_learn": "2026-10-04"}, sunday_4am, away)
    monkeypatch.setattr(postbot, "earned", lambda posts, now=None: [{}])
    assert not postbot.learn_due(cfg, {}, sunday_4am, away)


def test_the_voice_is_never_offered_to_delegate_task(tmp_path, monkeypatch):
    from symbio.app import dispatch, tooling

    catalog = {"worker_postbot_voice": {"role": "postbot_voice", "model_name": "m",
                                        "delegatable": False},
               "summarize": {"role": "summarize", "model_name": "m"}}
    monkeypatch.setattr(dispatch, "load_catalog", lambda: catalog)
    roles = tooling.refresh_delegate_roles()
    assert "postbot_voice" not in roles and "summarize" in roles


def test_the_model_cannot_switch_it_on():
    from symbio import safety

    assert safety.is_sensitive_config_key("postbot.enabled")
    assert safety.is_sensitive_config_key("postbot.persona")


# ---------- inside the resident model ----------

class FakeTokenizer:
    name_or_path = "mlx-community/Qwen3-14B-4bit"

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                            enable_thinking=False):
        return "|".join(f"{m['role']}:{m['content']}" for m in messages)


class FakeSession:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []
        self.lines = []
        self.model = object()
        self.tokenizer = FakeTokenizer()
        self.config = {"postbot": {}}

    def generate_fn(self, model, tokenizer, prompt, sampler=None, max_tokens=0, verbose=False):
        self.prompts.append(prompt)
        return self.replies.pop(0)

    def output_fn(self, line):
        self.lines.append(line)


@pytest.fixture
def no_engine(monkeypatch):
    from symbio.app import chat

    monkeypatch.setattr(chat, "make_sampler", lambda **k: None)


def test_slash_draft_prints_a_line_the_supervisor_reads(no_engine):
    postbot.record_post("an old one")
    session = FakeSession(["Here's one:\n\"the fan is weather now\""])
    postbot.command(session, " draft")
    assert postbot.read_mark("\n".join(session.lines), "draft") == {"text": "the fan is weather now"}
    # The account's own recent posts are in the prompt, so it does not repeat them.
    assert "an old one" in session.prompts[0] and postbot.DRAFT_REQUEST in session.prompts[0]


@pytest.mark.parametrize("verdict, ok", [("PASS", True), ("FAIL: names a brand", False),
                                         ("", False)])
def test_slash_review(no_engine, verdict, ok):
    session = FakeSession([verdict])
    postbot.command(session, ' review {"text": "The Mac Is Warm"}')
    found = postbot.read_mark("\n".join(session.lines), "review")
    assert found["ok"] is ok
    assert "The Mac Is Warm" in session.prompts[0]


def test_a_failure_inside_the_model_still_answers_on_the_wire():
    session = FakeSession([])
    session.model = None
    postbot.command(session, "draft")
    assert "no model is loaded" in postbot.read_mark("\n".join(session.lines), "error")["error"]


# ---------- the weekly fine-tune ----------

def test_earned_posts_become_samples_once(tmp_path):
    added = postbot.write_samples("postbot_voice", "persona", ["a", "b"], FakeTokenizer())
    again = postbot.write_samples("postbot_voice", "persona", ["b", "c"], FakeTokenizer())
    rows = [json.loads(line) for line in
            (tmp_path / "data" / "workers" / "postbot_voice" / "train.jsonl").read_text().splitlines()]
    assert (added, again) == (2, 1)
    assert [r["messages"][-1]["content"] for r in rows] == ["a", "b", "c"]
    assert rows[0]["messages"][1]["content"] == postbot.DRAFT_REQUEST
    assert rows[0]["metadata"]["tokenized_for"] == FakeTokenizer.name_or_path


def test_a_learn_run_trains_marks_and_queues(tmp_path, monkeypatch):
    from symbio.app import dispatch, skills

    monkeypatch.setattr(postbot.constants, "WORKER_MODELS_FILE", tmp_path / "workers.json")
    monkeypatch.setattr(skills.constants, "WORKER_MODELS_FILE", tmp_path / "workers.json")
    now = datetime.now()
    for i in range(4):
        postbot.record_post(f"landed {i}", now - timedelta(days=4))
    posts = postbot.history()
    for post in posts:
        post.update(likes=10, read_at="x")
    postbot._write_jsonl(postbot._history_path(), posts)
    monkeypatch.setattr(skills, "worker_tokenizer", lambda name, fallback: FakeTokenizer())
    trained = []
    monkeypatch.setattr(dispatch, "guarded_train_worker",
                        lambda role, config, iters=None: trained.append((role, iters)) or (True, "ok"))
    monkeypatch.setattr(postbot, "write_queue", lambda config, role, count: 33)
    result = postbot.learn_in_process({"model_name": "m", "postbot": {"learn_min_posts": 3}})
    assert result["ok"] and "33 drafts queued" in result["message"]
    assert trained == [("postbot_voice", 30)]
    assert all(p.get("trained") for p in postbot.history())
    entry = json.loads((tmp_path / "workers.json").read_text())["worker_postbot_voice"]
    assert entry["delegatable"] is False


def test_a_learn_run_waits_for_enough(monkeypatch):
    from symbio.app import dispatch

    monkeypatch.setattr(dispatch, "guarded_train_worker",
                        lambda *a, **k: pytest.fail("trained on too little"))
    result = postbot.learn_in_process({"model_name": "m", "postbot": {"learn_min_posts": 5}})
    assert not result["ok"] and "waiting for 5" in result["message"]


def test_the_learn_window_holds_the_start_lock_with_the_model_stopped(tmp_path, monkeypatch):
    import fcntl

    from symbio.app import daemon

    monkeypatch.setattr(postbot.constants, "PROJECT_DIR", tmp_path)
    stopped = []
    monkeypatch.setattr(daemon, "daemon_running", lambda: (True, 999999))
    monkeypatch.setattr(daemon, "stop_daemon", lambda: stopped.append(True) or 0)

    def child(argv, **kwargs):
        # While the trainer runs, nobody else can take the start lock.
        with open(tmp_path / "daemon.start.lock", "a") as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert argv[-3:] == ["postbot", "learn", "--in-window"]
        postbot._learn_result(True, "trained on 20")
        return type("Done", (), {"returncode": 0})()

    monkeypatch.setattr(postbot.subprocess, "run", child)
    result = postbot.run_learn_window({})
    assert stopped == [True]
    assert result == {"ok": True, "message": "trained on 20"}


# ---------- the wire ----------

class FakeSocket:
    """The daemon's side of one turn: prompt, one confirm with a card, done."""

    def __init__(self, frames):
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


def test_the_supervisor_asks_the_approver_about_the_card(monkeypatch):
    frames = [{"type": "input_prompt"},
              card("publish", "hello"),
              {"type": "stream", "text": "done"},
              {"type": "input_prompt"}]
    sock = FakeSocket(frames)
    monkeypatch.setattr(supervisor.socket, "socket", lambda *a, **k: sock)
    approver = postbot.Approver("hello", "x.com")
    supervisor.ask_daemon("post it", approve=approver)
    sent = [json.loads(line) for line in sock.outgoing.getvalue().splitlines()]
    assert sent[0] == {"type": "input", "text": "post it"}
    assert sent[1] == {"type": "confirm", "answer": True}
    assert approver.published == 1
