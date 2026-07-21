#!/usr/bin/env python3
"""End-to-end tests for the symbio.app agent loop, driven by a scripted
fake model so tool parsing, sandbox execution, observation feedback, cron
scheduling, and the max-rounds bound are exercised deterministically (no
model load needed)."""
import builtins
import json
from contextlib import contextmanager
from datetime import datetime

from symbio import constants
from symbio.app import chat, cron, learn, memory, sandbox, sessions, tooling, training, web
from symbio.app import config as app_config
from symbio.app.prompts import DEFAULT_SYSTEM_PROMPT, build_system_prompt
from test_utils import preserve_training_state


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, enable_thinking=False):
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        if add_generation_prompt:
            text += "\nassistant:"
        return text

    def encode(self, text, add_special_tokens=True):
        # "Tokens" are just words. _generate_reply only ever compares,
        # slices, and measures len() on these — never inspects the actual
        # int values — so a list[str] is a fully valid stand-in and lets
        # the fake stream_generate below reconstruct readable text.
        return text.split(" ")


class _FakeResponse:
    __slots__ = ("text", "token")

    def __init__(self, text, token):
        self.text = text
        self.token = token


class ScriptedSession:
    """Run chat_loop with scripted user inputs and model replies."""

    def __init__(self, user_inputs, model_replies, config=None):
        self.user_inputs = list(user_inputs)
        self.model_replies = list(model_replies)
        self.config = config
        self.prompts_seen = []
        # What a real KV cache would represent: everything fed across every
        # call so far, cache-aware callers may only feed the new suffix.
        self._fed_so_far: list[str] = []

    def fake_input(self, prompt_text=""):
        if not self.user_inputs:
            raise EOFError
        return self.user_inputs.pop(0)

    def fake_generate(self, model, tokenizer, prompt="", sampler=None, verbose=False, **kwargs):
        # Legacy blocking path — used when agent.prompt_cache_enabled=False.
        self.prompts_seen.append(prompt)
        if not self.model_replies:
            return "Nothing more to say."
        return self.model_replies.pop(0)

    def fake_stream_generate(self, model, tokenizer, prompt, max_tokens=256,
                             sampler=None, prompt_cache=None, **kwargs):
        self._fed_so_far.extend(prompt)
        self.prompts_seen.append(" ".join(self._fed_so_far))
        reply = self.model_replies.pop(0) if self.model_replies else "Nothing more to say."
        for i, word in enumerate(reply.split(" ")):
            text = word if i == 0 else " " + word
            self._fed_so_far.append(word)
            yield _FakeResponse(text, hash(word) & 0xFFFF)

    def run(self):
        real_input = builtins.input
        real_load = chat.load
        real_generate = chat.generate
        real_stream_generate = chat.stream_generate
        real_make_cache = chat.make_prompt_cache
        real_can_trim = chat.can_trim_prompt_cache
        real_trim = chat.trim_prompt_cache
        builtins.input = self.fake_input
        chat.load = lambda *a, **k: (object(), FakeTokenizer())
        chat.generate = self.fake_generate
        chat.stream_generate = self.fake_stream_generate
        chat.make_prompt_cache = lambda model: []
        chat.can_trim_prompt_cache = lambda cache: True
        chat.trim_prompt_cache = lambda cache, n: cache
        try:
            # ChatSession construction touches the real adapters/ directory
            # (checks adapter_config.json, marks it used) even with load()
            # faked out — guard it the same way training runs already are.
            with preserve_training_state(adapters=True):
                chat.chat_loop(self.config or app_config.load_config())
        finally:
            builtins.input = real_input
            chat.load = real_load
            chat.generate = real_generate
            chat.stream_generate = real_stream_generate
            chat.make_prompt_cache = real_make_cache
            chat.can_trim_prompt_cache = real_can_trim
            chat.trim_prompt_cache = real_trim


def test_system_prompt_substitutes_names():
    sp = build_system_prompt("Caine", "Huy")
    assert "Caine" in sp and "Huy" in sp, sp
    assert "{assistant_name}" not in sp and "{user_name}" not in sp, sp
    print("test_system_prompt_substitutes_names passed")


def test_system_prompt_seeds_missing_prompt_md():
    real_prompt = constants.PROMPT_FILE
    real_default = constants.PROMPT_DEFAULT_FILE
    seeded = constants.PROJECT_DIR / "prompt.md.seedtest"
    default_snapshot = constants.PROJECT_DIR / "prompt.md.default.seedtest"
    constants.PROMPT_FILE = seeded
    constants.PROMPT_DEFAULT_FILE = default_snapshot
    try:
        sp = build_system_prompt("Caine", "Huy")
        assert seeded.exists(), "prompt file was not seeded"
        # prompt.md stores the unformatted template; names are substituted at runtime.
        assert seeded.read_text(encoding="utf-8") == DEFAULT_SYSTEM_PROMPT
        assert default_snapshot.exists(), "default snapshot was not written"
        assert default_snapshot.read_text(encoding="utf-8") == DEFAULT_SYSTEM_PROMPT
    finally:
        constants.PROMPT_FILE = real_prompt
        constants.PROMPT_DEFAULT_FILE = real_default
        seeded.unlink(missing_ok=True)
        default_snapshot.unlink(missing_ok=True)
    assert "Caine" in sp and "Huy" in sp, sp
    assert "<cmd>" in sp, sp
    assert "at most ONE tool tag" in sp, sp
    print("test_system_prompt_seeds_missing_prompt_md passed")


def test_system_prompt_auto_updates_unchanged_prompt_md():
    """prompt.md is refreshed when it still matches the last shipped default."""
    real_prompt = constants.PROMPT_FILE
    real_default = constants.PROMPT_DEFAULT_FILE
    prompt_path = constants.PROJECT_DIR / "prompt.md.updatetest"
    default_path = constants.PROJECT_DIR / "prompt.md.default.updatetest"
    constants.PROMPT_FILE = prompt_path
    constants.PROMPT_DEFAULT_FILE = default_path
    try:
        # Simulate an old default snapshot and a prompt.md that matches it.
        old_default = "OLD PROMPT {assistant_name} {user_name}"
        prompt_path.write_text(old_default, encoding="utf-8")
        default_path.write_text(old_default, encoding="utf-8")

        sp = build_system_prompt("Caine", "Huy")
        assert prompt_path.read_text(encoding="utf-8") == DEFAULT_SYSTEM_PROMPT
        assert default_path.read_text(encoding="utf-8") == DEFAULT_SYSTEM_PROMPT
    finally:
        constants.PROMPT_FILE = real_prompt
        constants.PROMPT_DEFAULT_FILE = real_default
        prompt_path.unlink(missing_ok=True)
        default_path.unlink(missing_ok=True)
    assert "Caine" in sp and "Huy" in sp, sp
    print("test_system_prompt_auto_updates_unchanged_prompt_md passed")


def test_system_prompt_preserves_customized_prompt_md():
    """prompt.md is left alone when the user has edited it."""
    real_prompt = constants.PROMPT_FILE
    real_default = constants.PROMPT_DEFAULT_FILE
    prompt_path = constants.PROJECT_DIR / "prompt.md.customtest"
    default_path = constants.PROJECT_DIR / "prompt.md.default.customtest"
    constants.PROMPT_FILE = prompt_path
    constants.PROMPT_DEFAULT_FILE = default_path
    try:
        custom_text = "My custom prompt for {assistant_name} and {user_name}"
        prompt_path.write_text(custom_text, encoding="utf-8")
        default_path.write_text("SHIPPED DEFAULT", encoding="utf-8")

        sp = build_system_prompt("Caine", "Huy")
        assert prompt_path.read_text(encoding="utf-8") == custom_text
        assert default_path.read_text(encoding="utf-8") == DEFAULT_SYSTEM_PROMPT
    finally:
        constants.PROMPT_FILE = real_prompt
        constants.PROMPT_DEFAULT_FILE = real_default
        prompt_path.unlink(missing_ok=True)
        default_path.unlink(missing_ok=True)
    assert sp.startswith("My custom prompt for Caine and Huy"), sp
    assert "<tools>" in sp, sp
    print("test_system_prompt_preserves_customized_prompt_md passed")


def test_parse_and_strip_tool_tags():
    reply = (
        "Sure. <note title='Coffee'>Huy likes coffee.</note>"
        "<cmd>echo hi</cmd><digest /><train />"
        "<cron expr='*/5 * * * *'>hydrate</cron>"
        "<cron at='2026-12-31 23:59'>happy new year</cron>"
    )
    tools = tooling.parse_tools(reply)
    names = [name for name, _ in tools]
    assert names == [
        "write_note", "run_command", "digest_notes", "train_adapter",
        "schedule_job", "schedule_job",
    ], names
    assert tools[0][1] == {"title": "Coffee", "body": "Huy likes coffee."}
    assert tools[1][1] == {"cmd": "echo hi"}
    assert tools[4][1] == {"schedule": "*/5 * * * *", "text": "hydrate"}
    assert tools[5][1] == {"schedule": "at 2026-12-31 23:59", "text": "happy new year"}
    assert tooling.strip_tool_tags(reply) == "Sure.", repr(tooling.strip_tool_tags(reply))
    print("test_parse_and_strip_tool_tags passed")


def test_parse_hermes_tool_calls():
    reply = (
        "I'll run that for you.\n"
        '<tool_call>{"name": "terminal", "arguments": {"cmd": "ssh root@209.38.82.54"}}</tool_call>'
    )
    tools = tooling.parse_tools(reply)
    assert tools == [("run_command", {"cmd": "ssh root@209.38.82.54"})], tools
    assert tooling.strip_tool_tags(reply) == "I'll run that for you."
    print("test_parse_hermes_tool_calls passed")


def test_parse_mixed_hermes_and_legacy_tags():
    reply = (
        "Saving and searching.\n"
        '<tool_call>{"name": "write_note", "arguments": {"title": "Test", "body": "body"}}</tool_call>'
        "<search>news</search>"
    )
    tools = tooling.parse_tools(reply)
    names = [n for n, _ in tools]
    assert names == ["web_search", "write_note"], names
    print("test_parse_mixed_hermes_and_legacy_tags passed")


@contextmanager
def scratch_cron_file():
    real_file = constants.CRON_FILE
    constants.CRON_FILE = constants.PROJECT_DIR / "cron_jobs.test.json"
    try:
        constants.CRON_FILE.unlink(missing_ok=True)
        yield
    finally:
        constants.CRON_FILE.unlink(missing_ok=True)
        constants.CRON_FILE = real_file


def test_cron_matching():
    dt = datetime(2026, 7, 16, 9, 30)  # a Thursday
    assert cron.cron_matches("* * * * *", dt)
    assert cron.cron_matches("30 9 * * *", dt)
    assert not cron.cron_matches("31 9 * * *", dt)
    assert cron.cron_matches("*/15 * * * *", dt)
    assert not cron.cron_matches("*/7 * * * *", dt)
    assert cron.cron_matches("0-45 9 16 7 *", dt)
    assert cron.cron_matches("30 9 * * 4", dt)  # cron weekday: Thursday = 4
    assert not cron.cron_matches("30 9 * * 0", dt)
    assert cron.cron_matches("30 9 * * 0,4", dt)
    assert cron.validate_cron_expr("* * * *") is not None
    assert cron.validate_cron_expr("bogus * * * *") is not None
    assert cron.validate_cron_expr("*/10 8-18 * * 1-5") is None
    print("test_cron_matching passed")


def test_cron_jobs_fire_and_expire():
    with scratch_cron_file():
        config = app_config.load_config()
        one_shot = cron.add_cron_job("at 2026-01-01 09:00", "wish Huy a happy new year")
        assert one_shot["schedule"] == "at 2026-01-01 09:00", one_shot
        cron.add_cron_job("*/5 * * * *", "cmd:echo cron-ok")

        now = datetime(2026, 1, 1, 9, 5)
        events = cron.check_due_jobs(config, now=now)
        assert any("happy new year" in e for e in events), events
        assert any("cron-ok" in e for e in events), events

        # One-shot is gone; recurring fires at most once per minute.
        assert cron.check_due_jobs(config, now=now) == []
        events = cron.check_due_jobs(config, now=datetime(2026, 1, 1, 9, 10))
        assert len(events) == 1 and "cron-ok" in events[0], events

        # Future one-shots stay quiet; bad schedules are rejected up front.
        cron.add_cron_job("at 2099-01-01 00:00", "far future")
        assert cron.check_due_jobs(config, now=datetime(2026, 1, 1, 9, 11)) == []
        try:
            cron.add_cron_job("whenever", "x")
            raise AssertionError("expected ValueError for bad schedule")
        except ValueError:
            pass
    print("test_cron_jobs_fire_and_expire passed")


def test_agent_loop_schedules_job_from_tag():
    with scratch_cron_file():
        session = ScriptedSession(
            user_inputs=["Remind me every day at 9am to stretch.", "/quit", "n"],
            model_replies=[
                "Will do. <cron expr='0 9 * * *'>stretch</cron>",
                "Scheduled it, Huy.",
            ],
        )
        session.run()
        jobs = cron.load_cron_jobs()
        assert len(jobs) == 1 and jobs[0]["schedule"] == "0 9 * * *", jobs
        assert "Scheduled job" in session.prompts_seen[1], session.prompts_seen[1]
    print("test_agent_loop_schedules_job_from_tag passed")


def test_agent_loop_feeds_observation_back():
    session = ScriptedSession(
        user_inputs=["What files are in the sandbox?", "/quit", "n"],
        model_replies=[
            "Let me check. <cmd>echo loop-e2e-marker</cmd>",
            "The command printed loop-e2e-marker. Done.",
        ],
    )
    session.run()

    # Round 1 emits a tool; round 2 must see its output fed back as an observation.
    assert len(session.prompts_seen) == 2, len(session.prompts_seen)
    second = session.prompts_seen[1]
    assert "[System observation:" in second, second
    assert "loop-e2e-marker" in second, second
    assert "exited ok" in second, second
    # Every round grounds the model in wall-clock time and the host OS.
    assert "computer clock" in session.prompts_seen[0], session.prompts_seen[0]
    assert "[Environment:" in session.prompts_seen[0], session.prompts_seen[0]
    print("test_agent_loop_feeds_observation_back passed")


def test_agent_loop_stops_at_max_rounds():
    config = app_config.load_config()
    max_rounds = config["agent"]["max_tool_rounds"]
    # Distinct commands each round so the repeat guard doesn't cut in early.
    session = ScriptedSession(
        user_inputs=["Keep running commands forever.", "/quit", "n"],
        model_replies=[f"<cmd>echo round{i}</cmd>" for i in range(max_rounds + 5)],
    )
    session.run()
    assert len(session.prompts_seen) == max_rounds, len(session.prompts_seen)
    print("test_agent_loop_stops_at_max_rounds passed")


def test_agent_loop_breaks_on_repeated_tool_call():
    session = ScriptedSession(
        user_inputs=["what is the latest news?", "/quit", "n"],
        model_replies=["Here you go. <cmd>echo same-call</cmd>"] * 10,
    )
    session.run()
    # Round 1 executes the command; round 2 repeats it verbatim -> loop ends.
    assert len(session.prompts_seen) == 2, len(session.prompts_seen)
    print("test_agent_loop_breaks_on_repeated_tool_call passed")


def test_agent_loop_self_corrects_malformed_tag():
    """An unterminated tag (e.g. truncated by max_tokens) isn't silently
    treated as the final answer — it's fed back as an observation so the
    model can notice and retry, same spirit as a failed tool call."""
    real_search = web.web_search
    web.web_search = lambda q, c, max_results=5: (True, "1. Some result\n   https://example.com\n   Info.")
    try:
        session = ScriptedSession(
            user_inputs=["Search for something.", "/quit", "n"],
            model_replies=[
                "Let me check that. <search>who won the",  # truncated, no close
                "<search>who won the 2031 prize</search> Checking now.",  # retried, well-formed
                "It was won by nobody in particular.",  # final answer after the search executes
            ],
        )
        session.run()
    finally:
        web.web_search = real_search
    # Round 1: malformed -> self-correct. Round 2: well-formed <search> ->
    # executes as a real tool call. Round 3: final answer from the results.
    assert len(session.prompts_seen) == 3, len(session.prompts_seen)
    obs = session.prompts_seen[1]
    assert "unterminated" in obs.lower(), obs
    assert "try again" in obs.lower(), obs
    print("test_agent_loop_self_corrects_malformed_tag passed")


def test_agent_loop_self_corrects_only_once_per_turn():
    """A persistently malformed reply doesn't burn every round on retries —
    self-correction fires once, then normal end-of-turn handling resumes."""
    session = ScriptedSession(
        user_inputs=["Search for something.", "/quit", "n"],
        model_replies=[
            "<search>always truncated",
            # Still malformed, but with visible prose this time so the
            # (unrelated) blank-reply auto-search fallback doesn't also
            # kick in and add a third round — isolates just this behavior.
            "I couldn't complete that search. <search>still truncated",
        ],
    )
    session.run()
    # One self-correction retry, then the second (still-malformed) reply
    # ends the turn normally rather than retrying forever.
    assert len(session.prompts_seen) == 2, len(session.prompts_seen)
    print("test_agent_loop_self_corrects_only_once_per_turn passed")


def test_agent_loop_executes_one_tool_per_response():
    """A reply with multiple different tool tags must only run the first one."""
    session = ScriptedSession(
        user_inputs=["Run some stuff.", "/quit", "n"],
        model_replies=[
            "Starting. <cmd>echo first</cmd><cmd>echo second</cmd><cmd>echo third</cmd>",
            "Done after the first command.",
        ],
    )
    session.run()
    # Only one tool executed, so only one observation round then the final answer.
    assert len(session.prompts_seen) == 2, len(session.prompts_seen)
    observation = session.prompts_seen[1]
    assert "Command 'echo first' exited ok" in observation, observation
    assert "Output:\nfirst" in observation, observation
    assert "Output:\nsecond" not in observation, observation
    assert "Output:\nthird" not in observation, observation
    assert "ignored" in observation, observation
    print("test_agent_loop_executes_one_tool_per_response passed")


def test_strip_unterminated_tag():
    cut_off = "Here are the stories:\n1. Big story — <cmd>open 'https://example.com/very/long"
    assert tooling.strip_tool_tags(cut_off) == "Here are the stories:\n1. Big story —"
    assert tooling.parse_tools(cut_off) == []
    print("test_strip_unterminated_tag passed")


def test_execute_code_tool():
    config = app_config.load_config()
    # <py> tag parses and strips
    reply = "Sure. <py>print(6 * 7)</py>"
    tools = tooling.parse_tools(reply)
    assert tools == [("execute_code", {"code": "print(6 * 7)"})], tools
    assert tooling.strip_tool_tags(reply) == "Sure."

    ok, out = sandbox.run_python_code("import math\nprint(math.factorial(7))", config)
    assert ok and out == "5040", (ok, out)

    ok, out = sandbox.run_python_code("import os\nprint(os.listdir('/'))", config)
    assert not ok and "not allowed" in out, (ok, out)
    ok, out = sandbox.run_python_code("import socket", config)
    assert not ok and "not allowed" in out, (ok, out)
    ok, out = sandbox.run_python_code("print(", config)
    assert not ok and "Syntax error" in out, (ok, out)
    print("test_execute_code_tool passed")


def test_agent_loop_runs_python():
    session = ScriptedSession(
        user_inputs=["What is 12 factorial?", "/quit", "n"],
        model_replies=[
            "<py>import math\nprint(math.factorial(12))</py>",
            "12! = 479,001,600.",
        ],
    )
    session.run()
    assert len(session.prompts_seen) == 2, len(session.prompts_seen)
    assert "479001600" in session.prompts_seen[1], session.prompts_seen[1]
    assert "Python script exited ok" in session.prompts_seen[1], session.prompts_seen[1]
    print("test_agent_loop_runs_python passed")


_DDG_FIXTURE = """
<div class="result">
<a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fstory&amp;rut=x">Big <b>Story</b> Headline</a>
<a class="result__snippet" href="#">Something <b>important</b> happened today.</a>
</div>
"""


def test_web_tools():
    config = app_config.load_config()

    reply = "<search>latest news</search><read>https://example.com</read>"
    tools = tooling.parse_tools(reply)
    assert ("web_search", {"query": "latest news"}) in tools, tools
    assert ("read_page", {"url": "https://example.com"}) in tools, tools
    assert tooling.strip_tool_tags(reply) == ""

    assert web.html_to_text("<p>Hello <b>world</b><script>bad()</script></p>") == "Hello world"

    real_get = web._http_get
    web._http_get = lambda url, timeout=15: _DDG_FIXTURE
    try:
        ok, out = web.web_search("anything", config)
    finally:
        web._http_get = real_get
    assert ok, out
    assert "Big Story Headline" in out and "https://example.com/story" in out, out
    assert "important" in out, out

    ok, out = web.read_page("file:///etc/passwd", config)
    assert not ok and "http" in out, (ok, out)
    print("test_web_tools passed")


def test_agent_loop_answers_from_search():
    real_search = web.web_search
    web.web_search = lambda q, c, max_results=5: (True, "1. Rain expected\n   https://example.com/wx\n   Heavy rain tomorrow.")
    try:
        with scratch_notes_dir():
            session = ScriptedSession(
                user_inputs=["what is the latest news?", "/quit", "n"],
                model_replies=[
                    "<search>latest news</search>",
                    "The latest: heavy rain is expected tomorrow.",
                ],
            )
            session.run()
            # Ephemeral lookups (news/weather) are never remembered.
            learned = [t for t, _ in
                       ((f.read_text().splitlines()[0], f) for f in constants.NOTES_DIR.glob("*.md"))
                       if t.startswith("# Learned:")]
            assert learned == [], learned
    finally:
        web.web_search = real_search
    assert len(session.prompts_seen) == 2
    assert "Rain expected" in session.prompts_seen[1], session.prompts_seen[1]
    assert "Web search for 'latest news' succeeded" in session.prompts_seen[1]
    print("test_agent_loop_answers_from_search passed")


def test_agent_loop_remembers_research():
    real_search = web.web_search
    web.web_search = lambda q, c, max_results=5: (True, "1. Jorn Utzon\n   https://example.com/opera\n   The Dane behind the sails.")
    try:
        with scratch_notes_dir():
            session = ScriptedSession(
                user_inputs=["Who designed the Sydney Opera House?", "/quit", "n"],
                model_replies=[
                    "<search>Sydney Opera House architect</search>",
                    "The Sydney Opera House was designed by Jorn Utzon.",
                ],
            )
            session.run()
            notes = {f.name: f.read_text() for f in constants.NOTES_DIR.glob("*.md")}
            learned = {n: b for n, b in notes.items() if b.startswith("# Learned:")}
            assert len(learned) == 1, notes.keys()
            body = next(iter(learned.values()))
            assert "Who designed the Sydney Opera House?" in body, body
            assert "Jorn Utzon" in body, body
    finally:
        web.web_search = real_search
    print("test_agent_loop_remembers_research passed")


def test_remember_research_filters():
    config = app_config.load_config()
    with scratch_notes_dir():
        # Durable knowledge is saved.
        p = learn.remember_research(
            "Who designed the Sydney Opera House?",
            "The Sydney Opera House was designed by Jorn Utzon.", config)
        assert p is not None and p.read_text().startswith("# Learned:"), p
        # Exact repeat is deduped.
        assert learn.remember_research(
            "Who designed the Sydney Opera House?",
            "The Sydney Opera House was designed by Jorn Utzon.", config) is None
        # Ephemeral topics and trivial answers are skipped.
        assert learn.remember_research(
            "What's the weather in Tokyo?", "It is raining heavily in Tokyo at the moment.", config) is None
        assert learn.remember_research("Deep question?", "Yes.", config) is None
        # Config kill-switch.
        config["learn"]["remember_research"] = False
        assert learn.remember_research(
            "Who wrote Dune?", "Dune was written by Frank Herbert.", config) is None
    print("test_remember_research_filters passed")


@contextmanager
def scratch_config_file(initial: str = "{}"):
    real_cfg = constants.CONFIG_FILE
    constants.CONFIG_FILE = constants.PROJECT_DIR / "config.test.json"
    try:
        constants.CONFIG_FILE.write_text(initial)
        yield
    finally:
        constants.CONFIG_FILE.unlink(missing_ok=True)
        constants.CONFIG_FILE = real_cfg


def test_self_configuration():
    with scratch_config_file('{"agent": {"temperature": 0.1}, "unrelated": 42}'):
        config = app_config.load_config()

        tools = tooling.parse_tools("<config show /><config set='agent.temperature'>0.4</config>")
        assert ("config_show", {}) in tools, tools
        assert ("config_set", {"key": "agent.temperature", "value": "0.4"}) in tools, tools

        msg = app_config.set_config_value(config, "agent.temperature", "0.4")
        assert "Set agent.temperature" in msg, msg
        assert config["agent"]["temperature"] == 0.4
        saved = json.loads(constants.CONFIG_FILE.read_text())
        assert saved["agent"]["temperature"] == 0.4 and saved["unrelated"] == 42, saved

        assert "Unknown config key" in app_config.set_config_value(config, "nope.nada", "1")
        assert "Bad value" in app_config.set_config_value(config, "agent.max_tool_rounds", "many")
        assert "restart" in app_config.set_config_value(config, "model_name", "other-model")
        # The model may not loosen its own sandbox; the user may via /config.
        assert "user" in app_config.set_config_value(config, "sandbox.blocked_commands", '["rm"]')
        assert "Set sandbox" in app_config.set_config_value(
            config, "sandbox.blocked_commands", '["rm"]', allow_sandbox=True)
        assert "Set memory.enabled" in app_config.set_config_value(config, "memory.enabled", "false")
        assert config["memory"]["enabled"] is False
    print("test_self_configuration passed")


def test_agent_loop_applies_config_change():
    with scratch_config_file():
        session = ScriptedSession(
            user_inputs=["be more creative", "/quit", "n"],
            model_replies=["<config set='agent.temperature'>0.9</config>", "Done!"],
        )
        session.run()
        assert "Set agent.temperature = 0.9" in session.prompts_seen[1], session.prompts_seen[1]
        saved = json.loads(constants.CONFIG_FILE.read_text())
        assert saved["agent"]["temperature"] == 0.9, saved
    print("test_agent_loop_applies_config_change passed")


@contextmanager
def scratch_memory_files():
    real_mem, real_prof = constants.MEMORY_FILE, constants.PROFILE_FILE
    constants.MEMORY_FILE = constants.PROJECT_DIR / "agent_memory.test.md"
    constants.PROFILE_FILE = constants.PROJECT_DIR / "user_profile.test.md"
    try:
        yield
    finally:
        constants.MEMORY_FILE.unlink(missing_ok=True)
        constants.PROFILE_FILE.unlink(missing_ok=True)
        constants.MEMORY_FILE, constants.PROFILE_FILE = real_mem, real_prof


def test_curated_memory_store():
    config = app_config.load_config()
    with scratch_memory_files():
        reply = "<memory>Repo uses MLX.</memory><profile replace='all'>Huy likes bullets.</profile>"
        tools = tooling.parse_tools(reply)
        assert ("save_memory", {"store": "memory", "content": "Repo uses MLX.", "replace": False}) in tools, tools
        assert ("save_memory", {"store": "profile", "content": "Huy likes bullets.", "replace": True}) in tools, tools
        assert tooling.strip_tool_tags(reply) == ""

        msg = memory.save_memory("memory", "Repo uses MLX.", config)
        assert "Saved" in msg, msg
        memory.save_memory("profile", "Huy likes bullets.", config)
        block = memory.curated_memory_block(config)
        assert "Repo uses MLX." in block and "Huy likes bullets." in block, block

        # Over-limit append gets a consolidation nag; replace='all' shrinks it.
        msg = memory.save_memory("memory", "x" * 3000, config)
        assert "over the limit" in msg, msg
        msg = memory.save_memory("memory", "Only this.", config, replace=True)
        assert "over the limit" not in msg, msg
        assert constants.MEMORY_FILE.read_text() == "Only this.\n"
    print("test_curated_memory_store passed")


def test_memory_injected_and_flushed():
    config = app_config.load_config()
    flush_min = config["memory"]["flush_min_turns"]
    with scratch_memory_files():
        constants.MEMORY_FILE.write_text("Deploys happen on Fridays.\n")
        # Enough turns to cross the flush threshold, then /quit.
        chit_chat = [f"hello {i}" for i in range(flush_min)]
        session = ScriptedSession(
            user_inputs=chit_chat + ["/quit", "n"],
            model_replies=["Hi!"] * flush_min
            + ["<memory>Huy tests features right after asking for them.</memory>"],
        )
        session.run()
        # Always-on memory is in every prompt.
        assert "Deploys happen on Fridays." in session.prompts_seen[0], session.prompts_seen[0]
        # The flush turn ran and persisted the model's parting memory.
        saved = constants.MEMORY_FILE.read_text()
        assert "tests features right after asking" in saved, saved
    print("test_memory_injected_and_flushed passed")


def test_session_history_cross_session_recall():
    import shutil
    real_dir = constants.SESSIONS_DIR
    constants.SESSIONS_DIR = constants.PROJECT_DIR / "sessions.test"
    constants.SESSIONS_DIR.mkdir(exist_ok=True)
    try:
        s1 = ScriptedSession(
            user_inputs=["my project codename is quantum-kettle", "/quit", "n"],
            model_replies=["Noted — quantum-kettle it is."],
        )
        s1.run()
        files = list(constants.SESSIONS_DIR.glob("*.jsonl"))
        assert len(files) == 1, files
        rows = [json.loads(l) for l in files[0].read_text().splitlines()]
        assert rows[0]["role"] == "user" and "quantum-kettle" in rows[0]["content"], rows
        assert any(r["role"] == "assistant" for r in rows), rows

        hits = sessions.SessionStore.search("quantum kettle codename")
        assert hits and "quantum-kettle" in hits[0]["content"], hits
        assert sessions.SessionStore.search("quantum kettle", exclude_session=files[0].stem) == []

        # A later session retrieves the earlier one into context.
        s2 = ScriptedSession(
            user_inputs=["what was my project codename again?", "/quit", "n"],
            model_replies=["It's quantum-kettle."],
        )
        s2.run()
        first = s2.prompts_seen[0]
        assert "Past session" in first and "quantum-kettle" in first, first
    finally:
        shutil.rmtree(constants.SESSIONS_DIR, ignore_errors=True)
        constants.SESSIONS_DIR = real_dir
    print("test_session_history_cross_session_recall passed")


def test_rag_injects_saved_notes():
    note_path = memory.save_note(
        "Zephyr Project", "The Zephyr project deadline is March 3rd and uses Rust."
    )
    try:
        session = ScriptedSession(
            user_inputs=["What do you know about the Zephyr project?", "/quit", "n"],
            model_replies=["The Zephyr project is due March 3rd and uses Rust."],
        )
        session.run()
        first = session.prompts_seen[0]
        assert "Retrieved context" in first, first
        assert "March 3rd" in first, first
    finally:
        note_path.unlink(missing_ok=True)

    # Unrelated questions get no retrieval block.
    session = ScriptedSession(
        user_inputs=["hey", "/quit", "n"],
        model_replies=["Hey Huy!"],
    )
    session.run()
    assert "Retrieved context" not in session.prompts_seen[0]
    print("test_rag_injects_saved_notes passed")


class FakeBrowser:
    """Stands in for symbio.computer.BrowserSession in scripted sessions."""

    def __init__(self, confirm_fn=None):
        self.actions = []
        self.confirm_fn = confirm_fn

    def open(self, url):
        self.actions.append(("open", url))
        return f"Opened browser at {url}. Page title: Fake"

    def get_text(self):
        return "Fake page text about MLX."

    def click(self, selector="", text=""):
        self.actions.append(("click", selector or text))
        return "Clicked element containing text 'More'."

    def type_text(self, text, selector="", press_enter=False):
        return f"Typed '{text}'."

    def scroll(self, direction="down", amount=0):
        return f"Scrolled {direction} 800px."

    def close(self):
        return "Browser closed."


def test_parse_browser_tags():
    reply = (
        "<browse>https://a.b/c</browse><click>Sign in</click>"
        "<type enter='true'>lofi</type><scroll dir='up' /><scroll />"
    )
    tools = tooling.parse_tools(reply)
    names = [n for n, _ in tools]
    assert names == [
        "browser_open", "browser_click", "browser_type",
        "browser_scroll", "browser_scroll",
    ], names
    assert tools[0][1] == {"url": "https://a.b/c"}
    assert tools[1][1] == {"target": "Sign in"}
    assert tools[2][1] == {"text": "lofi", "enter": True}
    assert tools[3][1] == {"direction": "up"}
    assert tools[4][1] == {"direction": "down"}
    assert tooling.strip_tool_tags(reply) == "", repr(tooling.strip_tool_tags(reply))
    print("test_parse_browser_tags passed")


def test_agent_loop_browses():
    real_browser = chat.BrowserSession
    chat.BrowserSession = FakeBrowser
    try:
        with scratch_notes_dir():
            session = ScriptedSession(
                user_inputs=["Look up the MLX docs yourself.", "/quit", "n"],
                model_replies=[
                    "<browse>https://example.com/mlx</browse> Opening it.",
                    "The docs say: Fake page text about MLX.",
                ],
            )
            session.run()
            obs_prompt = session.prompts_seen[1]
            assert "Opened browser at https://example.com/mlx" in obs_prompt, obs_prompt
            assert "Fake page text about MLX" in obs_prompt, obs_prompt
            # A browse-backed answer is remembered as research.
            learned = [f for f in constants.NOTES_DIR.glob("*.md")
                       if f.read_text().startswith("# Learned:")]
            assert len(learned) == 1, learned
    finally:
        chat.BrowserSession = real_browser
    print("test_agent_loop_browses passed")


def test_digest_includes_curated_stores():
    import shutil
    import tempfile
    from pathlib import Path

    tok = FakeTokenizer()
    config = app_config.load_config()
    real = (constants.NOTES_DIR, constants.MEMORY_FILE, constants.PROFILE_FILE,
            constants.DIGEST_MANIFEST, constants.TRAIN_FILE)
    tmp = Path(tempfile.mkdtemp())
    constants.NOTES_DIR = tmp / "notes"
    constants.NOTES_DIR.mkdir()
    constants.MEMORY_FILE = tmp / "agent_memory.md"
    constants.PROFILE_FILE = tmp / "user_profile.md"
    constants.DIGEST_MANIFEST = tmp / "manifest.json"
    constants.TRAIN_FILE = tmp / "train.jsonl"
    try:
        constants.MEMORY_FILE.write_text("Deploys happen on Fridays.\n")
        constants.PROFILE_FILE.write_text(f"{config['user_name']} prefers concise replies.\n")
        added = training.digest_notes_to_training(tok, "SYS", config)
        assert added == 2, added
        data = constants.TRAIN_FILE.read_text()
        assert "Fridays" in data and "concise replies" in data, data
        # Unchanged stores are not re-digested.
        assert training.digest_notes_to_training(tok, "SYS", config) == 0
        # An updated profile is digested again.
        constants.PROFILE_FILE.write_text("Prefers bullet points now.\n")
        assert training.digest_notes_to_training(tok, "SYS", config) == 1
    finally:
        (constants.NOTES_DIR, constants.MEMORY_FILE, constants.PROFILE_FILE,
         constants.DIGEST_MANIFEST, constants.TRAIN_FILE) = real
        shutil.rmtree(tmp, ignore_errors=True)
    print("test_digest_includes_curated_stores passed")


def test_decay_research_notes():
    import shutil
    import tempfile
    from pathlib import Path

    config = app_config.load_config()
    config["learn"]["note_decay_days"] = 90
    real = (constants.NOTES_DIR, constants.NOTES_ARCHIVE_DIR,
            constants.TRAIN_FILE, constants.VALID_FILE)
    tmp = Path(tempfile.mkdtemp())
    constants.NOTES_DIR = tmp / "notes"
    constants.NOTES_ARCHIVE_DIR = constants.NOTES_DIR / "archive"
    constants.NOTES_ARCHIVE_DIR.mkdir(parents=True)
    constants.TRAIN_FILE = tmp / "train.jsonl"
    constants.VALID_FILE = tmp / "valid.jsonl"
    try:
        # An expired research note (filename timestamp far past the cutoff),
        # a fresh research note, and an old deliberate note.
        old_title = "Learned: Who won the 2020 title?"
        (constants.NOTES_DIR / "20200101_120000_old.md").write_text(
            f"# {old_title}\n\n**Answer (from web research):** Team A won it.")
        (constants.NOTES_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_fresh.md").write_text(
            "# Learned: Height of K2?\n\n**Answer (from web research):** 8,611 m.")
        (constants.NOTES_DIR / "20200101_120000_evergreen.md").write_text(
            "# Deploy checklist\n\n1. Run tests. 2. Tag the release.")
        # Training data holds the old note's two digested samples + unrelated.
        samples = [
            {"text": f"Write a markdown note titled '{old_title}'. Team A won it."},
            {"text": f"According to your notes, what do you know about '{old_title}'? Team A."},
            {"text": "What is your name? I am Caine."},
        ]
        constants.TRAIN_FILE.write_text(
            "\n".join(json.dumps(s) for s in samples) + "\n")

        archived = training.decay_research_notes(config)
        assert archived == ["20200101_120000_old.md"], archived
        assert (constants.NOTES_ARCHIVE_DIR / "20200101_120000_old.md").exists()
        remaining = sorted(f.name for f in constants.NOTES_DIR.glob("*.md"))
        # Fresh research and deliberate notes survive.
        assert len(remaining) == 2 and "20200101_120000_evergreen.md" in remaining, remaining
        # The decayed note's samples are gone; unrelated samples survive.
        data = constants.TRAIN_FILE.read_text()
        assert "Team A" not in data and "I am Caine" in data, data
        # Second run is a no-op; 0 disables decay entirely.
        assert training.decay_research_notes(config) == []
        config["learn"]["note_decay_days"] = 0
        (constants.NOTES_DIR / "20200101_120000_old2.md").write_text(
            "# Learned: something ancient\n\nAncient body text here.")
        assert training.decay_research_notes(config) == []
    finally:
        (constants.NOTES_DIR, constants.NOTES_ARCHIVE_DIR,
         constants.TRAIN_FILE, constants.VALID_FILE) = real
        shutil.rmtree(tmp, ignore_errors=True)
    print("test_decay_research_notes passed")


@contextmanager
def scratch_notes_dir():
    real_notes = constants.NOTES_DIR
    constants.NOTES_DIR = constants.PROJECT_DIR / "notes.test"
    constants.NOTES_DIR.mkdir(exist_ok=True)
    try:
        yield
    finally:
        import shutil
        shutil.rmtree(constants.NOTES_DIR, ignore_errors=True)
        constants.NOTES_DIR = real_notes


@contextmanager
def scratch_mistakes_dir():
    real_m, real_a = constants.MISTAKES_DIR, constants.MISTAKES_ARCHIVE_DIR
    constants.MISTAKES_DIR = constants.PROJECT_DIR / "mistakes.test"
    constants.MISTAKES_ARCHIVE_DIR = constants.MISTAKES_DIR / "archive"
    constants.MISTAKES_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        yield
    finally:
        import shutil
        shutil.rmtree(constants.MISTAKES_DIR, ignore_errors=True)
        constants.MISTAKES_DIR, constants.MISTAKES_ARCHIVE_DIR = real_m, real_a


def test_parse_skill_tag():
    reply = "Done! <skill name='Check disk'>1. df -h. 2. Report Use%.</skill> Saved."
    tools = tooling.parse_tools(reply)
    assert tools == [("save_skill", {"name": "Check disk", "steps": "1. df -h. 2. Report Use%."})], tools
    stripped = tooling.strip_tool_tags(reply)
    assert "<skill" not in stripped and "df -h" not in stripped, stripped
    print("test_parse_skill_tag passed")


def test_agent_loop_saves_skill():
    with scratch_notes_dir():
        session = ScriptedSession(
            user_inputs=["Remember how you fixed the wifi.", "/quit", "n"],
            model_replies=[
                "<skill name='Fix wifi'>1. Toggle wifi off. 2. Toggle it on.</skill> Saved the steps.",
                "It's saved as a skill now.",
            ],
        )
        session.run()
        assert "Saved skill note" in session.prompts_seen[1], session.prompts_seen[1]
        skills = memory.list_skills()
        assert len(skills) == 1 and skills[0][0] == "Skill: Fix wifi", skills
        body = skills[0][1].read_text()
        assert "Toggle wifi off" in body, body
    print("test_agent_loop_saves_skill passed")


def test_correction_detection_and_mining():
    config = app_config.load_config()
    history = [
        {"role": "user", "content": "What is the capital of Australia?"},
        {"role": "assistant", "content": "The capital of Australia is Sydney."},
    ]
    # Phrase-based detection (pre-append history).
    assert learn.looks_like_correction("No, that's wrong — it's Canberra.", history, config)
    # Repeating the same question also counts as a correction signal.
    assert learn.looks_like_correction("what is the capital of australia", history, config)
    # A normal follow-up does not.
    assert not learn.looks_like_correction("and what about New Zealand?", history, config)
    # Slash commands and empty input never count.
    assert not learn.looks_like_correction("/status", history, config)

    history += [
        {"role": "user", "content": "No, that's wrong — it's Canberra."},
        {"role": "user", "content": "[System observation: something]"},
        {"role": "assistant", "content": "You're right — the capital of Australia is Canberra."},
    ]
    sample = learn.find_correction_sample(history, config)
    assert sample is not None
    query, wrong, correction, correct = sample
    assert query == "What is the capital of Australia?", query
    assert "Sydney" in wrong and "Canberra" in correct, (wrong, correct)
    assert "wrong" in correction, correction
    print("test_correction_detection_and_mining passed")


def test_mistake_digest_and_threshold_training():
    import shutil
    import tempfile
    from pathlib import Path

    config = app_config.load_config()
    config["learn"]["mistake_threshold"] = 2
    config["learn"]["boost_factor"] = 2

    tmp = Path(tempfile.mkdtemp())
    real_train = constants.TRAIN_FILE
    constants.TRAIN_FILE = tmp / "train.jsonl"
    trained_with: list[int | None] = []
    real_run_training = training.run_training
    training.run_training = lambda cfg, iters=None: trained_with.append(iters) or True
    try:
        with scratch_mistakes_dir():
            learn.save_mistake_note("Q-alpha?", "wrong-a", "no,", "right-a")
            # Below threshold: nothing trains, note stays.
            assert not learn.maybe_train_on_mistakes(config, FakeTokenizer(), "SYS")
            assert learn.mistake_note_count() == 1
            assert trained_with == []

            learn.save_mistake_note("Q-beta?", "wrong-b", "no,", "right-b")
            # At threshold: digest (boosted), archive, train with batch iters.
            assert learn.maybe_train_on_mistakes(config, FakeTokenizer(), "SYS")
            assert learn.mistake_note_count() == 0
            archived = list(constants.MISTAKES_ARCHIVE_DIR.glob("*.md"))
            assert len(archived) == 2, archived
            assert trained_with == [config["learn"]["batch_train_iters"]], trained_with
            data = constants.TRAIN_FILE.read_text()
            # boost=2 -> each corrected answer appears twice.
            assert data.count("right-a") == 2 and data.count("right-b") == 2, data
    finally:
        training.run_training = real_run_training
        constants.TRAIN_FILE = real_train
        shutil.rmtree(tmp, ignore_errors=True)
    print("test_mistake_digest_and_threshold_training passed")


def test_correction_severity_grading():
    config = app_config.load_config()
    with scratch_mistakes_dir():
        # Mild rephrase.
        assert learn.correction_severity(
            "What is my name?", "actually, I meant the other one", config) == 1
        # Explicit "that's wrong".
        assert learn.correction_severity(
            "What is my name?", "No, that's wrong — it's Canberra.", config) == 2
        # Repeat offense: the same question was corrected before, pending...
        learn.save_mistake_note("What is my name?", "Bob", "no, it's Huy", "Huy", severity=2)
        assert learn.correction_severity(
            "What is my name?", "actually it's Huy", config) == 3
        # ...or already archived (punctuation/case don't matter).
        learn.archive_mistake_notes()
        assert learn.correction_severity(
            "what is my name", "actually it's Huy", config) == 3
        # A different question is not a repeat.
        assert learn.correction_severity(
            "What is your name?", "actually it's Caine", config) == 1
    print("test_correction_severity_grading passed")


def test_severity_scales_training_iters():
    import shutil
    import tempfile
    from pathlib import Path

    config = app_config.load_config()
    config["learn"]["mistake_threshold"] = 2
    config["learn"]["boost_factor"] = 1
    config["learn"]["batch_train_iters"] = 25
    config["learn"]["iters_per_severity"] = 5
    config["learn"]["max_batch_train_iters"] = 100

    tmp = Path(tempfile.mkdtemp())
    real_train = constants.TRAIN_FILE
    constants.TRAIN_FILE = tmp / "train.jsonl"
    trained_with: list[int | None] = []
    real_run_training = training.run_training
    training.run_training = lambda cfg, iters=None: trained_with.append(iters) or True
    try:
        with scratch_mistakes_dir():
            learn.save_mistake_note("Q-a?", "wrong-a", "that's wrong", "right-a", severity=2)
            learn.save_mistake_note("Q-b?", "wrong-b", "wrong again", "right-b", severity=3)
            assert learn.maybe_train_on_mistakes(config, FakeTokenizer(), "SYS")
            # Total severity 5 over 2 notes -> 25 + 5*(5-2) = 40 iters.
            assert trained_with == [40], trained_with
            # Severity multiplies the boost: the severity-3 answer appears 3x.
            data = constants.TRAIN_FILE.read_text()
            assert data.count("right-a") == 2 and data.count("right-b") == 3, data

        with scratch_mistakes_dir():
            # The cap stops a severe backlog from training forever.
            config["learn"]["max_batch_train_iters"] = 30
            learn.save_mistake_note("Q-c?", "w", "wrong", "right-c", severity=3)
            learn.save_mistake_note("Q-d?", "w", "wrong", "right-d", severity=3)
            assert learn.maybe_train_on_mistakes(config, FakeTokenizer(), "SYS")
            assert trained_with[-1] == 30, trained_with
    finally:
        training.run_training = real_run_training
        constants.TRAIN_FILE = real_train
        shutil.rmtree(tmp, ignore_errors=True)
    print("test_severity_scales_training_iters passed")


def test_agent_loop_captures_correction():
    with scratch_mistakes_dir():
        session = ScriptedSession(
            user_inputs=[
                "What is the capital of Australia?",
                "No, that's wrong — it's Canberra.",
                "/quit", "n",
            ],
            model_replies=[
                "The capital of Australia is Sydney.",
                "You're right — the capital of Australia is Canberra.",
            ],
        )
        session.run()
        notes = [f for f in constants.MISTAKES_DIR.glob("*.md") if f.is_file()]
        assert len(notes) == 1, notes
        body = notes[0].read_text()
        assert "What is the capital of Australia?" in body, body
        assert "Sydney" in body and "Canberra" in body, body
    print("test_agent_loop_captures_correction passed")


def test_sounds_unsure():
    for text in [
        "I'm not sure who holds that record.",
        "I don't know the answer to that.",
        "I don't have information about that event.",
        "As an AI, I cannot answer that.",
        "Hard to say without more data.",
    ]:
        assert learn.sounds_unsure(text), text
    for text in [
        "The capital of France is Paris.",
        "Done! I saved the note.",
        "2 to the power of 40 is 1,099,511,627,776.",
    ]:
        assert not learn.sounds_unsure(text), text
    print("test_sounds_unsure passed")


def test_sounds_like_tool_error():
    for observation in [
        "Command 'chrome' exited error.\nOutput:\nCommand not found: chrome",
        "Web search for 'X' failed.\nResults:\nNone",
        "Tool 'delegate_task' is disabled.",
        "Failed to save note: disk full",
        "Could not schedule job: bad cron expression",
        "Browser click error: the browser session broke and was reset.",
        "Click failed: no visible element with text 'Sign in'.",
        "Browser open blocked: example.com not yet approved.",
        "Worker gave an unrecognized action and stopped: maybe click something",
        "Worker did not finish within 4 round(s). Last action: scroll",
    ]:
        assert learn.sounds_like_tool_error(observation), observation
    for observation in [
        "Command 'echo hi' exited ok.\nOutput:\nhi",
        "Web search for 'database error fixes' succeeded.\nResults:\n"
        "1. How to fix a database error\n   https://example.com\n   Check the error logs.",
        "Scheduled job 1: 0 9 * * * — stretch",
        "Clicked element containing text 'Sign in'.",
        "Typed 'hello' and pressed Enter.",
    ]:
        assert not learn.sounds_like_tool_error(observation), observation
    print("test_sounds_like_tool_error passed")


def test_save_mistake_note_flattens_embedded_newlines():
    with scratch_mistakes_dir():
        original_query = "[System observation: Command 'chrome' exited error.\nOutput:\nCommand not found: chrome]"
        correct_answer = "'chrome' isn't a command here. <cmd>open -a 'Google Chrome'</cmd>"
        path = learn.save_mistake_note(
            original_query=original_query,
            wrong_answer="(a prior tool call failed; see the observation above)",
            correction="(automatic: the next tool call succeeded)",
            correct_answer=correct_answer,
        )
        body = path.read_text()
        # No line break survives inside a field, so digest_mistakes_to_training's
        # line-based parser captures the FULL value, not just its first line.
        for line in body.splitlines():
            if line.startswith("**Original question:**"):
                assert "Command not found: chrome" in line, line
            if line.startswith("**Correct answer:**"):
                assert "<cmd>open -a 'Google Chrome'</cmd>" in line, line
    print("test_save_mistake_note_flattens_embedded_newlines passed")


def test_agent_loop_captures_tool_mistake_and_fix():
    """A tool call that fails, followed by one that works, is captured as a
    mistake note automatically — the model learns from its own tool
    mistakes without needing the user to notice and correct anything."""
    with scratch_mistakes_dir():
        session = ScriptedSession(
            user_inputs=["Open Chrome.", "/quit", "n"],
            model_replies=[
                "<cmd>chrome</cmd>",
                "'chrome' isn't a command here — trying the native way. <cmd>open -a 'Google Chrome'</cmd>",
                "Done, opened Chrome for you.",
            ],
        )
        session.run()
        notes = [f for f in constants.MISTAKES_DIR.glob("*.md") if f.is_file()]
        assert len(notes) == 1, notes
        body = notes[0].read_text()
        assert "Command not found" in body, body
        assert "open -a 'Google Chrome'" in body, body
    print("test_agent_loop_captures_tool_mistake_and_fix passed")


def test_agent_loop_does_not_capture_when_error_never_gets_fixed():
    """If every attempt this turn keeps failing, there is no confirmed fix
    to learn from, so nothing should be saved."""
    with scratch_mistakes_dir():
        session = ScriptedSession(
            user_inputs=["Open Chrome.", "/quit", "n"],
            model_replies=["<cmd>chrome</cmd>", "<cmd>chromebrowser</cmd>",
                           "<cmd>chrome-app</cmd>", "<cmd>launch-chrome</cmd>", "<cmd>chrome-x</cmd>"],
        )
        session.run()
        notes = [f for f in constants.MISTAKES_DIR.glob("*.md") if f.is_file()]
        assert notes == [], notes
    print("test_agent_loop_does_not_capture_when_error_never_gets_fixed passed")


def test_agent_loop_auto_searches_when_unsure():
    real_search = web.web_search
    web.web_search = lambda q, c, max_results=5: (
        True, "1. Wilt Chamberlain's 100-point game\n   https://example.com/wilt\n   Scored 100 points in 1962.")
    try:
        with scratch_notes_dir():
            session = ScriptedSession(
                user_inputs=["Who holds the NBA single-game scoring record?", "/quit", "n"],
                model_replies=[
                    "Hmm, I'm not sure who holds that record.",
                    "It's Wilt Chamberlain, who scored 100 points in a single game in 1962.",
                ],
            )
            session.run()
            # Unsure reply with no tool call -> automatic search -> second round.
            assert len(session.prompts_seen) == 2, len(session.prompts_seen)
            obs = session.prompts_seen[1]
            assert "ran automatically" in obs, obs
            assert "Wilt Chamberlain" in obs, obs
            # The rescued answer counts as research and is remembered.
            learned = [f for f in constants.NOTES_DIR.glob("*.md")
                       if f.read_text().startswith("# Learned:")]
            assert len(learned) == 1, learned
            assert "100 points" in learned[0].read_text()

        # Confident answers never trigger the auto-search.
        session = ScriptedSession(
            user_inputs=["hey", "/quit", "n"],
            model_replies=["Hello Huy!"],
        )
        session.run()
        assert len(session.prompts_seen) == 1, len(session.prompts_seen)
    finally:
        web.web_search = real_search
    print("test_agent_loop_auto_searches_when_unsure passed")


def test_sounds_fabricated():
    # Fires only when the question asks for a figure AND the reply hedges one.
    for question, reply in [
        ("How tall is the Eiffel Tower?", "It's around 300 meters, I think."),
        ("How many moons does Saturn have?", "Probably about 80 or so."),
        ("What year did the Berlin Wall fall?", "I believe it was 1989."),
        ("What's the population of Iceland?", "Roughly 350,000 people live there."),
        ("How far is the Moon?", "It's 380,000 km away, give or take."),
    ]:
        assert learn.sounds_fabricated(question, reply), (question, reply)
    for question, reply in [
        # Confident numbers never trigger; corrections handle wrong ones.
        ("How tall is the Eiffel Tower?", "The Eiffel Tower is 330 metres tall."),
        # Hedge without any number.
        ("How many moons does Saturn have?", "I'd have to check, one moment."),
        # Question doesn't ask for a figure — casual numbers are fine.
        ("Can you set a timer?", "Sure, maybe 5 minutes?"),
        ("hey how's it going", "Great! I'm about 100% ready to help."),
    ]:
        assert not learn.sounds_fabricated(question, reply), (question, reply)
    print("test_sounds_fabricated passed")


def test_agent_loop_auto_searches_on_fabricated_number():
    real_search = web.web_search
    web.web_search = lambda q, c, max_results=5: (
        True, "1. Eiffel Tower - Wikipedia\n   https://example.com/eiffel\n   The tower is 330 metres tall.")
    try:
        with scratch_notes_dir():
            session = ScriptedSession(
                user_inputs=["How tall is the Eiffel Tower?", "/quit", "n"],
                model_replies=[
                    "It's around 300 meters, I think.",
                    "The Eiffel Tower is 330 metres tall.",
                ],
            )
            session.run()
            # Hedged figure with no tool call -> automatic search -> second round.
            assert len(session.prompts_seen) == 2, len(session.prompts_seen)
            obs = session.prompts_seen[1]
            assert "ran automatically" in obs, obs
            assert "330 metres" in obs, obs

        # Moderation: the per-session cap stops auto-search entirely at 0.
        capped_config = app_config.load_config()
        capped_config["web"]["auto_search_session_cap"] = 0
        session = ScriptedSession(
            user_inputs=["How tall is the Eiffel Tower?", "/quit", "n"],
            model_replies=["It's around 300 meters, I think."],
            config=capped_config,
        )
        session.run()
        assert len(session.prompts_seen) == 1, len(session.prompts_seen)
    finally:
        web.web_search = real_search
    print("test_agent_loop_auto_searches_on_fabricated_number passed")


def test_agent_loop_auto_searches_on_blank_reply():
    real_search = web.web_search
    web.web_search = lambda q, c, max_results=5: (
        True, "1. 2026 World Cup final\n   https://example.com/final\n   Spain won the 2026 World Cup.")
    try:
        with scratch_notes_dir():
            # "Look up" would normally suppress auto-search (the user already
            # asked for one), but a fully blank reply always searches anyway.
            session = ScriptedSession(
                user_inputs=["Look up who won the 2026 World Cup", "/quit", "n"],
                model_replies=[
                    "",
                    "Spain won the 2026 World Cup.",
                ],
            )
            session.run()
            assert len(session.prompts_seen) == 2, len(session.prompts_seen)
            obs = session.prompts_seen[1]
            assert "ran automatically" in obs, obs
            assert "came back blank" in obs, obs
            assert "Spain" in obs, obs
    finally:
        web.web_search = real_search
    print("test_agent_loop_auto_searches_on_blank_reply passed")


def test_sandbox_blocks_dangerous_commands():
    config = app_config.load_config()
    ok, out = sandbox.run_sandboxed("rm -rf /", config, interactive=False)
    assert not ok and "blocked" in out, (ok, out)
    ok, out = sandbox.run_sandboxed("echo sandbox-ok", config)
    assert ok and out == "sandbox-ok", (ok, out)
    print("test_sandbox_blocks_dangerous_commands passed")


def test_sandbox_blocked_command_permission_prompt():
    config = app_config.load_config()
    real_input = builtins.input

    # User approves: the blocked command runs.
    builtins.input = lambda *a: "y"
    try:
        ok, out = sandbox.run_sandboxed("bash -c 'echo approved-ok'", config)
    finally:
        builtins.input = real_input
    assert ok and out == "approved-ok", (ok, out)

    # User declines: still blocked.
    builtins.input = lambda *a: "n"
    try:
        ok, out = sandbox.run_sandboxed("bash -c 'echo approved-ok'", config)
    finally:
        builtins.input = real_input
    assert not ok and "blocked" in out, (ok, out)

    # Non-interactive callers (cron thread) never prompt.
    def _fail_input(*a):
        raise AssertionError("prompted in non-interactive mode")
    builtins.input = _fail_input
    try:
        ok, out = sandbox.run_sandboxed("bash -c 'echo approved-ok'", config, interactive=False)
    finally:
        builtins.input = real_input
    assert not ok and "blocked" in out, (ok, out)
    print("test_sandbox_blocked_command_permission_prompt passed")


def run_all():
    import shutil
    # Keep scripted-session chatter out of the real session store — it would
    # otherwise pollute cross-session RAG retrieval for the actual user.
    real_sessions = constants.SESSIONS_DIR
    constants.SESSIONS_DIR = constants.PROJECT_DIR / "sessions.suite"
    constants.SESSIONS_DIR.mkdir(exist_ok=True)
    try:
        _run_all_inner()
    finally:
        shutil.rmtree(constants.SESSIONS_DIR, ignore_errors=True)
        constants.SESSIONS_DIR = real_sessions


def _run_all_inner():
    with preserve_training_state():
        test_system_prompt_substitutes_names()
        test_system_prompt_seeds_missing_prompt_md()
        test_parse_and_strip_tool_tags()
        test_cron_matching()
        test_cron_jobs_fire_and_expire()
        test_execute_code_tool()
        test_agent_loop_runs_python()
        test_web_tools()
        test_agent_loop_answers_from_search()
        test_self_configuration()
        test_agent_loop_applies_config_change()
        test_curated_memory_store()
        test_memory_injected_and_flushed()
        test_session_history_cross_session_recall()
        test_rag_injects_saved_notes()
        test_parse_browser_tags()
        test_agent_loop_browses()
        test_parse_skill_tag()
        test_agent_loop_saves_skill()
        test_correction_detection_and_mining()
        test_mistake_digest_and_threshold_training()
        test_correction_severity_grading()
        test_severity_scales_training_iters()
        test_agent_loop_captures_correction()
        test_remember_research_filters()
        test_agent_loop_remembers_research()
        test_sounds_like_tool_error()
        test_save_mistake_note_flattens_embedded_newlines()
        test_agent_loop_captures_tool_mistake_and_fix()
        test_agent_loop_does_not_capture_when_error_never_gets_fixed()
        test_sounds_unsure()
        test_agent_loop_auto_searches_when_unsure()
        test_sounds_fabricated()
        test_agent_loop_auto_searches_on_fabricated_number()
        test_agent_loop_auto_searches_on_blank_reply()
        test_decay_research_notes()
        test_digest_includes_curated_stores()
        test_sandbox_blocks_dangerous_commands()
        test_sandbox_blocked_command_permission_prompt()
        test_agent_loop_feeds_observation_back()
        test_agent_loop_stops_at_max_rounds()
        test_agent_loop_breaks_on_repeated_tool_call()
        test_agent_loop_self_corrects_malformed_tag()
        test_agent_loop_self_corrects_only_once_per_turn()
        test_strip_unterminated_tag()
        test_agent_loop_schedules_job_from_tag()
    print("\nAll agent-loop tests passed.")


if __name__ == "__main__":
    run_all()
