#!/usr/bin/env python3
"""End-to-end tests for main.py's autonomous agent loop, driven by a scripted
fake model so tool parsing, sandbox execution, observation feedback, cron
scheduling, and the max-rounds bound are exercised deterministically (no
model load needed)."""
import builtins
import json
from contextlib import contextmanager
from datetime import datetime

import main
from test_utils import preserve_training_state


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, enable_thinking=False):
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        if add_generation_prompt:
            text += "\nassistant:"
        return text


class ScriptedSession:
    """Run chat_loop with scripted user inputs and model replies."""

    def __init__(self, user_inputs, model_replies):
        self.user_inputs = list(user_inputs)
        self.model_replies = list(model_replies)
        self.prompts_seen = []

    def fake_input(self, prompt_text=""):
        if not self.user_inputs:
            raise EOFError
        return self.user_inputs.pop(0)

    def fake_generate(self, model, tokenizer, prompt="", sampler=None, verbose=False, **kwargs):
        self.prompts_seen.append(prompt)
        if not self.model_replies:
            return "Nothing more to say."
        return self.model_replies.pop(0)

    def run(self):
        real_input = builtins.input
        real_load = main.load
        real_generate = main.generate
        builtins.input = self.fake_input
        main.load = lambda *a, **k: (object(), FakeTokenizer())
        main.generate = self.fake_generate
        try:
            main.chat_loop(main.load_config())
        finally:
            builtins.input = real_input
            main.load = real_load
            main.generate = real_generate


def test_system_prompt_substitutes_names():
    sp = main.build_system_prompt("Caine", "Huy")
    assert "Caine" in sp and "Huy" in sp, sp
    assert "{assistant_name}" not in sp and "{user_name}" not in sp, sp
    print("test_system_prompt_substitutes_names passed")


def test_system_prompt_seeds_missing_prompt_md():
    real_prompt = main.prompt
    seeded = main.PROJECT_DIR / "prompt.md.seedtest"
    main.prompt = seeded
    try:
        sp = main.build_system_prompt("Caine", "Huy")
        assert seeded.exists(), "prompt file was not seeded"
        assert seeded.read_text(encoding="utf-8") == main.DEFAULT_SYSTEM_PROMPT
    finally:
        main.prompt = real_prompt
        seeded.unlink(missing_ok=True)
    assert "Caine" in sp and "Huy" in sp, sp
    assert "<cmd>" in sp, sp
    print("test_system_prompt_seeds_missing_prompt_md passed")


def test_parse_and_strip_tool_tags():
    reply = (
        "Sure. <note title='Coffee'>Huy likes coffee.</note>"
        "<cmd>echo hi</cmd><digest /><train />"
        "<cron expr='*/5 * * * *'>hydrate</cron>"
        "<cron at='2026-12-31 23:59'>happy new year</cron>"
    )
    tools = main.parse_tools(reply)
    names = [name for name, _ in tools]
    assert names == [
        "write_note", "run_command", "digest_notes", "train_adapter",
        "schedule_job", "schedule_job",
    ], names
    assert tools[0][1] == {"title": "Coffee", "body": "Huy likes coffee."}
    assert tools[1][1] == {"cmd": "echo hi"}
    assert tools[4][1] == {"schedule": "*/5 * * * *", "text": "hydrate"}
    assert tools[5][1] == {"schedule": "at 2026-12-31 23:59", "text": "happy new year"}
    assert main.strip_tool_tags(reply) == "Sure.", repr(main.strip_tool_tags(reply))
    print("test_parse_and_strip_tool_tags passed")


@contextmanager
def scratch_cron_file():
    real_file = main.CRON_FILE
    main.CRON_FILE = main.PROJECT_DIR / "cron_jobs.test.json"
    try:
        main.CRON_FILE.unlink(missing_ok=True)
        yield
    finally:
        main.CRON_FILE.unlink(missing_ok=True)
        main.CRON_FILE = real_file


def test_cron_matching():
    dt = datetime(2026, 7, 16, 9, 30)  # a Thursday
    assert main.cron_matches("* * * * *", dt)
    assert main.cron_matches("30 9 * * *", dt)
    assert not main.cron_matches("31 9 * * *", dt)
    assert main.cron_matches("*/15 * * * *", dt)
    assert not main.cron_matches("*/7 * * * *", dt)
    assert main.cron_matches("0-45 9 16 7 *", dt)
    assert main.cron_matches("30 9 * * 4", dt)  # cron weekday: Thursday = 4
    assert not main.cron_matches("30 9 * * 0", dt)
    assert main.cron_matches("30 9 * * 0,4", dt)
    assert main.validate_cron_expr("* * * *") is not None
    assert main.validate_cron_expr("bogus * * * *") is not None
    assert main.validate_cron_expr("*/10 8-18 * * 1-5") is None
    print("test_cron_matching passed")


def test_cron_jobs_fire_and_expire():
    with scratch_cron_file():
        config = main.load_config()
        one_shot = main.add_cron_job("at 2026-01-01 09:00", "wish Huy a happy new year")
        assert one_shot["schedule"] == "at 2026-01-01 09:00", one_shot
        main.add_cron_job("*/5 * * * *", "cmd:echo cron-ok")

        now = datetime(2026, 1, 1, 9, 5)
        events = main.check_due_jobs(config, now=now)
        assert any("happy new year" in e for e in events), events
        assert any("cron-ok" in e for e in events), events

        # One-shot is gone; recurring fires at most once per minute.
        assert main.check_due_jobs(config, now=now) == []
        events = main.check_due_jobs(config, now=datetime(2026, 1, 1, 9, 10))
        assert len(events) == 1 and "cron-ok" in events[0], events

        # Future one-shots stay quiet; bad schedules are rejected up front.
        main.add_cron_job("at 2099-01-01 00:00", "far future")
        assert main.check_due_jobs(config, now=datetime(2026, 1, 1, 9, 11)) == []
        try:
            main.add_cron_job("whenever", "x")
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
        jobs = main.load_cron_jobs()
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
    config = main.load_config()
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


def test_strip_unterminated_tag():
    cut_off = "Here are the stories:\n1. Big story — <cmd>open 'https://example.com/very/long"
    assert main.strip_tool_tags(cut_off) == "Here are the stories:\n1. Big story —"
    assert main.parse_tools(cut_off) == []
    print("test_strip_unterminated_tag passed")


def test_execute_code_tool():
    config = main.load_config()
    # <py> tag parses and strips
    reply = "Sure. <py>print(6 * 7)</py>"
    tools = main.parse_tools(reply)
    assert tools == [("execute_code", {"code": "print(6 * 7)"})], tools
    assert main.strip_tool_tags(reply) == "Sure."

    ok, out = main.run_python_code("import math\nprint(math.factorial(7))", config)
    assert ok and out == "5040", (ok, out)

    ok, out = main.run_python_code("import os\nprint(os.listdir('/'))", config)
    assert not ok and "not allowed" in out, (ok, out)
    ok, out = main.run_python_code("import socket", config)
    assert not ok and "not allowed" in out, (ok, out)
    ok, out = main.run_python_code("print(", config)
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
    config = main.load_config()

    reply = "<search>latest news</search><read>https://example.com</read>"
    tools = main.parse_tools(reply)
    assert ("web_search", {"query": "latest news"}) in tools, tools
    assert ("read_page", {"url": "https://example.com"}) in tools, tools
    assert main.strip_tool_tags(reply) == ""

    assert main.html_to_text("<p>Hello <b>world</b><script>bad()</script></p>") == "Hello world"

    real_get = main._http_get
    main._http_get = lambda url, timeout=15: _DDG_FIXTURE
    try:
        ok, out = main.web_search("anything", config)
    finally:
        main._http_get = real_get
    assert ok, out
    assert "Big Story Headline" in out and "https://example.com/story" in out, out
    assert "important" in out, out

    ok, out = main.read_page("file:///etc/passwd", config)
    assert not ok and "http" in out, (ok, out)
    print("test_web_tools passed")


def test_agent_loop_answers_from_search():
    real_search = main.web_search
    main.web_search = lambda q, c, max_results=5: (True, "1. Rain expected\n   https://example.com/wx\n   Heavy rain tomorrow.")
    try:
        session = ScriptedSession(
            user_inputs=["what is the latest news?", "/quit", "n"],
            model_replies=[
                "<search>latest news</search>",
                "The latest: heavy rain is expected tomorrow.",
            ],
        )
        session.run()
    finally:
        main.web_search = real_search
    assert len(session.prompts_seen) == 2
    assert "Rain expected" in session.prompts_seen[1], session.prompts_seen[1]
    assert "Web search for 'latest news' succeeded" in session.prompts_seen[1]
    print("test_agent_loop_answers_from_search passed")


@contextmanager
def scratch_config_file(initial: str = "{}"):
    real_cfg = main.CONFIG_FILE
    main.CONFIG_FILE = main.PROJECT_DIR / "config.test.json"
    try:
        main.CONFIG_FILE.write_text(initial)
        yield
    finally:
        main.CONFIG_FILE.unlink(missing_ok=True)
        main.CONFIG_FILE = real_cfg


def test_self_configuration():
    with scratch_config_file('{"agent": {"temperature": 0.1}, "unrelated": 42}'):
        config = main.load_config()

        tools = main.parse_tools("<config show /><config set='agent.temperature'>0.4</config>")
        assert ("config_show", {}) in tools, tools
        assert ("config_set", {"key": "agent.temperature", "value": "0.4"}) in tools, tools

        msg = main.set_config_value(config, "agent.temperature", "0.4")
        assert "Set agent.temperature" in msg, msg
        assert config["agent"]["temperature"] == 0.4
        saved = json.loads(main.CONFIG_FILE.read_text())
        assert saved["agent"]["temperature"] == 0.4 and saved["unrelated"] == 42, saved

        assert "Unknown config key" in main.set_config_value(config, "nope.nada", "1")
        assert "Bad value" in main.set_config_value(config, "agent.max_tool_rounds", "many")
        assert "restart" in main.set_config_value(config, "model_name", "other-model")
        # The model may not loosen its own sandbox; the user may via /config.
        assert "user" in main.set_config_value(config, "sandbox.blocked_commands", '["rm"]')
        assert "Set sandbox" in main.set_config_value(
            config, "sandbox.blocked_commands", '["rm"]', allow_sandbox=True)
        assert "Set memory.enabled" in main.set_config_value(config, "memory.enabled", "false")
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
        saved = json.loads(main.CONFIG_FILE.read_text())
        assert saved["agent"]["temperature"] == 0.9, saved
    print("test_agent_loop_applies_config_change passed")


@contextmanager
def scratch_memory_files():
    real_mem, real_prof = main.MEMORY_FILE, main.PROFILE_FILE
    main.MEMORY_FILE = main.PROJECT_DIR / "agent_memory.test.md"
    main.PROFILE_FILE = main.PROJECT_DIR / "user_profile.test.md"
    try:
        yield
    finally:
        main.MEMORY_FILE.unlink(missing_ok=True)
        main.PROFILE_FILE.unlink(missing_ok=True)
        main.MEMORY_FILE, main.PROFILE_FILE = real_mem, real_prof


def test_curated_memory_store():
    config = main.load_config()
    with scratch_memory_files():
        reply = "<memory>Repo uses MLX.</memory><profile replace='all'>Huy likes bullets.</profile>"
        tools = main.parse_tools(reply)
        assert ("save_memory", {"store": "memory", "content": "Repo uses MLX.", "replace": False}) in tools, tools
        assert ("save_memory", {"store": "profile", "content": "Huy likes bullets.", "replace": True}) in tools, tools
        assert main.strip_tool_tags(reply) == ""

        msg = main.save_memory("memory", "Repo uses MLX.", config)
        assert "Saved" in msg, msg
        main.save_memory("profile", "Huy likes bullets.", config)
        block = main.curated_memory_block(config)
        assert "Repo uses MLX." in block and "Huy likes bullets." in block, block

        # Over-limit append gets a consolidation nag; replace='all' shrinks it.
        msg = main.save_memory("memory", "x" * 3000, config)
        assert "over the limit" in msg, msg
        msg = main.save_memory("memory", "Only this.", config, replace=True)
        assert "over the limit" not in msg, msg
        assert main.MEMORY_FILE.read_text() == "Only this.\n"
    print("test_curated_memory_store passed")


def test_memory_injected_and_flushed():
    config = main.load_config()
    flush_min = config["memory"]["flush_min_turns"]
    with scratch_memory_files():
        main.MEMORY_FILE.write_text("Deploys happen on Fridays.\n")
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
        saved = main.MEMORY_FILE.read_text()
        assert "tests features right after asking" in saved, saved
    print("test_memory_injected_and_flushed passed")


def test_rag_injects_saved_notes():
    note_path = main.save_note(
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


def test_sandbox_blocks_dangerous_commands():
    config = main.load_config()
    ok, out = main.run_sandboxed("rm -rf /", config)
    assert not ok and "blocked" in out, (ok, out)
    ok, out = main.run_sandboxed("echo sandbox-ok", config)
    assert ok and out == "sandbox-ok", (ok, out)
    print("test_sandbox_blocks_dangerous_commands passed")


def run_all():
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
        test_rag_injects_saved_notes()
        test_sandbox_blocks_dangerous_commands()
        test_agent_loop_feeds_observation_back()
        test_agent_loop_stops_at_max_rounds()
        test_agent_loop_breaks_on_repeated_tool_call()
        test_strip_unterminated_tag()
        test_agent_loop_schedules_job_from_tag()
    print("\nAll main.py agent-loop tests passed.")


if __name__ == "__main__":
    run_all()
