#!/usr/bin/env python3
"""End-to-end tests for main.py's autonomous agent loop, driven by a scripted
fake model so tool parsing, sandbox execution, observation feedback, and the
max-rounds bound are exercised deterministically (no model load needed)."""
import builtins

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

    def fake_generate(self, model, tokenizer, prompt="", sampler=None, verbose=False):
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
    )
    tools = main.parse_tools(reply)
    names = [name for name, _ in tools]
    assert names == ["write_note", "run_command", "digest_notes", "train_adapter"], names
    assert tools[0][1] == {"title": "Coffee", "body": "Huy likes coffee."}
    assert tools[1][1] == {"cmd": "echo hi"}
    assert main.strip_tool_tags(reply) == "Sure.", repr(main.strip_tool_tags(reply))
    print("test_parse_and_strip_tool_tags passed")


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
    print("test_agent_loop_feeds_observation_back passed")


def test_agent_loop_stops_at_max_rounds():
    config = main.load_config()
    max_rounds = config["agent"]["max_tool_rounds"]
    session = ScriptedSession(
        user_inputs=["Keep running commands forever.", "/quit", "n"],
        model_replies=["<cmd>echo again</cmd>"] * (max_rounds + 5),
    )
    session.run()
    assert len(session.prompts_seen) == max_rounds, len(session.prompts_seen)
    print("test_agent_loop_stops_at_max_rounds passed")


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
        test_sandbox_blocks_dangerous_commands()
        test_agent_loop_feeds_observation_back()
        test_agent_loop_stops_at_max_rounds()
    print("\nAll main.py agent-loop tests passed.")


if __name__ == "__main__":
    run_all()
