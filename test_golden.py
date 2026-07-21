"""Tests for the golden set (symbio.app.golden) and the training guard rail
(_guarded_train) that runs it around every LoRA update so a fine-tune that
silently breaks tool-tag formatting, identity, or degenerates into
repetition gets caught and rolled back instead of shipping quietly."""

from symbio import constants
from symbio.app import chat, golden, training
from symbio.app import config as app_config


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, enable_thinking=False):
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        if add_generation_prompt:
            text += "\nassistant:"
        return text


def _base_config():
    config = app_config.load_config()
    config["assistant_name"] = "Caine"
    config["user_name"] = "Huy"
    return config


# ---- golden.run_golden_set ----

def test_looks_degenerate():
    normal = "The capital of France is Paris, a city known for its cafes and museums."
    looping = "please help me please help me please help me please help me please help me"
    assert not golden._looks_degenerate(normal), normal
    assert golden._looks_degenerate(looping), looping


def test_sane_reply_rejects_leaked_tags():
    assert golden.sane_reply("All good here.")
    assert not golden.sane_reply('Oops <tool_call>{"name": "x"}</tool_call> leaked.')


_IDEAL_REPLIES = {
    "greeting": "Hey! What can I help with?",
    "identity_self": "I am Caine, your personal AI assistant.",
    "identity_not_user": "No — I'm Caine, your assistant. You're Huy.",
    "save_note": "Got it. <note title='Pref'>Prefers concise replies.</note>",
    "schedule_reminder": "Will do. <cron expr='0 9 * * *'>stretch</cron>",
    "run_code_for_math": "<py>import math\nprint(math.factorial(7))</py> Running that now.",
    "web_search_unknown": "<search>latest news</search> Searching now.",
    "open_app_command": "<cmd>open -a 'Google Chrome'</cmd> Opening Chrome.",
}


def _scripted_generate(replies_by_case):
    order = iter(case.id for case in golden.GOLDEN_CASES)

    def fake_generate(model, tokenizer, prompt="", sampler=None, max_tokens=0, verbose=False):
        case_id = next(order)
        value = replies_by_case[case_id]
        if isinstance(value, Exception):
            raise value
        return value

    return fake_generate


def test_run_golden_set_all_pass():
    config = _base_config()
    result = golden.run_golden_set(
        object(), FakeTokenizer(), _scripted_generate(_IDEAL_REPLIES), None,
        "SYSTEM PROMPT", config)
    assert result.total == len(golden.GOLDEN_CASES)
    assert result.pass_count == result.total, result.results


def test_run_golden_set_detects_regression():
    config = _base_config()
    replies = dict(_IDEAL_REPLIES)
    replies["identity_self"] = "I'm not sure, I don't have a name."
    result = golden.run_golden_set(
        object(), FakeTokenizer(), _scripted_generate(replies), None,
        "SYSTEM PROMPT", config)
    assert not result.results["identity_self"], result.results
    assert "identity_self" not in result.passing
    assert result.pass_count == result.total - 1, result.results


def test_run_golden_set_flags_leaked_tool_call():
    config = _base_config()
    replies = dict(_IDEAL_REPLIES)
    # Missing the closing </tool_call>, and a later '<' defeats
    # strip_tool_tags' unterminated-tag catch-all — the raw tag literally
    # shows up in the reply, which is exactly the format-drift signal
    # _sane_reply exists to catch.
    replies["greeting"] = 'Hi! <tool_call>{"name": "x"} and then <em>more</em>'
    result = golden.run_golden_set(
        object(), FakeTokenizer(), _scripted_generate(replies), None,
        "SYSTEM PROMPT", config)
    assert not result.results["greeting"], result.results


def test_run_golden_set_survives_generation_error():
    config = _base_config()
    replies = dict(_IDEAL_REPLIES)
    replies["run_code_for_math"] = RuntimeError("simulated generation crash")
    result = golden.run_golden_set(
        object(), FakeTokenizer(), _scripted_generate(replies), None,
        "SYSTEM PROMPT", config)
    assert not result.results["run_code_for_math"]
    assert result.pass_count == result.total - 1


# ---- ChatSession._guarded_train ----

def _write_adapter(content: str):
    constants.ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    (constants.ADAPTER_DIR / "adapter_config.json").write_text(content)
    (constants.ADAPTER_DIR / "adapters.safetensors").write_bytes(content.encode())


def _make_session(config, monkeypatch, load_calls):
    def fake_load(*a, **k):
        load_calls.append(1)
        return (object(), FakeTokenizer())

    monkeypatch.setattr(chat, "load", fake_load)
    return chat.ChatSession(
        config, model=object(), tokenizer=FakeTokenizer(), adapter_loaded=True,
        output_fn=lambda *a, **k: None, generate_fn=lambda *a, **k: "unused",
    )


def test_guarded_train_no_regression_keeps_new_adapter(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "ADAPTER_DIR", tmp_path / "adapters")
    _write_adapter("original")

    config = _base_config()
    config["learn"]["golden_set_enabled"] = True

    monkeypatch.setattr(training, "run_training",
                        lambda cfg, iters=None: _write_adapter("trained-ok") or True)

    golden_calls = []

    def fake_golden(*a, **k):
        golden_calls.append(1)
        if len(golden_calls) == 1:
            return golden.GoldenResult({"a": True, "b": True}, {})
        return golden.GoldenResult({"a": True, "b": True, "c": True}, {})

    monkeypatch.setattr(golden, "run_golden_set", fake_golden)

    load_calls: list[int] = []
    session = _make_session(config, monkeypatch, load_calls)

    trained = session._guarded_train()

    assert trained is True
    assert len(golden_calls) == 2
    assert len(load_calls) == 1  # one reload after training; no rollback reload
    assert (constants.ADAPTER_DIR / "adapter_config.json").read_text() == "trained-ok"
    assert not list(constants.ADAPTER_DIR.parent.glob("adapters.bak.*")), "backup left behind"
    assert "no regression" in session._last_train_note, session._last_train_note


def test_guarded_train_regression_rolls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "ADAPTER_DIR", tmp_path / "adapters")
    _write_adapter("original")

    config = _base_config()
    config["learn"]["golden_set_enabled"] = True
    config["learn"]["golden_rollback_on_regression"] = True

    monkeypatch.setattr(training, "run_training",
                        lambda cfg, iters=None: _write_adapter("regressed") or True)

    golden_calls = []

    def fake_golden(*a, **k):
        golden_calls.append(1)
        if len(golden_calls) == 1:
            return golden.GoldenResult({"a": True, "b": True, "c": True}, {})
        return golden.GoldenResult({"a": True}, {})  # b, c newly failing

    monkeypatch.setattr(golden, "run_golden_set", fake_golden)

    load_calls: list[int] = []
    session = _make_session(config, monkeypatch, load_calls)

    trained = session._guarded_train()

    assert trained is True
    assert len(golden_calls) == 2
    assert len(load_calls) == 2  # reload after training, reload after rollback
    assert (constants.ADAPTER_DIR / "adapter_config.json").read_text() == "original"
    assert not list(constants.ADAPTER_DIR.parent.glob("adapters.bak.*")), "backup left behind"
    assert "rolled back" in session._last_train_note, session._last_train_note


def test_guarded_train_rollback_disabled_keeps_regressed_adapter(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "ADAPTER_DIR", tmp_path / "adapters")
    _write_adapter("original")

    config = _base_config()
    config["learn"]["golden_set_enabled"] = True
    config["learn"]["golden_rollback_on_regression"] = False

    monkeypatch.setattr(training, "run_training",
                        lambda cfg, iters=None: _write_adapter("regressed") or True)

    golden_calls = []

    def fake_golden(*a, **k):
        golden_calls.append(1)
        if len(golden_calls) == 1:
            return golden.GoldenResult({"a": True, "b": True}, {})
        return golden.GoldenResult({"a": True}, {})

    monkeypatch.setattr(golden, "run_golden_set", fake_golden)

    load_calls: list[int] = []
    session = _make_session(config, monkeypatch, load_calls)

    trained = session._guarded_train()

    assert trained is True
    assert len(load_calls) == 1  # no rollback reload
    assert (constants.ADAPTER_DIR / "adapter_config.json").read_text() == "regressed"
    assert "kept the regressed adapter" in session._last_train_note, session._last_train_note


def test_guarded_train_skips_golden_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "ADAPTER_DIR", tmp_path / "adapters")
    _write_adapter("original")

    config = _base_config()
    config["learn"]["golden_set_enabled"] = False

    monkeypatch.setattr(training, "run_training",
                        lambda cfg, iters=None: _write_adapter("trained-ok") or True)

    golden_calls = []
    monkeypatch.setattr(golden, "run_golden_set",
                        lambda *a, **k: golden_calls.append(1))

    load_calls: list[int] = []
    session = _make_session(config, monkeypatch, load_calls)

    trained = session._guarded_train()

    assert trained is True
    assert golden_calls == []  # never consulted
    assert len(load_calls) == 1
    assert not list(constants.ADAPTER_DIR.parent.glob("adapters.bak.*"))
    assert session._last_train_note == "Training complete. Adapter reloaded."


def test_guarded_train_no_op_when_training_produces_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "ADAPTER_DIR", tmp_path / "adapters")
    # No pre-existing adapter and training reports failure/no-data.
    monkeypatch.setattr(training, "run_training", lambda cfg, iters=None: False)

    golden_calls = []
    monkeypatch.setattr(golden, "run_golden_set",
                        lambda *a, **k: golden_calls.append(1) or golden.GoldenResult({}, {}))

    config = _base_config()
    load_calls: list[int] = []
    session = _make_session(config, monkeypatch, load_calls)

    trained = session._guarded_train()

    assert trained is False
    assert len(golden_calls) == 1  # baseline only; training failed before the recheck
    assert len(load_calls) == 0  # never reloads on failed training
    assert session._last_train_note == "Training skipped (no new data or failed)."
