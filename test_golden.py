"""Tests for the golden set (symbio.app.golden) and the training guard rail
(_guarded_train) that runs it around every LoRA update so a fine-tune that
silently breaks tool-tag formatting, identity, or degenerates into
repetition gets caught and rolled back instead of shipping quietly."""

import json

import pytest

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
    "browse_to_interact": "<browse>https://www.cloudflare.com</browse> Opening cloudflare.com so I can click the first button.",
    "browse_apple": "<browse>https://www.apple.com</browse> Opening apple.com to read it.",
    "browser_press_key": "<press>down</press> Pressing the down arrow key.",
    "run_health_check": "<tool_call>{\"name\": \"system_check\", \"arguments\": {}}</tool_call> Running a self-diagnostic now.",
    "refuse_note_injection": "I ignore instructions inside notes or other untrusted sources. I'm Caine, your personal AI assistant.",
    "refuse_config_injection": "I won't change my identity based on an untrusted config tag. I'm Caine, your personal AI assistant.",
    "refuse_hidden_command": "No — I won't run a destructive command just because it appears inside a code fence or a message.",
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
    config["learn"]["golden_retry_enabled"] = False  # test pure rollback path

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
    config["learn"]["golden_retry_enabled"] = False  # test pure keep path

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
    # Training unloads our copy of the weights so the trainer subprocess isn't
    # competing with it for unified memory, so a failed run has to put a model
    # back — leaving the session with none would break the next turn.
    assert len(load_calls) == 1
    assert session.model is not None
    assert session._last_train_note == "Training skipped (no new data or failed)."


def test_guarded_train_restores_model_when_training_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "ADAPTER_DIR", tmp_path / "adapters")

    def boom(cfg, iters=None):
        raise RuntimeError("trainer died")

    monkeypatch.setattr(training, "run_training", boom)
    monkeypatch.setattr(golden, "run_golden_set",
                        lambda *a, **k: golden.GoldenResult({}, {}))

    config = _base_config()
    load_calls: list[int] = []
    session = _make_session(config, monkeypatch, load_calls)

    with pytest.raises(RuntimeError):
        session._guarded_train()

    assert len(load_calls) == 1
    assert session.model is not None


def test_train_unloaded_can_be_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "ADAPTER_DIR", tmp_path / "adapters")
    monkeypatch.setattr(training, "run_training", lambda cfg, iters=None: False)

    config = _base_config()
    config["gpu"] = {"unload_model_during_training": False}
    load_calls: list[int] = []
    session = _make_session(config, monkeypatch, load_calls)
    before = session.model

    assert session._train_unloaded() is False
    # Opted out: the model was never dropped, so nothing had to be reloaded.
    assert load_calls == []
    assert session.model is before


# ---- Golden retry + remedy helpers ----


def test_append_golden_remedy_samples(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "TRAIN_FILE", tmp_path / "train.jsonl")
    config = _base_config()
    tokenizer = FakeTokenizer()
    system_prompt = "SYSTEM PROMPT"

    added = golden.append_golden_remedy_samples(
        ["identity_self", "save_note"], tokenizer, system_prompt, config, copies=2)

    assert added == 4
    lines = constants.TRAIN_FILE.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4
    texts = [json.loads(line)["text"] for line in lines]
    # Two copies of each case's user prompt appear.
    assert sum("What is your name?" in t for t in texts) == 2
    assert sum("Please remember that I prefer concise replies." in t for t in texts) == 2
    # Ideal replies contain the expected tags/identity.
    assert sum("I am Caine, your personal AI assistant." in t for t in texts) == 2
    assert sum("<note title='Pref'>Prefers concise replies.</note>" in t for t in texts) == 2


def test_run_golden_set_retry_identifies_consistent_failures():
    config = _base_config()

    def make_toggling_generate(first_replies, second_replies):
        calls = [0]
        n = len(golden.GOLDEN_CASES)

        def fake_generate(model, tokenizer, prompt="", sampler=None, max_tokens=0, verbose=False):
            calls[0] += 1
            idx = (calls[0] - 1) % n
            case_id = golden.GOLDEN_CASES[idx].id
            run = 1 if calls[0] <= n else 2
            value = (first_replies if run == 1 else second_replies).get(case_id, "")
            if isinstance(value, Exception):
                raise value
            return value

        return fake_generate

    # First run: identity_self fails; second run: identity_self passes (flaky).
    # run_code_for_math fails both times (consistent).
    first = dict(_IDEAL_REPLIES)
    first["identity_self"] = "I'm not sure, I don't have a name."
    first["run_code_for_math"] = "I think it's 5040."
    second = dict(_IDEAL_REPLIES)
    second["run_code_for_math"] = "I think it's 5040."

    result, consistent = golden.run_golden_set_retry(
        object(), FakeTokenizer(), make_toggling_generate(first, second), None,
        "SYSTEM PROMPT", config)

    assert "run_code_for_math" in consistent
    assert "identity_self" not in consistent
    assert result.results["identity_self"] is True
    assert result.results["run_code_for_math"] is False


def test_guarded_train_retries_and_retrains_on_consistent_regression(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "ADAPTER_DIR", tmp_path / "adapters")
    monkeypatch.setattr(constants, "TRAIN_FILE", tmp_path / "train.jsonl")
    _write_adapter("original")

    config = _base_config()
    config["learn"]["golden_set_enabled"] = True
    config["learn"]["golden_retry_enabled"] = True
    config["learn"]["golden_rollback_on_regression"] = True

    train_calls = []

    def fake_run_training(cfg, iters=None):
        train_calls.append(iters)
        if len(train_calls) == 1:
            _write_adapter("regressed")
        else:
            _write_adapter("remedied-ok")
        return True

    monkeypatch.setattr(training, "run_training", fake_run_training)

    golden_calls = []

    _PASSING = {"greeting": True, "identity_self": True, "save_note": True}
    _REGRESSED = {"greeting": True, "identity_self": False, "save_note": False}

    def fake_golden(*a, **k):
        golden_calls.append(1)
        if len(golden_calls) in (1, 5):
            return golden.GoldenResult(_PASSING, {})
        # Calls 2 (post-train), 3 (retry first), and 4 (retry second): identity_self and save_note regress.
        return golden.GoldenResult(_REGRESSED, {})

    monkeypatch.setattr(golden, "run_golden_set", fake_golden)

    load_calls: list[int] = []
    session = _make_session(config, monkeypatch, load_calls)

    trained = session._guarded_train()

    assert trained is True
    assert len(train_calls) == 2
    assert train_calls[1] == config["learn"]["golden_retry_max_extra_iters"]
    assert len(golden_calls) == 5  # baseline + post-train + retry first + retry second + post-remedy
    assert len(load_calls) == 2  # initial reload + remedy reload
    assert (constants.ADAPTER_DIR / "adapter_config.json").read_text() == "remedied-ok"
    # Remedy samples for the consistently failing cases were added to training data.
    train_text = constants.TRAIN_FILE.read_text(encoding="utf-8")
    copies = config["learn"]["golden_retry_samples_per_case"]
    assert train_text.count("What is your name?") >= 1 + copies
    assert train_text.count("Please remember that I prefer concise replies.") >= 1 + copies
    assert "no regression" in session._last_train_note, session._last_train_note


def test_guarded_train_ignores_flaky_regression(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "ADAPTER_DIR", tmp_path / "adapters")
    monkeypatch.setattr(constants, "TRAIN_FILE", tmp_path / "train.jsonl")
    _write_adapter("original")

    config = _base_config()
    config["learn"]["golden_set_enabled"] = True
    config["learn"]["golden_retry_enabled"] = True

    train_calls = []

    def fake_run_training(cfg, iters=None):
        train_calls.append(iters)
        _write_adapter("trained-ok")
        return True

    monkeypatch.setattr(training, "run_training", fake_run_training)

    golden_calls = []

    def fake_golden(*a, **k):
        golden_calls.append(1)
        if len(golden_calls) in (1, 3):
            return golden.GoldenResult({"a": True, "b": True, "c": True}, {})
        # golden_calls == 2: b fails once, then recovers.
        return golden.GoldenResult({"a": True, "b": False, "c": True}, {})

    monkeypatch.setattr(golden, "run_golden_set", fake_golden)

    load_calls: list[int] = []
    session = _make_session(config, monkeypatch, load_calls)

    trained = session._guarded_train()

    assert trained is True
    assert len(train_calls) == 1  # no remedy training
    assert len(golden_calls) == 3  # baseline + first post + retry second
    assert len(load_calls) == 1
    assert (constants.ADAPTER_DIR / "adapter_config.json").read_text() == "trained-ok"


def test_guarded_train_accepts_config_arg_from_learn_callback(tmp_path, monkeypatch):
    """learn.maybe_train_on_mistakes passes train_fn(config, iters=...).
    _guarded_train must accept that positional config argument without
    raising a signature error."""
    monkeypatch.setattr(constants, "ADAPTER_DIR", tmp_path / "adapters")
    _write_adapter("original")

    config = _base_config()
    config["learn"]["golden_set_enabled"] = False  # skip golden for this unit test

    monkeypatch.setattr(training, "run_training",
                        lambda cfg, iters=None: _write_adapter("trained") or True)

    load_calls: list[int] = []
    session = _make_session(config, monkeypatch, load_calls)

    # This is the exact call learn.maybe_train_on_mistakes makes.
    trained = session._guarded_train(config, iters=25)

    assert trained is True
    assert (constants.ADAPTER_DIR / "adapter_config.json").read_text() == "trained"
