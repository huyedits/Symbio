"""Tests for the MoA dispatch mechanism (symbio.app.dispatch): the worker
catalog, WorkerPool's lazy-load/LRU-evict/idle-unload behavior, delegated
task execution, the <delegate> tag, and guarded_train_worker's golden-check
+ rollback story for a worker's own adapter."""

import json

import pytest

from symbio import constants
from symbio.app import chat, dispatch, golden, tooling, training
from symbio.app import config as app_config


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, enable_thinking=False):
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        if add_generation_prompt:
            text += "\nassistant:"
        return text


def _write_catalog(monkeypatch, tmp_path, entries):
    catalog_file = tmp_path / "worker_models.json"
    catalog_file.write_text(json.dumps(entries), encoding="utf-8")
    monkeypatch.setattr(constants, "WORKER_MODELS_FILE", catalog_file)


def _isolate_dirs(monkeypatch, tmp_path):
    # Mirror constants.py's own module-load-time mkdir for the real paths —
    # code like /status's ADAPTER_DIR.iterdir() assumes the dir exists.
    (tmp_path / "adapters").mkdir(parents=True, exist_ok=True)
    (tmp_path / "training_data").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(constants, "ADAPTER_DIR", tmp_path / "adapters")
    monkeypatch.setattr(constants, "WORKER_ADAPTERS_DIR", tmp_path / "adapters" / "workers")
    monkeypatch.setattr(constants, "DATA_DIR", tmp_path / "training_data")


# ---- catalog ----

def test_load_catalog_missing_file_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(constants, "WORKER_MODELS_FILE", tmp_path / "nope.json")
    assert dispatch.load_catalog() == {}


def test_catalog_entry_for_role(monkeypatch, tmp_path):
    _write_catalog(monkeypatch, tmp_path, {
        "s": {"model_name": "m/s", "role": "summarize"},
        "b": {"model_name": "m/b", "role": "browser"},
    })
    entry = dispatch.catalog_entry_for_role("summarize")
    assert entry is not None and entry["model_name"] == "m/s"
    assert dispatch.catalog_entry_for_role("nonexistent") is None


# ---- WorkerPool ----

def _fake_load_factory(load_calls):
    def fake_load(model_name, adapter_path=None):
        load_calls.append((model_name, adapter_path))
        return (object(), FakeTokenizer())
    return fake_load


def test_worker_pool_lazy_loads_once_per_role(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    _write_catalog(monkeypatch, tmp_path, {
        "s": {"model_name": "m/s", "role": "summarize"},
    })
    load_calls = []
    monkeypatch.setattr(dispatch, "load", _fake_load_factory(load_calls))

    pool = dispatch.WorkerPool({"dispatch": {}})
    r1 = pool.get("summarize")
    r2 = pool.get("summarize")
    assert r1 is not None and r2 is not None
    assert len(load_calls) == 1, "second get() should reuse the resident worker"
    assert pool.loaded_roles() == ["summarize"]


def test_worker_pool_unknown_role_returns_none(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    _write_catalog(monkeypatch, tmp_path, {})
    pool = dispatch.WorkerPool({"dispatch": {}})
    assert pool.get("nonexistent") is None


def test_worker_pool_evicts_lru_when_over_max_resident(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    _write_catalog(monkeypatch, tmp_path, {
        "s": {"model_name": "m/s", "role": "summarize"},
        "b": {"model_name": "m/b", "role": "browser"},
    })
    load_calls = []
    monkeypatch.setattr(dispatch, "load", _fake_load_factory(load_calls))

    pool = dispatch.WorkerPool({"dispatch": {"max_resident_workers": 1}})
    pool.get("summarize")
    pool.get("browser")  # should evict "summarize" (over the limit of 1)
    assert pool.loaded_roles() == ["browser"]

    # Getting "summarize" again reloads it — proves it was actually evicted.
    pool.get("summarize")
    assert len(load_calls) == 3, load_calls


def test_worker_pool_respects_higher_max_resident_workers(monkeypatch, tmp_path):
    """The user explicitly wants this to be a real, working setting for
    people with more RAM — not a permanently-1 placeholder."""
    _isolate_dirs(monkeypatch, tmp_path)
    _write_catalog(monkeypatch, tmp_path, {
        "s": {"model_name": "m/s", "role": "summarize"},
        "b": {"model_name": "m/b", "role": "browser"},
    })
    load_calls = []
    monkeypatch.setattr(dispatch, "load", _fake_load_factory(load_calls))

    pool = dispatch.WorkerPool({"dispatch": {"max_resident_workers": 2}})
    pool.get("summarize")
    pool.get("browser")
    assert sorted(pool.loaded_roles()) == ["browser", "summarize"], (
        "both should stay resident when max_resident_workers=2")


def test_worker_pool_evicts_idle_workers(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    _write_catalog(monkeypatch, tmp_path, {
        "s": {"model_name": "m/s", "role": "summarize"},
    })
    monkeypatch.setattr(dispatch, "load", _fake_load_factory([]))

    pool = dispatch.WorkerPool({"dispatch": {"worker_idle_unload_minutes": 10}})
    pool.get("summarize")
    # Simulate time passing by rewriting the recorded last-used timestamp.
    model, tok, ts = pool._resident["summarize"]
    pool._resident["summarize"] = (model, tok, ts - 11 * 60)
    pool._evict_idle()
    assert pool.loaded_roles() == []


def test_worker_pool_run_delegated_task_records_training_sample(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    _write_catalog(monkeypatch, tmp_path, {
        "s": {"model_name": "m/s", "role": "summarize"},
    })
    monkeypatch.setattr(dispatch, "load", _fake_load_factory([]))
    monkeypatch.setattr(dispatch, "generate",
                        lambda *a, **k: "A short summary of the text.")

    pool = dispatch.WorkerPool({"dispatch": {}})
    result = pool.run_delegated_task("summarize", "Some long text to summarize.")
    assert result == "A short summary of the text."

    train_file = constants.data_dir_for("summarize") / "train.jsonl"
    assert train_file.exists()
    assert "A short summary" in train_file.read_text()


def test_worker_pool_run_delegated_task_unknown_role(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    _write_catalog(monkeypatch, tmp_path, {})
    pool = dispatch.WorkerPool({"dispatch": {}})
    result = pool.run_delegated_task("nonexistent", "do something")
    assert "no worker" in result.lower()


def test_worker_pool_run_delegated_task_survives_generation_error(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    _write_catalog(monkeypatch, tmp_path, {
        "s": {"model_name": "m/s", "role": "summarize"},
    })
    monkeypatch.setattr(dispatch, "load", _fake_load_factory([]))

    def boom(*a, **k):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(dispatch, "generate", boom)
    pool = dispatch.WorkerPool({"dispatch": {}})
    result = pool.run_delegated_task("summarize", "text")
    assert "failed" in result.lower()


# ---- <delegate> tag ----

def test_parse_delegate_tag():
    reply = "Let me get that summarized. <delegate role='summarize'>The full page text here.</delegate>"
    tools = tooling.parse_tools(reply)
    assert tools == [("delegate_task", {"role": "summarize", "task": "The full page text here."})], tools
    assert tooling.strip_tool_tags(reply) == "Let me get that summarized."


def test_parse_delegate_tag_respects_enabled_groups():
    reply = "<delegate role='summarize'>text</delegate>"
    assert tooling.parse_tools(reply, enabled_groups={"delegate"}) != []
    assert tooling.parse_tools(reply, enabled_groups={"terminal"}) == []


# ---- ChatSession._execute_tool wiring ----

def _make_session(config, monkeypatch):
    monkeypatch.setattr(chat, "load", lambda *a, **k: (object(), FakeTokenizer()))
    return chat.ChatSession(
        config, model=object(), tokenizer=FakeTokenizer(), adapter_loaded=False,
        output_fn=lambda *a, **k: None, generate_fn=lambda *a, **k: "unused",
    )


def test_execute_tool_delegate_disabled_by_default(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    config = app_config.load_config()
    config["dispatch"]["enabled"] = False
    assert config["dispatch"]["enabled"] is False
    session = _make_session(config, monkeypatch)
    result = session._execute_tool("delegate_task", {"role": "summarize", "task": "x"})
    assert "disabled" in result.lower()


def test_execute_tool_delegate_enabled_routes_to_pool(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    _write_catalog(monkeypatch, tmp_path, {
        "s": {"model_name": "m/s", "role": "summarize"},
    })
    monkeypatch.setattr(dispatch, "load", _fake_load_factory([]))
    monkeypatch.setattr(dispatch, "generate", lambda *a, **k: "Summary text.")

    config = app_config.load_config()
    config["dispatch"]["enabled"] = True
    # A pre-existing config.json's own tools.enabled_groups (frozen at
    # whatever tools existed when it was written) wins over the default in
    # load_config()'s merge — explicitly include "delegate" here so this
    # test exercises the routing logic itself, not that merge behavior.
    config["tools"]["enabled_groups"] = list(config["tools"]["enabled_groups"]) + ["delegate"]
    session = _make_session(config, monkeypatch)
    result = session._execute_tool("delegate_task", {"role": "summarize", "task": "long text"})
    assert result == "Summary text."


# ---- guarded_train_worker ----

def _write_worker_adapter(role, content):
    d = constants.adapter_dir_for(role)
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_config.json").write_text(content)
    (d / "adapters.safetensors").write_bytes(content.encode())


def test_guarded_train_worker_no_prior_adapter_trains_without_golden_check(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    _write_catalog(monkeypatch, tmp_path, {
        "s": {"model_name": "m/s", "role": "summarize"},
    })
    monkeypatch.setattr(dispatch, "load", _fake_load_factory([]))
    monkeypatch.setattr(training, "run_training",
                        lambda cfg, iters=None, role=None, model_name=None:
                            _write_worker_adapter(role, "trained") or True)

    config = app_config.load_config()
    trained, msg = dispatch.guarded_train_worker("summarize", config)
    assert trained is True
    assert "trained" in msg.lower()
    assert (constants.adapter_dir_for("summarize") / "adapter_config.json").exists()


def test_guarded_train_worker_regression_rolls_back(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    _write_catalog(monkeypatch, tmp_path, {
        "s": {"model_name": "m/s", "role": "summarize"},
    })
    _write_worker_adapter("summarize", "original")

    monkeypatch.setattr(dispatch, "load", _fake_load_factory([]))
    monkeypatch.setattr(training, "run_training",
                        lambda cfg, iters=None, role=None, model_name=None:
                            _write_worker_adapter(role, "regressed") or True)

    calls = []

    def fake_golden(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            return golden.GoldenResult({"summarize_produces_output": True}, {})
        return golden.GoldenResult({"summarize_produces_output": False}, {})

    monkeypatch.setattr(golden, "run_golden_set", fake_golden)

    config = app_config.load_config()
    config["dispatch"]["worker_golden_rollback_on_regression"] = True
    trained, msg = dispatch.guarded_train_worker("summarize", config)
    assert trained is True
    assert "rolled back" in msg.lower()
    assert (constants.adapter_dir_for("summarize") / "adapter_config.json").read_text() == "original"


def test_guarded_train_worker_regression_remedies_then_succeeds(monkeypatch, tmp_path):
    """If a worker regresses but remedy samples push the golden check back to
    passing, the adapter should stay in place, not roll back."""
    _isolate_dirs(monkeypatch, tmp_path)
    _write_catalog(monkeypatch, tmp_path, {
        "s": {"model_name": "m/s", "role": "summarize"},
    })
    _write_worker_adapter("summarize", "original")

    train_calls = []

    def fake_run_training(cfg, iters=None, role=None, model_name=None):
        train_calls.append(iters)
        _write_worker_adapter(role, "remedied" if iters else "regressed")
        return True

    monkeypatch.setattr(dispatch, "load", _fake_load_factory([]))
    monkeypatch.setattr(training, "run_training", fake_run_training)

    calls = []

    def fake_golden(*a, **k):
        calls.append(1)
        # baseline passes, post-train fails, recheck fails, remedy recheck passes
        if len(calls) in (1, 5):
            return golden.GoldenResult({"summarize_produces_output": True}, {})
        return golden.GoldenResult({"summarize_produces_output": False}, {})

    monkeypatch.setattr(golden, "run_golden_set", fake_golden)

    config = app_config.load_config()
    config["dispatch"]["worker_golden_rollback_on_regression"] = True
    trained, msg = dispatch.guarded_train_worker("summarize", config)
    assert trained is True
    assert "rolled back" not in msg.lower()
    assert "50" in msg or "checks passing" in msg.lower()
    assert len(train_calls) == 2, "should train once, then remedy-train again"
    assert train_calls[1] == 50
    train_file = constants.data_dir_for("summarize") / "train.jsonl"
    assert train_file.exists()
    assert "bike lane network" in train_file.read_text()
    assert (constants.adapter_dir_for("summarize") / "adapter_config.json").read_text() == "remedied"


def test_guarded_train_worker_unknown_role(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    _write_catalog(monkeypatch, tmp_path, {})
    config = app_config.load_config()
    trained, msg = dispatch.guarded_train_worker("nonexistent", config)
    assert trained is False
    assert "no worker" in msg.lower()


# ---- browser delegation ----

class FakeBrowser:
    """Scripted page: each action returns a canned status, then the page
    text changes to whatever's configured for that step."""

    def __init__(self, pages, statuses=None):
        self.pages = list(pages)  # page text returned by get_text(), consumed one per call after actions
        self.statuses = statuses or {}
        self.calls: list[tuple] = []

    def get_text(self):
        return self.pages[0]

    def click(self, text=""):
        self.calls.append(("click", text))
        self.pages.pop(0)
        return self.statuses.get(("click", text), f"Clicked element containing text '{text}'.")

    def type_text(self, text="", press_enter=False):
        self.calls.append(("type", text))
        self.pages.pop(0)
        return f"Typed '{text}'."

    def scroll(self, direction="down"):
        self.calls.append(("scroll", direction))
        self.pages.pop(0)
        return f"Scrolled {direction} 800px."


def test_browser_delegation_clicks_then_finishes(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    _write_catalog(monkeypatch, tmp_path, {
        "b": {"model_name": "m/b", "role": "browser"},
    })
    monkeypatch.setattr(dispatch, "load", _fake_load_factory([]))

    actions = iter(["click: Sign in", "done: logged in"])
    monkeypatch.setattr(dispatch, "generate", lambda *a, **k: next(actions))

    browser = FakeBrowser(pages=["Sign in link visible.", "You are now logged in."])
    pool = dispatch.WorkerPool({"dispatch": {"max_worker_rounds": 4}})
    result = pool.run_delegated_task("browser", "Log in", browser=browser)

    assert "logged in" in result.lower()
    assert browser.calls == [("click", "Sign in")]


def test_browser_delegation_stops_after_max_rounds(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    _write_catalog(monkeypatch, tmp_path, {
        "b": {"model_name": "m/b", "role": "browser"},
    })
    monkeypatch.setattr(dispatch, "load", _fake_load_factory([]))
    monkeypatch.setattr(dispatch, "generate", lambda *a, **k: "scroll")

    browser = FakeBrowser(pages=["page"] * 10)
    pool = dispatch.WorkerPool({"dispatch": {"max_worker_rounds": 3}})
    result = pool.run_delegated_task("browser", "Find something", browser=browser)

    assert "did not finish" in result.lower()
    assert len(browser.calls) == 3


def test_browser_delegation_stops_on_unrecognized_action(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    _write_catalog(monkeypatch, tmp_path, {
        "b": {"model_name": "m/b", "role": "browser"},
    })
    monkeypatch.setattr(dispatch, "load", _fake_load_factory([]))
    monkeypatch.setattr(dispatch, "generate", lambda *a, **k: "I think I should look around")

    browser = FakeBrowser(pages=["page"])
    pool = dispatch.WorkerPool({"dispatch": {}})
    result = pool.run_delegated_task("browser", "Find something", browser=browser)

    assert "unrecognized" in result.lower()
    assert browser.calls == []


def test_browser_delegation_records_training_samples(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    _write_catalog(monkeypatch, tmp_path, {
        "b": {"model_name": "m/b", "role": "browser"},
    })
    monkeypatch.setattr(dispatch, "load", _fake_load_factory([]))
    monkeypatch.setattr(dispatch, "generate", lambda *a, **k: "done: nothing to do")

    browser = FakeBrowser(pages=["empty page"])
    pool = dispatch.WorkerPool({"dispatch": {}})
    pool.run_delegated_task("browser", "Check the page", browser=browser)

    train_file = constants.data_dir_for("browser") / "train.jsonl"
    assert train_file.exists()
    assert "done: nothing to do" in train_file.read_text()


def test_execute_tool_delegate_browser_role_passes_session_browser(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    _write_catalog(monkeypatch, tmp_path, {
        "b": {"model_name": "m/b", "role": "browser"},
    })
    monkeypatch.setattr(dispatch, "load", _fake_load_factory([]))
    monkeypatch.setattr(dispatch, "generate", lambda *a, **k: "done: ok")

    config = app_config.load_config()
    config["dispatch"]["enabled"] = True
    config["tools"]["enabled_groups"] = list(config["tools"]["enabled_groups"]) + ["delegate"]
    session = _make_session(config, monkeypatch)
    session.browser = FakeBrowser(pages=["some page"])

    result = session._execute_tool("delegate_task", {"role": "browser", "task": "check it"})
    assert "done: ok" in result.lower() or "finished" in result.lower()


# ---- /train_worker and /status wiring ----

def test_train_worker_command_missing_role_shows_usage(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    config = app_config.load_config()
    session = _make_session(config, monkeypatch)
    outputs = []
    session.output_fn = outputs.append

    session._handle_command("/train_worker")
    assert any("usage" in o.lower() for o in outputs)


def test_train_worker_command_trains_named_role(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    _write_catalog(monkeypatch, tmp_path, {
        "s": {"model_name": "m/s", "role": "summarize"},
    })
    monkeypatch.setattr(dispatch, "load", _fake_load_factory([]))
    monkeypatch.setattr(training, "run_training",
                        lambda cfg, iters=None, role=None, model_name=None:
                            _write_worker_adapter(role, "trained") or True)

    config = app_config.load_config()
    session = _make_session(config, monkeypatch)
    outputs = []
    session.output_fn = outputs.append

    session._handle_command("/train_worker summarize")
    assert any("trained" in o.lower() for o in outputs)


def test_status_shows_dispatch_state(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    config = app_config.load_config()
    session = _make_session(config, monkeypatch)
    outputs = []
    session.output_fn = outputs.append

    session._handle_command("/status")
    assert any("dispatch" in o.lower() for o in outputs)


# ---- adapter hot-swap ----
#
# Skill workers all run the headmaster's own base weights and differ only by a
# ~19 MB adapter, so loading a second full copy to switch between them costs
# gigabytes to change nothing else. These cover the swap and, more importantly,
# the cases where it must refuse: load_weights(strict=False) drops keys it
# cannot place, so a wrong-shaped adapter would leave the previous worker's
# weights attached and answer as the wrong specialist without any error.

class _FakeModel:
    """Stands in for an MLX model with LoRA layers already attached.

    The default fake in this file is a bare object(), which has no
    load_weights and so silently sends every swap down the fallback path.
    """

    def __init__(self):
        self.loaded = []

    def load_weights(self, path, strict=True):
        self.loaded.append(path)


def _swappable_load_factory(load_calls):
    def fake_load(model_name, adapter_path=None):
        load_calls.append((model_name, adapter_path))
        return (_FakeModel(), FakeTokenizer())
    return fake_load


def _adapter(tmp_path, role, shapes):
    """Write a stand-in adapter file for `role` and return its directory."""
    adapter_dir = tmp_path / "adapters" / "workers" / role
    adapter_dir.mkdir(parents=True, exist_ok=True)
    (adapter_dir / "adapters.safetensors").write_bytes(b"weights")
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    return adapter_dir


def _stub_shape_check(monkeypatch, verdict):
    monkeypatch.setattr(dispatch, "_adapter_fits_model",
                        lambda model, path: verdict)


def _two_skill_catalog(monkeypatch, tmp_path):
    _write_catalog(monkeypatch, tmp_path, {
        "skill_a": {"model_name": "same/model", "role": "alpha", "is_skill": True},
        "skill_b": {"model_name": "same/model", "role": "beta", "is_skill": True},
    })


def test_same_base_model_swaps_the_adapter_instead_of_reloading(
        monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    _two_skill_catalog(monkeypatch, tmp_path)
    _adapter(tmp_path, "alpha", {})
    _adapter(tmp_path, "beta", {})
    load_calls = []
    monkeypatch.setattr(dispatch, "load", _swappable_load_factory(load_calls))
    monkeypatch.setattr(dispatch.training, "mark_adapter_used", lambda **k: None)
    _stub_shape_check(monkeypatch, True)

    pool = dispatch.WorkerPool({"dispatch": {}})
    pool.get("alpha")
    assert len(load_calls) == 1

    result = pool.get("beta")

    assert result is not None
    assert len(load_calls) == 1, "switching skills must not reload the model"
    # Residency moves: the donor's model now carries beta's adapter.
    assert pool.loaded_roles() == ["beta"]


def test_a_mismatched_adapter_refuses_to_swap(monkeypatch, tmp_path):
    """The silent-wrong-answer case: strict=False would place nothing."""
    _isolate_dirs(monkeypatch, tmp_path)
    _two_skill_catalog(monkeypatch, tmp_path)
    _adapter(tmp_path, "alpha", {})
    _adapter(tmp_path, "beta", {})
    load_calls = []
    monkeypatch.setattr(dispatch, "load", _swappable_load_factory(load_calls))
    monkeypatch.setattr(dispatch.training, "mark_adapter_used", lambda **k: None)
    _stub_shape_check(monkeypatch, False)

    pool = dispatch.WorkerPool({"dispatch": {}})
    pool.get("alpha")
    pool.get("beta")

    assert len(load_calls) == 2, "a shape mismatch must fall back to a full load"


def test_different_base_models_never_swap(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    _write_catalog(monkeypatch, tmp_path, {
        "a": {"model_name": "big/model", "role": "alpha"},
        "b": {"model_name": "small/model", "role": "beta"},
    })
    _adapter(tmp_path, "alpha", {})
    _adapter(tmp_path, "beta", {})
    load_calls = []
    monkeypatch.setattr(dispatch, "load", _swappable_load_factory(load_calls))
    monkeypatch.setattr(dispatch.training, "mark_adapter_used", lambda **k: None)
    _stub_shape_check(monkeypatch, True)

    pool = dispatch.WorkerPool({"dispatch": {}})
    pool.get("alpha")
    pool.get("beta")

    assert len(load_calls) == 2, "different weights cannot share a model"


def test_a_worker_with_no_adapter_is_loaded_in_full(monkeypatch, tmp_path):
    """Reusing a model here would answer as the previous specialist."""
    _isolate_dirs(monkeypatch, tmp_path)
    _two_skill_catalog(monkeypatch, tmp_path)
    _adapter(tmp_path, "alpha", {})
    load_calls = []
    monkeypatch.setattr(dispatch, "load", _swappable_load_factory(load_calls))
    monkeypatch.setattr(dispatch.training, "mark_adapter_used", lambda **k: None)
    _stub_shape_check(monkeypatch, True)

    pool = dispatch.WorkerPool({"dispatch": {}})
    pool.get("alpha")
    pool.get("beta")          # beta has no adapter on disk

    assert len(load_calls) == 2


def test_hot_swap_can_be_switched_off(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    _two_skill_catalog(monkeypatch, tmp_path)
    _adapter(tmp_path, "alpha", {})
    _adapter(tmp_path, "beta", {})
    load_calls = []
    monkeypatch.setattr(dispatch, "load", _swappable_load_factory(load_calls))
    monkeypatch.setattr(dispatch.training, "mark_adapter_used", lambda **k: None)
    _stub_shape_check(monkeypatch, True)

    pool = dispatch.WorkerPool({"dispatch": {"hot_swap_adapters": False}})
    pool.get("alpha")
    pool.get("beta")

    assert len(load_calls) == 2


def test_shape_check_rejects_an_adapter_whose_tensors_do_not_fit(tmp_path):
    """Exercises the real check, not the stub."""
    import mlx.core as mx

    path = tmp_path / "adapters.safetensors"
    mx.save_safetensors(str(path), {"layers.0.lora_a": mx.zeros((8, 16))})

    class Model:
        def __init__(self, shape):
            self._shape = shape

        def parameters(self):
            return {"layers": [{"lora_a": mx.zeros(self._shape)}]}

    assert dispatch._adapter_fits_model(Model((8, 16)), path) is True
    assert dispatch._adapter_fits_model(Model((4, 16)), path) is False


# ---- refusing a second copy of the headmaster's weights ----
#
# The headmaster's model lives in ChatSession, never in WorkerPool._resident,
# so it is never a hot-swap donor: a worker on the headmaster's own model is
# always a full second copy of the largest allocation on the machine. That is
# what OOM'd a 16 GB Mac mini beside a training run, so it is refused unless
# the headmaster unloads first or the operator opts in.

def _headmaster_sized_catalog(monkeypatch, tmp_path):
    _write_catalog(monkeypatch, tmp_path, {
        "skill_a": {"model_name": "same/model", "role": "alpha", "is_skill": True},
        "small": {"model_name": "other/small", "role": "summarize"},
    })


def _refusal_pool(monkeypatch, tmp_path, dispatch_cfg):
    _isolate_dirs(monkeypatch, tmp_path)
    _headmaster_sized_catalog(monkeypatch, tmp_path)
    _adapter(tmp_path, "alpha", {})
    load_calls = []
    monkeypatch.setattr(dispatch, "load", _swappable_load_factory(load_calls))
    monkeypatch.setattr(dispatch.training, "mark_adapter_used", lambda **k: None)
    _stub_shape_check(monkeypatch, False)
    pool = dispatch.WorkerPool({"model_name": "same/model", "dispatch": dispatch_cfg})
    return pool, load_calls


def test_headmaster_sized_worker_is_refused_instead_of_doubling_ram(
        monkeypatch, tmp_path):
    pool, load_calls = _refusal_pool(monkeypatch, tmp_path, {})

    with pytest.raises(dispatch.SecondHeadmasterCopyRefused):
        pool.get("alpha")
    assert load_calls == [], "refused worker must not have been loaded anyway"


def test_a_worker_smaller_than_the_headmaster_is_never_refused(
        monkeypatch, tmp_path):
    pool, load_calls = _refusal_pool(monkeypatch, tmp_path, {})

    assert pool.get("summarize") is not None
    assert load_calls == [("other/small", None)]


def test_deep_sleep_allows_a_headmaster_sized_worker(monkeypatch, tmp_path):
    """The headmaster unloads itself first, so only one copy is ever resident."""
    pool, load_calls = _refusal_pool(
        monkeypatch, tmp_path, {"headmaster_deep_sleep_while_workers": True})

    assert pool.get("alpha") is not None
    assert len(load_calls) == 1


def test_the_second_copy_can_be_allowed_explicitly(monkeypatch, tmp_path):
    pool, load_calls = _refusal_pool(
        monkeypatch, tmp_path, {"allow_second_headmaster_copy": True})

    assert pool.get("alpha") is not None
    assert len(load_calls) == 1


def test_delegate_reports_a_refusal_rather_than_raising(monkeypatch, tmp_path):
    pool, _ = _refusal_pool(monkeypatch, tmp_path, {})

    reply = pool.run_delegated_task("alpha", "do the thing")

    assert "was not run" in reply
    assert "two full copies" in reply


# ---- naming LoRA recipe drift ----

def test_recipe_drift_names_the_mismatched_lora_keys(tmp_path):
    """The failure that started this: retraining the headmaster with masked
    prompts and narrower targets left every older skill adapter unswappable,
    with nothing in the logs saying which knob moved."""
    worker, headmaster = tmp_path / "w", tmp_path / "h"
    for d, keys in ((worker, None), (headmaster, ["self_attn.q_proj",
                                                  "self_attn.v_proj"])):
        d.mkdir()
        lora = {"rank": 8}
        if keys is not None:
            lora["keys"] = keys
        (d / "adapter_config.json").write_text(
            json.dumps({"num_layers": 8, "lora_parameters": lora}), encoding="utf-8")

    drift = dispatch.describe_recipe_drift(worker, headmaster)

    assert drift is not None
    assert "keys" in drift
    assert "self_attn.q_proj" in drift


def test_matching_recipes_report_no_drift(tmp_path):
    worker, headmaster = tmp_path / "w", tmp_path / "h"
    for d in (worker, headmaster):
        d.mkdir()
        (d / "adapter_config.json").write_text(
            json.dumps({"num_layers": 8, "lora_parameters": {"rank": 8}}),
            encoding="utf-8")

    assert dispatch.describe_recipe_drift(worker, headmaster) is None


def test_drift_is_unreported_rather_than_guessed_when_provenance_is_missing(
        tmp_path):
    """Older adapters predate the stamp; silence beats a fabricated diff."""
    worker, headmaster = tmp_path / "w", tmp_path / "h"
    worker.mkdir()
    headmaster.mkdir()
    (worker / "adapter_config.json").write_text("{}", encoding="utf-8")

    assert dispatch.describe_recipe_drift(worker, headmaster) is None


# ---- advertising real worker roles ----

def test_delegate_schema_lists_the_workers_that_exist(monkeypatch, tmp_path):
    """A model shown a made-up example role cannot route to a real skill."""
    _write_catalog(monkeypatch, tmp_path, {
        "skill_fix_wifi": {"model_name": "m", "role": "fix_wifi",
                           "is_skill": True, "skill_name": "Fix wifi"},
        "summarize_worker": {"model_name": "m", "role": "summarize",
                             "description": "Condense text."},
    })

    roles = tooling.refresh_delegate_roles()

    assert set(roles) == {"fix_wifi", "summarize"}
    schema = next(t for t in tooling.tool_schemas() if t["name"] == "delegate_task")
    role_schema = schema["parameters"]["properties"]["role"]
    assert set(role_schema["enum"]) == {"fix_wifi", "summarize"}
    # The bare slug, not the catalog key: `skill_fix_wifi` is what the old
    # description told the model to send, and catalog_entry_for_role would
    # never have resolved it.
    assert "skill_fix_wifi" not in role_schema["description"]
    assert "Fix wifi" in role_schema["description"]


def test_advertised_roles_are_the_ones_the_pool_can_resolve(monkeypatch, tmp_path):
    """The advertised name and the dispatcher's lookup key must agree."""
    _write_catalog(monkeypatch, tmp_path, {
        "skill_fix_wifi": {"model_name": "m", "role": "fix_wifi",
                           "is_skill": True, "skill_name": "Fix wifi"},
    })

    for role in tooling.refresh_delegate_roles():
        assert dispatch.catalog_entry_for_role(role) is not None


def test_an_empty_catalog_leaves_the_schema_alone(monkeypatch, tmp_path):
    _write_catalog(monkeypatch, tmp_path, {})
    assert tooling.refresh_delegate_roles() == []


# ---- skills are golden-checked at last ----
#
# WORKER_GOLDEN_CASES only ever held entries for 'summarize' and 'browser', so
# guarded_train_worker looked a skill up, got None, and took the early return:
# no baseline, no recheck, no rollback. The workers that retrain themselves
# unattended were the only ones with no regression gate.

def _skill_catalog(monkeypatch, tmp_path, steps="1. Toggle wifi off. 2. Toggle it on."):
    _write_catalog(monkeypatch, tmp_path, {
        "skill_fix_wifi": {
            "model_name": "m/8b", "role": "fix_wifi", "is_skill": True,
            "skill_name": "Fix wifi",
            # The *served* prompt carries the steps as a safety net.
            "system_prompt": f"You are the 'Fix wifi' skill.\nSteps:\n{steps}\n\nReply with the steps.",
        },
    })
    return steps


def test_a_skill_now_has_golden_cases(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    _skill_catalog(monkeypatch, tmp_path)
    from symbio.app import skill_eval

    assert dispatch.WORKER_GOLDEN_CASES.get("fix_wifi") is None, "still none by hand"
    derived = skill_eval.golden_cases_for_role("fix_wifi")
    assert derived and len(derived) >= 3


def test_a_regressed_skill_retrain_is_rolled_back(monkeypatch, tmp_path):
    """The whole point: a skill that forgets its steps must not ship."""
    _isolate_dirs(monkeypatch, tmp_path)
    _skill_catalog(monkeypatch, tmp_path)

    adapter_dir = tmp_path / "adapters" / "workers" / "fix_wifi"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "adapters.safetensors").write_bytes(b"before")

    monkeypatch.setattr(dispatch, "load",
                        lambda name, adapter_path=None: (object(), FakeTokenizer()))
    monkeypatch.setattr(dispatch.training, "run_training", lambda *a, **k: True)
    monkeypatch.setattr(dispatch.training, "mark_adapter_used", lambda **k: None)

    replies = iter(["1. Toggle wifi off. 2. Toggle it on."] * 5      # healthy baseline
                   + ["I don't know."] * 20)                        # broken after
    monkeypatch.setattr(dispatch, "generate",
                        lambda *a, **k: next(replies))

    restored = []
    real_restore = dispatch.training.restore_adapter
    monkeypatch.setattr(dispatch.training, "restore_adapter",
                        lambda backup, role=None: restored.append(role))

    trained, message = dispatch.guarded_train_worker("fix_wifi", {"dispatch": {}})

    assert trained is True
    assert restored == ["fix_wifi"], "a regressed skill must be rolled back"
    assert "regressed" in message and "rolled back" in message


def test_a_healthy_skill_retrain_is_kept(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    _skill_catalog(monkeypatch, tmp_path)

    adapter_dir = tmp_path / "adapters" / "workers" / "fix_wifi"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "adapters.safetensors").write_bytes(b"before")

    monkeypatch.setattr(dispatch, "load",
                        lambda name, adapter_path=None: (object(), FakeTokenizer()))
    monkeypatch.setattr(dispatch.training, "run_training", lambda *a, **k: True)
    monkeypatch.setattr(dispatch.training, "mark_adapter_used", lambda **k: None)
    monkeypatch.setattr(dispatch, "generate",
                        lambda *a, **k: "1. Toggle wifi off. 2. Toggle it on.")

    restored = []
    monkeypatch.setattr(dispatch.training, "restore_adapter",
                        lambda backup, role=None: restored.append(role))

    trained, message = dispatch.guarded_train_worker("fix_wifi", {"dispatch": {}})

    assert trained is True
    assert restored == [], "a skill that still recites its steps must be kept"


def test_a_skill_is_graded_without_its_steps_in_the_prompt(monkeypatch, tmp_path):
    """Otherwise the check measures copying, not the weights.

    A skill's served prompt includes the procedure as a safety net for a weak
    adapter. Grading against that prompt would let a broken adapter pass by
    reading its answer out of context, which is exactly the failure the gate
    exists to catch.
    """
    _isolate_dirs(monkeypatch, tmp_path)
    _skill_catalog(monkeypatch, tmp_path)

    adapter_dir = tmp_path / "adapters" / "workers" / "fix_wifi"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "adapters.safetensors").write_bytes(b"before")

    seen_prompts = []

    def fake_generate(model, tokenizer, prompt=None, **kw):
        seen_prompts.append(prompt or "")
        return "1. Toggle wifi off. 2. Toggle it on."

    monkeypatch.setattr(dispatch, "load",
                        lambda name, adapter_path=None: (object(), FakeTokenizer()))
    monkeypatch.setattr(dispatch.training, "run_training", lambda *a, **k: True)
    monkeypatch.setattr(dispatch.training, "mark_adapter_used", lambda **k: None)
    monkeypatch.setattr(dispatch.training, "restore_adapter", lambda *a, **k: None)
    monkeypatch.setattr(dispatch, "generate", fake_generate)

    dispatch.guarded_train_worker("fix_wifi", {"dispatch": {}})

    assert seen_prompts, "the golden set should have run"
    for prompt in seen_prompts:
        assert "Toggle wifi off" not in prompt, (
            "the procedure leaked into the prompt the adapter is graded under")


# ---- worker adapters must match their base model ----
#
# mlx_lm sizes LoRA layers to the new model then loads with strict=False, so an
# adapter from a different base is discarded silently: a 4B loaded with an 8B
# adapter (hidden 2560 vs 4096) comes up untrained with no error anywhere.

def _stamped_adapter(tmp_path, role, trained_for):
    d = tmp_path / "adapters" / "workers" / role
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_config.json").write_text(
        json.dumps({"model": trained_for, "num_layers": 8}), encoding="utf-8")
    (d / "adapters.safetensors").write_bytes(b"w")
    return d


def test_an_adapter_from_another_model_is_not_a_match(tmp_path):
    d = _stamped_adapter(tmp_path, "r", "Qwen/Qwen3-8B-MLX-4bit")
    assert dispatch.adapter_matches_model(d, "mlx-community/Qwen3-4B-4bit") is False


def test_the_same_weights_republished_still_match(tmp_path):
    d = _stamped_adapter(tmp_path, "r", "Qwen/Qwen3-8B-MLX-4bit")
    assert dispatch.adapter_matches_model(d, "mlx-community/Qwen3-8B-MLX-4bit") is True


def test_an_unstamped_adapter_is_given_the_benefit_of_the_doubt(tmp_path):
    """Older adapters predate the stamp; refusing them would be a regression."""
    d = tmp_path / "adapters" / "workers" / "r"
    d.mkdir(parents=True)
    (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert dispatch.adapter_matches_model(d, "any/model") is True


def test_a_mismatched_worker_adapter_is_not_loaded(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    _write_catalog(monkeypatch, tmp_path, {
        "w": {"model_name": "mlx-community/Qwen3-4B-4bit", "role": "r"},
    })
    _stamped_adapter(tmp_path, "r", "Qwen/Qwen3-8B-MLX-4bit")
    calls = []
    monkeypatch.setattr(dispatch, "load", _fake_load_factory(calls))
    monkeypatch.setattr(dispatch.training, "mark_adapter_used", lambda **k: None)

    dispatch.WorkerPool({"dispatch": {}}).get("r")

    assert calls == [("mlx-community/Qwen3-4B-4bit", None)], (
        "the stale adapter should have been skipped, not silently discarded")


def test_a_matching_worker_adapter_is_loaded(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    _write_catalog(monkeypatch, tmp_path, {
        "w": {"model_name": "mlx-community/Qwen3-4B-4bit", "role": "r"},
    })
    d = _stamped_adapter(tmp_path, "r", "mlx-community/Qwen3-4B-4bit")
    calls = []
    monkeypatch.setattr(dispatch, "load", _fake_load_factory(calls))
    monkeypatch.setattr(dispatch.training, "mark_adapter_used", lambda **k: None)

    dispatch.WorkerPool({"dispatch": {}}).get("r")

    assert calls == [("mlx-community/Qwen3-4B-4bit", str(d))]
