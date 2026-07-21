"""Tests for the MoA dispatch mechanism (symbio.app.dispatch): the worker
catalog, WorkerPool's lazy-load/LRU-evict/idle-unload behavior, delegated
task execution, the <delegate> tag, and guarded_train_worker's golden-check
+ rollback story for a worker's own adapter."""

import json

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
