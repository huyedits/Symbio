"""Binding learned updates into the base weights, and taking them back out
(symbio.app.weights), plus the chat hook that lets the model decide."""

import json
import os
import time
from pathlib import Path

import pytest

from symbio import constants
from symbio.app import chat, golden, training, weights
from symbio.app import config as app_config

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

BASE = "org/base-model"


@pytest.fixture(autouse=True)
def _scratch(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "WEIGHTS_DIR", tmp_path / "weights")
    monkeypatch.setattr(constants, "ADAPTER_DIR", tmp_path / "adapters")
    monkeypatch.setattr(constants, "GOLDEN_CASES_FILE", tmp_path / "golden_cases.json")
    weights._TRIAL.clear()
    yield
    weights._TRIAL.clear()


def _adapter(d, pairs, scale=2.0, kind="lora", model=BASE):
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter_config.json").write_text(json.dumps(
        {"model": model, "fine_tune_type": kind, "num_layers": 1,
         "lora_parameters": {"rank": 2, "scale": scale, "dropout": 0.0}}))
    mx.save_safetensors(str(d / "adapters.safetensors"), pairs)
    return d


class Tiny(nn.Module):
    def __init__(self, quantized=False):
        super().__init__()
        mx.random.seed(0)
        self.dense = nn.Linear(64, 32, bias=True)
        q = nn.Linear(64, 32, bias=False)
        self.quant = nn.QuantizedLinear.from_linear(q, 32, 4) if quantized else q


def _pair(out_dims=32, in_dims=64, r=2):
    return mx.random.normal((in_dims, r)) * 0.1, mx.random.normal((r, out_dims)) * 0.1


# ---- the arithmetic ----

def test_fold_adds_the_lora_update_to_a_dense_weight(tmp_path):
    m = Tiny()
    w0 = mx.array(m.dense.weight)
    a, b = _pair()
    ad = _adapter(tmp_path / "a", {"dense.lora_a": a, "dense.lora_b": b}, scale=2.0)
    assert weights.fold(m, [ad]) == {"dense": None}
    want = w0 + (2.0 * b.T) @ a.T
    assert mx.allclose(m.dense.weight, want, atol=1e-5).item()


def test_two_folds_sum_and_keep_the_bias(tmp_path):
    m = Tiny()
    w0, b0 = mx.array(m.dense.weight), mx.array(m.dense.bias)
    a1, b1 = _pair()
    a2, b2 = _pair()
    d1 = _adapter(tmp_path / "1", {"dense.lora_a": a1, "dense.lora_b": b1}, scale=1.0)
    d2 = _adapter(tmp_path / "2", {"dense.lora_a": a2, "dense.lora_b": b2}, scale=3.0)
    weights.fold(m, [d1, d2])
    want = w0 + b1.T @ a1.T + (3.0 * b2.T) @ a2.T
    assert mx.allclose(m.dense.weight, want, atol=1e-5).item()
    assert mx.array_equal(m.dense.bias, b0).item()


def test_a_quantized_layer_stays_quantized_and_moves_toward_the_update(tmp_path):
    m = Tiny(quantized=True)
    q = m.quant
    w0 = mx.dequantize(q.weight, q.scales, q.biases, group_size=q.group_size, bits=q.bits)
    a, b = _pair()
    # An update far below the 4-bit step: requantized at 4 bits it would
    # round away; at the 8-bit fold width it survives.
    ad = _adapter(tmp_path / "a", {"quant.lora_a": a, "quant.lora_b": b}, scale=1.0)
    layout = weights.fold(m, [ad])
    assert layout == {"quant": {"group_size": 64, "bits": 8, "mode": "affine"}}
    assert isinstance(m.quant, nn.QuantizedLinear) and m.quant.bits == 8
    q = m.quant
    w1 = mx.dequantize(q.weight, q.scales, q.biases, group_size=q.group_size, bits=q.bits)
    want = w0 + b.T @ a.T
    # Requantizing costs precision, but the folded weight is far closer to
    # base+update than to the base it started from.
    assert mx.abs(w1 - want).mean().item() < 0.1 * mx.abs(w0 - want).mean().item()


def test_an_update_for_a_missing_module_is_refused(tmp_path):
    a, b = _pair()
    ad = _adapter(tmp_path / "a", {"nope.lora_a": a, "nope.lora_b": b})
    with pytest.raises(ValueError, match="no such module"):
        weights.fold(Tiny(), [ad])


def test_dora_and_full_fine_tunes_are_not_foldable(tmp_path):
    a, b = _pair()
    assert "dora" in weights.foldable(_adapter(tmp_path / "d", {"dense.lora_a": a}, kind="dora"))
    assert weights.foldable(tmp_path / "missing") is not None
    assert weights.foldable(_adapter(tmp_path / "ok", {"dense.lora_a": a, "dense.lora_b": b})) is None


# ---- where loads go ----

def test_resolve_is_the_name_until_a_version_is_bound(tmp_path):
    assert weights.resolve(BASE) == BASE
    v = tmp_path / "v1"
    v.mkdir()
    weights.save_ledger({BASE: {"bound": [], "path": str(v)}})
    assert weights.resolve(BASE) == BASE          # no config.json: not a model
    (v / "config.json").write_text("{}")
    assert weights.resolve(BASE) == str(v)
    weights._TRIAL[BASE] = None                   # a trial of "no deltas at all"
    assert weights.resolve(BASE) == BASE


def test_every_model_load_goes_through_resolve(monkeypatch, tmp_path):
    from symbio.app import modelload
    v = tmp_path / "v1"
    v.mkdir()
    (v / "config.json").write_text("{}")
    weights.save_ledger({BASE: {"bound": [], "path": str(v)}})
    seen = []
    monkeypatch.setattr(modelload, "_load_with_backend",
                        lambda args, kwargs, config: seen.append(args[0]) or (None, None))
    modelload.load(BASE)
    modelload.load(BASE, resolve=False)
    assert seen == [str(v), BASE]


def test_the_trainer_trains_on_the_bound_version(tmp_path):
    from symbio import backend
    v = tmp_path / "v1"
    v.mkdir()
    (v / "config.json").write_text("{}")
    weights.save_ledger({BASE: {"bound": [], "path": str(v)}})
    lora = {"batch_size": 1, "num_layers": 2, "learning_rate": 1e-5, "steps_per_eval": 50,
            "max_seq_length": 512, "save_every": 50, "rank": 8, "dropout": 0.0, "scale": 20.0}
    argv = backend.trainer_command(BASE, "d", "a", lora, 5, "c.yaml", config={"backend": "mlx"})
    assert argv[argv.index("--model") + 1] == str(v)


# ---- what the model is asked ----

def _live_adapter(age_days):
    a, b = _pair()
    d = _adapter(constants.ADAPTER_DIR, {"dense.lora_a": a, "dense.lora_b": b})
    t = time.time() - age_days * 86400
    os.utime(d / "adapters.safetensors", (t, t))
    return d


def test_a_proven_adapter_is_a_bind_candidate_and_a_new_one_is_not():
    cfg = {"model_name": BASE}
    _live_adapter(0.5)
    assert weights.candidates(cfg) == []
    _live_adapter(5)
    (c,) = weights.candidates(cfg)
    assert c["kind"] == "bind" and c["options"][0][0] == "no"


def test_an_adapter_for_another_model_is_never_offered():
    _live_adapter(30)
    assert weights.candidates({"model_name": "org/other"}) == []


def test_a_fold_followed_by_lost_checks_is_an_unbind_candidate():
    weights.save_ledger({BASE: {"bound": [
        {"id": "adapter-1", "archive": "x", "bound_at": "2026-10-01T00:00:00",
         "passing_before": ["greeting", "identity"]}], "path": None}})
    cfg = {"model_name": BASE}
    assert weights.candidates(cfg, failing_now={"weather"}) == []
    (c,) = weights.candidates(cfg, failing_now={"identity", "weather"})
    assert c["kind"] == "unbind" and c["id"] == "adapter-1" and c["lost"] == ["identity"]


# ---- trial, keep, roll back ----

def _fake_materialize(monkeypatch, calls):
    def fake(base, dirs, out):
        calls.append([str(d) for d in dirs])
        out.mkdir(parents=True)
        (out / "config.json").write_text("{}")
        return True, "ok"
    monkeypatch.setattr(weights, "materialize", fake)
    monkeypatch.setattr(weights, "disk_room_for", lambda base: None)


def test_bind_archives_first_and_commit_makes_it_the_base(monkeypatch):
    calls = []
    _fake_materialize(monkeypatch, calls)
    _live_adapter(5)
    (cand,) = weights.candidates({"model_name": BASE})
    ok, _, plan = weights.stage(cand)
    assert ok and weights.resolve(BASE) == plan["path"]
    arch = plan["bound"][-1]["archive"]
    assert (constants.WEIGHTS_DIR / "archive") in [p for p in Path(arch).parents]
    assert calls == [[arch]]
    weights.commit(plan, {"a"}, {"a", "b"})
    weights._TRIAL.clear()
    assert weights.resolve(BASE) == plan["path"]
    e = weights.entry(BASE)
    assert e["bound"][0]["passing_before"] == ["a"] and e["events"][0]["kind"] == "bind"


def test_abandon_puts_loads_back_and_keeps_the_archive(monkeypatch):
    _fake_materialize(monkeypatch, [])
    _live_adapter(5)
    (cand,) = weights.candidates({"model_name": BASE})
    ok, _, plan = weights.stage(cand)
    weights.abandon(plan)
    assert weights.resolve(BASE) == BASE
    assert not Path(plan["path"]).exists()
    assert (Path(plan["bound"][-1]["archive"]) / "adapters.safetensors").exists()
    assert weights.load_ledger() == {}


def test_unbind_rebuilds_without_it_and_the_old_version_goes(monkeypatch):
    calls = []
    _fake_materialize(monkeypatch, calls)
    old = constants.WEIGHTS_DIR / "org_base-model" / "v1"
    old.mkdir(parents=True)
    (old / "config.json").write_text("{}")
    weights.save_ledger({BASE: {"bound": [
        {"id": "adapter-1", "archive": "/arch/1", "passing_before": ["identity"]},
        {"id": "adapter-2", "archive": "/arch/2", "passing_before": []}],
        "path": str(old)}})
    ok, _, plan = weights.stage({"kind": "unbind", "base": BASE, "id": "adapter-1"})
    assert ok and calls == [["/arch/2"]] and plan["path"].endswith("v2")
    weights.commit(plan, set(), {"identity"})
    assert not old.exists()
    assert [r["id"] for r in weights.entry(BASE)["bound"]] == ["adapter-2"]


def test_unbinding_the_last_delta_is_the_original_base(monkeypatch):
    calls = []
    _fake_materialize(monkeypatch, calls)
    weights.save_ledger({BASE: {"bound": [{"id": "adapter-1", "archive": "/a"}], "path": None}})
    ok, _, plan = weights.stage({"kind": "unbind", "base": BASE, "id": "adapter-1"})
    assert ok and plan["path"] is None and weights.resolve(BASE) == BASE


def test_no_room_on_disk_changes_nothing(monkeypatch):
    monkeypatch.setattr(weights, "disk_room_for", lambda base: "needs 9 GB")
    _live_adapter(5)
    (cand,) = weights.candidates({"model_name": BASE})
    ok, msg, _ = weights.stage(cand)
    assert not ok and "9 GB" in msg and not (constants.WEIGHTS_DIR / "archive").exists()


# ---- the model decides, the gate judges ----

class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=False, enable_thinking=False):
        return "\n".join(m["content"] for m in messages)


def _session(monkeypatch, out):
    config = app_config.load_config()
    config["model_name"] = BASE
    monkeypatch.setattr(chat, "load", lambda *a, **k: (object(), FakeTokenizer()))
    s = chat.ChatSession(config, model=object(), tokenizer=FakeTokenizer(),
                         adapter_loaded=True, output_fn=out.append,
                         generate_fn=lambda *a, **k: "unused")
    monkeypatch.setattr(s, "_mlx_generation", lambda: True)
    monkeypatch.setattr(s, "_prefill_system_prompt_cache", lambda **k: None)
    monkeypatch.setattr(s, "_reload_model", lambda: None)
    return s


def _say(choice):
    return lambda model, tok, cand, config: {"choice": choice, "yes": choice == "yes",
                                             "probs": {"yes": 0.9 if choice == "yes" else 0.1}}


def test_the_model_saying_no_changes_nothing(monkeypatch):
    out = []
    s = _session(monkeypatch, out)
    _live_adapter(5)
    monkeypatch.setattr(weights, "decide", _say("no"))
    monkeypatch.setattr(weights, "stage", lambda c: pytest.fail("staged without a yes"))
    assert s._consider_reshape(golden.GoldenResult({"a": True}, {})) is None
    assert any("Considered folding" in line and ": no" in line for line in out)


def test_a_bind_that_breaks_a_check_is_rolled_back(monkeypatch):
    out = []
    s = _session(monkeypatch, out)
    _fake_materialize(monkeypatch, [])
    _live_adapter(5)
    monkeypatch.setattr(weights, "decide", _say("yes"))
    monkeypatch.setattr(golden, "run_golden_set_retry", lambda *a, **k: (
        golden.GoldenResult({"a": False, "b": True}, {}), {"a"}))
    assert s._consider_reshape(golden.GoldenResult({"a": True, "b": True}, {})) is None
    assert weights.resolve(BASE) == BASE and weights.load_ledger() == {}
    assert (constants.ADAPTER_DIR / "adapters.safetensors").exists()
    assert any("Rolled back" in line for line in out)


def test_a_bind_that_holds_is_kept_and_the_live_adapter_retires(monkeypatch):
    out = []
    s = _session(monkeypatch, out)
    _fake_materialize(monkeypatch, [])
    _live_adapter(5)
    monkeypatch.setattr(weights, "decide", _say("yes"))
    after = golden.GoldenResult({"a": True, "b": True}, {})
    monkeypatch.setattr(golden, "run_golden_set_retry", lambda *a, **k: (after, set()))
    assert s._consider_reshape(golden.GoldenResult({"a": True, "b": False}, {})) is after
    assert weights.resolve(BASE) != BASE
    assert not (constants.ADAPTER_DIR / "adapters.safetensors").exists()
    rec = weights.entry(BASE)["bound"][0]
    assert (Path(rec["archive"]) / "adapters.safetensors").exists()
    assert s.adapter_loaded is False


def test_a_one_off_failure_is_noise_not_a_regression(monkeypatch):
    out = []
    s = _session(monkeypatch, out)
    _fake_materialize(monkeypatch, [])
    _live_adapter(5)
    monkeypatch.setattr(weights, "decide", _say("yes"))
    after = golden.GoldenResult({"a": True}, {})
    monkeypatch.setattr(golden, "run_golden_set_retry", lambda *a, **k: (after, set()))
    assert s._consider_reshape(golden.GoldenResult({"a": True}, {})) is after


def test_an_unbind_must_bring_back_what_it_was_blamed_for(monkeypatch):
    out = []
    s = _session(monkeypatch, out)
    _fake_materialize(monkeypatch, [])
    weights.save_ledger({BASE: {"bound": [
        {"id": "adapter-1", "archive": "/a", "bound_at": "2026-10-01T00:00:00",
         "passing_before": ["identity"]}], "path": None}})
    monkeypatch.setattr(weights, "decide", _say("yes"))
    still = golden.GoldenResult({"identity": False}, {})
    monkeypatch.setattr(golden, "run_golden_set_retry", lambda *a, **k: (still, {"identity"}))
    assert s._consider_reshape(golden.GoldenResult({"identity": False}, {})) is None
    assert [r["id"] for r in weights.entry(BASE)["bound"]] == ["adapter-1"]


def test_switched_off_by_config_asks_nothing(monkeypatch):
    out = []
    s = _session(monkeypatch, out)
    s.config["weights"] = {"self_reshape": False}
    _live_adapter(5)
    monkeypatch.setattr(weights, "decide", lambda *a: pytest.fail("asked while off"))
    assert s._consider_reshape(golden.GoldenResult({"a": True}, {})) is None


def test_the_training_round_asks_before_it_trains(monkeypatch):
    out = []
    s = _session(monkeypatch, out)
    order = []
    monkeypatch.setattr(golden, "run_golden_set",
                        lambda *a, **k: golden.GoldenResult({"a": True}, {}))
    monkeypatch.setattr(s, "_consider_reshape", lambda base: order.append("reshape"))
    monkeypatch.setattr(training, "backup_adapter", lambda *a, **k: order.append("backup"))
    monkeypatch.setattr(s, "_train_unloaded", lambda **k: False)
    monkeypatch.setattr(s, "_restore_model", lambda: None)
    s._guarded_train()
    assert order[:2] == ["reshape", "backup"]
