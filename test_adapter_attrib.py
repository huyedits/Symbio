#!/usr/bin/env python3
"""Attributing a regression to LoRA modules, and repairing instead of rolling back.

The claim under test is narrow and mechanical: LoRA's contribution is additive
per module, so switching one off is an EXPERIMENT rather than a guess. These
tests hold that mechanism to its two promises — an ablated adapter is byte-valid
and differs only where intended, and the search finds a culprit set whose
removal actually works.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from symbio.app import adapter_attrib as A

REAL_ADAPTER = Path("adapters.backup")
has_real = (REAL_ADAPTER / A.ADAPTER_FILE).exists()


def _fake_adapter(root: Path, layers=(0, 1, 2, 3), projections=("q_proj", "v_proj")):
    """A miniature adapter with the same key shape as a real one."""
    from safetensors.numpy import save_file

    root.mkdir(parents=True, exist_ok=True)
    weights = {}
    for layer in layers:
        for proj in projections:
            stem = f"model.layers.{layer}.self_attn.{proj}"
            weights[f"{stem}.lora_a"] = np.ones((8, 2), dtype=np.float32)
            weights[f"{stem}.lora_b"] = np.ones((2, 8), dtype=np.float32)
    save_file(weights, str(root / A.ADAPTER_FILE))
    (root / "adapter_config.json").write_text('{"fine_tune_type": "lora"}\n')
    return root


# ------------------------------------------------------------------ ablation

def test_modules_are_one_per_layer_and_projection(tmp_path):
    mods = A.modules(_fake_adapter(tmp_path / "a"))
    assert len(mods) == 8
    assert mods[0] == "model.layers.0.self_attn.q_proj"
    # Ordered by layer, so a bisection cuts along the model's own structure
    # rather than alphabetically — "the deepest layers" stays contiguous.
    assert [int(m.split(".")[2]) for m in mods] == [0, 0, 1, 1, 2, 2, 3, 3]


def test_ablation_zeroes_only_lora_b_of_the_named_modules(tmp_path):
    src = _fake_adapter(tmp_path / "src")
    mods = A.modules(src)
    drop = {mods[1], mods[4]}
    assert A.write_ablated(src, tmp_path / "out", drop) == 2

    before, after = A.load_weights(src), A.load_weights(tmp_path / "out")
    assert set(before) == set(after), "an ablation must not change the key set"
    for key in before:
        stem, part = key.rsplit(".", 1)
        if stem in drop and part == "lora_b":
            assert np.all(after[key] == 0), key
        else:
            assert np.array_equal(before[key], after[key]), key


def test_ablation_carries_the_config_across(tmp_path):
    """A weights file without adapter_config.json is not an adapter, and the
    loader says so."""
    src = _fake_adapter(tmp_path / "src")
    A.write_ablated(src, tmp_path / "out", [])
    assert (tmp_path / "out" / "adapter_config.json").exists()


def test_lora_a_is_left_alone(tmp_path):
    """B·A is zero if B is. Zeroing both would work too and destroy twice as
    much information for nothing — the A side is what the repair keeps if the
    module is ever brought back."""
    src = _fake_adapter(tmp_path / "src")
    mods = A.modules(src)
    A.write_ablated(src, tmp_path / "out", [mods[0]])
    after = A.load_weights(tmp_path / "out")
    assert np.all(after[f"{mods[0]}.lora_a"] == 1)
    assert np.all(after[f"{mods[0]}.lora_b"] == 0)


@pytest.mark.skipif(not has_real, reason="no real adapter on disk")
def test_a_real_adapter_ablates_cleanly():
    """The shapes above are a guess until a real adapter agrees with them."""
    mods = A.modules(REAL_ADAPTER)
    assert mods, "no modules found in the real adapter"
    assert all(".self_attn." in m for m in mods)
    assert len({m.rsplit(".", 1)[-1] for m in mods}) <= 4


# -------------------------------------------------------------------- search

def _planted(culprits):
    """A check that passes exactly when every planted culprit is switched off."""
    want = set(culprits)
    return lambda dropped: want <= set(dropped)


def test_a_single_culprit_is_found_and_minimal():
    mods = [f"m{i}" for i in range(16)]
    found, calls = A.bisect_blame(mods, _planted([mods[5]]))
    assert found == [mods[5]]
    assert calls < len(mods), "a bisection that costs a sweep is not a bisection"


def test_culprits_at_either_end_are_found():
    mods = [f"m{i}" for i in range(16)]
    for target in (mods[0], mods[-1]):
        found, _ = A.bisect_blame(mods, _planted([target]))
        assert found == [target]


def test_an_interaction_is_found_not_missed():
    """A plain binary search tries only halves, finds neither, and reports
    nothing. Two modules that only break things together are exactly the case
    that would be silently lost."""
    mods = [f"m{i}" for i in range(16)]
    want = [mods[2], mods[11]]
    found, _ = A.bisect_blame(mods, _planted(want))
    assert set(want) <= set(found)


def test_whatever_comes_back_actually_works():
    """The invariant that makes an early stop safe: the returned set always
    satisfies the check, even when the budget cut the search short. A superset
    repairs; it is just not minimal."""
    mods = [f"m{i}" for i in range(16)]
    for want in ([mods[3]], [mods[1], mods[9]], mods[4:7]):
        check = _planted(want)
        found, _ = A.bisect_blame(mods, check, max_calls=12)
        assert check(set(found)), (want, found)


def test_an_unattributable_regression_says_so():
    """Zeroing everything and still failing means the adapter is not what broke
    these cases — a real answer, and a different one from "no culprit found"."""
    mods = [f"m{i}" for i in range(8)]
    found, calls = A.bisect_blame(mods, lambda dropped: False)
    assert found == []
    assert calls == 1, "one evaluation should settle it"


def test_the_budget_is_honoured():
    calls = []

    def _greedy(dropped):
        calls.append(1)
        return len(dropped) >= 1

    A.bisect_blame([f"m{i}" for i in range(64)], _greedy, max_calls=5)
    assert len(calls) <= 5


# ---------------------------------------------------------------- quarantine

def test_quarantine_records_events_not_a_set(tmp_path):
    """The same module implicated twice, months apart, on two different cases
    is a much stronger signal than one implicated once, and a set throws that
    away."""
    A.record_quarantine(tmp_path, ["m1"], ["case_a"], "2026-09-01 10:00")
    A.record_quarantine(tmp_path, ["m1", "m2"], ["case_b"], "2026-09-14 11:00")
    entries = A.read_quarantine(tmp_path)
    assert len(entries) == 2
    assert entries[1]["cases"] == ["case_b"]
    assert A.repeat_offenders(tmp_path, minimum=2) == ["m1"]
    assert A.repeat_offenders(tmp_path, minimum=3) == []


def test_a_corrupt_quarantine_costs_only_itself(tmp_path):
    (tmp_path / A.QUARANTINE_FILE).write_text("{not json", encoding="utf-8")
    assert A.read_quarantine(tmp_path) == []


# --------------------------------------------------------- in-memory ablation

def test_switching_off_is_exact_and_reversible():
    mlx_nn = pytest.importorskip("mlx.nn")
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.tuner.lora import LoRALinear

    class Block(mlx_nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = LoRALinear.from_base(mlx_nn.Linear(8, 8), r=2)

    class Tiny(mlx_nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = [Block(), Block()]

    model = Tiny()
    mods = A.live_lora_modules(model)
    assert len(mods) == 2
    for module in mods.values():
        module.lora_b = mx.ones_like(module.lora_b)
    name = sorted(mods)[0]

    with A.switched_off(model, [name]):
        assert float(mx.sum(mods[name].lora_b)) == 0.0
        other = sorted(mods)[1]
        assert float(mx.sum(mods[other].lora_b)) != 0.0
    assert float(mx.sum(mods[name].lora_b)) != 0.0


def test_a_failed_evaluation_still_restores_the_model():
    """An evaluation that raises must not leave the resident model quietly
    missing half its adapter for the rest of the session."""
    mlx_nn = pytest.importorskip("mlx.nn")
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.tuner.lora import LoRALinear

    class Tiny(mlx_nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = LoRALinear.from_base(mlx_nn.Linear(8, 8), r=2)

    model = Tiny()
    mods = A.live_lora_modules(model)
    name = next(iter(mods))
    mods[name].lora_b = mx.ones_like(mods[name].lora_b)
    with pytest.raises(RuntimeError):
        with A.switched_off(model, [name]):
            raise RuntimeError("evaluation blew up")
    assert float(mx.sum(mods[name].lora_b)) != 0.0


def test_live_names_map_onto_file_names():
    """The file says model.layers.28...; the loaded tree may say layers.28...
    Matching them by suffix is what keeps the repair writable to disk."""
    mapped = A.match_file_names(
        ["layers.28.self_attn.q_proj"],
        ["model.layers.28.self_attn.q_proj", "model.layers.29.self_attn.v_proj"])
    assert mapped == {"layers.28.self_attn.q_proj":
                      "model.layers.28.self_attn.q_proj"}


def test_an_unwalkable_model_declines_instead_of_raising():
    """Repair sits in front of the rollback path, so anything it cannot handle
    must make it decline quietly — the caller's next move is the rollback that
    was always going to happen. Raising there took the whole training run down
    with an AttributeError."""
    assert A.live_lora_modules(object()) == {}
    assert A.live_lora_modules(None) == {}

    class Hostile:
        def named_modules(self):
            raise RuntimeError("not really a model")

    assert A.live_lora_modules(Hostile()) == {}


# ------------------------------------------------- damping, not just ablation

def test_scaling_multiplies_only_lora_b_of_the_named_modules(tmp_path):
    src = _fake_adapter(tmp_path / "src")
    mods = A.modules(src)
    assert A.write_scaled(src, tmp_path / "out", {mods[0]: 0.5}) == 1
    after = A.load_weights(tmp_path / "out")
    assert np.allclose(after[f"{mods[0]}.lora_b"], 0.5)
    assert np.allclose(after[f"{mods[0]}.lora_a"], 1.0), "the A side is untouched"
    assert np.allclose(after[f"{mods[1]}.lora_b"], 1.0), "other modules untouched"


def test_ablation_is_the_zero_case_of_scaling(tmp_path):
    """One code path, so the two can never disagree about what 'off' means."""
    src = _fake_adapter(tmp_path / "src")
    mods = A.modules(src)
    A.write_ablated(src, tmp_path / "a", [mods[0]])
    A.write_scaled(src, tmp_path / "b", {mods[0]: 0.0})
    a, b = A.load_weights(tmp_path / "a"), A.load_weights(tmp_path / "b")
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_damping_prefers_the_gentlest_level_that_works():
    """A module that misbehaves is rarely ONLY wrong. The smallest change that
    clears a failure keeps the most of what it learned, and is least likely to
    be fitting noise in a handful of sampled generations."""
    mlx_nn = pytest.importorskip("mlx.nn")
    mx = pytest.importorskip("mlx.core")
    from mlx_lm.tuner.lora import LoRALinear

    class Tiny(mlx_nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = LoRALinear.from_base(mlx_nn.Linear(8, 8), r=2)

    model = Tiny()
    mods = A.live_lora_modules(model)
    name = next(iter(mods))
    mods[name].lora_b = mx.ones_like(mods[name].lora_b)
    base = float(mx.sum(mods[name].lora_b))

    # Clears at 0.5 or below; 0.75 is not enough.
    factors, spent = A.minimal_damping(
        model, [name], lambda: float(mx.sum(mods[name].lora_b)) <= base * 0.5 + 1e-6)
    assert factors == {name: 0.5}, factors
    assert spent == 2, "should stop at the first level that works"
    assert float(mx.sum(mods[name].lora_b)) == base, "restored exactly"


def test_damping_reports_when_nothing_helps():
    """Not even switching it off: that says the adapter is not what misaligned,
    which is a different answer from 'needs more damping'."""
    mlx_nn = pytest.importorskip("mlx.nn")
    from mlx_lm.tuner.lora import LoRALinear

    class Tiny(mlx_nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = LoRALinear.from_base(mlx_nn.Linear(8, 8), r=2)

    factors, spent = A.minimal_damping(Tiny(), ["q_proj"], lambda: False)
    assert factors == {}
    assert spent == len(A.DAMPING_LEVELS)


def test_zero_is_still_on_the_ladder():
    """Some misalignments are not a matter of degree."""
    assert A.DAMPING_LEVELS[-1] == 0.0
    assert A.DAMPING_LEVELS == tuple(sorted(A.DAMPING_LEVELS, reverse=True))
