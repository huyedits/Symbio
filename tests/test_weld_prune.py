"""scripts/weld_prune.py: merging adapters into one file, pruning the rank
slices that carry nothing, and welding the survivor into the base through
symbio.app.weights' own bind flow."""

import json
import re
from pathlib import Path

import pytest
pytest.importorskip("mlx")  # MLX is Apple Silicon only; skip elsewhere
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.tuner.lora import LoRALinear

import weld_prune

MODE = "model.layers.0.self_attn.q_proj"
B = "org/base-model"


def _adapter(tmp_path, name, pairs, scale=20.0, model=B):
    """One adapter dir: {module: (A (in, r), B (r, out))} files + config."""
    d = tmp_path / name
    d.mkdir()
    tensors = {}
    for n, (a, b) in pairs.items():
        tensors[n + ".lora_a"] = mx.array(a)
        tensors[n + ".lora_b"] = mx.array(b)
    mx.save_safetensors(str(d / "adapters.safetensors"), tensors)
    rank = max(len(a[0]) for a, _ in pairs.values())
    cfg = {
        "fine_tune_type": "lora", "model": model, "num_layers": 48,
        "lora_parameters": {"scale": scale, "rank": rank,
                            "dropout": 0.0,
                            "keys": [re.sub(r"^model\.layers\.\d+\.", "", n)
                                     for n in pairs]},
    }
    (d / "adapter_config.json").write_text(json.dumps(cfg))
    return d


# ------------------------------------------------------------- merge

def test_merge_sums_the_deltas_exactly(tmp_path):
    """The concat in adapter space is the sum in ΔW space: that is the weld."""
    a1, b1 = [[1.0, 0.0], [0.0, 2.0], [0.5, 0.0]], [[1.0, 0.0], [0.0, 2.0]]
    a2, b2 = [[0.0, 1.0], [3.0, 0.0], [2.0, 0.0]], [[0.0, 1.0], [1.5, 0.0]]
    one = _adapter(tmp_path, "one", {MODE: (a1, b1)}, scale=10.0)
    two = _adapter(tmp_path, "two", {MODE: (a2, b2)}, scale=5.0)

    weld_prune.merge([one, two], tmp_path / "out")

    out = weld_prune.module_pairs(tmp_path / "out")
    (a, b), = out.values()
    assert a.shape == (3, 4) and b.shape == (4, 2)          # rank 2 + 2, padded none
    want = weld_prune.delta((mx.array(a1), mx.array(b1)), 10.0) \
        + weld_prune.delta((mx.array(a2), mx.array(b2)), 5.0)
    cfg = json.loads((tmp_path / "out" / "adapter_config.json").read_text())
    assert cfg["lora_parameters"]["scale"] == 10.0          # the first source's scale
    assert cfg["lora_parameters"]["rank"] == 4
    assert mx.allclose(weld_prune.delta(out[MODE],
                                        float(cfg["lora_parameters"]["scale"])), want)
    assert cfg["lora_parameters"]["keys"] == ["self_attn.q_proj"]


def test_merge_pads_modules_one_adapter_never_touched(tmp_path):
    """A module only the second adapter trained for keeps its ΔW, with the
    rank the first never contributed filled by zero — and the rank is one
    number for the whole file, so mlx_lm's single-rank loader is honest."""
    shared = MODE
    only2 = "model.layers.1.self_attn.q_proj"
    one = _adapter(tmp_path, "one", {shared: ([[1.0], [0.0]], [[2.0]])}, scale=20.0)
    two = _adapter(tmp_path, "two", {
        shared: ([[3.0], [0.0]], [[4.0]]),
        only2: ([[3.0], [0.0]], [[4.0]]),
    }, scale=10.0)

    weld_prune.merge([one, two], tmp_path / "out")

    out = weld_prune.module_pairs(tmp_path / "out")
    assert sorted(out) == [shared, only2]
    (a, b) = out[shared]
    (a2, b2) = out[only2]
    assert a.shape == (2, 2) and b.shape == (2, 1)          # shared: rank 1 + rank 1
    assert a2.shape == (2, 2) and b2.shape == (2, 1)        # only2: zero pad + rank 1
    assert mx.allclose(weld_prune.delta(out[shared], 20.0),
                       weld_prune.delta((mx.array([[1.0], [0.0]]), mx.array([[2.0]])), 20.0)
                       + weld_prune.delta((mx.array([[3.0], [0.0]]), mx.array([[4.0]])), 10.0))
    assert mx.allclose(weld_prune.delta(out[only2], 20.0),        # the merged file's
                       weld_prune.delta((mx.array([[3.0], [0.0]]), mx.array([[4.0]])), 10.0))
    # only2's pad (adapter one never trained it): the LAST rank slice is zero.
    assert float(mx.max(a2[:, 1] * a2[:, 1]).item()) == 0.0
    cfg = json.loads((tmp_path / "out" / "adapter_config.json").read_text())
    assert sorted(cfg["lora_parameters"]["keys"]) == \
        ["self_attn.q_proj"]                                # suffix style, de-layered


def test_the_merged_file_loads_through_mlx_lm(tmp_path):
    """The whole point of pinning one rank: mlx_lm builds its LoRA layers at
    the config's rank from every key at once, and a file whose modules
    disagree with it loads silently empty. The merged file must apply."""
    one = _adapter(tmp_path, "one", {MODE: ([[1.0], [0.0]], [[2.0]])}, scale=20.0)
    two = _adapter(tmp_path, "two", {MODE: ([[0.0], [3.0]], [[4.0]])}, scale=5.0)
    weld_prune.merge([one, two], tmp_path / "out")

    lin = LoRALinear.from_base(nn.Linear(2, 1, bias=False), r=2, scale=20.0)
    model = nn.Module()
    model.model = nn.Module()
    block = nn.Module()
    block.self_attn = nn.Module()
    block.self_attn.q_proj = lin
    model.model.layers = [block]          # a real model's layers: a plain list
    model.load_weights(str(tmp_path / "out" / "adapters.safetensors"), strict=False)
    got = dict(tree_flatten(model.parameters()))
    file = mx.load(str(tmp_path / "out" / "adapters.safetensors"))
    assert got[MODE + ".lora_a"].shape == (2, 2)
    assert mx.allclose(got[MODE + ".lora_a"], file[MODE + ".lora_a"])
    assert mx.allclose(got[MODE + ".lora_b"], file[MODE + ".lora_b"])


def test_merge_refuses_adapters_of_different_bases(tmp_path):
    """One file loads against one model. Root and a worker built on a
    different size would load 'fine' and change nothing — refuse instead."""
    one = _adapter(tmp_path, "one", {MODE: ([[1.0]], [[2.0]])}, model=B)
    two = _adapter(tmp_path, "two", {MODE: ([[3.0]], [[4.0]])}, model="other/model")

    with pytest.raises(ValueError, match="different bases"):
        weld_prune.merge([one, two], tmp_path / "out")


# ------------------------------------------------------------- prune

def test_prune_drops_the_slice_that_carries_nothing(tmp_path):
    """Energies [10, 0.01] with --keep 0.999: the second slice is noise, its
    outer product vanishes from the file, the first is untouched."""
    a = [[5.0, 0.5], [7.0, 0.5], [1.0, 0.5]]      # column norms: 10 and ~0.87...
    a = [[10.0, 0.0], [0.0, 0.0], [0.0, 1.0]]     # 10 and 1 — with B below: 0.01
    b = [[1.0, 0.0, 0.0], [0.0, 0.0, 0.01]]
    d = _adapter(tmp_path, "one", {MODE: (a, b)}, scale=1.0)

    stats = weld_prune.prune(d, tmp_path / "out", keep=0.999, min_rel=0.0)

    assert stats["dropped"] == []
    row = stats["modules"][MODE]
    assert row["rank"] == 1 and row["slices"] == 2
    (a2, b2), = weld_prune.module_pairs(tmp_path / "out").values()
    assert a2.shape == (3, 1) and b2.shape == (1, 3)
    assert mx.allclose(weld_prune.delta((a2, b2), 1.0),
                       weld_prune.delta((mx.array(a), mx.array(b)), 1.0)
                       - weld_prune.delta((mx.array([[0.0], [0.0], [1.0]]),
                                           mx.array([[0.0, 0.0, 0.01]])), 1.0))


def test_prune_writes_one_uniform_rank_and_pads_the_rest_with_zero(tmp_path):
    """Modules keep 2 and 1 slices: the file carries rank 2, the short one
    padded with a zero column/row so it still loads against the config."""
    kept2 = [[1.0, 0.4], [0.0, 2.0], [0.3, 0.0]]
    kept1 = [[2.0, 0.1], [0.0, 0.5]]
    b2 = [[1.0, 1.0, 0.0], [0.5, 1.0, 2.0]]
    b1 = [[3.0, 2.0], [0.0, 0.2]]
    d = _adapter(tmp_path, "one", {
        "model.layers.0.self_attn.q_proj": (kept2, b2),
        "model.layers.1.self_attn.q_proj": (kept1, b1),
    }, scale=1.0)

    stats = weld_prune.prune(d, tmp_path / "out", keep=0.999, min_rel=0.0)
    assert stats["rank_out"] == 2

    out = weld_prune.module_pairs(tmp_path / "out")
    assert out["model.layers.0.self_attn.q_proj"][0].shape == (3, 2)
    assert out["model.layers.1.self_attn.q_proj"][0].shape == (2, 2)
    a1, b1p = out["model.layers.1.self_attn.q_proj"]
    # The one dropped slice (energy ~0.1 next to ~7.2) is gone: col 1 empty.
    assert float(mx.max(a1[:, 1] * a1[:, 1]).item()) == 0.0
    assert mx.allclose(weld_prune.delta(out["model.layers.1.self_attn.q_proj"], 1.0),
                       weld_prune.delta((mx.array(kept1), mx.array(b1)), 1.0)
                       - weld_prune.delta((mx.array([[0.1], [0.5]]),
                                           mx.array([[0.0, 0.2]])), 1.0))


def test_prune_drops_a_module_lora_b_never_trained(tmp_path):
    """A fresh pair is rank whose B is all zeros: no ΔW at all. It leaves.""
    """
    d = _adapter(tmp_path, "one", {
        MODE: ([[1.0], [0.0]], [[2.0, 0.0, 0.0]]),
        "model.layers.1.self_attn.q_proj": ([[0.0], [0.0]], [[0.0, 0.0, 0.0]]),
    }, scale=20.0)

    stats = weld_prune.prune(d, tmp_path / "out", keep=0.999, min_rel=0.0)

    assert stats["dropped"] == ["model.layers.1.self_attn.q_proj"]
    out = weld_prune.module_pairs(tmp_path / "out")
    assert sorted(out) == [MODE]
    cfg = json.loads((tmp_path / "out" / "adapter_config.json").read_text())
    assert cfg["lora_parameters"]["rank"] == 1


def test_prune_with_full_keep_is_faithful(tmp_path):
    """--keep 1.0 keeps every trained slice: the ΔW must be exactly the old one."""
    a, b = [[1.0, -2.0], [0.5, 0.0]], [[2.0, 0.0, 1.0], [0.0, 1.0, 3.0]]
    d = _adapter(tmp_path, "one", {MODE: (a, b)}, scale=7.0)

    weld_prune.prune(d, tmp_path / "out", keep=1.0, min_rel=0.0)

    before = weld_prune.module_pairs(d)[MODE]
    after = weld_prune.module_pairs(tmp_path / "out")[MODE]
    assert mx.allclose(weld_prune.delta(after, 7.0), weld_prune.delta(before, 7.0))


def test_prune_refuses_an_adapter_that_is_all_fresh_pairs(tmp_path):
    """Everything is lora_b-zero: nothing carries anything, nothing to prune."""
    d = _adapter(tmp_path, "one", {MODE: ([[1.0]], [[0.0]])}, scale=20.0)

    with pytest.raises(ValueError, match="nothing to prune"):
        weld_prune.prune(d, tmp_path / "out", keep=0.999, min_rel=0.0)


def test_min_rel_cuts_what_coverage_would_keep(tmp_path):
    """--keep 1.0 keeps everything trained — unless --min-rel floors the
    weak slices: an early slice of a merged adapter, under the strongest."""
    a, b = [[9.0, 0.3], [0.0, 4.0]], [[2.0, 0.0], [1.0, 1.0]]   # energies [18, ~0.43]
    d = _adapter(tmp_path, "one", {MODE: (a, b)}, scale=1.0)

    full = weld_prune.prune(d, tmp_path / "full", keep=1.0, min_rel=0.0)
    cut = weld_prune.prune(d, tmp_path / "cut", keep=1.0, min_rel=0.5)

    assert full["rank_out"] == 2
    assert cut["rank_out"] == 1
    out = weld_prune.module_pairs(tmp_path / "cut")[MODE]
    assert mx.allclose(weld_prune.delta(out, 1.0),
                       weld_prune.delta((mx.array(a), mx.array(b)), 1.0)
                       - weld_prune.delta((mx.array([[0.3], [4.0]]),
                                           mx.array([[1.0, 1.0]])), 1.0))


# ------------------------------------------------------------- weld

@pytest.fixture()
def appflow(monkeypatch):
    """The app's own stage/commit/abandon, recorded instead of run, so the
    tests here never hold a real model or touch the real ledger."""
    import symbio.app.training as training
    import symbio.app.weights as weights

    calls = {"stage": [], "commit": [], "abandon": [], "removed": 0}

    def stage(cand):
        n = 1 + len(calls["stage"])
        calls["stage"].append(cand)
        plan = {"base": cand["base"], "kind": "bind", "id": f"w{n}",
                "bound": [{"id": f"w{n}", "archive": f"/arch/{n}"}],
                "path": f"/ver/{n}", "prev_path": None, "version": n}
        return True, "folded 1 delta(s) into 1 module(s)", plan

    def commit(plan, before, after):
        calls["commit"].append((plan, sorted(before), sorted(after)))

    monkeypatch.setattr(weights, "stage", stage)
    monkeypatch.setattr(weights, "commit", commit)
    monkeypatch.setattr(weights, "abandon", lambda p: calls["abandon"].append(p))
    monkeypatch.setattr(training, "remove_adapter", lambda: calls.update(
        removed=calls["removed"] + 1))
    return calls


def test_weld_builds_and_commits_a_bind_of_the_pruned_adapter(tmp_path, appflow):
    """The flow the in-chat bind runs, driving a pruned copy: stage gets a
    bind of the survivor, commit keeps it, and a stash dir's files stay."""
    a, b = [[2.0, 0.0], [1.0, 0.02]], [[1.0, 1.0, 0.0], [0.0, 2.0, 1.0]]
    d = _adapter(tmp_path, "one", {MODE: (a, b)}, scale=1.0)

    code = weld_prune.weld(d, base=B, keep=0.999, min_rel=0.0,
                           no_prune=False, yes=True)

    assert code == 0
    (cand,) = appflow["stage"]
    assert cand["kind"] == "bind" and cand["base"] == B
    assert Path(cand["adapter"]) != d                     # a prepared copy was welded
    out = weld_prune.module_pairs(cand["adapter"])
    assert out[MODE][0].shape[1] == 1                     # rank cut to what survived
    assert len(appflow["commit"]) == 1 and not appflow["abandon"]
    assert appflow["removed"] == 0                        # not the live dir


def test_weld_dry_run_builds_the_version_and_takes_it_back(tmp_path, appflow):
    d = _adapter(tmp_path, "one", {MODE: ([[2.0], [1.0]], [[1.0, 1.0, 0.0]])}, scale=1.0)

    code = weld_prune.weld(d, base=B, keep=0.999, min_rel=0.0,
                           no_prune=True, yes=False)

    assert code == 0
    assert appflow["stage"] and appflow["abandon"] and not appflow["commit"]
    assert appflow["removed"] == 0


def test_weld_without_pruning_welds_the_adapter_itself(tmp_path, appflow):
    d = _adapter(tmp_path, "one", {MODE: ([[2.0], [1.0]], [[1.0, 1.0, 0.0]])}, scale=1.0)

    weld_prune.weld(d, base=B, keep=0.999, min_rel=0.0, no_prune=True, yes=True)

    (cand,) = appflow["stage"]
    assert Path(cand["adapter"]) == d


def test_weld_removes_the_live_only_when_it_is_the_live(tmp_path, appflow, monkeypatch):
    import symbio.constants as constants

    d = _adapter(tmp_path, "one", {MODE: ([[2.0], [1.0]], [[1.0, 1.0, 0.0]])}, scale=1.0)

    weld_prune.weld(d, base=B, keep=0.999, min_rel=0.0, no_prune=True, yes=True)
    assert appflow["removed"] == 0                        # a stash dir: files stay

    monkeypatch.setattr(constants, "ADAPTER_DIR", d)
    weld_prune.weld(d, base=B, keep=0.999, min_rel=0.0, no_prune=True, yes=True)
    assert appflow["removed"] == 1                        # the live dir: cleared


def test_weld_refuses_an_adapter_from_another_base(tmp_path, appflow, monkeypatch):
    import symbio.app.weights as weights
    monkeypatch.setattr(weights, "resolve", lambda base: base)

    d = _adapter(tmp_path, "one", {MODE: ([[2.0], [1.0]], [[1.0, 1.0, 0.0]])},
                 scale=1.0, model="other/model")

    with pytest.raises(ValueError, match="other/model"):
        weld_prune.weld(d, base=B, keep=0.999, min_rel=0.0, no_prune=True, yes=True)
    assert not appflow["stage"]