#!/usr/bin/env python3
"""The paged expert layer, against the untouched SwitchLinear kernels.

Everything here runs at toy sizes (a 6-expert bank, small calls), which is
the point: paging is bookkeeping around kernels that are NOT re-written
here, so a small layer either matches the full-bank math exactly or the
pager is wrong. The SSD half — that eviction returns bytes and a cold row
re-faults from the file — is the real-model test's job (the 35B run); this
file pins the bookkeeping, the math, the growth cap, and what mlx_lm sees.
"""
import pytest

pytest.importorskip("mlx", reason="paged-moe is an MLX-only path")

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock  # noqa: E402
from mlx_lm.models.switch_layers import SwitchGLU  # noqa: E402
from symbio.app.paged_moe import (  # noqa: E402
    PagedSwitchGLU, _GlobalWarm, _layers_of, apply_paging, prime_resident)


def _block(hidden=64, moe=32, shared=32, experts=6, topk=2):
    args = SimpleNamespace(
        hidden_size=hidden, moe_intermediate_size=moe,
        shared_expert_intermediate_size=shared,
        norm_topk_prob=True, num_experts=experts, num_experts_per_tok=topk)
    return Qwen3NextSparseMoeBlock(args)


def _indices(shape, num_experts):
    return mx.random.randint(0, num_experts, shape)


def test_a_paged_layer_matches_the_untouched_switch_glu():
    src = SwitchGLU(128, 64, 6)
    paged = PagedSwitchGLU(src, hot=2, warm=_GlobalWarm(0))

    # decode-shaped call (unsorted path, batch 1)
    x = mx.expand_dims(mx.random.normal((1, 3, 128)), (-2, -3))
    idx = _indices((1, 3), 6)
    ref, got = src(x, idx), paged(x, idx)
    mx.eval(ref, got)
    assert mx.allclose(ref, got, atol=1e-5)

    # prefill-sized call arrives gather-sorted (SwitchGLU sorts >= 64), so
    # the test sorts the same way before calling; the raw unsorted big
    # shape is never a real call.
    x = mx.expand_dims(mx.random.normal((2, 40, 128)), (-2, -3))
    idx = _indices((2, 40), 6)
    from mlx_lm.models.switch_layers import _gather_sort
    x, idx, _inv = _gather_sort(x, idx)
    ref, got = src(x, idx), paged(x, idx)
    mx.eval(ref, got)
    assert mx.allclose(ref, got, atol=1e-5)


def test_the_layer_works_quantized_the_way_it_serves():
    """The real model is a 4-bit quantized checkpoint; the toy runs both
    paths so the composition is pinned in each."""
    from mlx_lm.models.switch_layers import SwitchLinear

    glu = SwitchGLU(64, 32, 6)
    for proj in (glu.gate_proj, glu.up_proj, glu.down_proj):
        setattr(glu, type(proj).__name__.replace("Linear", ""),
                None) if False else None
    # quantize the projections in place, the way mlx_lm.load does
    glu.gate_proj = glu.gate_proj.to_quantized(group_size=32, bits=4)
    glu.up_proj = glu.up_proj.to_quantized(group_size=32, bits=4)
    glu.down_proj = glu.down_proj.to_quantized(group_size=32, bits=4)

    paged = PagedSwitchGLU(glu, hot=2, warm=_GlobalWarm(0))
    x = mx.expand_dims(mx.random.normal((1, 8, 64)), (-2, -3))
    idx = _indices((1, 8), 6)
    ref, got = glu(x, idx), paged(x, idx)
    mx.eval(ref, got)
    assert mx.allclose(ref, got, atol=1e-5)


def test_eviction_bounds_the_warm_set_and_a_cold_expert_comes_back_correct():
    src = SwitchGLU(64, 32, 6)
    paged = PagedSwitchGLU(src, hot=2, warm=_GlobalWarm(0))
    x = mx.expand_dims(mx.random.normal((1, 1, 64)), (-2, -3))

    for expert in (0, 1, 2, 1):
        idx = mx.full((1, 1), expert, mx.int32)
        got = paged(x, idx)
        ref = src(x, idx)
        mx.eval(ref, got)
        assert mx.allclose(ref, got, atol=1e-5)

    assert len(paged._pages) <= 2


def test_the_block_level_math_is_unchanged_through_the_pager():
    block = _block()
    block.eval()
    x = mx.random.normal((2, 7, 64))
    ref = block(x)

    block.switch_mlp = PagedSwitchGLU(block.switch_mlp, hot=2,
                                      warm=_GlobalWarm(0))
    got = block(x)

    mx.eval(ref, got)
    assert mx.allclose(ref, got, atol=1e-4)


def test_the_block_survives_a_prefill_bigger_than_the_warm_set():
    """A call bigger than 64 tokens takes the sort/unsort path and routes to
    MORE experts than the warm set holds; the warm set still lands bounded
    after the call."""
    block = _block()
    block.eval()
    x = mx.random.normal((1, 500, 64))
    ref = block(x)

    block.switch_mlp = PagedSwitchGLU(block.switch_mlp, hot=2,
                                      warm=_GlobalWarm(0))
    got = block(x)

    mx.eval(ref, got)
    assert mx.allclose(ref, got, atol=2e-4)
    assert len(block.switch_mlp._pages) <= 2


def test_apply_paging_wraps_only_the_expert_layers():
    model = SimpleNamespace(model=SimpleNamespace(layers=[
        SimpleNamespace(mlp=_block()),
        SimpleNamespace(mlp=_block())]))

    wrapped = apply_paging(model, {"model": {
        "paged_experts_per_layer": 2, "paged_warm_gb": 1,
        "paged_experts_max_per_layer": 4}})
    for layer in wrapped.model.layers:
        assert isinstance(layer.mlp.switch_mlp, PagedSwitchGLU)
        assert layer.mlp.switch_mlp._cap == 4

    unchanged = apply_paging(model, {"model": {"paged_experts_per_layer": 0}})
    assert unchanged is model


def test_the_layers_are_found_under_every_known_wrapper():
    """Qwen3.5's multimodal Model nests the text stack as
    language_model.model.layers; plain ones use model.layers. The walker
    must not come back empty and silently page nothing."""
    def _blocks():
        return [SimpleNamespace(mlp=SimpleNamespace(switch_mlp=object())),
                SimpleNamespace(mlp=SimpleNamespace())]

    nested = SimpleNamespace(language_model=SimpleNamespace(
        model=SimpleNamespace(layers=_blocks())))
    assert len(_layers_of(nested)) == 2

    plain = SimpleNamespace(model=SimpleNamespace(layers=_blocks()))
    assert len(_layers_of(plain)) == 2

    bare = SimpleNamespace(layers=_blocks())
    assert len(_layers_of(bare)) == 2


def test_priming_evals_everything_but_the_cold_sources():
    """prime_resident is what keeps a lazy load lazy in the right places.
    The always-resident part evaluates into allocator memory; a paged bank's
    source subtree stays unevaluated. The sizes make it decisive: the cold
    source bank (~100 MB) dwarfs the resident dense layer (~4 KB), so an
    accidental materialization of the sources shows up as a 100x blow-up in
    active memory."""
    class Fake(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = nn.Linear(32, 32)      # the always-resident part
            self.mlp = SimpleNamespace(
                switch_mlp=PagedSwitchGLU(SwitchGLU(2048, 1024, 4), hot=1,
                                          warm=_GlobalWarm(0)))

    model = Fake()
    before = mx.get_active_memory()
    prime_resident(model)
    grown = mx.get_active_memory() - before

    dense_bytes = 32 * 32 * 4 + 32 * 4           # weights + bias, fp32
    assert grown <= dense_bytes * 2, (
        f"priming materialized {grown} bytes — the cold source pages "
        f"came home to RAM (they are ~100 MB of lazy bank)")


def test_the_parameters_the_model_reports_stay_the_sources():
    """The warm set is runtime cache: what mlx_lm saves, quantizes and
    trains against must keep the source's own leaves — same count, same
    multiset of shapes and dtypes — and nothing from the pager's pages may
    join them. (Leaf PATHS gain a `.source.` level, because the source
    module now sits under the pager; the tensors are the same.)"""
    block = _block()
    before = [(k, v.shape, str(v.dtype))
              for k, v in tree_flatten(block.parameters())]

    block.switch_mlp = PagedSwitchGLU(block.switch_mlp, hot=2,
                                      warm=_GlobalWarm(0))
    mx.eval(block(mx.random.normal((1, 3, 64))))  # warm set filled
    after = [(k, v.shape, str(v.dtype))
             for k, v in tree_flatten(block.parameters())]

    import collections

    def shapes_multiset(flat):
        return sorted(collections.Counter(
            (str(dt), tuple(sh)) for _, sh, dt in flat).items())

    assert shapes_multiset(after) == shapes_multiset(before)
    assert not any("slot" in str(k) or "bank" in str(k) or "page" in str(k)
                   for k, _, _ in after)


def _cached_snapshot(name):
    from pathlib import Path
    hub = Path(__import__("os").environ.get("HF_HOME",
                                            Path.home() / ".cache")) / "hub"
    d = hub / ("models--" + name.replace("/", "--"))
    snaps = d / "snapshots"
    if not snaps.is_dir():
        return None
    for snap in snaps.iterdir():
        if snap.is_dir():
            return snap
    return None


def test_row_reads_equal_the_lazy_load_byte_for_byte():
    """THE streaming primitive's contract: what the header+seek reader
    returns for expert e must be IDENTICAL to what MLX's own load of the
    same tensor holds at [e]. Runs against whatever LFM checkpoint is
    cached (skips if none is)."""
    from pathlib import Path
    import mlx.core as mx
    from symbio.app.paged_moe import _parse_shard_header, _RowReader

    snap = _cached_snapshot("LiquidAI/LFM2.5-8B-A1B-MLX-4bit")
    if snap is None:
        pytest.skip("no cached LFM checkpoint to test against")

    shard = sorted(snap.glob("*.safetensors"))[0]
    header = _parse_shard_header(shard)
    expertish = [k for k in header if "switch_mlp" in k and k.endswith("weight")]
    if not expertish:
        pytest.skip("the cached shard carries no expert-weight tensor")

    name = expertish[0]
    rows = int(header[name][3][0])   # shape[0] is field 3 now (path first)
    reader = _RowReader(header)
    lazy = mx.load(sh.__str__(), format="safetensors")[name] \
        if (sh := shard) is not None else None
    for e in (0, rows // 2, rows - 1):
        got = reader.read_row(name, e)
        ref = lazy[e]
        mx.eval(ref, got)
        assert mx.array_equal(ref, got), (
            f"row {e} of {name} differs: the byte-range reader is not "
            f"reading the same bytes the loader does")
        assert tuple(got.shape) == tuple(ref.shape)


def test_the_merged_gate_up_row_reads_split_by_bytes():
    """Qwen3.6 checkpoints ship gate|up merged (experts.gate_up_proj, the
    row is (2*out, packed)); the reader splits the byte-range rather than
    materializing the whole runtime tensor. Synthetic shard, exact check."""
    import numpy as np
    from pathlib import Path
    from types import SimpleNamespace
    from mlx_lm.models.switch_layers import SwitchGLU
    from symbio.app.paged_moe import build_row_reader

    E, OUT, PACKED = 4, 32, 2
    merged_w = np.random.randint(0, 2**31, size=(E, 2 * OUT, PACKED),
                                 dtype=np.uint32)
    merged_s = (np.random.rand(E, 2 * OUT, PACKED)
                .astype(np.float16) + 0.5)
    down_w = np.random.randint(0, 2**31, size=(E, 64, PACKED),
                               dtype=np.uint32)
    down_s = (np.random.rand(E, 64, PACKED).astype(np.float16) + 0.5)
    tmp = Path(__import__("tempfile").mkdtemp())
    shard = tmp / "model.safetensors"
    from safetensors.numpy import save_file as _st_save
    _st_save({"model.layers.3.mlp.experts.gate_up_proj.weight": merged_w,
              "model.layers.3.mlp.experts.gate_up_proj.scales": merged_s,
              "model.layers.3.mlp.experts.down_proj.weight": down_w,
              "model.layers.3.mlp.experts.down_proj.scales": down_s},
             str(shard))

    glu = SwitchGLU(64, OUT, E)
    glu.gate_proj = glu.gate_proj.to_quantized(group_size=32, bits=4)
    glu.up_proj = glu.up_proj.to_quantized(group_size=32, bits=4)
    glu.down_proj = glu.down_proj.to_quantized(group_size=32, bits=4)
    # The layer sits at list index 3 and the FILE names it model.layers.3:
    # the enumerate index is the absolute layer index, which is the file's
    # numbering. (Dense layers 0-2 are padding; the walker skips them.)
    layers = [SimpleNamespace(), SimpleNamespace(), SimpleNamespace(),
              SimpleNamespace(mlp=SimpleNamespace(switch_mlp=glu))]
    reader = build_row_reader(str(tmp), layers, "mlp")
    assert reader is not None, "the merged layout failed to map"

    paged = PagedSwitchGLU(glu, hot=1, warm=_GlobalWarm(0),
                           reader=reader,
                           row_names=reader.layer_names[3])
    assert paged._merged_gate_up
    parts = paged._materialize(3)
    gate_w, up_w = (parts[0][0], parts[1][0])
    mx.eval(gate_w, up_w)
    ref = mx.load(str(shard), format="safetensors")[
        "model.layers.3.mlp.experts.gate_up_proj.weight"]
    assert mx.array_equal(gate_w, ref[3][:OUT])
    assert mx.array_equal(up_w, ref[3][OUT:])
    # and the scales split the same way
    assert mx.array_equal(parts[0][1],
                          mx.load(str(shard), format="safetensors")[
       "model.layers.3.mlp.experts.gate_up_proj.scales"][3][:OUT])


def test_growth_buys_back_ssd_reads_only_up_to_the_global_budget():
    """A layer whose working set keeps missing its warm set is re-reading
    the SSD; the pager buys that back in RAM by growing — one expert per
    growth, gated by the GLOBAL warm-byte ledger, never past it, and each
    growth is logged. Without a budget there is no growth at all: a fixed
    warm set is a decision, not an accident."""
    from symbio.app import paged_moe as pm

    src = SwitchGLU(64, 32, 6)
    x = mx.expand_dims(mx.random.normal((1, 1, 64)), (-2, -3))

    growing = PagedSwitchGLU(src, hot=1, warm=_GlobalWarm(0))
    growing_row_bytes = growing._row_bytes
    budget = int(2.5 * growing_row_bytes)
    growing = PagedSwitchGLU(src, hot=1, warm=_GlobalWarm(budget))
    for turn in range(pm.THRASH_EXPERT_RELOADS + 8):
        growing(x, mx.full((1, 1), turn % 4, mx.int32))
    assert growing.hot == 2, "growth stopped at the global ledger's line"
    assert len(growing._pages) <= growing.hot
    assert growing._registry.bytes <= growing._registry.budget

    fixed = PagedSwitchGLU(src, hot=1, warm=_GlobalWarm(0))
    for turn in range(pm.THRASH_EXPERT_RELOADS + 8):
        fixed(x, mx.full((1, 1), turn % 4, mx.int32))
    assert fixed.hot == 1

    # The ledger tracks what the warm set holds.
    assert growing._registry.bytes >= 0
    assert growing._registry.bytes <= growing._registry.budget