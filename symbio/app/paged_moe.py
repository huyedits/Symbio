"""Expert paging for MoE models: the cold expert rows stay on the SSD.

The 16 GB machine cannot hold a 20+ GB 4-bit MoE resident, and the plain mmap
path is worse than either: stock mlx_lm.load evals EVERY parameter when it
finishes (measured 2026-10-15: Qwen3.5-35B-A3B-4bit returned from load with
18.16 GB already materialized). So paging owns the load path:

* the load runs LAZY (mlx_lm.load's `lazy=True`), leaving every weight as a
  file-backed mmap;
* the pager wraps each MoE layer's switch_mlp BEFORE anything evaluates, and
  primes ONLY the always-resident part (attention, DeltaNet, norms, router,
  shared expert, embeddings, head — 1.28 GB on that 35B, measured).

How the warm set works — the unit is the EXPERT PER LAYER, not the
projection: a MoE layer routes tokens through gate/up/down with the SAME
indices, so its three projections page together, under one expert key. Warm
rows are owned copies (`+ 0` forces a fresh buffer on eval), dropped on
eviction so their bytes return to the allocator while the mmap source keeps
its place on disk.

SSD protection (the SSD is not to get hammered):

* NOTHING here reads on a timer — the only reads are router-driven misses,
  and macOS's file cache dedupes repeat touches after that;
* a reload counter per layer (new experts streamed in a sliding 60 s window)
  watches for THRASH: a layer whose working set keeps missing its warm set.
  Over the threshold the layer's warm set GROWS a step — but growth is gated
  by a GLOBAL warm-byte budget (model.paged_warm_gb): the RAM it costs is
  bounded for the WHOLE model, not per bank. Each growth is logged with the
  measured counts; at the budget line the layer grinds honestly (the log
  says the numbers) rather than the SSD eating silently.
* with no warm budget set, there is no growth at all: a fixed warm set is a
  decision, not an accident.

Measured on Qwen3.5-35B-A3B (256 experts, 8-of-256 routing, 40 layers): a
short conversation keeps ~50 distinct experts warm per layer per minute —
scattered, NOT the ~15%-concentrated routing mlx-moe measured on
Qwen3-Coder-Next. ~50 warm experts/layer is ~3.4 GB of rows; that fit is
exactly what paged_warm_gb decides. (The per-projection warm set that
preceded this grew 120 banks x 48 rows = 9.2 GB and OOM'd Metal
mid-generation — the same routing scatter tripled.)

The pages are deliberately NOT nn.Module-tracked state: mlx_lm walks
Module.parameters() to save adapters and quantize checkpoints, and warm
rows are runtime cache, not model state. The parameters the model reports
and saves are exactly the source's own, unchanged.
"""
from __future__ import annotations

from collections import OrderedDict, deque
from threading import Lock
from typing import Any
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

# The thrash window and threshold, per LAYER, counted in experts streamed.
# 40 new experts inside a minute means the warm set keeps missing.
THRASH_WINDOW_S = 60.0
THRASH_EXPERT_RELOADS = 40


def _parse_shard_header(path):
    """One safetensors shard's tensor table: {name: (data_start, length,
    shape, np_dtype)}. The header is an 8-byte little-endian JSON length,
    then the JSON, then the data at data_start. data_offsets are relative
    to the data block."""
    import json as _json
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = _json.loads(f.read(n))
    data_start = 8 + n
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        out[name] = (str(path), data_start + start, end - start,
                     tuple(meta["shape"]), meta["dtype"])
    return out


_NP_DTYPES = {
    "U32": "uint32", "I32": "int32", "I64": "int64",
    "F16": "float16", "F32": "float32", "F64": "float64",
    "U8": "uint8", "I8": "int8", "BOOL": "bool",
}

# numpy has no bfloat16; BF16 tensors read as uint16 bits and MLX's
# numpy-style .view reinterprets them (a CAST would treat the bits as an
# integer value — measured 2026-10-15: 2.0's bits became 16384.0).
_MX_BIT_VIEWS = {"BF16": mx.bfloat16}


class _RowReader:
    """Byte-level row reads from the weight files: THE streaming primitive.

    MLX's lazy path materializes a WHOLE tensor on first touch (measured
    2026-10-15: the first slice of a bank pulled ~0.3 GB and every later
    slice was free), so a lazy slice cannot read one expert. The safetensors
    header can. Rows along axis 0 are contiguous C-order blocks, so expert
    e's row of any tensor is one seek + one read of prod(shape[1:]) x
    itemsize bytes — and the packed-weights parts a 4-bit checkpoint serves
    are U32/F16 on both sides, whose numpy dtypes are exact. The reader
    keeps one readonly mmap per shard open for the process lifetime; a read
    is seek + read + np.frombuffer + mx.array — sub-millisecond per row.
    """

    def __init__(self, entries: dict):
        # name -> (path, data_start, row_len, shape, np_dtype_name)
        self.entries = entries
        self.layer_names: dict = {}   # layer idx -> {proj: {part: name}}
        self._maps = {}
        self.total_bytes_read = 0
        self.read_bytes_clock: deque = deque()   # (time, bytes)

    def read_row(self, name: str, e: int) -> "mx.array":
        """One axis-0 row of a checkpoint tensor, by bytes.

        Safetensors stores C-order tensor data contiguously, so row e is
        the bytes [row_bytes * e, row_bytes * (e+1)) past the tensor's data
        start; the header carries the shape, and the row's own shape is the
        tensor's shape[1:]. The read is seek + read — nothing else of the
        tensor is touched."""
        path, data_start, total_len, shape, dt = self.entries[name]
        m = self._maps.get(path)
        if m is None:
            m = self._maps[path] = open(path, "rb")
        rows = int(shape[0])
        if rows <= 0 or total_len % rows:
            raise ValueError(
                f"{name}: {rows} rows do not divide {total_len} bytes; "
                f"the streaming reader needs an axis-0 row to be a byte "
                f"range")
        row_len = total_len // rows
        m.seek(data_start + row_len * e)
        buf = m.read(row_len)
        self.total_bytes_read += row_len
        self.read_bytes_clock.append((time.monotonic(), row_len))
        np = __import__("numpy")
        tail = [int(d) for d in shape[1:]]
        if dt in _MX_BIT_VIEWS:
            # No numpy bfloat16: the bits come through uint16 and MLX's
            # view() reinterprets them, shape intact.
            view = np.frombuffer(buf, dtype=np.dtype("uint16"))
            return mx.array(view.reshape(tail)).view(_MX_BIT_VIEWS[dt])
        view = np.frombuffer(buf, dtype=np.dtype(_NP_DTYPES[dt]))
        return mx.array(view.reshape(tail))


class _GlobalWarm:
    """The whole model's warm-byte ledger, shared by every paged layer.

    Growth asks it whether the budget has room; evictions hand it back. A
    lock because the daemon's generation thread is the ordinary writer and
    a worker can hold a second copy of the model.
    """

    __slots__ = ("budget", "bytes", "lock")

    def __init__(self, budget_bytes: int):
        self.budget = int(budget_bytes)
        self.bytes = 0
        self.lock = Lock()

    def can_take(self, item_bytes: int) -> bool:
        if self.budget <= 0:
            return False         # no budget: the warm set never grows
        with self.lock:
            return self.bytes + item_bytes <= self.budget

    def add(self, item_bytes: int) -> None:
        with self.lock:
            self.bytes += item_bytes

    def drop(self, item_bytes: int) -> None:
        with self.lock:
            self.bytes = max(0, self.bytes - item_bytes)


class PagedSwitchGLU(nn.Module):
    """Stands in for SparseMoeBlock.switch_mlp (SwitchGLU): the same call,
    the same kernels, with the layer's expert rows paged as one warm set.

    The composition is the source's own — sort (many tokens), up/gate, the
    SwiGLU activation between them, down, unsort — and each projection's
    kernel (gather_qmm / gather_mm) runs against a bank stacked from ONLY
    the rows this call routes to. Nothing about the math is re-implemented;
    which rows are resident is what is decided here.

    The source module is kept as `source`, so the parameters the model
    reports, saves and trains against are the source's own; `_pages` is
    runtime state. The layer's warm set is keyed by expert id and holds its
    three projections together.
    """

    def __init__(self, source: nn.Module, hot: int, warm: _GlobalWarm,
                 cap: int = 0, reader: "_RowReader | None" = None,
                 row_names: dict | None = None):
        super().__init__()
        self.source = source
        self._activation = source.activation
        self._registry = warm
        self.hot = max(1, int(hot))
        self._cap = max(int(cap), 0)
        self._reader = reader              # byte-level row reads, if mapped
        self._rows = row_names             # proj name -> {part: tensor name}
        # The merged gate|up layout (Qwen checkpoints ship
        # experts.gate_up_proj; mlx_lm's sanitize splits it in memory): the
        # row reader splits it by bytes instead.
        self._merged_gate_up = bool(row_names and str(
            row_names.get("gate_proj", {}).get("weight", "")).endswith(
            "gate_up_proj.weight"))
        self._quantized = hasattr(source.gate_proj, "scales")

        self._projs = []
        for name in ("gate_proj", "up_proj", "down_proj"):
            proj = getattr(source, name)
            self._projs.append({
                "name": name,
                "w": proj["weight"] if self._quantized else proj.weight,
                "s": proj["scales"] if self._quantized else None,
                "zp": proj.get("biases") if self._quantized else None,
                "add": proj["bias"] if "bias" in proj else None,
                "group_size": getattr(proj, "group_size", 0),
                "bits": getattr(proj, "bits", 0),
                "mode": getattr(proj, "mode", "affine"),
            })
        self._e = self._projs[0]["w"].shape[0]

        # Warm rows: expert id -> ((gw, sg, zpg, addg), (uw, ...), (dw, ...))
        # — one part per projection. NOT nn.Module-tracked: parameters()
        # must never see them (see module docstring).
        self._pages: OrderedDict[int, tuple] = OrderedDict()
        self._row_bytes = self._row_bytes()
        self._reload_clock: deque[float] = deque()

        # Inference machinery. Adapters are trained on the model's own
        # parameters, never through this cache; pointed at a paged layer,
        # training gets the error, not a silently cold bank mid-optimizer.
        # eval() is what makes that honest: a frozen module still defaults
        # to training=True in mlx.nn.
        self.freeze()
        self.eval()

    # ---- bookkeeping -------------------------------------------------------
    def _row_bytes(self) -> int:
        """One expert's whole triple of rows, in bytes."""
        total = 0
        for p in self._projs:
            per = 1
            for dim in p["w"].shape[1:]:
                per *= int(dim)
            per *= int(p["w"].dtype.size)
            if p["s"] is not None:
                sb = 1
                for dim in p["s"].shape[1:]:
                    sb *= int(dim)
                per += sb * int(p["s"].dtype.size)
                if p["zp"] is not None:
                    zb = 1
                    for dim in p["zp"].shape[1:]:
                        zb *= int(dim)
                    per += zb * int(p["zp"].dtype.size)
            if p["add"] is not None:
                ab = 1
                for dim in p["add"].shape[1:]:
                    ab *= int(dim)
                per += ab * int(p["add"].dtype.size)
            total += int(per)
        return max(1, total)

    def _materialize(self, e: int) -> tuple:
        """Expert e's parts across the three projections, as fresh owned
        copies.

        Two paths, and the choice IS the SSD story. With a row reader (the
        layer's parts mapped to (shard, byte-range) entries from the
        checkpoint's own headers) the parts are READ from the files — one
        seek+read each, sub-millisecond, and NOTHING of the lazy bank comes
        resident. Without one, the lazy-slice fallback: `+ 0` gives each a
        fresh buffer on eval — and MLX's lazy path materializes the WHOLE
        tensor on first touch (measured 2026-10-15: the first slice of a
        bank pulled ~0.3 GB, every later slice free), so the machine's own
        crude streaming is fast on the second read and resident forever."""
        if self._rows is not None:
            parts = []
            merged_row = None
            for p in self._projs:
                entry = self._rows[p["name"]]
                if self._merged_gate_up and p["name"] in ("gate_proj",
                                                          "up_proj"):
                    # The checkpoint stores gate|up merged along the OUT
                    # axis: (2*out, packed). One byte-read serves both
                    # halves — gate the first half, up the second.
                    if merged_row is None:
                        merged_row = self._reader.read_row(
                            entry["weight"], e)
                    out_half = p["w"].shape[1]
                    w = (merged_row[:out_half] if p["name"] == "gate_proj"
                         else merged_row[out_half:])
                    mscales = self._rows[p["name"]].get("scales")
                    s = None
                    if p["s"] is not None and mscales:
                        s_row = self._reader.read_row(mscales, e)
                        s = (s_row[:out_half] if p["name"] == "gate_proj"
                             else s_row[out_half:])
                    zp = None
                    if p["s"] is not None and entry.get("biases"):
                        z_row = self._reader.read_row(entry["biases"], e)
                        zp = (z_row[:out_half]
                              if p["name"] == "gate_proj"
                              else z_row[out_half:])
                    add = None
                else:
                    w = self._reader.read_row(entry["weight"], e)
                    s = (self._reader.read_row(entry["scales"], e)
                         if p["s"] is not None else None)
                    # A runtime zero-points tensor may exist while the file
                    # carries none (affine quantize's zero-points are all
                    # zeros and the kernel needs none); a missing file half
                    # reads as None, not as a crash.
                    zp = (self._reader.read_row(entry["biases"], e)
                          if p["zp"] is not None and "biases" in entry
                          else None)
                    add = (self._reader.read_row(entry["bias"], e)
                           if p["add"] is not None and "bias" in entry
                           else None)
                parts.append((w, s, zp, add))
            return tuple(parts)
        parts = []
        for p in self._projs:
            w = p["w"][e:e + 1] + 0
            s = p["s"][e:e + 1] + 0 if p["s"] is not None else None
            zp = p["zp"][e:e + 1] + 0 if p["zp"] is not None else None
            add = p["add"][e:e + 1] + 0 if p["add"] is not None else None
            parts.append((w, s, zp, add))
        mx.eval(*[arr for part in parts for arr in part if arr is not None])
        return tuple(parts)

    def _part(self, proj_state, part):
        """(w, s, zp) as the kernel call wants them — scales/zero-points
        exist only on a quantized source."""
        w, s, zp, _add = part
        return (w, s, zp)

    # ---- the source module's own composition -------------------------------
    def __call__(self, x: mx.array, indices: mx.array) -> mx.array:
        import os

        dbg = os.environ.get("PAGED_MOE_DEBUG")
        if dbg:
            print(f"[pageddbg] enter L{self._idx}: active "
                  f"{mx.get_active_memory()/2**30:.2f} · "
                  f"peak {mx.get_peak_memory()/2**30:.2f} GB · "
                  f"pages {len(self._pages)}", flush=True)
        if self.training:
            raise RuntimeError(
                "the paged expert layer is not a training path; train "
                "adapters against the model's own parameters with paging "
                "off (clear model.paged_experts_per_layer).")

        x = mx.expand_dims(x, (-2, -3))
        do_sort = indices.size >= 64
        idx = indices
        inv_order = None
        if do_sort:
            x, idx, inv_order = _gather_sort(x, indices)

        flat = idx.reshape(-1).tolist()
        ids = sorted({int(i) for i in flat})
        pages = self._pages

        if dbg:
            probe = self._pages
            first = ids[0]
            if first not in pages:
                one = self._materialize(first)
                mx.eval(*[a for part in one for a in part if a is not None])
                print(f"[pageddbg] after ONE materialize L{self._idx}: "
                      f"active {mx.get_active_memory()/2**30:.2f} GB", flush=True)

        rows: dict[int, tuple] = {}
        for e in ids:
            if e in pages:
                pages.move_to_end(e)
                rows[e] = pages[e]
                continue
            while len(pages) >= self.hot:
                evicted, _old = pages.popitem(last=False)
                self._registry.drop(self._row_bytes)
            pages[e] = self._materialize(e)
            self._registry.add(self._row_bytes)
            rows[e] = pages[e]

        # Thrash watch: this layer streamed NEW experts constantly = the
        # warm set is under-budgeted. Growth is a step of one, GATED by the
        # global warm-byte budget, and every growth is logged with the
        # measured counts.
        now = time.monotonic()
        clock = self._reload_clock
        clock.append(now)
        while clock and now - clock[0] > THRASH_WINDOW_S:
            clock.popleft()
        if (len(clock) >= THRASH_EXPERT_RELOADS and self.hot < self._e and (
                self._cap == 0 or self.hot < self._cap)):
            if self._registry.can_take(self._row_bytes):
                self.hot += 1
                print(f"  [paged-moe] a layer streamed {len(clock)} new "
                      f"experts in {THRASH_WINDOW_S:.0f}s — warm set grown "
                      f"to {self.hot} experts/layer (global warm budget "
                      f"{self._registry.budget / 2**30:.1f} GB, "
                      f"{self._registry.bytes / 2**30:.1f} GB in use); the "
                      f"working set wants RAM, not SSD.")

        # Token indices -> stack positions. One small lut assignment keeps
        # the remap inside the lazy graph. The kernel's sorted-path hint is
        # left OFF: a warm page's stack position implies nothing about id
        # order, and the kernel's own sort costs a fraction of a disk miss.
        lut = mx.full((self._e,), -1, mx.int32)
        lut[mx.array(ids, dtype=mx.int32)] = mx.arange(len(ids))
        local = lut[idx]

        parts_by_proj = [[rows[e][pi] for e in ids]
                         for pi in range(len(self._projs))]

        # The composition, projection by projection, in the source's order:
        x_up = self._run(self._projs[1], parts_by_proj[1], x, local)
        x_gate = self._run(self._projs[0], parts_by_proj[0], x, local)
        mid = self._activation(x_up, x_gate)
        out = self._run(self._projs[2], parts_by_proj[2], mid, local)

        if do_sort:
            out = _scatter_unsort(out, inv_order, indices.shape)
        return out.squeeze(-2)

    def _run(self, proj_state, parts, x: mx.array,
             local: mx.array) -> mx.array:
        """One projection's kernel against the stacked bank of parts. The
        kernels are the source's own: gather_qmm for a quantized source,
        gather_mm for a float one."""
        w_bank = mx.stack([part[0] for part in parts])
        name = proj_state["name"]
        if proj_state["s"] is not None:
            s_bank = mx.stack([part[1] for part in parts])
            zp_bank = mx.stack([part[2] for part in parts]) \
                if proj_state["zp"] is not None else None
            out = mx.gather_qmm(
                x, w_bank, s_bank, zp_bank,
                rhs_indices=local,
                transpose=True,
                group_size=proj_state["group_size"], bits=proj_state["bits"],
                mode=proj_state["mode"],
                sorted_indices=False)
            if proj_state["add"] is not None:
                a_bank = mx.stack([part[3] for part in parts])
                out = out + mx.expand_dims(a_bank[local], -2)
            return out
        out = mx.gather_mm(
            x, w_bank.swapaxes(-1, -2),
            rhs_indices=local,
            sorted_indices=False)
        if proj_state["add"] is not None:
            a_bank = mx.stack([part[3] for part in parts])
            out = out + mx.expand_dims(a_bank[local], -2)
        return out


def _layers_of(model: Any):
    """The decoder layers, wherever the wrapper put them. Qwen3.5's
    multimodal Model nests the text stack as language_model.model.layers;
    plain text models use model.layers; some classes expose layers
    directly."""
    root = model
    for chain in (("language_model", "model"),
                  ("model",),
                  ()):
        probe = root
        for name in chain:
            if not hasattr(probe, name):
                probe = None
                break
            probe = getattr(probe, name)
        if probe is not None and hasattr(probe, "layers"):
            return probe.layers
    return getattr(model, "layers", []) or []


def prime_resident(model: Any) -> None:
    """Eval every parameter EXCEPT the paged expert sources.

    A lazy load leaves the whole model as file-backed mmap. The
    always-resident part — attention, DeltaNet, norms, the router, the
    shared expert, embeddings, the head — is what every single call
    touches, so it is evaluated here, once, into RAM. The stacked expert
    banks under `switch_mlp.source` stay unevaluated: those bytes live on
    the SSD until a call routes to them and the pager pulls the row it
    needs. (Measured cost of skipping this eval: the whole model
    materializes at first forward instead — an 18 GB wired bill for a 16
    GB machine.)
    """
    if not callable(getattr(model, "parameters", None)):
        return  # not a model — nothing to prime (unit-test fakes pass here)
    resident = []
    for key, arr in tree_flatten(model.parameters()):
        key = key[1:] if key.startswith(".") else key
        if ".source." in key:
            continue
        resident.append(arr)
    # CHUNKED eval, deliberately NOT one eval of everything: a single
    # command buffer carrying every kernel of a 20 GB model needs more
    # command-buffer resources than Metal will grant — it surfaces as
    # "Insufficient Memory" at whatever eval happens later (or, at bigger
    # sizes, as a balloon-then-SIGKILL). One small buffer at a time keeps
    # both the GPU pool and the RAM bounded while priming.
    chunk = 64
    for i in range(0, len(resident), chunk):
        mx.eval(*resident[i:i + chunk])


def apply_paging(model: Any, config: dict | None = None,
                 hot: int | None = None, prime: bool = True,
                 model_source: str | None = None) -> Any:
    """Walk a loaded model and put every SparseMoE layer behind a pager.

    `hot` is warm EXPERTS per layer; when it is not given it is read from
    model.paged_experts_per_layer (0 = off, the default — paging changes
    nothing until a model needs it). model.paged_warm_gb is the whole
    model's warm-byte cap — the knob that keeps the SSD quiet without
    letting the warm set eat the machine — and
    model.paged_experts_max_per_layer stays as an optional per-layer
    ceiling. Dense MLP layers and every attention, embedding and shared
    expert stay untouched: they are the always-resident budget, not the
    bank.

    model_source is where the weights live (a repo id the hub resolves, or
    a directory): with it, the expert parts are read at ROW granularity
    from the files — TRUE SSD streaming, nothing of an untouched bank comes
    resident. Without it (or when the checkpoint's tensors don't map), the
    layers fall back to lazy slices: MLX materializes a whole bank on
    first touch.

    A lazy load is REQUIRED either way (mlx_lm evals everything otherwise):
    the caller passes lazy=True — modelload does when paging is on — and
    the priming here evaluates the always-resident part so the model is
    ready to generate.
    """
    model_cfg = (config or {}).get("model", {}) or {}
    if hot is None:
        hot = int(model_cfg.get("paged_experts_per_layer", 0) or 0)
    if hot <= 0:
        return model
    warm_gb = float(model_cfg.get("paged_warm_gb", 0) or 0)
    cap = int(model_cfg.get("paged_experts_max_per_layer", 0) or 0)
    warm = _GlobalWarm(int(warm_gb * 2 ** 30))
    layers = _layers_of(model)

    # One row reader per holder style ("mlp" / "feed_forward"); layers wrap
    # against the reader that mapped THEIR checkpoint names.
    readers = {}
    if model_source:
        for name in ("mlp", "feed_forward"):
            if any(hasattr(getattr(l, name, None), "switch_mlp")
                   for l in layers):
                r = build_row_reader(model_source, layers, name)
                if r is not None:
                    readers[name] = r

    wrapped = 0
    for idx, layer in enumerate(layers):
        holder_name = next((n for n in ("mlp", "feed_forward")
                            if hasattr(getattr(layer, n, None), "switch_mlp")),
                           None)
        if holder_name is None:
            continue
        holder = getattr(layer, holder_name)
        switch = holder.switch_mlp
        if isinstance(switch, PagedSwitchGLU):
            continue
        reader = readers.get(holder_name)
        row_names = (reader.layer_names.get(idx) if reader is not None
                     else None)
        paged_switch = PagedSwitchGLU(switch, hot, warm, cap,
                                      reader=reader, row_names=row_names)
        paged_switch._idx = idx
        holder.switch_mlp = paged_switch
        wrapped += 1
    streamed = sum(1 for r in readers.values() for idxs in [r.layer_names]
                   for _ in idxs)
    if wrapped:
        print(f"  [paged-moe] {wrapped} expert layers behind a warm set of "
              f"{hot} experts (global warm budget {warm_gb or '∞'} GB); "
              f"{streamed} layers stream rows from the weight files, the "
              f"others fall back to whole-bank lazy reads. Set "
              f"model.paged_experts_per_layer: 0 to load resident.")
    if prime:
        prime_resident(model)
    return model


def build_row_reader(model_source: str, layers: list,
                     holder_name: str) -> "_RowReader | None":
    """Map every paged layer's expert parts to their (shard, byte-range)
    entries, parsed from the checkpoint's own safetensors headers.

    Returns None (and says so) when the checkpoint's tensors don't match
    the paged layer's shape of the world — the layers then fall back to
    lazy slices: the machine's own whole-tensor streaming, honest but not
    fine-grained.
    """
    from pathlib import Path as _Path

    root = _Path(model_source)
    if not root.is_dir():
        try:
            from huggingface_hub import snapshot_download
            # ASK ONLY FOR WHAT THE READER READS: mlx_lm's own download
            # excludes README/.gitattributes from its patterns, so a full
            # snapshot check reports those as missing FOREVER — which once
            # silently killed the mapping on every mlm-served checkpoint
            # (measured 2026-10-15, the 35B and the 24B both). The shards
            # are complete; only asking for them matters.
            root = _Path(snapshot_download(
                model_source, local_files_only=True,
                allow_patterns=["*.safetensors"]))
        except Exception as e:
            print(f"  [paged-moe] cannot see the model's files from here "
                  f"({type(e).__name__}); layers fall back to lazy "
                  f"whole-bank reads")
            return None
    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        return None
    entries: dict = {}
    for shard in shards:
        entries.update(_parse_shard_header(shard))

    # A checkpoint names a layer's tensors under one of three key shapes,
    # tried in order: (1) the runtime's split form (switch_mlp.gate_proj …),
    # (2) Qwen's merged gate|up pair (experts.gate_up_proj — mlx_lm's
    # sanitize splits it in memory; here the split is a byte-range of the
    # row: the row is (2*out, packed), gate the first half, up the rest),
    # (3) a bare per-projection experts form. The file keeps its config's
    # "model." prefix; in-memory trees do not.
    def _find(idx, holder, proj, part):
        for pattern in (
            "language_model.model.layers.{i}.{h}.switch_mlp.{p}.{t}",
            "model.layers.{i}.{h}.switch_mlp.{p}.{t}",
            "layers.{i}.{h}.switch_mlp.{p}.{t}",
            "language_model.model.layers.{i}.{h}.experts.{p}.{t}",
            "model.layers.{i}.{h}.experts.{p}.{t}",
        ):
            if pattern is None:
                continue
            name = pattern.format(i=idx, h=holder, p=proj, t=part)
            if name in entries:
                return name
        return None

    reader = _RowReader(entries)
    expected = mapped = 0
    for idx, layer in enumerate(layers):
        switch = getattr(getattr(layer, holder_name, None), "switch_mlp",
                         None)
        if switch is None:
            continue
        expected += 1
        quantized = hasattr(switch.gate_proj, "scales")
        layer_names = {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            parts = {}
            for part in ("weight", "scales", "biases", "bias"):
                hit = _find(idx, holder_name, proj, part)
                if hit is None and proj in ("gate_proj", "up_proj"):
                    merged = _find(idx, holder_name, "gate_up_proj", part)
                    if merged is not None:
                        parts[part] = merged
                        continue
                if hit:
                    parts[part] = hit
            if not ("weight" in parts
                    and ("scales" in parts or not quantized)):
                layer_names = {}
                break
            layer_names[proj] = parts
        if layer_names:
            mapped += 1
            reader.layer_names[idx] = layer_names

    if mapped != expected:
        print(f"  [paged-moe] only {mapped}/{expected} layers' expert "
              f"tensors mapped to file ranges; falling back to lazy "
              f"slices (whole-tensor residency)")
        return None
    return reader