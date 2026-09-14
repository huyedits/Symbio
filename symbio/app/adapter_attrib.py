"""Which part of an adapter caused the regression — asked by switching it off.

When a fine-tune regresses, the whole adapter is currently thrown away. That is
the safe move and an expensive one: a run that improved fifteen things and broke
one loses all sixteen.

It does not have to be a guess which part broke it, because LoRA is additive and
linear. Each targeted module contributes `scale · B·A` on top of a frozen weight,
so zeroing `lora_b` for one module makes that module contribute EXACTLY nothing
while every key, shape and dtype in the file stays as it was — the loader cannot
tell the difference. That turns "which tensors caused this" into an experiment
you can actually run: switch a subset off, re-run the cases that regressed, and
see whether they come back.

On the live 14B adapter that search space is 16 modules — 8 layers (28-35) by
two projections (q_proj, v_proj) — so delta-debugging finds a minimal culprit
set in a handful of evaluations rather than sixteen.

Two things follow from an answer:

  * REPAIR. Zero the implicated modules, re-run the FULL battery, and keep the
    adapter if it now passes. The other fifteen modules' gains survive.
  * QUARANTINE. Remember the modules, and zero them again after the next run.

What this does NOT claim: that the implicated module is "the cause" in any
deeper sense. A regression is usually distributed across an adapter, and what
delta-debugging finds is the smallest set whose removal fixes the SYMPTOM. That
is the useful question, and it is the only one an experiment can answer.

Nor is it free of noise. Golden cases are sampled generations, so a case can
flip on its own; `check` is the caller's to make repeat-resistant, and if it is
not, this will chase noise very efficiently.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Callable, Iterable, Sequence

ADAPTER_FILE = "adapters.safetensors"

# The half of a LoRA pair that is zero at initialisation, and so the half to
# zero when switching a module off: B·A is zero if B is, and a file whose keys
# and shapes are untouched loads exactly as before.
_B_SUFFIXES = ("lora_b", "lora_B", "lora_B.weight")


def _weights_path(adapter_dir: Path) -> Path:
    return Path(adapter_dir) / ADAPTER_FILE


def load_weights(adapter_dir: Path) -> dict:
    """Every tensor in an adapter, as numpy arrays."""
    from safetensors.numpy import load_file

    return load_file(str(_weights_path(adapter_dir)))


def modules(adapter_dir: Path) -> list[str]:
    """The LoRA modules in an adapter — one per (layer, projection) pair.

    Ordered by layer then projection so a bisection splits along the model's
    own structure rather than alphabetically, which keeps a culprit that is
    really "the deepest layers" contiguous and findable in one cut.
    """
    names: set[str] = set()
    for key in load_weights(adapter_dir):
        stem = key.rsplit(".", 1)[0]
        if key.rsplit(".", 1)[-1].startswith("lora"):
            names.add(stem)
        else:
            names.add(stem)

    def _sort_key(name: str):
        parts = name.split(".")
        layer = next((int(p) for p in parts if p.isdigit()), -1)
        return (layer, name)

    return sorted(names, key=_sort_key)


def write_ablated(src_dir: Path, dst_dir: Path, drop: Iterable[str]) -> int:
    """Copy an adapter with `drop`'s modules switched off. Returns how many.

    Everything else in the directory comes along — adapter_config.json above
    all, because the loader reads it and a weights file without one is not an
    adapter.
    """
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    drop = set(drop)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        if item.is_file() and item.name != ADAPTER_FILE:
            shutil.copy2(item, dst_dir / item.name)

    from safetensors.numpy import save_file

    weights = load_weights(src_dir)
    zeroed = set()
    for key, tensor in weights.items():
        stem = key.rsplit(".", 1)[0]
        if stem in drop and key.rsplit(".", 1)[-1] in _B_SUFFIXES:
            weights[key] = tensor * 0
            zeroed.add(stem)
    save_file(weights, str(_weights_path(dst_dir)))
    return len(zeroed)


def bisect_blame(candidates: Sequence[str],
                 check: Callable[[frozenset[str]], bool],
                 max_calls: int = 24) -> tuple[list[str], int]:
    """The smallest subset of `candidates` whose removal satisfies `check`.

    `check(dropped)` answers one question: with these modules switched off, do
    the cases that regressed pass again?

    This is delta-debugging (ddmin), not a plain binary search, and the
    difference matters: a regression can need two modules removed together,
    and a binary search that tries only halves would find neither and report
    nothing. ddmin widens its granularity when no single chunk works, so an
    interaction is found rather than missed.

    Returns (culprits, evaluations_spent). An empty list means the regression
    is not attributable to any subset — zeroing EVERYTHING did not bring the
    cases back — which is itself worth knowing: it says the adapter is not
    what broke them, so the battery is flaky or something else moved.

    `max_calls` is a real bound, not a safety net: every evaluation is a batch
    of sampled generations, and minimality is worth far less than finishing.
    Measured on the live 16-module adapter, a single culprit costs 5-9
    evaluations but a two-module interaction costs 50 — so the budget stops
    the search early and returns the smallest set CONFIRMED so far. That set
    is always a real answer: `current` is only ever replaced by a subset that
    passed `check`, so the value returned satisfies it too. It may just be
    larger than strictly necessary, which for repair means zeroing a module
    that did not need it — cheap, and honest about which it was.
    """
    calls = 0

    def _check(subset: Iterable[str]) -> bool:
        nonlocal calls
        calls += 1
        return bool(check(frozenset(subset)))

    current = list(candidates)
    if not current or calls >= max_calls:
        return [], calls
    # The sanity gate. Without it, ddmin happily "minimises" its way to a
    # single module on a battery that was never going to pass anyway.
    if not _check(current):
        return [], calls

    granularity = 2
    while len(current) > 1 and calls < max_calls:
        size = max(1, len(current) // granularity)
        chunks = [current[i:i + size] for i in range(0, len(current), size)]
        for chunk in chunks:
            if calls >= max_calls:
                break
            if _check(chunk):
                current, granularity = chunk, 2
                break
        else:
            # No single chunk does it on its own: the culprit spans chunks.
            # Try each complement, which is what finds a two-module
            # interaction without testing every pair.
            for chunk in chunks:
                if calls >= max_calls:
                    break
                complement = [c for c in current if c not in chunk]
                if complement and _check(complement):
                    current = complement
                    granularity = max(2, granularity - 1)
                    break
            else:
                if granularity >= len(current):
                    break
                granularity = min(len(current), granularity * 2)
                continue
    return current, calls


# --------------------------------------------------------- ablating in memory
#
# The search has to evaluate a dozen or more subsets, and writing each one to
# disk and reloading the model would cost minutes per evaluation on a 14B —
# hours for one attribution, which is the same as not having it. So the search
# switches modules off on the RESIDENT model instead: mlx_lm's LoRALinear keeps
# lora_a/lora_b as live attributes, and assigning a zero matrix to lora_b makes
# that module contribute nothing, instantly and exactly. The originals are held
# and put back, so the model is bit-identical afterwards.


def live_lora_modules(model) -> dict:
    """Every LoRA layer on a loaded model, by its dotted path.

    Empty for anything that is not a walkable model — a base model with no
    adapter applied, or a stand-in handed in by a test. Repair is an
    optimisation on top of rollback, so a model it cannot read must make it
    decline, never raise: the caller's next move is the rollback that was
    always going to happen.
    """
    walk = getattr(model, "named_modules", None)
    if not callable(walk):
        return {}
    found = {}
    try:
        for name, module in walk():
            if hasattr(module, "lora_a") and hasattr(module, "lora_b"):
                found[name] = module
    except Exception:
        return {}
    return found


class switched_off:
    """Context manager: zero these modules' lora_b, then restore them exactly.

    Restoration is unconditional — an evaluation that raises must not leave the
    resident model quietly missing half its adapter for the rest of the
    session, which would be a far worse bug than the regression being chased.
    """

    def __init__(self, model, names: Iterable[str]):
        self.modules = live_lora_modules(model)
        self.names = [n for n in names if n in self.modules]
        self.saved: dict[str, object] = {}

    def __enter__(self):
        import mlx.core as mx

        for name in self.names:
            module = self.modules[name]
            self.saved[name] = module.lora_b
            module.lora_b = mx.zeros_like(module.lora_b)
        return self

    def __exit__(self, *exc):
        for name, original in self.saved.items():
            self.modules[name].lora_b = original
        self.saved.clear()
        return False


def match_file_names(live_names: Iterable[str],
                     file_names: Iterable[str]) -> dict[str, str]:
    """Map live module paths onto the names used inside the adapter file.

    The two differ by a prefix — a file says `model.layers.28.self_attn.q_proj`
    where the loaded tree may say `layers.28.self_attn.q_proj` — so they are
    matched on the longest common suffix rather than assumed equal.
    """
    out: dict[str, str] = {}
    files = list(file_names)
    for live in live_names:
        for candidate in files:
            if candidate == live or candidate.endswith("." + live) \
                    or live.endswith("." + candidate):
                out[live] = candidate
                break
    return out


# ------------------------------------------------------------- the quarantine

QUARANTINE_FILE = "quarantine.json"


def quarantine_path(adapter_dir: Path) -> Path:
    return Path(adapter_dir) / QUARANTINE_FILE


def read_quarantine(adapter_dir: Path) -> list[dict]:
    path = quarantine_path(adapter_dir)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def record_quarantine(adapter_dir: Path, culprits: Sequence[str],
                      cases: Sequence[str], when: str) -> list[dict]:
    """Remember that these modules broke these cases, and when.

    Kept as a list of events rather than a set of names because the same
    module implicated twice, months apart, on two different cases is a much
    stronger signal than one implicated once — and a set throws that away.
    """
    entries = read_quarantine(adapter_dir)
    entries.append({
        "modules": list(culprits),
        "cases": list(cases),
        "at": when,
    })
    try:
        quarantine_path(adapter_dir).write_text(
            json.dumps(entries, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass
    return entries


def repeat_offenders(adapter_dir: Path, minimum: int = 2) -> list[str]:
    """Modules implicated in at least `minimum` separate regressions."""
    counts: dict[str, int] = {}
    for entry in read_quarantine(adapter_dir):
        for name in entry.get("modules", []):
            counts[name] = counts.get(name, 0) + 1
    return sorted(name for name, n in counts.items() if n >= minimum)
