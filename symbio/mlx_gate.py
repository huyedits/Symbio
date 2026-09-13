"""Lazy, legible access to the MLX inference engine.

`import symbio` must succeed on a host with no engine. Config constants, the
load/save helpers and SessionStore have nothing to do with a model, and they
used to be held hostage to one: an eager `from mlx_lm import load` in the
legacy modules made `import symbio` itself die with a traceback ending in
"No module named 'mlx_lm'" (measured 2026-08-27 from a bare install, and
independently by two external reviews since).

The engine is imported here, and only here, at model-load time. The legacy
entry points that need engine symbols (chat, llm, agent) resolve them through
`attr()` on first use, so importing the package costs nothing. The only
failure a user should ever see is the install hint, raised from the same
place a raw traceback used to surface.

This module must never import mlx or mlx_lm itself — that is the point.

Boundary: this gate covers the whole repo — the legacy `symbio` package and
`symbio.app` alike. The app tree used to be engine-on by design (its package
__init__ patched mlx-lm at import time); it is equally engine-free now, with
symbio.app.mlx_compat registering its model patches as a prepare hook instead.
Importing `symbio.app` still guarantees the hook is registered, because its
package __init__ imports mlx_compat for the side effect — so the compat-before-
load invariant holds from either tree, and a clean install can import
anything short of actually driving a model.
"""
from __future__ import annotations

from importlib import import_module
from threading import Lock

# Byte-identical to the old agent.py import guard, so anything matching on the
# text (a desktop crash dialog, a log scrape) keeps working untouched.
HINT = (
    "Symbio's inference engine is not installed.\n"
    "\n"
    "  On Apple Silicon:  pip install 'symbio-cli[mlx]'\n"
    "\n"
    "MLX runs on macOS with an M-series chip only — it publishes no Linux "
    "or Windows wheels — so on other platforms symbio-cli installs but "
    "cannot run a model yet. A CUDA backend is in progress: "
    "https://github.com/huyedits/Symbio"
)

_ENGINE_BASE = "mlx_lm"
# The vision worker rides a separate package (mlx_vlm depends on mlx_lm, so
# importing it first would pull the engine in bypassing the prepare hooks).
# Treat its absence as the same install problem.
_ENGINE_NAMES = ("mlx_lm", "mlx", "mlx_vlm")
_lock = Lock()
_loaded: dict[str, object] = {}
_prepare_hooks: list = []


def _is_engine_missing(exc: ModuleNotFoundError) -> bool:
    return any(
        exc.name == name or (exc.name or "").startswith(name + ".")
        for name in _ENGINE_NAMES
    )


def on_engine_prepare(hook) -> None:
    """Register a hook that must run before the engine is first imported.

    symbio.app.mlx_compat registers the model patches here. Because importing
    the mlx_lm package for a model is always preceded by this gate, the patch
    ordering invariant ("compat before any mlx_lm.load()") holds everywhere —
    for the legacy tree as much as for symbio.app — without the compat module
    itself importing the engine at module scope.
    """
    _prepare_hooks.append(hook)


def _prepare_engine() -> None:
    for hook in _prepare_hooks:
        try:
            hook()
        except ModuleNotFoundError as exc:
            if not _is_engine_missing(exc):
                raise
            raise ModuleNotFoundError(HINT) from exc


def require(module_name: str) -> object:
    """The module named, imported once. A missing engine fails with the hint.

    Any ModuleNotFoundError whose name is the engine or an mlx_lm submodule is
    an engine-absent failure and gets the install hint with the original
    exception chained; anything else re-raises untouched. Registered prepare
    hooks (mlx_compat's model patches) run before the first engine import.
    """
    module = _loaded.get(module_name)
    if module is None:
        _prepare_engine()
        with _lock:
            module = _loaded.get(module_name)
            if module is None:
                try:
                    module = import_module(module_name)
                except ModuleNotFoundError as exc:
                    if not _is_engine_missing(exc):
                        raise
                    raise ModuleNotFoundError(HINT) from exc
                _loaded[module_name] = module
    return module


def attr(spec: str) -> object:
    """One engine symbol by dotted path: ``"mlx_lm.load"`` or
    ``"mlx_lm.sample_utils.make_sampler"``. The deepest module is imported
    lazily, then the final name is read off it."""
    head, _, tail = spec.rpartition(".")
    if not head:
        return require(_ENGINE_BASE)
    return getattr(require(head), tail)
