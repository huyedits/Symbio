"""Verify the boot-time KV-cache prefetch against a real model.

The 10 unit tests in test_prompt_cache.py cover this logic with fakes. What
fakes cannot tell you is whether reading a real multi-hundred-MB safetensors
cache on a background thread, while mlx_lm.load() is allocating its own
buffers, actually works on this machine and produces a cache the model
accepts. That is what this checks.

USAGE — two steps, one model load each.

  1. Produce a cache file by using the app normally, if you have not already:
         python main.py
     Say anything, then exit. On a clean exit it writes
     cache/system_prompt.safetensors.

  2. Run this from the repo root (NOT from a worktree — a worktree reseeds
     security.md to the permissive default, which changes the system prompt
     and therefore the cache signature):
         python verify_prefetch.py

Exit code 0 means the prefetch ran, was consumed, and the cache was accepted.
Non-zero means it fell back to a normal load, with the reason printed.

Safety: refuses to start while an mlx_lm trainer is alive, and waits out a
recently-exited one, because a trainer's Metal teardown racing a fresh
multi-GB allocation is this machine's documented kernel-panic pattern.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from symbio import constants
from symbio.app import chat, config as config_mod


def _trainer_alive() -> str | None:
    """Return a description of a live mlx_lm trainer, or None."""
    try:
        out = subprocess.run(["pgrep", "-fl", "mlx_lm"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return None
    for line in out.splitlines():
        if "lora" in line or "--train" in line:
            return line.strip()
    return None


def _guard(settle_seconds: float = 30.0) -> None:
    alive = _trainer_alive()
    if alive:
        sys.exit(f"REFUSING: a trainer is running, loading a model now is the "
                 f"panic pattern.\n  {alive}\nWait for it to finish, then retry.")
    print(f"  No trainer running. Settling {settle_seconds:.0f}s in case one "
          f"just exited...")
    time.sleep(settle_seconds)


class Probe:
    """The real cache methods, bound onto the attributes they touch.

    Same technique as test_prompt_cache.py's FakeSession, but pointed at the
    real config, the real cache file, and a real model load.
    """

    _log_info = chat.ChatSession._log_info
    _prompt_cache_signature = chat.ChatSession._prompt_cache_signature
    _load_persisted_prompt_cache = chat.ChatSession._load_persisted_prompt_cache
    _start_prompt_cache_prefetch = chat.ChatSession._start_prompt_cache_prefetch
    _take_prefetched_cache = chat.ChatSession._take_prefetched_cache

    def __init__(self, cfg):
        self.config = cfg
        self.adapter_loaded = (constants.ADAPTER_DIR / "adapter_config.json").exists()
        self._prompt_cache = None
        self._cached_prompt_ids = None
        self._prefetch_thread = None
        self._prefetched_cache = None
        self.logged: list[str] = []
        self.logger = type("L", (), {"info": lambda s, m: self.logged.append(m)})()


def main() -> int:
    path = constants.PROMPT_CACHE_FILE
    if not path.exists():
        return _fail(f"No cache file at {path}\n"
                     f"  Run `python main.py` once and exit cleanly to write one.")

    size_mb = path.stat().st_size / 1024 ** 2
    print(f"\nKV prefetch verification")
    print(f"  cache file : {path} ({size_mb:.0f} MB)")

    cfg = config_mod.load_config()
    model_name = cfg.get("model_name")
    print(f"  model      : {model_name}")
    print(f"  adapter    : {'loaded' if (constants.ADAPTER_DIR / 'adapter_config.json').exists() else 'none'}")

    _guard()

    probe = Probe(cfg)

    # Count reads of the cache file. The whole point of the prefetch is that
    # the file is read ONCE, on the background thread, and handed to
    # _load_persisted_prompt_cache — which must not read it a second time.
    reads: list[float] = []
    real_loader = chat.load_prompt_cache

    def counting_loader(*a, **kw):
        reads.append(time.monotonic())
        return real_loader(*a, **kw)

    chat.load_prompt_cache = counting_loader
    try:
        t0 = time.monotonic()
        probe._start_prompt_cache_prefetch()
        started = time.monotonic() - t0
        print(f"\n  prefetch thread started at +{started * 1000:.0f} ms")

        from mlx_lm import load
        print(f"  loading weights (this is the window the read overlaps)...")
        model, tokenizer = load(model_name)
        loaded_at = time.monotonic() - t0
        print(f"  weights loaded at +{loaded_at:.2f} s")

        if not reads:
            return _fail("the prefetch thread never read the file")
        read_at = reads[0] - t0
        print(f"  cache read began at +{read_at * 1000:.0f} ms "
              f"({'DURING the load' if read_at < loaded_at else 'AFTER the load'})")

        # Rebuild the exact ids the running app signs with. This must match
        # _prefill_system_prompt_cache byte for byte — system prompt rendered
        # with an EMPTY user turn, no generation prompt, thinking off — or the
        # signature will not match and this reports a failure that isn't one.
        system_prompt = chat.prompts.build_system_prompt(
            probe.config["assistant_name"], probe.config["user_name"])
        templated = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": ""}],
            tokenize=False, add_generation_prompt=False, enable_thinking=False,
        )
        system_ids = tokenizer.encode(templated)
        print(f"  system prefix    : {len(system_ids)} tokens")
        accepted = probe._load_persisted_prompt_cache(system_ids)

        print(f"\n  file reads total : {len(reads)}  (1 = prefetch consumed, 2 = re-read)")
        print(f"  cache accepted   : {accepted}")
        for line in probe.logged:
            print(f"    log: {line}")

        if len(reads) != 1:
            return _fail(f"expected exactly 1 read, saw {len(reads)} — "
                         f"the prefetch was not consumed")
        if not accepted:
            return _fail("signature rejected the cache — it was read but not "
                         "usable. Expected if the model or adapter changed "
                         "since the file was written; a bug otherwise.")
        if read_at >= loaded_at:
            return _fail("the read did not overlap the load — the prefetch "
                         "gained nothing")

        n = len(probe._cached_prompt_ids or [])
        kib = 2 * 36 * 8 * 128 * 2 / 1024
        print(f"\n  PASS — {n} tokens restored without re-prefilling them.")
        print(f"  Approx KV held: {n * kib / 1024:.0f} MiB at {kib:.0f} KiB/token.")
        return 0
    finally:
        chat.load_prompt_cache = real_loader


def _fail(msg: str) -> int:
    print(f"\n  FAIL — {msg}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
