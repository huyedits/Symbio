"""Configuration: defaults, loading, and model-driven self-configuration."""

import copy
import json
import os
from typing import Any

from symbio import constants


DEFAULT_CONFIG: dict[str, Any] = {
    "model_name": "Qwen/Qwen3-0.6B",
    "assistant_name": "",
    "user_name": "",
    "lora": {
        "rank": 8,
        "dropout": 0.0,
        "scale": 20.0,
        # How many transformer blocks get adapters, counted from the output
        # end — mlx_lm always takes a trailing run, never a chosen set. Note
        # that 0 (or negative) means ALL blocks, not none.
        "num_layers": 8,
        # Which modules inside those blocks get adapters. Empty = every
        # projection (mlx_lm's default). Block-relative names such as
        # "self_attn.q_proj" narrow it; full paths such as
        # "model.layers.12.mlp.up_proj" are matched against the whole model
        # and are the only way to target specific blocks. See
        # training.validate_lora_keys for what is checked before a run.
        "keys": [],
        # Trade compute for activation memory. Worth turning on before
        # lowering max_seq_length, since activations are what scale with it.
        "grad_checkpoint": False,
        "batch_size": 1,
        "learning_rate": 1e-4,
        # Floor for a full retrain, not the count itself: run_training scales
        # iterations to cover the corpus `epochs` times (see
        # training.iters_for_corpus), because a fixed count over a growing
        # corpus means each retrain sees an arbitrary subset of it.
        "iters": 300,
        "epochs": 2,
        "max_iters": 2000,
        # How long to wait after a trainer child exits before loading a model
        # again. The child hands every Metal buffer back in one bulk teardown,
        # and a multi-gigabyte allocation landing in the middle of that is what
        # panics IOGPUFamily on Apple silicon — three kernel panics in twenty
        # minutes on 2026-08-17. vm_stat reports the pages free immediately,
        # so this floor covers the driver-side reclaim that cannot be observed
        # from userspace; settle_free_gb then polls for the part that can.
        # Set to 0 to disable (there is no userspace fix, only less exposure).
        "settle_after_training_seconds": 15,
        "settle_free_gb": 6.0,
        "settle_timeout_seconds": 180,
        # Ceiling on how many passes the `iters` floor may buy a small corpus.
        # Without it the floor stops being a floor: at 6 samples the old 150
        # meant 25 epochs, which memorises the samples instead of learning
        # from them.
        "max_epochs": 4,
        # Compute the loss only over the assistant's answer. Without this the
        # system prompt — ~99% of every sample's tokens, and identical across
        # all of them — dominates the gradient, and the run measures how well
        # the model memorised a constant it is handed at inference anyway.
        "mask_prompt": True,
        # Must exceed a whole sample: system prompt (tool catalog stripped by
        # training.strip_tool_catalog) + user turn + assistant turn. Seed
        # samples run ~2,200 tokens, so anything under ~2,560 silently cuts the
        # assistant turn off and the sample teaches nothing. run_training warns
        # when the corpus does not fit.
        "max_seq_length": 3072,
        "steps_per_eval": 100,
        # Validation batches per evaluation. mlx_lm's own default of 25 rescores
        # the whole split every time, which on ~2k-token samples costs minutes
        # per check and can dominate a run. This is a plateau detector, not a
        # benchmark.
        "val_batches": 8,
        # Checkpoint interval, and it must divide steps_per_eval. A crash
        # between checkpoints destroys everything since the last one, and an 8B
        # LoRA step runs ~20s here, so 100 iters is a ~33 minute hole. The
        # divisibility matters as much as the size: early stopping restores the
        # best-scoring *evaluated* step, so an interval that never lands on an
        # eval point leaves every best step without a file and hands the run's
        # output to _restore_best's fallback.
        # 25 divides the 100 above; config.json's 30/15 pairing does too.
        "save_every": 25,
        # Early stopping: kill training if validation loss stops improving.
        "early_stop_enabled": True,
        "early_stop_patience": 2,
        "early_stop_min_delta": 0.005,
    },
    "agent": {
        "max_tool_rounds": 3,
        "history_limit": 20,
        "sandbox_timeout": 30,
        "code_timeout": 60,
        "max_output_len": 4000,
        # Page text is read deliberately ("read the page"), so it gets a
        # larger budget than generic command output — 4000 chars cut
        # example.com-sized pages mid-sentence. Raising it costs prompt:
        # measured on the 14B, decode falls from ~26 tok/s at 1.5k of
        # context to ~10 tok/s at 6k, so this is a speed dial too.
        "max_page_chars": 12000,
        # The unasked-for snapshot appended after every browser action.
        # Kept small on purpose: it lands on EVERY action, not just the
        # ones where the page is the answer.
        "browser_peek_chars": 1500,
        # Short replies keep the model fast: tags + short prose fit easily;
        # long answers can still be requested explicitly. 128 is a sweet spot
        # for quick chat on local MLX.
        "max_reply_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
        "cron_poll_seconds": 20,
        "stream_output": True,
        # Surface the model's Qwen3 thinking block to the user as a
        # "[Reasoning] …" block before the answer. Set false to hide it.
        "show_reasoning": True,
        # How hard the model is asked to think before answering: none, low,
        # medium or flurry (see chat.THINKING_LEVELS). Change it live with
        # /think. "none" ends the prompt with an empty closed think block, so
        # the model answers directly; the others leave the block open and give
        # the reasoning a token allowance on top of max_reply_tokens.
        #
        # Defaults to none because that is how the tool-call round already ran
        # (chat.py used think=round_num > 0), and it is the setting the 21-case
        # battery scored 19/21 on.
        "thinking_level": "low",
        # Speculative decoding: a small model drafts, the real one verifies.
        # Empty disables it. Keep num_draft_tokens low on hybrid
        # linear-attention models (Qwen3.5) — deeper drafts lose there.
        "draft_model": "",
        "num_draft_tokens": 1,
        "prompt_cache_enabled": True,
        # Keep the warmed system-prefix KV cache on disk between runs, so a
        # restart reloads it instead of re-prefilling ~4.3k tokens through the
        # model. Costs a few hundred MB in cache/ (KV for the whole prefix);
        # set false to trade the faster start back for the disk.
        "persist_prompt_cache": True,
        # Start reading that persisted cache off disk *while* the weights are
        # still loading, rather than after. The two are the slowest parts of
        # boot and contend for almost nothing, so overlapping them hides the
        # read almost entirely. Only the read is overlapped — the cache is not
        # materialized onto the GPU until the weight load has finished, so
        # there is never a second Metal client during the load window.
        # Set false on a memory-tight machine: the read raises peak page-cache
        # residency at the moment the load is already at its high-water mark.
        "prefetch_prompt_cache_during_load": True,
        # How long a chat front-end should wait before showing a "thinking…"
        # placeholder if the model has not emitted a visible token yet.
        "first_chunk_timeout_ms": 600,
        # Maximum character budget for the retained conversation window. One
        # giant observation (e.g. a full web page) can otherwise bloat every
        # later turn even with a turn-count history limit.
        "max_history_chars": 4000,
        # Lower temperature during tool-use rounds makes the model follow the
        # prompt's tag rules (browse vs cmd, press vs fake keydown) more
        # strictly instead of drifting into prose or hallucinated commands.
        "tool_use_temperature": 0.2,
        # Near-greedy sampling on a small, heavily fine-tuned model degenerates
        # into repetition loops on out-of-distribution input. agent.py has had
        # this since the start; the CLI's own loop (chat.py) and the worker
        # dispatch never did, and both ran without any penalty at all.
        "repetition_penalty": 1.15,
        "repetition_context_size": 64,
        # Speed preset: "balanced" (default) or "fast". "fast" trades context
        # length and RAG budget for snappier turns.
        "speed_mode": "balanced",
        # When true, editing or overwriting an existing file inside the project
        # directory first creates a numbered backup (e.g. file.txt.1.bak). The
        # user can disable this in setup or config if they prefer in-place edits.
        "backup_before_edit": True,
    },
    "browser": {
        # Browser automation is off by default. When enabled, the agent launches
        # its own isolated Google Chrome / Chromium via Playwright, not the
        # user's personal browser profile. It must still ask for confirmation
        # the first time it visits a new domain.
        "enabled": False,
    },
    "web": {
        "search_results": 5,
        "http_timeout": 15,
        "auto_search_when_unsure": True,
        "auto_search_session_cap": 20,
    },
    "sandbox": {
        "blocked_commands": [
            "rm", "sudo", "su", "dd", "mkfs", "fdisk", "mount", "umount",
            "chmod", "chown", "curl", "wget", "ssh", "scp",
            "python", "python3", "perl", "ruby", "php", "node", "npm",
            "bash", "sh", "zsh", "fish",
        ],
        "blocked_shells": ["bash", "sh", "zsh", "fish"],
        "shell_allow_localhost": True,
        "shell_allow_remote_hosts": True,
        "blocked_imports": [
            "os", "sys", "subprocess", "pathlib", "shutil", "socket", "http",
            "urllib", "ftplib", "smtplib", "imaplib", "pickle", "ctypes",
            "multiprocessing", "threading", "tempfile", "asyncio", "importlib",
            "pkgutil", "site", "builtins",
        ],
    },
    "telegram": {
        "enabled": False,
        "bot_token": "",
        "allowed_chat_ids": [],
        "confirm_dangerous": True,
    },
    "rag": {
        "enabled": True,
        # Tight retrieval budget: small top-k and low token cap keep prompt
        # prefill fast while still giving the model relevant notes/sessions.
        "top_k": 3,
        "max_context_tokens": 800,
        "sources": ["notes", "sessions"],
    },
    # Metal/unified-memory ceilings for the long-lived chat process. MLX keeps
    # freed GPU buffers in a cache for reuse, which is the right default for a
    # process that owns the GPU alone — but this one spawns `mlx_lm lora` as a
    # second Metal client during training, and a hoarded cache in the parent is
    # memory the trainer cannot have. Capping it trades a little allocator churn
    # for far less pressure during the window that has the most of it.
    "gpu": {
        # MLX buffer cache ceiling, in MB. 0 disables caching entirely,
        # -1 leaves MLX's default alone.
        "cache_limit_mb": 1024,
        # Wired (non-swappable) memory ceiling, in MB. -1 leaves the default.
        # Only raise this if you know the machine's headroom.
        "wired_limit_mb": -1,
        # Drop the in-process model before spawning the LoRA trainer, so only
        # one copy of the weights is resident at a time.
        "unload_model_during_training": True,
        # Refuse to spawn a trainer when free RAM cannot hold it. What this
        # prevents is not a failed run: a trainer the machine cannot hold gets
        # Jetsam-killed, and on a 16 GB Mac that has already cost a reboot.
        # Set false to train anyway on a machine you have measured yourself.
        "memory_preflight": True,
    },
    # Self-pruning of the stores RAG reads back (see symbio/app/prune.py).
    # Runs once per boot so junk the agent wrote down stops being retrieved
    # as context on every later turn.
    "prune": {
        "enabled": True,
        "on_boot": True,
        "notes": True,
        "sessions": True,
        # Identical (role, content) entries kept per session before the rest
        # are treated as a stuck loop and dropped.
        "session_max_copies": 2,
    },
    "memory": {
        "enabled": True,
        "memory_char_limit": 2200,
        "profile_char_limit": 1375,
        "nudge_interval": 10,
        "flush_min_turns": 6,
        # Compact the curated stores when the machine is under memory pressure,
        # without waiting for the canary to notice adherence has degraded.
        #
        # Be clear about what this does and does not do: compacting shrinks the
        # PROMPT, not the process. The stores are markdown files of a few KB, so
        # freeing them returns almost no RAM. What it buys is a shorter context
        # at the moment the machine is least able to afford a long one — on a
        # 16 GB Mac with a 14B resident, a big prompt is what tips the KV cache
        # into swap.
        "auto_compact_enabled": True,
        # 0.75 sits in the band Huy asked for (~70-80%). A resident 14B alone is
        # already ~55%, so anything much lower would fire on every turn of a
        # normal session.
        "auto_compact_ram_fraction": 0.75,
        # Compaction costs a model call to summarise. Without a floor between
        # attempts, sustained pressure — which is the normal state while a model
        # is resident — would compact every single turn.
        "auto_compact_cooldown_turns": 10,
    },
    "learn": {
        "enabled": True,
        "auto": True,
        "auto_train": True,
        "remember_research": True,
        "note_decay_days": 90,
        "mistake_threshold": 5,
        "batch_train_iters": 25,
        "iters_per_severity": 5,
        "max_batch_train_iters": 100,
        "boost_factor": 3,
        "severe_correction_phrases": [
            "wrong", "incorrect", "you misunderstood", "fix it", "not what",
        ],
        "golden_set_enabled": True,
        "golden_max_tokens": 150,
        # Held-out generalisation check after each training round. Measures
        # only — never rolls back, since failing these early is normal.
        "wildcard_check_enabled": True,
        "wildcard_max_tokens": 200,
        "golden_regression_threshold": 0,
        "golden_rollback_on_regression": True,
        "golden_retry_enabled": True,
        "golden_retry_max_extra_iters": 50,
        "golden_retry_samples_per_case": 3,
        # Also teach golden cases that were already failing when a training
        # round started, not just ones that round broke. Without it a case
        # only ever gets remedy samples the round it regresses, so anything
        # that slipped through stays failing forever.
        "golden_teach_baseline_failures": True,
        "adapter_idle_reminder_enabled": True,
        "adapter_idle_days": 30,
        "correction_phrases": [
            "no,", "not quite", "that's wrong", "incorrect", "wrong",
            "you misunderstood", "try again", "actually", "i meant",
            "i said", "i asked", "not what", "that's not", "you're wrong",
            "fix it", "correction", "rephrase",
        ],
    },
    "eval": {
        "max_eval_tokens": 512,
    },
    "tools": {
        "enabled_groups": [
            "memory", "notes", "terminal", "code", "web_search",
            "digest", "train", "cron", "config", "delegate", "system",
        ],
    },
    "remote": {
        # Hosts available to the run_remote tool. Each alias maps to connection
        # details. If ssh_key is omitted, the agent relies on ssh-agent or
        # default ~/.ssh identities. Shell access is never interactive; use
        # key-based auth or ssh-agent.
        "hosts": {},
    },
    "safety": {
        # Prompt-injection defenses and risk-based escalation.
        "enabled": True,
        # Risk scores: 0=safe, 1=notice, 2=alert, 3=requires approval.
        # Actions at or above this score require explicit confirmation.
        "require_confirm_score": 3,
        # Actions at or above this score get an alert appended to the
        # tool observation so the model sees what it is doing.
        "log_score": 2,
        # Ask before running a tool that has never run here, on a turn where
        # retrieved text entered the context — the shape an injected action
        # takes. Escalates to a confirmation, never to a refusal: this is a
        # behavioural guess, and a wrong guess must cost a keystroke rather
        # than the feature. Each tool asks at most once, then it is baselined
        # in tool_baseline.json.
        "provenance_enabled": True,
        # Ask before a shell/filesystem/settings call on a turn where the user
        # requested no action at all. Escalates to a prompt, never a refusal.
        "intent_gate_enabled": True,
    },
    "dispatch": {
        # Off by default: MoA delegation loads and runs additional models
        # on your machine, which is a bigger behavior/resource change than
        # anything else here — opt in deliberately.
        "enabled": False,
        # The model a worker runs. Deliberately smaller than the headmaster:
        # a worker does one narrow job under a short prompt, and running it at
        # headmaster size means a second full-size copy of the weights resident
        # beside the one already loaded. Set to null to use model_name instead.
        "worker_model_name": "mlx-community/Qwen3-4B-4bit",
        # Switch between workers that share a base model by replacing only
        # their LoRA tensors, instead of loading a second copy of the weights.
        # Skill workers all run the headmaster's own model, so this turns a
        # multi-gigabyte reload into a ~19 MB one. Refuses the swap and falls
        # back to a full load whenever the adapter's shape does not match.
        # When retrieval matches a skill that has its own trained worker, tell
        # the model the worker exists instead of leaving it to infer that from
        # the tool schema. A suggestion, not a route: retrieval is fuzzy, and a
        # wrong hard route costs the whole turn where a wrong hint costs a line.
        "suggest_skill_workers": True,
        # Unload the headmaster before a worker runs and reload it afterwards.
        # A skill worker runs the headmaster's own model and the headmaster is
        # never a hot-swap donor (its model lives in ChatSession, not the pool),
        # so without this every skill dispatch holds two full copies of those
        # weights. Costs a reload per delegation; that is the price of a skill
        # worker at headmaster size on a machine that cannot hold both.
        # The worker is unloaded again before the headmaster comes back, so
        # this really does mean one model resident at a time — reloading the
        # headmaster on top of a still-resident worker was the same double
        # residency one turn later, and it OOM'd the machine.
        "headmaster_deep_sleep_while_workers": False,
        "hot_swap_adapters": True,
        # When a headmaster-sized worker cannot be hot-swapped, the fallback is
        # a second full copy of the headmaster's weights. That is correct and
        # unaffordable: on 16 GB, beside a training run, it is what takes the
        # machine down rather than the turn. Refuse by default and say why;
        # turn this on only if you have the RAM to hold both.
        "allow_second_headmaster_copy": False,
        "max_resident_workers": 1,
        "worker_idle_unload_minutes": 10,
        "max_worker_rounds": 4,
        "worker_golden_set_enabled": True,
        "worker_golden_regression_threshold": 0,
        # Applies to hand-written eval_tasks.json only. Derived checks grade a
        # reply by how much of the steps text it reproduces, so a worker that
        # learns to perform its skill fails them by definition; those report
        # without reverting (see skill_eval.has_custom_tasks).
        "worker_golden_rollback_on_regression": True,
        # Append each worker reply to that worker's training corpus. Off by
        # default: nothing validates the reply first, so this trains a worker
        # on its own unchecked output.
        "capture_worker_samples": False,
        # When a worker adapter regresses on a golden case, synthesize extra
        # training samples from the case's ideal reply and do a targeted
        # retrain before deciding whether to roll back.
        "worker_golden_retry_enabled": True,
        "worker_golden_retry_max_extra_iters": 50,
        "worker_golden_retry_samples_per_case": 3,
    },
    # Anonymous telemetry + /feedback. Off by default; requires an explicit
    # Y/N consent (run_setup_wizard or /telemetry) before anything is sent.
    # No endpoint -> records are kept locally under telemetry/ and never sent.
    "telemetry": {
        "enabled": False,          # collection+send allowed (set by consent)
        "consented": False,        # user has answered the Y/N (Y or N both -> True)
        "endpoint": "",            # Cloudflare Worker URL, https://x.workers.dev/ingest
        "shared_secret": "",       # bearer key sent as the X-Telemetry-Secret header
        "feedback_enabled": True,  # /feedback toggle (/feedback on|off)
        "ping_daily": True,        # auto telemetry ping at most once per day
    },
}

# Keys that must survive a restart to take effect.
_RESTART_KEYS = {"model_name"}


def _env_list(key: str) -> list[str] | None:
    """Parse a comma-separated env var into a list, or None if unset/empty."""
    raw = os.environ.get(key, "").strip()
    if not raw:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def _apply_env_overrides(config: dict[str, Any]) -> None:
    """Let environment variables override key identity/model/secret settings."""
    if os.environ.get("SYMBIO_MODEL_NAME", "").strip():
        config["model_name"] = os.environ["SYMBIO_MODEL_NAME"].strip()
    if os.environ.get("SYMBIO_ASSISTANT_NAME", "").strip():
        config["assistant_name"] = os.environ["SYMBIO_ASSISTANT_NAME"].strip()
    if os.environ.get("SYMBIO_USER_NAME", "").strip():
        config["user_name"] = os.environ["SYMBIO_USER_NAME"].strip()

    token = os.environ.get("SYMBIO_TELEGRAM_TOKEN", "").strip()
    if token:
        config.setdefault("telegram", {})
        config["telegram"]["bot_token"] = token
        config["telegram"]["enabled"] = True

    allowed = _env_list("SYMBIO_TELEGRAM_ALLOWED_CHAT_IDS")
    if allowed:
        config.setdefault("telegram", {})
        try:
            config["telegram"]["allowed_chat_ids"] = [int(x) for x in allowed]
        except ValueError:
            print("[Config warning] SYMBIO_TELEGRAM_ALLOWED_CHAT_IDS contains non-integer values; ignored.")


def save_config(config: dict[str, Any]) -> None:
    """Persist the merged config back to config.json.

    Writes the full merged config so runtime changes (e.g. /auto-index on)
    survive restarts."""
    try:
        constants.CONFIG_FILE.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    except Exception as e:
        print(f"[Config warning] Could not write {constants.CONFIG_FILE}: {e}")


def load_config() -> dict[str, Any]:
    """Load config.json if present; merge with sensible defaults and env overrides."""
    # Deep copy: callers (e.g. set_config_value) mutate nested sections, and a
    # shallow copy would poison DEFAULT_CONFIG for every later load.
    config = copy.deepcopy(DEFAULT_CONFIG)
    if constants.CONFIG_FILE.exists():
        try:
            user_config = json.loads(constants.CONFIG_FILE.read_text(encoding="utf-8"))
            config.update(user_config)
            for section in ("lora", "agent", "rag", "memory", "web", "sandbox", "learn", "telegram", "tools", "dispatch", "archive", "prune", "gpu"):
                if section in user_config:
                    config[section] = {**DEFAULT_CONFIG.get(section, {}), **user_config[section]}
        except Exception as e:
            print(f"[Config warning] Could not read {constants.CONFIG_FILE}: {e}")
    _apply_env_overrides(config)
    _apply_speed_mode(config)
    return config


def apply_gpu_limits(config: dict[str, Any]) -> None:
    """Apply the config's Metal memory ceilings to this process.

    Safe to call more than once, and a no-op when the limits are set to -1 or
    when the installed MLX predates these calls. Importing mlx.core lazily
    keeps config loading usable in processes that never touch the GPU.
    """
    gpu = config.get("gpu", {})
    try:
        import mlx.core as mx
    except Exception:
        return

    for key, setter_name, scale in (
        ("cache_limit_mb", "set_cache_limit", 1024 * 1024),
        ("wired_limit_mb", "set_wired_limit", 1024 * 1024),
    ):
        value = gpu.get(key, -1)
        if value is None or int(value) < 0:
            continue
        setter = getattr(mx, setter_name, None)
        if setter is None:
            # Older MLX exposed these under mx.metal.
            setter = getattr(getattr(mx, "metal", None), setter_name, None)
        if setter is None:
            continue
        try:
            setter(int(value) * scale)
        except Exception:
            # A ceiling we could not set is not worth failing a startup over.
            pass


# Speed preset: applied after user config/env overrides so the user can still
# override individual keys, but flipping one switch retunes the whole agent.
_SPEED_PRESETS: dict[str, dict[str, Any]] = {
    "balanced": {
        "agent": {
            "max_reply_tokens": 128,
            "history_limit": 20,
            "max_history_chars": 4000,
            "first_chunk_timeout_ms": 600,
            "tool_use_temperature": 0.2,
        },
        "rag": {"top_k": 3, "max_context_tokens": 800},
    },
    "fast": {
        "agent": {
            "max_reply_tokens": 64,
            "history_limit": 12,
            "max_history_chars": 2500,
            "first_chunk_timeout_ms": 400,
            "tool_use_temperature": 0.1,
        },
        "rag": {"top_k": 2, "max_context_tokens": 400},
    },
}


def _apply_speed_mode(config: dict[str, Any]) -> None:
    """If agent.speed_mode is set to a known preset, apply the preset values
    unless the user explicitly overrode a specific key in config.json/env."""
    mode = config.get("agent", {}).get("speed_mode", "balanced")
    preset = _SPEED_PRESETS.get(mode)
    if preset is None:
        return

    # Capture which keys the user actually supplied so we don't clobber them.
    user_supplied: set[str] = set()
    if constants.CONFIG_FILE.exists():
        try:
            user = json.loads(constants.CONFIG_FILE.read_text(encoding="utf-8"))
            if "agent" in user:
                user_supplied.update(f"agent.{k}" for k in user["agent"].keys())
            if "rag" in user:
                user_supplied.update(f"rag.{k}" for k in user["rag"].keys())
        except Exception:
            pass

    for section, values in preset.items():
        for k, v in values.items():
            key_path = f"{section}.{k}"
            if key_path not in user_supplied:
                config.setdefault(section, {})[k] = v


def config_show(config: dict[str, Any]) -> str:
    """Return config as pretty JSON, with sensitive values redacted."""
    safe = copy.deepcopy(config)
    if safe.get("telegram", {}).get("bot_token"):
        safe["telegram"]["bot_token"] = "***REDACTED***"
    return json.dumps(safe, indent=2)


def _coerce_like(current: Any, raw: str) -> Any:
    """Parse raw into the same type as the current value."""
    if isinstance(current, bool):
        if raw.lower() in ("true", "yes", "on", "1"):
            return True
        if raw.lower() in ("false", "no", "off", "0"):
            return False
        raise ValueError(f"Expected true/false, got {raw!r}")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        value = json.loads(raw)
        if not isinstance(value, list):
            raise ValueError("Expected a JSON list")
        return value
    return raw


def set_config_value(config: dict[str, Any], key: str, raw_value: str,
                     allow_sandbox: bool = False) -> str:
    """Set a dotted config key (e.g. agent.temperature), persist it to
    config.json, and apply it to the running config. Returns a status message."""
    key = key.strip()
    if key.startswith("sandbox.") and not allow_sandbox:
        return "sandbox.* settings can only be changed by the user via /config set."

    # Resolve the dotted path against the live config to validate it exists.
    parts = key.split(".")
    node: Any = config
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            return f"Unknown config key: {key}"
        node = node[part]
    leaf = parts[-1]
    if not isinstance(node, dict) or leaf not in node or isinstance(node[leaf], dict):
        return f"Unknown config key: {key}"

    try:
        value = _coerce_like(node[leaf], raw_value.strip())
    except Exception as e:
        return f"Bad value for {key}: {e}"
    node[leaf] = value

    # Persist into config.json without disturbing unrelated user settings.
    user_config: dict[str, Any] = {}
    if constants.CONFIG_FILE.exists():
        try:
            user_config = json.loads(constants.CONFIG_FILE.read_text(encoding="utf-8"))
        except Exception:
            user_config = {}
    target = user_config
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[leaf] = value
    constants.CONFIG_FILE.write_text(json.dumps(user_config, indent=2) + "\n", encoding="utf-8")

    note = " (takes effect after restart)" if parts[0] in _RESTART_KEYS else ""
    return f"Set {key} = {value!r}{note}."


_TELEGRAM_TOKEN_ENV = "SYMBIO_TELEGRAM_TOKEN"


def get_telegram_token(config: dict[str, Any], input_fn=input) -> str | None:
    """Return the Telegram bot token, in order of preference:
    1. SYMBIO_TELEGRAM_TOKEN environment variable
    2. config["telegram"]["bot_token"]
    3. Prompt the user and persist to config.json
    Returns None if the user declines to provide a token.
    """
    token = os.environ.get(_TELEGRAM_TOKEN_ENV, "").strip()
    if token:
        return token
    token = (config.get("telegram", {}) or {}).get("bot_token", "").strip()
    if token:
        return token
    try:
        token = input_fn(
            "Enter your Telegram bot token from @BotFather (or press Enter to skip): "
        ).strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not token:
        return None
    if not constants.CONFIG_FILE.exists():
        constants.CONFIG_FILE.write_text("{}", encoding="utf-8")
    try:
        user_config = json.loads(constants.CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        user_config = {}
    user_config.setdefault("telegram", {})["bot_token"] = token
    constants.CONFIG_FILE.write_text(json.dumps(user_config, indent=2) + "\n", encoding="utf-8")
    print("[Telegram] Token saved to config.json. Consider using SYMBIO_TELEGRAM_TOKEN instead.")
    return token
