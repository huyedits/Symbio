"""The interactive chat REPL: slash commands, the autonomous agent loop,
and the growth loop (memory nudges, exit flush, cron surfacing)."""

import gc
import hashlib
import json
import logging
import os
import re
import shlex
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.generate import generate, generate_step, stream_generate
from mlx_lm.models.cache import (
    can_trim_prompt_cache, load_prompt_cache, make_prompt_cache,
    save_prompt_cache, trim_prompt_cache,
)
from mlx_lm.sample_utils import make_logits_processors, make_sampler

from symbio.rag import Retriever
from symbio import constants
from symbio.config import _adapter_matches_model
from symbio.computer import BrowserSession
from symbio import safety
from symbio.tools import tool_few_shots
from symbio.app import cron, dispatch, golden, health, learn, local_telemetry, memory, mcp_bridge, pending, prompts, prune, sandbox, security, sessions, setup, skills, tooling, training, web
from symbio.app.config import apply_gpu_limits, config_show, set_config_value
try:
    # tag_rag lives at the repo root rather than inside the package, so it is
    # only importable when the root is on sys.path — true for `./symb` (which
    # runs `python -m symbio.app.cli` from the checkout) and false for the
    # installed `symbio` console script, whose sys.path[0] is venv/bin. That
    # made an optional feature fatal to startup: tag indexing is off by default
    # and _ensure_tag_index() already refuses to build one without
    # rag.broad_tags, so nothing here needs it to import.
    from tag_rag import TagIndex
except ModuleNotFoundError:  # pragma: no cover - depends on how it was launched
    TagIndex = None

# ---------------------------------------------------------------------------
# chat.py was one 5,885-line module holding every layer of the REPL at once.
# The pieces below now live in their own files; they are imported back into
# this namespace rather than referenced through it, because `chat.X` is the
# name the rest of the project -- and roughly thirty test modules that
# monkeypatch these -- already reach for. Moving code out must not move the
# handles it is held by.
# ---------------------------------------------------------------------------
from symbio.app.chat_constants import (  # noqa: F401  (re-exported; see above)
    THINKING_LEVELS, THINKING_ORDER, _QUIT, _HANDLED, _WEB_TOOLS, _BROWSER_TOOLS,
    _BROWSER_ACTION_TOOLS, _MAX_TOOL_RETRIES, _MAX_RATE_LIMIT_RETRIES,
    _MAX_RATE_LIMIT_WAIT, _TELEGRAM_CONFIRM_TOOLS, _INTERNAL_TO_HERMES_NAME,
    _internal_to_hermes_name, _common_prefix_len, _COMPLETION_CLAIM, _CLAIM_HEDGE,
    _claims_completion
)
from symbio.app.chat_text import (  # noqa: F401  (re-exported; see above)
    _GUI_APP_ALIASES, _GUI_APP_STEMS, _gui_app_from_stem, _gui_app_for,
    _looks_like_shell_command, _VERIFICATION_FOLLOWUPS, _VERIFICATION_TRAILING,
    _looks_like_verification_followup, _EXPLICIT_SEARCH_RE,
    _SEARCH_FILLER_STOPWORDS, _subjectless_search_command, _QUERY_STOPWORDS,
    _queries_overlap, _GREETING_WORDS, _GREETING_FILLERS, _is_greeting,
    _is_action_request, _ACTION_VERBS, _PATHY_RE, _IMPERATIVE_DO_RE,
    _asks_for_action, _repair_project_path_command, _annotate_sandbox_cwd,
    _project_paths_in, _is_substantive, _is_navigation_only, _last_exchange,
    _AFFECT_FRUSTRATION, _AFFECT_IMPATIENCE, _AFFECT_CONFUSED, _AFFECT_GRATEFUL,
    _AFFECT_HAPPY, _AFFECT_CURIOUS, _AFFECT_EXASPERATION,
    _AFFECT_EXASPERATION_NORM, _CMD_START_RE, infer_user_affect, _MOOD_TAG_RE,
    _VALID_MOODS
)
from symbio.app.chat_ui import (  # noqa: F401  (re-exported; see above)
    _persist_health_report, _make_chat_logger, _RAINBOW_COLORS, rainbow, _Spinner,
    _adapter_trained_at, _adapter_iters, _fmt_ago, learn_progress_line,
    adapter_status_value, print_banner
)
from symbio.app.chat_commands import CommandsMixin
from symbio.app.chat_tools import ToolsMixin
from symbio.app.chat_turn import AgentTurnMixin


def _browser_peek(browser: BrowserSession, config: dict | None = None) -> str:
    """Best-effort snapshot of the live page after a browser action, so the
    model sees what its click/type/scroll did without asking."""
    try:
        text = browser.get_text()
    except Exception:
        return ""
    if text.startswith("Browser "):  # error string from get_text itself
        return ""
    limit = int(config.get("agent", {}).get("browser_peek_chars", 1500)) if config else 1500
    return "\n\nPage text now:\n" + text[:limit]


class ChatSession(AgentTurnMixin, ToolsMixin, CommandsMixin):
    """One interactive chat session: model, stores, browser, cron thread.

    Non-terminal front-ends can supply:
      - model/tokenizer/adapter_loaded to reuse a loaded model
      - input_fn(prompt) -> str  to replace builtins.input
      - output_fn(text)            to replace print for user-facing output
      - confirm_fn(prompt) -> bool for yes/no gates (blocked commands, domains)
    """

    def __init__(self, config: dict[str, Any], model=None, tokenizer=None,
                 adapter_loaded: bool | None = None,
                 input_fn=None, output_fn=None, confirm_fn=None,
                 generate_fn=None, stream_fn=None, stream_chunk_fn=None,
                 stream_prefix: bool = True, owner: str | None = None):
        # Last URL successfully opened in the controllable browser; used to
        # auto-recover when a later click/type/scroll/press finds the browser
        # session was reset or never opened.
        self._last_browsed_url: str = ""
        self.config = config
        self.owner = owner
        self.input_fn = input_fn if input_fn is not None else input
        self.output_fn = output_fn if output_fn is not None else print
        self.confirm_fn = confirm_fn
        self.generate_fn = generate_fn if generate_fn is not None else generate
        self.stream_fn = stream_fn if stream_fn is not None else stream_generate
        # Called with each safe chunk of text as a reply streams in (e.g.
        # incremental terminal printing or a throttled Telegram message
        # edit). None means no live output — replies are shown once
        # complete, same as before streaming existed.
        self.stream_chunk_fn = stream_chunk_fn
        # Whether to prepend the assistant-name prefix to streamed chunks.
        # Terminal front-ends want it for alignment; chat front-ends like
        # Telegram supply their own sender context, so the prefix is noise.
        self.stream_prefix = stream_prefix
        # KV-cache reuse across generate calls (see _generate_reply);
        # invalidated whenever the prompt's actual prefix changes out from
        # under it (adapter reload, a generation that errored mid-stream).
        self._prompt_cache: list | None = None
        self._cached_prompt_ids: list[int] | None = None
        # Speculative decoding's draft model, loaded on first use. A small
        # model proposes several tokens, the real one verifies them in a single
        # pass; on a memory-bound Mac that is where the speedup comes from.
        # None means it is off, either unconfigured or it failed to load.
        self._draft_model = None
        self._draft_tried = False
        # The system-prompt prefill runs on a daemon thread at boot so the user
        # can read the banner and type their first message while the ~4000-token
        # system+tools prefix is processed into the KV cache, instead of paying
        # that cost as a blocking "warming prompt cache…" step. _await_prefill()
        # joins it before any generation (so the model is never used by two
        # threads at once).
        self._prefill_thread: threading.Thread | None = None
        # A persisted prompt cache is over a gigabyte of safetensors, and
        # reading it used to start only once load() had finished — so the two
        # slowest parts of boot ran back to back when they have nothing to say
        # to each other. This thread pulls the file's bytes into the OS page
        # cache underneath the weight load, so the real read on the prefill
        # thread finds them in memory. See _start_prompt_cache_prefetch.
        self._prefetch_thread: threading.Thread | None = None
        self.enabled_groups: set[str] = set(
            config.get("tools", {}).get("enabled_groups", [])
        )
        # Simple timing record for the most recent turn; surfaced in /status
        # and used by front-ends to report latency.
        self.last_turn_timings: dict[str, float | None] = {}
        # Both of these MUTATE the module-level tool registry, so they have to
        # run BEFORE the system prompt is built — the prompt embeds the <tools>
        # catalog, and a prompt built ahead of them is a prompt no turn will
        # ever use again.
        #
        # They used to run ~40 lines below this, after _finish_model_setup had
        # already prefilled and persisted a KV cache keyed on the stale prefix.
        # Measured on this install: the prefilled prompt was 5,741 tokens, the
        # prompt every turn actually builds was 6,164, and they shared only
        # 5,221 — so 938 tokens were re-prefilled on every single turn for the
        # life of the session, and the 74s boot prefill was partly spent on a
        # prefix that was thrown away before the first reply.
        #
        # Load any custom MCP tools the user has previously built so they are
        # available to the model without restarting the process.
        try:
            tooling.refresh_mcp_tools(self.config)
        except Exception:
            pass
        # Advertise the workers that actually exist. Without this the model is
        # shown a made-up example role and cannot reliably delegate to a skill
        # at all, which is what kept saved skills reachable only through RAG.
        try:
            tooling.refresh_delegate_roles()
        except Exception:
            pass
        self.system_prompt = prompts.build_system_prompt(
            config["assistant_name"], config["user_name"], config
        )
        self._refresh_sampler()

        self.adapter_config = constants.ADAPTER_DIR / "adapter_config.json"
        # adapter_loaded state is "unknown" when no model is supplied. We infer
        # presence cheaply from disk without loading weights, so the banner can
        # still show something useful before the model wakes up.
        self.adapter_loaded = adapter_loaded if adapter_loaded is not None else self.adapter_config.exists()
        # model/tokenizer may be None until _ensure_model_loaded() runs.
        self.model: Any | None = model if model is not None else None
        self.tokenizer: Any | None = tokenizer if tokenizer is not None else None
        self._model_lock = threading.Lock()
        self._model_loaded = model is not None and tokenizer is not None
        self._model_load_error: str | None = None
        # Health report placeholder must exist before any model setup that might
        # read it, and must not be overwritten after _run_post_load_self_check().
        self._health_report: dict[str, Any] = {"healthy": True, "errors": [], "warnings": []}
        # If a caller already handed us a loaded model, do all the post-load
        # setup immediately (same behavior as before lazy loading).
        if self._model_loaded:
            if adapter_loaded is None:
                self.adapter_loaded = self.adapter_config.exists()
            self._finish_model_setup()
            self._run_post_load_self_check()

        self.history: list[dict[str, str]] = []
        self.session_id = f"{datetime.now():%Y-%m-%d_%H-%M-%S-%f}"

        # If a caller already handed us a loaded model, the self-check ran
        # before session_id existed; re-persist now that we have one.
        if self._model_loaded and self._health_report.get("_persisted") is None:
            self._run_post_load_self_check()

        # Skill notes touched this session; used to append health errors and
        # user corrections to the matching sidecar files.
        self._skill_notes_used: set[Path] = set()
        self._skill_health_recorded: set[Path] = set()
        self.session_store = sessions.SessionStore(self.session_id)
        # Past sessions are retrievable; the live one is excluded to avoid echo.
        self.retriever = Retriever(config, session_store=self.session_store,
                                   exclude_session_id=self.session_id,
                                   llm_fn=self._generate_tag_metadata)
        self.tag_index: TagIndex | None = None
        self.browser = BrowserSession(confirm_fn=self.confirm_fn)
        # Worker models are loaded lazily on first delegated task — this
        # just holds the (empty) pool, no extra RAM until dispatch.enabled
        # and something actually delegates. Status messages go through the
        # same output channel as tool observations so you can see workers
        # loading and tasks delegating.
        self.dispatch = dispatch.WorkerPool(
            config,
            status_fn=self.output_fn,
            before_worker_fn=self._sleep_headmaster,
            after_worker_fn=self._wake_headmaster,
        )
        self.logger = _make_chat_logger()
        # Tidy the retrieval stores once the pieces it reports through exist
        # (session_id to skip the live log, retriever to drop its note cache,
        # logger to swallow a failure). Deliberately not in
        # _finish_model_setup: this is pure file maintenance and must still
        # happen when the model is loaded lazily or never.
        if self.config.get("prune", {}).get("on_boot", True):
            self._self_prune()
        # Pick up whatever the last process was in the middle of. Cheap, and
        # runs before the model loads: this is reading a small JSON file and,
        # at most, copying an adapter directory back over a truncated one.
        self._recover_pending()
        self.user_turns = 0
        self.auto_searches = 0
        # Resolved subject for a subjectless "check online"-style command this
        # turn, so the web_search tool can override a hallucinated query and
        # the research note can be filed under the real question. None on a
        # normal turn.
        self._search_subject: str | None = None
        # Human-readable outcome of the last _guarded_train() call, surfaced
        # verbatim as the train_adapter tool's observation.
        self._last_train_note = ""

        # Background scheduler: fires due cron jobs, prints a notice
        # immediately, and queues the event for the model's next turn.
        self.cron_events: list[str] = []
        self.cron_lock = threading.Lock()
        self._last_auto_archive: float = 0.0
        threading.Thread(target=self._cron_worker, daemon=True).start()

        # Background tag-index maintenance. Runs only when enabled and only
        # while the model is idle, so it never races with generation.
        self._index_lock = threading.Lock()
        self._indexing_now = False
        self._index_stop = threading.Event()
        threading.Thread(target=self._background_index_worker, daemon=True).start()

    # ---- Infrastructure ----

    def _refresh_sampler(self, tool_use: bool = False):
        temp = self.config["agent"].get("tool_use_temperature") if tool_use else None
        if temp is None:
            temp = self.config["agent"]["temperature"]
        self.sampler = make_sampler(
            temp=temp,
            top_p=self.config["agent"]["top_p"],
        )
        # Without this the headmaster generates with no repetition penalty at
        # all. agent.py has carried one since the beginning, with the reason
        # written next to it — "near-greedy sampling on a small overfit model
        # degenerates into repetition loops on out-of-distribution input" — but
        # agent.py is not the loop the CLI runs. Observed live 2026-08-24:
        # single-token answers ("wegen", ".", "+") and a worker that replied
        # with "![](https://777.777.777.77777777777..." for 300 characters.
        agent_cfg = self.config.get("agent", {})
        penalty = float(agent_cfg.get("repetition_penalty", 1.15) or 0)
        self.logits_processors = make_logits_processors(
            repetition_penalty=penalty or None,
            repetition_context_size=int(agent_cfg.get("repetition_context_size", 64)),
        ) if penalty else None

    def _cron_worker(self):
        while True:
            time.sleep(int(self.config["agent"]["cron_poll_seconds"]))
            try:
                fired = cron.check_due_jobs(self.config)
            except Exception:
                continue
            if fired:
                with self.cron_lock:
                    self.cron_events.extend(fired)
                for ev in fired:
                    self.output_fn(f"\n  [Cron] {ev.splitlines()[0]}")
            try:
                if self.config.get("archive", {}).get("auto", False):
                    interval = int(self.config["archive"].get("auto_poll_seconds", 3600))
                    now = time.time()
                    if now - self._last_auto_archive >= interval:
                        self._last_auto_archive = now
                        archived = skills.archive_idle_items(self.config)
                        n_notes = len(archived.get("notes", []))
                        n_adapters = len(archived.get("adapters", []))
                        if n_notes or n_adapters:
                            self.output_fn(
                                f"\n  [Archive] Auto-archived {n_notes} idle note(s) and {n_adapters} idle adapter(s)."
                            )
            except Exception:
                pass

    def _drop_prompt_cache(self, reason: str, tell_user: bool = False):
        """Throw away the warmed KV cache, on the record.

        Dropping it costs a full re-prefill — 7,329 tokens and 64 seconds on
        the 14B — and until now every one of the ten call sites did it in
        silence. From the outside that is a turn which sometimes takes a
        minute for no visible reason, and nothing in the logs to say which
        site fired. This module already learned that lesson once, on the
        prefill that was dead in the running app while every switch that
        controlled it read as enabled; a stall nobody can attribute is the
        same failure wearing different clothes.

        Prefer trusting the prefix diff in _generate_reply over calling this.
        It compares tokens exactly and trims the cache to the common prefix,
        so a system prompt that grew a block, or a store that was rewritten,
        costs only the tokens after the change. This is for the cases where
        the cached tensors are genuinely unusable — the weights are gone, the
        cache was mutated mid-token, or it came from another process.
        """
        had_cache = self._prompt_cache is not None
        self._prompt_cache = None
        self._cached_prompt_ids = None
        if not had_cache:
            return
        # The cache is already gone by here. Nothing below may raise, or a
        # logging problem becomes a failed turn — the exact shape of the bug
        # _log_info was written to fix.
        try:
            self._log_info(f"prompt cache dropped: {reason}")
        except Exception:
            pass
        if tell_user:
            try:
                self.output_fn(
                    f"  [Cache] Dropped the warmed prompt cache ({reason}); "
                    f"the next reply re-reads the prompt and will be slow.")
            except Exception:
                pass

    def _unload_model(self):
        """Drop the in-process model and release its GPU buffers.

        The caller is responsible for getting a model back before the next
        generation (_reload_model, _wake_headmaster, or _ensure_model_loaded,
        all of which reload from nothing).
        """
        # Whatever the KV cache holds refers to weights we are about to drop.
        self._drop_prompt_cache("the model weights are being unloaded")
        # Don't leave a reader running into a reload: the prefetch is only
        # ever useful to the load it was started under.
        self._await_prompt_cache_prefetch()
        if getattr(self, "model", None) is not None:
            del self.model
        self.model = None
        self.tokenizer = None
        gc.collect()
        try:
            mx.clear_cache()
        except Exception:
            pass

    def _reload_model(self) -> str | None:
        """Reload model+adapter after training; returns an error message or None."""
        # Free the old weights *before* loading the new ones. Loading first
        # leaves both copies resident at once, which on an 8B model is a
        # multi-gigabyte spike in unified memory for no benefit.
        self._unload_model()
        try:
            self.model, self.tokenizer = load(
                self.config["model_name"], adapter_path=str(constants.ADAPTER_DIR)
            )
            self.adapter_loaded = True
            training.mark_adapter_used()
            # Same cold-cache problem as _wake_headmaster: _unload_model above
            # dropped the KV cache, and without this the first turn after a
            # retrain re-processes the whole prefix. No prefetch here, though —
            # training rewrote the adapter, so the persisted cache's signature
            # cannot match and reading it would be pure waste. This prefill is
            # a real one, which is exactly why it belongs on a background
            # thread rather than in front of the user's next message.
            self._prefill_system_prompt_cache(show_spinner=False)
            return None
        except Exception as e:
            # We already dropped the previous model, so returning here would
            # leave the session with no model at all. Fall back to the base
            # weights and report the adapter failure.
            try:
                self.model, self.tokenizer = load(self.config["model_name"])
                self.adapter_loaded = False
            except Exception as base_exc:
                return f"{e} (base model reload also failed: {base_exc})"
            return str(e)

    def _sleep_headmaster(self):
        """Unload the headmaster model from RAM so a worker can run alone.

        The model is reloaded on the next generation. We only do this when
        dispatch.headmaster_deep_sleep_while_workers is true.
        """
        if not getattr(self, "model", None):
            return
        self._status("  [Dispatch] Headmaster going to sleep (unloading the base model)...")
        self._unload_model()
        self._status("  [Dispatch] Headmaster asleep.")

    def _wake_headmaster(self):
        """Reload the headmaster model after a worker finishes, warm.

        Waking used to hand back a model with a cold KV cache. _unload_model
        drops _prompt_cache along with the weights — it has to, the cached
        values belong to weights that no longer exist — and nothing rebuilt it,
        so the turn after every delegation silently re-processed the whole
        ~4000-token system+tools prefix through the model. The delegation saved
        RAM and spent that saving on latency the user paid at the worst moment:
        immediately after waiting for a worker.

        So the wake mirrors the boot path. The cache file is warmed into the
        page cache before load() and overlaps it, and the prefill runs after
        — which, since the model, adapter and prompt are all unchanged since
        boot, is a signature hit on the persisted file rather than a real
        prefill.
        """
        if getattr(self, "model", None) is not None:
            return
        self._status("  [Dispatch] Headmaster waking up (reloading the base model)...")
        self._start_prompt_cache_prefetch()
        try:
            # The same compatibility check boot makes. This used to ask only
            # whether an adapter *exists*, so waking after a worker dispatch
            # re-applied one trained for a different base — boot had already
            # refused that adapter and said so, and the wake silently put it
            # back. Every generation afterwards died in the matmul:
            #     [matmul] Last dimension of first input with shape
            #     (1,512,5120) must match second to last dimension of second
            #     input with shape (4096,8)
            # 5120 is the served model's width, 4096 the adapter's. A session
            # that boots fine and breaks the first time a worker runs is worse
            # than one that never loads the adapter at all.
            if self.adapter_config.exists() and not _adapter_matches_model(self.config):
                self.model, self.tokenizer = load(self.config["model_name"])
                self.adapter_loaded = False
            elif self.adapter_config.exists():
                self.model, self.tokenizer = load(
                    self.config["model_name"], adapter_path=str(constants.ADAPTER_DIR)
                )
                self.adapter_loaded = True
            else:
                self.model, self.tokenizer = load(self.config["model_name"])
                self.adapter_loaded = False
            training.mark_adapter_used()
            self._prefill_system_prompt_cache(show_spinner=False)
            self._status("  [Dispatch] Headmaster awake.")
        except Exception as e:
            self._status(f"  [Dispatch] Headmaster reload failed: {e}")

    def _status(self, message: str):
        self.output_fn(message)

    def _ensure_model_loaded(self):
        """Load the model on first use. Idempotent and thread-safe.

        When the caller already supplied a model, this returns immediately.
        All model-dependent setup (adapter check, training seed, health checks,
        KV-cache warmup) happens here instead of in __init__ so the CLI can
        show its prompt before any heavy work.
        """
        if self._model_loaded and self.model is not None and self.tokenizer is not None:
            return
        with self._model_lock:
            # Double-checked locking: another thread may have finished while we
            # were acquiring the lock.
            if self._model_loaded and self.model is not None and self.tokenizer is not None:
                return
            if self._model_load_error is not None:
                raise RuntimeError(self._model_load_error)

            # Cap MLX's buffer cache before the first allocation, so a long
            # chat session doesn't sit on GPU memory a training run needs.
            apply_gpu_limits(self.config)

            # Start reading the persisted KV cache off disk now, so those
            # hundreds of megabytes stream in while load() is busy with the
            # weights instead of after it. Pure I/O on a daemon thread; it
            # touches neither the model nor the GPU. Started before the
            # "Waking model..." line so a slow first read still overlaps the
            # whole load, and deliberately inside the try below's scope so a
            # load failure still leaves it harmless (it is only ever consumed
            # by the prefill, which does not run if the load failed).
            self._start_prompt_cache_prefetch()

            # Don't spin during load(): HuggingFace's download progress bar
            # animates the same terminal line via \r, and the two carriages
            # overprint each other into garbage. A static label is enough
            # while the download bar shows progress.
            self.output_fn(" Waking model...")
            try:
                if self.adapter_config.exists() and not _adapter_matches_model(self.config):
                    self.output_fn(
                        " [Warning] Existing adapter was trained for a different model."
                        " Loading base model only."
                    )
                    self.model, self.tokenizer = load(self.config["model_name"])
                    self.adapter_loaded = False
                elif self.adapter_config.exists():
                    self.output_fn(" Loading adapter...")
                    try:
                        self.model, self.tokenizer = load(
                            self.config["model_name"], adapter_path=str(constants.ADAPTER_DIR)
                        )
                        self.adapter_loaded = True
                    except Exception as e:
                        self.output_fn(f" Could not load adapter: {e}")
                        self.output_fn(" Falling back to base model...")
                        self.model, self.tokenizer = load(self.config["model_name"])
                        self.adapter_loaded = False
                else:
                    self.model, self.tokenizer = load(self.config["model_name"])
                    self.adapter_loaded = False

                local_telemetry.log_event(
                    "model", model=self.config["model_name"], adapter=self.adapter_loaded,
                )

                # Warmup (KV-cache prefill + seed notes) has no progress bar of
                # its own, so this is where the spinner actually earns its keep.
                spinner = _Spinner("Waking model...")
                spinner.start()
                try:
                    self._finish_model_setup()
                    self._model_loaded = True
                    self._model_load_error = None
                finally:
                    spinner.stop()
            except Exception as e:
                self._model_load_error = str(e)
                self.output_fn(f" Failed to load model: {e}")
                raise
            self._run_post_load_self_check()

    def _finish_model_setup(self):
        """Run all work that requires a loaded tokenizer/model.

        Called either immediately (when a model is supplied by the caller) or
        from _ensure_model_loaded() the first time the model is needed.
        """
        self._check_idle_adapter()

        # Seed identity notes + clean training corpus on first run.
        memory.ensure_seed_notes(self.config)
        memory.ensure_capability_notes()
        training.seed_training_data(self.tokenizer, self.system_prompt, self.config)
        training.clean_training_duplicates(max_copies=3)

        # Warm the KV cache with the system prompt so the first real turn skips
        # re-processing it. This is guarded so fake-model tests skip it.
        # No inner spinner: when called from _ensure_model_loaded() the outer
        # 'Waking model...' spinner is already active.
        self._prefill_system_prompt_cache(show_spinner=False)

    def _self_prune(self, dry_run: bool = False,
                    announce: bool = True) -> dict[str, Any]:
        """Drop junk out of the stores RAG reads back.

        Notes and past sessions are retrieval sources, so a bad write keeps
        costing the agent long after the turn that made it. Pruning at boot
        means each start is a little cleaner than the last. Never fatal: a
        failure here must not stop the session from coming up. The live
        session is excluded — it is still being appended to.
        """
        cfg = self.config.get("prune", {})
        if not cfg.get("enabled", True):
            return {"notes": [], "sessions": [], "total": 0}
        try:
            report = prune.prune_all(
                self.config, dry_run=dry_run, exclude_session=self.session_id)
        except Exception as e:
            self.logger.warning(f"Self-prune failed: {e}")
            return {"notes": [], "sessions": [], "total": 0}
        if report["total"] and announce:
            self.output_fn(
                f"  [Tidy] Pruned {len(report['notes'])} junk note(s) and "
                f"{report['total'] - len(report['notes'])} duplicate log "
                f"entr(ies) from retrieval.")
        # Retrieval caches the note list; drop it so the pruned notes stop
        # being served from memory for the rest of the session.
        if report["total"] and not dry_run:
            self.retriever.invalidate_cache()
        return report

    def _recover_pending(self):
        """Report work the last process did not finish, and repair its damage.

        The repair is automatic because there is only one right answer to it:
        a run killed partway through leaves an adapter directory the trainer
        was still writing, next to a complete copy of the last good one, and
        loading the truncated version as if it were trained is worse than any
        cost of putting the backup back.

        Re-running the training is not automatic. It is minutes of GPU and a
        second full copy of the weights, and starting one unprompted at boot
        is close to a description of how the machine went down. The list is
        printed; /resume runs it.
        """
        try:
            repairs = pending.recover(restore_fn=training.restore_adapter)
            outstanding = pending.describe_outstanding()
        except Exception as e:
            self.logger.warning(f"Pending-task recovery failed: {e}")
            return
        for line in repairs:
            self.output_fn(f"  [Resume] {line}")
        if outstanding:
            self.output_fn(
                f"  [Resume] {len(outstanding)} unfinished task(s) carried "
                f"over. Run /resume to pick them up, /resume clear to drop them.")
            for line in outstanding:
                self.output_fn(f"    - {line}")

    def _cmd_resume(self, arg: str = ""):
        """List, run, or drop the work carried over from a previous session."""
        arg = (arg or "").strip().lower()
        outstanding = pending.outstanding()
        if arg == "clear":
            dropped = pending.clear()
            self.output_fn(f"  [Resume] Dropped {dropped} carried-over task(s).")
            return
        if not outstanding:
            self.output_fn("  [Resume] Nothing carried over — every task finished.")
            return
        if arg != "run":
            self.output_fn(f"  [Resume] {len(outstanding)} task(s) waiting:")
            for line in pending.describe_outstanding():
                self.output_fn(f"    - {line}")
            self.output_fn("  [Resume] /resume run to start them, "
                           "/resume clear to drop them.")
            return

        # Strictly one at a time, and the headmaster's own run last: each is a
        # trainer holding a full copy of the weights, and overlapping two of
        # them is the failure that made any of this necessary.
        for task in sorted(outstanding, key=lambda t: t.get("kind") == "train_headmaster"):
            kind, role = task.get("kind"), task.get("role")
            self.output_fn(f"  [Resume] {task.get('detail', kind)}...")
            if kind == "train_worker" and role:
                trained, msg = dispatch.guarded_train_worker(role, self.config)
                self.output_fn(f"  [Resume] {msg}")
            elif kind == "train_headmaster":
                self._guarded_train()
            else:
                self.output_fn(
                    f"  [Resume] Nothing knows how to re-run '{kind}'; "
                    f"leaving it on the list.")

    def _run_post_load_self_check(self):
        """AI-driven feature verification after the model has finished loading.

        Runs outside the "Waking model..." spinner so user-facing output is
        not interleaved with the load progress."""
        try:
            self._health_report = health.verify_enabled_features(
                self.config, verbose=True, output_fn=self.output_fn,
                # This runs after the model finished loading, so hand over the
                # live tokenizer rather than making the check load its own.
                tokenizer=self.tokenizer,
            )
        except Exception as e:
            self._health_report = {
                "healthy": False,
                "errors": [{"name": "self_check", "message": f"Self-check crashed: {e}"}],
            }

        # Persist the report so external tools and future sessions can audit it.
        try:
            _persist_health_report(self.session_id, self._health_report)
        except Exception:
            pass

    def _ensure_draft_model(self):
        """The configured draft model, loaded once, or None when there isn't one.

        A draft that fails to load must not take the session with it — the
        model still generates fine without one, just slower.
        """
        if self._draft_tried:
            return self._draft_model
        self._draft_tried = True
        path = str(self.config.get("agent", {}).get("draft_model", "") or "").strip()
        if not path or self.stream_fn is not stream_generate:
            return None
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = constants.PROJECT_DIR / candidate
        try:
            self._draft_model, _ = load(str(candidate))
        except Exception as e:
            self.output_fn(f"  [Draft] speculative decoding off: {e}")
            self._draft_model = None
        return self._draft_model

    def _new_prompt_cache(self):
        """An empty cache in the layout the generation path expects.

        With a draft model that is the two models' caches concatenated, which
        is how mlx-lm's speculative step splits them apart again.
        """
        caches = make_prompt_cache(self.model)
        draft = self._ensure_draft_model()
        if draft is not None:
            caches = caches + make_prompt_cache(draft)
        return caches

    def _kv_quant_kwargs(self) -> dict:
        """KV-cache quantization options, or {} when it is switched off.

        A KV cache is stored at the model's activation precision — BF16 — so
        the persisted system-prompt cache is 1.8 GB for a 6.6k-token prefix on
        the 14B plus its draft. Quantizing it trades a little accuracy per
        cached value for roughly a quarter of the bytes, which on a 16 GB box
        is the difference between the model fitting beside a browser and
        jetsam picking a winner.

        This is a memory knob, not a speed one: quantizing adds arithmetic to
        every prefill step, so a machine with headroom to spare will measure it
        as a small loss. It pays only where the pressure itself is what is
        slowing generation down.

        Off unless agent.kv_bits is set, and only ever passed to the real MLX
        generator — front-ends and tests inject stream_fns that never took
        these arguments.
        """
        if self.stream_fn is not stream_generate:
            return {}
        agent_cfg = self.config.get("agent", {})
        bits = agent_cfg.get("kv_bits")
        if not bits:
            return {}
        return {
            "kv_bits": int(bits),
            "kv_group_size": int(agent_cfg.get("kv_group_size", 64)),
            "quantized_kv_start": int(agent_cfg.get("quantized_kv_start", 0)),
        }

    def _prefill_new_cache(self, ids):
        """A fresh cache with `ids` already processed into it.

        Both models have to be walked over the same tokens: the draft predicts
        from its own cache, so if it were left behind it would be guessing from
        a different context than the target verifies against, and every draft
        would be rejected.
        """
        kv_kw = self._kv_quant_kwargs()
        model_cache = make_prompt_cache(self.model)
        for _ in generate_step(mx.array(ids), self.model, max_tokens=0,
                               sampler=self.sampler, prompt_cache=model_cache,
                               **kv_kw):
            pass
        draft = self._ensure_draft_model()
        if draft is None:
            return model_cache
        draft_cache = make_prompt_cache(draft)
        for _ in generate_step(mx.array(ids), draft, max_tokens=0,
                               sampler=self.sampler, prompt_cache=draft_cache,
                               **kv_kw):
            pass
        return model_cache + draft_cache

    def _prefill_system_prompt_cache(self, show_spinner: bool = True):
        """Process the system prompt through the model once at boot so the
        first user turn skips re-processing it. This is a pure latency win;
        failures are silently ignored and the chat loop falls back to cold
        generation.

        Runs on a daemon thread so the user can start typing their first
        message while the ~4000-token system+tools prefix is processed into
        the KV cache — instead of blocking boot for that long. _await_prefill()
        joins the thread before any generation.

        Only runs with the real MLX model/stream path — tests and front-ends
        that inject fake objects should not trigger a real model call.
        """
        if self.stream_fn is not stream_generate:
            return
        if self.model is None or self.tokenizer is None:
            return
        if not isinstance(self.model, nn.Module):
            return
        agent_cfg = self.config.get("agent", {})
        if not agent_cfg.get("prompt_cache_enabled", True):
            return

        def _prefill():
            # Block the background note-indexer from touching the model while
            # the prefill runs; it sleeps while _indexing_now is True. The main
            # thread is blocked on input() here, and _await_prefill() joins us
            # before it generates, so the model is never used concurrently.
            self._indexing_now = True
            try:
                # The Mistral template only renders the system message inside a
                # following user [INST] block, so apply_chat_template([system])
                # alone emits just BOS and caches nothing useful. Render the
                # system prompt with an empty user turn to get the real
                # system+tools prefix, then prefill those ids. The empty user's
                # closing [/INST] becomes a few stale tokens on the first real
                # turn and is trimmed by the cache-diff logic in _generate_reply.
                templated = self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": self.system_prompt},
                     {"role": "user", "content": ""}],
                    tokenize=False, add_generation_prompt=False, enable_thinking=False,
                )
                system_ids = self.tokenizer.encode(templated)
                if not system_ids:
                    return
                # A cache saved by an earlier run covers this exact prefix and
                # these exact weights — load it and skip the prefill entirely.
                # This used to be switched off whenever a draft model was
                # configured, on the grounds that the file holds one model's
                # layers while the live cache is two models' concatenated. The
                # concatenation round-trips fine (measured: save, load, and
                # speculative generation byte-identical to the live cache); the
                # only real hazard was the split landing in the wrong place, and
                # draft_sig in the signature is what rules that out. With a
                # draft configured — which is the shipped default — the effect
                # was that no cache was ever written and every single boot
                # re-prefilled the whole system prompt through the 14B.
                if (self.config.get("agent", {}).get("persist_prompt_cache", True)
                        and self._load_persisted_prompt_cache(system_ids)):
                    return
                # max_tokens=0 processes the prompt into the KV cache and
                # stops before generating any output tokens.
                self._prompt_cache = self._prefill_new_cache(system_ids)
                self._cached_prompt_ids = list(system_ids)
                # Persist the cache while it holds exactly the system prefix.
                # Saving at exit instead would store the whole conversation,
                # which the next run's prefix diff could not reuse.
                if self.config.get("agent", {}).get("persist_prompt_cache", True):
                    self._save_persisted_prompt_cache(system_ids)
            except Exception as e:
                # Prefill is an optimization, never a hard requirement. Clear any
                # partial state so the next generation rebuilds cleanly.
                #
                # But say why. Swallowing this silently is how the prefill came
                # to be dead in the running app while every switch that controls
                # it read as enabled: no cache file was ever written, every turn
                # reported "cached 0", and nothing anywhere said a word. The
                # save path next to this one already logs its failures; this one
                # not doing so hid a broken feature rather than a slow one.
                self._drop_prompt_cache(f"boot prefill failed: {e!r}")
            finally:
                self._indexing_now = False

        # Run it here, on the calling thread, NOT on a background one.
        #
        # This used to start a daemon thread so the user could type their first
        # message while the ~5k-token prefix warmed. That never once worked.
        # generate_step calls mx.eval on the cache state, MLX's stream registry
        # is thread-local, and evaluating a cache built from main-thread model
        # weights on another thread raises
        #     RuntimeError: There is no Stream(cpu, N) in current thread.
        # after zero yields, on every boot, with this MLX build. The bare
        # except below swallowed it in silence, so the feature reported as
        # enabled while never having run: no cache file was ever written and
        # every turn logged "cached 0".
        #
        # Entering the main thread's stream inside the worker does not fix it —
        # tried, still raises — because the failing stream is a cpu one owned by
        # the arrays, not the device stream the worker enters.
        #
        # So this now blocks. It costs a full prefill on a first boot, once —
        # measured at 65s on the 14B: the whole point of persisting the cache
        # is that every later boot loads the file instead of prefilling, and
        # _start_prompt_cache_prefetch warms that file underneath the model
        # load. Paying it once to make the feature real beats an unblocking
        # optimization that never produced a cache.
        _prefill()
        self._prefill_thread = None

    def _log_info(self, message: str):
        """Log only once the session logger exists.

        Model setup — and the prefill it starts — runs from __init__ before
        self.logger is assigned, so an unguarded log call here raises
        AttributeError. That used to be swallowed by the prefill's own
        catch-all, which then cleared the freshly warmed cache: a logging
        detail silently undoing the whole optimization.
        """
        logger = getattr(self, "logger", None)
        if logger is not None:
            logger.info(message)

    def _prompt_cache_signature(self, system_ids: list[int]) -> dict[str, str]:
        """Identity of the prefix a persisted cache was built from.

        A KV cache is only reusable if both the tokens *and* the weights that
        produced it are unchanged. The token ids cover the system prompt,
        prompt.md edits, the tool catalog and the user's names; the model name
        and adapter fingerprint cover the weights — swapping an adapter leaves
        the ids identical while making every cached value wrong.

        The draft model is in here because a speculative cache is the two
        models' layers concatenated, and the split is recomputed from the
        target at load time. Loading a draft-less file into a draft session
        (or the reverse, or after the draft changed) would put the split in
        the wrong place, which is the failure the persisted cache used to be
        switched off entirely to avoid.
        """
        adapter_sig = "none"
        if self.adapter_loaded:
            weights = constants.ADAPTER_DIR / "adapters.safetensors"
            try:
                st = weights.stat()
                adapter_sig = f"{st.st_mtime_ns}:{st.st_size}"
            except OSError:
                adapter_sig = "missing"
        # The draft half, keyed on what actually LOADED rather than on what
        # config asks for: a draft that failed to load leaves a target-only
        # cache, while a file written by a run where it did load carries extra
        # layers that would push the split past the end. Config alone cannot
        # tell those two apart.
        #
        # Inline, and reached through getattr, on purpose. Several callers bind
        # these cache methods onto a duck-typed stand-in instead of building a
        # ChatSession, so a new helper method here is a method they do not have
        # — a private note to a future edit: keep this function's dependencies
        # to attributes, not to methods.
        draft_sig = getattr(self, "_draft_sig", None)
        if draft_sig is None:
            draft_sig = "none"
            resolve = getattr(self, "_ensure_draft_model", None)
            if resolve is not None:
                try:
                    draft = resolve()
                except Exception:
                    draft = None
                if draft is not None:
                    draft_sig = (
                        f"{self.config.get('agent', {}).get('draft_model', '')}"
                        f":{len(make_prompt_cache(draft))}")
            self._draft_sig = draft_sig
        # A cache written at BF16 and one written at 4 bits hold the same
        # tokens and the same weights but different tensors, and load_prompt_cache
        # will happily hand back the wrong one. Toggling agent.kv_bits has to
        # discard the file, not reuse it.
        # Inline, not via _kv_quant_kwargs, for the reason given above the
        # draft_sig block: callers bind these cache methods onto duck-typed
        # stand-ins, so this function may only reach for attributes.
        _agent = self.config.get("agent", {})
        _bits = _agent.get("kv_bits")
        kv_sig = "none" if not _bits else (
            f"{int(_bits)}:{int(_agent.get('kv_group_size', 64))}"
            f":{int(_agent.get('quantized_kv_start', 0))}")
        ids_bytes = ",".join(map(str, system_ids)).encode()
        return {
            "model_name": str(self.config.get("model_name", "")),
            "adapter_sig": adapter_sig,
            "draft_sig": draft_sig,
            "kv_sig": kv_sig,
            "ids_sha": hashlib.sha256(ids_bytes).hexdigest(),
            "n_tokens": str(len(system_ids)),
        }

    def _start_prompt_cache_prefetch(self):
        """Begin reading the persisted KV cache while the weights are loading.

        The two slowest things at boot are the weight load and the prompt-cache
        read, and they were strictly sequential: the read only began once
        _finish_model_setup started the prefill, which is after load() returns.
        They contend for almost nothing — one is file I/O, the other is mostly
        CPU placing tensors — so running the read underneath the load hides it
        almost entirely, and the first turn is warm the moment the model is.

        What this thread must NOT do is touch MLX at all. It used to call
        load_prompt_cache here, on the theory that the arrays come back lazy
        and so no GPU buffer is allocated until the prefill thread evaluates
        them. Lazy is not the same as thread-free: the arrays still capture the
        stream of the thread that made them, and the mx.eval in
        _load_persisted_prompt_cache then died on the main thread with

            There is no Stream(cpu, 0) in current thread.

        — the same thread-local stream registry that keeps the prefill itself
        on the calling thread (see _prefill_system_prompt_cache). That handler
        deleted the file and re-prefilled, so the persisted cache was rebuilt
        and thrown away on every single boot: measured at 76s to the prompt on
        a warm cache that should have cost about ten.

        So this reads bytes and nothing else. Warming the page cache is the
        part that actually overlaps the load — the array construction was
        never the expensive half — and it cannot produce an object bound to
        the wrong thread because it produces no objects at all.
        """
        if not self.config.get("agent", {}).get("prompt_cache_enabled", True):
            return
        if not self.config.get("agent", {}).get("persist_prompt_cache", True):
            return
        if not self.config.get("agent", {}).get(
                "prefetch_prompt_cache_during_load", True):
            return
        path = constants.PROMPT_CACHE_FILE
        if not path.exists():
            return

        def _prefetch():
            try:
                with open(path, "rb", buffering=0) as f:
                    buf = bytearray(8 << 20)
                    while f.readinto(buf):
                        pass
            except Exception:
                # Nothing is owed here. A failed warm just means the real read
                # on the prefill thread goes to disk, which is what it did
                # before this existed. A bad file is still diagnosed there.
                pass

        self._prefetch_thread = threading.Thread(target=_prefetch, daemon=True)
        self._prefetch_thread.start()

    def _await_prompt_cache_prefetch(self):
        """Block until the page-cache warm has finished.

        Nothing is handed back — the prefetch's only product is bytes in the
        OS page cache. Waiting still matters: letting the real read start while
        the warm is mid-file has the two of them queueing on the same device
        for the same blocks, which is slower than either alone.
        """
        thread = self._prefetch_thread
        if thread is not None:
            thread.join()
            self._prefetch_thread = None

    def _load_persisted_prompt_cache(self, system_ids: list[int]) -> bool:
        """Restore the warmed system-prefix cache from disk.

        Returns True if the cache was loaded and is safe to use. Reading a
        gigabyte off an SSD is roughly an order of magnitude cheaper than
        re-running the prefill through the model, which is the whole point —
        and cheaper still when _start_prompt_cache_prefetch has already pulled
        those bytes into the page cache underneath the weight load.

        The read happens HERE, on the calling thread, and not on the prefetch
        thread: MLX arrays belong to the stream of the thread that built them,
        so a cache read anywhere else cannot be evaluated by the prefill. See
        _start_prompt_cache_prefetch for what that cost.
        """
        path = constants.PROMPT_CACHE_FILE
        if not path.exists():
            return False
        want = self._prompt_cache_signature(system_ids)
        self._await_prompt_cache_prefetch()
        try:
            cache, meta = load_prompt_cache(str(path), return_metadata=True)
        except Exception as e:
            # A truncated or version-mismatched file is not worth keeping.
            self._log_info(f"Prompt cache unreadable, discarding: {e}")
            path.unlink(missing_ok=True)
            return False
        differing = [k for k, v in want.items() if meta.get(k) != v]
        if differing:
            # The model, adapter, draft or prompt changed since it was written.
            # Say WHICH. This branch throws away a 1.7 GB file and buys the next
            # boot a 74-second prefill, and it used to do that without a word —
            # so a cache that never once hit was indistinguishable from a cache
            # that was working fine. `ids_sha` here means the system prompt
            # moved, which is the one worth naming, because it is usually a
            # bug in what built the prompt rather than a real change.
            self._log_info(
                "Prompt cache signature mismatch, discarding: "
                + ", ".join(f"{k} {meta.get(k)!r} != {want[k]!r}"
                            for k in differing))
            path.unlink(missing_ok=True)
            return False
        # Loading a cache is not the same as being able to use one: nothing
        # would notice a cache that cannot be materialized until generation,
        # which would then die mid-turn. Touching the state now moves that
        # failure to the one place equipped to handle it — prefill just runs
        # normally instead.
        #
        # The file is NOT deleted on this path. Whatever went wrong here is a
        # property of this process, not of the bytes on disk; deleting them
        # buys the next boot a full prefill for a condition it may not even
        # share. Only an unreadable file or a stale signature is worth a
        # discard, and both are handled above.
        try:
            mx.eval([c.state for c in cache])
        except Exception as e:
            self._log_info(f"Prompt cache unusable in this process, keeping "
                           f"the file and prefilling instead: {e}")
            return False
        self._prompt_cache = cache
        self._cached_prompt_ids = list(system_ids)
        # A hit used to be the only outcome that said nothing, which is how a
        # cache that never once loaded looked exactly like one that always did.
        self._log_info(f"Prompt cache hit: {len(system_ids)} tokens, "
                       f"{path.stat().st_size / 1e6:.0f} MB")
        return True

    def _save_persisted_prompt_cache(self, system_ids: list[int]):
        """Write the freshly warmed cache so the next start can skip prefill."""
        path = constants.PROMPT_CACHE_FILE
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write beside the target and rename, so a crash mid-write can
            # never leave a half-file that the next boot would try to load.
            # The temp name must already end in .safetensors: save_prompt_cache
            # appends that extension itself, so a plain ".tmp" would land on a
            # different path than the one being renamed.
            tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
            save_prompt_cache(str(tmp), self._prompt_cache,
                              self._prompt_cache_signature(system_ids))
            tmp.replace(path)
            self._log_info(
                f"Saved prompt cache: {len(system_ids)} tokens, "
                f"{path.stat().st_size / 1e6:.0f} MB")
        except Exception as e:
            # Purely an optimization — a failed write must not affect the run.
            self._log_info(f"Could not save prompt cache: {e}")
            path.with_name(f"{path.stem}.tmp{path.suffix}").unlink(missing_ok=True)

    def _await_prefill(self):
        """Block until the background system-prompt prefill (if any) has
        finished, so the model is never used by two threads at once. After this
        returns, _prompt_cache/_cached_prompt_ids are either populated (prefill
        succeeded) or None (prefill failed or never started), and the cache path
        falls back to cold generation in the latter case."""
        thread = self._prefill_thread
        if thread is None:
            return
        thread.join()
        self._prefill_thread = None

    def thinking_setting(self) -> tuple[bool, int]:
        """Resolve agent.thinking_level to (enable_thinking, reasoning budget).

        An unknown level falls back to "none" rather than raising: this is read
        on every generation, and a typo in config.json should cost the feature,
        not the session.
        """
        level = str(self.config.get("agent", {}).get("thinking_level", "none")).lower()
        return THINKING_LEVELS.get(level, THINKING_LEVELS["none"])

    def _skill_example_generator(self):
        """A teacher callable for skills._seed_worked_examples, or None.

        The load is deferred into the callable on purpose. /new-skill is a slash
        command, so on the lazy CLI path no model is resident when the user types
        it -- checking here returned None every time and the worker silently got
        recall-only seeds. The callable runs on the training thread instead,
        where blocking to load the headmaster is free, and where
        unload_model_during_training frees it again before the worker trains.
        """
        def _teach(prompt: str, max_tokens: int = 700) -> str:
            from mlx_lm import generate
            from symbio.app import eval as eval_mod, tooling

            self._ensure_model_loaded()
            if self.model is None or self.tokenizer is None:
                raise RuntimeError("no headmaster resident to write examples")

            rendered = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            # Greedy: seed data has no business being a dice roll, and a
            # temperature sample here is one bad draw away from teaching the
            # worker something malformed it will then repeat forever.
            raw = generate(
                self.model, self.tokenizer, prompt=rendered,
                sampler=eval_mod._make_sampler(
                    {**self.config,
                     "agent": {**self.config.get("agent", {}), "temperature": 0.0}}),
                max_tokens=max_tokens, verbose=False)
            return tooling.strip_tool_tags(
                tooling.strip_reasoning_block(raw)).strip()

        return _teach

    def _generate_reply(
        self,
        messages: list[dict[str, str]],
        chunk_prefix: str = "",
        timings: dict[str, float | None] | None = None,
        think: bool = True,
        reasoning_budget: int = 0,
    ) -> tuple[str, bool]:
        """Generate the next reply for `messages`.

        When agent.prompt_cache_enabled, reuses the model's KV cache across
        calls: only the token-level suffix that's new since the last call
        (an exact longest-common-prefix diff, not a string heuristic) is
        actually prefilled — the system prompt and unchanged history are
        served from cache instead of reprocessed every round. This is what
        makes multi-round tool loops (e.g. a browser click sequence) and
        ordinary turn-to-turn chat fast; see _common_prefix_len.

        The system prompt is also pre-encoded once per change so we don't
        re-tokenize it on every turn.

        When self.stream_chunk_fn is set (and agent.stream_output), also
        streams tag-stripped text to it live via tooling.StreamingStripper,
        prefixed with `chunk_prefix` on the first visible chunk.

        Returns (raw_reply, streamed_live) — streamed_live is True iff
        something was actually shown via stream_chunk_fn this call, so the
        caller knows whether the final consolidated print is still needed.
        """
        agent_cfg = self.config["agent"]
        self._ensure_model_loaded()
        # The system-prompt prefill may still be running on its background
        # thread (started at boot). Join it before touching the model so the
        # prefill thread and this generation never use the model concurrently.
        self._await_prefill()
        # Start the spinner before tokenizing the prompt: apply_chat_template
        # plus the encode passes (the full prompt for token counting, then the
        # conversation tail) and the cache-prefix diff take a visible beat on
        # longer sessions and otherwise read as a dead gap after the user hits
        # enter. This early spinner hands off to the generation spinner below.
        tokenizing_spinner = _Spinner("thinking…")
        tokenizing_spinner.start()
        try:
            # enable_thinking must match how the adapter was TRAINED.
            # training.build_chat_training_sample renders every sample with
            # enable_thinking=False, so the corpus contains only empty think
            # blocks — the model is fine-tuned to skip reasoning and answer
            # directly. Inviting real reasoning here creates a train/serve
            # mismatch whose failure mode is reasoning text surfacing as the
            # reply ("The assistant already greeted the user." in place of a
            # greeting). The golden set and eval both grade with False too,
            # so a mismatch here is invisible to the regression net.
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=think,
            )
            # Encode the FULL templated prompt (system + tools + conversation).
            # The Mistral chat template only renders the system message inside a
            # following user [INST] block, so apply_chat_template([system]) on
            # its own emits just BOS — and the old "encode system separately,
            # encode the rest separately, then concatenate" splice silently
            # dropped the entire system prompt + tool catalog (and produced a
            # double-BOS). The model then had no idea what tools it has.
            # Templating the whole message list renders everything correctly.
            ids = self.tokenizer.encode(prompt_text)
            prompt_tokens = len(ids)
            if timings is not None:
                timings["prompt_tokens"] = prompt_tokens
                timings["prompt_chars"] = len(prompt_text)
            # Reasoning gets its own allowance on top of the reply budget.
            # Sharing one number means a turn that thinks is a turn that stops
            # mid-answer, which reads as the model breaking rather than as the
            # dial being turned up.
            max_tokens = int(agent_cfg["max_reply_tokens"]) + max(0, int(reasoning_budget))

            if not agent_cfg.get("prompt_cache_enabled", True):
                # Caching off: the exact original call, unchanged.
                gen_start = time.perf_counter()
                self._indexing_now = True
                try:
                    text = self.generate_fn(
                        self.model, self.tokenizer, prompt=prompt_text, sampler=self.sampler,
                        max_tokens=max_tokens, verbose=False,
                    )
                    # Cut at the end-of-turn marker (mirrors the streaming stop).
                    # Only cut if the think block is closed — <end> inside an
                    # unclosed think block is mid-reasoning, not the real end.
                    m = tooling.END_TURN_RE.search(text)
                    if m:
                        think_open = tooling._QWEN_THINK_OPEN
                        think_close = tooling._QWEN_THINK_CLOSE
                        prefix = text[:m.start()]
                        if prefix.count(think_open) <= prefix.count(think_close):
                            text = prefix
                finally:
                    self._indexing_now = False
                if timings is not None:
                    timings["gen_ms"] = (time.perf_counter() - gen_start) * 1000
                    timings["ttft_ms"] = timings["gen_ms"]
                return text, False

            # Reuse the KV cache across calls: only the token-level suffix that's
            # new since the last call (an exact longest-common-prefix diff) is
            # prefilled — the system prompt, tools, and unchanged history are
            # served from cache. The system prompt is prefilled at boot (see
            # _prefill_system_prompt_cache) so the first turn feeds only the
            # user message, not the whole system+tools prefix.
            reused = _common_prefix_len(self._cached_prompt_ids, ids)
            if timings is not None:
                timings["cached_tokens"] = reused
                timings["new_tokens"] = len(ids) - reused
            if self._prompt_cache is None or reused == 0:
                self._prompt_cache = self._new_prompt_cache()
                feed = ids
            else:
                stale = len(self._cached_prompt_ids) - reused
                if stale and can_trim_prompt_cache(self._prompt_cache):
                    trim_prompt_cache(self._prompt_cache, stale)
                elif stale:
                    self._prompt_cache = self._new_prompt_cache()
                    reused = 0
                feed = ids[reused:] if reused else ids
            if not feed:
                # The new prompt is exactly what the cache already holds —
                # a resample of an unchanged prompt. Generation still needs
                # at least one input token, so hand back the last one; but
                # evict it from the cache first, otherwise the cache would
                # hold ids[-1] twice while _cached_prompt_ids below records
                # it once, and every later prefix diff would trim against a
                # length that is off by one.
                if can_trim_prompt_cache(self._prompt_cache):
                    trim_prompt_cache(self._prompt_cache, 1)
                    feed = ids[-1:]
                else:
                    self._prompt_cache = self._new_prompt_cache()
                    feed = ids
        finally:
            tokenizing_spinner.stop()

        use_stream = self.stream_chunk_fn is not None and agent_cfg.get("stream_output", True)
        stripper = tooling.StreamingStripper(
            show_reasoning=agent_cfg.get("show_reasoning", True)
        ) if use_stream else None
        shown = False
        # The reply prefix ("Caine: ") attaches to the ANSWER, not to a
        # "[Reasoning] …" block the stripper emits first — so it is deferred
        # until the first non-reasoning chunk.
        answer_prefix_emitted = False
        first_token_time: float | None = None
        gen_start = time.perf_counter()
        prompt_tokens = len(ids)
        cached_tokens = reused
        new_tokens = prompt_tokens - cached_tokens
        spinner_label = (
            f"thinking…  [prompt {prompt_tokens} | cached {cached_tokens} | new {new_tokens}]"
        )
        spinner = _Spinner(spinner_label)
        spinner.start()

        def _emit(text: str, is_reasoning: bool = False):
            if self.stream_chunk_fn is None or not text:
                return
            nonlocal shown, answer_prefix_emitted
            if not shown:
                shown = True
                # Stop the spinner and clear its line before the first visible
                # chunk, otherwise the spinner thread keeps overwriting the
                # streaming reply.
                spinner.stop()
            # Reasoning now streams chunk by chunk, so only its first chunk
            # carries REASONING_MARKER — testing for the marker alone would
            # have read every later reasoning chunk as the start of the answer
            # and stamped "Caine   : " into the middle of the thought. The
            # stripper says which kind of text it just handed over.
            if not answer_prefix_emitted and not is_reasoning and not text.startswith(
                    tooling.REASONING_MARKER):
                answer_prefix_emitted = True
                if chunk_prefix:
                    self.stream_chunk_fn(chunk_prefix)
            self.stream_chunk_fn(text)

        text_parts: list[str] = []
        gen_ids: list[int] = []
        gen_tokens = 0
        raw_acc = ""
        # Mark the model as busy so the background indexer waits.
        self._indexing_now = True
        try:
            # A draft model turns each generation step into "guess several,
            # verify in one pass". Passed only when one actually loaded, so the
            # ordinary path is byte-for-byte what it was.
            _draft = self._ensure_draft_model()
            _spec_kw = {} if _draft is None else {
                "draft_model": _draft,
                "num_draft_tokens": int(
                    self.config.get("agent", {}).get("num_draft_tokens", 1)),
            }
            # Only for the real generator: front-ends and tests inject their own
            # stream_fn, and those fakes take the arguments this loop has always
            # passed. speculative_generate_step accepts it too, so the draft
            # path is covered.
            if self.stream_fn is stream_generate and getattr(
                    self, "logits_processors", None):
                _spec_kw["logits_processors"] = self.logits_processors
            # Same quantization the cache was prefilled under. Passing it here
            # too keeps the live cache and the persisted one the same shape;
            # speculative_generate_step takes these as well, so the draft path
            # is covered.
            _spec_kw.update(self._kv_quant_kwargs())
            for response in self.stream_fn(
                self.model, self.tokenizer, feed, max_tokens=max_tokens,
                sampler=self.sampler, prompt_cache=self._prompt_cache,
                **_spec_kw,
            ):
                if first_token_time is None:
                    # The moment the model produced its first token, which is
                    # what "time to first token" means and what the prefill
                    # work above is spent on. This was declared and never
                    # assigned, so ttft_ms fell through to gen_ms below and
                    # /status reported the whole generation as the latency to
                    # first token — the one number you would use to tell a slow
                    # prefill from a slow decode, reading as neither.
                    first_token_time = time.perf_counter()
                text_parts.append(response.text)
                raw_acc += response.text
                gen_ids.append(response.token)
                gen_tokens += 1
                spinner.set_gen_tokens(gen_tokens)
                if stripper is not None:
                    safe = stripper.feed(response.text)
                    if safe:
                        _emit(safe, stripper.chunk_is_reasoning)
                else:
                    _emit(response.text)
                # Stop the instant the explicit end-of-turn marker streams out,
                # so a model that forgets <|im_end|> can't loop, repeating tool
                # calls, until max_tokens. The marker is stripped from display
                # by StreamingStripper/strip_tool_tags, so it never shows.
                # BUT: only stop if the think block is closed. A model that
                # emits <end> inside an unclosed think block is still mid-
                # reasoning — stopping there leaves a partial JSON fragment
                # that strip_reasoning_block treats as the answer, causing
                # spurious "malformed tool call" errors on every turn.
                if tooling.END_TURN_RE.search(raw_acc):
                    # Count think open/close delimiters in raw_acc. If there
                    # are more opens than closes, the think block is unclosed
                    # and <end> is inside reasoning — ignore it.
                    think_open = tooling._QWEN_THINK_OPEN
                    think_close = tooling._QWEN_THINK_CLOSE
                    opens = raw_acc.count(think_open)
                    closes = raw_acc.count(think_close)
                    m = tooling.END_TURN_RE.search(raw_acc)
                    if opens <= closes and not raw_acc[m.end():].strip():
                        break
        except BaseException:
            # The real MLX cache may already be mutated beyond what our
            # bookkeeping reflects (interrupted mid-token) — never trust a
            # stale cache after this; the next call rebuilds it from zero.
            self._drop_prompt_cache("generation was interrupted mid-token")
            raise
        finally:
            self._indexing_now = False
            spinner.stop()

        if stripper is not None:
            tail = stripper.finish()
            if tail:
                _emit(tail, stripper.chunk_is_reasoning)
            # Only close the line if something was actually written to it.
            # When nothing streamed (the whole reply was a tool tag, or
            # reasoning that stayed hidden) the spinner's stop() already
            # cleared its line, so the cursor is at column 0 and no newline is
            # owed — and emitting the reply prefix there printed an answerless
            # "Caine   : " above every [Tool]/[Blank] notice. The consolidated
            # print below handles that case: streamed_live is False, so the
            # caller prints the real reply with its own prefix.
            if self.stream_chunk_fn is not None and shown:
                self.stream_chunk_fn("\n")

        if timings is not None:
            timings["gen_ms"] = (time.perf_counter() - gen_start) * 1000
            timings["ttft_ms"] = (
                (first_token_time - gen_start) * 1000
                if first_token_time is not None else timings["gen_ms"])

        self._cached_prompt_ids = ids + gen_ids
        return "".join(text_parts), shown

    def _generate_tag_metadata(self, prompt: str) -> str:
        """Generate a tag-indexing response using the already-loaded MLX model.

        Uses the same chat-template path as normal generation but with a
        higher token limit and lower temperature so the output is stable JSON.
        """
        self._ensure_model_loaded()
        # The background note-indexer calls this; the boot prefill sets
        # _indexing_now so the indexer waits, but join the prefill here too in
        # case some other caller reaches this before the prefill is done.
        self._await_prefill()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=training.THINKING_ENABLED
        )
        # Make a fresh sampler for deterministic JSON.
        sampler = make_sampler(temp=0.1, top_p=0.9)
        try:
            text = self.generate_fn(
                self.model, self.tokenizer, prompt=prompt_text, sampler=sampler,
                max_tokens=2048, verbose=False,
            )
            # Strip reasoning artifacts that smaller local models sometimes emit
            # (the Qwen3 thinking block, plus the plain-text variants).
            text = tooling.strip_reasoning_block(text)
            for pattern in (
                r"\bthinking\b.*?/\bthinking\b",
                r"\breasoning\b.*?/\breasoning\b",
            ):
                text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
            return text.strip()
        except Exception as e:
            return ""

    def _ensure_tag_index(self) -> bool:
        """Initialize self.tag_index if needed. Returns True if ready."""
        rag_cfg = self.config.get("rag", {})
        broad_tags = rag_cfg.get("broad_tags", [])
        if not broad_tags:
            return False
        if TagIndex is None:
            self.output_fn(
                "  Tag indexing needs tag_rag.py from the project root, which "
                "this launch cannot import. Run ./symb instead of symbio.")
            return False
        if self.tag_index is None:
            db_path = rag_cfg.get("tag_index_db", "notes/tags.db")
            db_path = Path(db_path)
            if not db_path.is_absolute():
                db_path = constants.PROJECT_DIR / db_path
            self.tag_index = TagIndex(
                notes_dir=constants.NOTES_DIR,
                db_path=db_path,
                broad_tags=broad_tags,
                llm_fn=self._generate_tag_metadata,
            )
        return True

    def _cmd_index_notes(self, force: bool = False) -> None:
        """Index or reindex notes using the in-session loaded model."""
        if not self._ensure_tag_index():
            self.output_fn("  No broad_tags configured. Add them to config.json under rag.broad_tags.")
            return

        self.output_fn("  Indexing notes with the loaded model...")
        stats = self.tag_index.index_all(force=force)
        self.output_fn(
            f"  Done. Indexed: {stats['indexed']}, failed: {stats['failed']}, "
            f"removed stale: {stats['removed']}, skipped: {stats.get('skipped', 0)}"
        )
        if stats.get("errors"):
            self.output_fn("  Failures:")
            for err in stats["errors"]:
                self.output_fn(f"    • {err}")
            self.output_fn(
                "  Tip: if every file failed, the model is not producing valid JSON metadata.\n"
                "       Try a larger model or lower the broad_tag guardrail."
            )
        self.retriever.invalidate_cache()

    def _background_index_worker(self) -> None:
        """Daemon thread that periodically reindexes notes when idle.

        Only runs when:
        - rag.tag_index_enabled is true
        - rag.auto_index_enabled is true
        - the model is not currently generating a reply
        """
        rag_cfg = self.config.get("rag", {})
        if not rag_cfg.get("tag_index_enabled") or not rag_cfg.get("auto_index_enabled"):
            return

        interval = max(30, int(rag_cfg.get("auto_index_interval_seconds", 300)))

        while not self._index_stop.is_set():
            # Sleep in small chunks so shutdown is responsive.
            for _ in range(interval):
                if self._index_stop.is_set():
                    return
                time.sleep(1)

            # Wait until the model is free.
            while self._indexing_now and not self._index_stop.is_set():
                time.sleep(0.5)
            if self._index_stop.is_set():
                return

            if not self._ensure_tag_index():
                continue

            with self._index_lock:
                try:
                    stats = self.tag_index.index_all(force=False)
                    if stats["indexed"] or stats["failed"] or stats.get("removed"):
                        self.output_fn(
                            f"[auto-index] Indexed {stats['indexed']}, failed {stats['failed']}, "
                            f"removed {stats['removed']} (skipped {stats.get('skipped', 0)})"
                        )
                    if stats.get("errors"):
                        for err in stats["errors"]:
                            self.output_fn(f"[auto-index] failure: {err}")
                    if stats["indexed"] or stats.get("removed"):
                        self.retriever.invalidate_cache()
                except Exception as exc:
                    self.output_fn(f"[auto-index] error: {exc}")

    def _check_idle_adapter(self):
        """A saved adapter that exists on disk but wasn't loaded this session
        (e.g. after switching to an incompatible model) sits there unused. If
        it's been idle longer than learn.adapter_idle_days, ask whether to
        remove it. Declining or asking to keep it both just reset the grace
        period so the reminder does not repeat every session — nothing is
        ever deleted unless the user explicitly agrees to remove it."""
        if not self.adapter_config.exists():
            return
        if self.adapter_loaded:
            # Actively in use this session; that alone counts as "used".
            training.mark_adapter_used()
            return

        learn_cfg = self.config.get("learn", {})
        if not learn_cfg.get("adapter_idle_reminder_enabled", True):
            return

        last_used = training.adapter_last_used()
        if last_used is None:
            # First time this adapter's idle state has been tracked.
            training.mark_adapter_used()
            return

        idle_days = (datetime.now() - last_used).days
        threshold = int(learn_cfg.get("adapter_idle_days", 30))
        if idle_days < threshold:
            return

        try:
            answer = self.input_fn(
                f"  A saved LoRA adapter hasn't been used in {idle_days} day(s) "
                f"(not loaded with the current model). Remove it to free up "
                f"space? [y/N]: "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return

        if answer in ("y", "yes", "remove"):
            training.remove_adapter()
            self.output_fn("  Removed the unused adapter.")
        else:
            training.mark_adapter_used()
            self.output_fn("  Keeping the adapter.")

    def _restore_model(self):
        """Reload the model if something unloaded it. No-op when one is loaded."""
        if getattr(self, "model", None) is not None:
            return
        err = self._reload_model()
        if err:
            self.output_fn(f"  [Train] Model reload failed: {err}")

    def _train_unloaded(self, iters: int | None = None) -> bool:
        """Run LoRA training with our own copy of the weights evicted first.

        `mlx_lm lora` runs as a child process and is a second, independent
        Metal client: it loads the same model again and adds optimizer state
        on top. Holding our copy for the duration means two full models plus
        gradients compete for unified memory, which is how these runs end up
        swapping — and memory pressure is the condition under which Apple's
        GPU driver has been observed to fall over.

        On success the model is left unloaded, because every caller reloads
        the freshly trained adapter anyway. On a skipped run, a failure, or
        an exception, the previous model is restored before returning.
        """
        # Both branches end the same way: a trainer child process has exited,
        # and whoever called this reloads the adapter immediately afterwards.
        # That reload is a multi-gigabyte allocation landing in the middle of
        # the child's bulk Metal teardown, which is the sequence that panics
        # the driver — so the wait goes here, once, rather than at each of the
        # several places that reload.
        if not self.config.get("gpu", {}).get("unload_model_during_training", True):
            trained = training.run_training(self.config, iters=iters)
            training.settle_after_trainer_exit(self.config, status_fn=self.output_fn)
            return trained

        self.output_fn("  [Train] Unloading model so the trainer has the GPU to itself...")
        self._unload_model()
        trained = False
        try:
            trained = training.run_training(self.config, iters=iters)
            return trained
        finally:
            training.settle_after_trainer_exit(self.config, status_fn=self.output_fn)
            if not trained:
                self._restore_model()

    def _guarded_train(self, config: dict[str, Any] | None = None, iters: int | None = None) -> bool:
        """Run LoRA training, reload the adapter, then check it against the
        golden set (a fixed battery of prompts covering identity and
        tool-tag formatting — see symbio.app.golden). A regression, a case
        that passed before this training round but fails after, rolls the
        adapter back automatically so a bad fine-tune never silently ships
        as the new default behavior. Mirrors training.run_training's bool
        contract so it's a drop-in replacement everywhere training is
        triggered (slash command, tool call, end-of-session, /learn).

        `config` is accepted (and ignored) so this method can be passed
        directly to learn.maybe_train_on_mistakes, which expects a
        `train_fn(config, iters=...)` signature."""
        learn_cfg = self.config.get("learn", {})
        golden_on = learn_cfg.get("golden_set_enabled", True)

        self.output_fn("  [Train] Running pre-train golden checks...")
        baseline = None
        if golden_on:
            baseline = golden.run_golden_set(
                self.model, self.tokenizer, self.generate_fn, self.sampler,
                self.system_prompt, self.config, self.enabled_groups)
            self.output_fn(
                f"  [Train] Baseline golden checks: "
                f"{baseline.pass_count}/{baseline.total} passing."
            )
            # Cases that were already failing when this round started never
            # enter the regression set below (baseline.passing - after.passing
            # can only contain cases that passed first), so nothing downstream
            # ever teaches them: a contract an earlier fine-tune broke stays
            # broken through every later /train, and the golden report just
            # keeps reprinting it. Give a standing failure the same remedy
            # samples a fresh regression would get, so ordinary training walks
            # it back instead of preserving it.
            standing = sorted(set(baseline.results) - baseline.passing)
            if standing and learn_cfg.get("golden_teach_baseline_failures", True):
                added = golden.append_golden_remedy_samples(
                    standing, self.tokenizer, self.system_prompt, self.config,
                    copies=int(learn_cfg.get("golden_retry_samples_per_case", 3)))
                if added:
                    self.output_fn(
                        f"  [Train] Injected {added} remedy sample(s) for "
                        f"{len(standing)} standing failure(s): {', '.join(standing)}")
        self.output_fn("  [Train] Backing up current adapter before training...")
        backup_dir = training.backup_adapter() if golden_on else None
        # Between here and discard_adapter_backup the previous adapter exists
        # only inside backup_dir, and the adapter directory itself is whatever
        # the trainer has written so far. A crash in that window used to leave
        # both facts on the floor: an orphaned .bak nobody knew was live, and a
        # half-written adapter that loaded as the real one.
        task_id = pending.open_task(
            "train_headmaster", "training for the headmaster adapter",
            backup_dir=str(backup_dir) if backup_dir else None)

        try:
            trained = self._train_unloaded(iters=iters)
            local_telemetry.log_event("train", iters=iters, ok=bool(trained))
            if not trained or not self.adapter_config.exists():
                # Covers the "trained but no adapter on disk" case, which
                # _train_unloaded treats as success and so leaves unloaded.
                self._restore_model()
                self._last_train_note = "Training skipped (no new data or failed)."
                return trained

            self.output_fn("  [Train] Adapter trained. Reloading model...")
            err = self._reload_model()
            if err:
                self.output_fn(f"  [Train] Adapter reload failed: {err}")
                self._last_train_note = f"Training done but reload failed: {err}"
                return True

            if not golden_on or baseline is None:
                self.output_fn("  [Train] Adapter reloaded.")
                self.output_fn(f"  [Train] {adapter_status_value(self.config, True)}")
                self._last_train_note = "Training complete. Adapter reloaded."
                return True

            self.output_fn("  [Train] Running post-train golden checks...")
            after = golden.run_golden_set(
                self.model, self.tokenizer, self.generate_fn, self.sampler,
                self.system_prompt, self.config, self.enabled_groups)
            self.output_fn(
                f"  [Train] Post-train golden checks: "
                f"{after.pass_count}/{after.total} passing."
            )
            regressions = sorted(baseline.passing - after.passing)
            threshold = int(learn_cfg.get("golden_regression_threshold", 0))

            if len(regressions) > threshold and learn_cfg.get("golden_retry_enabled", True):
                self.output_fn(
                    f"  [Golden] Double-checking {len(regressions)} regression(s)...")
                recheck, consistent = golden.run_golden_set_retry(
                    self.model, self.tokenizer, self.generate_fn, self.sampler,
                    self.system_prompt, self.config, self.enabled_groups)
                flaky = sorted(set(regressions) - consistent)
                if flaky:
                    self.output_fn(
                        f"  [Golden] {len(flaky)} regression(s) passed on recheck: "
                        f"{', '.join(flaky)}")
                if not consistent:
                    self.output_fn(
                        "  [Golden] All regressions were flaky; using recheck result.")
                    after = recheck
                    regressions = sorted(baseline.passing - after.passing)
                else:
                    self.output_fn(
                        f"  [Golden] {len(consistent)} case(s) consistently failing: "
                        f"{', '.join(sorted(consistent))}")
                    extra_iters = int(learn_cfg.get("golden_retry_max_extra_iters", 50))
                    copies = int(learn_cfg.get("golden_retry_samples_per_case", 3))
                    added = golden.append_golden_remedy_samples(
                        sorted(consistent), self.tokenizer, self.system_prompt,
                        self.config, copies=copies)
                    if added:
                        self.output_fn(
                            f"  [Train] Injected {added} remedy sample(s) for consistent failures.")
                        self.output_fn(
                            f"  [Train] Running targeted remedy training ({extra_iters} iters)...")
                        trained2 = self._train_unloaded(iters=extra_iters)
                        if trained2:
                            reload_err2 = self._reload_model()
                            if reload_err2:
                                self.output_fn(
                                    f"  [Train] Remedy reload failed: {reload_err2}")
                            else:
                                self.output_fn(
                                    "  [Train] Remedy adapter reloaded. Re-checking golden set...")
                                after = golden.run_golden_set(
                                    self.model, self.tokenizer, self.generate_fn, self.sampler,
                                    self.system_prompt, self.config, self.enabled_groups)
                                self.output_fn(
                                    f"  [Golden] Post-remedy checks: "
                                    f"{after.pass_count}/{after.total} passing.")
                                regressions = sorted(baseline.passing - after.passing)
                                # The same flaky filter the first round got.
                                # Without it this single measurement decides
                                # the rollback, and the recheck a few lines
                                # above has just finished proving that two of
                                # these fifteen cases flip between identical
                                # runs. Observed: a run that went 10/15 ->
                                # 12/15, fixing four checks including a
                                # prompt-injection refusal, was discarded on
                                # two unrechecked regressions — noise deciding
                                # the fate of two and a half hours of GPU.
                                if (len(regressions) > threshold
                                        and learn_cfg.get("golden_retry_enabled", True)):
                                    recheck2, consistent2 = golden.run_golden_set_retry(
                                        self.model, self.tokenizer, self.generate_fn,
                                        self.sampler, self.system_prompt, self.config,
                                        self.enabled_groups)
                                    flaky2 = sorted(set(regressions) - consistent2)
                                    if flaky2:
                                        self.output_fn(
                                            f"  [Golden] {len(flaky2)} post-remedy "
                                            f"regression(s) passed on recheck: "
                                            f"{', '.join(flaky2)}")
                                        after = recheck2
                                        regressions = sorted(
                                            baseline.passing - after.passing)
                    else:
                        self.output_fn("  [Train] No remedy samples could be generated.")

            if len(regressions) > threshold:
                self.output_fn(
                    f"  [Golden] Regression: {len(regressions)} case(s) newly "
                    f"failing ({', '.join(regressions)}).")
                rolled_back = False
                if not learn_cfg.get("golden_rollback_on_regression", True):
                    self.output_fn("  [Golden] Rollback disabled in config; keeping the regressed adapter.")
                elif backup_dir is None:
                    self.output_fn("  [Golden] No prior adapter to roll back to; keeping the regressed adapter.")
                else:
                    training.restore_adapter(backup_dir)
                    reload_err = self._reload_model()
                    if reload_err:
                        self.output_fn(f"  [Golden] Rollback reload failed: {reload_err}")
                    else:
                        self.output_fn("  [Golden] Rolled back to the previous adapter.")
                        rolled_back = True
                self._last_train_note = (
                    f"Training complete but regressed on {len(regressions)} check(s) "
                    f"({', '.join(regressions)}); " + (
                        "rolled back to the previous adapter."
                        if rolled_back else "kept the regressed adapter."
                    )
                )
            else:
                self.output_fn(
                    f"  [Golden] {after.pass_count}/{after.total} checks passing "
                    f"(baseline {baseline.pass_count}/{baseline.total}) — no regression.")
                self._last_train_note = (
                    f"Training complete. Adapter reloaded "
                    f"({after.pass_count}/{after.total} golden checks passing, no regression)."
                )
            self._run_wildcard_check(learn_cfg)
            self.output_fn(f"  [Train] {adapter_status_value(self.config, self.adapter_loaded)}")
            return True
        finally:
            pending.finish(task_id)
            training.discard_adapter_backup(backup_dir)

    def _run_wildcard_check(self, learn_cfg: dict[str, Any]):
        """Score the reloaded adapter on subjects absent from the corpus.

        Runs here because the model is already loaded — the check costs a
        handful of short generations and no extra load. It never rolls back:
        the golden set guards against breaking known behaviour, while this
        only reports whether a rule reached past the samples that taught it.
        Failing wildcards early is expected, so treating them as a gate would
        block nearly every retrain.
        """
        if not learn_cfg.get("wildcard_check_enabled", True):
            return
        try:
            from symbio.app import wildcards

            self.output_fn("  [Wild] Checking held-out cases...")
            result = wildcards.run_check(
                self.model, self.tokenizer, self.generate_fn, self.sampler,
                self.system_prompt, self.config)
            failed = [t["id"] for t in result.tasks if not t["passed"]]
            entry = wildcards.record_run(
                result.pass_count, result.total, failed,
                note=self._last_train_note or "",
                adapter_loaded=self.adapter_loaded)
            self.output_fn(f"  [Wild] {wildcards.format_result(entry)}")
            if entry.get("delta") is not None and entry["delta"] > 0:
                self.output_fn(
                    "  [Wild] Generalising better than the last adapter.")
        except Exception as exc:
            # A measurement must never break the training it is measuring.
            self.output_fn(f"  [Wild] Held-out check skipped: {exc}")

    def _trim_history(self):
        """Keep the most recent messages, but also cap the total token
        budget of the retained window so one giant observation (e.g. a full
        web page dumped by a browser action) cannot bloat every later turn.
        """
        limit = self.config["agent"]["history_limit"]
        while len(self.history) > limit + 8:
            self.history.pop(0)
        # Hard token budget: drop oldest messages until the retained window is
        # under roughly half the model's typical context budget. This is a
        # cheap safety valve; exact token counts are computed later in
        # _generate_reply, but dropping by message count avoids repeatedly
        # tokenizing here.
        max_history_chars = int(self.config["agent"].get("max_history_chars", 12000))
        while len(self.history) > 2:
            window = [
                m.get("content", "") for m in self.history[-limit:]
                if isinstance(m.get("content"), str)
            ]
            if sum(len(c) for c in window) <= max_history_chars:
                break
            self.history.pop(0)
        # Popping one message at a time can leave a tool observation at the
        # front whose assistant turn — the call that produced it — has just
        # been dropped. The window then opens on a result with no request,
        # which is not invalid for the template but is a claim about work the
        # model can no longer see itself having asked for. (This history holds
        # only user/assistant roles; observations are user messages, so the
        # orphaned-tool-role problem that affects agent.py cannot arise here.)
        while (len(self.history) > 1
               and self.history[0].get("role") == "user"
               and str(self.history[0].get("content", "")).startswith(
                   "[System observation:")):
            self.history.pop(0)


    def _nudge_block(self) -> str:
        nudge_every = self.config["memory"]["nudge_interval"]
        if not (self.config["memory"]["enabled"] and nudge_every
                and self.user_turns % nudge_every == 0):
            return ""
        return (
            f"\n\n[Reminder: if this session taught you anything durable about "
            f"{self.config['user_name']} or how to do your job, save it now with "
            f"<memory> or <profile> — only what was actually said, with no "
            f"inferred or invented details. Skip if nothing is worth keeping.]"
        )

    def _record_health_errors_for_skill(self, note_path: Path):
        """If the session health report has errors/warnings, record them once
        into the sidecar of a skill note that is being used this session."""
        if note_path in self._skill_health_recorded:
            return
        issues = (self._health_report.get("errors") or []) + (self._health_report.get("warnings") or [])
        if not issues:
            return
        summary = "\n".join(f"{i['name']}: {i['message']}" for i in issues)
        try:
            skills.record_skill_error(note_path, f"Session health issues at startup:\n{summary}")
            self._skill_health_recorded.add(note_path)
        except Exception:
            pass

    def _learn_from_correction(self, verbose: bool = False):
        """Capture the last (question -> corrected answer) pair as a mistake
        note; at the configured threshold, retrain and reload the adapter.
        Also append the correction to every skill note used this session."""
        sample = learn.find_correction_sample(self.history, self.config)
        if sample is None:
            if verbose:
                self.output_fn("  No recent correction detected. Say something like "
                      "\"No, the answer is ...\" first, then run /learn.")
            return
        severity = learn.correction_severity(sample[0], sample[2], self.config)
        path = learn.save_mistake_note(*sample, severity=severity)
        self.output_fn(f"  [Learn] Correction captured (severity {severity}): {path.name}")

        correction_text = (
            f"Original question: {sample[0]}\n"
            f"Wrong answer: {sample[1]}\n"
            f"Correction: {sample[2]}\n"
            f"Correct answer: {sample[3]}"
        )
        # Only skills the correction is actually about. _skill_notes_used is
        # everything *retrieved* this session, cumulatively — and retrieval is
        # fuzzy on purpose, so that set is far wider than "was involved in the
        # wrong answer". Measured: a correction about which text editor the
        # user prefers was filed against Device Awareness, folding a fitted
        # sheet, and descaling a kettle, because those notes had been
        # retrieved at some point. Those sidecars feed skill retraining, so a
        # Helix correction would have been trained into the fitted-sheet
        # adapter — cross-contamination straight through the per-skill
        # isolation that exists to prevent exactly that.
        for note_path in self._skill_notes_used:
            if not learn.correction_concerns_skill(correction_text, note_path):
                continue
            try:
                skills.record_skill_correction(note_path, correction_text)
            except Exception:
                pass

        learn.maybe_train_on_mistakes(
            self.config, self.tokenizer, self.system_prompt, train_fn=self._guarded_train)

    def _decay_stale_notes(self) -> list[str]:
        """Archive expired 'Learned:' research notes and purge their training
        samples before digesting, so stale web facts are neither retrained
        nor served by RAG."""
        decayed = training.decay_research_notes(self.config)
        if decayed:
            self.retriever.invalidate_cache()
            days = self.config["learn"].get("note_decay_days", 90)
            self.output_fn(
                f"  [Decay] Archived {len(decayed)} research note(s) older than "
                f"{days} days: " + ", ".join(decayed))
        return decayed

    # ---- The autonomous agent loop ----

    def _canary_due(self) -> bool:
        """True when the periodic adherence check should run this turn.

        Asking the model to end every reply with an invisible mark was tried
        first and measured at **0 out of 333 real replies** — a per-reply
        formatting rule loses to everything else in a 2,000-token system prompt,
        so the checker would have warned every session after three turns. The
        model's own `<end>` tag fares better but still only 59%, which is not an
        adherence signal either.

        What the model does do reliably is answer a direct question, which is
        why the on-demand canary works. So this runs that same check on a timer
        instead of waiting for the user to think of it.
        """
        cfg = self.config.get("memory", {})
        if not cfg.get("canary_auto_check_enabled", True):
            return False
        interval = int(cfg.get("canary_check_interval_turns", 25))
        if interval <= 0:
            return False
        # getattr: several call sites build ChatSession without __init__.
        turns = getattr(self, "_turns_since_canary", 0) + 1
        self._turns_since_canary = turns
        if turns < interval:
            return False
        self._turns_since_canary = 0
        return True

    def _run_canary_check(self, canary_phrase: str) -> bool:
        """Ask for the phrase and return True when the model produced it."""
        check_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content":
                f"What is the canary phrase? Reply with only '{canary_phrase}' "
                f"and nothing else."},
        ]
        try:
            check_prompt = self.tokenizer.apply_chat_template(
                check_messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=training.THINKING_ENABLED,
            )
            reply = self.generate_fn(
                self.model, self.tokenizer, prompt=check_prompt,
                sampler=self.sampler, max_tokens=256, verbose=False,
            ).strip()
            reply = tooling.strip_reasoning_block(reply)
        except Exception as exc:
            self.logger.warning("canary check could not run: %s", exc)
            return True      # unknown is not evidence of failure
        return canary_phrase in reply

    def _periodic_canary_check(self) -> None:
        """Warn the user when the system prompt has stopped reaching the model."""
        if not self._canary_due():
            return
        phrase = "SYMBIO_CANARY_v1"
        if self._run_canary_check(phrase):
            self.logger.info("canary_ok")
            return
        self.output_fn(
            "  [Canary] The model no longer repeats the system prompt's hidden "
            "phrase, so the prompt is not reaching it — context has probably "
            "grown too long. Compacting memory; if replies keep drifting, "
            "/quit and start a fresh session.")
        for store in ("memory", "profile"):
            try:
                msg, archived = memory.compact_store(store, self.config)
                if archived is not None:
                    self.output_fn(f"  [Canary] {msg}")
            except Exception as exc:
                self.output_fn(f"  [Canary] Could not compact {store}: {exc}")
        self.retriever.invalidate_cache()
        self._drop_prompt_cache("the periodic canary check failed", tell_user=True)
        try:
            safety.log_security_event("canary_auto_check_failed",
                                      {"history_len": len(self.history)})
        except Exception:
            pass

    def _auto_compact_if_under_pressure(self) -> None:
        """Compact the curated stores when the machine is short on memory.

        The canary already compacts, but only after the model has visibly
        stopped following its system prompt — by then the turn is already bad.
        This fires on the machine's own signal instead, before generation.

        It shrinks the prompt, not the process: the stores are a few KB, so no
        meaningful RAM comes back. The value is a shorter context at the moment
        a long one is most expensive — with a 14B resident on 16 GB, the KV
        cache is what tips into swap.
        """
        cfg = self.config.get("memory", {})
        if not cfg.get("enabled", True) or not cfg.get("auto_compact_enabled", True):
            return
        # Increment first, then test, so "cooldown_turns: 10" means the next
        # compaction becomes possible on the 10th turn after one — reading the
        # counter before bumping it made the real gap 11.
        cooldown = int(cfg.get("auto_compact_cooldown_turns", 10))
        # getattr, not a plain attribute: several call sites build ChatSession
        # without running __init__, so per-session state added here cannot
        # depend on the constructor having run. Defaults to the cooldown so an
        # unconstructed session behaves like a fresh one.
        self._turns_since_auto_compact = getattr(
            self, "_turns_since_auto_compact", cooldown) + 1
        if self._turns_since_auto_compact < cooldown:
            return

        from symbio.app import training as _training
        used = _training.ram_used_fraction()
        threshold = float(cfg.get("auto_compact_ram_fraction", 0.75))
        if used is None or used < threshold:
            return

        self.output_fn(
            f"  [Memory] RAM at {used * 100:.0f}% (threshold "
            f"{threshold * 100:.0f}%) — compacting stores to shorten the prompt.")
        compacted_any = False
        for store in ("memory", "profile"):
            try:
                msg, archived = memory.compact_store(store, self.config)
            except Exception as exc:
                self.output_fn(f"  [Memory] Could not compact {store}: {exc}")
                continue
            if archived is not None:
                compacted_any = True
                self.output_fn(f"  [Memory] {msg}")
        if compacted_any:
            # The stores are part of the cached system prefix; a stale cache
            # would keep serving the pre-compaction text.
            self.retriever.invalidate_cache()
            self._drop_prompt_cache("the curated stores were compacted",
                                    tell_user=True)
        self._turns_since_auto_compact = 0


    # ---- Main loop ----

    def run(self):
        dataset_size = constants.TRAIN_FILE.stat().st_size if constants.TRAIN_FILE.exists() else 0
        print_banner(self.config, self.adapter_loaded, dataset_size, output_fn=self.output_fn)

        while True:
            try:
                user_input = self.input_fn(f"{self.config['user_name']:8}: ").strip()
            except (EOFError, KeyboardInterrupt):
                self.output_fn("")
                user_input = "/quit"

            if user_input.startswith("/"):
                if self._handle_command(user_input) == _QUIT:
                    break
                continue

            if not user_input:
                continue

            self._agent_turn(user_input)

        try:
            self.browser.close()
        except Exception:
            pass

        # ---- End of Session ----
        if self.history:
            try:
                save = self.input_fn("\n Save conversation for training? [y/N]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                save = "n"
            if save in ("y", "yes"):
                saved_count = training.save_history_pairs(
                    self.history, self.tokenizer, self.system_prompt)
                self.output_fn(f"    Appended {saved_count} exchange(s) to {constants.TRAIN_FILE}")

                try:
                    train_now = self.input_fn("  Train now? [y/N]: ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    train_now = "n"
                if train_now in ("y", "yes"):
                    self._guarded_train()


def chat_loop(config: dict[str, Any], model=None, tokenizer=None,
              adapter_loaded: bool | None = None,
              generate_fn=None, stream_fn=None,
              stream_chunk_fn=None,
              input_fn=None, output_fn=None, confirm_fn=None):
    """Run the interactive chat loop.

    The CLI passes no extras and gets a real model load. Tests can inject
    a fake model/tokenizer and generation functions to drive the loop without
    loading weights.
    """
    if stream_chunk_fn is None:
        stream_chunk_fn = lambda s: print(s, end="", flush=True)
    # Never run with a blank identity: a skipped wizard or a reset config.json
    # can leave names empty, which blanks the chat banner and the input prompt.
    # Fill sane defaults in-memory now (cheap); persist only on a real CLI run
    # (below) so test runs with an injected model don't rewrite the user's
    # config.json. This is the backstop for is_first_run no longer re-launching
    # the wizard over empty names.
    _identity_filled = setup.ensure_identity_defaults(config)
    if output_fn is None:
        # The background note-indexer (and other daemon threads) call output_fn
        # while the main thread is blocked in input() with the 'Huy : ' readline
        # prompt drawn. A bare print() clobbers that prompt and readline never
        # redraws it, so the terminal looks frozen after the [auto-index] line.
        # When a background thread prints, escape to a fresh line first and then
        # redraw the prompt + any in-progress input so the session stays
        # responsive. No-op for non-interactive (piped/test) runs.
        try:
            import readline as _readline
        except ImportError:
            _readline = None
        _main_thread = threading.main_thread()
        _user_prompt = f"{config['user_name']:8}: "

        def _cli_output(message=""):
            if threading.current_thread() is _main_thread or not sys.stdin.isatty() or _readline is None:
                print(message)
                return
            sys.stdout.write("\n")
            print(message)
            try:
                sys.stdout.write(_user_prompt + _readline.get_line_buffer())
                _readline.redisplay()
                sys.stdout.flush()
            except Exception:
                print(_user_prompt, end="", flush=True)

        output_fn = _cli_output
    session = ChatSession(
        config,
        model=model, tokenizer=tokenizer, adapter_loaded=adapter_loaded,
        generate_fn=generate_fn, stream_fn=stream_fn,
        stream_chunk_fn=stream_chunk_fn,
        input_fn=input_fn, output_fn=output_fn, confirm_fn=confirm_fn,
        owner="cli",
    )
    # When the CLI itself runs, load and warm the model before showing the
    # banner or interactive prompt. Tests that inject a model or generation
    # functions skip this so they remain lightweight.
    is_real_cli_run = (
        model is None
        and tokenizer is None
        and generate_fn is None
        and stream_fn is None
    )
    if is_real_cli_run:
        # Say so when the policy is not the one the last session ran under.
        # Nothing inside the assistant can write security.md, so a change means
        # a human edited it — worth one line, because it is the file that
        # decides every refusal.
        try:
            changed = security.check_stamp()
            if changed:
                output_fn(f"  [Security] {changed}")
        except OSError:
            pass
        session._ensure_model_loaded()
        # Persist the identity defaults filled above (only on a real CLI run,
        # so tests with injected models don't rewrite the user's config.json).
        if _identity_filled:
            try:
                from symbio.app.config import save_config
                save_config(config)
            except Exception:
                pass
        # Telemetry: count this session and maybe fire one daily ping. Cheap and
        # local-first: maybe_daily_ping is a no-op when telemetry is off or not
        # consented, and only stamps state on a successful send/save.
        try:
            from symbio.app import telemetry
            _tstate = telemetry.load_state()
            telemetry.record_session(_tstate)
            telemetry.save_state(_tstate)
            telemetry.maybe_daily_ping(config, _tstate)
        except Exception:
            pass
    session.run()
