"""The interactive chat REPL: slash commands, the autonomous agent loop,
and the growth loop (memory nudges, exit flush, cron surfacing)."""

import json
import logging
import re
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from mlx_lm import load, generate
from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache
from mlx_lm.sample_utils import make_sampler

from rag import Retriever
from symbio import constants
from symbio.computer import BrowserSession
from symbio.app import cron, dispatch, golden, health, learn, memory, mcp_bridge, prompts, sandbox, sessions, setup, skills, tooling, training, web
from symbio.app.config import config_show, set_config_value


def _looks_like_shell_command(cmd: str) -> bool:
    """Return True if a command uses shell syntax that shlex+no-shell can't handle.

    Pipes, redirects, command separators, subshells, globs, and env var
    assignments all need a real shell interpreter. Simple space-separated
    commands (including URLs) stay in the direct sandbox path.
    """
    shell_tokens = {"|", "&&", "||", ";", "&", "<", ">", "$(", "`", "*", "$", "{", "}"}
    for token in shell_tokens:
        if token in cmd:
            return True
    # Glob characters only count when not inside a URL/query string.
    if "?" in cmd and "?" not in cmd.split()[-1].lstrip("https://").rstrip("/?"):
        return True
    if "*" in cmd and not any(s.endswith(("*", "?")) for s in cmd.split()):
        return True
    # Bare environment variable assignment (e.g. FOO=bar ./x)
    first_word = cmd.split(None, 1)[0] if cmd.strip() else ""
    if "=" in first_word and not first_word.startswith("-"):
        return True
    return False


def _persist_health_report(session_id: str, report: dict[str, Any]):
    """Write the session health report to both a per-session file and a
    rolling 'latest' file inside sessions/."""
    constants.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    session_path = constants.SESSIONS_DIR / f"{session_id}_health.json"
    session_path.write_text(json.dumps(report, indent=2, default=str) + "\n",
                            encoding="utf-8")
    latest_path = constants.SESSIONS_DIR / "latest_health.json"
    latest_path.write_text(json.dumps(report, indent=2, default=str) + "\n",
                           encoding="utf-8")


def _make_chat_logger() -> logging.Logger:
    logger = logging.getLogger("chat")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # One handler per session; drop stale ones so lines don't fan out to
    # every log file ever opened in this process.
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    path = constants.LOG_DIR / f"chat_{datetime.now():%Y-%m-%d_%H-%M-%S}.log"
    fh = logging.FileHandler(path, delay=True)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
    logger.addHandler(fh)
    return logger


class _Spinner:
    """Terminal spinner shown while waiting for visible model output.

    Runs on a daemon thread and anchors itself with carriage returns; stop()
    erases the line so streamed text can take its place. No-op when stdout
    is not a TTY (tests, pipes, or non-terminal front-ends).
    """

    _FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, label: str = "thinking…"):
        self.label = label
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.active = sys.stdout.isatty()
        self._start_time: float | None = None
        self._gen_tokens = 0
        self._lock = threading.Lock()

    def set_gen_tokens(self, n: int):
        with self._lock:
            self._gen_tokens = n

    def start(self):
        if not self.active or self._thread is not None:
            return
        self._stop_event.clear()
        self._start_time = time.perf_counter()

        def _spin():
            i = 0
            while not self._stop_event.wait(0.08):
                elapsed = time.perf_counter() - self._start_time
                frame = self._FRAMES[i % len(self._FRAMES)]
                with self._lock:
                    gen_tokens = self._gen_tokens
                tok_info = f" | generated {gen_tokens} tokens" if gen_tokens else ""
                if elapsed >= 5:
                    label = f"{self.label} ({int(elapsed)}s){tok_info}"
                else:
                    label = f"{self.label}{tok_info}"
                sys.stdout.write(f"\r{frame} {label}")
                sys.stdout.flush()
                i += 1

        self._thread = threading.Thread(target=_spin, daemon=True)
        self._thread.start()

    def stop(self):
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join()
        self._thread = None
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()



def print_banner(config: dict[str, Any], adapter_loaded: bool, dataset_size: int,
                 output_fn=print):
    note_count = len(list(constants.NOTES_DIR.glob("*.md")))
    output_fn("\n" + "=" * 50)
    output_fn(f"  {config['assistant_name'].upper()} — PERSONAL CHAT-FINETUNE CLI")
    output_fn(f"   Model  : {config['model_name']}")
    output_fn(f"   User   : {config['user_name']}")
    output_fn(f"   LoRA   : {'YES' if adapter_loaded else 'None (base)'}")
    output_fn(f"   Data   : {dataset_size:,} bytes")
    output_fn(f"   Notes  : {note_count}")
    output_fn("-" * 50)
    output_fn("Commands: /quit  /save  /train  /retrain  /train_worker  /golden  /learn  /forget_last  /status  /prune  /selfcheck  /setup  /compact  /help")
    output_fn("         /run <cmd>  /note [title]  /notes  /new-skill <name>  /skills  /skill-adapters  /digest  /cron  /config  /archive  /restore")
    output_fn("         /build-mcp <name> | <description>  /mcp-tools  /hosts")
    output_fn("  (Caine can also use <note>, <cmd>, <py>, <digest />, <train />, <cron> by itself)")
    output_fn("-" * 50)


def _browser_peek(browser: BrowserSession) -> str:
    """Best-effort snapshot of the live page after a browser action, so the
    model sees what its click/type/scroll did without asking."""
    try:
        text = browser.get_text()
    except Exception:
        return ""
    if text.startswith("Browser "):  # error string from get_text itself
        return ""
    return "\n\nPage text now:\n" + text[:1500]


_QUIT = "quit"
_HANDLED = "handled"

# Tool names whose observations bring outside information into the turn;
# a turn that used any of these is a research turn worth remembering.
_WEB_TOOLS = {
    "web_search", "read_page",
}

_BROWSER_TOOLS = {
    "browser_open", "browser_click", "browser_type", "browser_scroll", "browser_press",
}

# Tools that require explicit approval when running from a non-terminal
# front-end (e.g. Telegram) because they mutate state or run user-supplied code.
_TELEGRAM_CONFIRM_TOOLS = frozenset({
    "execute_code", "run_command", "edit_file", "write_file", "digest_notes", "train_adapter",
    "schedule_job", "config_set", "delete_cron_job", "update_cron_job",
})

# Map internal tool names back to Hermes-style names for <tool_response> labels.
_INTERNAL_TO_HERMES_NAME: dict[str, str] = {
    "run_command": "terminal",
}


def _internal_to_hermes_name(name: str) -> str:
    return _INTERNAL_TO_HERMES_NAME.get(name, name)


def _common_prefix_len(a: list[int] | None, b: list[int]) -> int:
    """Length of the exact matching prefix of two token-id lists. Token
    level, not string level: chat templates concatenate per-turn, but
    re-encoding a string *substring* independently is not guaranteed to
    match the tokenization of encoding the whole string and slicing (BPE
    merges can cross the cut boundary) — comparing already-encoded ids
    sidesteps that entirely."""
    if not a:
        return 0
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class ChatSession:
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
        self.enabled_groups: set[str] = set(
            config.get("tools", {}).get("enabled_groups", [])
        )
        # Simple timing record for the most recent turn; surfaced in /status
        # and used by front-ends to report latency.
        self.last_turn_timings: dict[str, float | None] = {}
        self.system_prompt = prompts.build_system_prompt(
            config["assistant_name"], config["user_name"]
        )
        self._system_prompt_text: str = self.system_prompt
        self._cached_system_ids: list[int] | None = None
        self._refresh_sampler()

        self.output_fn(" Loading model...")
        self.adapter_config = constants.ADAPTER_DIR / "adapter_config.json"
        self.adapter_loaded = adapter_loaded if adapter_loaded is not None else False
        if model is not None and tokenizer is not None:
            self.model, self.tokenizer = model, tokenizer
            if adapter_loaded is None:
                self.adapter_loaded = self.adapter_config.exists()
        elif self.adapter_config.exists():
            self.output_fn(" Found existing adapter. Loading it...")
            try:
                self.model, self.tokenizer = load(
                    config["model_name"], adapter_path=str(constants.ADAPTER_DIR)
                )
                self.adapter_loaded = True
            except Exception as e:
                self.output_fn(f" Could not load adapter: {e}")
                self.output_fn(" Falling back to base model...")
                self.model, self.tokenizer = load(config["model_name"])
        else:
            self.model, self.tokenizer = load(config["model_name"])

        self._check_idle_adapter()

        # Seed identity notes + clean training corpus on first run.
        memory.ensure_seed_notes(config)
        training.seed_training_data(self.tokenizer, self.system_prompt, config)
        training.clean_training_duplicates(max_copies=3)

        # AI-driven feature verification: run only the checks that match
        # enabled features, auto-fix safe failures, and store the report for
        # later tool access / user surfacing.
        try:
            self._health_report = health.verify_enabled_features(
                self.config, verbose=True, output_fn=self.output_fn
            )
        except Exception as e:
            self._health_report = {
                "healthy": False,
                "errors": [{"name": "self_check", "message": f"Self-check crashed: {e}"}],
            }

        self.history: list[dict[str, str]] = []
        self.session_id = f"{datetime.now():%Y-%m-%d_%H-%M-%S-%f}"

        # Persist the report so external tools and future sessions can audit it.
        try:
            _persist_health_report(self.session_id, self._health_report)
        except Exception:
            pass
        # Load any custom MCP tools the user has previously built so they are
        # available to the model without restarting the process.
        try:
            tooling.refresh_mcp_tools()
        except Exception:
            pass
        # Skill notes touched this session; used to append health errors and
        # user corrections to the matching sidecar files.
        self._skill_notes_used: set[Path] = set()
        self._skill_health_recorded: set[Path] = set()
        self.session_store = sessions.SessionStore(self.session_id)
        # Past sessions are retrievable; the live one is excluded to avoid echo.
        self.retriever = Retriever(config, session_store=self.session_store,
                                   exclude_session_id=self.session_id)
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
        self.user_turns = 0
        self.auto_searches = 0
        # Human-readable outcome of the last _guarded_train() call, surfaced
        # verbatim as the train_adapter tool's observation.
        self._last_train_note = ""

        # Background scheduler: fires due cron jobs, prints a notice
        # immediately, and queues the event for the model's next turn.
        self.cron_events: list[str] = []
        self.cron_lock = threading.Lock()
        self._last_auto_archive: float = 0.0
        threading.Thread(target=self._cron_worker, daemon=True).start()

    # ---- Infrastructure ----

    def _refresh_sampler(self, tool_use: bool = False):
        temp = self.config["agent"].get("tool_use_temperature") if tool_use else None
        if temp is None:
            temp = self.config["agent"]["temperature"]
        self.sampler = make_sampler(
            temp=temp,
            top_p=self.config["agent"]["top_p"],
        )

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

    def _reload_model(self) -> str | None:
        """Reload model+adapter after training; returns an error message or None."""
        # New weights make any existing KV cache meaningless.
        self._prompt_cache = None
        self._cached_prompt_ids = None
        try:
            self.model, self.tokenizer = load(
                self.config["model_name"], adapter_path=str(constants.ADAPTER_DIR)
            )
            self.adapter_loaded = True
            training.mark_adapter_used()
            return None
        except Exception as e:
            return str(e)

    def _sleep_headmaster(self):
        """Unload the headmaster model from RAM so a worker can run alone.

        The model is reloaded on the next generation. We only do this when
        dispatch.headmaster_deep_sleep_while_workers is true.
        """
        if not getattr(self, "model", None):
            return
        self._status("  [Dispatch] Headmaster going to sleep (unloading 8B model)...")
        self._prompt_cache = None
        self._cached_prompt_ids = None
        # Drop the MLX model reference. Garbage collection / metal cache
        # cleanup happens automatically once nothing references the arrays.
        del self.model
        self.model = None
        self.tokenizer = None
        import gc
        gc.collect()
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass
        self._status("  [Dispatch] Headmaster asleep.")

    def _wake_headmaster(self):
        """Reload the headmaster model after a worker finishes."""
        if getattr(self, "model", None) is not None:
            return
        self._status("  [Dispatch] Headmaster waking up (reloading 8B model)...")
        try:
            if self.adapter_config.exists():
                self.model, self.tokenizer = load(
                    self.config["model_name"], adapter_path=str(constants.ADAPTER_DIR)
                )
                self.adapter_loaded = True
            else:
                self.model, self.tokenizer = load(self.config["model_name"])
                self.adapter_loaded = False
            training.mark_adapter_used()
            self._status("  [Dispatch] Headmaster awake.")
        except Exception as e:
            self._status(f"  [Dispatch] Headmaster reload failed: {e}")

    def _status(self, message: str):
        self.output_fn(message)

    def _encode_system_prompt(self) -> list[int]:
        """Encode just the system message once and cache it. The cache is
        invalidated when the system prompt text changes (e.g. identity edits)."""
        if self._cached_system_ids is None or self.system_prompt != self._system_prompt_text:
            self._system_prompt_text = self.system_prompt
            self._cached_system_ids = self.tokenizer.encode(
                self.tokenizer.apply_chat_template(
                    [{"role": "system", "content": self.system_prompt}],
                    tokenize=False, add_generation_prompt=False, enable_thinking=False,
                )
            )
        return self._cached_system_ids

    def _generate_reply(
        self,
        messages: list[dict[str, str]],
        chunk_prefix: str = "",
        timings: dict[str, float | None] | None = None,
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
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        prompt_tokens = len(self.tokenizer.encode(prompt_text))
        if timings is not None:
            timings["prompt_tokens"] = prompt_tokens
            timings["prompt_chars"] = len(prompt_text)
        max_tokens = int(agent_cfg["max_reply_tokens"])

        if not agent_cfg.get("prompt_cache_enabled", True):
            # Caching off: the exact original call, unchanged.
            spinner = _Spinner()
            spinner.start()
            gen_start = time.perf_counter()
            try:
                text = self.generate_fn(
                    self.model, self.tokenizer, prompt=prompt_text, sampler=self.sampler,
                    max_tokens=max_tokens, verbose=False,
                )
            finally:
                spinner.stop()
            if timings is not None:
                timings["gen_ms"] = (time.perf_counter() - gen_start) * 1000
                timings["ttft_ms"] = timings["gen_ms"]
            return text, False

        # Avoid re-encoding the full system prompt every turn: cache its ids
        # and splice them with the encoded rest of the conversation.
        system_ids = self._encode_system_prompt()
        if messages and messages[0].get("role") == "system":
            rest = messages[1:]
        else:
            rest = messages
            system_ids = []
        rest_ids = self.tokenizer.encode(
            self.tokenizer.apply_chat_template(
                rest, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
        ) if rest else []
        ids = system_ids + rest_ids

        reused = _common_prefix_len(self._cached_prompt_ids, ids)
        if timings is not None:
            timings["cached_tokens"] = reused
            timings["new_tokens"] = len(ids) - reused
        if self._prompt_cache is None or reused == 0:
            self._prompt_cache = make_prompt_cache(self.model)
            feed = ids
        else:
            stale = len(self._cached_prompt_ids) - reused
            if stale and can_trim_prompt_cache(self._prompt_cache):
                trim_prompt_cache(self._prompt_cache, stale)
            elif stale:
                self._prompt_cache = make_prompt_cache(self.model)
                reused = 0
            feed = ids[reused:] if reused else ids
        if not feed:
            feed = ids[-1:]

        use_stream = self.stream_chunk_fn is not None and agent_cfg.get("stream_output", True)
        stripper = tooling.StreamingStripper() if use_stream else None
        shown = False
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

        def _emit(text: str):
            if self.stream_chunk_fn is None or not text:
                return
            nonlocal shown
            if not shown:
                shown = True
                if chunk_prefix:
                    self.stream_chunk_fn(chunk_prefix)
            self.stream_chunk_fn(text)

        text_parts: list[str] = []
        gen_ids: list[int] = []
        gen_tokens = 0
        try:
            for response in self.stream_fn(
                self.model, self.tokenizer, feed, max_tokens=max_tokens,
                sampler=self.sampler, prompt_cache=self._prompt_cache,
            ):
                text_parts.append(response.text)
                gen_ids.append(response.token)
                gen_tokens += 1
                spinner.set_gen_tokens(gen_tokens)
                if stripper is not None:
                    safe = stripper.feed(response.text)
                    if safe:
                        _emit(safe)
                else:
                    _emit(response.text)
        except BaseException:
            # The real MLX cache may already be mutated beyond what our
            # bookkeeping reflects (interrupted mid-token) — never trust a
            # stale cache after this; the next call rebuilds it from zero.
            self._prompt_cache = None
            self._cached_prompt_ids = None
            raise
        finally:
            spinner.stop()

        if stripper is not None:
            tail = stripper.finish()
            if tail:
                _emit(tail)
            if self.stream_chunk_fn is not None:
                self.stream_chunk_fn("\n")

        if timings is not None:
            timings["gen_ms"] = (time.perf_counter() - gen_start) * 1000
            if timings.get("ttft_ms") is None:
                timings["ttft_ms"] = timings["gen_ms"]

        self._cached_prompt_ids = ids + gen_ids
        return "".join(text_parts), shown

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
        self.output_fn("  [Train] Backing up current adapter before training...")
        backup_dir = training.backup_adapter() if golden_on else None

        try:
            trained = training.run_training(self.config, iters=iters)
            if not trained or not self.adapter_config.exists():
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
                        trained2 = training.run_training(self.config, iters=extra_iters)
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
            return True
        finally:
            training.discard_adapter_backup(backup_dir)

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

    # ---- Slash commands ----

    def _handle_command(self, user_input: str) -> str:
        """Handle a /command; returns _QUIT or _HANDLED."""
        cmd = user_input.lower()

        if cmd in ("/quit", "/q", "/exit"):
            self._memory_flush()
            self.output_fn(" Exiting chat.")
            return _QUIT

        if cmd == "/forget_last":
            removed = 0
            while self.history and self.history[-1]["role"] == "assistant":
                self.history.pop()
                removed += 1
            while (
                self.history
                and self.history[-1]["role"] == "user"
                and not self.history[-1]["content"].startswith("[System observation:")
            ):
                self.history.pop()
                removed += 1
            self.output_fn("  Forgot last exchange." if removed else " Nothing to forget.")

        elif cmd == "/save":
            if not self.history:
                self.output_fn(" Nothing to save yet.")
            else:
                saved_count = training.save_history_pairs(
                    self.history, self.tokenizer, self.system_prompt)
                self.output_fn(f" Saved {saved_count} exchange(s) to training data.")

        elif cmd == "/train":
            self._guarded_train()

        elif cmd == "/retrain":
            self._cmd_retrain()

        elif cmd.startswith("/train_worker"):
            parts = user_input.split(None, 1)
            role = parts[1].strip() if len(parts) == 2 else ""
            if not role:
                self.output_fn("  Usage: /train_worker <role>  (e.g. /train_worker summarize)")
            else:
                trained, msg = dispatch.guarded_train_worker(role, self.config)
                self.output_fn(f"  [Worker] {msg}")

        elif cmd == "/golden":
            result = golden.run_golden_set(
                self.model, self.tokenizer, self.generate_fn, self.sampler,
                self.system_prompt, self.config, self.enabled_groups)
            self.output_fn(f"  [Golden] {result.pass_count}/{result.total} checks passing:")
            for case in golden.GOLDEN_CASES:
                mark = "PASS" if result.results.get(case.id) else "FAIL"
                self.output_fn(f"    [{mark}] {case.id} — {case.description}")

        elif cmd == "/digest":
            self._decay_stale_notes()
            added = training.digest_notes_to_training(
                self.tokenizer, self.system_prompt, self.config)
            if added:
                self.output_fn(f"  Digested {added} new note samples into training data.")
            else:
                self.output_fn("  No new or changed notes to digest.")

        elif cmd.startswith("/run"):
            self._cmd_run(user_input[4:].strip())

        elif cmd.startswith("/note"):
            self._cmd_note(user_input[5:].strip())

        elif cmd == "/learn":
            self._learn_from_correction(verbose=True)

        elif cmd == "/skills":
            skills = memory.list_skills()
            if not skills:
                self.output_fn("  No skills saved yet.")
            else:
                self.output_fn(f"  {len(skills)} skill(s):")
                for title, path in skills:
                    self.output_fn(f"    - {title}  ({path.name})")

        elif cmd.startswith("/new-skill"):
            rest = user_input[len("/new-skill"):].strip()
            if not rest:
                self.output_fn("  Usage: /new-skill <name> | <steps>")
            else:
                if "|" in rest:
                    name, steps = rest.split("|", 1)
                else:
                    name, steps = rest, ""
                name = name.strip()
                steps = steps.strip()
                if not name:
                    self.output_fn("  Usage: /new-skill <name> | <steps>")
                else:
                    try:
                        result = memory.save_skill(
                            name,
                            steps or "(no steps provided yet)",
                            config=self.config,
                            tokenizer=self.tokenizer,
                            auto_train_adapter=True,
                        )
                        if isinstance(result, dict) and "role" in result:
                            self.output_fn(
                                f"  Created skill note and adapter for '{name}'. "
                                f"Worker role: {result['role']}. Training started in the background."
                            )
                        else:
                            self.output_fn(f"  Created skill note for '{name}'.")
                    except Exception as e:
                        self.output_fn(f"  Failed to create skill adapter: {e}")

        elif cmd == "/skill-adapters":
            adapters = skills.list_skill_adapters()
            if not adapters:
                self.output_fn("  No skill adapters active.")
            else:
                self.output_fn(f"  {len(adapters)} active skill adapter(s):")
                for meta in adapters:
                    self.output_fn(
                        f"    - {meta['name']}  (role={meta['role']}, "
                        f"last_used={meta.get('last_used','never')})"
                    )

        elif cmd.startswith("/build-mcp"):
            rest = user_input[len("/build-mcp"):].strip()
            if "|" in rest:
                name, description = rest.split("|", 1)
            else:
                name, description = rest, ""
            name = name.strip()
            description = description.strip() or name
            if not name:
                self.output_fn("  Usage: /build-mcp <name> | <description>")
            else:
                try:
                    from symbio.app import mcp_tools
                    result = mcp_tools.build_mcp_tool(
                        name,
                        description,
                        model=self.model,
                        tokenizer=self.tokenizer,
                        generate_fn=self.generate_fn,
                        config=self.config,
                    )
                    # Refresh the in-memory tool registry so the new MCP tool
                    # is available immediately in this session.
                    tooling.refresh_mcp_tools()
                    self.output_fn(f"  {result['message']}")
                    self.output_fn(f"  Tool name: {result['tool_name']}")
                    self.output_fn(f"  Smoke test: {'PASS' if result['smoke_ok'] else 'FAIL'}")
                    if result.get("smoke_error"):
                        self.output_fn(f"    Error: {result['smoke_error']}")
                except Exception as e:
                    self.output_fn(f"  Failed to build MCP tool: {e}")

        elif cmd == "/mcp-tools":
            from symbio.app import mcp_tools
            tools = mcp_tools.list_mcp_tools()
            if not tools:
                self.output_fn("  No MCP tools built yet.")
            else:
                self.output_fn(f"  {len(tools)} MCP tool(s):")
                for meta in tools:
                    self.output_fn(f"    - {meta['name']}  ({meta['schema'].get('name')})")

        elif cmd == "/hosts":
            hosts = self.config.get("remote", {}).get("hosts", {})
            if not hosts:
                self.output_fn("  No remote hosts configured.")
                self.output_fn("  Usage: /config set remote.hosts '{\"alias\": {\"hostname\": \"...\", \"user\": \"...\"}}'")
            else:
                self.output_fn(f"  {len(hosts)} remote host(s):")
                for alias, cfg in hosts.items():
                    hostname = cfg.get("hostname", alias)
                    user = cfg.get("user")
                    port = cfg.get("port", 22)
                    display = f"{user}@{hostname}" if user else hostname
                    if port != 22:
                        display += f" (port {port})"
                    self.output_fn(f"    - {alias}: {display}")

        elif cmd == "/archive":
            try:
                archived = skills.archive_idle_items(self.config)
                notes = archived.get("notes", [])
                adapters = archived.get("adapters", [])
                if notes or adapters:
                    self.output_fn(f"  Archived {len(notes)} idle note(s) and {len(adapters)} idle adapter(s).")
                    for n in notes:
                        self.output_fn(f"    note: {Path(n).name}")
                    for a in adapters:
                        self.output_fn(f"    adapter: {Path(a).name}")
                else:
                    self.output_fn("  Nothing idle to archive.")
            except Exception as e:
                self.output_fn(f"  Archival failed: {e}")

        elif cmd.startswith("/restore"):
            rest = user_input[len("/restore"):].strip()
            parts = rest.split(None, 1)
            if len(parts) != 2 or parts[0] not in ("note", "adapter"):
                self.output_fn("  Usage: /restore note <filename>  or  /restore adapter <role>")
            else:
                kind, name = parts
                try:
                    if kind == "note":
                        restored = skills.restore_archived_note(name)
                        if restored:
                            self.retriever.invalidate_cache()
                            self.output_fn(f"  Restored note: {restored.name}")
                        else:
                            self.output_fn(f"  No archived note named '{name}'.")
                    else:
                        restored = skills.restore_archived_adapter(name)
                        if restored:
                            self.output_fn(f"  Restored adapter for role: {name}")
                        else:
                            self.output_fn(f"  No archived adapter for role '{name}'.")
                except Exception as e:
                    self.output_fn(f"  Restore failed: {e}")

        elif cmd == "/notes":
            files = sorted(constants.NOTES_DIR.glob("*.md"))
            if not files:
                self.output_fn("  No notes yet.")
            else:
                self.output_fn(f"  {len(files)} note(s):")
                for f in files:
                    self.output_fn(f"    - {f.name}")

        elif cmd == "/health":
            report = health.system_check(self.config)
            self.output_fn("  [Health check]")
            self.output_fn(json.dumps(report, indent=2, default=str))

        elif cmd == "/selfcheck":
            report = health.verify_enabled_features(self.config, verbose=True, output_fn=self.output_fn)
            self._health_report = report

        elif cmd == "/setup":
            parts = user_input.split(None, 2)[1:]
            if parts and parts[0].lower() == "wizard":
                self.config = setup.run_setup_wizard(
                    self.config, input_fn=self.input_fn, output_fn=self.output_fn
                )
                self.system_prompt = prompts.build_system_prompt(
                    self.config["assistant_name"], self.config["user_name"]
                )
                self._cached_system_ids = None
                self.output_fn("  Setup complete. Some changes may need a restart to take full effect.")
            elif not self.config.get("assistant_name") or not self.config.get("user_name"):
                self.config = setup.run_setup_wizard(
                    self.config, input_fn=self.input_fn, output_fn=self.output_fn
                )
                self.system_prompt = prompts.build_system_prompt(
                    self.config["assistant_name"], self.config["user_name"]
                )
                self._cached_system_ids = None
            else:
                self.output_fn("  Run /setup wizard to re-run the full setup, or use /config to change individual settings.")

        elif cmd == "/compact":
            parts = user_input.split(None, 2)[1:]
            store = parts[0].lower() if parts else "memory"
            if store not in ("memory", "profile"):
                self.output_fn("  Usage: /compact [memory|profile]")
            else:
                def _summarize(text: str) -> str:
                    return str(self.generate_fn(
                        self.model, self.tokenizer, prompt=text, sampler=self.sampler,
                        max_tokens=512, verbose=False,
                    )).strip()
                msg, _ = memory.compact_store(store, self.config, summarize_fn=_summarize)
                self.retriever.invalidate_cache()
                self.output_fn(f"  {msg}")

        elif cmd == "/status":
            files = sorted(constants.NOTES_DIR.glob("*.md"))
            data_size = constants.TRAIN_FILE.stat().st_size if constants.TRAIN_FILE.exists() else 0
            adapter_files = list(constants.ADAPTER_DIR.glob("adapters.*"))
            adapter_kb = sum(
                f.stat().st_size for f in constants.ADAPTER_DIR.iterdir() if f.is_file()) // 1024
            self.output_fn(f"  Model: {self.config['model_name']}")
            self.output_fn(f"  Assistant: {self.config['assistant_name']} | User: {self.config['user_name']}")
            self.output_fn(f"  Notes: {len(files)}")
            self.output_fn(f"  Training data: {data_size:,} bytes")
            self.output_fn(f"  Adapter loaded: {'YES' if self.adapter_loaded else 'NO'}")
            self.output_fn(f"  Adapter files: {len(adapter_files)} ({adapter_kb:,} KB)")
            last_used = training.adapter_last_used()
            if last_used is not None:
                idle_days = (datetime.now() - last_used).days
                self.output_fn(f"  Adapter last used: {idle_days} day(s) ago")
            dispatch_on = self.config.get("dispatch", {}).get("enabled", False)
            loaded_workers = self.dispatch.loaded_roles()
            self.output_fn(
                f"  Dispatch: {'ON' if dispatch_on else 'off'}"
                + (f" — loaded worker(s): {', '.join(loaded_workers)}" if loaded_workers else "")
            )
            timings = getattr(self, "last_turn_timings", {}) or {}
            if timings.get("total_ms"):
                self.output_fn("  Last turn latency:")
                for key in ("rag_ms", "prompt_ms", "ttft_ms", "gen_ms", "tools_ms", "total_ms"):
                    val = timings.get(key)
                    label = key.replace("_ms", "").upper()
                    self.output_fn(
                        f"    {label}: {val:.0f}ms" if val is not None else f"    {label}: —"
                    )
                prompt_tokens = timings.get("prompt_tokens")
                cached = timings.get("cached_tokens")
                new = timings.get("new_tokens")
                if prompt_tokens is not None:
                    self.output_fn(
                        f"    Prompt: {prompt_tokens} tokens "
                        f"(cached {cached or 0}, new {new or 0})"
                    )

        elif cmd.startswith("/config"):
            parts = user_input.split(None, 3)[1:]
            if not parts or parts[0].lower() == "show":
                self.output_fn(config_show(self.config))
            elif parts[0].lower() == "set" and len(parts) == 3:
                msg = set_config_value(self.config, parts[1], parts[2], allow_sandbox=True)
                self.output_fn(f"  {msg}")
                # Re-run feature verification after a config change so the AI
                # immediately notices if the new value broke something.
                if not msg.startswith("Unknown") and not msg.startswith("Bad"):
                    self.output_fn("  [Re-checking enabled features...]")
                    report = health.verify_enabled_features(
                        self.config, verbose=True, output_fn=self.output_fn
                    )
                    self._health_report = report
            else:
                self.output_fn("  Usage: /config [show] | /config set <dotted.key> <value>")

        elif cmd.startswith("/cron"):
            self._cmd_cron(user_input)

        elif cmd == "/prune":
            info = training.prune_adapters()
            if info["removed"]:
                self.output_fn(f"  Removed {len(info['removed'])} stale checkpoint(s):")
                for name in info["removed"]:
                    self.output_fn(f"    - {name}")
            else:
                self.output_fn("  No stale checkpoints to remove.")
            self.output_fn(f"  Current adapter footprint: {info['total_kb']:,} KB")
            self.output_fn("  Note: mlx_lm LoRA adapters do not support true weight pruning; keeping rank low and removing checkpoints is the practical way to stay small.")

        elif cmd in ("/help", "/h", "/?"):
            data_size = constants.TRAIN_FILE.stat().st_size if constants.TRAIN_FILE.exists() else 0
            print_banner(self.config, self.adapter_loaded, data_size, output_fn=self.output_fn)

        else:
            self.output_fn("  Unknown command. Type /help for the command list.")

        return _HANDLED

    def _cmd_retrain(self):
        """Run a full adapter rebuild from scratch inside the chat session."""
        from symbio.app.retrain import retrain_model

        # CLI convenience: require explicit confirmation because this deletes the adapter.
        if self.input_fn(
            "  [Retrain] This will DELETE the current LoRA adapter and retrain from scratch. "
            "Type 'retrain' to continue: "
        ).strip().lower() != "retrain":
            self.output_fn("  [Retrain] Cancelled.")
            return

        self.output_fn("  [Retrain] Rebuilding adapter from scratch...")
        # Sleep the headmaster to free RAM before loading the base model for retraining.
        self._sleep_headmaster()
        try:
            ok = retrain_model(self.config, digest=True, seed=True)
        finally:
            self._wake_headmaster()
        if ok:
            self.adapter_loaded = (constants.ADAPTER_DIR / "adapter_config.json").exists()
            self.output_fn("  [Retrain] Done. Reloaded headmaster.")
        else:
            self.output_fn("  [Retrain] Failed — see output above.")

    def _cmd_run(self, shell_cmd: str):
        if not shell_cmd:
            self.output_fn("  Usage: /run <command>")
            return
        self.output_fn(f"\n  $ {shell_cmd}")
        ok, output = sandbox.run_sandboxed(shell_cmd, self.config, confirm_fn=self.confirm_fn)
        self.output_fn(f"  [{'ok' if ok else 'err'}]")
        for line in output.splitlines():
            self.output_fn(f"  {line}")
        training.append_chat_pair(
            user_msg=f"Run this sandbox command and show the output:\n{shell_cmd}",
            assistant_msg=output,
            tokenizer=self.tokenizer,
            system_prompt=self.system_prompt,
        )
        self.output_fn("  -> Logged to training data.\n")

    def _cmd_note(self, title: str):
        if not title:
            title = self.input_fn("  Note title: ").strip()
        if not title:
            self.output_fn("  Cancelled.")
            return
        body = ""
        self.output_fn("  Content (empty line to finish):")
        try:
            while True:
                line = self.input_fn()
                if line == "":
                    break
                body += line + "\n"
        except (EOFError, KeyboardInterrupt):
            pass
        if not body.strip():
            self.output_fn("  Empty note, cancelled.")
            return
        path = memory.save_note(title, body.strip())
        self.retriever.invalidate_cache()
        self.output_fn(f"  Saved: {path.name}")

    def _cmd_cron(self, user_input: str):
        import shlex
        try:
            parts = shlex.split(user_input)[1:]
        except ValueError as e:
            self.output_fn(f"  Parse error: {e}")
            return
        sub = parts[0].lower() if parts else "list"
        if sub == "list":
            jobs = cron.load_cron_jobs()
            if not jobs:
                self.output_fn("  No scheduled jobs.")
            for j in jobs:
                self.output_fn(f"  [{j['id']}] {j['schedule']} — {j['text']}")
        elif sub == "add" and len(parts) >= 3:
            try:
                job = cron.add_cron_job(
                    parts[1], " ".join(parts[2:]),
                    blocked_commands=set(self.config["sandbox"].get("blocked_commands", [])),
                    owner=self.owner,
                )
                self.output_fn(f"  Added job {job['id']}: {job['schedule']} — {job['text']}")
            except ValueError as e:
                self.output_fn(f"  {e}")
        elif sub in ("update", "edit") and len(parts) >= 4:
            try:
                job = cron.update_cron_job(
                    int(parts[1]), parts[2], " ".join(parts[3:]),
                    blocked_commands=set(self.config["sandbox"].get("blocked_commands", [])),
                    owner=self.owner,
                )
                self.output_fn(f"  Updated job {job['id']}: {job['schedule']} — {job['text']}")
            except ValueError as e:
                self.output_fn(f"  {e}")
        elif sub == "rm" and len(parts) == 2:
            try:
                cron.delete_cron_job(int(parts[1]), owner=self.owner)
                self.output_fn(f"  Removed job {parts[1]}.")
            except ValueError as e:
                self.output_fn(f"  {e}")
        else:
            self.output_fn('  Usage: /cron [list] | /cron add "<cron expr | at YYYY-MM-DD HH:MM>" <text> | /cron update <id> "<schedule>" <text> | /cron rm <id>')

    # ---- Growth loop ----

    def _memory_flush(self):
        """One last turn on /quit to persist memories before context is lost."""
        flush_min = self.config["memory"]["flush_min_turns"]
        if not (self.config["memory"]["enabled"] and flush_min
                and self.user_turns >= flush_min and self.history):
            return
        self.output_fn(" Letting the model save memories before exit...")
        flush_messages = [{"role": "system", "content": (
            self.system_prompt + memory.curated_memory_block(self.config)
            + prompts.env_note() + prompts.time_note()
        )}]
        flush_messages.extend(self.history[-self.config["agent"]["history_limit"]:])
        flush_messages.append({"role": "user", "content": (
            "[Session ending. If this conversation contained anything durable "
            "worth keeping — facts about the user, lessons learned, procedures "
            "that worked — save it now with <memory>, <profile>, or <note>. "
            "Record only what was actually said or observed in this session; "
            "never add inferred, assumed, or invented details. "
            "Reply with just the tags, or 'nothing to save'.]"
        )})
        try:
            flush_prompt = self.tokenizer.apply_chat_template(
                flush_messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
            flush_reply = self.generate_fn(
                self.model, self.tokenizer, prompt=flush_prompt, sampler=self.sampler,
                max_tokens=int(self.config["agent"]["max_reply_tokens"]), verbose=False,
            )
            for name, params in tooling.parse_tools(flush_reply, self.enabled_groups):
                if name == "save_memory":
                    msg = memory.save_memory(params["store"], params["content"], self.config,
                                             replace=params.get("replace", False))
                    self.output_fn(f"  [Memory] {msg}")
                elif name == "write_note":
                    p = memory.save_note(params["title"], params["body"])
                    self.output_fn(f"  [Memory] Saved note: {p.name}")
        except KeyboardInterrupt:
            self.output_fn("\n  [Memory flush interrupted — exiting without saving.]")
        except Exception as e:
            self.output_fn(f"  [Memory flush skipped: {e}]")

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
        for note_path in self._skill_notes_used:
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

    def _agent_turn(self, user_input: str):
        self.logger.info(f"User: {user_input}")
        self.session_store.log("user", user_input)
        turn_start = time.perf_counter()
        timings: dict[str, float | None] = {
            "rag_ms": None,
            "prompt_ms": None,
            "ttft_ms": None,
            "gen_ms": None,
            "tools_ms": None,
            "total_ms": None,
        }

        # Detect corrections against the pre-append history: the last real
        # user turn is still the question the assistant just answered.
        is_correction = learn.looks_like_correction(user_input, self.history, self.config)

        # Surface any cron events that fired since the last turn.
        with self.cron_lock:
            due_events, self.cron_events[:] = list(self.cron_events), []
        if due_events:
            self.history.append({
                "role": "user",
                "content": "[System observation: " + "\n".join(due_events) + "]",
            })

        self.history.append({"role": "user", "content": user_input})

        # Unbounded knowledge: pull relevant saved notes into this turn's
        # context. Retrieval text never enters history or training data.
        rag_context = self.retriever.build_context(user_input)
        if self.retriever.rag_cfg.get("enabled", True):
            for r in self.retriever.retrieve(user_input):
                if r.get("source") == "note" and r.get("path"):
                    note_path = Path(r["path"])
                    try:
                        skills.record_note_usage(note_path)
                    except Exception:
                        pass
                    if skills._is_skill_note(note_path):
                        self._skill_notes_used.add(note_path)
                        self._record_health_errors_for_skill(note_path)
        rag_block = f"\n\n{rag_context}" if rag_context else ""
        timings["rag_ms"] = (time.perf_counter() - turn_start) * 1000

        # Live-reload: config changes and prompt.md edits apply on the next turn.
        self._refresh_sampler()
        self.system_prompt = prompts.build_system_prompt(
            self.config["assistant_name"], self.config["user_name"]
        )
        timings["prompt_ms"] = (time.perf_counter() - turn_start) * 1000

        self.user_turns += 1
        nudge_block = self._nudge_block()

        max_rounds = self.config["agent"]["max_tool_rounds"]
        executed_calls: set[str] = set()
        web_used = False
        auto_searched = False
        self_corrected = False
        final_display = ""
        consecutive_tool_rounds = 0
        # The exact "[System observation: ...]" text of the most recent
        # tool failure this turn, if any — used to capture (saw this error
        # -> did this instead, which worked) as a mistake-note training
        # sample the moment a later tool call actually succeeds. Cleared on
        # any success so only a confirmed fix gets saved, not a mere retry.
        pending_tool_error: str | None = None
        for _ in range(max_rounds):
            # Once we are inside a tool-followup round, lower the temperature
            # so the model sticks to the tag grammar instead of drifting into
            # prose or inventing fake commands.
            if consecutive_tool_rounds:
                self._refresh_sampler(tool_use=True)
            gen_start = time.perf_counter()
            # Keep the system message fixed so the KV cache survives across turns.
            # Per-turn context (RAG, memory, env, time, nudges) is prepended to
            # the latest real user message, so the fixed system prompt stays
            # identical and chat-template role alternation remains strict.
            messages = [{"role": "system", "content": self.system_prompt}]
            context_block = (
                memory.curated_memory_block(self.config) + rag_block
                + prompts.env_note() + prompts.time_note() + nudge_block
            ).lstrip()
            working_history = list(self.history[-self.config["agent"]["history_limit"]:])
            if context_block:
                for i in range(len(working_history) - 1, -1, -1):
                    if (
                        working_history[i]["role"] == "user"
                        and not str(working_history[i]["content"]).startswith("[System observation:")
                    ):
                        working_history[i] = {
                            "role": "user",
                            "content": context_block + "\n\n" + working_history[i]["content"],
                        }
                        break
            messages.extend(working_history)

            chunk_prefix = f"{self.config['assistant_name']:8}: " if self.stream_prefix else ""
            try:
                raw_reply, streamed_live = self._generate_reply(
                    messages, chunk_prefix=chunk_prefix, timings=timings)
                reply = raw_reply.strip()
            except KeyboardInterrupt:
                # Ctrl-C during a slow generation abandons the turn, not the app.
                self.output_fn("\n  [Generation interrupted.]")
                break
            except Exception as e:
                self.output_fn(f"[MLX Error: {e}]")
                break

            tools = tooling.parse_tools(reply, self.enabled_groups)
            display = tooling.strip_tool_tags(reply)

            if display.strip():
                final_display = display
                if not streamed_live:
                    self.output_fn(f"{self.config['assistant_name']:8}: {display}")
                self.logger.info(f"{self.config['assistant_name']}: {display}")
                self.session_store.log("assistant", display)

            # Never re-run a tool call already executed this turn — a model
            # that repeats itself would otherwise loop until max_rounds.
            fresh_tools = [
                (n, p) for n, p in tools
                if json.dumps([n, p], sort_keys=True) not in executed_calls
            ]

            if not fresh_tools:
                self.history.append({"role": "assistant", "content": reply})
                self._trim_history()
                # A tag that looked like a tool call but never resolved
                # (unterminated, or invalid JSON) is a formatting mistake,
                # not a normal reply — surface it as an observation so the
                # model can notice and retry, instead of silently treating
                # the mangled leftovers as the final answer. Once per turn.
                malformed = tooling.detect_malformed_tag(reply)
                if malformed and not self_corrected:
                    self_corrected = True
                    self.output_fn(f"  [Format] {malformed}")
                    self.history.append({"role": "user", "content": (
                        f"[System observation: {malformed} Check your tag "
                        f"syntax (matching open/close tags, valid JSON "
                        f"inside <tool_call>) and try again, or continue "
                        f"without it.]"
                    )})
                    self._trim_history()
                    continue
                # Don't let the model fill knowledge gaps by guessing: an
                # unsure-sounding answer, or a hedged made-up figure for a
                # numeric question, with no tool call triggers one automatic
                # web search so it can answer from results. Moderation: once
                # per turn, never after real web use, never when the user
                # already asked to search, and capped per session so a
                # runaway loop can't hammer the search engine.
                user_asked_web_search = any(
                    marker in user_input.lower() for marker in
                    ("news", "search", "look up", "lookup", "find", "latest", "current",
                     "both sides", "perspective", "balanced", "compare", "conclude")
                )
                # Browser follow-ups (click/scroll/type/press/browse) are never
                # knowledge-gap searches; auto-searching them wastes a turn and
                # creates bogus research notes.
                browser_followup = any(
                    marker in user_input.lower() for marker in
                    ("click", "scroll", "type ", "press ", "browse ", "open ", "go to ")
                ) and not any(
                    marker in user_input.lower() for marker in
                    ("search", "news", "weather", "look up", "find online")
                )
                unsure = bool(display.strip()) and learn.sounds_unsure(display)
                fabricated = (not unsure and bool(display.strip())
                              and learn.sounds_fabricated(user_input, display))
                # A turn that ends with no visible answer at all is the model
                # blanking out entirely — always search then, even when the
                # user's wording asked for one (they asked and got nothing).
                blanked = not final_display.strip()
                # Trivial acknowledgments ("ok", "yes", "go on", "continue") are
                # never a reason to auto-search; just ask the user to clarify.
                trivial_ack = bool(user_input.strip()) and len(user_input.strip().split()) <= 2 and any(
                    marker in user_input.lower() for marker in
                    ("ok", "okay", "yes", "sure", "go on", "go ahead", "continue", "proceed")
                )
                session_cap = int(self.config["web"].get("auto_search_session_cap", 20))
                if (self.config["web"].get("auto_search_when_unsure", True)
                        and not auto_searched and not web_used and not browser_followup
                        and not trivial_ack
                        and (blanked or not user_asked_web_search)
                        and self.auto_searches < session_cap
                        and (unsure or fabricated or blanked)):
                    auto_searched = True
                    web_used = True
                    self.auto_searches += 1
                    reason = ("hedged a made-up-sounding figure" if fabricated
                              else "sounded unsure" if unsure
                              else "came back blank")
                    self.output_fn(f"  [Auto-search] Reply {reason} — searching the web...")
                    ok, out = web.web_search(user_input, self.config)
                    self.history.append({"role": "user", "content": (
                        f"[System observation: Your answer {reason}, so a web "
                        f"search for '{user_input}' ran automatically "
                        f"({'succeeded' if ok else 'failed'}).\nResults:\n{out}\n"
                        f"Answer from these results, citing the exact figure they "
                        f"give. If they don't help, say plainly that you could not "
                        f"find it — do not guess.]"
                    )})
                    self._trim_history()
                    continue
                # Normal turn (or pure repetition): stop.
                break

            # Only execute the first fresh tool per response. Multiple tools in
            # one reply cause bursts (e.g. five <search> tags at once) and can
            # overwhelm the model with parallel observations.
            name, params = fresh_tools[0]
            tool_key = json.dumps([name, params], sort_keys=True)
            executed_calls.add(tool_key)
            extra = fresh_tools[1:]

            # There are tools to execute
            self.history.append({"role": "assistant", "content": reply})
            consecutive_tool_rounds += 1

            self.output_fn(f"  [Tool: {name}]")
            if name in _WEB_TOOLS:
                web_used = True
            observation = self._execute_tool(name, params)
            if extra:
                ignored = ", ".join(n for n, _ in extra)
                observation += (
                    f"\n[Note: {ignored} were also requested in the same reply but "
                    f"ignored — use at most one tool tag per response.]"
                )

            # A tool call that fails and is then followed by one that works
            # is exactly the "made a mistake, then fixed it" pattern already
            # hand-seeded in seed_training_data — capture it automatically
            # from real usage too, via the same mistake-note pipeline that
            # already threshold-batches and golden-checks conversational
            # corrections, so the model learns from its own tool mistakes
            # without needing the user to notice and correct it.
            if pending_tool_error is not None and not learn.sounds_like_tool_error(observation):
                path = learn.save_mistake_note(
                    original_query=pending_tool_error,
                    wrong_answer="(a prior tool call failed; see the observation above)",
                    correction="(automatic: the next tool call succeeded)",
                    correct_answer=reply,
                )
                self.output_fn(f"  [Learn] Tool mistake captured: {path.name}")
                learn.maybe_train_on_mistakes(
                    self.config, self.tokenizer, self.system_prompt, train_fn=self._guarded_train)
            pending_tool_error = (
                f"[System observation: {observation}]" if learn.sounds_like_tool_error(observation)
                else None
            )

            self.output_fn(f"  [Observation] {observation.replace(chr(10), chr(10) + '  ')}")
            timings["tools_ms"] = (time.perf_counter() - gen_start) * 1000
            # Present results in Hermes-style <tool_response> JSON so the model
            # learns the structured format, while keeping a plain-text fallback
            # for models that have not switched to Hermes calls yet.
            hermes_name = _internal_to_hermes_name(name)
            response_json = json.dumps({"name": hermes_name, "content": observation}, ensure_ascii=False)
            self.history.append({"role": "user", "content": (
                f"[System observation: {observation}]\n"
                f"<tool_response>{response_json}</tool_response>"
            )})
            self._trim_history()

        timings["total_ms"] = (time.perf_counter() - turn_start) * 1000
        self.last_turn_timings = timings
        self.logger.info(f"Timings: {timings}")

        if is_correction:
            # The corrected answer is now in history; capture and maybe retrain.
            self._learn_from_correction()
        elif web_used and final_display:
            # Web research produced an answer: remember durable knowledge so
            # it is retrievable later and trained into the weights on digest.
            note = learn.remember_research(user_input, final_display, self.config)
            if note:
                self.retriever.invalidate_cache()
                self.output_fn(f"  [Learn] Remembered research: {note.name}")

    def _resolve_project_path(self, raw_path: str) -> Path | None:
        """Normalize a user-supplied path so it stays inside the project dir."""
        raw_path = raw_path.strip()
        if not raw_path:
            return None
        target = Path(raw_path)
        if not target.is_absolute():
            target = constants.PROJECT_DIR / target
        try:
            target.resolve().relative_to(constants.PROJECT_DIR.resolve())
        except ValueError:
            return None
        return target

    def _make_backup(self, path: Path) -> Path:
        """Create a numbered .bak sibling for an existing file."""
        counter = 1
        while True:
            candidate = path.parent / f"{path.name}.{counter}.bak"
            if not candidate.exists():
                break
            counter += 1
            if counter > 9999:
                raise RuntimeError("Could not find a free backup slot")
        candidate.write_bytes(path.read_bytes())
        return candidate

    def _handle_file_tool(self, name: str, params: dict[str, Any]) -> str:
        path = self._resolve_project_path(params.get("path", ""))
        if path is None:
            return f"Invalid path: {params.get('path')!r}. Must be inside the project directory."

        if name == "read_file":
            if not path.exists():
                return f"File not found: {path.relative_to(constants.PROJECT_DIR)}"
            try:
                text = path.read_text(encoding="utf-8")
            except Exception as e:
                return f"Could not read {path.name}: {e}"
            max_len = self.config["agent"].get("max_output_len", 4000)
            if len(text) > max_len:
                text = text[:max_len] + "\n... (truncated)"
            return f"Contents of {path.relative_to(constants.PROJECT_DIR)}:\n{text}"

        # Mutating file tools: backup by default unless explicitly disabled.
        backup_default = self.config.get("agent", {}).get("backup_before_edit", True)
        backup = params.get("backup")
        if backup is None:
            backup = backup_default

        if name == "write_file":
            try:
                if path.exists() and backup:
                    bak = self._make_backup(path)
                    msg = f"Backed up original to {bak.name}. "
                else:
                    msg = ""
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(params.get("content", ""), encoding="utf-8")
                self.retriever.invalidate_cache()
                return f"{msg}Wrote {path.relative_to(constants.PROJECT_DIR)}."
            except Exception as e:
                return f"Failed to write {path.name}: {e}"

        if name == "edit_file":
            if not path.exists():
                return f"File not found: {path.relative_to(constants.PROJECT_DIR)}"
            try:
                original = path.read_text(encoding="utf-8")
            except Exception as e:
                return f"Could not read {path.name}: {e}"
            old_string = params.get("old_string", "")
            new_string = params.get("new_string", "")
            if old_string not in original:
                return (
                    f"Could not find the exact old_string in {path.relative_to(constants.PROJECT_DIR)}. "
                    "Use read_file to see the current contents, then retry with the exact text."
                )
            if backup:
                bak = self._make_backup(path)
                msg = f"Backed up original to {bak.name}. "
            else:
                msg = ""
            path.write_text(original.replace(old_string, new_string, 1), encoding="utf-8")
            self.retriever.invalidate_cache()
            return f"{msg}Edited {path.relative_to(constants.PROJECT_DIR)}."

    def _execute_tool(self, name: str, params: dict[str, Any]) -> str:
        # Respect tool-group enable/disable settings.
        group = tooling.tool_group(name)
        enabled_groups = getattr(self, "enabled_groups", None)
        if group is not None and enabled_groups is not None and group not in enabled_groups:
            return f"Tool '{name}' is disabled."

        # Non-terminal front-ends (Telegram) ask before state-mutating tools.
        if self.confirm_fn is not None and name in _TELEGRAM_CONFIRM_TOOLS:
            prompt = self._tool_confirm_prompt(name, params)
            if not self.confirm_fn(prompt):
                return f"Tool '{name}' was not approved."

        # A tool failing outright (e.g. clicking before the browser was ever
        # opened) must never crash the whole session — every branch below
        # already tries to catch its own likely failures, but this is the
        # backstop for anything that slips through. It becomes an
        # observation the model — and the tool-mistake-learning pipeline in
        # _agent_turn — can react to, same as any other tool failure.
        try:
            return self._dispatch_tool(name, params)
        except Exception as e:
            return f"Tool '{name}' failed unexpectedly: {e}"

    def _dispatch_tool(self, name: str, params: dict[str, Any]) -> str:
        if name == "write_note":
            try:
                p = memory.save_note(params["title"], params["body"])
                self.retriever.invalidate_cache()
                return f"Saved note: {p.name}"
            except Exception as e:
                return f"Failed to save note: {e}"

        if name == "save_skill":
            try:
                result = memory.save_skill(
                    params["name"],
                    params["steps"],
                    config=self.config,
                    tokenizer=self.tokenizer,
                    auto_train_adapter=True,
                )
                self.retriever.invalidate_cache()
                note_path = result.get("note_path", "")
                role = result.get("role", "")
                msg = result.get("message", "")
                return f"Saved skill note: {Path(note_path).name}\n  Worker role: {role}\n  {msg}"
            except Exception as e:
                return f"Failed to save skill: {e}"

        if name in ("read_file", "edit_file", "write_file"):
            return self._handle_file_tool(name, params)

        if name == "run_command":
            cmd = params["cmd"].strip()
            # Shell-heavy commands (pipes, redirections, globs, semicolons) are
            # routed through the local shell instead of shlex+no-shell, so the
            # user gets the behavior they expect from a normal terminal.
            if _looks_like_shell_command(cmd):
                ok, out = sandbox.run_shell(cmd, self.config, confirm_fn=self.confirm_fn)
                return f"Shell command exited {'ok' if ok else 'error'}.\nOutput:\n{out}"
            ok, out = sandbox.run_sandboxed(params["cmd"], self.config, confirm_fn=self.confirm_fn)
            return f"Command '{params['cmd']}' exited {'ok' if ok else 'error'}.\nOutput:\n{out}"

        if name == "run_remote":
            ok, out = sandbox.run_remote(
                params["host"], params["command"], self.config, confirm_fn=self.confirm_fn
            )
            return f"Remote '{params['host']}' command exited {'ok' if ok else 'error'}.\nOutput:\n{out}"

        if name == "execute_code":
            ok, out = sandbox.run_python_code(params["code"], self.config)
            return f"Python script exited {'ok' if ok else 'error'}.\nOutput:\n{out}"

        if name == "web_search":
            ok, out = web.web_search(params["query"], self.config)
            return f"Web search for '{params['query']}' {'succeeded' if ok else 'failed'}.\nResults:\n{out}"

        if name == "read_page":
            ok, out = web.read_page(params["url"], self.config)
            return f"Reading {params['url']} {'succeeded' if ok else 'failed'}.\nContent:\n{out}"

        if name == "browser_open":
            if not self.config.get("browser", {}).get("enabled", False):
                return (
                    "Browser automation is disabled. If you want me to open my "
                    "own Google Chrome window, enable it with "
                    "<config set=\"browser.enabled\">true</config>."
                )
            out = self.browser.open(params["url"])
            if "blocked" not in out and "error" not in out.lower():
                self._last_browsed_url = params["url"]
                out += _browser_peek(self.browser)
            return out

        browser_action_tools = {
            "browser_click": lambda: self.browser.click(
                selector=params["target"] if params["target"].startswith(("#", ".", "//", "[")) else "",
                text=params["target"] if not params["target"].startswith(("#", ".", "//", "[")) else "",
            ),
            "browser_type": lambda: self.browser.type_text(params["text"], press_enter=params["enter"]),
            "browser_scroll": lambda: self.browser.scroll(params["direction"]),
            "browser_press": lambda: self.browser.press(params["key"]),
            "browser_close": lambda: self.browser.close(),
        }

        if name in browser_action_tools:
            if not self.config.get("browser", {}).get("enabled", False):
                return (
                    "Browser automation is disabled. Enable it with "
                    "<config set=\"browser.enabled\">true</config> so I can use "
                    "my own Chrome window."
                )
            out = browser_action_tools[name]()
            if "Browser is not open" in out:
                out = (
                    f"{out} Use <browse>https://...</browse> to load a page first, "
                    "then retry the action."
                )
            return out + _browser_peek(self.browser)

        if name == "save_memory":
            return memory.save_memory(params["store"], params["content"], self.config,
                                      replace=params.get("replace", False))

        if name == "compact_memory":
            store = params.get("store", "memory")
            def _summarize(text: str) -> str:
                return str(self.generate_fn(
                    self.model, self.tokenizer, prompt=text, sampler=self.sampler,
                    max_tokens=512, verbose=False,
                )).strip()
            msg, _ = memory.compact_store(store, self.config, summarize_fn=_summarize)
            self.retriever.invalidate_cache()
            return msg

        if name == "config_show":
            return f"Current configuration:\n{config_show(self.config)}"

        if name == "config_set":
            return set_config_value(self.config, params["key"], params["value"])

        if name == "digest_notes":
            try:
                decayed = self._decay_stale_notes()
                cnt = training.digest_notes_to_training(
                    self.tokenizer, self.system_prompt, self.config)
                msg = f"Digested {cnt} new training samples from notes."
                if decayed:
                    msg += (f" Archived {len(decayed)} stale research note(s) "
                            f"past their decay age.")
                return msg
            except Exception as e:
                return f"Digest error: {e}"

        if name == "schedule_job":
            try:
                job = cron.add_cron_job(
                    params["schedule"], params["text"],
                    blocked_commands=set(self.config["sandbox"].get("blocked_commands", [])),
                    owner=self.owner,
                )
                return f"Scheduled job {job['id']}: {job['schedule']} — {job['text']}"
            except ValueError as e:
                return f"Could not schedule job: {e}"

        if name == "list_cron_jobs":
            jobs = cron.list_cron_jobs()
            if not jobs:
                return "No scheduled jobs."
            lines = ["Scheduled jobs:"]
            for job in jobs:
                owner_tag = f" (owner: {job['owner']})" if job.get("owner") else ""
                lines.append(f"  {job['id']}: {job['schedule']} — {job['text']}{owner_tag}")
            return "\n".join(lines)

        if name == "delete_cron_job":
            try:
                job = cron.delete_cron_job(int(params["job_id"]), owner=self.owner)
                return f"Deleted job {job['id']}: {job['schedule']} — {job['text']}"
            except (ValueError, KeyError) as e:
                return f"Could not delete job: {e}"

        if name == "update_cron_job":
            try:
                job = cron.update_cron_job(
                    int(params["job_id"]),
                    schedule=params.get("schedule"),
                    text=params.get("text"),
                    blocked_commands=set(self.config["sandbox"].get("blocked_commands", [])),
                    owner=self.owner,
                )
                return f"Updated job {job['id']}: {job['schedule']} — {job['text']}"
            except (ValueError, KeyError) as e:
                return f"Could not update job: {e}"

        if name == "brain_solve":
            prompt = params.get("prompt", "").strip()
            if not prompt:
                return "No prompt provided to brain_solve."
            use_frontier = bool(params.get("use_frontier", False))
            result = mcp_bridge.brain_solve(prompt, use_frontier=use_frontier)
            if not result.get("success"):
                err = result.get("error", "unknown error")
                return f"brain_solve failed: {err}"
            source = result.get("source", "unknown")
            fallback = " (frontier fallback)" if result.get("fallback") else ""
            return f"[{source}{fallback}] {result['output']}"

        if name == "train_adapter":
            self._guarded_train()
            return self._last_train_note

        if name == "retrain_adapter":
            self._cmd_retrain()
            return self._last_train_note

        if name == "system_check":
            report = health.system_check(self.config)
            return json.dumps(report, indent=2, default=str)

        if name == "verify_features":
            report = health.verify_enabled_features(self.config, verbose=False)
            self._health_report = report
            return json.dumps(report, indent=2, default=str)

        if name == "delegate_task":
            if not self.config.get("dispatch", {}).get("enabled", False):
                return "Delegation is disabled (dispatch.enabled is off)."
            return self.dispatch.run_delegated_task(
                params["role"], params["task"], browser=self.browser)

        if name == "add_golden_case":
            return self._add_golden_case(params)

        if name.startswith("mcp_"):
            from symbio.app import mcp_tools
            tool_name = name[4:]
            ok, output = mcp_tools.execute_mcp_tool(tool_name, params, self.config)
            return f"MCP tool '{name}' {'succeeded' if ok else 'failed'}.\nOutput:\n{output}"

        return f"Unknown tool: {name}"

    def _add_golden_case(self, params: dict[str, Any]) -> str:
        """Append a new case to golden_cases.json and return a status message."""
        from symbio.app import golden as golden_mod

        case_id = params.get("id", "").strip()
        if not case_id:
            return "add_golden_case requires an id."
        if not re.match(r"^[a-z0-9_]+$", case_id):
            return "Golden case id must be lowercase letters, digits, and underscores."

        description = params.get("description", "").strip() or case_id
        prompt = params.get("prompt", "").strip()
        if not prompt:
            return "add_golden_case requires a prompt."
        requirements = params.get("requirements", [])
        if not isinstance(requirements, list) or not requirements:
            return "add_golden_case requires at least one requirement."

        data: dict[str, Any] = {}
        if constants.GOLDEN_CASES_FILE.exists():
            try:
                data = json.loads(constants.GOLDEN_CASES_FILE.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    data = {}
            except Exception:
                data = {}

        if case_id in data:
            return f"Golden case '{case_id}' already exists; edit {constants.GOLDEN_CASES_FILE.name} directly to change it."

        entry: dict[str, Any] = {
            "description": description,
            "prompt": prompt,
            "requirements": requirements,
        }
        ideal_reply = params.get("ideal_reply", "").strip()
        if ideal_reply:
            entry["ideal_reply"] = ideal_reply

        data[case_id] = entry
        try:
            constants.GOLDEN_CASES_FILE.write_text(
                json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        except Exception as e:
            return f"Could not write golden_cases.json: {e}"

        # Validate by loading it.
        try:
            golden_mod.load_user_golden_cases()
        except Exception as e:
            return f"Saved, but the case failed validation: {e}"

        return (
            f"Added golden case '{case_id}' to {constants.GOLDEN_CASES_FILE.name}. "
            "It will be included in the next pre/post-train golden check."
        )

    @staticmethod
    def _tool_confirm_prompt(name: str, params: dict[str, Any]) -> str:
        """User-friendly prompt shown by non-terminal front-ends before
        state-mutating tools."""
        if name == "execute_code":
            code = params.get("code", "").replace("\n", " ")[:200]
            return f"Run the following Python code?\n{code}"
        if name == "run_command":
            cmd = params.get("cmd", "").replace("\n", " ")[:200]
            return f"Run this shell command?\n{cmd}"
        if name == "config_set":
            return f"Change config '{params.get('key')}' to '{params.get('value')}'?"
        if name == "schedule_job":
            return f"Schedule job '{params.get('schedule')}' with text '{params.get('text')}'?"
        if name == "delete_cron_job":
            return f"Delete scheduled job {params.get('job_id')}?"
        if name == "update_cron_job":
            return (f"Update scheduled job {params.get('job_id')} to "
                    f"'{params.get('schedule')}' with text '{params.get('text')}'?")
        if name == "digest_notes":
            return "Digest all notes into training data?"
        if name == "train_adapter":
            return "Start LoRA training? This may take a while."
        if name == "retrain_adapter":
            return (
                "⚠️  Start a FULL adapter rebuild? This will DELETE the current LoRA "
                "adapter and retrain from scratch. This cannot be undone."
            )
        return f"Allow tool '{name}'?"

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
            save = self.input_fn("\n Save conversation for training? [y/N]: ").strip().lower()
            if save in ("y", "yes"):
                saved_count = training.save_history_pairs(
                    self.history, self.tokenizer, self.system_prompt)
                self.output_fn(f"    Appended {saved_count} exchange(s) to {constants.TRAIN_FILE}")

                if self.input_fn("  Train now? [y/N]: ").strip().lower() in ("y", "yes"):
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
    if output_fn is None:
        output_fn = print
    ChatSession(
        config,
        model=model, tokenizer=tokenizer, adapter_loaded=adapter_loaded,
        generate_fn=generate_fn, stream_fn=stream_fn,
        stream_chunk_fn=stream_chunk_fn,
        input_fn=input_fn, output_fn=output_fn, confirm_fn=confirm_fn,
        owner="cli",
    ).run()
