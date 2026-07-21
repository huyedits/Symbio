"""The interactive chat REPL: slash commands, the autonomous agent loop,
and the growth loop (memory nudges, exit flush, cron surfacing)."""

import json
import logging
import threading
import time
from datetime import datetime
from typing import Any

from mlx_lm import load, generate
from mlx_lm.generate import stream_generate
from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache
from mlx_lm.sample_utils import make_sampler

from rag import Retriever
from symbio import constants
from symbio.computer import BrowserSession
from symbio.app import cron, dispatch, golden, learn, memory, prompts, sandbox, sessions, tooling, training, web
from symbio.app.config import config_show, set_config_value


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
    output_fn("Commands: /quit  /save  /train  /train_worker  /golden  /learn  /forget_last  /status  /prune  /help")
    output_fn("         /run <cmd>  /note [title]  /notes  /skills  /digest  /cron  /config")
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
    "web_search", "read_page", "browser_open",
    "browser_click", "browser_type", "browser_scroll",
}

# Tools that require explicit approval when running from a non-terminal
# front-end (e.g. Telegram) because they mutate state or run user-supplied code.
_TELEGRAM_CONFIRM_TOOLS = frozenset({
    "execute_code", "run_command", "digest_notes", "train_adapter", "schedule_job", "config_set",
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
                 generate_fn=None, stream_fn=None, stream_chunk_fn=None):
        self.config = config
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
        # KV-cache reuse across generate calls (see _generate_reply);
        # invalidated whenever the prompt's actual prefix changes out from
        # under it (adapter reload, a generation that errored mid-stream).
        self._prompt_cache: list | None = None
        self._cached_prompt_ids: list[int] | None = None
        self.enabled_groups: set[str] = set(
            config.get("tools", {}).get("enabled_groups", [])
        )
        self.system_prompt = prompts.build_system_prompt(
            config["assistant_name"], config["user_name"]
        )
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

        self.history: list[dict[str, str]] = []
        self.session_id = f"{datetime.now():%Y-%m-%d_%H-%M-%S-%f}"
        self.session_store = sessions.SessionStore(self.session_id)
        # Past sessions are retrievable; the live one is excluded to avoid echo.
        self.retriever = Retriever(config, session_store=self.session_store,
                                   exclude_session_id=self.session_id)
        self.browser = BrowserSession(confirm_fn=self.confirm_fn)
        # Worker models are loaded lazily on first delegated task — this
        # just holds the (empty) pool, no extra RAM until dispatch.enabled
        # and something actually delegates.
        self.dispatch = dispatch.WorkerPool(config)
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
        threading.Thread(target=self._cron_worker, daemon=True).start()

    # ---- Infrastructure ----

    def _refresh_sampler(self):
        self.sampler = make_sampler(
            temp=self.config["agent"]["temperature"],
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

    def _generate_reply(self, messages: list[dict[str, str]], chunk_prefix: str = "") -> tuple[str, bool]:
        """Generate the next reply for `messages`.

        When agent.prompt_cache_enabled, reuses the model's KV cache across
        calls: only the token-level suffix that's new since the last call
        (an exact longest-common-prefix diff, not a string heuristic) is
        actually prefilled — the system prompt and unchanged history are
        served from cache instead of reprocessed every round. This is what
        makes multi-round tool loops (e.g. a browser click sequence) and
        ordinary turn-to-turn chat fast; see _common_prefix_len.

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
        max_tokens = int(agent_cfg["max_reply_tokens"])

        if not agent_cfg.get("prompt_cache_enabled", True):
            # Caching off: the exact original call, unchanged.
            text = self.generate_fn(
                self.model, self.tokenizer, prompt=prompt_text, sampler=self.sampler,
                max_tokens=max_tokens, verbose=False,
            )
            return text, False

        ids = self.tokenizer.encode(prompt_text)
        reused = _common_prefix_len(self._cached_prompt_ids, ids)
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

        def _emit(text: str):
            nonlocal shown
            if not shown:
                shown = True
                if chunk_prefix:
                    self.stream_chunk_fn(chunk_prefix)
            self.stream_chunk_fn(text)

        text_parts: list[str] = []
        gen_ids: list[int] = []
        try:
            for response in self.stream_fn(
                self.model, self.tokenizer, feed, max_tokens=max_tokens,
                sampler=self.sampler, prompt_cache=self._prompt_cache,
            ):
                text_parts.append(response.text)
                gen_ids.append(response.token)
                if stripper is not None:
                    safe = stripper.feed(response.text)
                    if safe:
                        _emit(safe)
        except BaseException:
            # The real MLX cache may already be mutated beyond what our
            # bookkeeping reflects (interrupted mid-token) — never trust a
            # stale cache after this; the next call rebuilds it from zero.
            self._prompt_cache = None
            self._cached_prompt_ids = None
            raise

        if stripper is not None:
            tail = stripper.finish()
            if tail:
                _emit(tail)
        if shown:
            self.stream_chunk_fn("\n")

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

    def _guarded_train(self, iters: int | None = None) -> bool:
        """Run LoRA training, reload the adapter, then check it against the
        golden set (a fixed battery of prompts covering identity and
        tool-tag formatting — see symbio.app.golden). A regression, a case
        that passed before this training round but fails after, rolls the
        adapter back automatically so a bad fine-tune never silently ships
        as the new default behavior. Mirrors training.run_training's bool
        contract so it's a drop-in replacement everywhere training is
        triggered (slash command, tool call, end-of-session, /learn)."""
        learn_cfg = self.config.get("learn", {})
        golden_on = learn_cfg.get("golden_set_enabled", True)

        baseline = None
        if golden_on:
            baseline = golden.run_golden_set(
                self.model, self.tokenizer, self.generate_fn, self.sampler,
                self.system_prompt, self.config, self.enabled_groups)
        backup_dir = training.backup_adapter() if golden_on else None

        try:
            trained = training.run_training(self.config, iters=iters)
            if not trained or not self.adapter_config.exists():
                self._last_train_note = "Training skipped (no new data or failed)."
                return trained

            err = self._reload_model()
            if err:
                self.output_fn(f"  [Train] Adapter reload failed: {err}")
                self._last_train_note = f"Training done but reload failed: {err}"
                return True

            if not golden_on or baseline is None:
                self.output_fn("  [Train] Adapter reloaded.")
                self._last_train_note = "Training complete. Adapter reloaded."
                return True

            after = golden.run_golden_set(
                self.model, self.tokenizer, self.generate_fn, self.sampler,
                self.system_prompt, self.config, self.enabled_groups)
            regressions = sorted(baseline.passing - after.passing)
            threshold = int(learn_cfg.get("golden_regression_threshold", 0))

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
        while len(self.history) > self.config["agent"]["history_limit"] + 8:
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

        elif cmd == "/notes":
            files = sorted(constants.NOTES_DIR.glob("*.md"))
            if not files:
                self.output_fn("  No notes yet.")
            else:
                self.output_fn(f"  {len(files)} note(s):")
                for f in files:
                    self.output_fn(f"    - {f.name}")

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

        elif cmd.startswith("/config"):
            parts = user_input.split(None, 3)[1:]
            if not parts or parts[0].lower() == "show":
                self.output_fn(config_show(self.config))
            elif parts[0].lower() == "set" and len(parts) == 3:
                self.output_fn(f"  {set_config_value(self.config, parts[1], parts[2], allow_sandbox=True)}")
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
                    blocked_commands=set(self.config["sandbox"].get("blocked_commands", []))
                )
                self.output_fn(f"  Added job {job['id']}: {job['schedule']} — {job['text']}")
            except ValueError as e:
                self.output_fn(f"  {e}")
        elif sub == "rm" and len(parts) == 2:
            jobs = cron.load_cron_jobs()
            kept = [j for j in jobs if str(j["id"]) != parts[1]]
            cron.save_cron_jobs(kept)
            self.output_fn(f"  Removed job {parts[1]}." if len(kept) < len(jobs)
                  else f"  No job with id {parts[1]}.")
        else:
            self.output_fn('  Usage: /cron [list] | /cron add "<cron expr | at YYYY-MM-DD HH:MM>" <text> | /cron rm <id>')

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

    def _learn_from_correction(self, verbose: bool = False):
        """Capture the last (question -> corrected answer) pair as a mistake
        note; at the configured threshold, retrain and reload the adapter."""
        sample = learn.find_correction_sample(self.history, self.config)
        if sample is None:
            if verbose:
                self.output_fn("  No recent correction detected. Say something like "
                      "\"No, the answer is ...\" first, then run /learn.")
            return
        severity = learn.correction_severity(sample[0], sample[2], self.config)
        path = learn.save_mistake_note(*sample, severity=severity)
        self.output_fn(f"  [Learn] Correction captured (severity {severity}): {path.name}")
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
        rag_block = f"\n\n{rag_context}" if rag_context else ""

        # Live-reload: config changes and prompt.md edits apply on the next turn.
        self._refresh_sampler()
        self.system_prompt = prompts.build_system_prompt(
            self.config["assistant_name"], self.config["user_name"]
        )

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
            messages = [{"role": "system", "content": (
                self.system_prompt + memory.curated_memory_block(self.config) + rag_block
                + prompts.env_note() + prompts.time_note() + nudge_block
            )}]
            messages.extend(self.history[-self.config["agent"]["history_limit"]:])

            try:
                raw_reply, streamed_live = self._generate_reply(
                    messages, chunk_prefix=f"{self.config['assistant_name']:8}: ")
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
                unsure = bool(display.strip()) and learn.sounds_unsure(display)
                fabricated = (not unsure and bool(display.strip())
                              and learn.sounds_fabricated(user_input, display))
                # A turn that ends with no visible answer at all is the model
                # blanking out entirely — always search then, even when the
                # user's wording asked for one (they asked and got nothing).
                blanked = not final_display.strip()
                session_cap = int(self.config["web"].get("auto_search_session_cap", 20))
                if (self.config["web"].get("auto_search_when_unsure", True)
                        and not auto_searched and not web_used
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

        if name == "write_note":
            try:
                p = memory.save_note(params["title"], params["body"])
                self.retriever.invalidate_cache()
                return f"Saved note: {p.name}"
            except Exception as e:
                return f"Failed to save note: {e}"

        if name == "save_skill":
            try:
                p = memory.save_skill(params["name"], params["steps"])
                self.retriever.invalidate_cache()
                return f"Saved skill note: {p.name}"
            except Exception as e:
                return f"Failed to save skill: {e}"

        if name == "run_command":
            ok, out = sandbox.run_sandboxed(params["cmd"], self.config, confirm_fn=self.confirm_fn)
            return f"Command '{params['cmd']}' exited {'ok' if ok else 'error'}.\nOutput:\n{out}"

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
            out = self.browser.open(params["url"])
            if "blocked" not in out and "error" not in out.lower():
                out += _browser_peek(self.browser)
            return out

        if name == "browser_click":
            target = params["target"]
            if target.startswith(("#", ".", "//", "[")):
                out = self.browser.click(selector=target)
            else:
                out = self.browser.click(text=target)
            return out + _browser_peek(self.browser)

        if name == "browser_type":
            out = self.browser.type_text(params["text"], press_enter=params["enter"])
            return out + _browser_peek(self.browser)

        if name == "browser_scroll":
            return self.browser.scroll(params["direction"]) + _browser_peek(self.browser)

        if name == "save_memory":
            return memory.save_memory(params["store"], params["content"], self.config,
                                      replace=params.get("replace", False))

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
                    blocked_commands=set(self.config["sandbox"].get("blocked_commands", []))
                )
                return f"Scheduled job {job['id']}: {job['schedule']} — {job['text']}"
            except ValueError as e:
                return f"Could not schedule job: {e}"

        if name == "train_adapter":
            self._guarded_train()
            return self._last_train_note

        if name == "delegate_task":
            if not self.config.get("dispatch", {}).get("enabled", False):
                return "Delegation is disabled (dispatch.enabled is off)."
            return self.dispatch.run_delegated_task(
                params["role"], params["task"], browser=self.browser)

        return f"Unknown tool: {name}"

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
        if name == "digest_notes":
            return "Digest all notes into training data?"
        if name == "train_adapter":
            return "Start LoRA training? This may take a while."
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


def chat_loop(config: dict[str, Any]):
    ChatSession(config, stream_chunk_fn=lambda s: print(s, end="", flush=True)).run()
