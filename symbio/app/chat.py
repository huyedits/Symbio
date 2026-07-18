"""The interactive chat REPL: slash commands, the autonomous agent loop,
and the growth loop (memory nudges, exit flush, cron surfacing)."""

import json
import logging
import threading
import time
from datetime import datetime
from typing import Any

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

from rag import Retriever
from symbio import constants
from symbio.computer import BrowserSession
from symbio.app import cron, learn, memory, prompts, sandbox, sessions, tooling, training, web
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


def print_banner(config: dict[str, Any], adapter_loaded: bool, dataset_size: int):
    note_count = len(list(constants.NOTES_DIR.glob("*.md")))
    print("\n" + "=" * 50)
    print(f"  {config['assistant_name'].upper()} — PERSONAL CHAT-FINETUNE CLI")
    print(f"   Model  : {config['model_name']}")
    print(f"   User   : {config['user_name']}")
    print(f"   LoRA   : {'YES' if adapter_loaded else 'None (base)'}")
    print(f"   Data   : {dataset_size:,} bytes")
    print(f"   Notes  : {note_count}")
    print("-" * 50)
    print("Commands: /quit  /save  /train  /learn  /forget_last  /status  /prune  /help")
    print("         /run <cmd>  /note [title]  /notes  /skills  /digest  /cron  /config")
    print("  (Caine can also use <note>, <cmd>, <py>, <digest />, <train />, <cron> by itself)")
    print("-" * 50)


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


class ChatSession:
    """One interactive chat session: model, stores, browser, cron thread."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.system_prompt = prompts.build_system_prompt(
            config["assistant_name"], config["user_name"]
        )
        self._refresh_sampler()

        print(" Loading model...")
        self.adapter_config = constants.ADAPTER_DIR / "adapter_config.json"
        self.adapter_loaded = False
        if self.adapter_config.exists():
            print(" Found existing adapter. Loading it...")
            try:
                self.model, self.tokenizer = load(
                    config["model_name"], adapter_path=str(constants.ADAPTER_DIR)
                )
                self.adapter_loaded = True
            except Exception as e:
                print(f" Could not load adapter: {e}")
                print(" Falling back to base model...")
                self.model, self.tokenizer = load(config["model_name"])
        else:
            self.model, self.tokenizer = load(config["model_name"])

        # Seed identity notes + clean training corpus on first run.
        memory.ensure_seed_notes(config)
        training.seed_training_data(self.tokenizer, self.system_prompt, config)

        self.history: list[dict[str, str]] = []
        self.session_id = f"{datetime.now():%Y-%m-%d_%H-%M-%S-%f}"
        self.session_store = sessions.SessionStore(self.session_id)
        # Past sessions are retrievable; the live one is excluded to avoid echo.
        self.retriever = Retriever(config, session_store=self.session_store,
                                   exclude_session_id=self.session_id)
        self.browser = BrowserSession()
        self.logger = _make_chat_logger()
        self.user_turns = 0

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
                    print(f"\n  [Cron] {ev.splitlines()[0]}")

    def _reload_model(self) -> str | None:
        """Reload model+adapter after training; returns an error message or None."""
        try:
            self.model, self.tokenizer = load(
                self.config["model_name"], adapter_path=str(constants.ADAPTER_DIR)
            )
            self.adapter_loaded = True
            return None
        except Exception as e:
            return str(e)

    def _trim_history(self):
        while len(self.history) > self.config["agent"]["history_limit"] + 8:
            self.history.pop(0)

    # ---- Slash commands ----

    def _handle_command(self, user_input: str) -> str:
        """Handle a /command; returns _QUIT or _HANDLED."""
        cmd = user_input.lower()

        if cmd in ("/quit", "/q", "/exit"):
            self._memory_flush()
            print(" Exiting chat.")
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
            print("  Forgot last exchange." if removed else " Nothing to forget.")

        elif cmd == "/save":
            if not self.history:
                print(" Nothing to save yet.")
            else:
                saved_count = training.save_history_pairs(
                    self.history, self.tokenizer, self.system_prompt)
                print(f" Saved {saved_count} exchange(s) to training data.")

        elif cmd == "/train":
            trained = training.run_training(self.config)
            if trained and self.adapter_config.exists():
                print("\n Reloading model with updated adapter...")
                err = self._reload_model()
                print(f" Could not reload adapter: {err}" if err else "  Adapter reloaded.")

        elif cmd == "/digest":
            added = training.digest_notes_to_training(
                self.tokenizer, self.system_prompt, self.config)
            if added:
                print(f"  Digested {added} new note samples into training data.")
            else:
                print("  No new or changed notes to digest.")

        elif cmd.startswith("/run"):
            self._cmd_run(user_input[4:].strip())

        elif cmd.startswith("/note"):
            self._cmd_note(user_input[5:].strip())

        elif cmd == "/learn":
            self._learn_from_correction(verbose=True)

        elif cmd == "/skills":
            skills = memory.list_skills()
            if not skills:
                print("  No skills saved yet.")
            else:
                print(f"  {len(skills)} skill(s):")
                for title, path in skills:
                    print(f"    - {title}  ({path.name})")

        elif cmd == "/notes":
            files = sorted(constants.NOTES_DIR.glob("*.md"))
            if not files:
                print("  No notes yet.")
            else:
                print(f"  {len(files)} note(s):")
                for f in files:
                    print(f"    - {f.name}")

        elif cmd == "/status":
            files = sorted(constants.NOTES_DIR.glob("*.md"))
            data_size = constants.TRAIN_FILE.stat().st_size if constants.TRAIN_FILE.exists() else 0
            adapter_files = list(constants.ADAPTER_DIR.glob("adapters.*"))
            adapter_kb = sum(
                f.stat().st_size for f in constants.ADAPTER_DIR.iterdir() if f.is_file()) // 1024
            print(f"  Model: {self.config['model_name']}")
            print(f"  Assistant: {self.config['assistant_name']} | User: {self.config['user_name']}")
            print(f"  Notes: {len(files)}")
            print(f"  Training data: {data_size:,} bytes")
            print(f"  Adapter loaded: {'YES' if self.adapter_loaded else 'NO'}")
            print(f"  Adapter files: {len(adapter_files)} ({adapter_kb:,} KB)")

        elif cmd.startswith("/config"):
            parts = user_input.split(None, 3)[1:]
            if not parts or parts[0].lower() == "show":
                print(config_show(self.config))
            elif parts[0].lower() == "set" and len(parts) == 3:
                print(f"  {set_config_value(self.config, parts[1], parts[2], allow_sandbox=True)}")
            else:
                print("  Usage: /config [show] | /config set <dotted.key> <value>")

        elif cmd.startswith("/cron"):
            self._cmd_cron(user_input)

        elif cmd == "/prune":
            info = training.prune_adapters()
            if info["removed"]:
                print(f"  Removed {len(info['removed'])} stale checkpoint(s):")
                for name in info["removed"]:
                    print(f"    - {name}")
            else:
                print("  No stale checkpoints to remove.")
            print(f"  Current adapter footprint: {info['total_kb']:,} KB")
            print("  Note: mlx_lm LoRA adapters do not support true weight pruning; keeping rank low and removing checkpoints is the practical way to stay small.")

        elif cmd in ("/help", "/h", "/?"):
            data_size = constants.TRAIN_FILE.stat().st_size if constants.TRAIN_FILE.exists() else 0
            print_banner(self.config, self.adapter_loaded, data_size)

        else:
            print("  Unknown command. Type /help for the command list.")

        return _HANDLED

    def _cmd_run(self, shell_cmd: str):
        if not shell_cmd:
            print("  Usage: /run <command>")
            return
        print(f"\n  $ {shell_cmd}")
        ok, output = sandbox.run_sandboxed(shell_cmd, self.config)
        print(f"  [{'ok' if ok else 'err'}]")
        for line in output.splitlines():
            print(f"  {line}")
        training.append_chat_pair(
            user_msg=f"Run this sandbox command and show the output:\n{shell_cmd}",
            assistant_msg=output,
            tokenizer=self.tokenizer,
            system_prompt=self.system_prompt,
        )
        print("  -> Logged to training data.\n")

    def _cmd_note(self, title: str):
        if not title:
            title = input("  Note title: ").strip()
        if not title:
            print("  Cancelled.")
            return
        body = ""
        print("  Content (empty line to finish):")
        try:
            while True:
                line = input()
                if line == "":
                    break
                body += line + "\n"
        except (EOFError, KeyboardInterrupt):
            pass
        if not body.strip():
            print("  Empty note, cancelled.")
            return
        path = memory.save_note(title, body.strip())
        self.retriever.invalidate_cache()
        print(f"  Saved: {path.name}")

    def _cmd_cron(self, user_input: str):
        import shlex
        try:
            parts = shlex.split(user_input)[1:]
        except ValueError as e:
            print(f"  Parse error: {e}")
            return
        sub = parts[0].lower() if parts else "list"
        if sub == "list":
            jobs = cron.load_cron_jobs()
            if not jobs:
                print("  No scheduled jobs.")
            for j in jobs:
                print(f"  [{j['id']}] {j['schedule']} — {j['text']}")
        elif sub == "add" and len(parts) >= 3:
            try:
                job = cron.add_cron_job(parts[1], " ".join(parts[2:]))
                print(f"  Added job {job['id']}: {job['schedule']} — {job['text']}")
            except ValueError as e:
                print(f"  {e}")
        elif sub == "rm" and len(parts) == 2:
            jobs = cron.load_cron_jobs()
            kept = [j for j in jobs if str(j["id"]) != parts[1]]
            cron.save_cron_jobs(kept)
            print(f"  Removed job {parts[1]}." if len(kept) < len(jobs)
                  else f"  No job with id {parts[1]}.")
        else:
            print('  Usage: /cron [list] | /cron add "<cron expr | at YYYY-MM-DD HH:MM>" <text> | /cron rm <id>')

    # ---- Growth loop ----

    def _memory_flush(self):
        """One last turn on /quit to persist memories before context is lost."""
        flush_min = self.config["memory"]["flush_min_turns"]
        if not (self.config["memory"]["enabled"] and flush_min
                and self.user_turns >= flush_min and self.history):
            return
        print(" Letting the model save memories before exit...")
        flush_messages = [{"role": "system", "content": (
            self.system_prompt + memory.curated_memory_block(self.config)
            + prompts.env_note() + prompts.time_note()
        )}]
        flush_messages.extend(self.history[-self.config["agent"]["history_limit"]:])
        flush_messages.append({"role": "user", "content": (
            "[Session ending. If this conversation contained anything durable "
            "worth keeping — facts about the user, lessons learned, procedures "
            "that worked — save it now with <memory>, <profile>, or <note>. "
            "Reply with just the tags, or 'nothing to save'.]"
        )})
        try:
            flush_prompt = self.tokenizer.apply_chat_template(
                flush_messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
            flush_reply = generate(
                self.model, self.tokenizer, prompt=flush_prompt, sampler=self.sampler,
                max_tokens=int(self.config["agent"]["max_reply_tokens"]), verbose=False,
            )
            for name, params in tooling.parse_tools(flush_reply):
                if name == "save_memory":
                    msg = memory.save_memory(params["store"], params["content"], self.config,
                                             replace=params.get("replace", False))
                    print(f"  [Memory] {msg}")
                elif name == "write_note":
                    p = memory.save_note(params["title"], params["body"])
                    print(f"  [Memory] Saved note: {p.name}")
        except KeyboardInterrupt:
            print("\n  [Memory flush interrupted — exiting without saving.]")
        except Exception as e:
            print(f"  [Memory flush skipped: {e}]")

    def _nudge_block(self) -> str:
        nudge_every = self.config["memory"]["nudge_interval"]
        if not (self.config["memory"]["enabled"] and nudge_every
                and self.user_turns % nudge_every == 0):
            return ""
        return (
            f"\n\n[Reminder: if this session taught you anything durable about "
            f"{self.config['user_name']} or how to do your job, save it now with "
            f"<memory> or <profile>. Skip if nothing is worth keeping.]"
        )

    def _learn_from_correction(self, verbose: bool = False):
        """Capture the last (question -> corrected answer) pair as a mistake
        note; at the configured threshold, retrain and reload the adapter."""
        sample = learn.find_correction_sample(self.history, self.config)
        if sample is None:
            if verbose:
                print("  No recent correction detected. Say something like "
                      "\"No, the answer is ...\" first, then run /learn.")
            return
        path = learn.save_mistake_note(*sample)
        print(f"  [Learn] Correction captured: {path.name}")
        trained = learn.maybe_train_on_mistakes(self.config, self.tokenizer, self.system_prompt)
        if trained and self.adapter_config.exists():
            err = self._reload_model()
            print(f"  [Learn] Adapter reload failed: {err}" if err
                  else "  [Learn] Adapter updated and reloaded.")

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
        final_display = ""
        for _ in range(max_rounds):
            messages = [{"role": "system", "content": (
                self.system_prompt + memory.curated_memory_block(self.config) + rag_block
                + prompts.env_note() + prompts.time_note() + nudge_block
            )}]
            messages.extend(self.history[-self.config["agent"]["history_limit"]:])

            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            try:
                raw_reply = generate(
                    self.model, self.tokenizer, prompt=prompt, sampler=self.sampler,
                    max_tokens=int(self.config["agent"]["max_reply_tokens"]), verbose=False,
                )
                reply = raw_reply.strip()
            except KeyboardInterrupt:
                # Ctrl-C during a slow generation abandons the turn, not the app.
                print("\n  [Generation interrupted.]")
                break
            except Exception as e:
                print(f"[MLX Error: {e}]")
                break

            tools = tooling.parse_tools(reply)
            display = tooling.strip_tool_tags(reply)

            if display.strip():
                final_display = display
                print(f"{self.config['assistant_name']:8}: {display}")
                self.logger.info(f"{self.config['assistant_name']}: {display}")
                self.session_store.log("assistant", display)

            # Never re-run a tool call already executed this turn — a model
            # that repeats itself would otherwise loop until max_rounds.
            fresh_tools = [
                (n, p) for n, p in tools
                if json.dumps([n, p], sort_keys=True) not in executed_calls
            ]

            if not fresh_tools:
                # Normal turn (or pure repetition): store the reply and stop.
                self.history.append({"role": "assistant", "content": reply})
                self._trim_history()
                break
            for n, p in fresh_tools:
                executed_calls.add(json.dumps([n, p], sort_keys=True))

            # There are tools to execute
            self.history.append({"role": "assistant", "content": reply})

            observations = []
            for name, params in fresh_tools:
                print(f"  [Tool: {name}]")
                if name in _WEB_TOOLS:
                    web_used = True
                observations.append(self._execute_tool(name, params))

            # Feed observations back as a system/user turn
            obs_text = "\n".join(observations)
            print(f"  [Observation] {obs_text.replace(chr(10), chr(10) + '  ')}")
            self.history.append({"role": "user", "content": f"[System observation: {obs_text}]"})
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
                print(f"  [Learn] Remembered research: {note.name}")

    def _execute_tool(self, name: str, params: dict[str, Any]) -> str:
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
            ok, out = sandbox.run_sandboxed(params["cmd"], self.config)
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
                cnt = training.digest_notes_to_training(
                    self.tokenizer, self.system_prompt, self.config)
                return f"Digested {cnt} new training samples from notes."
            except Exception as e:
                return f"Digest error: {e}"

        if name == "schedule_job":
            try:
                job = cron.add_cron_job(params["schedule"], params["text"])
                return f"Scheduled job {job['id']}: {job['schedule']} — {job['text']}"
            except ValueError as e:
                return f"Could not schedule job: {e}"

        if name == "train_adapter":
            trained = training.run_training(self.config)
            if trained and self.adapter_config.exists():
                err = self._reload_model()
                if err:
                    return f"Training done but reload failed: {err}"
                return "Training complete. Adapter reloaded."
            return "Training skipped (no new data or failed)."

        return f"Unknown tool: {name}"

    # ---- Main loop ----

    def run(self):
        dataset_size = constants.TRAIN_FILE.stat().st_size if constants.TRAIN_FILE.exists() else 0
        print_banner(self.config, self.adapter_loaded, dataset_size)

        while True:
            try:
                user_input = input(f"{self.config['user_name']:8}: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
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
            save = input("\n Save conversation for training? [y/N]: ").strip().lower()
            if save in ("y", "yes"):
                saved_count = training.save_history_pairs(
                    self.history, self.tokenizer, self.system_prompt)
                print(f"    Appended {saved_count} exchange(s) to {constants.TRAIN_FILE}")

                if input("  Train now? [y/N]: ").strip().lower() in ("y", "yes"):
                    trained = training.run_training(self.config)
                    if trained and self.adapter_config.exists():
                        print("\n Reloading model...")
                        err = self._reload_model()
                        if err:
                            print(f" Could not reload: {err}")


def chat_loop(config: dict[str, Any]):
    ChatSession(config).run()
