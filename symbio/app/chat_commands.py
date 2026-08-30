"""Slash commands: everything the user types starting with "/".

A mixin rather than a module of functions, because every command reads and
mutates live session state (the model, the history, the config, the browser).
ChatSession inherits it; the methods are unchanged and still take `self`.
"""

import json
import re
import shlex
import threading
from datetime import datetime
from pathlib import Path

from symbio import constants
from symbio.app import (
    cron, dispatch, golden, health, local_telemetry, memory, pending, prompts,
    sandbox, security, setup, skills, tooling, training,
)
from symbio.app.config import config_show, set_config_value
from symbio.app.chat_constants import (
    _HANDLED, _QUIT, THINKING_LEVELS, THINKING_ORDER,
)
from symbio.app.chat_text import _gui_app_for
from symbio.app.chat_ui import (
    _adapter_iters, _adapter_trained_at, _fmt_ago, learn_progress_line,
    print_banner, rainbow,
)


class CommandsMixin:
    """Slash-command handling for ChatSession."""

    def _golden_corpus_command(self, action: str) -> None:
        """`/golden audit` and `/golden prune`: check the training corpus for
        samples that answer a golden case's own prompt in a way that case
        grades as a failure, and optionally drop them.

        A failing golden case says the model got a prompt wrong; it cannot say
        why. When the reason is that the corpus teaches both answers, more
        training is not a fix — the counter-examples have to go first."""
        hits = golden.find_corpus_contradictions(
            self.config, enabled_groups=self.enabled_groups)
        if not hits:
            self.output_fn("  [Golden] No training samples contradict a golden case.")
            return

        counts: dict[str, int] = {}
        for hit in hits:
            counts[hit.case_id] = counts.get(hit.case_id, 0) + 1
        self.output_fn(
            f"  [Golden] {len(hits)} sample(s) teach against a golden case:")
        for case_id, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            self.output_fn(f"    {case_id}: {n} sample(s)")
        for hit in hits:
            # The answer is what contradicts the case; the reasoning block in
            # front of it is just noise in a listing this long.
            reply = " ".join(tooling.strip_reasoning_block(hit.reply).split())
            self.output_fn(
                f"    {hit.path.name}:{hit.line_no} [{hit.case_id}] {reply[:120]}"
                f"{'...' if len(reply) > 120 else ''}")

        if action != "prune":
            self.output_fn(
                "  [Golden] Run /golden prune to drop them, then /train.")
            return
        if not self._yes_no(
                f"  Delete {len(hits)} contradicting training sample(s)? [y/N] "):
            self.output_fn("  [Golden] Left the corpus alone.")
            return
        dropped = golden.drop_corpus_contradictions(
            self.config, enabled_groups=self.enabled_groups)
        total = sum(dropped.values())
        self.output_fn(
            f"  [Golden] Dropped {total} sample(s); a timestamped copy of each "
            "file was kept. Run /train to relearn the contracts.")

    def _yes_no(self, prompt: str) -> bool:
        """Local Y/N prompt: uses confirm_fn if a front-end supplied one, else
        reads a line from input_fn. Used by /telemetry's consent re-prompt."""
        if self.confirm_fn is not None:
            try:
                return self.confirm_fn(prompt)
            except Exception:
                pass
        try:
            ans = self.input_fn(prompt).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return ans in ("y", "yes", "true", "1", "on")

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

        elif cmd.startswith("/resume"):
            # `cmd` is the whole line, not the first word, so an equality test
            # here silently drops every argument form — /resume listed fine
            # and /resume run fell through to "Unknown command".
            parts = user_input.split(None, 1)
            self._cmd_resume(parts[1] if len(parts) == 2 else "")

        elif cmd.startswith("/train_worker"):
            parts = user_input.split(None, 1)
            role = parts[1].strip() if len(parts) == 2 else ""
            if not role:
                self.output_fn("  Usage: /train_worker <role>  (e.g. /train_worker summarize)")
            else:
                trained, msg = dispatch.guarded_train_worker(role, self.config)
                self.output_fn(f"  [Worker] {msg}")

        elif cmd == "/security":
            path = constants.SECURITY_FILE
            self.output_fn(f"  [Security] Policy: {path}")
            self.output_fn(f"  [Security] Digest: {security.policy_digest()[:16] or '(missing)'}")
            self.output_fn(
                "  [Security] Not writable from inside the assistant: no tool "
                "call, shell command, or script can change it.")
            self.output_fn(
                f"  [Security] To change it, edit {path.name} yourself; it "
                "takes effect on the next turn.")
            if path.exists():
                self.output_fn("")
                for line in path.read_text(encoding="utf-8").rstrip().splitlines():
                    self.output_fn(f"    {line}")

        elif cmd.startswith("/golden ") and cmd.split(None, 1)[1].strip() in ("audit", "prune"):
            self._golden_corpus_command(cmd.split(None, 1)[1].strip())

        elif cmd == "/golden":
            result = golden.run_golden_set(
                self.model, self.tokenizer, self.generate_fn, self.sampler,
                self.system_prompt, self.config, self.enabled_groups)
            self.output_fn(f"  [Golden] {result.pass_count}/{result.total} checks passing:")
            # all_golden_cases(), not GOLDEN_CASES: user-defined cases from
            # golden_cases.json are run and counted, so they have to be listed
            # too or the report silently omits the ones it just graded.
            for case in golden.all_golden_cases():
                mark = "PASS" if result.results.get(case.id) else "FAIL"
                self.output_fn(f"    [{mark}] {case.id} — {case.description}")
                if result.results.get(case.id):
                    continue
                # Show what the model actually said. Without this a failure is
                # just a name, and diagnosing it means re-running the case by
                # hand outside the CLI — which is how "<cmd>" coming out as
                # "/cmd>" stayed invisible: right command, one wrong token,
                # nothing to parse, and no way to see it from this report.
                reply = " ".join(result.replies.get(case.id, "").split())
                if reply:
                    self.output_fn(
                        f"           reply: {reply[:200]}"
                        f"{'...' if len(reply) > 200 else ''}")
            if result.pass_count < result.total:
                self.output_fn(
                    "  [Golden] /golden audit checks whether the corpus itself "
                    "teaches against a failing case.")

        elif cmd == "/wildcards":
            from symbio.app import wildcards as _wild

            result = _wild.run_check(
                self.model, self.tokenizer, self.generate_fn, self.sampler,
                self.system_prompt, self.config)
            failed = [t["id"] for t in result.tasks if not t["passed"]]
            entry = _wild.record_run(result.pass_count, result.total, failed,
                                     note="manual /wildcards run",
                                     adapter_loaded=self.adapter_loaded)
            self.output_fn(f"  [Wild] {_wild.format_result(entry)}")
            for task in result.tasks:
                mark = "PASS" if task["passed"] else "FAIL"
                self.output_fn(f"    [{mark}] {task['id']}")
            history = _wild.load_history()
            if len(history) > 1:
                trend = " → ".join(str(h["score"]) for h in history[-6:])
                self.output_fn(f"  [Wild] Trend (last {min(6, len(history))}): {trend}")

        elif cmd == "/digest":
            self._decay_stale_notes()
            added = training.digest_notes_to_training(
                self.tokenizer, self.system_prompt, self.config)
            if added:
                self.output_fn(f"  Digested {added} new note samples into training data.")
            else:
                self.output_fn("  No new or changed notes to digest.")

        elif cmd.startswith("/index-notes"):
            rest = user_input[len("/index-notes"):].strip()
            force = rest == "--force"
            self._cmd_index_notes(force=force)

        elif cmd.startswith("/auto-index"):
            rest = user_input[len("/auto-index"):].strip().lower()
            if rest in ("on", "true", "yes", "1"):
                self.config.setdefault("rag", {})["auto_index_enabled"] = True
                from symbio.app.config import save_config
                save_config(self.config)
                self.output_fn("  Auto-index enabled. Notes will be reindexed in the background.")
                # If the worker thread is already running but was disabled by config
                # at startup, it exited; restart it.
                if self._index_stop.is_set():
                    self._index_stop.clear()
                    threading.Thread(target=self._background_index_worker, daemon=True).start()
            elif rest in ("off", "false", "no", "0"):
                self.config.setdefault("rag", {})["auto_index_enabled"] = False
                from symbio.app.config import save_config
                save_config(self.config)
                self.output_fn("  Auto-index disabled.")
            else:
                state = "ON" if self.config.get("rag", {}).get("auto_index_enabled") else "OFF"
                self.output_fn(f"  Auto-index is {state}.")
                self.output_fn("  Usage: /auto-index on | /auto-index off")

        elif cmd.startswith("/run"):
            self._cmd_run(user_input[4:].strip())

        # `/notes` is a prefix match away from `/note`, and this branch is
        # first, so it used to swallow it: typing /notes opened the note
        # composer with "s" as the title and consumed the next line as the
        # body, while the real /notes handler further down was unreachable
        # code for a command the banner advertises. Found by typing it into
        # a real session, which is the only place the two are adjacent.
        elif cmd.startswith("/note") and cmd.rstrip() != "/notes":
            self._cmd_note(user_input[5:].strip())

        elif cmd == "/learn":
            self._learn_from_correction(verbose=True)

        elif cmd == "/skills":
            # Not `skills`: binding that name anywhere in this function makes
            # it local for the whole of it, so the module import at the top
            # stops resolving and /skill-adapters — hundreds of lines below,
            # in the same function — dies with UnboundLocalError before it
            # runs a line of its own.
            saved_skills = memory.list_skills()
            if not saved_skills:
                self.output_fn("  No skills saved yet.")
            else:
                self.output_fn(f"  {len(saved_skills)} skill(s):")
                for title, path in saved_skills:
                    self.output_fn(f"    - {title}  ({path.name})")

        elif cmd.startswith("/new-skill"):
            rest = user_input[len("/new-skill"):].strip()
            name, steps = (rest.split("|", 1) + [""])[:2] if rest else ("", "")
            name, steps = name.strip(), steps.strip()
            # The steps are not optional, and omitting them used to be quiet:
            # the skill was created with the placeholder "(no steps provided
            # yet)" as its procedure and a background fine-tune started on it.
            # memory.save_skill refuses that now; this says so before the work
            # starts, and shows the pipe, which is the part people miss.
            if not name or not steps:
                self.output_fn("  Usage: /new-skill <name> | <steps>")
                self.output_fn(
                    "  The steps go after the pipe, on the same line:")
                self.output_fn(
                    "    /new-skill Rotate Keys | 1. Read the current key. "
                    "2. Issue a new one. 3. Retire the old one.")
                if name and not steps:
                    self.output_fn(
                        f"  Nothing was created for '{name}' — a skill with no "
                        f"procedure cannot be trained or retrieved.")
            else:
                try:
                    result = memory.save_skill(
                        name,
                        steps,
                        config=self.config,
                        tokenizer=self.tokenizer,
                        auto_train_adapter=True,
                        example_generator=self._skill_example_generator(),
                        history=list(self.history),
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
                    tooling.refresh_mcp_tools(self.config)
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

        elif cmd.startswith("/backup"):
            # training.backup_adapter has always existed — run_training calls it
            # so the golden set can roll a regression back — but nothing exposed
            # it to the user. Taking a snapshot before a deliberate experiment
            # (a new base model, a corpus change) meant knowing the internals or
            # copying the directory by hand.
            label = user_input[len("/backup"):].strip() or None
            if label and not re.fullmatch(r"[\w.-]+", label):
                self.output_fn(
                    "  Label must be letters, digits, dot, dash or underscore "
                    "— it becomes a directory name.")
            else:
                try:
                    made = training.backup_adapter(label=label)
                    if made is None:
                        self.output_fn(
                            "  No adapter to back up yet — nothing has been trained "
                            "for this model.")
                    else:
                        size_mb = sum(
                            f.stat().st_size for f in made.rglob("*") if f.is_file()
                        ) // (1024 * 1024)
                        self.output_fn(f"  Backed up to {made.name} ({size_mb} MB)")
                        self.output_fn(f"  Roll back with: /restore-adapter {made.name}")
                except Exception as e:
                    self.output_fn(f"  Backup failed: {e}")

        elif cmd.startswith("/restore-adapter"):
            name = user_input[len("/restore-adapter"):].strip()
            backups = sorted(
                (p for p in constants.ADAPTER_DIR.parent.glob(
                    f"{constants.ADAPTER_DIR.name}.*.bak") if p.is_dir()),
                key=lambda p: p.stat().st_mtime, reverse=True)
            if not name:
                if not backups:
                    self.output_fn("  No adapter backups yet. Take one with /backup.")
                else:
                    self.output_fn("  Adapter backups, newest first:")
                    for p in backups[:15]:
                        when = datetime.fromtimestamp(p.stat().st_mtime)
                        self.output_fn(f"    {p.name}  ({when:%Y-%m-%d %H:%M})")
                    self.output_fn("  Roll back with: /restore-adapter <name>")
            else:
                target = next((p for p in backups if p.name == name), None)
                if target is None:
                    self.output_fn(f"  No backup named '{name}'. /restore-adapter lists them.")
                else:
                    try:
                        # Snapshot what is about to be overwritten. Restoring is
                        # how you undo a bad train; without this, restoring the
                        # wrong one is an undo you cannot undo.
                        training.backup_adapter(label="PRE_RESTORE")
                        training.restore_adapter(target)
                        self.output_fn(f"  Restored {target.name}.")
                        self.output_fn("  Restart Symbio to load it.")
                    except Exception as e:
                        self.output_fn(f"  Restore failed: {e}")

        elif cmd.startswith("/archive"):
            # startswith, not equality: `cmd` is the whole input line, so an
            # equality test drops every argument. The README documents
            # `/archive --dry-run` and it answered "Unknown command" — and
            # even when matched, dry_run was never passed through, so the
            # documented preview did not exist in chat at all.
            arg = user_input[len("/archive"):].strip().lower()
            dry_run = arg in ("--dry-run", "-n", "dry", "dry-run", "preview")
            try:
                archived = skills.archive_idle_items(self.config, dry_run=dry_run)
                notes = archived.get("notes", [])
                adapters = archived.get("adapters", [])
                if notes or adapters:
                    verb = "Would archive" if dry_run else "Archived"
                    self.output_fn(f"  {verb} {len(notes)} idle note(s) and {len(adapters)} idle adapter(s).")
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
            report = health.verify_enabled_features(
                self.config, verbose=True, output_fn=self.output_fn,
                tokenizer=self.tokenizer)
            self._health_report = report

        elif cmd == "/setup":
            parts = user_input.split(None, 2)[1:]
            if parts and parts[0].lower() == "wizard":
                self.config = setup.run_setup_wizard(
                    self.config, input_fn=self.input_fn, output_fn=self.output_fn
                )
                self.system_prompt = prompts.build_system_prompt(
                    self.config["assistant_name"], self.config["user_name"],
                    self.config
                )
                # Identity changed → the prefilled KV cache holds the old system
                # prompt's tokens, so drop it. The next turn rebuilds a fresh
                # cache instead of mismatching the prefix and re-prefilling.
                self._prompt_cache = None
                self._cached_prompt_ids = None
                self.output_fn("  Setup complete. Some changes may need a restart to take full effect.")
            elif not self.config.get("assistant_name") or not self.config.get("user_name"):
                self.config = setup.run_setup_wizard(
                    self.config, input_fn=self.input_fn, output_fn=self.output_fn
                )
                self.system_prompt = prompts.build_system_prompt(
                    self.config["assistant_name"], self.config["user_name"],
                    self.config
                )
                self._prompt_cache = None
                self._cached_prompt_ids = None
            else:
                self.output_fn("  Run /setup wizard to re-run the full setup, or use /config to change individual settings.")

        elif cmd == "/compact":
            parts = user_input.split(None, 2)[1:]
            store = parts[0].lower() if parts else "memory"
            if store not in ("memory", "profile"):
                self.output_fn("  Usage: /compact [memory|profile]")
            else:
                # /compact can be the user's first input, before any _agent_turn
                # joined the boot prefill — make sure that background prefill is
                # done before we use the model to summarize.
                self._await_prefill()
                def _summarize(text: str) -> str:
                    return str(self.generate_fn(
                        self.model, self.tokenizer, prompt=text, sampler=self.sampler,
                        max_tokens=512, verbose=False,
                    )).strip()
                msg, _ = memory.compact_store(store, self.config, summarize_fn=_summarize)
                self.retriever.invalidate_cache()
                self.output_fn(f"  {msg}")

        elif cmd.startswith("/think"):
            parts = cmd.split(None, 1)
            current = str(self.config.get("agent", {}).get(
                "thinking_level", "none")).lower()
            if len(parts) < 2:
                self.output_fn("  " + rainbow("Thinking dial") + ":")
                for name in THINKING_ORDER:
                    on, budget = THINKING_LEVELS[name]
                    marker = "*" if name == current else " "
                    room = f"+{budget} tokens to reason in" if on else "answer directly"
                    label = rainbow(name) if name == current else name
                    self.output_fn(f"    [{marker}] {label:<8} {room}")
                self.output_fn("  Turn it with: /think none|low|medium|flurry")
            else:
                want = parts[1].strip().lower()
                if want not in THINKING_LEVELS:
                    self.output_fn(
                        f"  Unknown level: {want}. "
                        f"Pick one of {', '.join(THINKING_ORDER)}.")
                else:
                    from symbio.app.config import save_config
                    self.config.setdefault("agent", {})["thinking_level"] = want
                    save_config(self.config)
                    on, budget = THINKING_LEVELS[want]
                    room = (f"reasoning gets {budget} tokens of its own"
                            if on else "answering directly, no reasoning")
                    self.output_fn(f"  Thinking set to {rainbow(want)} — {room}.")

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
            trained_at = _adapter_trained_at()
            if trained_at is not None:
                iters = _adapter_iters()
                self.output_fn(
                    f"  Adapter trained: {_fmt_ago(trained_at)}"
                    + (f" ({iters} iters)" if iters is not None else "")
                )
            last_used = training.adapter_last_used()
            if last_used is not None:
                idle_days = (datetime.now() - last_used).days
                self.output_fn(f"  Adapter last used: {idle_days} day(s) ago")
            self.output_fn(f"  Learn: {learn_progress_line(self.config)}")
            carried_over = pending.describe_outstanding()
            if carried_over:
                self.output_fn(f"  Unfinished tasks: {len(carried_over)} (/resume)")
                for line in carried_over:
                    self.output_fn(f"    - {line}")
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
                        self.config, verbose=True, output_fn=self.output_fn,
                        tokenizer=self.tokenizer,
                    )
                    self._health_report = report
            else:
                self.output_fn("  Usage: /config [show] | /config set <dotted.key> <value>")

        elif cmd.startswith("/cron"):
            self._cmd_cron(user_input)

        elif cmd.startswith("/tidy"):
            # /prune is adapter checkpoints; this prunes what RAG reads back.
            dry = user_input[len("/tidy"):].strip().lower() in ("dry", "--dry", "dry-run")
            report = self._self_prune(dry_run=dry, announce=False)
            if not report["total"]:
                self.output_fn("  Nothing to tidy — notes and session logs are clean.")
            else:
                verb = "Would archive" if dry else "Archived"
                for n in report["notes"]:
                    self.output_fn(f"    note: {n['name']} — {n['reason']}")
                dropped = report["total"] - len(report["notes"])
                self.output_fn(
                    f"  {verb} {len(report['notes'])} note(s); "
                    f"{'would drop' if dry else 'dropped'} {dropped} "
                    f"duplicate log entr(ies) across "
                    f"{len(report['sessions'])} session file(s).")
                if dry:
                    self.output_fn("  (dry run — nothing was changed. Run /tidy to apply.)")

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

        elif cmd.startswith("/telemetry"):
            from symbio.app import telemetry
            from symbio.app.config import save_config
            rest = user_input[len("/telemetry"):].strip().lower()
            tcfg = self.config.setdefault("telemetry", {})
            # `/telemetry` with no argument, or `/telemetry activity [days]`,
            # reads the local activity log back. It had been written to since
            # day one and never once read — log_path() existed and nothing
            # called it — so a MEDIUM-risk security alert and a tool failing
            # four calls in five both sat in the file unnoticed.
            if rest in ("", "all") or rest.split()[0] in ("activity", "all"):
                parts = rest.split()
                verbose = "all" in parts
                days = next((int(p) for p in parts if p.isdigit()), None)
                report = local_telemetry.summarise(days=days)
                self.output_fn(local_telemetry.format_summary(report, verbose=verbose))
                return True
            if rest in ("on", "enable", "true", "yes", "1"):
                # Re-ask consent with the full data set disclosed, honoring the
                # "required consent" rule: the user can say No and keep going.
                self.output_fn(telemetry.consent_summary(self.config))
                if self._yes_no("  Enable anonymous telemetry? [y/N]: "):
                    tcfg["enabled"] = True
                    tcfg["consented"] = True
                    save_config(self.config)
                    self.output_fn("  Telemetry enabled. Set telemetry.endpoint to send to your worker;")
                    self.output_fn("  with no endpoint, records are kept locally under telemetry/.")
                else:
                    tcfg["consented"] = True
                    tcfg["enabled"] = False
                    save_config(self.config)
                    self.output_fn("  Telemetry remains off. (Consent recorded.) /telemetry on re-asks anytime.")
            elif rest in ("off", "disable", "false", "no", "0"):
                tcfg["enabled"] = False
                save_config(self.config)
                self.output_fn("  Telemetry disabled. /telemetry on to re-enable (re-asks consent).")
            else:
                enabled = tcfg.get("enabled", False)
                fb = tcfg.get("feedback_enabled", True)
                endpoint = tcfg.get("endpoint", "") or "(none — local only)"
                self.output_fn(f"  Telemetry: {'ON' if enabled else 'off'}  |  consented: {'yes' if tcfg.get('consented') else 'not yet asked'}")
                self.output_fn(f"  /feedback: {'ON' if fb else 'off'}  |  endpoint: {endpoint}")
                self.output_fn("  /telemetry on  — re-asks consent (shows the full data set first)")
                self.output_fn("  /telemetry off — disable")

        elif cmd.startswith("/feedback"):
            from symbio.app import telemetry
            from symbio.app.config import save_config
            rest = user_input[len("/feedback"):].strip()
            tcfg = self.config.setdefault("telemetry", {})
            if rest.lower() in ("on", "enable", "true", "yes", "1"):
                tcfg["feedback_enabled"] = True
                save_config(self.config)
                self.output_fn("  /feedback enabled. /feedback <your message> to send.")
            elif rest.lower() in ("off", "disable", "false", "no", "0"):
                tcfg["feedback_enabled"] = False
                save_config(self.config)
                self.output_fn("  /feedback disabled. /feedback on to bring it back.")
            elif not rest:
                fb = tcfg.get("feedback_enabled", True)
                self.output_fn(f"  /feedback is {'ON' if fb else 'off'}.")
                self.output_fn("  /feedback <your message>  — send feedback")
                self.output_fn("  /feedback on | /feedback off — toggle")
            else:
                if not tcfg.get("feedback_enabled", True):
                    self.output_fn("  /feedback is disabled. /feedback on to bring it back.")
                else:
                    state = telemetry.load_state()
                    ok, msg = telemetry.send_feedback(rest, self.config, state)
                    if ok:
                        self.output_fn(f"  Feedback {msg}.")
                        if "feedback.txt" in msg:
                            self.output_fn("  Open that file and submit it as a PR, or paste the block")
                            self.output_fn("  into a GitHub Discussion. /feedback off to disable.")
                    else:
                        self.output_fn(f"  Could not save feedback: {msg}")

        elif cmd == "/standing" or cmd.startswith("/standing "):
            # The user's own view of, and veto over, the one store that is
            # served to the model as trusted. A channel they cannot inspect or
            # revoke is not one they can reasonably be asked to trust.
            rest = user_input.split(None, 1)
            arg = rest[1].strip().lower() if len(rest) > 1 else ""
            if arg == "clear":
                self.output_fn(f"  {memory.clear_standing_instructions()}")
                self._prompt_cache = None
                self._cached_prompt_ids = None
            else:
                entries = memory.list_standing_instructions()
                if not entries:
                    self.output_fn("  No standing instructions. Ask for one in chat "
                                   "(\"from now on, ...\") or /standing clear to reset.")
                else:
                    self.output_fn(f"  Standing instructions ({constants.STANDING_FILE.name}):")
                    for entry in entries:
                        self.output_fn(f"    - {entry}")
                    self.output_fn("  /standing clear removes them all.")

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
        # The same GUI-app recovery the model's <cmd> path gets. Without it
        # `/run chrome` typed by hand died on "command not found" while the
        # identical command from the model was quietly corrected to `open -a` —
        # the person driving got less help than the thing being driven.
        # No mistake note here: the user typed this, and their typo is not a
        # lesson about the model's behaviour.
        if not ok:
            app = _gui_app_for(shell_cmd, output)
            if app:
                shell_cmd = f"open -a {shlex.quote(app)}"
                self.output_fn(f"  [Shell] that names a GUI app; retrying as:")
                self.output_fn(f"\n  $ {shell_cmd}")
                ok, output = sandbox.run_sandboxed(
                    shell_cmd, self.config, confirm_fn=self.confirm_fn)
        self.output_fn(f"  [{'ok' if ok else 'err'}]")
        for line in output.splitlines():
            self.output_fn(f"  {line}")
        # A silent command is not a training example. `open -a`, mkdir, touch
        # and most successful writes print nothing, and logging those pairs
        # teaches the model that the correct answer to a command is no answer —
        # the exact behaviour that shows up as a blank reply. Seen while
        # testing: one `/run` of `open -a 'Google Chrome'` wrote a sample whose
        # assistant turn was empty.
        if output.strip():
            training.append_chat_pair(
                user_msg=f"Run this sandbox command and show the output:\n{shell_cmd}",
                assistant_msg=output,
                tokenizer=self.tokenizer,
                system_prompt=self.system_prompt,
            )
            self.output_fn("  -> Logged to training data.\n")
        else:
            self.output_fn("  -> No output; not logged to training data.\n")

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
        # Keep the system prompt as the only trusted system content.
        # Memory and RAG live in user-role context so they cannot override it.
        flush_messages = [{"role": "system", "content": (
            self.system_prompt + prompts.env_note() + prompts.time_note()
        )}]
        flush_messages.extend(self.history[-self.config["agent"]["history_limit"]:])
        memory_block = memory.curated_memory_block(self.config)
        flush_messages.append({"role": "user", "content": (
            (memory_block + "\n\n" if memory_block else "")
            + "[Session ending. If this conversation contained anything durable "
            "worth keeping — facts about the user, lessons learned, procedures "
            "that worked — save it now with <memory>, <profile>, or <note>. "
            "Record only what was actually said or observed in this session; "
            "never add inferred, assumed, or invented details. "
            "Reply with just the tags, or 'nothing to save'.]"
        )})
        try:
            flush_prompt = self.tokenizer.apply_chat_template(
                flush_messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=training.THINKING_ENABLED,
            )
            flush_reply = self.generate_fn(
                self.model, self.tokenizer, prompt=flush_prompt, sampler=self.sampler,
                max_tokens=int(self.config["agent"]["max_reply_tokens"]), verbose=False,
            )
            # The model may reason before emitting the tags; parse only the
            # answer so reasoning text can't be mistaken for a tool call.
            flush_reply = tooling.strip_reasoning_block(flush_reply)
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
