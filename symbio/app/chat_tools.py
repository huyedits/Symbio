"""Tool execution: resolving a parsed tool call to a real effect.

_dispatch_tool is the switchboard -- one branch per tool name -- with
_execute_tool wrapping it in confirmation and telemetry, and the file helpers
handling the project-path resolution and backups that the file tools share.

A mixin rather than a module of functions: dispatch needs the session's
browser, config, confirmation callback and output function. ChatSession
inherits it.
"""

import json
import re
import shlex
from pathlib import Path
from typing import Any

from symbio import constants, safety
from symbio.app import (
    cron, health, learn, local_telemetry, mcp_bridge, memory, sandbox,
    security, tooling, training, web,
)
from symbio.app.config import config_show, set_config_value
from symbio.app.chat_constants import _TELEGRAM_CONFIRM_TOOLS
from symbio.app.chat_text import (
    _annotate_sandbox_cwd, _gui_app_for, _looks_like_shell_command,
    _queries_overlap, _repair_project_path_command,
)


def _browser_peek(browser, config=None) -> str:
    """Read the current page, resolved through chat at call time.

    Deliberately not an import-time binding and not a method. The tests stub
    the page reader with `setattr(chat, "_browser_peek", ...)`, which an
    import-time `from ... import _browser_peek` would not see; and several
    drive _dispatch_tool with a duck-typed stand-in for the session rather
    than a real ChatSession, which a `self._peek_browser()` would not find.
    Going through the module on every call satisfies both.
    """
    from symbio.app import chat

    return chat._browser_peek(browser, config)


class ToolsMixin:
    """Tool dispatch and execution for ChatSession."""

    def _resolve_project_path(self, raw_path: str) -> Path | None:
        """Normalize a user-supplied path so it stays inside the project dir."""
        raw_path = raw_path.strip()
        if not raw_path:
            return None
        target = Path(raw_path)
        if not target.is_absolute():
            target = constants.PROJECT_DIR / target
        elif not target.exists():
            # A rooted path that names nothing at the filesystem root, but does
            # name something inside the project, is a project path the model
            # wrote with a leading slash. Observed live 2026-08-24: asked for
            # the size of symbio/app/chat.py it sent "/symbio/app/chat.py",
            # which resolved outside the project, tripped the path_escape risk
            # flag, asked the user to approve a HIGH-risk action, and then
            # failed anyway with "Must be inside the project directory".
            #
            # This can only ever move a path INTO the project — the
            # relative_to check below still runs, and a rooted path that does
            # exist is left alone — so it narrows what is reachable rather than
            # widening it.
            relocated = constants.PROJECT_DIR / raw_path.lstrip("/")
            if relocated.exists():
                target = relocated
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
        # The security policy is not writable from inside the assistant, by any
        # route, before any other gate gets a say. A refusal, not a
        # confirmation prompt: every other high-risk call ends in "ask the
        # user", which is the right answer when the user is the one who wants
        # it and the wrong one here, because the attack this guards against is
        # precisely an instruction that arrived pretending to be them. A file
        # that can be unlocked by a convincing enough message is not locked.
        blocked = security.block_reason(name, params)
        if blocked is not None:
            # Two different refusals now come through here — a write to the
            # policy, and a command that would destroy the assistant's own
            # state. Log them apart: one is someone probing the rules, the
            # other is an `rm -rf adapters` that nearly happened, and reading
            # them as the same event hides which.
            kind = ("self_destruction_blocked"
                    if security.text_destroys_vital(
                        params.get(security.FREE_TEXT_TOOLS.get(name, ""), ""))
                    else "policy_write_blocked")
            safety.log_security_event(kind, {"tool": name, "params": params})
            return blocked

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

        # Risk-based escalation: the more dangerous an action is, the louder
        # the alert. High-risk actions require explicit approval; medium-risk
        # ones run but annotate the observation so the model sees the warning.
        risk = safety.assess_tool_risk(
            name, params, self.config,
            # What the user actually typed this turn. A note that only repeats
            # their own words is a record of the request, not an injection
            # laundered into storage — see safety.echoes_live_user.
            user_text=getattr(self, "_user_text_this_turn", ""))
        # Where did this call come from? A tool that has never run here, on a
        # turn that pulled in retrieved text, is the shape an injected action
        # takes — so ask, rather than assume the model chose it freely.
        # Only when someone can actually answer. This escalation exists to turn
        # a suspicious call into a question; with no confirm_fn — a scripted
        # run, a test, the Telegram bot mid-poll — there is nobody to ask, and
        # "ask" silently degrades into "refuse". Blocking a real action on a
        # behavioural guess that was never even voiced is worse than not
        # guessing, so headless runs keep the risk score they earned.
        # "Is there anyone to ask?" is safety's question to answer, not this
        # module's. Asking it as `confirm_fn is not None` disabled both
        # escalations in the interactive CLI — which supplies no confirm_fn and
        # prompts on the TTY instead — while leaving them on for the front-ends
        # that do supply one. The guard was off wherever a human was actually
        # sitting there.
        if safety.can_prompt(self.confirm_fn):
            risk = safety.assess_provenance(
                name, risk, self.config,
                untrusted_in_context=getattr(self, "_untrusted_this_turn", False))
            # Provenance stops firing once a tool is familiar; this does not.
            # A shell call on a turn that asked for nothing is the model's own
            # idea however many times it has run before.
            risk = safety.assess_request_intent(
                name, params, risk, self.config,
                user_asked_for_action=getattr(
                    self, "_action_asked_this_turn", True))
        allowed, reason = safety.maybe_confirm(name, params, risk, self.config, self.confirm_fn)
        # `reason` is non-None only when the gate actually asked; combined with
        # `allowed` that means the user was shown this call and said yes. The
        # annotation below needs to carry that, or the model re-litigates an
        # action its own user already authorised.
        user_approved = allowed and reason is not None
        if not allowed:
            safety.log_security_event("tool_blocked", {
                "tool": name, "params": params, "risk": risk, "reason": reason,
            })
            return (
                f"Tool '{name}' was not approved (risk score {risk['risk_score']}/3: "
                f"{', '.join(risk['flags'])})."
            )

        # A tool failing outright (e.g. clicking before the browser was ever
        # opened) must never crash the whole session — every branch below
        # already tries to catch its own likely failures, but this is the
        # backstop for anything that slips through. It becomes an
        # observation the model — and the tool-mistake-learning pipeline in
        # _agent_turn — can react to, same as any other tool failure.
        try:
            observation = self._dispatch_tool(name, params)
        except Exception as e:
            return f"Tool '{name}' failed unexpectedly: {e}"

        # Only now does this tool stop being novel. Recording it before the
        # confirmation would let a refused call teach the baseline that it was
        # normal — the next identical attempt would sail through unasked.
        safety.record_tool_use(name)

        log_score = self.config.get("safety", {}).get("log_score", 2)
        if risk["risk_score"] >= log_score:
            annotation = safety.risk_annotation(risk, approved=user_approved)
            observation += annotation
            safety.log_security_event("tool_executed", {
                "tool": name, "params": params, "risk": risk, "annotation": annotation,
            })
        return observation

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
                    example_generator=self._skill_example_generator(),
                    history=list(self.history),
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
                if "no such file" in out.lower():
                    repaired = _repair_project_path_command(cmd)
                    if repaired:
                        self._status(f"  [Shell] that path is not under "
                                     f"{constants.SANDBOX_DIR.name}/; retrying "
                                     f"with the project path.")
                        ok2, out2 = sandbox.run_shell(
                            repaired, self.config, confirm_fn=self.confirm_fn)
                        if "no such file" not in out2.lower():
                            return (f"Shell command '{repaired}' exited "
                                    f"{'ok' if ok2 else 'error'}.\nOutput:\n{out2}")
                out = _annotate_sandbox_cwd(cmd, out)
                return f"Shell command exited {'ok' if ok else 'error'}.\nOutput:\n{out}"
            ok, out = sandbox.run_sandboxed(params["cmd"], self.config, confirm_fn=self.confirm_fn)

            # Launch a GUI app the way macOS actually launches one.
            #
            # 481 of the 542 run_command failures in the activity log are this
            # single mistake: the model trying to start a desktop app by a CLI
            # name that has never existed.
            #
            #     241  chrome            120  chromebrowser
            #     120  chrome-app
            #
            # env_note() already tells it, in the prompt, on every single turn,
            # that "GUI apps have no CLI names like 'chrome'" and to use
            # `open -a 'Google Chrome'`. It read that and did it anyway, 481
            # times. Another sentence of prompt is not the fix; this is
            # deterministic and belongs in code.
            if not ok:
                app = _gui_app_for(params["cmd"], out)
                if app:
                    self._status(f"  [Shell] '{params['cmd'].strip()}' is a GUI app; "
                                 f"launching it with open -a '{app}'.")
                    retry = f"open -a {shlex.quote(app)}"
                    ok, out = sandbox.run_sandboxed(
                        retry, self.config, confirm_fn=self.confirm_fn)
                    local_telemetry.log_event(
                        "gui_launch_recover", asked=params["cmd"].strip(),
                        app=app, ok=ok)

                    # Recover the action AND keep the lesson.
                    #
                    # This turn's failure-then-fix is normally what feeds the
                    # mistake-note loop: a tool call that fails followed by one
                    # that works gets captured, digested, and trained on. By
                    # recovering internally the loop would never see a failure,
                    # and the model would go on emitting `chrome` forever with
                    # nothing to learn from.
                    #
                    # Writing the note here keeps that signal. Worth noting the
                    # loop had this exact lesson available 481 times already and
                    # the mistake kept happening — so this is not a substitute
                    # for the deterministic fix above, it is the training data
                    # the fix would otherwise have destroyed.
                    if ok:
                        try:
                            learn.save_mistake_note(
                                original_query=f"launch the {app} app",
                                wrong_answer=f"<cmd>{params['cmd'].strip()}</cmd>",
                                # Carry the real failure text, not a paraphrase.
                                # The note is training data, and the model
                                # needs to see the error it actually produced.
                                correction=(
                                    f"Command not found: {params['cmd'].strip()}. "
                                    f"GUI apps have no CLI name; launch them "
                                    f"with open -a."),
                                correct_answer=f"<cmd>{retry}</cmd>",
                            )
                        except Exception:
                            # Never let bookkeeping fail a turn that worked.
                            pass

                    return (f"Command '{retry}' exited {'ok' if ok else 'error'}.\n"
                            f"Output:\n{out}")
            # Repair a path that names a real project file wrongly, then run
            # it again — the same failure-then-fix shape as the GUI-app
            # recovery below, and for the same reason: the model was told the
            # correct path in the observation and reissued the wrong one.
            if not ok and "no such file" in out.lower():
                repaired = _repair_project_path_command(params["cmd"])
                if repaired:
                    self._status(f"  [Shell] that path is not under "
                                 f"{constants.SANDBOX_DIR.name}/; retrying with "
                                 f"the project path.")
                    ok, out = sandbox.run_sandboxed(
                        repaired, self.config, confirm_fn=self.confirm_fn)
                    if ok:
                        return (f"Command '{repaired}' exited ok.\n"
                                f"Output:\n{out}")
            out = _annotate_sandbox_cwd(params["cmd"], out)
            if ok and not out.strip():
                # Same trap as execute_code: many successful commands are
                # silent (open -a, mkdir, touch, cp), and a bare "exited ok"
                # with an empty Output block invites the model to describe a
                # result it never saw.
                return (f"Command '{params['cmd']}' exited ok and printed no "
                        f"output. That means it ran, not that it produced a "
                        f"result — report only that it ran.")
            return f"Command '{params['cmd']}' exited {'ok' if ok else 'error'}.\nOutput:\n{out}"

        if name == "run_remote":
            ok, out = sandbox.run_remote(
                params["host"], params["command"], self.config, confirm_fn=self.confirm_fn
            )
            return f"Remote '{params['host']}' command exited {'ok' if ok else 'error'}.\nOutput:\n{out}"

        if name == "execute_code":
            ok, out = sandbox.run_python_code(params["code"], self.config)
            if ok and not out.strip():
                # "exited ok" over an empty Output block reads as success and
                # says nothing, and the model fills the silence rather than
                # reporting it. Observed live 2026-08-24: asked to compute
                # 4839*27104 it ran a script that never printed, then answered
                # "4839 × 27104 = 130,875,64" — a fabricated number, and not
                # even a well-formed one. Naming the silence gives it something
                # to act on instead of a void to guess into.
                return ("Python script exited ok but printed NOTHING, so it "
                        "produced no result. A value is only visible if the "
                        "script prints it — call print() on what you want back, "
                        "then run it again. Do not state a result you have not "
                        "seen in this output.")
            return f"Python script exited {'ok' if ok else 'error'}.\nOutput:\n{out}"

        if name == "web_search":
            query = params.get("query", "") or ""
            # If the user gave a subjectless "check online" command, the model
            # had no topic and may have hallucinated a query unrelated to the
            # conversation. Override it with the resolved previous question
            # unless the model's query already mentions a signature word from
            # that question (in which case it bound the right subject itself).
            subject = getattr(self, "_search_subject", None)
            if subject and not _queries_overlap(query, subject):
                self.output_fn(
                    f"  [Auto-correct] 'search' command had no subject — "
                    f"searching the previous question instead of "
                    f"'{query[:60]}'.")
                query = subject
            ok, out = web.web_search(query, self.config)
            return f"Web search for '{query}' {'succeeded' if ok else 'failed'}.\nResults:\n{out}"

        if name == "read_page":
            url = params.get("url", "")
            if not url:
                return "Read page error: no URL provided."
            ok, out = web.read_page(url, self.config)
            if not ok:
                return f"Reading {url} failed.\nContent:\n{out}"
            # Say what this content is NOT. read_page runs the page through
            # html_to_text, so every tag and attribute is gone before the model
            # sees it — and the model cannot tell a stripped page from a page
            # that never had markup. Observed live 2026-08-24: asked for raw
            # HTML it called this, announced "The raw HTML content is:" over
            # tagless text, was corrected, called the same tool again, and
            # concluded "the page doesn't include any HTML tags" about a page
            # whose every row carries a data-testid. It blamed the page for the
            # tool's behaviour, and never reached for fetch_html, which was
            # enabled the whole time.
            return (
                f"Reading {url} succeeded.\nContent (TEXT ONLY — all HTML tags "
                f"and attributes were stripped; this is not markup, and their "
                f"absence here says nothing about the page):\n{out}\n"
                f"[If you need tags, attributes or selectors such as "
                f"data-testid, call fetch_html on the same URL — it returns the "
                f"raw markup.]")

        if name == "fetch_html":
            url = params.get("url", "")
            if not url:
                return "Fetch error: no URL provided."
            ok, out = web.fetch_html(url, self.config)
            if not ok:
                return out
            # Raw markup is the most attacker-controllable text this assistant
            # ingests, and unlike read_page's output nothing has stripped the
            # comments, hidden elements or attribute values where an
            # instruction can sit. RAG context and saved memory are already
            # wrapped this way; fetched markup has more reason to be, not less.
            scan = safety.scan_for_injection(out, self.config)
            self._untrusted_this_turn = True
            return (f"Fetched {url}.\n"
                    + safety.wrap_untrusted("web page markup", out, scan))

        if name == "browser_open":
            if not self.config.get("browser", {}).get("enabled", False):
                return (
                    "Browser automation is disabled. If you want me to open my "
                    "own Google Chrome window, enable it with "
                    "<config set=\"browser.enabled\">true</config>."
                )
            url = params.get("url", "")
            if not url:
                return "Browser open error: no URL provided. Please specify a URL to open."
            out = self.browser.open(url)
            if "blocked" not in out and "error" not in out.lower():
                self._last_browsed_url = url
                out += _browser_peek(self.browser, self.config)
            return out

        if name == "browser_get_text":
            if not self.config.get("browser", {}).get("enabled", False):
                return "Browser automation is disabled."
            if not self.browser.is_open:   # a property, not a method
                return ("The browser is not open. Use browser_open with a URL "
                        "first, then read the page.")
            text = self.browser.get_text()
            if text.startswith("Browser "):  # error string from get_text itself
                return text
            # Page text is attacker-controllable in exactly the way fetched
            # markup is, and this is the one browser tool whose whole output is
            # page content rather than an action result.
            scan = safety.scan_for_injection(text, self.config)
            self._untrusted_this_turn = True
            limit = int(self.config["agent"].get(
                "max_page_chars", self.config["agent"].get("max_output_len", 4000)))
            if len(text) > limit:
                text = text[:limit] + (
                    f"\n... (truncated at {limit} characters; raise "
                    f"agent.max_page_chars to read more)")
            return safety.wrap_untrusted("page text", text, scan)

        browser_action_tools = {
            "browser_click": lambda: self.browser.click(
                selector=params.get("target", "") if str(params.get("target", "")).startswith(("#", ".", "//", "[")) else "",
                text=params.get("target", "") if not str(params.get("target", "")).startswith(("#", ".", "//", "[")) else "",
            ),
            "browser_type": lambda: self.browser.type_text(params.get("text", ""), press_enter=params.get("enter", False)),
            "browser_scroll": lambda: self.browser.scroll(params.get("direction", "down")),
            "browser_press": lambda: self.browser.press(params.get("key", "")),
            "browser_close": lambda: self.browser.close(),
        }

        if name in browser_action_tools:
            if not self.config.get("browser", {}).get("enabled", False):
                return (
                    "Browser automation is disabled. Enable it with "
                    "<config set=\"browser.enabled\">true</config> so I can use "
                    "my own Chrome window."
                )
            # Validate required parameters for browser actions so malformed
            # tool calls produce clear, actionable errors instead of crashing.
            if name == "browser_click" and not params.get("target"):
                return (
                    "Click failed: missing 'target'. "
                    "Retry now with the exact visible text inside the tag, e.g. "
                    "<click>Mac</click>. "
                    "Do not explain the failure — just emit the corrected click tag."
                )
            if name == "browser_type" and not params.get("text"):
                return (
                    "Type failed: missing 'text'. "
                    "Retry now with <type>text to type</type>. "
                    "Do not explain the failure — just emit the corrected type tag."
                )
            if name == "browser_press" and not params.get("key"):
                return (
                    "Press failed: missing 'key'. "
                    "Retry now with <press>down</press>. "
                    "Do not explain the failure — just emit the corrected press tag."
                )
            def _act() -> str:
                """Run the action, turning a raised 'not open' into the same
                string the other paths return.

                The browser reports a closed session two different ways and the
                activity log shows both: 314 failures came back as a returned
                "Browser click error: Browser is not open...", and another 122
                as "Tool 'browser_click' failed unexpectedly: Browser is not
                open..." — an exception caught by the generic handler upstream.
                Recovering only the returned form would leave more than a
                quarter of the failures untouched for no reason.
                """
                try:
                    return browser_action_tools[name]()
                except Exception as exc:
                    if "browser is not open" in str(exc).lower():
                        # name already reads "browser_click"; prefixing another
                        # "Browser" gives "Browser browser_click error".
                        return f"{name} error: {exc}"
                    raise

            out = _act()

            # Reopen and retry once when the page is gone.
            #
            # This is the single largest tool failure in the system. Of 567
            # browser_click calls in the local activity log, 453 failed, and
            # 450 of those failed with "Browser is not open" — the model
            # clicking at a page that was never opened or whose session was
            # reset. The old behaviour was to append a sentence telling it to
            # open a page first, which it had already been told and which
            # plainly was not working.
            #
            # _last_browsed_url has existed since the beginning for exactly
            # this, described in its own comment as being "used to auto-recover
            # when a later click/type/scroll/press finds the browser session
            # was reset or never opened". It was assigned and never once read.
            #
            # Recovery is only attempted for a real action (closing a browser
            # by reopening it first is absurd), only when there is a URL this
            # session already opened successfully, and only once — a retry loop
            # against a page that will not load is worse than a clear failure.
            if (
                "Browser is not open" in out
                and name != "browser_close"
                and self._last_browsed_url
            ):
                self._status(f"  [Browser] Session was closed; reopening "
                             f"{self._last_browsed_url} to retry {name}.")
                reopened = self.browser.open(self._last_browsed_url)
                if "blocked" not in reopened and "error" not in reopened.lower():
                    out = _act()
                    local_telemetry.log_event(
                        "browser_recover", url=self._last_browsed_url, tool=name,
                        ok="Browser is not open" not in out,
                    )

            if "Browser is not open" in out:
                out = (
                    f"{out} Use <browse>https://...</browse> to load a page first, "
                    "then retry the action."
                )
            return out + _browser_peek(self.browser, self.config)

        if name == "save_memory":
            return memory.save_memory(
                params["store"], params["content"], self.config,
                replace=params.get("replace", False),
                user_text=getattr(self, "_user_text_this_turn", ""))

        if name == "set_standing_instruction":
            # The live user's turn is passed through, not looked up: the store
            # is only writable from a turn the user actually typed, and that is
            # checked in save_standing_instruction rather than trusted here.
            out = memory.save_standing_instruction(
                params["instruction"], self.config,
                user_text=getattr(self, "_user_text_this_turn", ""),
                replace=params.get("replace", False))
            # A new standing instruction changes the system prompt, so the
            # warmed KV cache no longer matches its own prefix.
            self._prompt_cache = None
            self._cached_prompt_ids = None
            return out

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
            report = health.verify_enabled_features(
                self.config, verbose=False, tokenizer=self.tokenizer)
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

        ideal_reply = params.get("ideal_reply", "").strip()

        # Golden cases shape future training; reject injected prompts/replies.
        scan = safety.scan_for_injection(
            f"{prompt}\n{ideal_reply}", self.config
        )
        if scan["risk_score"] >= 2:
            safety.log_security_event("golden_case_injection_refused", {
                "id": case_id, "flags": scan["flags"], "snippet": scan["snippet"],
            })
            return (
                f"Refused to add golden case '{case_id}': prompt/ideal_reply "
                f"contains possible injection ({', '.join(scan['flags'])})."
            )

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
