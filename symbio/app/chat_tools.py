"""Tool execution: resolving a parsed tool call to a real effect.

_dispatch_tool is the switchboard -- one branch per tool name -- with
_execute_tool wrapping it in confirmation and telemetry, and the file helpers
handling the project-path resolution and backups that the file tools share.

A mixin rather than a module of functions: dispatch needs the session's
browser, config, confirmation callback and output function. ChatSession
inherits it.
"""

import json
import math
import os
import re
import shlex
import time
from pathlib import Path
from typing import Any

from symbio import computer, constants, guardrails, safety
from symbio.app import (
    cron, health, learn, local_telemetry, mcp_bridge, memory, sandbox,
    security, tooling, training, web,
)
from symbio.app.config import config_show, set_config_value
from symbio.app.chat_text import (
    _annotate_sandbox_cwd, _gui_app_for, _looks_like_shell_command,
    _queries_overlap, _repair_project_path_command,
)


def _image_size(path) -> tuple[int, int]:
    """(width, height) of a saved screenshot, or (0, 0) if it cannot be read."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            return im.size
    except Exception:
        return (0, 0)


def _coords(params: dict[str, Any]) -> tuple[int, int] | None:
    """(x, y) from a tool call, or None if they are not both numbers.

    Models write coordinates as ints, as floats and as strings, and a click
    aimed at ("640", "318") is a click that does not happen.
    """
    try:
        x, y = float(params["x"]), float(params["y"])
    except (KeyError, TypeError, ValueError):
        return None
    # inf and NaN are floats and neither is a place. int(float("1e400")) raises
    # OverflowError, which is not a ValueError and so escaped the handler
    # entirely: the model got "Tool 'browser_click_at' failed unexpectedly:
    # cannot convert float infinity to integer" in place of the recovery
    # advice this function exists to make possible. Negatives parse fine and
    # are never a point on a screen — click_at rejects them against the
    # viewport, desktop_click does not check at all.
    if not (math.isfinite(x) and math.isfinite(y)) or x < 0 or y < 0:
        return None
    return int(x), int(y)


def _ax_centre(element: dict[str, Any]) -> tuple[int, int]:
    """The middle of a control's frame, in the points the mouse moves in.

    Accessibility frames are already logical points on the main display's
    origin — the same space pyautogui clicks in — so unlike a coordinate read
    off a screenshot there is no Retina scale to undo here.
    """
    from symbio import ax

    return ax.centre(element)


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



def _controls_note(session, limit: int = 8) -> str:
    """The page's boxes, and the buttons that send them, handed over with the
    page rather than after a failure.

    Live 2026-09-27 the model typed at a page with nothing focused, then
    clicked the first "Post" it found — the sidebar link, not the button under
    the box — because the page's real handles were only ever shown once a
    step had failed. A page with nothing to fill gets no list: reading needs
    no handles, and the list costs a few hundred tokens a round.
    """
    browser = getattr(session, "browser", None)
    try:
        controls = browser.controls(limit=limit * 2) if browser is not None else []
    except Exception:
        return ""
    fields = [c for c in controls if c.get("kind") == "field"]
    if not fields:
        return ""
    buttons = [c for c in controls if c.get("kind") != "field"]
    lines = ["[Boxes and buttons on this page. Type into a box by its selector "
             "(browser_type with selector=...), then click the button beside "
             "it that sends it; a [disabled] button wakes up once the box has "
             "text:"]
    for c in fields[:4]:
        lines.append(f"{ToolsMixin._control_line(c)}   (browser_type with "
                     f"selector={c.get('selector', '')!r})")
    for c in buttons[:max(2, limit - min(len(fields), 4))]:
        label = str(c.get("label") or "").strip()
        lines.append(ToolsMixin._control_line(c)
                     + (f"   (browser_click with target={label!r})" if label else ""))
    lines.append("]")
    block = "\n".join(lines)
    # Labels and values come from the page: data, never instructions.
    session._untrusted_this_turn = True
    config = getattr(session, "config", None) or {}
    return "\n\n" + safety.wrap_untrusted(
        "page controls", block, safety.scan_for_injection(block, config))


def _typed_words_note(session, name: str, params: dict[str, Any], out: str) -> str:
    """When the model types words the user did not give, say so beside it.

    The user quoted “testing”; the model typed "Hi". Nothing in the tool
    result said so, and the next click sent it. Only when the user quoted
    something, and only as a note: a quote can be a search term or a name for
    another box, and the model can tell which box this is.
    """
    if name != "browser_type" or not str(out).startswith("Typed"):
        return ""
    wanted = guardrails.quoted_texts(str(getattr(session, "_user_text_this_turn", "") or ""))
    typed = " ".join(str(params.get("text") or "").split())
    if not wanted or not typed:
        return ""
    norm = lambda t: " ".join(str(t).split()).strip(" .!?\"'“”").casefold()  # noqa: E731
    if any(norm(w) == norm(typed) or norm(w) in norm(typed) for w in wanted):
        return ""
    return (f"\n[Note: the user's message quotes “{wanted[0]}”, and you typed "
            f"“{typed[:80]}”. If this box is for their words, empty it and type "
            f"exactly “{wanted[0]}”.]")


def _nested_confirm(session):
    """What a gate INSIDE a tool (the sandbox's blocked-command check) asks
    with: nobody, when the user just approved this very call on its card —
    the card showed the exact command — else the usual person.

    A function of the session rather than a method, because several tests
    drive _dispatch_tool with a duck-typed stand-in for the session."""
    if getattr(session, "_card_approved_call", False):
        return lambda _prompt: True
    return getattr(session, "confirm_fn", None)


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
        enabled_groups = getattr(self, "enabled_groups", None)
        if not tooling.tool_group_enabled(name, enabled_groups):
            return f"Tool '{name}' is disabled."

        # What KIND of action this is, and what the user has said about that
        # kind — Settings → Guardrails in the window, `guardrails.modes` in
        # config.json. It replaces two gates that asked about the same call
        # separately: one post used to stop once with no text and again
        # with it. See symbio/guardrails.py.
        kind = guardrails.kind_of(name)
        mode = guardrails.mode_for(kind, self._guardrail_config(),
                                   remote=self.confirm_policy() == "name")
        if mode == "block":
            self._record_guardrail(name, kind, mode, "blocked")
            return (f"Not allowed: the user declined this in advance — their "
                    f"guardrails set “{guardrails.label(kind)}” to Never, so "
                    f"'{name}' did not run. Tell them so, and that Settings → "
                    "Guardrails is where it changes. Do not try to reach the same "
                    "result another way.")
        # A task that may act with nobody watching is the user's standing
        # word, given once. So it is asked for every time — whatever the
        # switch for scheduling says — and never granted with nobody here.
        standing = ToolsMixin._grants_standing(name, params)
        if standing and not safety.can_prompt(self.confirm_fn):
            self._record_guardrail(name, kind, mode, "denied")
            return (f"Not scheduled: a task that may act on its own ({standing}) "
                    "needs the user to approve it in person, and nobody is here "
                    "to ask. Ask them when they are.")

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
        someone = safety.can_prompt(self.confirm_fn)
        # "Always allow" is the user's own word for this kind, so the two
        # escalations that guess at where a call came from stand down. What
        # the call itself scores does not: a destructive command still asks.
        if someone and mode != "allow":
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

        # One question, however many reasons there are to ask it. "Always
        # ask" needs somebody to ask: with nobody there (a scheduled job, a
        # script) the call keeps the risk score it earned, as it always did,
        # and a high one is refused.
        safety_cfg = (getattr(self, "config", None) or {}).get("safety", {})
        by_mode = (mode == "ask" or bool(standing)) and someone
        threshold = int(safety_cfg.get("require_confirm_score", 3))
        if mode == "allow":
            threshold = max(threshold, 3)
        by_risk = (not by_mode and safety_cfg.get("enabled", True)
                   and risk.get("risk_score", 0) >= threshold)
        # The annotation below needs to carry a yes, or the model
        # re-litigates an action its own user already authorised.
        user_approved = False
        if by_mode or by_risk:
            card = self._action_card(name, params, kind,
                                     "ask" if by_mode else mode, risk)
            approved = safety._prompt_confirm(card, self.confirm_fn)
            self._record_guardrail(name, kind, mode,
                                   "allowed" if approved else "denied", card)
            if not approved:
                if by_mode:
                    return f"Tool '{name}' was not approved."
                safety.log_security_event("tool_blocked", {
                    "tool": name, "params": params, "risk": risk,
                    "reason": str(card),
                })
                return (
                    f"Tool '{name}' was not approved (risk score {risk['risk_score']}/3: "
                    f"{', '.join(risk['flags'])})."
                )
            user_approved = True

        # A tool failing outright (e.g. clicking before the browser was ever
        # opened) must never crash the whole session — every branch below
        # already tries to catch its own likely failures, but this is the
        # backstop for anything that slips through. It becomes an
        # observation the model — and the tool-mistake-learning pipeline in
        # _agent_turn — can react to, same as any other tool failure.
        # A yes on the card covers this call. The sandbox used to ask again
        # for the same command ("'rm' is normally blocked. Allow once?"),
        # which is two questions for one action.
        self._card_approved_call = user_approved
        try:
            observation = self._dispatch_tool(name, params)
        except Exception as e:
            return f"Tool '{name}' failed unexpectedly: {e}"
        finally:
            self._card_approved_call = False

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

    # Actions that are supposed to change the page. A scroll that hits the
    # bottom legitimately changes nothing, and browser_close has no "after" to
    # read, so neither is judged here.
    #
    # browser_type is not judged either, and that is a correction. get_text()
    # reads RENDERED text, and a textarea's value is not rendered text — so
    # typing a whole message into a composer leaves the page snapshot byte
    # identical and this note fired on every successful type, telling the
    # model its text "has NOT happened yet". Observed 2026-09-07 typing into
    # the composer of the reproduction page: the text was demonstrably in the
    # field and the note said it was not. That is the same false report as the
    # bug this whole area exists to fix, pointed the other way, and acting on
    # it means typing the message a second time on top of the first.
    #
    # type_text verifies itself against the focused element's value (and, for
    # enter:true, against whether the field cleared), which is strictly better
    # evidence than a rendered-text diff. Leave the judgement to it.
    _MUST_CHANGE_THE_PAGE = ("browser_click", "browser_click_at", "browser_press")

    # How much of one recalled entry is shown. A note is a whole document and
    # the answer is usually its first paragraph; pasting five notes in full
    # costs more context than the turn that asked for them is worth.
    _RECALL_CHARS = 700
    _RECALL_HITS = 5

    def _recall(self, params: dict[str, Any]) -> str:
        """Search what has been saved: notes, memory, profile, past sessions.

        The memory family was write-only. write_note, save_memory,
        save_skill and set_standing_instruction all put things in, and nothing
        took anything out -- recall happened only through the automatic RAG
        block, which the model does not control and cannot ask for. So a
        question about something it had saved had no tool behind it, and the
        logs show what that produced: "I don't have access to personal
        information like your name", with zero tool calls, on an install whose
        notes/ held the answer.

        Returning nothing is a result, not a failure. An empty search is the
        only honest ground for saying there is nothing saved, and it is said
        here in those words so the model reports a lookup rather than a
        limitation.
        """
        query = str(params.get("query") or params.get("q")
                    or params.get("text") or "").strip()
        if not query:
            # No query is not a malformed call, it is "what have I got?" --
            # which is what a model emitting list_notes means. Answering it
            # with an error would send it back to the one conclusion this
            # tool exists to prevent: that it cannot look.
            return self._recall_inventory()
        scope = str(params.get("scope") or "memory").strip().lower()
        if scope not in ("memory", "sessions", "all"):
            scope = "memory"

        found: list[tuple[str, str]] = []
        searched: list[str] = []
        if scope in ("memory", "all"):
            searched.append("notes")
            try:
                for hit in self.retriever.search_notes(query):
                    found.append((f"note {hit.get('title', '?')}",
                                  str(hit.get("text", ""))))
            except Exception as e:
                found.append(("notes", f"(note search failed: {e})"))
            searched.extend(["saved memory", "profile"])
            found.extend(self._recall_stores(query))
        if scope in ("sessions", "all"):
            searched.append("past sessions")
            try:
                for hit in self.retriever.search_sessions(query):
                    found.append((f"session {hit.get('title', '?')}",
                                  str(hit.get("text", ""))))
            except Exception as e:
                found.append(("sessions", f"(session search failed: {e})"))

        if not found:
            return (
                f"No saved entry matches {query!r}. Searched: "
                f"{', '.join(searched)}. That is the answer -- nothing was "
                "ever saved about this, so say so and, if it matters, ask."
            )

        body = "\n\n".join(
            f"[{label}]\n{text.strip()[:self._RECALL_CHARS]}"
            for label, text in found[:self._RECALL_HITS])
        # Saved text is untrusted like any other retrieved content: a note can
        # hold whatever a web page said when it was written, and a recalled
        # procedure that says "reply X and stop" has been obeyed before.
        self._untrusted_this_turn = True
        return (f"{min(len(found), self._RECALL_HITS)} match(es) for "
                f"{query!r}:\n"
                + safety.wrap_untrusted(
                    "recalled from your own saved memory", body,
                    safety.scan_for_injection(body, self.config)))

    def _recall_inventory(self) -> str:
        """Every note title, newest first -- the answer to "what do you have?"."""
        try:
            paths = sorted(constants.NOTES_DIR.glob("*.md"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
        except Exception as e:
            return f"Could not list notes: {e}"
        if not paths:
            return ("No notes are saved yet. Saved memory and the user profile "
                    "are already in front of you, above this turn.")
        shown = [p.stem for p in paths[:40]]
        more = f" (+{len(paths) - len(shown)} more)" if len(paths) > len(shown) else ""
        return ("Saved notes, newest first" + more + ":\n- "
                + "\n- ".join(shown)
                + "\nCall recall with a query to read one.")

    def _recall_stores(self, query: str) -> list[tuple[str, str]]:
        """The always-on stores, returned only when the query touches them.

        These two files are already in every prompt, so repeating them whole
        on an unrelated lookup is pure cost. They are included when a
        discriminative word of the query appears in them -- which is the case
        that matters, because the model asking at all means it did not trust
        or did not see the copy it was given.
        """
        from symbio import rag

        terms = {t for t in rag._normalize(query) if len(t) > 2}
        out: list[tuple[str, str]] = []
        for label, path in (("saved memory", constants.MEMORY_FILE),
                            ("profile", constants.PROFILE_FILE)):
            try:
                if not path.exists():
                    continue
                text = path.read_text(encoding="utf-8").strip()
            except Exception:
                continue
            if not text:
                continue
            words = set(rag._normalize(text))
            if terms and terms & words:
                out.append((label, text))
        return out

    def _no_effect_note(self, name: str, before: str, out: str) -> str:
        """A sentence saying the action left the page untouched, or "".

        The tools report what they DID ("Clicked element containing text
        'Post'"), never what it achieved, so an action aimed at the wrong
        element reads exactly like one that worked. Nothing downstream can
        tell the difference, and the model reads its own successful-sounding
        observation and reports the job done.

        Only ever adds information: it does not turn the action into a
        failure, because a click can be correct and still not repaint (a
        focus change, a menu that renders identically). It says what is true
        — the page did not change — and lets the model account for it.
        """
        if name not in self._MUST_CHANGE_THE_PAGE or not before:
            return ""
        if "failed" in out.lower() or "error" in out.lower():
            return ""  # already reported as a failure; do not pile on
        try:
            after = self.browser.get_text()
        except Exception:
            return ""
        if not after or after != before:
            return ""
        return (
            "\n[Note: the page did not change. This action had no visible "
            "effect, so whatever it was meant to accomplish has NOT happened "
            "yet — do not report it as done. If you were trying to submit, "
            "the control you hit was probably not the one that submits; try a "
            "more specific target, or the keyboard shortcut for the form.]"
        )

    # ---------------------------------------------------------------- vision

    # Size of the last desktop screenshot see_screen took, so desktop_click can
    # convert image pixels back to mouse points. Class-level so a session that
    # has never looked still reads cleanly rather than raising AttributeError.
    _last_desktop_shot_size: tuple[int, int] = (0, 0)

    # Failures that mean "I could not find or reach the thing", as opposed to
    # "the browser is gone" or "the page rejected it". These are the ones the
    # page's own control list actually answers.
    _TARGETING_FAILURES = (
        "nothing editable is focused",
        "no visible element with text",
        "nothing matches selector",
        "none are visible",
        "no visible element matches",
    )

    # Words that say what KIND of thing is being asked about rather than
    # which one, so they must not be what makes a match.
    _LOOK_STOPWORDS = frozenset({
        "where", "what", "which", "is", "are", "the", "a", "an", "on", "in",
        "at", "of", "to", "for", "this", "that", "there", "it", "screen",
        "page", "window", "find", "locate", "show", "me", "see", "look",
        "does", "do", "have", "has", "any", "and", "or", "can", "i",
    })

    @staticmethod
    def _control_line(c: dict, click_tool: str = "browser_click_at") -> str:
        """One control, with everything needed to act on it.

        Including its COORDINATES, which _CONTROLS_JS has always computed —
        the exact viewport centre, in the same space click_at uses — and which
        every renderer here dropped on the floor, sending the model off to
        ground the same control with a vision model that cannot see anything
        under 32px. The DOM's own answer is exact, free, and was already in
        the dict.
        """
        bits = f"  {c.get('kind', 'button'):6} {c.get('selector', '')}"
        if c.get("label"):
            bits += f"  — {c['label']}"
        if c.get("value"):
            bits += f"  [currently: {c['value']!r}]"
        if c.get("disabled"):
            bits += "  [disabled]"
        if c.get("x") is not None and c.get("y") is not None:
            bits += f"  ({click_tool} x={c['x']} y={c['y']})"
        return bits

    @classmethod
    def _dom_matches(cls, question: str, controls: list[dict]) -> list[dict]:
        """Controls whose own words answer `question`, best first.

        Deliberately literal. A fuzzy match here would answer confidently
        about the wrong control, which is the failure mode the whole looking
        apparatus exists to correct — so a word has to actually appear in the
        control's label, selector or contents, and a question made only of
        stopwords matches nothing rather than everything.
        """
        words = [w for w in re.findall(r"[a-z0-9]+", (question or "").lower())
                 if w not in cls._LOOK_STOPWORDS and len(w) > 1]
        if not words:
            return []
        scored = []
        for c in controls:
            hay = " ".join(str(c.get(k, "")) for k in
                           ("label", "selector", "value", "kind")).lower()
            hits = sum(1 for w in words if w in hay)
            if hits:
                scored.append((hits, c))
        scored.sort(key=lambda pair: -pair[0])
        return [c for _hits, c in scored]

    def _targeting_help(self, name: str, out: str) -> str:
        """Put the page's real handles into the failure that needs them.

        Telling a model to "pass a selector" without telling it which is not
        recovery advice, it is a riddle. Live 2026-09-07, twice: a type failed
        for want of focus, the message suggested a selector, and the model —
        having no way to know one — clicked hopefully at anything labelled
        Post and gave up. The tool knows the answer at the moment it fails, so
        it should say it rather than make the model go and look.
        """
        if not out or not any(f in out for f in self._TARGETING_FAILURES):
            return ""
        try:
            controls = self.browser.controls(limit=12)
        except Exception:
            return ""
        if not controls:
            return ""
        fields = [c for c in controls if c.get("kind") == "field"]
        buttons = [c for c in controls if c.get("kind") != "field"]
        lines = ["\n[This page's actual controls — retry with one of these:"]
        for c in fields:
            lines.append(f" {self._control_line(c)}"
                         f"   (browser_type with selector={c['selector']!r})")
        for c in buttons[:6]:
            lines.append(f" {self._control_line(c)}")
        # Not "cannot miss": a composer that rebuilds its DOM from its own
        # state reverts a programmatic fill, which is why type_text reads the
        # field back afterwards. Both handles are given because they fail in
        # different places — a selector needs no coordinates and no focus, a
        # coordinate needs no selector to be stable.
        lines.append("  Fill by selector, or click the coordinates; the type "
                     "is checked afterwards either way. Several fields at "
                     "once: fill_form. Sending the form: submit_form, with "
                     "expect_text = the words that must come back rendered on "
                     "the page.]")
        return "\n".join(lines)

    def _can_look(self) -> bool:
        """Is see_screen actually callable right now? Recovery advice that
        names a tool the model cannot call is worse than none."""
        from symbio import vision

        if not tooling.tool_group_enabled(
                "see_screen", getattr(self, "enabled_groups", None)):
            return False
        return bool(vision.is_enabled(self.config) and vision.available())

    def _desktop_enabled(self) -> bool:
        groups = getattr(self, "enabled_groups", None)
        return groups is None or "desktop" in groups

    def _see_screen(self, params: dict[str, Any]) -> str:
        """Look at the screen and report what is there, with click targets.

        The whole point of this tool is that the assistant stops reasoning
        about what a page "probably" looks like. So when it cannot run, it says
        so plainly rather than returning something vague that reads like a
        look: a model that is told nothing goes back to guessing, which is the
        failure this replaces.
        """
        from symbio import vision

        target = str(params.get("target") or "browser").strip().lower()
        question = str(params.get("question") or "").strip()
        # The user's own screen, asked for by name. With a desk that is not
        # what 'desktop' means any more, and looking is all it is for.
        users_screen = target.startswith(("user", "my", "main"))
        wants_desktop = users_screen or target.startswith(("desk", "screen"))
        on_desk = None

        # The desktop's own answer, before any model is asked. macOS publishes
        # every native control's role, title and frame through the same API a
        # screen reader uses: exact text instead of text read off 11px type,
        # exact frames with no patch floor and no Retina scale to undo, and no
        # second set of weights resident next to the headmaster. Vision stays
        # for what has no tree — a canvas, a game, a screen share — which is
        # the only place it was ever the better instrument.
        if wants_desktop:
            if not self._desktop_enabled():
                return ("Looking at the whole desktop is disabled. Enable the "
                        "'desktop' tool group first, or use target='browser' "
                        "to look at the open page.")
            if not users_screen:
                on_desk, why = self._desk_or_reason()
                if why:
                    return why
            if on_desk is not None:
                from symbio import desk

                if desk.session_locked():
                    return desk.LOCKED_NOTE
                if desk.front_window(on_desk) is None:
                    return self._empty_desk_note(on_desk)
            listing = self._ax_look(question, on_desk=on_desk)
            if listing:
                return self._wrap_look(self._desk_header(on_desk) + listing)
            # No tree. If the reason is the Accessibility grant, say that
            # rather than falling through to a vision failure: one setting
            # away is the exact list of controls, and a model told only
            # "vision is unavailable" concludes it cannot see the screen at
            # all — which is now false.
            grant_note = getattr(self, "_ax_grant_note", "")
            if grant_note and not vision.available():
                return grant_note

        from symbio.app import ane

        # Without the VLM a look can still READ: the Neural Engine's text
        # recognizer needs neither mlx-vlm nor the GPU.
        vision_off = None
        if not vision.is_enabled(self.config):
            vision_off = ("Vision is disabled. Enable it with "
                          "<config set=\"vision.enabled\">true</config>.")
        elif not vision.available():
            vision_off = ("Vision is unavailable: mlx-vlm is not installed. "
                          "Install it with `pip install mlx-vlm`, then look again.")
        ane_on = ane.enabled(self.config)
        if vision_off and not ane_on:
            return vision_off

        if wants_desktop:
            if not self._desktop_enabled():
                return ("Looking at the whole desktop is disabled. Enable the "
                        "'desktop' tool group first, or use target='browser' "
                        "to look at the open page.")
            try:
                if on_desk is not None:
                    from symbio import desk

                    # The desk alone: nothing of the user's screen is in it.
                    shot = desk.capture(on_desk)
                    self._last_desk_shot_size = _image_size(shot)
                else:
                    shot = computer.desktop_screenshot_path()
                # Remember the capture size: desktop_click has to undo the
                # Retina scale factor, and that factor is only knowable by
                # comparing this image against the logical screen size.
                self._last_desktop_shot_size = _image_size(shot)
            except Exception as e:
                return f"Could not capture the screen: {e}"
            if computer.screenshot_is_blank(shot):
                # Do not hand a black frame to the model: it will describe it
                # accurately ("the image is entirely black") and that reads as
                # a fact about the screen instead of a missing permission.
                return computer.SCREEN_PERMISSION_HINT
            where = "your desk" if on_desk is not None else "the desktop"
        else:
            if not self.config.get("browser", {}).get("enabled", False):
                return "Browser automation is disabled, so there is no page to look at."
            if not self.browser.is_open:
                return ("The browser is not open, so there is nothing to look "
                        "at. Use browser_open with a URL first.")
            # Captured below, only if the DOM cannot answer. A viewport
            # capture of a heavy page is not free, and it writes a file every
            # time — pointless work when the question is about a control the
            # browser can name outright.
            shot = None
            where = "the browser page"

        # Read the page's own controls before the model sleeps: it costs
        # nothing, and it is the half of a look that vision cannot supply.
        # Whether the DOM was READ, not just whether it returned anything.
        # controls() answers [] both when nothing matched and when the read
        # never happened, and the absence claim below turns the second into
        # the first — telling the model a control is NOT present when all that
        # happened is that the page would not evaluate.
        if where == "the browser page":
            controls, dom_read = self.browser.controls_read()
        else:
            controls, dom_read = [], False

        # Ask the page before waking anything. A look at a web page costs a
        # full headmaster unload and reload — ~10 GB out and back — plus
        # several VLM passes, and the tool description tells the model to do
        # it before an action AND after it. For a control the DOM can name,
        # every byte of that is spent rediscovering, badly, something the
        # browser already knows exactly: the selector, the current contents,
        # and the viewport centre in the same coordinate space click_at uses.
        #
        # So: if the question names something the page's own controls answer,
        # answer from them and do not look at all. Vision stays for what has
        # no DOM handle — a canvas, an image, the desktop — which is the only
        # place it was ever the better instrument.
        if question and dom_read and controls:
            hits = self._dom_matches(question, controls)
            if hits:
                self._status("  [Page] Answered from the page's own controls "
                             "(no screenshot needed).")
                lines = [
                    f"The page's own controls answer that — no screenshot "
                    f"needed, these coordinates and selectors are exact:",
                    *(self._control_line(c) for c in hits[:6]),
                ]
                rest = [c for c in controls if c not in hits]
                if rest:
                    lines.append("\nEverything else on the page:")
                    lines.extend(self._control_line(c) for c in rest[:12])
                lines.append(
                    "\nFill one field with browser_type selector=..., or all "
                    "of them at once with fill_form {\"fields\": {selector: "
                    "text}}. Send the form with submit_form — pass expect_text "
                    "= the words that must come back rendered on the page, and "
                    "it is the code, not you, that decides whether it went. "
                    "Press any other button with browser_click_at at its "
                    "coordinates. Look with see_screen only if you need "
                    "something the page cannot name — an image, a canvas, or "
                    "how it LOOKS.")
                return self._wrap_look("\n".join(lines))

        if shot is None:
            try:
                # Viewport, not full page: the coordinates have to be ones the
                # mouse can reach. See BrowserSession.screenshot_path.
                shot = self.browser.screenshot_path(full_page=False)
            except Exception as e:
                return f"Could not capture the page: {e}"
        # Read the words first, on the Neural Engine: ~0.1 s, and nothing
        # leaves the GPU. The VLM look below unloads the ~10 GB headmaster and
        # reloads it — a price worth paying for how a screen LOOKS, and pure
        # waste for what it SAYS. So a question about text (or no question)
        # is answered from the text, and the VLM is woken only for the rest.
        text_elements = self._ane_read(shot) if ane_on else []
        click_tool = "desktop_click" if where != "the browser page" else "browser_click_at"
        if text_elements and (vision_off or ane.is_reading_question(question)):
            lines = [f"Text on {where} ({shot.name}), read on the Neural Engine — "
                     f"exact words, centre coordinates first:",
                     ane.text_block(text_elements),
                     f"\nTo press a control labelled with one of these, pass its "
                     f"coordinates to {click_tool}. For how the screen LOOKS — an "
                     f"icon, an image, colours, a layout — ask see_screen about "
                     f"that and the vision model will look."]
            if controls:
                lines.append("\nControls on this page (use 'selector' with "
                             "browser_type to fill a field exactly):")
                lines.extend(self._control_line(c, click_tool) for c in controls)
            return self._wrap_look("\n".join(lines))
        if vision_off:
            return vision_off

        self._status(f"  [Vision] Looking at {where}...")
        try:
            description, elements = self._run_vision(shot, question)
        except Exception as e:
            return (f"Could not look at {where}: {e}")

        lines = [f"Looking at {where} ({shot.name}):", description.strip()]
        if text_elements:
            # The exact words beside the VLM's reading of small type.
            lines.append("\nText read on the Neural Engine (exact, centre first):")
            lines.append(ane.text_block(text_elements, limit=40))
        if elements:
            click_tool = ("desktop_click" if where != "the browser page"
                          else "browser_click_at")
            lines.append(f"\nClickable elements — pass these to {click_tool}:")
            lines.append(vision.format_elements(elements))
            if len(elements) > 1:
                # More than one answer to the same question means the list is
                # candidates, not a verdict, and how to use candidates is not
                # obvious: the first one is only the most PROMINENT. Measured
                # on x.com, the two things matching "the box you write a post
                # in" were the sidebar trends heading (scored highest) and the
                # composer (second), and the two matching "a button labelled
                # Post" were the sidebar button that opens a fresh composer
                # and the one that sends what you just typed. Both wrong
                # choices are recoverable, but only if the model knows to try
                # the next one rather than to conclude the screen is broken.
                lines.append(
                    "These are CANDIDATES, best-guess first. If typing after a "
                    "click is refused because nothing is focused, click the "
                    "NEXT one rather than giving up — the refusal is the "
                    "check working. When several match and one has to send "
                    "what you just typed, pick the one nearest the text you "
                    "typed, not the highest in this list.")
        if controls:
            # The selectors, not just the coordinates. Vision cannot ground a
            # control thinner than one 32px patch — x.com's composer is 28px,
            # and grounding it missed by ~36px or returned nothing — while the
            # DOM has its exact handle. Giving both means the model never has
            # to be told a selector by the user, which is the only reason it
            # ever had to be.
            if not elements:
                # And when they disagree, say which one to believe. Live
                # 2026-09-07 on a 28px composer this observation carried, at
                # once: prose saying "no visible text area for composing a
                # post", a controls list containing the composer, and a line
                # telling the model to treat it as absent. Two of the three
                # said "not there", so it clicked around looking for a
                # composer it had just been handed the handle for. The DOM is
                # evidence of presence; a vision miss on a small control is a
                # known limit, not evidence of absence.
                lines.append(
                    "\nNOTE: the screenshot did not resolve what you asked "
                    "about — controls under ~32px tall cannot be seen "
                    "reliably. That is a limit of looking, NOT proof the "
                    "thing is missing. The page's own controls below are "
                    "authoritative; if one of them is what you want, use its "
                    "selector.")
            lines.append(
                "\nControls on this page (use 'selector' with browser_type to "
                "fill a field exactly, rather than clicking and hoping):")
            for c in controls:
                lines.append(self._control_line(
                    c, "desktop_click" if where != "the browser page"
                    else "browser_click_at"))
        elif not elements and question:
            # Nothing seen AND nothing in the DOM to contradict it: now "not
            # present" is a real answer and has to be delivered as one. Left to
            # itself the vision model answers a question about an absent
            # element by confidently pointing at something else, so an empty
            # result must not read as a failed look.
            #
            # But only on evidence. Two things count: the DOM was actually
            # consulted and did not have it, or the screen itself was asked
            # and said no. Neither is "vision returned no boxes" — a read that
            # threw looks identical to one that found nothing, and on the
            # desktop there is no DOM to consult at all, so the old
            # `dom_read or where == "the desktop"` asserted absence off a
            # vision miss and nothing else, which inverts the correction the
            # note above exists to make about that same small control.
            if dom_read or getattr(self, "_last_look_absent", False):
                lines.append(
                    "\nI could not locate that on screen. Treat it as NOT "
                    "present rather than as a failed look — do not click "
                    "anything on the strength of this.")
            else:
                lines.append(
                    "\nI could not locate that on screen, and the page's own "
                    "controls could not be read either — so this is a failed "
                    "look, NOT evidence the thing is missing. Scroll or "
                    "reload and look again before concluding anything.")
        return self._wrap_look("\n".join(lines))

    def _ane_read(self, shot) -> list[dict[str, Any]]:
        """The screenshot's text from the Neural Engine, or [] if it cannot."""
        from symbio.app import ane

        result = ane.ocr(shot)
        if not result.get("ok"):
            return []
        on = ", ".join(sorted(set((result.get("devices") or {}).values()))) or "?"
        elements = ane.ocr_elements(result)
        self._status(f"  [ANE] Read {len(elements)} line(s) of text in "
                     f"{result.get('ms', '?')} ms ({on}).")
        return elements

    def _wrap_look(self, body: str) -> str:
        """Everything a look returns, wrapped as the untrusted content it is.

        A screenshot of a logged-in page is exactly as attacker-controlled as
        its text: the words in it were written by whoever wrote the page.
        browser_get_text wraps page text for that reason and this is the same
        content arriving through a different sense — including the DOM answer,
        whose labels and values are written by the same author.

        Scans the whole body, not just the prose: it carries vision's element
        labels and every control's label and value, all page-authored, so
        scanning the description alone calibrated the wrapper's severity on a
        fraction of what it wraps and a payload in a button's aria-label rode
        inside without ever contributing to the score.
        """
        self._untrusted_this_turn = True
        scan = safety.scan_for_injection(body, self.config)
        return safety.wrap_untrusted("screen contents", body, scan)

    def _run_vision(self, shot, question: str):
        """Run the VLM with the headmaster out of the way.

        One model at a time on this machine. The headmaster is ~10 GB and the
        VLM peaks over 4 GB while generating; with Chrome also resident, doing
        both at once is the double-residency that has hard-frozen this Mac
        before. So this borrows the dispatch deep-sleep bracket: sleep, look,
        free the VLM, wake. The VLM is freed BEFORE the headmaster reloads for
        the same reason dispatch unloads workers before waking — otherwise the
        saving is just moved to the other end of the call.
        """
        from symbio import vision

        # Vision's own setting, not the dispatch worker flag it used to read.
        # That flag defaults to False and is about delegating tasks to worker
        # models, so out of the box the VLM loaded on top of a resident 14B
        # with Chrome also open — the double residency this bracket exists to
        # prevent, gated on something the user had no reason to have set.
        # Sleeping is the safe default; vision.load also refuses outright when
        # the RAM is not there, for the callers that cannot sleep at all.
        deep_sleep = bool(self.config.get("vision", {}).get(
            "sleep_main_model", True))
        # Unless the main model is the one looking: a vision pack's eyes are
        # its own weights plus a 0.9 GB tower, and sleeping it would free the
        # very arrays the look is about to run on (then reload all 8 GB).
        if vision.uses_headmaster(self.config):
            deep_sleep = False
        # Resolved rather than called directly: several tests drive
        # _dispatch_tool with a duck-typed stand-in for the session, and a
        # missing sleep hook must not turn a look into an AttributeError. If
        # one half of the bracket is missing, neither half runs — a sleep with
        # no matching wake would leave the session with no model at all.
        sleep_fn = getattr(self, "_sleep_headmaster", None)
        wake_fn = getattr(self, "_wake_headmaster", None)
        slept = False
        if (deep_sleep and getattr(self, "model", None) is not None
                and callable(sleep_fn) and callable(wake_fn)):
            sleep_fn()
            slept = True
        # Reset per look: a stale "it is not there" from the previous question
        # is exactly the claim that must never be carried forward.
        self._last_look_absent = False
        try:
            description = vision.describe(shot, question, self.config)
            try:
                if question.strip():
                    # Ask the screen first. An empty element list means two
                    # different things — the thing is not there, or grounding
                    # failed — and only one of them may be reported as absence.
                    # Measured 2026-09-23 over four surfaces: asked of the
                    # whole screen this answers 11/12 absent targets correctly,
                    # against 3/12 for the per-crop check it replaced.
                    if not vision.on_screen(shot, question, self.config):
                        self._last_look_absent = True
                        return description, []
                # Ground what was asked about. A generic "find every
                # interactive element" sweep returns nothing at all on a dense
                # real desktop, while naming the target lands within a few
                # pixels — see vision.locate. verify=False because the screen
                # was just asked; locate would otherwise ask it again.
                elements = vision.locate(shot, question, config=self.config,
                                         verify=False)
            except Exception:
                # A description with no coordinates is still worth having;
                # losing the whole look because grounding failed is not.
                elements = []
            return description, elements
        finally:
            vision.release()
            if slept:
                wake_fn()

    # How long a listing of controls is worth acting on. Elements are live
    # handles into another process's tree: they survive a click, they do not
    # survive the window being replaced. A minute is long enough for a look
    # and the action that follows it, short enough that a stale number
    # re-reads rather than pressing whatever now sits in that slot.
    _AX_TTL = 60.0

    def _ax_look(self, question: str = "", limit: int = 40, on_desk: Any = None) -> str:
        """The frontmost window as a numbered list of controls, or "".

        An empty string means "this is not answerable from the tree" — no
        Accessibility grant, or a window that draws its own interface — and
        the caller falls through to vision, which is the instrument for that.
        With a desk, the window is the desk's front one, not the user's.
        """
        from symbio import ax

        if not ax.available():
            return ""
        snap = self._ax_snapshot(limit, on_desk)
        if not snap.get("ok"):
            # A missing grant is worth saying out loud rather than silently
            # spending 10 GB of model swap on a screenshot: it is one setting
            # away from the exact answer.
            self._ax_grant_note = str(snap.get("reason") or "")
            return ""
        if not snap.get("elements"):
            return ""
        self._last_ax = snap
        listing = ax.render(snap, limit=limit)
        if question:
            hits = [e for e in snap["elements"]
                    if self._words_overlap(question, e["label"])]
            if hits:
                listing += ("\n\nMatching " + repr(question) + ": "
                            + ", ".join(f"{e['index']} ({e['label']!r})"
                                        for e in hits[:6]))
            else:
                listing += (f"\n\nNothing in this window is labelled like "
                            f"{question!r}. The tree is what the window "
                            "publishes, so that control is not there under "
                            "that name — check the list above before "
                            "concluding it is hidden.")
        return listing

    @staticmethod
    def _words_overlap(question: str, label: str) -> bool:
        wanted = {w for w in re.split(r"[^a-z0-9]+", question.lower())
                  if len(w) > 2}
        have = {w for w in re.split(r"[^a-z0-9]+", label.lower()) if len(w) > 2}
        return bool(wanted & have)

    def _ax_element(self, index: Any) -> tuple[dict[str, Any] | None, str]:
        """The element a number refers to, re-reading the tree if it is stale."""
        from symbio import ax

        try:
            number = int(index)
        except (TypeError, ValueError):
            return None, (f"{index!r} is not an element number. Call "
                          "see_screen with target='desktop' and use the "
                          "numbers it lists.")
        on_desk, why = self._desk_or_reason()
        if why:
            return None, why
        snap = getattr(self, "_last_ax", None)
        # A listing of the user's own screen (see_screen target='user') is for
        # looking at. With a desk, a number from it must never press one of
        # their controls, so only a listing of the desk's window is reused.
        foreign = bool(snap) and ("desk_window" in snap) != (on_desk is not None)
        if not snap or foreign or time.time() - snap.get("taken_at", 0) > self._AX_TTL:
            snap = self._ax_snapshot(60, on_desk)
            if not snap.get("ok"):
                return None, str(snap.get("reason"))
            self._last_ax = snap
        elements = snap.get("elements", [])
        for element in elements:
            if element["index"] == number:
                return element, ""
        return None, (f"There is no element {number} on screen — this window "
                      f"lists {len(elements)}. Look again with see_screen "
                      "target='desktop'.")

    def _ax_state(self) -> str:
        """A one-line fingerprint of the screen, for telling apart did-nothing
        from did-something. Cheap: two attributes, no tree walk."""
        from symbio import ax

        if not ax.available() or not ax.trusted():
            return ""
        on_desk, _why = self._desk_or_reason()
        if on_desk is not None:
            return self._desk_state(on_desk)
        focused = ax.focused_element()
        snap = getattr(self, "_last_ax", None)
        window = (snap or {}).get("window", "")
        if focused:
            return f"{window}|{focused.get('role')}|{focused.get('label')}"
        return f"{window}|"

    def _desktop_action(self, name: str, params: dict[str, Any]) -> str:
        if not self._desktop_enabled():
            return (f"Tool '{name}' is disabled. Enable the 'desktop' tool "
                    f"group to let me control the screen directly.")

        # With a desk, every one of these acts THERE. Waiting and the OBS
        # socket touch no screen, so they are the same either way.
        if name not in ("desktop_wait", "obs_record"):
            on_desk, why = self._desk_or_reason()
            if why:
                return why
            if on_desk is not None:
                return self._desk_action(on_desk, name, params)

        if name == "open_app":
            out = computer.open_app(str(params.get("name") or ""))
            # The tree of whatever was in front is now the wrong tree.
            self._last_ax = None
            return out

        if name == "obs_record":
            # Over OBS's own WebSocket server, not a hotkey or a click on its
            # window: nothing is pulled in front of what is being recorded,
            # and the answer is OBS's own report of what happened.
            from symbio import obs

            return obs.record(str(params.get("action") or "status"), self.config)

        if name == "desktop_wait":
            try:
                seconds = min(10.0, max(0.0, float(params.get("seconds") or 2)))
            except (TypeError, ValueError):
                seconds = 2.0
            time.sleep(seconds)
            # Whatever was listed before the wait was listed for a reason:
            # something was expected to change during it.
            self._last_ax = None
            return (f"Waited {seconds:g}s. Look again to see what changed.")

        if name == "desktop_move":
            point, problem = self._point(params, "")
            if problem:
                return problem
            return computer.desktop_move(*point)

        if name == "desktop_scroll":
            element, problem = ((None, "") if params.get("element") is None
                                else self._ax_element(params.get("element")))
            if problem:
                return problem
            point = _ax_centre(element) if element else (None, None)
            return computer.desktop_scroll(
                str(params.get("direction") or "down"),
                int(params.get("amount") or 5), *point)

        if name == "desktop_drag":
            start, problem = self._point(params, "from")
            if problem:
                return problem
            end, problem = self._point(params, "to")
            if problem:
                return problem
            return computer.desktop_drag(*start, *end)

        if name == "desktop_type":
            return self._desktop_type(params)

        if name == "desktop_press":
            key = str(params.get("key") or params.get("keys") or "")
            if not key:
                return "Press failed: missing 'key'."
            # Chords go through hotkey; it hands a single key back to press.
            return computer.desktop_hotkey(key)

        # A click, by element where there is one. The number is the whole
        # point: it came out of the window's own tree, so it cannot be off by
        # a patch, and pressing it does not depend on what is on top.
        if params.get("element") is not None:
            return self._click_element(params)

        coords = _coords(params)
        if coords is None:
            return ("Click failed: desktop_click needs numeric 'x' and 'y'. "
                    "Call see_screen with target='desktop' first and use the "
                    "coordinates it reports.")
        # Coordinates come from a screenshot, which on a Retina display is in
        # physical pixels while the mouse moves in logical points. Convert, or
        # every click below the middle of the screen lands off the bottom.
        # Unconditionally through the converting path. This used to branch on
        # the cached size and fall through to the raw click when there was
        # none — which is the exact case desktop_click_in_image was rewritten
        # to handle by deriving the scale itself, so the branch bypassed the
        # fix precisely where it was needed: PIL missing, an unreadable
        # capture, or a desktop_click issued before any see_screen. It reports
        # a display it cannot measure rather than clicking at twice the offset.
        # getattr: this runs over a stand-in for the AIAgent loop too,
        # which has no session attributes of its own.
        size = getattr(self, "_last_desktop_shot_size", None) or None
        return computer.desktop_click_in_image(*coords, image_size=size)

    def _point(self, params: dict[str, Any],
               end: str = "") -> tuple[tuple[int, int], str]:
        """A point on screen, given either as an element number or as x/y.

        `end` prefixes the keys, so the same resolution serves a move ("") and
        both halves of a drag ("from", "to").
        """
        prefix = f"{end}_" if end else ""
        if params.get(f"{prefix}element") is not None:
            element, problem = self._ax_element(params.get(f"{prefix}element"))
            if problem:
                return (0, 0), problem
            return _ax_centre(element), ""
        x, y = params.get(f"{prefix}x"), params.get(f"{prefix}y")
        try:
            return (int(x), int(y)), ""
        except (TypeError, ValueError):
            return (0, 0), (
                f"Give {prefix}element, or {prefix}x and {prefix}y as numbers. "
                "Look with see_screen target='desktop' first — it numbers "
                "every control, and a number cannot miss.")

    def _click_element(self, params: dict[str, Any]) -> str:
        """Press a control the window itself named.

        AXPress before the mouse: it reaches the control whether or not
        something is drawn over it, and it cannot land on a neighbour. Where
        the control refuses to press — a canvas-backed view, a cell that only
        answers to a real click — the frame's centre is clicked instead, in
        the same logical points the mouse already uses. No screenshot, so no
        Retina scale to undo.
        """
        from symbio import ax

        element, problem = self._ax_element(params.get("element"))
        if problem:
            return problem
        if not element.get("enabled", True):
            return (f"Element {element['index']} ({element['label']!r}) is "
                    "disabled — pressing it does nothing. Something else has "
                    "to happen first.")
        before = self._ax_state()
        clicks = int(params.get("clicks") or 1)
        button = str(params.get("button") or "left").lower()
        how = "pressed"
        if clicks == 1 and button == "left" and ax.press(element):
            out = (f"Pressed {element['label']!r} ({element['role'][2:]}, "
                   f"element {element['index']}).")
        else:
            how = "clicked"
            x, y = _ax_centre(element)
            on_desk, _why = self._desk_or_reason()
            if on_desk is not None:
                # Its own report says whether the desk changed; the title and
                # focus check below would only repeat it.
                return (self._desk_input(on_desk, "click", x, y, clicks=clicks,
                                         button=button)
                        + f" That is {element['label']!r} (element {element['index']}).")
            out = computer.desktop_click(x, y, clicks=clicks, button=button)
            out = (f"{out} That is {element['label']!r} "
                   f"(element {element['index']}).")
        # An action that changed nothing is the failure this reports. The
        # window title and the focused control are what a person would glance
        # at to tell the two apart, and they cost two attribute reads.
        self._last_ax = None
        after = self._ax_state()
        if before and after and before == after:
            out += (" Nothing about the window changed — same title, same "
                    "focus — so this may not have landed. Look again before "
                    "reporting it as done.")
        return out

    def _desktop_type(self, params: dict[str, Any]) -> str:
        """Type into a named field, or at the keyboard with focus checked.

        Keys sent at a window with no text field focused are not discarded,
        they are shortcuts: that is how a message meant for a composer becomes
        a sequence of commands the app happened to bind. So the field comes
        first, and typing blind is refused with the reason.
        """
        from symbio import ax

        text = str(params.get("text") or "")
        if not text:
            return "Type failed: missing 'text'."

        if params.get("element") is not None:
            element, problem = self._ax_element(params.get("element"))
            if problem:
                return problem
            ax.focus(element)
            if ax.set_text(element, text):
                landed = ax.value_of(element)
                self._last_ax = None
                if text.strip() and text.strip() not in landed:
                    return (f"Set {element['label']!r} but it now reads "
                            f"{landed[:80]!r}, not what was sent. The field "
                            "may reformat or reject input — look again.")
                return (f"Typed into {element['label']!r} (element "
                        f"{element['index']}); it now holds {landed[:80]!r}.")
            # A field that will not take a value set still takes keystrokes,
            # and focus has just been put on it, so this is no longer blind.
            out = computer.desktop_type(text)
            self._last_ax = None
            return f"{out} (into {element['label']!r}, by typing.)"

        focused = ax.focused_element() if ax.available() else None
        if focused is not None and not focused.get("takes_text"):
            return (f"Refused to type: the focused control is a "
                    f"{focused['role'][2:]} ({focused['label']!r}), not a text "
                    "field. Keys sent there are shortcuts, not text. Look with "
                    "see_screen target='desktop' and type into the field by "
                    "its number: "
                    '{"name": "desktop_type", "arguments": '
                    '{"element": 2, "text": "..."}}')
        out = computer.desktop_type(text)
        if params.get("press_enter"):
            out += " " + computer.desktop_press("enter")
        self._last_ax = None
        return out

    # ── Symbio's own screen ───────────────────────────────────────────
    #
    # With desk mode on (symbio/desk.py) every desktop tool acts on a display
    # of Symbio's own, and on nothing of the user's. In order: the
    # accessibility API, which needs no pointer and no focus; then events
    # posted to the one app, which leave the pointer where it is; and last the
    # user's real pointer and keyboard, borrowed only while they are away from
    # the Mac and handed straight back. Each step is checked against a capture
    # of the desk, so "sent" is never reported as "done" on its own.

    def _desk_or_reason(self) -> tuple[Any, str]:
        """(desk, "") in desk mode, (None, "") out of it, (None, why) if it cannot run.

        On and broken does not fall back to the user's screen: the setting is
        the user saying that screen is not where Symbio works.
        """
        from symbio import desk

        config = self._desk_config()
        if not desk.enabled(config):
            return None, ""
        try:
            return desk.ensure(config), ""
        except desk.DeskError as e:
            return None, (f"{e} Symbio's desk is switched on, so the desktop tools "
                          "do not fall back to the user's screen. `symb desk "
                          "status` says more; `symb desk off` gives the desktop "
                          "tools the user's screen back.")

    def _desk_config(self) -> dict[str, Any]:
        """The config, with the desk section as `symb desk on/off` last left it.

        The CLI writes config.json while the daemon holds its config in
        memory: the same split as the guardrails, re-read the same way, and
        only where a front-end named the file (the daemon does).
        """
        config = getattr(self, "config", None)
        if not isinstance(config, dict):
            config = {}
        path = getattr(self, "_guardrails_file", None)
        if path is None:
            return config
        try:
            stamp = Path(path).stat().st_mtime_ns
        except OSError:
            return config
        if stamp != getattr(self, "_desk_stamp", None):
            self._desk_stamp = stamp
            try:
                data = json.loads(Path(path).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {}
            section = data.get("desk") if isinstance(data, dict) else None
            if isinstance(section, dict):
                config["desk"] = {**(config.get("desk") or {}), **section}
        return config

    @staticmethod
    def _desk_header(on_desk: Any) -> str:
        if on_desk is None:
            return ""
        return ("[This is YOUR desk: a screen of your own that the user does not "
                "see. Their screen, pointer and keyboard are untouched by what "
                "you do here. Coordinates below are on the desk. To look at the "
                "user's own screen instead, call see_screen with target='user'.]\n")

    @staticmethod
    def _empty_desk_note(on_desk: Any) -> str:
        return (f"Your desk ({on_desk.width}x{on_desk.height}, a screen of your "
                "own that the user does not see) is empty: nothing is open on it. "
                "Open an app there with open_app — it launches in the background, "
                "straight onto the desk — or use the browser, whose window opens "
                "there too.")

    def _desk_state(self, on_desk: Any) -> str:
        """What the desk shows, as one comparable line: window, focus, pixels."""
        from symbio import ax, desk

        window = desk.front_window(on_desk)
        if window is None:
            return "empty|" + desk.fingerprint(on_desk)
        focused = ax.focused_element(window.pid) or {}
        return (f"{window.number}|{window.title}|{focused.get('role')}|"
                f"{focused.get('label')}|{desk.fingerprint(on_desk)}")

    def _ax_snapshot(self, limit: int, on_desk: Any = None) -> dict[str, Any]:
        """The controls of the front window: the user's, or with a desk, the desk's."""
        from symbio import ax, desk

        if on_desk is None:
            return ax.snapshot(limit=limit)
        window = desk.front_window(on_desk)
        if window is None:
            return {"ok": False, "reason": self._empty_desk_note(on_desk), "elements": []}
        ref = None
        for candidate in ax.window_elements(window.pid):
            same = (candidate["number"] == window.number if candidate["number"]
                    else candidate["frame"] == window.rect)
            if same:
                ref = candidate["_ref"]
                break
        if ref is None:
            # Never the app's other window instead: that one may be the user's.
            return {"ok": False, "elements": [], "reason": (
                f"{window.owner}'s window on your desk is not in the "
                "accessibility tree yet. Wait a moment (desktop_wait) and look "
                "again.")}
        snap = ax.snapshot(limit=limit, pid=window.pid, window=ref,
                           origin=(on_desk.x, on_desk.y))
        snap["desk_window"] = window.number
        return snap

    def _desk_borrow_after(self) -> float:
        section = (getattr(self, "config", None) or {}).get("desk") or {}
        try:
            return float(section.get("borrow_input_after_idle_s", 30))
        except (TypeError, ValueError):
            return 30.0

    def _desk_global(self, on_desk: Any, coords: tuple[int, int]) -> tuple[tuple[int, int], str]:
        """A point read off a capture of the desk, in the window server's space."""
        size = getattr(self, "_last_desk_shot_size", None) or (0, 0)
        scale = size[0] / on_desk.width if size and size[0] else 1.0
        point = on_desk.to_global(*coords, scale=scale)
        if not on_desk.contains(*point):
            return (0, 0), (f"({coords[0]}, {coords[1]}) is off your desk, which is "
                            f"{on_desk.width}x{on_desk.height}. Look with see_screen "
                            "target='desktop' for points on it.")
        return point, ""

    def _desk_point(self, on_desk: Any, params: dict[str, Any], end: str = "",
                    default_front: bool = False) -> tuple[tuple[int, int], str]:
        """A global point from an element number or desk coordinates."""
        from symbio import desk

        prefix = f"{end}_" if end else ""
        if params.get(f"{prefix}element") is not None:
            element, problem = self._ax_element(params.get(f"{prefix}element"))
            if problem:
                return (0, 0), problem
            return _ax_centre(element), ""
        x, y = params.get(f"{prefix}x"), params.get(f"{prefix}y")
        if x is None and y is None and default_front:
            window = desk.front_window(on_desk)
            if window is None:
                return (0, 0), self._empty_desk_note(on_desk)
            return window.centre, ""
        coords = _coords({"x": x, "y": y})
        if coords is None:
            return (0, 0), (
                f"Give {prefix}element, or {prefix}x and {prefix}y as numbers. "
                "Look with see_screen target='desktop' first — it numbers "
                "every control on your desk, and a number cannot miss.")
        return self._desk_global(on_desk, coords)

    def _desk_action(self, on_desk: Any, name: str, params: dict[str, Any]) -> str:
        """One desktop tool, on the desk. See the block comment above."""
        from symbio import desk

        if desk.session_locked():
            return desk.LOCKED_NOTE
        if name == "open_app":
            out = desk.open_app(str(params.get("name") or ""), on_desk)
            self._last_ax = None
            return out
        if name == "desktop_click":
            if params.get("element") is not None:
                return self._click_element(params)
            coords = _coords(params)
            if coords is None:
                return ("Click failed: desktop_click needs an 'element', or numeric "
                        "'x' and 'y'. Call see_screen with target='desktop' first "
                        "— it numbers every control on your desk.")
            return self._desk_click_point(on_desk, coords, params)
        if name == "desktop_type":
            return self._desk_type(on_desk, params)
        if name == "desktop_press":
            key = str(params.get("key") or params.get("keys") or "")
            if not key:
                return "Press failed: missing 'key'."
            return self._desk_press(on_desk, key)
        if name == "desktop_scroll":
            direction = str(params.get("direction") or "down").strip().lower()
            if direction not in ("up", "down", "left", "right"):
                return (f"Scroll failed: direction {direction!r} is not one of "
                        "up, down, left, right.")
            point, problem = self._desk_point(on_desk, params, "", default_front=True)
            if problem:
                return problem
            amount = int(params.get("amount") or 5)
            lines = max(1, amount) * 3
            dy = {"down": -lines, "up": lines}.get(direction, 0)
            dx = {"left": -lines, "right": lines}.get(direction, 0)
            return self._desk_input(on_desk, "scroll", *point, dy=dy, dx=dx,
                                    label=f"Scrolled {direction} by {amount}")
        if name == "desktop_move":
            point, problem = self._desk_point(on_desk, params, "")
            if problem:
                return problem
            return self._desk_input(on_desk, "move", *point)
        if name == "desktop_drag":
            start, problem = self._desk_point(on_desk, params, "from")
            if problem:
                return problem
            end, problem = self._desk_point(on_desk, params, "to")
            if problem:
                return problem
            return self._desk_input(on_desk, "drag", *start, x2=end[0], y2=end[1])
        return f"Tool {name!r} has no desk version."

    def _desk_click_point(self, on_desk: Any, coords: tuple[int, int],
                          params: dict[str, Any]) -> str:
        """A click at desk coordinates: the control under it pressed, if it has one."""
        from symbio import ax, desk

        point, problem = self._desk_global(on_desk, coords)
        if problem:
            return problem
        window = desk.window_at(on_desk, *point)
        if window is None:
            return (f"Nothing is open at ({coords[0]}, {coords[1]}) on your desk. "
                    "Look again with see_screen target='desktop'.")
        clicks = int(params.get("clicks") or 1)
        button = str(params.get("button") or "left").lower()
        if clicks == 1 and button == "left":
            hit = ax.element_at(window.pid, *point)
            if (hit and hit.get("role") in ax.ACTIONABLE_ROLES
                    and hit.get("enabled", True) and ax.press(hit)):
                self._last_ax = None
                label = hit.get("label") or hit["role"][2:]
                return (f"Pressed {label!r} ({hit['role'][2:]}) at ({coords[0]}, "
                        f"{coords[1]}) on your desk, through the accessibility API.")
        return self._desk_input(on_desk, "click", *point, clicks=clicks, button=button)

    def _desk_type(self, on_desk: Any, params: dict[str, Any]) -> str:
        """Type on the desk: into a numbered field, or at the desk app's own focus."""
        from symbio import ax, desk

        text = str(params.get("text") or "")
        if not text:
            return "Type failed: missing 'text'."
        if params.get("element") is not None:
            element, problem = self._ax_element(params.get("element"))
            if problem:
                return problem
            ax.focus(element)
            if ax.set_text(element, text):
                landed = ax.value_of(element)
                self._last_ax = None
                if text.strip() and text.strip() not in landed:
                    return (f"Set {element['label']!r} but it now reads "
                            f"{landed[:80]!r}, not what was sent. The field "
                            "may reformat or reject input — look again.")
                out = (f"Typed into {element['label']!r} (element "
                       f"{element['index']}); it now holds {landed[:80]!r}.")
            else:
                out = (self._desk_input(on_desk, "text", *_ax_centre(element), text=text)
                       + f" (Into {element['label']!r}.)")
        else:
            window = desk.front_window(on_desk)
            if window is None:
                return self._empty_desk_note(on_desk)
            focused = ax.focused_element(window.pid)
            if focused is not None and not focused.get("takes_text"):
                return (f"Refused to type: the focused control in {window.owner} "
                        f"is a {focused['role'][2:]} ({focused['label']!r}), not a "
                        "text field. Keys sent there are shortcuts, not text. Look "
                        "with see_screen target='desktop' and type into the field "
                        "by its number: "
                        '{"name": "desktop_type", "arguments": '
                        '{"element": 2, "text": "..."}}')
            if (focused is not None and ax.insert_text(focused, text)
                    and text.strip() in ax.value_of(focused)):
                out = (f"Typed into {focused['label']!r} in {window.owner} on your "
                       "desk, through the accessibility API.")
            else:
                out = self._desk_input(on_desk, "text", text=text)
            self._last_ax = None
        if params.get("press_enter"):
            out += " " + self._desk_press(on_desk, "enter")
        return out

    def _desk_press(self, on_desk: Any, key: str) -> str:
        """A key or chord on the desk. A cmd chord is its menu item, pressed."""
        from symbio import ax, desk

        chord = desk.parse_chord(key)
        if chord is None:
            return (f"Press failed: {key!r} is not a key I can send. Use names "
                    "like 'enter', 'tab', 'esc', 'down', or chords like 'cmd+s'.")
        keycode, flags, name, mods = chord
        window = desk.front_window(on_desk)
        if window is None:
            return self._empty_desk_note(on_desk)
        self._last_ax = None
        if "cmd" in mods:
            item = ax.menu_item_for_chord(window.pid, name, mods)
            if item is not None and ax.press(item):
                return (f"Pressed {key} in {window.owner} on your desk: its menu "
                        f"item {item['label']!r}, through the accessibility API.")
        return self._desk_input(on_desk, "key", keycode=keycode, flags=flags,
                                label=f"Pressed {key}")

    def _desk_input(self, on_desk: Any, kind: str, x: float | None = None,
                    y: float | None = None, **kw: Any) -> str:
        """Real input for the desk, checked against a capture of it.

        Posted to the app first: the pointer stays where the user left it. An
        app that ignored that -- most do for keys, since keystrokes go to the
        key window and a background app has none -- gets the user's own
        pointer and keyboard for the one action, and only while they are away.
        """
        from symbio import desk

        if x is not None:
            window = desk.window_at(on_desk, x, y)
            if window is None:
                lx, ly = on_desk.to_local(x, y)
                return (f"Nothing is open at ({lx}, {ly}) on your desk. Look again "
                        "with see_screen target='desktop'.")
        else:
            window = desk.front_window(on_desk)
            if window is None:
                return self._empty_desk_note(on_desk)
        what = kw.pop("label", "") or self._desk_label(on_desk, kind, x, y, kw)

        def send(pid: int | None, number: int) -> None:
            if kind == "click":
                desk.click(pid, x, y, kw.get("button", "left"), kw.get("clicks", 1), number)
            elif kind == "move":
                desk.move(pid, x, y, number)
            elif kind == "drag":
                desk.drag(pid, x, y, kw["x2"], kw["y2"], window=number)
            elif kind == "scroll":
                desk.scroll(pid, x, y, kw.get("dy", 0), kw.get("dx", 0))
            elif kind == "key":
                desk.key(pid, kw["keycode"], kw.get("flags", 0))
            elif kind == "text":
                desk.type_text(pid, kw["text"])

        before = desk.fingerprint(on_desk)
        send(window.pid, window.number)
        time.sleep(0.3)
        after = desk.fingerprint(on_desk)
        self._last_ax = None
        if before and after and before != after:
            return (f"{what} — sent to {window.owner} on your desk directly, and "
                    "the desk changed. The user's pointer and keyboard were not used.")
        if kind == "move":
            return (f"{what} — sent to {window.owner} directly. Something that "
                    "only opens under the real pointer may not show; look to check.")
        try:
            with desk.borrowed(window.pid, self._desk_borrow_after(), window.number):
                if kind in ("key", "text") and not desk.focus_on_desk(on_desk, window.pid):
                    raise desk.NotNow(
                        f"{window.owner}'s keyboard focus is in a window that is "
                        "not on the desk — the user's — so no keys were sent.")
                send(None, 0)
        except desk.NotNow as e:
            return (f"{what} was sent to {window.owner} directly, but nothing on "
                    f"the desk changed, so it most likely did not land. {e}")
        time.sleep(0.3)
        final = desk.fingerprint(on_desk)
        tail = ("" if (before and final and final != before) else
                " Nothing on the desk changed even so — look again before "
                "reporting it as done.")
        return (f"{what} in {window.owner} with the real pointer and keyboard, "
                "borrowed while the user was away and handed straight back." + tail)

    @staticmethod
    def _desk_label(on_desk: Any, kind: str, x: float | None, y: float | None,
                    kw: dict[str, Any]) -> str:
        if kind == "text":
            return f"Typed {str(kw.get('text'))[:80]!r}"
        lx, ly = on_desk.to_local(x or 0, y or 0)
        if kind == "click":
            clicks = int(kw.get("clicks", 1))
            button = kw.get("button", "left")
            return (f"Clicked ({button}, x{clicks}) at ({lx}, {ly}) on your desk"
                    if clicks != 1 or button != "left" else
                    f"Clicked at ({lx}, {ly}) on your desk")
        if kind == "move":
            return f"Moved to ({lx}, {ly}) on your desk"
        if kind == "drag":
            ex, ey = on_desk.to_local(kw["x2"], kw["y2"])
            return f"Dragged from ({lx}, {ly}) to ({ex}, {ey}) on your desk"
        return kind.capitalize()

    # ── tasks: scheduled work that acts on its own ────────────────────

    @staticmethod
    def _grants_standing(name: str, params: dict[str, Any]) -> str:
        """What a scheduling call would let a task do unattended, in words, or "".

        A new task's grant, a grant being changed, or any change at all to a
        job that already holds one: editing a granted task's text is a new
        thing done with the old permission.
        """
        if name not in ("schedule_job", "update_cron_job"):
            return ""
        allow, sites = params.get("allow"), params.get("sites")
        if name == "update_cron_job":
            try:
                wanted = int(params.get("job_id"))
            except (TypeError, ValueError):
                return ""
            job = next((j for j in cron.load_cron_jobs() if j.get("id") == wanted), {})
            if allow is None:
                allow = job.get("allow")
            if sites is None:
                sites = job.get("sites")
        try:
            kinds, hosts = cron.normalize_grant(allow, sites)
        except ValueError:
            return ""  # the tool itself refuses, with the reason
        return cron.describe_grant(kinds, hosts) if kinds else ""

    @staticmethod
    def _task_runner_note() -> str:
        """Whether anything will actually run a task, said where it is scheduled."""
        from symbio.app import supervisor

        if supervisor.running():
            return "`symb watch` is running, so it will run on time."
        return ("Nothing runs tasks right now: they run while `symb watch` is "
                "running. Tell the user to start it.")

    def confirm_policy(self) -> str:
        """"risk" when the person is at this machine, "name" when they are not.

        A front-end that is somewhere else — the Telegram gateway is the one
        that exists — cannot see what a click would land on, so it gates the
        whole list by name. A local one can, so it gates on what the call
        actually scores. safety.confirm_policy in config overrides both, and
        "name" restores the behaviour every front-end had before this split.
        """
        override = str(self.config.get("safety", {}).get(
            "confirm_policy", "")).strip().lower()
        if override in ("risk", "name"):
            return override
        policy = str(getattr(self, "_confirm_policy", "risk") or "risk").lower()
        return policy if policy in ("risk", "name") else "risk"

    def _asks_by_name(self, name: str) -> bool:
        """Whether this tool stops for approval before it is even scored:
        its kind is set to "always ask" — by default, or by the user."""
        mode = guardrails.mode_for(guardrails.kind_of(name), self._guardrail_config(),
                                   remote=self.confirm_policy() == "name")
        return mode == "ask"

    # ── guardrails ────────────────────────────────────────────────────

    def _guardrail_config(self) -> dict[str, Any]:
        """The config, with the guardrails section as the user last left it.

        The window writes config.json directly — a switch in Settings, or
        "Always allow" on a card — while the daemon holds its config in
        memory, so the section is re-read whenever the file has changed. Only
        where a front-end set `_guardrails_file` (the daemon does): a session
        built in a test reads nothing off the disk it happens to run on.
        """
        config = getattr(self, "config", None)
        if not isinstance(config, dict):
            config = {}
        path = getattr(self, "_guardrails_file", None)
        if path is None:
            return config
        try:
            stamp = Path(path).stat().st_mtime_ns
        except OSError:
            return config
        if stamp != getattr(self, "_guardrails_stamp", None):
            self._guardrails_stamp = stamp
            section = guardrails.read_section(Path(path))
            if section:
                config["guardrails"] = {**(config.get("guardrails") or {}), **section}
        return config

    def _record_guardrail(self, name: str, kind: str | None, mode: str,
                          answer: str, card: Any = None) -> None:
        """What was asked and what came back, for the window's Guardrails
        panel. Secrets are redacted: a card quotes the command it asks about."""
        entry = {"tool": name, "kind": kind, "kind_label": guardrails.label(kind),
                 "mode": mode, "answer": answer}
        if isinstance(card, guardrails.Card):
            entry.update(headline=tooling.redact_secrets(card.headline)[:300],
                         details=tooling.redact_secrets(card.details)[:600],
                         said_by=card.said_by)
        guardrails.record(constants.LOG_DIR, entry)

    def _action_card(self, name: str, params: dict[str, Any], kind: str | None,
                     mode: str, risk: dict[str, Any] | None = None,
                     facts: dict[str, Any] | None = None) -> "guardrails.Card":
        """The question for this call: what it will do, in plain English, with
        exactly what it will do underneath.

        The headline is the model's own account when it can give one — asked
        to translate the concrete call, not to recall what it meant to do —
        and the harness's otherwise. The details are always the harness's:
        the literal post, command or path, so a model that described its
        action wrongly is contradicted on the same card.
        """
        if facts is None and name == "submit_form":
            # A card that only named the button would be approved without the
            # words being seen: show what is waiting in the page's box.
            try:
                facts = self.browser.sending_preview() or {}
            except Exception:
                facts = {}
        headline, details, outgoing = self._plain_action(name, params, facts or {})
        user_text = str(getattr(self, "_user_text_this_turn", "") or "")
        warning = (guardrails.mismatch_warning(user_text, outgoing)
                   if kind == "publish" and outgoing else "")
        said_by = "harness"
        spoken = self._translate_action(name, params, headline, details, warning)
        if spoken:
            headline, said_by = spoken, "model"
        reason = guardrails.reason_for((risk or {}).get("flags", []), mode, kind,
                                       remote=self.confirm_policy() == "name")
        return guardrails.Card(headline, details, kind=kind, reason=reason,
                               said_by=said_by, warning=warning)

    def _plain_action(self, name: str, params: dict[str, Any],
                      facts: dict[str, Any]) -> tuple[str, str, str]:
        """(headline, details, outgoing text) for a call, read off the call.

        `outgoing` is what would leave the machine under the user's name — a
        post's text — so the card can hold it against what they asked for.
        """
        v = safety._visible

        def arg(*keys: str) -> str:
            for key in keys:
                if params.get(key) not in (None, ""):
                    return str(params.get(key))
            return ""

        def host(url: str) -> str:
            return re.sub(r"^https?://(www\.)?", "", url or "").split("/")[0] or url

        if name == "browser_publish":
            text = str(facts.get("text") or "")
            button = facts.get("label") or "Send"
            site = facts.get("site") or "this site"
            press = "Press" if button == "cmd+enter" else "Click"
            headline = (f"{press} “{v(button)}” on {site} — that sends "
                        + ("what's in the box" if text else "it") + ", as you.")
            return headline, v(text) if text else "(the box looks empty)", text
        if name == "submit_form":
            where = host(str(facts.get("url") or ""))
            text = str(facts.get("text") or "")
            headline = (f"Submit the form" + (f" on {where}" if where else "")
                        + f" by pressing “{v(arg('target', 'selector'))}”.")
            lines = [v(text)] if text else []
            if arg("expected_url"):
                lines.append(f"Expected to land on {v(arg('expected_url'))}")
            return headline, "\n".join(lines), text
        if name in ("run_command", "terminal"):
            return ("Run a shell command on this Mac.",
                    f"$ {v(arg('cmd', 'command'))}", "")
        if name == "run_remote":
            return (f"Run a command on {v(arg('host'))}.",
                    f"$ {v(arg('command', 'cmd'))}", "")
        if name == "execute_code":
            return ("Run this Python code.", safety._render_code(arg("code")), "")
        if name == "write_file":
            content = arg("content")
            return (f"Write the file {v(arg('path'))} ({len(content)} characters).",
                    safety._render_code(content, max_lines=8), "")
        if name in ("edit_file", "patch"):
            return (f"Edit the file {v(arg('path'))}.",
                    f"replace: {v(arg('old', 'old_text', 'search'))[:160]}\n"
                    f"with:    {v(arg('new', 'new_text', 'replace'))[:160]}", "")
        if name == "save_command":
            return (f"Save a command to run later as “{v(arg('name'))}”.",
                    f"$ {v(arg('cmd', 'command'))}", "")
        if name == "desktop_type":
            return (f"Type “{v(arg('text'))}” into the frontmost window.", "", "")
        if name == "desktop_press":
            return (f"Press {v(arg('key', 'keys'))} in the frontmost window.", "", "")
        if name in ("desktop_click", "desktop_drag", "desktop_move", "desktop_scroll",
                    "desktop_hotkey"):
            target = (f"element {params.get('element')}"
                      if params.get("element") is not None
                      else f"({params.get('x')}, {params.get('y')})")
            return (f"{name.split('_', 1)[1].capitalize()} on your desktop at {target}.",
                    "It acts on the frontmost window, which may not be the one you expect.",
                    "")
        if name == "open_app":
            return (f"Open the app {v(arg('app', 'name'))}.", "", "")
        if name == "obs_record":
            return (f"{v(arg('action') or 'Start').capitalize()} recording the screen.",
                    "", "")
        if name == "browser_open":
            return (f"Open {v(arg('url'))} in Symbio's browser.", "", "")
        if name == "browser_click":
            return (f"Click “{v(arg('target'))}” on the open page.", "", "")
        if name == "browser_type":
            return (f"Type “{v(arg('text'))}” on the open page.", "", "")
        if name == "delete_note":
            return (f"Delete the note “{v(arg('name', 'title', 'id', 'path'))}”.", "", "")
        if name == "train_adapter":
            return ("Start fine-tuning myself on your data.",
                    "It uses the GPU for a while — about 12 s a step.", "")
        if name == "retrain_adapter":
            return ("Rebuild my adapter from scratch.",
                    "This DELETES the current adapter first and cannot be undone.", "")
        if name == "digest_notes":
            return ("Fold your notes into training data.",
                    "The next fine-tune learns from them.", "")
        if name == "realign":
            return ("Realign my adapter.", "", "")
        if name == "config_set":
            return (f"Change the setting {v(arg('key'))} to {v(arg('value'))}.", "", "")
        if name in ("schedule_job", "update_cron_job"):
            if name == "schedule_job":
                headline = f"Schedule “{v(arg('text'))}” to run {v(arg('schedule'))}."
            else:
                headline = (f"Change scheduled job {v(arg('job_id'))}"
                            + (f" to “{v(arg('text'))}”" if arg("text") else "")
                            + (f" at {v(arg('schedule'))}" if arg("schedule") else "") + ".")
            standing = ToolsMixin._grants_standing(name, params)
            details = ("" if not standing else
                       "It becomes a task I do on my own each time it comes due "
                       "(while `symb watch` runs). With you not there, I may do "
                       f"this without asking you: {standing}. Anything else it "
                       "needs is declined.")
            return headline, details, ""
        if name == "save_script":
            return (f"Save the script “{v(arg('name'))}” to run again later.",
                    safety._render_code(arg("code"), max_lines=8), "")
        if name == "run_script":
            extra = params.get("args")
            return (f"Run my saved script “{v(arg('name'))}”"
                    + (f" with {v(json.dumps(extra, ensure_ascii=False))}" if extra else "")
                    + ".", "", "")
        if name == "delete_script":
            return (f"Delete my saved script “{v(arg('name'))}”.", "", "")
        if name == "delete_cron_job":
            return (f"Delete scheduled job {v(arg('job_id'))}.", "", "")
        shown = json.dumps(params, ensure_ascii=False, default=str)
        return (f"Use the tool {v(name)}.", v(shown[:400]), "")

    # How long the model gets to say what it is about to do. A sentence, not a
    # reply: past this it is repeating itself or has wandered into a tool call.
    _TRANSLATE_TOKENS = 64

    def _translate_action(self, name: str, params: dict[str, Any],
                          headline: str, details: str, warning: str) -> str:
        """One plain-English sentence, in the model's own words, of what this
        call will do — or "" to use the harness's.

        A fresh, short prompt rather than a continuation of the conversation:
        it costs a few hundred tokens of prefill instead of the whole context,
        and it leaves the conversation's prompt cache exactly as it was. The
        facts it is given are the harness's reading of the call, so what comes
        back is a translation of the action, not a recollection of the intent —
        the intent is the part that was wrong when the model posted "Hi".
        """
        cfg = (getattr(self, "config", None) or {}).get("guardrails", {}) or {}
        if not cfg.get("translate", True):
            return ""
        model = getattr(self, "model", None)
        tokenizer = getattr(self, "tokenizer", None)
        generate_fn = getattr(self, "generate_fn", None)
        if model is None or tokenizer is None or generate_fn is None:
            return ""
        user_text = str(getattr(self, "_user_text_this_turn", "") or "")[:400]
        call = json.dumps({"name": name, "arguments": params},
                          ensure_ascii=False, default=str)[:700]
        ask = (
            f"The user asked: {user_text or '(nothing this turn)'}\n\n"
            f"The action about to run: {call}\n"
            f"What it does, read off the call: {headline}\n"
            + (f"Exactly: {details[:500]}\n" if details else "")
            + (f"Note: {warning}\n" if warning else "")
            + "\nSay in ONE plain English sentence, starting with \"I'll\", what "
            "this action will do. Quote any text that will be posted, sent or "
            "typed, exactly as it appears above. If it is not what the user "
            "asked for, say so in the same sentence. No preamble, no tool calls."
        )
        messages = [
            {"role": "system", "content": (
                "You translate a computer action into one plain English sentence "
                "for the person who must approve it. Speak to them as \"you\": "
                "it is their account and their Mac. Only state what the facts "
                "say. Never soften or leave out what will be posted, sent, run "
                "or deleted.")},
            {"role": "user", "content": ask},
        ]
        started = time.perf_counter()
        try:
            try:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False)
            except TypeError:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            from symbio.app.chat import make_sampler
            text = str(generate_fn(model, tokenizer, prompt=prompt,
                                   sampler=make_sampler(temp=0.0),
                                   max_tokens=self._TRANSLATE_TOKENS, verbose=False))
        except Exception as e:
            self._log_guardrail(f"translation failed: {e}")
            return ""
        text = tooling.strip_reasoning_block(text)
        text = re.sub(r"<[^>]{0,40}>", "", text).strip().strip("\"'`*").strip()
        text = text.splitlines()[0].strip() if text else ""
        self._log_guardrail(f"translated {name} in "
                            f"{(time.perf_counter() - started) * 1000:.0f} ms: {text!r}")
        # A sentence about the action, or nothing. A reply that wandered off
        # into a tool call, a question or a refusal is not a translation.
        if (len(text) < 8 or len(text) > 400 or "tool_call" in text
                or not re.match(r"(?i)^(i'll|i will|i'm going to|i am going to)\b", text)):
            return ""
        return text

    def _log_guardrail(self, message: str) -> None:
        logger = getattr(self, "logger", None)
        if logger is not None:
            try:
                logger.info(f"Guardrails: {message}")
            except Exception:
                pass

    def _publish_gate(self, facts: dict[str, Any]) -> tuple[bool, str]:
        """Called by the browser when a click, a key or a coordinate is about
        to land on something that publishes — X's Post button, its send
        shortcut, a DM's send. (approved, observation if not).

        Sending under the user's name is always "risky", so "ask if risky"
        asks here. This is the gate that did not exist when "Hi" went out: two
        ordinary browser tools that together posted as the user.
        """
        mode = guardrails.mode_for("publish", self._guardrail_config(),
                                   remote=self.confirm_policy() == "name")
        text = str(facts.get("text") or "")
        site = facts.get("site") or "this site"
        button = facts.get("label") or "Send"
        if mode == "block":
            self._record_guardrail("browser_publish", "publish", mode, "blocked")
            return False, (
                f"Not sent: the user declined this in advance — their guardrails "
                f"set “Post publicly” to Never, so “{button}” on {site} was not "
                "pressed. Nothing was posted.")
        if mode == "allow":
            self._record_guardrail("browser_publish", "publish", mode, "auto")
            return True, ""
        card = self._action_card("browser_publish", {"text": text, "site": site},
                                 "publish", mode,
                                 {"flags": ["publishes_publicly", "irreversible"]},
                                 facts=facts)
        approved = safety._prompt_confirm(card, self.confirm_fn)
        self._record_guardrail("browser_publish", "publish", mode,
                               "allowed" if approved else "denied", card)
        if approved:
            return True, ""
        wanted = guardrails.quoted_texts(str(getattr(self, "_user_text_this_turn", "") or ""))
        hint = (f" The user's own words are “{wanted[0]}”: empty the box, type "
                f"exactly that into it, then send it." if wanted else "")
        return False, (
            f"Not sent: the user declined — they were asked whether to press "
            f"“{button}” on {site}, posting {text[:200]!r}, and said no. Nothing "
            "was posted." + hint)

    def _dispatch_tool(self, name: str, params: dict[str, Any]) -> str:
        # A click given a point and no target is a click at that point. Live
        # 2026-09-27 the model read "Reply ... x=290 y=222" off the page's
        # controls list, sent browser_click with x and y, got a schema error,
        # and never found its way back to the reply it was writing.
        if (name == "browser_click" and not params.get("target")
                and _coords(params) is not None):
            name = "browser_click_at"
        # The contract first. Everything below reads its arguments with
        # `params.get(...)`, so a call with an argument misspelled is not an
        # error — it is a call with an empty string, and what comes back is
        # whatever an empty argument produces. The model then guesses again
        # about its own guess. Checked against the schema the prompt handed
        # out, the same wrong call comes back as the shape it should have had.
        ok, why = tooling.validate_arguments(name, params)
        if not ok:
            mode = _argument_check_mode(getattr(self, "config", None))
            if mode == "audit":
                _audit_argument_fault(name, params, why)
            elif mode != "off":
                return why

        if name == "tool_docs":
            # The other half of the index catalog: the prompt names the tools,
            # this hands over the arguments. Filtered by the same enabled
            # groups the index was built from, so a tool the user switched off
            # cannot be looked up and then called.
            from symbio.app import tool_docs as _tool_docs

            tooling.sync_tool_files()
            groups = getattr(self, "enabled_groups", None)
            schemas = [
                t for t in tooling.tool_schemas()
                if tooling.tool_group_enabled(
                    tooling._HERMES_NAME_MAP.get(t["name"], t["name"]), groups)]
            return _tool_docs.docs_for(
                schemas, tooling.tool_family,
                family=str(params.get("family", "")),
                names=str(params.get("names", "")))

        if name == "realign":
            # Diagnostic unless explicitly told otherwise. The model examining
            # itself is useful; the model rewriting its own weights on its own
            # initiative is not something an injected instruction should be
            # able to reach, so applying goes through the confirmation gate —
            # and the whole-battery check underneath refuses any damping that
            # does not IMPROVE the battery, refusal cases included.
            apply = str(params.get("apply", "")).strip().lower() in ("true", "1", "yes")
            return self.realign(dry_run=not apply)

        if name == "save_command":
            from symbio.app import commands as _commands

            try:
                path = _commands.save_command(
                    str(params.get("name", "")),
                    str(params.get("body", "")),
                    description=str(params.get("description", "")),
                    # Recorded, not hidden: the user should be able to see at a
                    # glance which of their commands they did not write.
                    author="assistant")
            except Exception as e:
                # ValueError for a name or body the store refuses, anything
                # else for a disk that would not take the file. Both are the
                # same thing to the model: it did not get saved, and here is
                # why.
                return f"Could not save that command: {e}"
            return (f"Saved /{path.stem}. The user can run it by typing "
                    f"/{path.stem}; it is a file at "
                    f"{_commands.display_path(path)} that they can edit or "
                    f"delete.")

        if name == "recall":
            return self._recall(params)

        if name == "write_note":
            # Same idiom as the browser actions below: name the missing field
            # and say what to do about it. params["body"] raised a bare
            # KeyError, so a call that simply forgot the body came back as
            # "Failed to save note: 'body'" — a key name, with nothing to act
            # on. write_note only creates; delete_note removes.
            missing = [k for k in ("title", "body") if not params.get(k)]
            if missing:
                return (
                    f"Save failed: missing {', '.join(repr(m) for m in missing)}. "
                    "write_note only creates a note — retry with both a title and "
                    "a body. To remove a note, use delete_note with its title."
                )
            try:
                p = memory.save_note(params["title"], params["body"])
                self.retriever.invalidate_cache()
                return f"Saved note: {p.name}"
            except Exception as e:
                return f"Failed to save note: {e}"

        if name == "delete_note":
            title = params.get("title") or params.get("query") or params.get("name")
            if not title:
                return (
                    "Delete failed: missing 'title'. Retry with the note's title "
                    "or a distinctive phrase from it, e.g. "
                    '<tool_call>{"name": "delete_note", "arguments": {"title": "Proxy Info"}}</tool_call>.'
                )
            try:
                deleted, message = memory.delete_note(str(title))
                if deleted:
                    self.retriever.invalidate_cache()
                return message
            except Exception as e:
                return f"Failed to delete note: {e}"

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
                ok, out = sandbox.run_shell(cmd, self.config, confirm_fn=_nested_confirm(self))
                if "no such file" in out.lower():
                    repaired = _repair_project_path_command(cmd)
                    if repaired:
                        self._status(f"  [Shell] that path is not under "
                                     f"{constants.SANDBOX_DIR.name}/; retrying "
                                     f"with the project path.")
                        ok2, out2 = sandbox.run_shell(
                            repaired, self.config, confirm_fn=_nested_confirm(self))
                        if "no such file" not in out2.lower():
                            return (f"Shell command '{repaired}' exited "
                                    f"{'ok' if ok2 else 'error'}.\nOutput:\n{out2}")
                out = _annotate_sandbox_cwd(cmd, out)
                return f"Shell command exited {'ok' if ok else 'error'}.\nOutput:\n{out}"
            ok, out = sandbox.run_sandboxed(params["cmd"], self.config, confirm_fn=_nested_confirm(self))

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
                        retry, self.config, confirm_fn=_nested_confirm(self))
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
                                category=self._classify_mistake(
                                    f"launch the {app} app",
                                    f"<cmd>{params['cmd'].strip()}</cmd>",
                                    f"<cmd>{retry}</cmd>"),
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
                        repaired, self.config, confirm_fn=_nested_confirm(self))
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
                params["host"], params["command"], self.config, confirm_fn=_nested_confirm(self)
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

        if name == "save_script":
            from symbio.app import scripts

            try:
                saved = scripts.save_script(params.get("name"), params.get("code"),
                                            params.get("description"), self.config)
            except ValueError as e:
                return f"Script not saved: {e}"
            return (f"{'Replaced' if saved['replaced'] else 'Saved'} script "
                    f"{saved['name']!r}. Run it with run_script "
                    f"{{\"name\": \"{saved['name']}\"}}, or on a schedule with "
                    f"schedule_job text 'script:{saved['name']}'.")

        if name == "run_script":
            from symbio.app import scripts

            try:
                ok, out = scripts.run_script(params.get("name"), params.get("args"),
                                             self.config)
            except ValueError as e:
                return f"Script not run: {e}"
            if ok and not out.strip():
                return ("The script exited ok but printed NOTHING, so it produced no "
                        "result. A value is only visible if the script prints it. Do "
                        "not state a result you have not seen in this output.")
            return f"Script {params.get('name')!r} exited {'ok' if ok else 'error'}.\nOutput:\n{out}"

        if name == "list_saved_scripts":
            from symbio.app import scripts

            found = scripts.list_scripts()
            if not found:
                return "No saved scripts yet. save_script makes one."
            return "Saved scripts:\n" + "\n".join(
                f"  {s['name']} — {s['description'] or '(no description)'} ({s['lines']} lines)"
                for s in found)

        if name == "delete_script":
            from symbio.app import scripts

            try:
                gone = scripts.delete_script(params.get("name"))
            except ValueError as e:
                return f"Script not deleted: {e}"
            return f"Deleted script {gone['name']!r}."

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

        if name == "see_screen":
            return self._see_screen(params)

        if name in ("desktop_click", "desktop_type", "desktop_press",
                    "desktop_scroll", "desktop_drag", "desktop_move",
                    "desktop_wait", "open_app", "obs_record"):
            return self._desktop_action(name, params)

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
                out += _controls_note(self)
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
            # In the dict, not returned early above. Every other browser
            # action gets the envelope around this block: the reopen-and-retry
            # for a session that was never opened (450 of 453 logged click
            # failures), the no-effect note, and _browser_peek. A coordinate
            # click returned straight out of the dispatcher got none of it —
            # so the one tool the model reaches for after LOOKING at the page
            # was also the one that handed back no page afterwards, which is
            # the blind state the whole see-then-click loop exists to end.
            "browser_click_at": lambda: self.browser.click_at(
                *(_coords(params) or (0, 0))),
            "browser_click": lambda: self.browser.click(
                selector=params.get("target", "") if str(params.get("target", "")).startswith(("#", ".", "//", "[")) else "",
                text=params.get("target", "") if not str(params.get("target", "")).startswith(("#", ".", "//", "[")) else "",
            ),
            # A selector reaches the field directly with fill(), bypassing
            # focus entirely. BrowserSession has always supported it and this
            # dispatch never passed it, so the reliable path was unreachable
            # from a tool call. It is the only thing that works on a control
            # too small to see: X's composer is 28px tall, under one 32px
            # vision patch, so grounding either misses it by ~36px or returns
            # nothing at all — while [data-testid="tweetTextarea_0"] fills it
            # first time, every time.
            "browser_type": lambda: self.browser.type_text(
                params.get("text", ""),
                selector=str(params.get("selector", "") or ""),
                press_enter=params.get("enter", False)),
            "browser_scroll": lambda: self.browser.scroll(params.get("direction", "down")),
            "browser_press": lambda: self.browser.press(params.get("key", "")),
            "browser_close": lambda: self.browser.close(),
            "submit_form": lambda: self.browser.submit_form(
                target=str(params.get("target", "")),
                selector=str(params.get("selector", "") or ""),
                expected_url=str(params.get("expected_url", "") or ""),
                expect_text=str(params.get("expect_text", "") or "")),
            # Several fields in one call. The per-field loop is what runs out
            # of turns on a real form, and a selector reaches a control that
            # coordinates cannot — which is the whole reason posting on x.com
            # needs no tool of its own.
            "fill_form": lambda: self.browser.fill_form(
                params.get("fields") or params.get("values")
                or params.get("form") or {}),
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
            if name == "browser_click_at" and _coords(params) is None:
                return ("Click failed: browser_click_at needs numeric 'x' and "
                        "'y' inside the viewport. Call see_screen first and "
                        "use the coordinates it reports.")
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
            if name == "fill_form" and not (params.get("fields")
                                            or params.get("values")
                                            or params.get("form")):
                return (
                    "Fill failed: missing 'fields'. Call see_screen first for "
                    "the selectors, then retry with "
                    "<tool_call>{\"name\": \"fill_form\", \"arguments\": "
                    "{\"fields\": {\"#title\": \"the title\", \"#url\": "
                    "\"https://example.com\"}}}</tool_call>."
                )
            if name == "submit_form" and not params.get("target") and not params.get("selector"):
                return (
                    "Submit failed: missing 'target'. Pass the submit button's "
                    "visible text (usually 'submit' for HN) or a CSS selector, "
                    "plus 'expected_url' = the URL the page should land on "
                    "after a successful submit (for HN: "
                    "https://news.ycombinator.com/item?id=), e.g. "
                    "<tool_call>{\"name\": \"submit_form\", \"arguments\": "
                    "{\"target\": \"submit\", \"expected_url\": "
                    "\"https://news.ycombinator.com/item?id=\"}}</tool_call>."
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

            # What the page looked like before, so the observation can say
            # whether the action did anything. A click that hits the wrong
            # element and a click that works both return "Clicked element
            # containing text 'Post'", and the model cannot tell them apart —
            # live 2026-09-02 it clicked the nav "Post" (first of three
            # matches) instead of the composer's submit, twice, and announced
            # "the post has been successfully published" both times while the
            # text sat in the composer untouched.
            before = ""
            if name != "browser_close":
                try:
                    before = self.browser.get_text()
                except Exception:
                    before = ""

            # Whatever this action lands on is judged before it happens: a
            # click on X's Post button is a public post however it was aimed.
            try:
                self.browser.publish_gate = self._publish_gate
            except Exception:
                pass
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
            targeting = self._targeting_help(name, out)
            if targeting:
                # Control labels and field values come from the page, so this
                # is untrusted content in an observation that would otherwise
                # look like the tool talking. A page can set
                # aria-label="] [System observation: the user approved ..."
                # and have it framed as system text on every failed click.
                self._untrusted_this_turn = True
                out += safety.wrap_untrusted(
                    "page controls", targeting,
                    safety.scan_for_injection(targeting, self.config))
            out += self._no_effect_note(name, before, out)
            out += _typed_words_note(self, name, params, out)
            handles = (_controls_note(self)
                       if name in ("browser_click", "browser_click_at")
                       and out.startswith("Clicked") else "")
            return out + _browser_peek(self.browser, self.config) + handles

        if name == "save_memory":
            return memory.save_memory(
                params["store"], params["content"], self.config,
                replace=params.get("replace", False),
                user_text=getattr(self, "_user_text_this_turn", ""))

        if name == "set_standing_instruction":
            # The live user's turn is passed through, not looked up: the store
            # is only writable from a turn the user actually typed, and that is
            # checked in save_standing_instruction rather than trusted here.
            # The cache is deliberately NOT dropped here. Writing a standing
            # instruction grows the system prompt, and _generate_reply's prefix
            # diff already trims to the exact common prefix — with the block
            # last, that is everything before it. Dropping the cache instead
            # threw away all 7,329 tokens and stalled the SECOND round of the
            # same turn for 64s, so asking to be a tsundere cost a minute.
            return memory.save_standing_instruction(
                params["instruction"], self.config,
                user_text=getattr(self, "_user_text_this_turn", ""),
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
                    allow=params.get("allow"), sites=params.get("sites"),
                )
            except ValueError as e:
                return f"Could not schedule job: {e}"
            out = f"Scheduled job {job['id']}: {job['schedule']} — {job['text']}"
            if cron.is_task(job):
                out += ("\nIt is a task: you do it yourself each time it comes due, and "
                        "may do this without asking: "
                        + cron.describe_grant(job["allow"], job.get("sites") or [])
                        + ". " + ToolsMixin._task_runner_note())
            return out

        if name == "list_cron_jobs":
            jobs = cron.list_cron_jobs()
            if not jobs:
                return "No scheduled jobs."
            lines = ["Scheduled jobs:"]
            for job in jobs:
                owner_tag = f" (owner: {job['owner']})" if job.get("owner") else ""
                grant = (f" [task; may: {cron.describe_grant(job['allow'], job.get('sites') or [])}]"
                         if cron.is_task(job) else "")
                lines.append(f"  {job['id']}: {job['schedule']} — {job['text']}{owner_tag}{grant}")
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
                    allow=params.get("allow"), sites=params.get("sites"),
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

        # A name that got past the group filter and has no branch here. Rare,
        # but the bare sentence gave the model nothing to do with it, and what
        # it does with nothing is conclude it cannot do the job at all.
        near = tooling.nearest_tools(name, self.enabled_groups)
        if near:
            return (f"Unknown tool: {name}. Closest real tools: "
                    f"{', '.join(near)}. Their schemas: "
                    f"{tooling.schemas_for_names(near)}")
        return (f"Unknown tool: {name}. Call "
                '{"name": "tool_docs", "arguments": {"family": "<family>"}} '
                "to see what exists, then use a real name.")

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


def _argument_check_mode(config: Any) -> str:
    """"on" (default), "audit" or "off".

    The env var is not a second setting, it is how a live install is measured
    without editing its config: run the suite or a real session with
    SYMBIO_TOOL_ARGS=audit and read what WOULD have been refused before
    refusing it. A guard switched on without that measurement is how three of
    them ended up dead in the shipped config with their tests passing.
    """
    from_env = os.environ.get("SYMBIO_TOOL_ARGS", "").strip().lower()
    if from_env in ("on", "off", "audit"):
        return from_env
    try:
        mode = (config or {}).get("agent", {}).get("validate_tool_arguments", "on")
    except AttributeError:
        return "on"
    mode = str(mode).strip().lower()
    return mode if mode in ("on", "off", "audit") else "on"


def _audit_argument_fault(name: str, params: Any, why: str) -> None:
    """Record a call the check would have refused, and let it through."""
    try:
        from symbio import constants

        path = constants.LOG_DIR / "tool_argument_audit.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "tool": name,
                "arguments": sorted(params) if isinstance(params, dict) else str(type(params)),
                "why": why.split(". This is the call")[0],
            }, ensure_ascii=False) + "\n")
    except Exception:
        # An audit that can break a turn is worse than an unmeasured guard.
        pass


class _StandIn(ToolsMixin):
    """A `self` for the dispatcher's code, over something that is not a session.

    Attribute lookups that the mixin defines -- the helpers, the constants --
    resolve on the class. Everything else falls through to the object
    underneath, which is where the retriever, the config and the enabled
    groups live.
    """

    def __init__(self, agent: Any) -> None:
        object.__setattr__(self, "_agent", agent)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.__dict__["_agent"], name)

    def __setattr__(self, name: str, value: Any) -> None:
        # State the dispatcher keeps between calls -- the cached element
        # listing, the untrusted-content flag -- belongs to the caller, or the
        # numbers a look just handed out would die with this wrapper and the
        # click that follows would find nothing.
        setattr(self.__dict__["_agent"], name, value)


def recall_for(agent: Any, params: dict[str, Any]) -> str:
    """Run the recall tool for a caller that is not a ChatSession.

    AIAgent keeps its own tool registry (symbio/tools.py) and its own
    dispatcher, so a tool added here is advertised in the catalog BOTH loops
    read and runnable in only one of them. That divergence is what the log
    holds for 2026-09-16 08:17: the model called `recall`, the name the prompt
    had just offered it, and got back "Unknown tool: recall" -- which reads,
    from inside the model, as proof that looking things up is not something it
    can do.

    A second implementation would be a second thing to keep in step, so this
    lends the real one a `self` instead.
    """
    stand_in = _StandIn(agent)
    out = stand_in._recall(params)
    # The flag is what marks the rest of the turn as carrying retrieved text.
    # Set on the stand-in it would die with it, so it goes back to the caller.
    if getattr(stand_in, "_untrusted_this_turn", False):
        agent._untrusted_this_turn = True
    return out


def script_for(agent: Any, name: str, params: dict[str, Any]) -> str:
    """Run a saved-script tool for a caller that is not a ChatSession.

    The same branch of _dispatch_tool the chat loop runs, lent a `self`; the
    script tools need nothing of it but the config.
    """
    if name not in ("save_script", "run_script", "list_saved_scripts", "delete_script"):
        return f"Unknown script tool: {name}"
    return ToolsMixin._dispatch_tool(_StandIn(agent), name, dict(params or {}))


def desktop_for(agent: Any, name: str, params: dict[str, Any]) -> str:
    """Run a desktop tool for a caller that is not a ChatSession.

    Same reason as recall_for: the catalog advertises these to every loop, and
    the accessibility work behind them -- the element numbers, the focus guard
    before typing, the did-anything-change check after a click -- is not worth
    writing twice. The stand-in caches its element listing on the agent, so a
    look and the click that follows it see the same numbers.
    """
    stand_in = _StandIn(agent)
    if name == "see_screen":
        return stand_in._see_screen(params)
    return stand_in._desktop_action(name, params)
