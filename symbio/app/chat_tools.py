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

from symbio import computer, constants, safety
from symbio.app import (
    cron, health, learn, local_telemetry, mcp_bridge, memory, sandbox,
    security, tooling, training, web,
)
from symbio.app.config import config_show, set_config_value
from symbio.app.chat_constants import (
    _ALWAYS_CONFIRM_TOOLS, _LOCAL_TRUSTED_TOOLS, _TELEGRAM_CONFIRM_TOOLS)
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

        # Ask by name — before the risk scorer gets a say — for the actions
        # whose cost does not depend on their arguments, and, when the person
        # is somewhere else, for the ones they cannot judge from there.
        if self.confirm_fn is not None and self._asks_by_name(name):
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

        # The desktop's own answer, before any model is asked. macOS publishes
        # every native control's role, title and frame through the same API a
        # screen reader uses: exact text instead of text read off 11px type,
        # exact frames with no patch floor and no Retina scale to undo, and no
        # second set of weights resident next to the headmaster. Vision stays
        # for what has no tree — a canvas, a game, a screen share — which is
        # the only place it was ever the better instrument.
        if target.startswith("desk") or target.startswith("screen"):
            if not self._desktop_enabled():
                return ("Looking at the whole desktop is disabled. Enable the "
                        "'desktop' tool group first, or use target='browser' "
                        "to look at the open page.")
            listing = self._ax_look(question)
            if listing:
                return self._wrap_look(listing)
            # No tree. If the reason is the Accessibility grant, say that
            # rather than falling through to a vision failure: one setting
            # away is the exact list of controls, and a model told only
            # "vision is unavailable" concludes it cannot see the screen at
            # all — which is now false.
            grant_note = getattr(self, "_ax_grant_note", "")
            if grant_note and not vision.available():
                return grant_note

        if not vision.is_enabled(self.config):
            return ("Vision is disabled. Enable it with "
                    "<config set=\"vision.enabled\">true</config>.")
        if not vision.available():
            return ("Vision is unavailable: mlx-vlm is not installed. "
                    "Install it with `pip install mlx-vlm`, then look again.")

        if target.startswith("desk") or target.startswith("screen"):
            if not self._desktop_enabled():
                return ("Looking at the whole desktop is disabled. Enable the "
                        "'desktop' tool group first, or use target='browser' "
                        "to look at the open page.")
            try:
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
            where = "the desktop"
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

        # Read the screenshot's text before waking anything. A named control
        # usually carries its name, and the OS reads that in well under a
        # second with an exact box — where the VLM costs the headmaster's
        # unload and reload plus ~6s a question. ocr.find declines anything
        # ambiguous, and a decline or a miss (an icon has no text) falls
        # through to vision: it never means "not there".
        if question and self.config.get("vision", {}).get("ocr", True):
            from symbio import ocr

            hits = ocr.find(shot, question)
            if hits:
                self._status("  [Vision] Found it by reading the screen's "
                             "text (no vision model needed).")
                click_tool = ("desktop_click" if where == "the desktop"
                              else "browser_click_at")
                return self._wrap_look("\n".join([
                    f"Read the text on {where} ({shot.name}); exactly one "
                    f"piece of text matches what you asked about:",
                    vision.format_elements(hits),
                    f"\nPass its centre to {click_tool}. If this text is not "
                    f"the control you meant, say how it differs and look "
                    f"again with a question naming what sets it apart.",
                ]))

        self._status(f"  [Vision] Looking at {where}...")
        try:
            description, elements = self._run_vision(shot, question)
        except Exception as e:
            return (f"Could not look at {where}: {e}")

        lines = [f"Looking at {where} ({shot.name}):", description.strip()]
        if elements:
            click_tool = ("desktop_click" if where == "the desktop"
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
                    c, "desktop_click" if where == "the desktop"
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

    def _ax_look(self, question: str = "", limit: int = 40) -> str:
        """The frontmost window as a numbered list of controls, or "".

        An empty string means "this is not answerable from the tree" — no
        Accessibility grant, or a window that draws its own interface — and
        the caller falls through to vision, which is the instrument for that.
        """
        from symbio import ax

        if not ax.available():
            return ""
        snap = ax.snapshot(limit=limit)
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
        snap = getattr(self, "_last_ax", None)
        if not snap or time.time() - snap.get("taken_at", 0) > self._AX_TTL:
            snap = ax.snapshot(limit=60)
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

        if name == "open_app":
            out = computer.open_app(str(params.get("name") or ""))
            # The tree of whatever was in front is now the wrong tree.
            self._last_ax = None
            return out

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
        """Whether this tool stops for approval before it is even scored."""
        if name in _ALWAYS_CONFIRM_TOOLS:
            return True
        return self.confirm_policy() == "name" and name in _LOCAL_TRUSTED_TOOLS

    def _dispatch_tool(self, name: str, params: dict[str, Any]) -> str:
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

        if name == "see_screen":
            return self._see_screen(params)

        if name in ("desktop_click", "desktop_type", "desktop_press",
                    "desktop_scroll", "desktop_drag", "desktop_move",
                    "desktop_wait", "open_app"):
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

        if name == "post_to_x":
            if not self.config.get("browser", {}).get("enabled", False):
                return "Browser automation is disabled, so there is nothing to post with."
            if not self.browser.is_open:
                return ("The browser is not open. Open https://x.com/home with "
                        "browser_open first, check you are signed in, then post.")
            out = self.browser.post_to_x(str(params.get("text") or ""))
            self._untrusted_this_turn = True   # the timeline was read back
            return out

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
        if name == "submit_form":
            return (f"Submit the form on the live page? target='{params.get('target')}' "
                    f"expected to land on '{params.get('expected_url')}'.")
        return f"Allow tool '{name}'?"


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
