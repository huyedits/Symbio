"""The autonomous agent turn: one user message driven to completion.

This is the loop that generates a reply, parses tool calls out of it, runs
them, feeds the observations back, and decides when the turn is finished --
plus the learning that hangs off the end of it (mood, affect, research notes).

A mixin rather than a module of functions: the turn reads and mutates nearly
all of the session's live state. ChatSession inherits it.
"""

import json
import time
from pathlib import Path
from typing import Any

from symbio import constants, safety
from symbio.tools import tool_few_shots
from symbio.app import (
    learn, local_telemetry, memory, prompts, skills, tooling, training, web,
)
from symbio.app.chat_constants import (
    _BROWSER_ACTION_TOOLS, _MAX_RATE_LIMIT_RETRIES, _MAX_RATE_LIMIT_WAIT,
    _MAX_TOOL_RETRIES, _WEB_TOOLS, _claims_completion, _internal_to_hermes_name,
)
from symbio.app.chat_text import (
    _EXPLICIT_SEARCH_RE, _MOOD_TAG_RE, _VALID_MOODS, _asks_for_action,
    _is_action_request, _is_greeting, _is_navigation_only, _is_substantive,
    _last_exchange, _looks_like_verification_followup,
    _subjectless_search_command, infer_user_affect,
)


class AgentTurnMixin:
    """The agent loop for ChatSession."""

    def _agent_turn(self, user_input: str):
        # The boot system-prompt prefill may still be running on its background
        # thread. Join it before any model use this turn (canary, memory flush,
        # tool summarizers, _generate_reply) so the model is never used by two
        # threads at once. No-op once the prefill has finished.
        self._await_prefill()
        self._auto_compact_if_under_pressure()
        self._periodic_canary_check()
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

        # Surface any cron events that fired since the last turn. Treat their
        # text as untrusted: a malicious reminder could try to inject commands.
        with self.cron_lock:
            due_events, self.cron_events[:] = list(self.cron_events), []
        if due_events:
            due_text = "\n".join(due_events)
            cron_scan = safety.scan_for_injection(due_text, self.config)
            wrapped_events = safety.wrap_untrusted("scheduled event", due_text, cron_scan)
            self.history.append({
                "role": "user",
                "content": "[System observation: scheduled event(s)]\n" + wrapped_events,
            })

        # Canary check: if the user asks for the canary phrase, the model must
        # echo it back. Failing is a signal that the system prompt is being
        # ignored or context has degraded.
        canary_phrase = "SYMBIO_CANARY_v1"
        lower_input = user_input.lower()
        is_canary_request = (
            "canary" in lower_input
            or "repeat the hidden phrase" in lower_input
            or f"repeat {canary_phrase.lower()}" in lower_input
        )
        canary_failed = False
        if is_canary_request:
            self.history.append({"role": "user", "content": user_input})
            # Skip normal processing: run a single-shot generation just to check
            # whether the model still follows the system prompt.
            check_messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"What is the canary phrase? Reply with only '{canary_phrase}' and nothing else."},
            ]
            try:
                check_prompt = self.tokenizer.apply_chat_template(
                    check_messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=training.THINKING_ENABLED,
                )
                check_reply = self.generate_fn(
                    self.model, self.tokenizer, prompt=check_prompt, sampler=self.sampler,
                    # Room for a thinking block plus the one-word answer.
                    max_tokens=256, verbose=False,
                ).strip()
                # The model may reason before echoing the phrase; the check
                # grades the answer, not the reasoning.
                check_reply = tooling.strip_reasoning_block(check_reply)
            except Exception as e:
                self.output_fn(f"[Canary check failed: {e}]")
                canary_failed = True
                check_reply = ""
            if canary_phrase not in check_reply:
                canary_failed = True
                self.output_fn(
                    "  [Canary] The model did not repeat the canary phrase — "
                    "system-prompt adherence may have degraded. Compacting memory to reduce context pressure."
                )
                # Compact both curated stores to shrink context.
                for store in ("memory", "profile"):
                    if constants.PROFILE_FILE.exists() or constants.MEMORY_FILE.exists():
                        try:
                            msg, _ = memory.compact_store(store, self.config)
                            self.output_fn(f"  [Canary] {msg}")
                        except Exception as exc:
                            self.output_fn(f"  [Canary] Could not compact {store}: {exc}")
                self.retriever.invalidate_cache()
                safety.log_security_event("canary_failed", {
                    "reply": check_reply,
                    "prompt_tokens": timings.get("prompt_tokens"),
                    "new_tokens": timings.get("new_tokens"),
                })
            else:
                self.output_fn(f"  [Canary] OK — the model still follows the system prompt.")
            return

        self.history.append({"role": "user", "content": user_input})
        local_telemetry.log_event("turn", user=user_input)

        # A subjectless "check online" / "search it" / "look it up" gives the
        # 8B model no topic to bind the search to, so it hallucinates a query
        # unrelated to the conversation ("Who is the CEO of Apple Inc." when
        # asked to look up Windows 11 pricing). Resolve the previous unanswered
        # question as the search subject; the web_search tool layer overrides
        # any hallucinated query with this subject, and the research note is
        # filed under it instead of under the bare command.
        self._search_subject = None
        if _subjectless_search_command(user_input):
            subj_q, _subj_a = _last_exchange(self.history)
            if subj_q:
                self._search_subject = subj_q
                # Fold the nudge into the user's own turn instead of appending a
                # second user message: two consecutive user turns break the
                # Mistral chat template's strict role alternation ("After the
                # optional system message, conversation roles must alternate
                # user/assistant/user/assistant/..."). The template allows only
                # one optional system message at the start, so a mid-conversation
                # system injection has to ride inside a user turn.
                self.history[-1]["content"] += (
                    "\n\n[System: the user asked you to search online but gave no "
                    f"subject. They mean your previous unanswered question: "
                    f"\"{subj_q}\". Call web_search for exactly that question, "
                    f"then answer from the results. Do NOT search for anything "
                    f"else or change the subject.]"
                )
                self._trim_history()

        # Short verification follow-ups ("are you sure?", "check again") give
        # the 8B model almost no signal, so at low temperature it derails —
        # reciting its identity or regurgitating an earlier topic instead of
        # re-examining the answer it just gave. Inject a contextual nudge that
        # embeds the actual previous Q&A so the model has the full prior context
        # inline and knows to verify (search if uncertain), rather than having
        # to dig it out of history itself.
        if _looks_like_verification_followup(user_input):
            q, a = _last_exchange(self.history)
            if q and a:
                a_short = a if len(a) <= 600 else a[:600] + "…"
                nudge = (
                    "[System: the user doubts your previous answer and asks you to "
                    f"re-check it.\nPrevious question: {q}\n"
                    f"Your previous answer: {a_short}\n"
                    "Re-examine that answer. If you are not certain it is correct, "
                    "call web_search to verify and then give a corrected answer. "
                    "Do not recite your identity or change the subject.]"
                )
            else:
                nudge = (
                    "[System: the user is asking you to re-examine your previous answer. "
                    "Briefly restate what you last claimed, then verify it — if you are "
                    "not certain, call web_search and answer from the results. "
                    "Do not recite your identity or change the subject.]"
                )
            # Fold into the user's own turn, not a separate message — see the
            # subjectless-search block above for why two consecutive user turns
            # break the Mistral chat template's role alternation.
            self.history[-1]["content"] += f"\n\n{nudge}"

        # The user's mood this turn is inferred by the model itself, not here:
        # Caine reads tone from language (the way a language model naturally
        # does) and emits a <mood>tag</mood> at the start of its reply, which
        # the tool loop parses and surfaces as [Mood: tag]. infer_user_affect
        # is only a fallback for turns where the model omits the tag. No
        # pre-generation nudge — the model adapts its own tone per the system
        # prompt once it has read the mood.

        # Unbounded knowledge: pull relevant saved notes into this turn's
        # context. Retrieval text never enters history or training data.
        rag_context = self.retriever.build_context(user_input)
        rag_results: list[dict[str, Any]] = []
        # Skill workers whose notes retrieval matched this turn.
        suggested_roles: list[str] = []
        if self.retriever.rag_cfg.get("enabled", True):
            for r in self.retriever.retrieve(user_input):
                rag_results.append(r)
                if r.get("source") == "note" and r.get("path"):
                    note_path = Path(r["path"])
                    try:
                        skills.record_note_usage(note_path)
                    except Exception:
                        pass
                    if skills._is_skill_note(note_path):
                        self._skill_notes_used.add(note_path)
                        self._record_health_errors_for_skill(note_path)
                        role = None
                        try:
                            role = skills.delegatable_role_for_note(
                                note_path, self.config)
                        except Exception:
                            pass
                        if role and role not in suggested_roles:
                            suggested_roles.append(role)
        rag_block = f"\n\n{rag_context}" if rag_context else ""
        # Retrieval matched a skill that has its own trained worker. Say so
        # rather than routing on it: a suggestion the model can decline costs a
        # line of context when retrieval is wrong, where hard routing would
        # hand the whole turn to the wrong specialist.
        if suggested_roles and self.config.get("dispatch", {}).get(
                "suggest_skill_workers", True):
            offers = ", ".join(f"<delegate role='{r}'>the task</delegate>"
                               for r in suggested_roles)
            # Retrieval usually surfaces about two candidates, so the offer has
            # to carry enough to tell them apart. Each worker's own recorded
            # reason for existing does that; without it the model is choosing
            # between bare role names it has no basis to rank.
            reasons = ""
            if len(suggested_roles) > 1:
                from symbio.app import dispatch as _dispatch

                lines = []
                for role in suggested_roles:
                    entry = _dispatch.catalog_entry_for_role(role) or {}
                    why = (entry.get("routing_rationale") or "").strip()
                    if why:
                        lines.append(f"  - {role}: {why.splitlines()[0]}")
                if lines:
                    reasons = "\n Which one:\n" + "\n".join(lines)
            rag_block += (
                f"\n\n[System note: this request matches a skill that has its own "
                f"trained worker. The procedure is in that worker's weights, so "
                f"prefer handing it over with {offers} rather than answering from "
                f"memory. Ignore this if the request is not actually about that "
                f"skill.{reasons}]"
            )
        rag_ms = (time.perf_counter() - turn_start) * 1000
        timings["rag_ms"] = rag_ms
        # Surface that retrieval ran and what it pulled in, so the user can see
        # the agent isn't sitting silent before the spinner starts. Hits are
        # always shown; a no-match is only mentioned when retrieval was slow
        # enough that the pause would otherwise look like a stall.
        # Retrieved notes are text the assistant did not write this turn and the
        # user did not type — the carrier injection actually travels on. Record
        # that it entered, so assess_provenance can tell a tool the user asked
        # for from one that appeared right after some retrieved text did.
        self._untrusted_this_turn = bool(rag_results)
        # Whether the user asked for anything to be done this turn; read by the
        # intent gate in _execute_tool.
        self._action_asked_this_turn = _asks_for_action(user_input)
        # The user's own words this turn. The write-scanner reads it to tell a
        # note that records what they just said from one that carries an
        # instruction out of retrieved text.
        self._user_text_this_turn = user_input

        if self.retriever.rag_cfg.get("enabled", True):
            if rag_results:
                labels = sorted({
                    r.get("broad_tag") or r.get("title", "?")
                    for r in rag_results if r.get("source") == "note"
                })[:3]
                extra = f" ({', '.join(labels)})" if labels else ""
                self.output_fn(f"  [RAG] {len(rag_results)} hit(s){extra} · {rag_ms:.0f}ms")
            elif rag_ms > 100:
                self.output_fn(f"  [RAG] no notes matched · {rag_ms:.0f}ms")

        # Live-reload: config changes and prompt.md edits apply on the next turn.
        self._refresh_sampler()
        self.system_prompt = prompts.build_system_prompt(
            self.config["assistant_name"], self.config["user_name"], self.config
        )
        timings["prompt_ms"] = (time.perf_counter() - turn_start) * 1000

        self.user_turns += 1
        nudge_block = self._nudge_block()

        max_rounds = self.config["agent"]["max_tool_rounds"]
        executed_calls: set[str] = set()
        # executed_calls is "calls not to repeat", and the retry path below
        # DISCARDS from it when a call fails and is still retry-eligible — so
        # it is empty after a failed tool, and cannot answer "did anything
        # run this turn". The end-of-turn fallback needs that second question.
        any_tool_ran = False
        # How many times each call has failed this turn, so a retry after a
        # fixed precondition is allowed but a persistently failing call is not.
        failed_calls: dict[str, int] = {}
        # Counted separately from failed_calls: a rate limit is not the call
        # going wrong, so it must not consume the budget for one that is.
        rate_limited_calls: dict[str, int] = {}
        web_used = False
        auto_searched = False
        self_corrected = False
        final_display = ""
        # User mood this turn, resolved from the model's own <mood> tag (or
        # the lexicon heuristic fallback). Surfaced once as [Mood: ...].
        mood = "neutral"
        mood_decided = False
        consecutive_tool_rounds = 0
        scrolls_this_turn = 0
        _MAX_SCROLLS_PER_TURN = 5
        # The exact "[System observation: ...]" text of the most recent
        # tool failure this turn, if any — used to capture (saw this error
        # -> did this instead, which worked) as a mistake-note training
        # sample the moment a later tool call actually succeeds. Cleared on
        # any success so only a confirmed fix gets saved, not a mere retry.
        pending_tool_error: str | None = None
        # Track whether the last tool executed this turn was a browser action
        # that failed. If the model then tries to end the turn without another
        # tool tag, we nudge it to retry — otherwise Caine just explains the
        # failure and gives up after one attempt.
        pending_browser_error: str | None = None
        # Set the moment the user declines anything, and never cleared before
        # the turn ends: a "no" applies to the rest of the turn, not just to
        # the one tool that asked.
        user_refused_this_turn = False
        browser_retry_nudged = False
        blank_retry_nudged = False
        claim_nudged = False
        unparsed_tag_nudged = False
        echo_retry_nudged = False
        # The round index used to select thinking (think=round_num > 0);
        # agent.thinking_level owns that now, so nothing reads the counter.
        for _round_num in range(max_rounds):
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
            # Browser state: tell the model what page is open so it doesn't
            # forget between turns and try to reopen or use run_command.
            browser_note = ""
            if self.browser.is_open:
                browser_note = "\n\n[" + self.browser.status() + "]"
            # Static parts first, volatile parts last.
            #
            # This block is prepended to the last user message, so everything
            # from the first byte that differs from last turn has to be
            # re-prefilled. env_note is fixed for the life of the machine;
            # sitting it after rag_block meant a new retrieval hit pushed it
            # into the re-prefilled region every time, for nothing.
            #
            # Measured on a 349-token block: when the clock ticks alone this
            # changes nothing (344 tokens reused either way), but when the RAG
            # hit also changes — the common case, since retrieval runs per
            # query — reuse goes from 141 tokens to 243.
            context_block = (
                memory.curated_memory_block(self.config) + prompts.env_note()
                + rag_block + prompts.time_note() + nudge_block
                + browser_note
            ).lstrip()
            # Greeting guard: the small model sometimes invents random tool
            # calls for "hi" instead of just greeting back. Prepend a one-
            # line nudge that anchors it to the greeting few-shot example.
            if _is_greeting(user_input) and not executed_calls:
                context_block = (
                    "[This is a greeting. Reply with a short friendly greeting "
                    "and ask what the user needs. Do NOT call any tool — just "
                    "say hi back.]\n\n" + context_block
                ).lstrip()
            working_history = list(self.history[-self.config["agent"]["history_limit"]:])
            if context_block:
                attached = False
                for i in range(len(working_history) - 1, -1, -1):
                    if (
                        working_history[i]["role"] == "user"
                        and not str(working_history[i]["content"]).startswith("[System observation:")
                    ):
                        working_history[i] = {
                            "role": "user",
                            "content": context_block + "\n\n" + working_history[i]["content"],
                        }
                        attached = True
                        break
                # First turn: no user message in history yet. Prepend the
                # context block as a standalone system-observation user turn
                # so greeting guards and env notes actually reach the model.
                if not attached:
                    working_history.insert(0, {
                        "role": "user",
                        "content": f"[System observation: {context_block}]",
                    })
            # Canonical tool-use examples (open/close an app, disk space,
            # weather, post-tool acknowledgement, etc.). The agent stack
            # (symbio/agent.py) always injects these; the app stack did not,
            # so the model never saw a worked <cmd>/<search> example and fell
            # back to giving manual steps. They are constant across turns, so
            # they fold into the cached system+few-shot prefix at no cost.
            messages.extend(tool_few_shots(self.config))
            messages.extend(working_history)

            chunk_prefix = f"{self.config['assistant_name']:8}: " if self.stream_prefix else ""
            # Resample once if the reply has a dangling (truncated) tool call —
            # the model started emitting a tool tag but hit max_tokens or got
            # cut off. A fresh sample usually completes it, avoiding a system-
            # observation round-trip that pollutes history.
            gen_aborted = False
            for _sample_attempt in range(2):
                # _generate_reply clears the cache in its own error path before
                # re-raising, so whether one was in play has to be recorded here
                # or the handler below can never tell.
                _had_prompt_cache = self._prompt_cache is not None
                try:
                    _think, _budget = self.thinking_setting()
                    raw_reply, streamed_live = self._generate_reply(
                        messages, chunk_prefix=chunk_prefix, timings=timings,
                        think=_think, reasoning_budget=_budget,
                        )
                    # The thinking block is surfaced to the user (streamed by
                    # StreamingStripper, or printed below when not streaming);
                    # the reply itself stays reasoning-free so tools and
                    # history never see it.
                    reasoning = tooling.extract_reasoning(raw_reply)
                    reply = tooling.clean_response(
                        tooling.strip_reasoning_block(raw_reply)).strip()
                    self.logger.info(f"RAW_REPLY: {raw_reply!r}")
                except KeyboardInterrupt:
                    self.output_fn("\n  [Generation interrupted.]")
                    gen_aborted = True
                    break
                except Exception as e:
                    # The warmed prompt cache is an optimization, and the turn
                    # must not die with it. A cache persisted by an earlier run
                    # can reference an MLX stream that does not exist in this
                    # process ("There is no Stream(cpu, N) in current thread");
                    # loading it succeeds and generation is where it explodes,
                    # so the prefill guard never sees it. Drop it, delete the
                    # stale file so the next run does not inherit the same
                    # crash, and take the second attempt without it.
                    if _had_prompt_cache and _sample_attempt == 0:
                        self.output_fn("  [Cache] Warmed prompt cache unusable "
                                       "here; discarding it and retrying.")
                        self._drop_prompt_cache(
                            "the warmed cache was unusable in this process")
                        try:
                            constants.PROMPT_CACHE_FILE.unlink(missing_ok=True)
                        except OSError:
                            pass
                        continue
                    self.output_fn(f"[MLX Error: {e}]")
                    gen_aborted = True
                    break
                # Check for dangling tool calls (truncated mid-JSON or
                # unterminated tag). If clean, stop; otherwise resample.
                if not tooling.detect_malformed_tag(reply):
                    break
                # Only a sample that showed the user nothing can be quietly
                # retried. Once text has streamed to the screen, a second
                # sample would print a whole second reply underneath the
                # first — so leave it to the self-correction observation
                # below, which repairs the turn without duplicating output.
                if streamed_live:
                    break
            if gen_aborted:
                break

            # The model emits <mood>tag</mood> at the start of its reply to show
            # how it read the user's tone (it catches things a regex misses —
            # e.g. a lone raised-voice word like "DOINGG"). StreamingStripper
            # already hid the tag while streaming; strip it from the reply so
            # it never reaches history/display/parse_tools, and surface the
            # detected mood once. If the model gave no tag this turn, fall
            # back to the lexicon heuristic.
            m_match = _MOOD_TAG_RE.search(reply)
            if m_match:
                reply = _MOOD_TAG_RE.sub("", reply).strip()
                tag = m_match.group(1).lower()
                if not mood_decided:
                    mood = tag if tag in _VALID_MOODS else "neutral"
                    mood_decided = True
                    self.output_fn(f"  [Mood: {mood}]")
            elif not mood_decided:
                mood = infer_user_affect(user_input)
                mood_decided = True
                self.output_fn(f"  [Mood: {mood}]")

            tools = tooling.parse_tools(reply, self.enabled_groups)
            display = tooling.strip_tool_tags(reply)

            # A tool tag this module cannot parse is the quietest failure in
            # the loop: it matches nothing, strip_tool_tags removes it from the
            # display, and the turn ends having done nothing, told the user
            # nothing and given the model nothing to correct. 22 of the
            # declared tools have no <name>arg</name> form, so this is reachable
            # for most of the catalog. Say so once per turn, with the syntax
            # that does work.
            if not tools and not unparsed_tag_nudged:
                unparsed = tooling.unparsed_tool_tags(reply)
                if unparsed:
                    unparsed_tag_nudged = True
                    names = ", ".join(unparsed)
                    self.output_fn(f"  [Tool] <{unparsed[0]}> is not a tag I can "
                                   f"parse; telling the model the right form.")
                    self.history.append({"role": "assistant", "content": reply})
                    self.history.append({"role": "user", "content": (
                        f"[System observation: you wrote <{names}> as a tag. "
                        f"That is a tool NAME, not a tag this system parses, so "
                        f"nothing ran and nothing changed. Call it in the "
                        f"JSON form instead, exactly:\n"
                        f'<tool_call>{{"name": "{unparsed[0]}", "arguments": '
                        f'{{...}}}}</tool_call>\n'
                        f"Do not describe the call — emit it.]"
                    )})
                    self._trim_history()
                    continue

            # A model that emits the same tool tag over and over in one
            # response (e.g. "<scroll/> Scrolling down. <scroll/> Scrolling
            # down. …") is looping. Catch it here so the display text from
            # the stripped tags doesn't flood the user's screen, and nudge
            # the model to break out on the next round.
            if len(tools) >= 4 and not echo_retry_nudged:
                from collections import Counter
                tool_counts = Counter(n for n, _ in tools)
                most_common, count = tool_counts.most_common(1)[0]
                # Fire when one tool is >=80% of all calls and there are
                # at least 4 of it — catches both "49 scrolls" and
                # "1 browser_open + 49 scrolls".
                if count >= 4 and count / len(tools) >= 0.8:
                    echo_retry_nudged = True
                    self.output_fn(
                        f"  [Loop] {count}× <{most_common}/> in one reply "
                        f"({len(tools)} total tags) — regenerating...")
                    self.history.append({"role": "user", "content": (
                        f"[System observation: your last reply contained "
                        f"{count} copies of the <{most_common}/> tag. Do not "
                        f"repeat the same tool call. If scrolling isn't "
                        f"revealing new information, stop and work with "
                        f"what you can see, or try a different approach.]"
                    )})
                    self._trim_history()
                    continue

            # The model wrote the harness's own scaffold, or looped one line.
            # Either way this is not a reply. Checked here, ahead of the
            # display/log block below, because a copy written to the session
            # store comes back through retrieval later and reinforces the
            # habit — catching it further down would filter the symptom while
            # still recording the cause.
            scaffold_echo = (learn.looks_like_observation_echo(display)
                             or learn.looks_like_tool_result_echo(display)
                             or learn.looks_degenerate(display))
            # Handing the user's own instruction back to them only counts as a
            # failure when nothing was actually called. "Opening the GitHub
            # page in browser." is a fine thing to say alongside a browser_open
            # call and a fabrication without one.
            user_echo = (not tools
                         and learn.looks_like_user_echo(display, user_input))
            if not echo_retry_nudged and (scaffold_echo or user_echo):
                echo_retry_nudged = True
                if user_echo:
                    self.output_fn(
                        "  [Echo] Reply restated the request as if done; "
                        "regenerating...")
                    correction = (
                        "you repeated the user's own instruction back as "
                        "though you had carried it out. No tool ran, so "
                        "nothing happened. Either call the tool that would "
                        "actually do it, or say plainly what you can and "
                        "cannot do."
                    )
                else:
                    self.output_fn(
                        "  [Echo] Reply impersonated a system observation; "
                        "regenerating...")
                    correction = (
                        "you wrote text in one of the system's own forms "
                        "('[System observation: ...]', '[Begin untrusted "
                        "...]', 'Opened browser at ... Page title: ...'), or "
                        "repeated one line over and over. Those forms are how "
                        "the system speaks to you — they are never part of "
                        "your reply, and you must never invent one. In "
                        "particular, never write a tool result yourself: if "
                        "you did not receive one, the tool did not run."
                    )
                self.history.append({"role": "user", "content": (
                    f"[System observation: your last reply was discarded — "
                    f"{correction} Answer the user directly now, in your own "
                    f"voice, once.]"
                )})
                self._trim_history()
                continue

            if display.strip():
                final_display = display
                if not streamed_live:
                    # Streaming showed nothing (streaming off, or the whole
                    # reply was a tool tag) — surface the reasoning here so it
                    # is not lost, then the answer.
                    if reasoning and self.config["agent"].get("show_reasoning", True):
                        self.output_fn(f"{tooling.REASONING_MARKER}{reasoning}")
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
                # A syntactically perfect call to a tool that does not exist
                # is not "malformed" and reaches none of the checks below — it
                # is filtered out by the group filter and the turn ends on
                # whatever prose the model wrote next to it, which is usually a
                # claim that the call worked. Tell it the real names, once per
                # turn.
                dropped = tooling.dropped_tool_calls(reply, self.enabled_groups)
                if dropped and not self_corrected:
                    self_corrected = True
                    unknown = [n for n, why in dropped if why == "unknown"]
                    disabled = [n for n, why in dropped if why == "disabled"]
                    self.output_fn(
                        f"  [Tool] dropped call to "
                        f"{', '.join(n for n, _ in dropped)} — retrying.")
                    detail = []
                    if unknown:
                        detail.append(
                            f"No tool named {', '.join(unknown)} exists. The "
                            f"tools you can call are: "
                            f"{', '.join(tooling.enabled_tool_names(self.enabled_groups))}.")
                    if disabled:
                        detail.append(
                            f"{', '.join(disabled)} is turned off in this "
                            f"configuration, so it did nothing.")
                    self.history.append({"role": "user", "content": (
                        f"[System observation: that tool call did not run and "
                        f"returned nothing. {' '.join(detail)} You received no "
                        f"result from it, so do not describe one. The user "
                        f"said: \"{user_input.strip()}\". Call a real tool "
                        f"from that list now, or answer without one.]"
                    )})
                    self._trim_history()
                    continue

                # A tag that looked like a tool call but never resolved
                # (unterminated, or invalid JSON) is a formatting mistake,
                # not a normal reply — surface it as an observation so the
                # model can notice and retry, instead of silently treating
                # the mangled leftovers as the final answer. Once per turn.
                malformed = tooling.detect_malformed_tag(reply)
                if malformed and not self_corrected:
                    self_corrected = True
                    # Show the user a terse note (the raw mangled text is
                    # unreadable); the full snippet is still passed to the
                    # model via the system observation below for self-correction.
                    snippet = " ".join(malformed.split())[:80]
                    self.output_fn(f"  [Format] malformed tool call ({snippet}) — retrying.")
                    # Include the user's original request so the model
                    # doesn't lose context and invent a random action.
                    user_reminder = (
                        f" The user said: \"{user_input.strip()}\". "
                        f"Respond to THAT request."
                    )
                    self.history.append({"role": "user", "content": (
                        f"[System observation: {malformed} Check your tag "
                        f"syntax (matching open/close tags, valid JSON "
                        f"inside <tool_call>) and try again, or continue "
                        f"without it.{user_reminder}]"
                    )})
                    self._trim_history()
                    continue

                # Browser actions often fail on the first target (element not
                # visible yet, text mismatch, selector typo). If the model tries
                # to end the turn after a browser failure without issuing another
                # tool tag, force it to retry once — don't let it give up and
                # explain the failure.
                if pending_browser_error and not browser_retry_nudged:
                    browser_retry_nudged = True
                    self.output_fn(
                        "  [Browser] Previous action failed; prompting retry...")
                    self.history.append({"role": "user", "content": (
                        f"[System observation: {pending_browser_error} "
                        "Do not explain the failure. Retry the browser action "
                        "with a different exact visible text or selector. "
                        "Use browser_get_text if needed. Do not end the turn "
                        "until the user's request is completed.]"
                    )})
                    self._trim_history()
                    continue

                # The model produced no visible answer and no new tool call.
                # Most often a Qwen3 thinking block (or a lone <mood> tag) that
                # clean_response()/mood-stripping reduced to nothing. Don't let
                # the turn die silent — nudge it once to answer or continue.
                # Fires for a mid-task blank (a tool already ran) and for a
                # greeting blank ("hi" → only a mood tag, nothing else): without
                # the greeting branch a first-round blank would skip straight to
                # auto-searching the greeting and answer with random web results.
                # A non-greeting first-round blank is left to the auto-search path
                # below — a real question that blanks should search, not nudge.

                action_req = _is_action_request(user_input)
                # any_tool_ran, not executed_calls: the retry path discards a
                # failed call from executed_calls so it can be attempted again,
                # which empties the set precisely when a tool has just FAILED —
                # the moment the model most needs prompting. Live 2026-08-25: a
                # wc on a bad path failed, the observation explained exactly how
                # to recover, the nudge did not fire because the set was empty,
                # and the turn ended on "Running 'wc -c' on the file." with no
                # answer.
                if (not _is_substantive(display) and not blank_retry_nudged
                        and (any_tool_ran or _is_greeting(user_input)
                             or action_req)):
                    blank_retry_nudged = True
                    self.output_fn(
                        "  [Blank] Reply came back empty; "
                        "prompting the model to respond...")
                    if action_req and not any_tool_ran:
                        act_hint = (
                            "The user asked you to perform an action (open/go to/"
                            "click/press/read a page) but you emitted no tool call. "
                            "Emit one of these tags exactly, on its own: "
                            "<browse>https://...</browse> to open a page, "
                            "<read>https://...</read> to read one, "
                            "<click>visible text</click>, <press>Enter</press>, "
                            "<scroll />. Then answer in one short line. "
                        )
                    else:
                        act_hint = ""
                    mid_task = "You are mid-task — a tool already ran this turn " \
                               "and the user's request is not yet answered. " if any_tool_ran else ""
                    self.history.append({"role": "user", "content": (
                        "[System observation: your last reply was empty after "
                        "removing internal reasoning and the mood tag. "
                        + act_hint + mid_task +
                        "Answer the user now, or continue with the next tool "
                        "call if you are mid-task. The <mood> tag is metadata, "
                        "not your reply — always follow it with a real response. "
                        "Do not end the turn with no visible output.]"
                    )})
                    self._trim_history()
                    continue

                # A fluent reply claiming work it never did. The blank-retry
                # branch above cannot see this one: it fires on an EMPTY reply,
                # and 2026-08-26's "I've run the scrape script for you...
                # processed 25 rows... 23 in clean and 2 in quarantine" was a
                # perfectly formed sentence backed by zero tool calls, zero
                # fetches and no files. Push it to actually act; if it repeats
                # the claim, say so in the open rather than pass it on.
                if (not any_tool_ran and _is_substantive(display)
                        and _claims_completion(display)):
                    if not claim_nudged:
                        claim_nudged = True
                        self.output_fn("  [Unverified] Reply claims completed work "
                                       "but no tool ran; asking it to actually do it...")
                        self.history.append({"role": "user", "content": (
                            "[System observation: your reply states you already "
                            "ran/fetched/saved something, but you emitted no tool "
                            "call this turn, so nothing was executed and nothing "
                            "was written. Do not describe results you have not "
                            "produced. Either emit the tool call now and report "
                            "what it actually returns, or say plainly that you "
                            "have not done it and give the user the command to "
                            "run.]"
                        )})
                        self._trim_history()
                        continue
                    self.output_fn("  [Unverified] The model repeated a completion "
                                   "claim with no tool call — treat the result "
                                   "below as NOT performed.")

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
                # A confident-sounding non-answer to a price/figure question —
                # "it depends on the device, check the official website" with no
                # number — is the model papering over a gap without committing
                # to a figure. sounds_fabricated misses it (no digit to hedge),
                # so detect the deflection explicitly.
                evasive = (not unsure and not fabricated and bool(display.strip())
                           and learn.sounds_evasive(user_input, display))
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
                # If the user explicitly told it to search and the model waffled
                # (described searching instead of calling <search>), force one —
                # don't leave the user stranded with a confident non-answer.
                forced_by_explicit = bool(_EXPLICIT_SEARCH_RE.search(user_input))
                gap_trigger = ((blanked or not user_asked_web_search)
                               and (unsure or fabricated or evasive or blanked))
                if (self.config["web"].get("auto_search_when_unsure", True)
                        and not auto_searched and not web_used and not browser_followup
                        and not trivial_ack
                        and not _is_greeting(user_input)
                        and self.auto_searches < session_cap
                        and (forced_by_explicit or gap_trigger)):
                    auto_searched = True
                    web_used = True
                    self.auto_searches += 1
                    reason = ("ignored an explicit request to search" if forced_by_explicit
                              else "hedged a made-up-sounding figure" if fabricated
                              else "deflected a price question without a figure" if evasive
                              else "sounded unsure" if unsure
                              else "came back blank")
                    # A subjectless "check online" command has no topic of its
                    # own — search the resolved previous question, not the bare
                    # command text (which would query the engine for "check
                    # online" and return junk).
                    search_query = self._search_subject or user_input
                    self.output_fn(f"  [Auto-search] Reply {reason} — searching the web...")
                    ok, out = web.web_search(search_query, self.config)
                    self.history.append({"role": "user", "content": (
                        f"[System observation: Your answer {reason}, so a web "
                        f"search for '{search_query}' ran automatically "
                        f"({'succeeded' if ok else 'failed'}).\nResults:\n{out}\n"
                        f"Answer from these results, citing the exact figure they "
                        f"give. If they don't help, say plainly that you could not "
                        f"find it — do not guess.]"
                    )})
                    self._trim_history()
                    continue
                # Normal turn (or pure repetition): stop.
                # BUT: if the user asked for an action (open/click/type/etc.)
                # and the model only talked about doing it without actually
                # calling a tool, nudge it once to use the right tool.
                # any_tool_ran, not executed_calls: the retry path empties
                # executed_calls when a call FAILS, so this fired mid-task on a
                # turn where execute_code had already run three times. And the
                # hint below was browser-only, so it answered a Python task with
                # "emit <browse>" — live 2026-08-25 the model dutifully switched
                # to browser_press and hit "Browser is not open". A nudge that
                # names one toolset drags every unfinished turn towards it.
                if action_req and not any_tool_ran and not blank_retry_nudged:
                    blank_retry_nudged = True
                    self.output_fn(
                        "  [Action] Model described the action but didn't "
                        "call a tool — prompting to retry...")
                    self.history.append({"role": "user", "content": (
                        "[System observation: you described what you would do "
                        "but did not actually call a tool. Emit the call you "
                        "intended, on its own. To act on a web page: "
                        "<browse>https://...</browse> to open one, "
                        "<click>visible text</click>, <type>words</type>, "
                        "<press>Enter</press>. To read one: "
                        "<read>https://...</read> for its text, "
                        "<fetch_html>https://...</fetch_html> for its markup. "
                        "To compute, fetch or write files: <py>...</py>. "
                        "To run a command: <cmd>...</cmd>. Pick the one that "
                        "fits what you were already doing — do not switch "
                        "toolset. Then answer in one short line.]"
                    )})
                    self._trim_history()
                    continue
                break

            # Only execute the first fresh tool per response. Multiple tools in
            # one reply cause bursts (e.g. five <search> tags at once) and can
            # overwhelm the model with parallel observations.
            name, params = fresh_tools[0]
            tool_key = json.dumps([name, params], sort_keys=True)
            executed_calls.add(tool_key)
            any_tool_ran = True
            extra = fresh_tools[1:]

            # There are tools to execute
            self.history.append({"role": "assistant", "content": reply})
            # The visible part of this reply was already logged to the session
            # store above; logging the raw form here too would write a second,
            # near-identical entry for every tool turn and let RAG retrieve
            # both. It would also put literal tool-call tags into retrievable
            # context, which the model can echo back as if they were its own.
            # The tool's observation is logged below — that is the part of a
            # tool turn worth recalling later.
            consecutive_tool_rounds += 1

            self.output_fn(f"  [Tool: {name}]")
            if name in _WEB_TOOLS:
                web_used = True
            if user_refused_this_turn:
                # Observed live: browser_open was denied at the domain prompt,
                # and the very next round the model ran `open -a 'Google
                # Chrome'` through run_command and reported success. The
                # sandbox is a denylist, so `open` was never going to stop it
                # — but no denylist should have to. A refusal is about the
                # action the user was asked about, not the tool that happened
                # to ask, so once one is given nothing else runs this turn.
                observation = (
                    "Blocked: the user declined this action earlier in this "
                    "turn. Do not attempt it by other means. Tell them it was "
                    "not done.")
                self.output_fn(f"  [Safety] {observation}")
            else:
                # A model that scrolls without finding what it wants will
                # scroll forever — the page never changes enough to satisfy
                # it, and each <scroll/> is a distinct tag the dedup logic
                # can't catch. Cap it per turn so the agent falls back to
                # reading whatever is visible instead of scrolling into a
                # loop that only stops when max_tool_rounds runs out.
                if name == "browser_scroll" and scrolls_this_turn >= _MAX_SCROLLS_PER_TURN:
                    observation = (
                        f"Scrolled {scrolls_this_turn} times already this turn. "
                        f"Stop scrolling and work with the page text you can see. "
                        f"If the information is not on this page, try a different "
                        f"URL or a web search instead."
                    )
                else:
                    observation = self._execute_tool(name, params)
                    if name == "browser_scroll":
                        scrolls_this_turn += 1
                if learn.is_user_refusal(observation):
                    user_refused_this_turn = True
            local_telemetry.log_event(
                "tool", name=name, ok=not learn.sounds_like_tool_error(observation),
                result=observation,
            )
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
            # Anonymous tool-error counter for telemetry (no content, just +1).
            if learn.sounds_like_tool_error(observation):
                try:
                    from symbio.app import telemetry
                    _tstate = telemetry.load_state()
                    telemetry.record_error(_tstate)
                    telemetry.save_state(_tstate)
                except Exception:
                    pass
            # A call is remembered as "already done" so a repetitive model
            # can't loop on it — but only once it actually worked. A call that
            # failed because of a precondition the model then fixed (clicking
            # before the page was open, the common case) must be allowed to
            # run again, or the retry is silently swallowed by the fresh-tool
            # filter and the turn ends on a bare "Clicking the button." with
            # nothing clicked. Retries stay capped so a call that keeps
            # failing still can't spin.
            # A refusal is the exception: no retry turns a "no" into a "yes",
            # and re-running the call just puts the same confirmation prompt
            # in front of the user again. Observed live — one denied
            # browser_open asked twice in a single turn.
            if (learn.sounds_like_tool_error(observation)
                    and not learn.is_user_refusal(observation)):
                failed_calls[tool_key] = failed_calls.get(tool_key, 0) + 1
                if failed_calls[tool_key] < _MAX_TOOL_RETRIES:
                    executed_calls.discard(tool_key)
            # A rate limit is the one failure whose correct response is the
            # SAME call again, and it fell between every existing case: the
            # script that received a 429 ran fine, so sounds_like_tool_error is
            # False, failed_calls never counted it, and the key stayed in
            # executed_calls — where the fresh-tool filter drops the reissue as
            # "already done". The model could not repeat the call at all, and
            # no amount of extra tool rounds would have changed that. Measured
            # live 2026-08-27 against an API that 429s twice before serving.
            #
            # web_search is excluded because its observation is page content:
            # results *about* rate limiting are not the search being limited.
            elif (name != "web_search"
                    and learn.is_rate_limited(observation)
                    and not learn.is_user_refusal(observation)):
                rate_limited_calls[tool_key] = rate_limited_calls.get(tool_key, 0) + 1
                if rate_limited_calls[tool_key] < _MAX_RATE_LIMIT_RETRIES:
                    executed_calls.discard(tool_key)
                    # Honour the server's own number so the retry is not spent
                    # arriving too early, but never hand the CLI a long stall.
                    wait = learn.retry_after_seconds(observation) or 1.0
                    wait = min(wait, _MAX_RATE_LIMIT_WAIT)
                    self.output_fn(
                        f"  [Retry] rate limited; waiting {wait:g}s before "
                        f"allowing the same call again.")
                    time.sleep(wait)
            # Track browser-action failures so we can force a retry if the
            # model tries to end the turn without another tool tag.
            # A refusal is excluded here for the same reason: this nudge tells
            # the model "do not end the turn until the request is completed",
            # which against a denied request is an instruction to keep asking.
            if (name in _BROWSER_ACTION_TOOLS
                    and learn.sounds_like_tool_error(observation)
                    and not learn.is_user_refusal(observation)):
                pending_browser_error = observation
            else:
                pending_browser_error = None

            self.output_fn(f"  [Observation] {observation.replace(chr(10), chr(10) + '  ')}")
            timings["tools_ms"] = (time.perf_counter() - gen_start) * 1000
            # Web results must ground the answer: tell the model to answer from
            # them and admit it couldn't find the answer, so an 8B model can't
            # just regurgitate its confident prior instead of using the results.
            # Appended after the user-facing print so the grounding reaches the
            # model/history but not the terminal line.
            if name in _WEB_TOOLS:
                observation += (
                    "\n\n[Answer ONLY from the results above. If they do not state "
                    "the answer, say plainly that you could not find it — do not "
                    "repeat your earlier claim or guess.]"
                )
            if learn.is_user_refusal(observation):
                # Without this the turn ends on the sentence the model wrote
                # *before* the tool ran — "Opening apple.com for you." — which
                # reports an action the user had just blocked. Seen live: the
                # denial changed nothing about the final answer.
                observation += (
                    "\n\n[The user declined this. It did NOT happen. Say plainly "
                    "that you did not do it because they declined, and do not "
                    "describe it as done or in progress. Do not try another way.]"
                )
            # Present results in Hermes-style <tool_response> JSON so the model
            # learns the structured format, while keeping a plain-text fallback
            # for models that have not switched to Hermes calls yet.
            hermes_name = _internal_to_hermes_name(name)
            response_json = json.dumps({"name": hermes_name, "content": observation}, ensure_ascii=False)
            self.history.append({"role": "user", "content": (
                f"[System observation: {observation}]\n"
                f"<tool_response>{response_json}</tool_response>"
            )})
            # Log the tool result to the session store so RAG can retrieve
            # past observations (e.g. "what did the page say last time").
            self.session_store.log("tool", f"{name}: {observation}")
            self._trim_history()

            # Pure navigation is complete the moment the page is open. Stop
            # here instead of re-prompting the model with the freshly-loaded
            # page — that re-prompt is what makes it auto-click elements it sees
            # ("Continue", "Stream now"). The model's pre-tool prose already
            # stands as the user-facing reply. Requests that also want info
            # ("go to cloudflare pricing") are not navigation-only, so the loop
            # continues and the model can read on.
            if (name == "browser_open"
                    and _is_navigation_only(user_input)
                    and not learn.sounds_like_tool_error(observation)):
                break

        # A turn must never end with nothing on screen. The blank-reply nudge
        # above fires at most once per turn; when the retry comes back blank as
        # well — a Qwen3 thinking block that never closed, most often — the loop
        # simply breaks and the user is handed their prompt back with no answer
        # and no explanation. Seen live 2026-08-24 on a run_remote turn: the
        # command ran, its output was sitting right there in the observation,
        # and the whole session printed one assistant line for three exchanges.
        #
        # Deliberately NOT written into final_display: no answer was produced,
        # and the research-note path below must not memorize this as one.
        if not _is_substantive(final_display):
            self.output_fn(
                f"{self.config['assistant_name']:8}: "
                + ("(No reply — I could not summarize it. The tool output above "
                   "is the result.)" if any_tool_ran else
                   "(No reply — the model returned only internal reasoning.)"))
            self.logger.info("Turn ended with no visible reply.")

        timings["total_ms"] = (time.perf_counter() - turn_start) * 1000
        self.last_turn_timings = timings
        self.logger.info(f"Timings: {timings}")

        if is_correction:
            # The corrected answer is now in history; capture and maybe retrain.
            self._learn_from_correction()
        elif web_used and final_display:
            # Web research produced an answer: remember durable knowledge so
            # it is retrievable later and trained into the weights on digest.
            # But don't memorize a suspect answer — one given under a doubt/
            # verification followup, or one that hedged or couldn't find the
            # fact — that would bake an unverified (or non-)fact into the
            # weights. Let the user confirm it first.
            suspect = (_looks_like_verification_followup(user_input)
                       or learn.sounds_unsure(final_display))
            if not suspect:
                # File the note under the real question, not a bare "check
                # online" command — otherwise the note gets titled "Learned:
                # check online" and trains a command string into the weights.
                research_q = self._search_subject or user_input
                note = learn.remember_research(research_q, final_display, self.config)
                if note:
                    self.retriever.invalidate_cache()
                    self.output_fn(f"  [Learn] Remembered research: {note.name}")
