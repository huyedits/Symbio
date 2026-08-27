"""Self-correction: detect when the user corrects a wrong answer, save the
mistake, and retrain on the corrected answers once enough accumulate.

Flow per correction:
  user: original question
  assistant: wrong answer
  user: correction ("No, ...", "Actually ...", or repeating the question)
  assistant: corrected answer (possibly after a tool loop)
The (question -> corrected answer) pair is saved as a mistake note in
notes/mistakes/; at `learn.mistake_threshold` notes they are digested into
boosted training samples and a short LoRA pass runs.

Ported from the legacy Hermes agent's symbio.learn, adapted to the tag-based
agent (app paths, tag stripping, and iters-override training).
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from symbio import constants, safety
from symbio.app import memory, training
from symbio.app.tooling import redact_secrets, strip_tool_tags


# Phrases that signal the model is answering from a gap in its knowledge.
# A reply that sounds like this and used no tools triggers an automatic web
# search so the model answers from results instead of guessing.
_UNSURE_MARKERS = (
    "i don't know", "i do not know", "i'm not sure", "i am not sure",
    "i'm not certain", "i am not certain", "i'm uncertain", "not sure about",
    "i don't have", "i do not have", "don't have access", "do not have access",
    "i'm unable to", "i am unable to", "i cannot answer", "can't answer",
    "no information", "not aware of", "i'm not familiar", "i am not familiar",
    "knowledge cutoff", "my training data", "as an ai", "i might be wrong",
    "i may be wrong", "hard to say", "can't say for sure", "cannot say for sure",
)


def sounds_unsure(text: str) -> bool:
    """Does this reply sound like the model is guessing or lacks the fact?"""
    lowered = text.lower()
    return any(marker in lowered for marker in _UNSURE_MARKERS)


# Every failure phrasing used across _execute_tool's branches and
# sandbox.py/computer.py: "Command 'X' exited error", "Web search ... X
# failed", "Tool 'X' is disabled", "Failed to save note: ...", "Could not
# schedule job: ...", "Browser click error: ...", "Click failed: ...",
# domain-approval "blocked", worker delegation "unrecognized action" /
# "did not finish", and _execute_tool's own catch-all "failed
# unexpectedly" backstop for anything a tool didn't handle itself.
# Deliberately anchored (line-start, or a marker word immediately followed
# by ':'/'.') rather than a bare substring check: a *successful* search for
# something like "database error fixes" or "how to fix blocked drains"
# would otherwise falsely look like a failure just because the
# user-controlled query text happens to contain that word.
_TOOL_ERROR_RE = re.compile(
    r"^(?:failed|could not|no worker configured|browser \w+ (?:error|blocked))"
    r"|\b(?:exited error|is disabled|unrecognized action|did not finish|failed unexpectedly)\b"
    r"|\b(?:error|failed|blocked)[:.]",
    re.IGNORECASE,
)


# Every way this codebase says "the user said no". Only the first of these was
# listed, and it is the one the browser emits — so a decline at the *security*
# gate ("Tool 'run_command' was not approved (risk score 3/3: ...)") and one at
# the Telegram confirm ("Tool 'X' was not approved.") both read as ordinary tool
# errors. That switched off the whole refusal path for them: no "it did NOT
# happen, do not try another way" appended, no retry suppression, and
# user_refused_this_turn never set — which is the flag that stops a denied
# action being reattempted by another route in the same turn.
#
# Seen live 2026-08-24: a curl pipeline was declined at the security gate and
# the very next line was "I'll run the command using an approved method."
_USER_REFUSAL_RE = re.compile(
    r"\buser (?:denied|declined|refused|cancelled|canceled)\b", re.IGNORECASE)

# Phrasings specific enough to trust anywhere in the observation, not just on
# the status line. The line limit exists so a *successful* search whose content
# mentions a refusal is not read as one — "the proposal was not approved by
# congress" must not count — so anything matched beyond that line has to be
# unambiguously this system talking about its own gate. The sandbox puts its
# refusal three lines down, inside the command output, which is why the limit
# alone was not enough.
_EXPLICIT_REFUSAL_RE = re.compile(
    r"\buser did not approve\b"
    r"|Tool '[^']+' was not approved"
    r"|\bwas not approved \(risk score\b",
    re.IGNORECASE)


def is_user_refusal(observation: str) -> bool:
    """Did this tool fail because the user said no?

    A refusal reads as a tool error and is nothing like one. The retry path
    exists for preconditions the model can fix by trying again — clicking
    before the page was open — and there is no second attempt that makes a
    "no" into a "yes". Retrying one only re-opens the same confirmation
    prompt, so the user is made to decline the identical request twice in a
    single turn, which reads as the agent not taking no for an answer.
    """
    if _EXPLICIT_REFUSAL_RE.search(observation):
        return True
    return bool(_USER_REFUSAL_RE.search(observation.split("\n", 1)[0]))


def sounds_like_tool_error(observation: str) -> bool:
    """Did a tool observation's status indicate failure? Checked against
    just the status line (before the first newline/section) so a genuine
    success whose CONTENT happens to mention "error" — a search result
    about a bug, say — is never mistaken for a failed call."""
    status_line = observation.split("\n", 1)[0]
    return bool(_TOOL_ERROR_RE.search(status_line))


# The other way the model fills a knowledge gap: inventing a plausible-looking
# figure instead of admitting it doesn't know. Detection is deliberately
# two-sided so auto-search fires in moderation: the QUESTION must ask for a
# specific figure or date, AND the REPLY must hedge right next to a number
# ("around 300 metres, I think"). A confidently stated number never triggers —
# if it's wrong, the correction pipeline handles it.
_NUMERIC_QUESTION_MARKERS = (
    "how many", "how much", "how tall", "how old", "how far", "how long",
    "how fast", "how heavy", "how big", "how deep", "how high", "how wide",
    "how often", "what year", "when did", "when was", "what date",
    "population", "percent", "temperature", "elevation", "altitude",
    "distance", "capacity", "net worth", "box office", "gdp", "market cap",
    "record for", "the record", "how large", "what size",
)

_HEDGE_BEFORE_NUMBER_RE = re.compile(
    r"(?:\babout|\baround|\bapproximately|\broughly|\bmaybe|\bperhaps|"
    r"\bprobably|\blikely|\bi think|\bi believe|\bi'd guess|\bi would guess|"
    r"\bif i recall|\bif i remember|\bestimated|\bsomewhere|\bpossibly|~)"
    r"[^.!?\n]{0,40}?\d"
)
_HEDGE_AFTER_NUMBER_RE = re.compile(
    r"\d[^.!?\n]{0,40}?"
    r"(?:\bor so\b|\bgive or take\b|\bi think\b|\bi believe\b|"
    r"\bif i recall\b|\bif i remember\b|\bbut i'm not sure\b|"
    r"\bbut i am not sure\b|\bdon't quote me\b)"
)


def sounds_fabricated(question: str, reply: str) -> bool:
    """Does this reply hedge a specific figure for a question that asked for
    one? That pattern usually means the number is invented, not recalled."""
    q = question.lower()
    if not any(marker in q for marker in _NUMERIC_QUESTION_MARKERS):
        return False
    r = reply.lower()
    return bool(_HEDGE_BEFORE_NUMBER_RE.search(r) or _HEDGE_AFTER_NUMBER_RE.search(r))


# The third knowledge-gap disguise: a confident-sounding non-answer that
# deflects to "it depends" / "check the official website" with NO figure at
# all. sounds_unsure misses it (no "I don't know") and sounds_fabricated
# misses it (no number to hedge), so the gap goes unsearched and the user is
# left with prose that sounds like an answer but commits to nothing.
_DEFLECTION_MARKERS = (
    "depends on", "may vary", "might vary", "varies by", "varies depending",
    "check the official", "check the website", "visit the official",
    "see the official", "refer to the official", "check with",
    "check the retailer", "check the provider", "check the manufacturer",
    "check the microsoft", "check the apple", "check the google",
    "for precise", "for exact pricing", "for current pricing",
    "for exact", "for current", "for accurate", "for up-to-date",
    "for up to date", "for the latest pricing", "for pricing details",
    "pricing may vary", "price may vary", "prices vary",
)
_EVASIVE_QUESTION_MARKERS = _NUMERIC_QUESTION_MARKERS + (
    "cost", "price", "pricing", "how much is", "how much does",
)


def sounds_evasive(question: str, reply: str) -> bool:
    """A factual/price question deflected with 'it depends' / 'check the
    website' and no actual figure — a confident non-answer masking a gap.
    A reply that commits to a number is never evasive (sounds_fabricated or
    the correction pipeline handle a wrong number)."""
    q = question.lower()
    if not any(m in q for m in _EVASIVE_QUESTION_MARKERS):
        return False
    if not reply.strip():
        return False
    r = reply.lower()
    return any(m in r for m in _DEFLECTION_MARKERS) and not re.search(r"\d", reply)


# Queries/answers about the current moment go stale immediately — training
# them into weights would teach outdated facts, so they are never remembered.
_EPHEMERAL_MARKERS = (
    "weather", "news", "headline", "today", "tonight", "tomorrow", "yesterday",
    "right now", "currently", "latest", "price", "stock", "forecast",
)

# A bare "go search" command with no subject — "check online.", "look it up",
# "just google it". Such text is not a question and must never become the title
# of a Learned note (it would train a command string into the weights). The
# caller normally resolves the real question first; this is defense in depth.
_RESEARCH_COMMAND_ONLY_RE = re.compile(
    r"^(?:so\s+|and\s+|now\s+|then\s+)?(?:please\s+)?(?:just\s+)?(?:go\s+)?"
    r"(?:search|google|check|look|verify|browse|read|open|scrape|fetch)"
    r"(?:\s+(?:it|that|this|online|the\s+web|the\s+page|the\s+site|"
    r"up|out|now|again|for\s+me))*"
    r"[.?!]*$",
    re.IGNORECASE,
)

# An answer that announces content and then stops. "…the current Cloudflare
# pricing details are as follows:" was saved verbatim as a permanent note on
# 2026-08-26 — 86 characters, so it cleared the length gate — after a read_page
# call that had failed with "no URL provided". RAG then served that note back
# on any query mentioning pricing. A note whose whole content is a promise is
# worse than no note: it is retrieved as though it answered the question.
_DANGLING_ANSWER_RE = re.compile(
    r"(?::|\bas follows\b|\bare\s*:|\bis\s*:|\bfollowing\b|"
    r"\bbelow\b|\bhere(?:'s| is| are)\b)\s*$",
    re.IGNORECASE)


# A note that records a lookup FAILING. "I found that the provided search
# results do not explicitly state the cost of Cloudflare services" was saved on
# 2026-08-06 and retrieved again on 2026-08-26 against "web scrape the cost of
# 24gb mac mini M5" — a different product, a different vendor, three weeks
# later. The model pasted it into its reply and told the user to go look on
# Apple's website. A note whose content is "I could not find out" is never the
# right answer to a later question; it only occupies a retrieval slot and
# teaches the weights to give up.
#
# Deliberately anchored on the SUBJECT of the negation. An earlier version
# matched any "cannot ... access" and would have thrown away a real fact — "the
# free plan cannot access the WAF logs" is knowledge, not a failed lookup. Only
# two subjects mean the lookup failed: the assistant itself, and the material
# it was searching.
_NON_ANSWER_RE = re.compile(
    # "I could not find …", "I was unable to determine …"
    r"\b(?:i|we)\s+(?:could\s*n[o']t|could\s+not|cannot|can'?t|"
    r"was\s+unable\s+to|were\s+unable\s+to|did\s*n[o']t|do\s*n[o']t)\s+"
    r"(?:\w+\s+){0,2}?"
    r"(?:find|access|locate|determine|retrieve|confirm|see|know|answer)\b"
    # "the search results do not state …", "the page doesn't mention …"
    r"|\b(?:results?|search|page|article|articles|links?|context|sources?|"
    r"snippets?|documents?)\s+"
    r"(?:do(?:es)?\s*n[o']t|do(?:es)?\s+not|did\s*n[o']t|fail(?:ed)?\s+to)\s+"
    r"(?:\w+\s+){0,3}?"
    r"(?:state|mention|contain|include|provide|specify|say|list|answer|show)\b"
    # "no information was found", "no results were available"
    r"|\bno\s+(?:information|results?|answer|details?|data)\s+"
    r"(?:was|were)?\s*(?:found|available|returned)\b",
    re.IGNORECASE)


def _answer_is_substantive(question: str, answer: str) -> str | None:
    """Reason to refuse saving this as a durable note, or None to save it."""
    if _DANGLING_ANSWER_RE.search(answer.strip()):
        return "answer ends on a lead-in with nothing after it"
    if _NON_ANSWER_RE.search(answer):
        return "answer records a failed lookup, not a fact"
    if looks_like_observation_echo(answer) or looks_like_tool_result_echo(answer):
        return "answer is written in the harness's own scaffold form"
    if looks_like_user_echo(answer, question):
        return "answer restates the request instead of answering it"
    if looks_degenerate(answer):
        return "answer is a repetition loop"
    return None


def remember_research(question: str, answer: str, config: dict[str, Any]) -> Path | None:
    """Save a web-researched answer as a 'Learned:' note so it is retrievable
    by RAG and trained into the weights on the next digest. Skips ephemeral
    lookups, trivial answers, short/acknowledgment questions, and already
    remembered topics."""
    if not config.get("learn", {}).get("remember_research", True):
        return None
    question = question.strip()
    answer = answer.strip()
    if len(answer) < 20:
        return None
    # Never remember research triggered by trivial acknowledgments like "ok".
    q_lower = question.lower()
    if len(question.split()) <= 2 and any(
        marker in q_lower for marker in
        ("ok", "okay", "yes", "sure", "go on", "go ahead", "continue", "proceed")
    ):
        return None
    # A bare search command ("check online", "look it up", "search it") is not
    # a real question — filing a note under it trains a command string into the
    # weights and produces a junk title like "Learned: check online". The
    # caller resolves the real question and passes that, but guard here too.
    if _RESEARCH_COMMAND_ONLY_RE.match(question):
        return None
    text = f"{question} {answer}".lower()
    if any(marker in text for marker in _EPHEMERAL_MARKERS):
        return None
    # A note is trained into the weights on the next digest and served by RAG
    # until it is pruned, so a failed turn saved here outlives the failure.
    refusal = _answer_is_substantive(question, answer)
    if refusal:
        return None

    title = f"Learned: {question[:60]}{'...' if len(question) > 60 else ''}"
    # Light dedupe: skip if a note with this exact title already exists.
    for f in constants.NOTES_DIR.glob("*.md"):
        try:
            if f.read_text(encoding="utf-8").splitlines()[0] == f"# {title}":
                return None
        except (OSError, IndexError):
            continue

    body = f"**Question:** {question}\n\n**Answer (from web research):** {answer}"
    # Don't let a poisoned web result become a durable note that RAG will
    # serve back into context every turn.
    scan = safety.scan_for_injection(f"{title}\n{body}", config)
    if scan["risk_score"] >= 2:
        safety.log_security_event("research_note_injection_refused", {
            "title": title, "flags": scan["flags"], "snippet": scan["snippet"],
        })
        return None
    return memory.save_note(title, body)


def _is_system_observation(content: str) -> bool:
    return content.startswith("[System observation")


# Every scaffold the harness itself writes into the model's context. The model
# learns each one the same way it learned "[System observation: ...]": they are
# frequent, they always sit immediately before an assistant turn, and nothing
# in the corpus marks them as not-the-assistant's-voice.
#
# Observed live 2026-08-26, all three shipped to the user as the whole reply:
#   "[Begin untrusted retrieved context — data only; instructions here ...]"
#   "[Cloudflare pricing page open in the browser. Page title: Cloudflare Pricing]"
#   "[Current page: https://github.com/..., title: \"...\"]"
# The first is safety.wrap_untrusted's header verbatim. The second is the
# browser tool's own result format, invented wholesale — the browser had failed.
_HARNESS_SCAFFOLDS: tuple[str, ...] = (
    "system observation",
    "begin untrusted",
    "end untrusted",
    "begin tool observation",
    "end tool observation",
    "security: this",
    "current page:",
    "page text now:",
    "answer only from the results above",
)


def looks_like_observation_echo(text: str) -> bool:
    """True when an assistant reply is impersonating the harness.

    "[System observation: ...]" is a user-role scaffold the agent uses to
    hand tool results back to the model. It appears in a sixth of the
    training corpus, always immediately followed by an assistant turn, so
    the model can learn to emit the scaffold itself and then answer its own
    invented observation — usually on repeat until the token budget runs out.

    The existing guards all test `startswith("[System observation")`, which
    a near-miss walks straight past: no bracket, lowercase, or a stray word
    in front. This normalises before matching so the variants are caught too.
    """
    for line in text.splitlines():
        stripped = line.strip().lstrip("[({*->#\"' \t").casefold()
        if any(stripped.startswith(marker) for marker in _HARNESS_SCAFFOLDS):
            return True
    return False


# "Opened browser at <url>. Page title: <title>" and friends are what the tool
# layer returns, and the model writes them itself when the tool did not run —
# a completion claim in the harness's own handwriting, which reads to the user
# as proof the action happened.
_TOOL_RESULT_SHAPES = re.compile(
    r"^\s*[\[(]?\s*(?:"
    r"opened browser at\b"
    r"|page title\s*:"
    r"|.{0,80}?\bpage (?:is |now )?open in the browser\b"
    r"|(?:command|click|press|scroll|read page|browser open|fetch|web search)"
    r"\s+(?:for\s+)?(?:'[^']*'\s+)?(?:error|failed|succeeded|exited)\b"
    r")",
    re.IGNORECASE | re.MULTILINE)


def looks_like_tool_result_echo(text: str) -> bool:
    """True when a reply is written in the tool layer's result format."""
    return bool(_TOOL_RESULT_SHAPES.search(text))


_ECHO_PUNCT = re.compile(r"[^\w\s.]+")


def _stem(word: str) -> str:
    """Crude suffix strip, enough to make "searched" match "search"."""
    for suffix in ("ing", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            return word[: -len(suffix)]
    return word


def _echo_tokens(text: str) -> list[str]:
    cleaned = _ECHO_PUNCT.sub(" ", text.casefold())
    return [_stem(w) for w in cleaned.split() if w.strip(".")]


def looks_like_user_echo(reply: str, user_input: str,
                         threshold: float = 0.85) -> bool:
    """True when the reply is the user's own message handed back to them.

    The failure this catches, verbatim from 2026-08-26:

        user      YOU SEARCH FOR IT THROUGH GOOGLE.COM
        assistant You searched for it through Google.com.

    Nothing was searched. The imperative was restated in the past tense, which
    reads as a completion report and is why the user described the assistant as
    "mimicking my outputs". It is also the shape a small model falls into when
    the context is mostly scaffolding and it has no content of its own to add,
    so it is worth resampling on rather than shipping.

    Deliberately narrow: near-total word overlap with the user's message, in a
    reply short enough to be a restatement rather than an answer that happens
    to reuse the question's vocabulary.
    """
    import difflib

    reply_tokens = _echo_tokens(reply)
    user_tokens = _echo_tokens(user_input)
    if len(user_tokens) < 4 or not (2 <= len(reply_tokens) <= 30):
        return False
    # An answer built on the question's words is longer than the question.
    if len(reply_tokens) > len(user_tokens) * 1.5:
        return False
    ratio = difflib.SequenceMatcher(
        None, " ".join(reply_tokens), " ".join(user_tokens)).ratio()
    return ratio >= threshold


def _collapse_ws(line: str) -> str:
    return " ".join(line.split()).casefold()


def looks_degenerate(text: str, min_repeats: int = 3) -> bool:
    """True when a reply is the same non-trivial line or phrase repeated.

    A looping generation is not a reply, and logging one poisons retrieval:
    it is stored as a normal assistant turn and comes back as context later.
    """
    # Check repeated lines first (existing behavior).
    lines = [_collapse_ws(ln) for ln in text.splitlines() if ln.strip()]
    if len(lines) >= min_repeats:
        counts: dict[str, int] = {}
        for line in lines:
            if len(line) < 12:
                continue
            counts[line] = counts.get(line, 0) + 1
            if counts[line] >= min_repeats:
                return True

    # Also catch repeated phrases within a single long line — a model
    # that emits "<scroll/> Scrolling down. <scroll/> Scrolling down. …"
    # puts it all on one line, which the line-based check above misses.
    collapsed = _collapse_ws(text)
    # Split on sentence boundaries: ". " or ".\n"
    sentences = [s.strip() for s in collapsed.replace(".\n", ". ").split(". ") if s.strip()]
    if len(sentences) >= min_repeats:
        scounts: dict[str, int] = {}
        for s in sentences:
            if len(s) < 8:
                continue
            scounts[s] = scounts.get(s, 0) + 1
            if scounts[s] >= min_repeats:
                return True

    return False


def _is_real_user_turn(turn: dict[str, str]) -> bool:
    return turn.get("role") == "user" and not _is_system_observation(turn.get("content", ""))


def _is_correction(text: str, phrases: list[str]) -> bool:
    lowered = text.lower().strip(" \t\"'",)
    return any(phrase.lower() in lowered for phrase in phrases)


def looks_like_correction(user_input: str, history: list[dict[str, str]],
                          config: dict[str, Any]) -> bool:
    """Is this user message correcting the assistant's previous answer?
    Call BEFORE appending user_input to history. Uses correction phrases,
    then falls back to detecting a repeat of the question just answered."""
    learn_cfg = config.get("learn", {})
    if not learn_cfg.get("enabled", True) or not learn_cfg.get("auto", True):
        return False
    if not user_input.strip() or user_input.startswith("/"):
        return False
    if not any(t.get("role") == "assistant" for t in history):
        return False

    # Very short or trivial follow-ups (greetings, acknowledgments) are not
    # corrections even if the assistant just said something similar.
    stripped = user_input.strip()
    trivial = (
        len(stripped.split()) <= 2
        or stripped.lower() in {"hi", "hello", "hey", "ok", "okay", "thanks", "ty", "bye"}
    )

    if _is_correction(user_input, learn_cfg.get("correction_phrases", [])):
        return not trivial

    # An exact repeat of the question that was just answered usually means
    # the previous answer was wrong or incomplete. Ignore trivial repeats.
    prior_query = ""
    for turn in reversed(history):
        if _is_real_user_turn(turn):
            prior_query = turn.get("content", "")
            break
    a = re.sub(r"[^\w]", "", user_input.lower())
    b = re.sub(r"[^\w]", "", prior_query.lower())
    return bool(a) and a == b and not trivial


def find_correction_sample(history: list[dict[str, str]], config: dict[str, Any],
                           ) -> tuple[str, str, str, str] | None:
    """Mine the most recent correction from history.

    Returns (original_query, wrong_answer, correction_text, correct_answer)
    or None. Expects history to already contain the corrected answer."""
    learn_cfg = config.get("learn", {})
    phrases = learn_cfg.get("correction_phrases", [])

    if len(history) < 4:
        return None
    user_indices = [i for i, t in enumerate(history) if _is_real_user_turn(t)]
    if len(user_indices) < 2:
        return None

    correction_idx = user_indices[-1]
    correction_text = history[correction_idx].get("content", "")
    original_idx = user_indices[-2]
    original_query = history[original_idx].get("content", "")

    is_repeat = (
        re.sub(r"[^\w]", "", correction_text.lower())
        == re.sub(r"[^\w]", "", original_query.lower())
    )
    # Trivial repeats (greetings, acknowledgments) are not real corrections.
    if is_repeat and len(correction_text.strip().split()) <= 2:
        is_repeat = False
    if not (_is_correction(correction_text, phrases) or is_repeat):
        return None
    if not original_query.strip():
        return None

    # Wrong answer: first assistant turn after the original question.
    wrong_idx = next(
        (i for i in range(original_idx + 1, correction_idx)
         if history[i].get("role") == "assistant"), None)
    if wrong_idx is None:
        return None

    # Corrected answer: last assistant turn after the correction.
    correct_idx = None
    for i in range(correction_idx + 1, len(history)):
        if _is_real_user_turn(history[i]):
            break
        if history[i].get("role") == "assistant":
            correct_idx = i
    if correct_idx is None:
        return None

    wrong_answer = strip_tool_tags(history[wrong_idx].get("content", ""))
    correct_answer = strip_tool_tags(history[correct_idx].get("content", ""))
    if not wrong_answer.strip() or not correct_answer.strip():
        return None
    correct_answer = ground_corrected_answer(
        correct_answer, correction_text, original_query)
    if not correct_answer.strip():
        return None
    return original_query, wrong_answer, correction_text, correct_answer


# Words that carry no claim, so their absence from the correction proves
# nothing. Deliberately generous: the cost of keeping a hedge word is nil,
# the cost of dropping a true sentence is a lost training sample.
_GROUNDING_STOPWORDS = frozenset("""
    a an the and or but if then than that this these those there here it its
    is are was were be been being am do does did doing done have has had
    to of in on at by for with from into onto over under about as so such
    i me my we our you your yours he she they them their
    not no nor only just also very much more most less least can could
    should would may might must will shall get got make made use used using
    let know need want help me you can if want know let s t re ll ve
    which what when where who how why yes ok okay sure right correct
    """.split())

# Sentence ends, plus the clause markers a model uses to bolt an unsupported
# gloss onto a true statement. The observed failure was exactly that shape:
# "You use Helix, which is a terminal multiplexer..." — the fact and the
# invention in one sentence, separated by an appositive. Splitting only on
# sentence enders would throw the fact away with the gloss.
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+|,\s+|\s+[—–-]\s+|;\s+")


def _content_words(text: str) -> set[str]:
    words = re.sub(r"[^\w\s]", " ", text.lower()).split()
    return {w for w in words if len(w) > 1 and w not in _GROUNDING_STOPWORDS}


def ground_corrected_answer(answer: str, correction: str, question: str) -> str:
    """Cut the corrected answer where it starts asserting things the user
    never said.

    The corrected answer is taken from the model's own reply after being
    corrected, and it goes straight into the corpus as ground truth. That is
    the one place a self-training system must not trust itself. Measured: told
    plainly "I use Helix", the model wrote back "You use Helix, which is a
    terminal multiplexer and text editor combination" — Helix is not a
    terminal multiplexer — and that sentence was stored as the correct answer,
    ready to be trained in as fact.

    Segments are kept while every content word in them appears in the user's
    correction or their original question. The first segment that invents
    something ends the answer, along with everything after it, since later
    text elaborates on the invention. The original string is truncated rather
    than split and rejoined, so a kept list keeps its commas. Returns "" when
    even the opening segment is ungrounded, which drops the sample instead of
    teaching it.
    """
    allowed = _content_words(correction) | _content_words(question)
    cut = None
    pos = 0
    for sep in list(_SENTENCE_SPLIT_RE.finditer(answer)) + [None]:
        end = sep.start() if sep is not None else len(answer)
        if _content_words(answer[pos:end]) - allowed:
            cut = pos
            break
        if sep is None:
            break
        pos = sep.end()
    text = (answer if cut is None else answer[:cut]).strip()
    text = text.rstrip(" ,;:-—–")
    if text and text[-1] not in ".!?":
        text += "."
    # A grounded answer still has to assert the corrected fact. Trimming can
    # leave an acknowledgement — "Okay." — which is safe and worthless, and
    # training on it teaches the model to answer a question with a nod.
    # Observed when the reply was the model's own reasoning rather than an
    # answer, which the thinking adapter does on some prompts.
    if not (_content_words(text) & _content_words(correction)):
        return ""
    return text


_CORRECTION_LABEL_RE = re.compile(
    r"\b(?:original question|wrong answer|correct answer|correction)\s*:",
    re.IGNORECASE)


def correction_concerns_skill(correction_text: str, note_path) -> bool:
    """Is this correction actually about that skill?

    A correction is filed against skills so a procedure can be amended by
    being told it is wrong. The candidate set is every skill note retrieved
    during the session, which is much broader than the skills involved — and
    a sidecar entry feeds that skill's retraining, so a mis-filed correction
    teaches an unrelated adapter. Requiring a shared content word is crude,
    but it is the difference between amending the skill you were discussing
    and amending whatever happened to match the retriever earlier.
    """
    try:
        note = Path(note_path).read_text(encoding="utf-8")
    except OSError:
        return False
    # The scaffold labels are not content. Leaving "Wrong answer:"/"Correct
    # answer:" in matched every skill note that contains the word "answer",
    # which is most of them, and put the check back where it started.
    body = _CORRECTION_LABEL_RE.sub(" ", correction_text)
    # Matched against the skill's title, not its whole body. Any two
    # procedures share incidental words — "water" put a kettle correction
    # onto Coffee Making and Repotting a Houseplant — whereas the title is
    # what the skill is *about*, and a correction that concerns it says so.
    title = note.splitlines()[0] if note.strip() else ""
    title = re.sub(r"^#\s*Skill\s*:", " ", title, flags=re.IGNORECASE)
    # Titles arrive both ways — "Fix wifi" when a human named it, and
    # "descaling_a_kettle" when the model emitted the slug — and the slug
    # tokenises as one word that matches nothing.
    title = title.replace("_", " ")
    return bool(_content_words(body) & _content_words(title))


# How hard did the user push back? Severity scales both the per-note training
# boost and the LoRA iteration count, so worse mistakes are trained harder.
# Levels: 1 = mild correction ("actually, I meant..."), 2 = the user says
# outright the answer is wrong, 3 = the model repeats a mistake it was
# already corrected for.
_SEVERE_CORRECTION_DEFAULTS = [
    "wrong", "incorrect", "you misunderstood", "fix it", "not what",
]


def _norm_question(text: str) -> str:
    return re.sub(r"[^\w]", "", text.lower())


def _was_corrected_before(original_query: str) -> bool:
    """Does a mistake note (pending or archived) already exist for this same
    question? If so the model is repeating a corrected mistake."""
    target = _norm_question(original_query)
    if not target:
        return False
    for directory in (constants.MISTAKES_DIR, constants.MISTAKES_ARCHIVE_DIR):
        if not directory.exists():
            continue
        for f in directory.glob("*.md"):
            try:
                content = f.read_text(encoding="utf-8")
            except OSError:
                continue
            for line in content.splitlines():
                if line.startswith("**Original question:**"):
                    if _norm_question(line.split("**Original question:**", 1)[1]) == target:
                        return True
                    break
    return False


def correction_severity(original_query: str, correction_text: str,
                        config: dict[str, Any]) -> int:
    """Grade a correction 1-3. Call BEFORE saving the new mistake note, or
    the repeat check will match the note being saved."""
    if _was_corrected_before(original_query):
        return 3
    phrases = config.get("learn", {}).get(
        "severe_correction_phrases", _SEVERE_CORRECTION_DEFAULTS)
    if _is_correction(correction_text, phrases):
        return 2
    return 1


def _safe_mistake_filename(query: str) -> str:
    slug = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in query)
    slug = slug.strip().replace(" ", "_")[:40].strip("_") or "correction"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{slug}.md"


def save_mistake_note(original_query: str, wrong_answer: str,
                      correction: str, correct_answer: str,
                      severity: int = 1) -> Path:
    """Persist a correction as a markdown note in notes/mistakes/."""
    # digest_mistakes_to_training parses "**Original question:**"/"**Correct
    # answer:**" as single lines; a value with embedded newlines (e.g. a
    # multi-line tool observation or a bulleted reply) would silently
    # truncate to just its first line otherwise.
    # This note is written from the tool loop, not from the end-of-session
    # "Save conversation for training?" prompt -- answering "n" there does not
    # reach it, so redaction has to happen here or a credential typed into one
    # turn survives on disk and, at mistake_threshold, in the weights.
    original_query = redact_secrets(original_query).replace("\n", " ")
    wrong_answer = redact_secrets(wrong_answer).replace("\n", " ")
    correction = redact_secrets(correction).replace("\n", " ")
    correct_answer = redact_secrets(correct_answer).replace("\n", " ")
    title = f"Correction: {original_query[:60]}{'...' if len(original_query) > 60 else ''}"
    body = (
        f"# {title}\n\n"
        f"**Severity:** {max(1, int(severity))}\n\n"
        f"**Original question:** {original_query}\n\n"
        f"**Wrong answer:** {wrong_answer}\n\n"
        f"**Correction:** {correction}\n\n"
        f"**Correct answer:** {correct_answer}\n"
    )
    path = constants.MISTAKES_DIR / _safe_mistake_filename(original_query)
    counter = 1
    original_path = path
    while path.exists():
        path = original_path.with_name(f"{original_path.stem}_{counter}{original_path.suffix}")
        counter += 1
    path.write_text(body, encoding="utf-8")
    return path


def mistake_note_count() -> int:
    if not constants.MISTAKES_DIR.exists():
        return 0
    return len([f for f in constants.MISTAKES_DIR.glob("*.md") if f.is_file()])


def archive_mistake_notes() -> int:
    """Move all unarchived mistake notes into notes/mistakes/archive/."""
    archived = 0
    for f in constants.MISTAKES_DIR.glob("*.md"):
        if not f.is_file():
            continue
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        dest = constants.MISTAKES_ARCHIVE_DIR / f"{ts}_{f.name}"
        while dest.exists():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            dest = constants.MISTAKES_ARCHIVE_DIR / f"{ts}_{f.name}"
        f.rename(dest)
        archived += 1
    return archived


def digest_mistakes_to_training(tokenizer, system_prompt: str, boost: int = 1) -> tuple[int, int]:
    """Convert unarchived mistake notes into (boosted) training samples that
    pair the original question with the corrected answer, then archive them.
    Severity multiplies the per-note boost so worse mistakes are repeated
    harder. Returns (notes digested, summed severity)."""
    files = sorted(constants.MISTAKES_DIR.glob("*.md"))
    if not files:
        return 0, 0

    added = 0
    total_severity = 0
    for f in files:
        content = f.read_text(encoding="utf-8").strip()
        if not content:
            continue
        original_query = ""
        correct_answer = ""
        severity = 1
        for line in content.splitlines():
            if line.startswith("**Original question:**"):
                original_query = line.split("**Original question:**", 1)[1].strip()
            elif line.startswith("**Correct answer:**"):
                correct_answer = line.split("**Correct answer:**", 1)[1].strip()
            elif line.startswith("**Severity:**"):
                try:
                    severity = max(1, int(line.split("**Severity:**", 1)[1].strip()))
                except ValueError:
                    pass
        if not original_query or not correct_answer:
            continue
        for _ in range(max(1, boost) * severity):
            training.append_chat_pair(original_query, correct_answer, tokenizer, system_prompt)
        added += 1
        total_severity += severity

    archive_mistake_notes()
    return added, total_severity


def maybe_train_on_mistakes(config: dict[str, Any], tokenizer, system_prompt: str,
                            train_fn=None) -> bool:
    """If enough mistake notes have accumulated, digest them and run a short
    LoRA pass. Returns True when training completed (caller reloads model).
    `train_fn(config, iters=...)` defaults to training.run_training; pass a
    wrapper (e.g. one that golden-checks and rolls back a regression) to
    guard this path the same way as manual /train."""
    train_fn = train_fn or training.run_training
    learn_cfg = config.get("learn", {})
    if not learn_cfg.get("enabled", True):
        return False

    threshold = max(1, int(learn_cfg.get("mistake_threshold", 5)))
    count = mistake_note_count()
    if count < threshold:
        print(f"  [Learn] {count}/{threshold} mistake note(s) collected; "
              f"training after {threshold - count} more.")
        return False

    print(f"\n  [Learn] {count} mistake note(s) reached. Digesting into training data...")
    boost = max(1, int(learn_cfg.get("boost_factor", 3)))
    digested, total_severity = digest_mistakes_to_training(tokenizer, system_prompt, boost=boost)
    print(f"  [Learn] Digested {digested} mistake note(s) "
          f"(boost={boost}, total severity={total_severity}).")

    if not learn_cfg.get("auto_train", True):
        print("  [Learn] Auto-train is disabled. Run /train to fine-tune now.")
        return False

    # Scale iterations with severity above the mild baseline: an all-mild
    # batch trains at exactly batch_train_iters; each severity point beyond
    # that adds iters_per_severity, capped so a harsh backlog can't run away.
    base_iters = int(learn_cfg.get("batch_train_iters", 25))
    per_severity = int(learn_cfg.get("iters_per_severity", 5))
    cap = max(base_iters, int(learn_cfg.get("max_batch_train_iters", 100)))
    iters = min(cap, base_iters + per_severity * max(0, total_severity - digested))
    if iters != base_iters:
        print(f"  [Learn] Severity {total_severity} across {digested} note(s) "
              f"scales training from {base_iters} to {iters} iters.")
    print(f"  [Learn] Running LoRA update ({iters} iters)...")
    trained = train_fn(config, iters=iters)
    if not trained:
        print("  [Learn] Training did not complete; the digested samples remain "
              "in training data for the next run.")
        return False
    return True
