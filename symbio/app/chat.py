"""The interactive chat REPL: slash commands, the autonomous agent loop,
and the growth loop (memory nudges, exit flush, cron surfacing)."""

import gc
import hashlib
import json
import logging
import re
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
from mlx_lm.sample_utils import make_sampler

from rag import Retriever
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


# Short verification follow-ups ("are you sure?", "check again") carry almost
# no signal, so a small low-temperature model derails — reciting its identity
# or regurgitating an earlier topic instead of re-examining the answer it just
# gave. Detected here so the turn loop can inject a contextual nudge.
_VERIFICATION_FOLLOWUPS = {
    # Direct re-check requests.
    "are you sure", "you sure", "sure", "sure?", "really", "really?",
    "certain", "certain?", "is that right", "is that correct", "that right",
    "check again", "again", "verify", "double check", "doublecheck",
    "recheck", "redo", "rethink", "reconsider",
    # Subtler hedging / doubt a user fires back at a factual claim.
    "i don't think so", "i dont think so",
    "not sure", "not so sure", "not convinced", "not really",
    "not sure about that", "not sure about this",
    "doesn't sound right", "doesnt sound right", "sounds wrong", "sounds off",
    "are you certain", "are you positive",
    "are you sure about that", "are you sure about this",
    "how do you know", "how do you know that", "how are you sure", "how so",
    "hmm", "wait", "wait what", "wait really", "huh",
    "nope", "nah", "i doubt it", "doubt it", "skeptical",
}

# Cues that also work at the END of a longer pushback, e.g.
# "that's qatar, not america. check again". Bare ambiguous single tokens
# (again/sure/really/wait/hmm) are excluded here — they only qualify as short
# standalone follow-ups, not as trailing cues, so a sentence that merely ends
# with one of them isn't mistaken for doubt. "you think" is deliberately
# excluded so "what do you think" (asking for an opinion) never trips.
_VERIFICATION_TRAILING = {
    "check again", "are you sure", "is that right", "is that correct",
    "double check", "double-check", "doublecheck", "recheck", "rethink",
    "reconsider", "verify that", "verify it", "verify this",
    "are you certain", "are you positive",
    "are you sure about that", "are you sure about this",
    "how do you know", "how do you know that", "how are you sure",
    "doesn't sound right", "doesnt sound right", "sounds wrong", "sounds off",
    "doesn't sound right to me",
    "i don't think so", "i dont think so", "not so sure", "not convinced",
    "not sure about that", "not sure about this",
    "i doubt it",
}


def _looks_like_verification_followup(text: str) -> bool:
    """Is this a follow-up asking the model to re-examine its last answer?

    Fires on (a) a short ≤4-word hedging phrase (with an optional leading "no"
    negation, so "no. check again" matches), or (b) a longer pushback (≤20
    words) that ends with an unambiguous verification cue like "check again".
    Normal turns and standalone questions are unaffected.
    """
    stripped = text.strip().lower()
    if not stripped or stripped.startswith("/"):
        return False
    # (a) short exact hedging phrase, allowing a leading negation.
    short = re.sub(r"^no[.,]?\s+", "", stripped).strip("?.!\"' ")
    if len(short.split()) <= 4 and short in _VERIFICATION_FOLLOWUPS:
        return True
    # (b) longer pushback ending in an unambiguous verification cue.
    tail = re.sub(r"[?.!\"']+$", "", stripped).strip()
    if len(tail.split()) <= 20 and any(
        tail == p or tail.endswith(" " + p) for p in _VERIFICATION_TRAILING
    ):
        return True
    return False


# Imperative "search for me" phrasings. When the user explicitly tells the model
# to search and it waffles (describes searching instead of calling <search>),
# the turn loop forces a web search so the user isn't left stranded. Word-
# boundary anchored so "research" (which contains "search") never matches.
_EXPLICIT_SEARCH_RE = re.compile(
    r"\b(?:search\s+it|search\s+online|search\s+the\s+web|search\s+now|"
    r"google\s+it|look\s+it\s+up|just\s+search|check\s+online|check\s+the\s+web|"
    r"look\s+online|verify\s+online)\b",
    re.IGNORECASE,
)

# Function/filler words (len>=4) that never constitute a search subject on
# their own. Words shorter than 4 chars (it, has, the, is, of, on, in, at,
# do, to, be, a, an, as, by, he, we, me, my, no, so, or, up, us) are dropped
# by the len>=4 signature filter in _subjectless_search_command, so they
# don't need listing here. A "check online" command is subjectless when,
# after dropping the matched command phrase, no substantive word survives —
# e.g. "it has check online" -> "it has" -> no signature word -> inject the
# previous question; "check online who won the world cup" -> "world" survives
# -> the model can bind the topic itself.
_SEARCH_FILLER_STOPWORDS = {
    "does", "much", "will", "would", "should", "could", "that", "this",
    "with", "from", "about", "they", "them", "have", "been", "were",
    "your", "their", "when", "where", "which", "there", "here", "than",
    "then", "what", "whats", "just", "into", "onto", "over", "sure",
    "know", "think", "thought", "said", "told", "means", "again",
    "never", "always", "very", "more", "most", "some", "such", "only",
    "also", "even", "because", "though", "however", "anyway", "really",
    "actually", "convinced", "done", "went", "getting", "going",
    "already", "still", "happened", "happening", "happens", "dont",
}


def _subjectless_search_command(text: str) -> bool:
    """A "go search" command whose own words carry no searchable subject —
    "check online.", "it has check online", "search it", "just google it".
    The small model has nothing to bind the search to and will hallucinate an
    unrelated query, so the caller injects the previous unanswered question
    as the subject instead.

    Detects this by dropping the matched command phrase (e.g. "check online")
    and checking whether any substantive word (len>=4, not a function-word
    stopword) survives in the remainder. "it has" -> none -> subjectless;
    "who won the world cup" -> "world" -> has a subject.
    """
    t = text.strip().lower().rstrip("?.!")
    if not t or len(t.split()) > 6:
        return False
    m = _EXPLICIT_SEARCH_RE.search(t)
    if not m:
        return False
    remainder = (t[:m.start()] + " " + t[m.end():]).strip()
    signature = {w for w in re.findall(r"[a-z0-9]{4,}", remainder)
                 if w not in _SEARCH_FILLER_STOPWORDS}
    return not signature


# Stopwords excluded when comparing a model-emitted search query to the
# resolved subject, so a coincidence like "does" in both doesn't make a
# hallucinated query look on-topic.
_QUERY_STOPWORDS = {
    "does", "much", "will", "would", "should", "could", "that", "this",
    "with", "from", "about", "they", "them", "have", "has", "been",
    "were", "your", "their", "when", "where", "which", "there", "here",
    "than", "then", "what", "whats",
}


def _queries_overlap(model_query: str, subject: str) -> bool:
    """Does the model's search query mention any signature word (len>=4, not a
    stopword) from the subject question? If not, the model likely hallucinated
    an unrelated topic and the caller should override the query."""
    subj_words = {w for w in re.findall(r"[a-z0-9]{4,}", subject.lower())
                  if w not in _QUERY_STOPWORDS}
    if not subj_words:
        return True  # can't tell; trust the model
    model_lower = (model_query or "").lower()
    return any(w in model_lower for w in subj_words)

# Greeting-only messages ("hi", "hi caine", "hello there", "hey", "good
# morning"). A blank reply to one must never trigger a web search (we'd
# "search" the word "hi" and get nonsense) and must never be "learned" as
# research. Matched by tokenizing so "hi" can't false-fire on "this" etc.
_GREETING_WORDS = {
    "hi", "hello", "hey", "yo", "sup", "howdy", "hiya", "heya",
    "morning", "evening", "afternoon", "greetings", "whatsup", "whats",
    "hola", "aloha",
}
_GREETING_FILLERS = {
    "there", "you", "all", "guys", "man", "dude", "bro", "mate", "good",
    "caine", "symbio", "bot", "ai",
}


def _is_greeting(text: str) -> bool:
    t = text.strip().lower()
    if not t:
        return False
    words = re.findall(r"[a-z']+", t)
    if not words or len(words) > 4:
        return False
    return all(w in _GREETING_WORDS or w in _GREETING_FILLERS for w in words)


# A user turn that asks Caine to *do* something with a tool — open/navigate to a
# page, click/press/type/scroll, or read a URL. When the model blanks on one of
# these (no tool call, no prose), the turn must not die silent: the user is
# waiting for an action, so we nudge the model to emit the right tool rather than
# leaving them with nothing. Matches the same verbs browser_followup uses, plus
# an explicit URL, so "now go to cloudflare pricing" / "read the webpage at …"
# are caught.
def _is_action_request(text: str) -> bool:
    t = text.strip().lower()
    if not t:
        return False
    if "http" in t or ".com" in t:
        return True
    return any(m in t for m in (
        "go to ", "open ", "browse ", "click ", "press ", "type ", "scroll ",
        "read the webpage", "read the page", "read this", "navigate", "visit ",
    ))


# A pure navigation request: "open/go to/visit <site>" with NO ask for info or
# further on-page action. Once the page is open, the task is done — re-prompting
# the model with the freshly-loaded page only invites it to auto-click elements
# it happens to see (the "Stream now"/"Continue" problem). Used to stop the
# tool loop right after a successful browser_open so the model's pre-tool prose
# (e.g. "Opening Apple.com.") stands as the reply. Requests that also want info
# ("go to cloudflare pricing", "open X and tell me…") are NOT navigation-only,
# so the loop continues and the model can read/summarize the page.
def _is_navigation_only(text: str) -> bool:
    t = text.strip().lower()
    if not t:
        return False
    if not any(m in t for m in ("open ", "go to ", "visit ", "browse ", "navigate ")):
        return False
    if any(m in t for m in (
        "click", "press", "type ", "scroll", "read", "tell", "what", "find",
        "show", "price", "pricing", "cost", "how much", "list", "summary",
        "summari", "extract", "who", "when", "where", "why", "and then", "then ",
    )):
        return False
    return True


def _last_exchange(history: list[dict[str, str]]) -> tuple[str | None, str | None]:
    """Return (last real user question, last assistant answer) from history.

    Skips system observations and injected [System: …] nudges so a verification
    follow-up can be given the full prior context inline rather than relying on
    the model to dig it out of history itself.
    """
    answer = None
    ans_idx = None
    for i in range(len(history) - 1, -1, -1):
        turn = history[i]
        if turn.get("role") == "assistant" and turn.get("content", "").strip():
            answer = turn["content"].strip()
            ans_idx = i
            break
    if answer is None:
        return None, None
    question = None
    for i in range(ans_idx - 1, -1, -1):
        turn = history[i]
        if turn.get("role") != "user":
            continue
        content = turn.get("content", "")
        if content.startswith("[System"):
            continue
        if content.strip():
            question = content.strip()
            break
    return question, answer


# Per-turn user affect: a lightweight read of how the user is feeling from one
# message, so Caine can adapt tone/directness instead of waffling cheerfully
# while the user is frustrated. Heuristic only — caps/punctuation + a small
# lexicon — good enough to spot the states that should change how it replies
# (frustration/impatience most of all), not a fine-grained emotion model.

_AFFECT_FRUSTRATION = {
    "frick", "insufferable", "annoying", "stupid", "wtf", "ugh", "bruh",
    "damn", "ridiculous", "useless", "broken", "hate", "crap", "garbage",
    "trash", "awful", "not working", "doesn't work", "fix this", "serious",
    "for real", "cmon", "c'mon", "ffs",
}
_AFFECT_IMPATIENCE = {
    "still not", "why won't", "why is it", "why are you", "come on", "just do",
    "enough", "yet again", "i keep", "still doesn't", "again and again",
}
_AFFECT_CONFUSED = {
    "confused", "don't get", "dont get", "don't understand", "dont understand",
    "what do you mean", "i'm lost", "im lost", "make sense", "lost",
    "what is going on", "not following",
}
_AFFECT_GRATEFUL = {
    "thank", "thanks", "thank you", "appreciate", "love ya", "love you",
    "awesome", "great job", "nice work", "perfect", "<3",
}
_AFFECT_HAPPY = {
    "lol", "haha", "yay", "love this", "sweet", "amazing", "woohoo",
    "🔥", "😊", "😄", "🥳", "😂",
}
_AFFECT_CURIOUS = {
    "what if", "i wonder", "how about", "what happens", "curious",
    "let's try", "brainstorm", "idea: ",
}
# Exasperated rhetorical closes — the message ends with one of these (often
# with a single raised-voice emphasis word, e.g. "what are you DOINGG").
# Matched against a repeat-normalized, trailing-punctuation-stripped copy of
# the lowercased text, and required at the *end* so a genuine "what are you
# doing tomorrow" does not trip it.
_AFFECT_EXASPERATION = (
    "what are you doing",
    "what are you even doing",
    "what are you even",
    "what are you talking about",
    "what are you on about",
    "what is wrong with you",
    "what is your problem",
    "are you kidding me",
    "are you kidding me right now",
    "are you kidding",
    "are you serious right now",
    "are you serious",
    "why are you like this",
    "are you actually doing",
    "are you done",
    "are you finished",
)
# Same phrases with repeated letters collapsed, so misspelled emphasis
# ("doingg", "soooo") matches them and legit doubles ("kidding" -> "kiding")
# line up on both sides of the comparison.
_AFFECT_EXASPERATION_NORM = tuple(
    re.sub(r"(.)\1+", r"\1", p) for p in _AFFECT_EXASPERATION
)

# Command-start verbs, to tell an all-caps imperative ("TELL ME SEARCH IT
# ONLINE") from an all-caps question ("WHO IS THE CEO") when there's no
# question mark. Question auxiliaries (do/does/is/are) are deliberately
# excluded so "DO YOU KNOW" doesn't read as a command.
_CMD_START_RE = re.compile(
    r"^\s*(?:tell|search|find|get|show|give|make|run|open|fix|check|look|go|"
    r"stop|help|explain|list|try|call|fetch|write|read|delete|create|start|"
    r"shut|send|bring|take|install|build|deploy|test|restart|update|download|"
    r"copy|clear|reset|quit|exit|please|just|now)\b", re.IGNORECASE,
)

# Tone adaptation for the detected mood now lives in the system prompt (the
# model reads the mood itself and adjusts), so there is no per-turn nudge
# dict here.


def infer_user_affect(text: str) -> str:
    """Best-guess the user's current mood from a single message.

    Returns one of: frustrated, impatient, confused, grateful, happy, curious,
    neutral. Coarse by design — it only needs to spot the states that should
    change how Caine responds (mainly frustration/impatience), not be a
    fine-grained emotion model.
    """
    if not text or not text.strip():
        return "neutral"
    lower = text.lower()
    letters = [c for c in text if c.isalpha()]
    caps_ratio = (sum(1 for c in letters if c.isupper()) / len(letters)) if letters else 0.0
    # Raised voice: heavy caps across enough letters, paired with exclamations
    # so acronyms ("NASA", "CEO") don't read as shouting.
    shouting = caps_ratio > 0.45 and len(letters) >= 5
    excl = text.count("!") >= 2

    # Impatience (persistence over time: "still not", "why won't", "i keep")
    # is checked before frustration so "still not working" reads as impatient
    # rather than angry — the two share some negativity but differ in tone.
    if any(w in lower for w in _AFFECT_IMPATIENCE):
        return "impatient"
    # Exasperated rhetorical close ("...what are you DOINGG", "are you kidding
    # me"): normalize exaggerated repeats ("doingg" -> "doing") so misspelled
    # emphasis still matches, then test whether the message ends with one of
    # these phrases. End-position avoids "what are you doing tomorrow" (a
    # genuine question) reading as frustration.
    _exasp_norm = re.sub(r"(.)\1+", r"\1", lower)
    _exasp_tail = re.sub(r"[?!.\s]+$", "", _exasp_norm)
    if any(_exasp_tail.endswith(p) for p in _AFFECT_EXASPERATION_NORM):
        return "frustrated"
    # All-caps counts as frustration only when it's a command or exclamatory
    # (raised voice), not an all-caps question — "WHO IS THE CEO" is just a
    # question typed in caps, "TELL ME SEARCH IT ONLINE" is a frustrated order.
    imperative = bool(_CMD_START_RE.match(text))
    if any(w in lower for w in _AFFECT_FRUSTRATION) or (shouting and (imperative or excl)):
        return "frustrated"
    if any(w in lower for w in _AFFECT_CONFUSED):
        return "confused"
    if any(w in lower for w in _AFFECT_GRATEFUL):
        return "grateful"
    if any(w in lower for w in _AFFECT_HAPPY):
        return "happy"
    if any(w in lower for w in _AFFECT_CURIOUS):
        return "curious"
    return "neutral"


# --- Model-emitted mood tag -------------------------------------------------
# The model itself reads the user's tone (it was trained on human language, so
# it catches anger/frustration/sadness/joy that a regex misses — e.g. a single
# raised-voice emphasis word like "DOINGG"). It emits <mood>tag</mood> at the
# start of its reply; StreamingStripper + strip_tool_tags hide that tag from
# the user, and the turn loop parses it to surface [Mood: tag]. The lexicon
# heuristic (infer_user_affect) is only a fallback for turns where the model
# omits the tag.
_MOOD_TAG_RE = re.compile(r"<mood>\s*([a-zA-Z]+)\s*</mood>", re.IGNORECASE)
_VALID_MOODS = {
    "angry", "frustrated", "impatient", "confused", "sad", "anxious",
    "grateful", "happy", "excited", "curious", "neutral",
}


def _persist_health_report(session_id: str, report: dict[str, Any]):
    """Write the session health report to both a per-session file and a
    rolling 'latest' file inside sessions/."""
    constants.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    session_path = constants.SESSIONS_DIR / f"{session_id}_health.json"
    report["_persisted"] = True
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
    constants.LOG_DIR.mkdir(parents=True, exist_ok=True)
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

    def set_label(self, label: str):
        with self._lock:
            self.label = label

    def start(self):
        if self._thread is not None:
            return
        if not self.active:
            # No animation without a TTY, but silence is not the alternative:
            # a turn that prints nothing between the prompt and the reply looks
            # like a hang, and the wait here is tens of seconds on an 8B. One
            # static line costs nothing and cannot be mistaken for frozen.
            sys.stdout.write(f"  {self.label}\n")
            sys.stdout.flush()
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



def _adapter_trained_at() -> datetime | None:
    """mtime of the adapter weights — the last time a LoRA run wrote them."""
    weights = constants.ADAPTER_DIR / "adapters.safetensors"
    if weights.exists():
        try:
            return datetime.fromtimestamp(weights.stat().st_mtime)
        except OSError:
            return None
    return None


def _adapter_iters() -> int | None:
    """iters recorded in the last training run's adapter_config.json."""
    cfg = constants.ADAPTER_DIR / "adapter_config.json"
    if not cfg.exists():
        return None
    try:
        return int(json.loads(cfg.read_text(encoding="utf-8")).get("iters", 0)) or None
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _fmt_ago(then: datetime, now: datetime | None = None) -> str:
    """Compact '2h ago'-style relative time."""
    now = now or datetime.now()
    secs = int((now - then).total_seconds())
    if secs < 60:
        return "just now"
    mins = secs // 60
    if mins < 60:
        return f"{mins}m ago"
    hrs = mins // 60
    if hrs < 48:
        return f"{hrs}h ago"
    return f"{hrs // 24}d ago"


def learn_progress_line(config: dict[str, Any]) -> str:
    """One-line summary of the self-finetune loop: mistake counter state.

    e.g. '3/5 mistakes to next tune', '5/5 mistakes — tuning due', or
    'learn: off' when the loop is disabled."""
    learn_cfg = config.get("learn", {}) or {}
    if not learn_cfg.get("enabled", True):
        return "learn: off"
    threshold = max(1, int(learn_cfg.get("mistake_threshold", 5)))
    count = learn.mistake_note_count()
    suffix = "" if learn_cfg.get("auto_train", True) else " (auto-train off)"
    if count >= threshold:
        return f"{count}/{threshold} mistakes — tuning due{suffix}"
    return f"{count}/{threshold} mistakes to next tune{suffix}"


def adapter_status_value(config: dict[str, Any], adapter_loaded: bool) -> str:
    """Legible adapter + learn state, e.g.

    'loaded (trained 2h ago, 50 iters) · 3/5 mistakes to next tune'
    'none (base) · 3/5 mistakes to next tune'
    """
    progress = learn_progress_line(config)
    if not adapter_loaded:
        return f"none (base) · {progress}"
    bits: list[str] = []
    trained = _adapter_trained_at()
    if trained is not None:
        bits.append(f"trained {_fmt_ago(trained)}")
    iters = _adapter_iters()
    if iters is not None:
        bits.append(f"{iters} iters")
    detail = f" ({', '.join(bits)})" if bits else ""
    return f"loaded{detail} · {progress}"


def print_banner(config: dict[str, Any], adapter_loaded: bool, dataset_size: int,
                 output_fn=print):
    note_count = len(list(constants.NOTES_DIR.glob("*.md")))
    output_fn("\n" + "=" * 50)
    output_fn(f"  {config['assistant_name'].upper()} — PERSONAL CHAT-FINETUNE CLI")
    output_fn(f"   Model  : {config['model_name']}")
    output_fn(f"   User   : {config['user_name']}")
    output_fn(f"   LoRA   : {adapter_status_value(config, adapter_loaded)}")
    output_fn(f"   Data   : {dataset_size:,} bytes")
    output_fn(f"   Notes  : {note_count}")
    output_fn("-" * 50)
    output_fn("Commands: /quit  /save  /train  /retrain  /train_worker  /resume  /golden [audit|prune]  /security  /learn  /forget_last  /status  /prune  /selfcheck  /setup  /compact  /help")
    output_fn("         /run <cmd>  /note [title]  /notes  /index-notes [--force]  /auto-index on|off  /new-skill <name>  /skills  /skill-adapters  /digest  /cron  /config  /archive  /restore")
    output_fn("         /build-mcp <name> | <description>  /mcp-tools  /hosts  /telemetry on|off  /feedback <text>")
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

# Browser actions that can sensibly be retried with a different target.
_BROWSER_ACTION_TOOLS = {
    "browser_click", "browser_type", "browser_scroll", "browser_press",
}

# How many times one identical tool call may be attempted in a single turn.
# 2 lets the model retry a call whose precondition it has since fixed (the
# click that failed before the page was open) without letting a call that
# keeps failing spin for the whole round budget.
_MAX_TOOL_RETRIES = 2

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
        # The system-prompt prefill runs on a daemon thread at boot so the user
        # can read the banner and type their first message while the ~4000-token
        # system+tools prefix is processed into the KV cache, instead of paying
        # that cost as a blocking "warming prompt cache…" step. _await_prefill()
        # joins it before any generation (so the model is never used by two
        # threads at once).
        self._prefill_thread: threading.Thread | None = None
        # A persisted prompt cache is a few hundred MB of safetensors, and
        # reading it used to start only once load() had finished — so the two
        # slowest parts of boot ran back to back when they have nothing to say
        # to each other. The read is pure file I/O and the weight load is
        # mostly decompress-and-place, so the read is started *before* load()
        # and overlaps it; by the time the model is resident the bytes are
        # already in the page cache. See _start_prompt_cache_prefetch.
        self._prefetch_thread: threading.Thread | None = None
        # (cache, metadata) once the prefetch has read the file, or None. The
        # cache is deliberately left unevaluated here: materializing it would
        # allocate GPU buffers on a second thread while load() is allocating
        # its own, and this codebase has a standing kernel-panic problem with
        # concurrent Metal clients. mx.eval stays on the prefill thread, which
        # only runs after load() has returned.
        self._prefetched_cache: tuple[list, dict] | None = None
        self.enabled_groups: set[str] = set(
            config.get("tools", {}).get("enabled_groups", [])
        )
        # Simple timing record for the most recent turn; surfaced in /status
        # and used by front-ends to report latency.
        self.last_turn_timings: dict[str, float | None] = {}
        self.system_prompt = prompts.build_system_prompt(
            config["assistant_name"], config["user_name"]
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

    def _unload_model(self):
        """Drop the in-process model and release its GPU buffers.

        The caller is responsible for getting a model back before the next
        generation (_reload_model, _wake_headmaster, or _ensure_model_loaded,
        all of which reload from nothing).
        """
        # Whatever the KV cache holds refers to weights we are about to drop.
        self._prompt_cache = None
        self._cached_prompt_ids = None
        # A prefetch that nobody consumed is a few hundred megabytes with no
        # reader, which is the opposite of what it was added for. Dropping the
        # weights is the point at which it can no longer be claimed.
        self._take_prefetched_cache()
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
        self._status("  [Dispatch] Headmaster going to sleep (unloading 8B model)...")
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

        So the wake mirrors the boot path. The cache read starts before load()
        and overlaps it, and the prefill runs afterwards — which, since the
        model, adapter and prompt are all unchanged since boot, is a signature
        hit on the persisted file rather than a real prefill.
        """
        if getattr(self, "model", None) is not None:
            return
        self._status("  [Dispatch] Headmaster waking up (reloading 8B model)...")
        self._start_prompt_cache_prefetch()
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
                self.config, verbose=True, output_fn=self.output_fn
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
                if (self.config.get("agent", {}).get("persist_prompt_cache", True)
                        and self._load_persisted_prompt_cache(system_ids)):
                    return
                self._prompt_cache = make_prompt_cache(self.model)
                # max_tokens=0 processes the prompt into the KV cache and stops
                # before generating any output tokens.
                for _ in generate_step(
                    mx.array(system_ids),
                    self.model,
                    max_tokens=0,
                    sampler=self.sampler,
                    prompt_cache=self._prompt_cache,
                ):
                    pass
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
                self._log_info(f"Prompt cache prefill failed: {e!r}")
                self._prompt_cache = None
                self._cached_prompt_ids = None
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
        # So this now blocks. It costs ~24s on a first boot, once: the whole
        # point of persisting the cache is that every later boot loads the file
        # instead of prefilling, and _start_prompt_cache_prefetch overlaps even
        # that read with the model load. Paying 24s once to make the feature
        # real beats an unblocking optimization that never produced a cache.
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
        """
        adapter_sig = "none"
        if self.adapter_loaded:
            weights = constants.ADAPTER_DIR / "adapters.safetensors"
            try:
                st = weights.stat()
                adapter_sig = f"{st.st_mtime_ns}:{st.st_size}"
            except OSError:
                adapter_sig = "missing"
        ids_bytes = ",".join(map(str, system_ids)).encode()
        return {
            "model_name": str(self.config.get("model_name", "")),
            "adapter_sig": adapter_sig,
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

        What this thread must NOT do is touch the GPU. Materializing the cache
        here would put a second Metal client in the same window as the weight
        load, which is the shape that panics IOGPUFamily on this hardware. So
        it stops at the read: load_prompt_cache leaves the arrays lazy, and the
        mx.eval that actually allocates stays on the prefill thread, after the
        load has finished. The signature check needs a tokenizer that does not
        exist yet, so it also waits — a mismatched file costs one wasted read,
        which is exactly what it cost before.
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
                cache, meta = load_prompt_cache(str(path), return_metadata=True)
                self._prefetched_cache = (cache, meta)
            except Exception:
                # Nothing is owed here. A failure leaves _prefetched_cache None
                # and _load_persisted_prompt_cache reads the file itself, which
                # is what it did before this existed — including the unlink of
                # a truncated file, so the error is still handled, just later.
                self._prefetched_cache = None

        self._prefetch_thread = threading.Thread(target=_prefetch, daemon=True)
        self._prefetch_thread.start()

    def _take_prefetched_cache(self) -> tuple[list, dict] | None:
        """Hand over the prefetched (cache, metadata), waiting for it if needed.

        Consumed exactly once: a KV cache is mutated in place by generation, so
        handing the same object to a second caller would give two readers one
        buffer. After this returns the prefetch is spent and a later miss falls
        back to reading the file.
        """
        thread = self._prefetch_thread
        if thread is not None:
            thread.join()
            self._prefetch_thread = None
        taken = self._prefetched_cache
        self._prefetched_cache = None
        return taken

    def _load_persisted_prompt_cache(self, system_ids: list[int]) -> bool:
        """Restore the warmed system-prefix cache from disk.

        Returns True if the cache was loaded and is safe to use. Reading a few
        hundred MB off an SSD is roughly an order of magnitude cheaper than
        re-running the prefill through the model, which is the whole point —
        and cheaper still when _start_prompt_cache_prefetch already read it
        underneath the weight load, in which case there is nothing left to wait
        for here.
        """
        path = constants.PROMPT_CACHE_FILE
        if not path.exists():
            return False
        want = self._prompt_cache_signature(system_ids)
        prefetched = self._take_prefetched_cache()
        if prefetched is not None:
            cache, meta = prefetched
        else:
            try:
                cache, meta = load_prompt_cache(str(path), return_metadata=True)
            except Exception as e:
                # A truncated or version-mismatched file is not worth keeping.
                self._log_info(f"Prompt cache unreadable, discarding: {e}")
                path.unlink(missing_ok=True)
                return False
        if any(meta.get(k) != v for k, v in want.items()):
            # The model, adapter or prompt changed since it was written.
            path.unlink(missing_ok=True)
            return False
        # Loading a cache is not the same as being able to use one. A file
        # written by an earlier process can carry arrays bound to an MLX stream
        # that does not exist here, and nothing notices until generation, which
        # then dies mid-turn. Touching the state now moves that failure to the
        # one place equipped to handle it: prefill just runs normally instead.
        try:
            mx.eval([c.state for c in cache])
        except Exception as e:
            self._log_info(f"Prompt cache unusable in this process, discarding: {e}")
            path.unlink(missing_ok=True)
            return False
        self._prompt_cache = cache
        self._cached_prompt_ids = list(system_ids)
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
                enable_thinking=training.THINKING_ENABLED,
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
            max_tokens = int(agent_cfg["max_reply_tokens"])

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
                    self._prompt_cache = make_prompt_cache(self.model)
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

        def _emit(text: str):
            if self.stream_chunk_fn is None or not text:
                return
            nonlocal shown, answer_prefix_emitted
            if not shown:
                shown = True
                # Stop the spinner and clear its line before the first visible
                # chunk, otherwise the spinner thread keeps overwriting the
                # streaming reply.
                spinner.stop()
            if not answer_prefix_emitted and not text.startswith(tooling.REASONING_MARKER):
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
            for response in self.stream_fn(
                self.model, self.tokenizer, feed, max_tokens=max_tokens,
                sampler=self.sampler, prompt_cache=self._prompt_cache,
            ):
                text_parts.append(response.text)
                raw_acc += response.text
                gen_ids.append(response.token)
                gen_tokens += 1
                spinner.set_gen_tokens(gen_tokens)
                if stripper is not None:
                    safe = stripper.feed(response.text)
                    if safe:
                        _emit(safe)
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
                    if opens <= closes:
                        break
        except BaseException:
            # The real MLX cache may already be mutated beyond what our
            # bookkeeping reflects (interrupted mid-token) — never trust a
            # stale cache after this; the next call rebuilds it from zero.
            self._prompt_cache = None
            self._cached_prompt_ids = None
            raise
        finally:
            self._indexing_now = False
            spinner.stop()

        if stripper is not None:
            tail = stripper.finish()
            if tail:
                _emit(tail)
            # If nothing was emitted during the stream (e.g. the entire reply
            # was a tool tag or was held back as ambiguous), make sure a
            # newline is still sent so the terminal cursor is in the right
            # place and any later non-streamed print starts on a fresh line.
            if self.stream_chunk_fn is not None:
                if not shown and chunk_prefix:
                    self.stream_chunk_fn(chunk_prefix + "\n")
                else:
                    self.stream_chunk_fn("\n")

        if timings is not None:
            timings["gen_ms"] = (time.perf_counter() - gen_start) * 1000
            if timings.get("ttft_ms") is None:
                timings["ttft_ms"] = timings["gen_ms"]

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
            enable_thinking=training.THINKING_ENABLED,
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
                note=self._last_train_note or "")
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

    # ---- Slash commands ----

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
                                     note="manual /wildcards run")
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

        elif cmd.startswith("/note"):
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
                    self.config["assistant_name"], self.config["user_name"]
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
                        self.config, verbose=True, output_fn=self.output_fn
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

    def _agent_turn(self, user_input: str):
        # The boot system-prompt prefill may still be running on its background
        # thread. Join it before any model use this turn (canary, memory flush,
        # tool summarizers, _generate_reply) so the model is never used by two
        # threads at once. No-op once the prefill has finished.
        self._await_prefill()
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
            self.config["assistant_name"], self.config["user_name"]
        )
        timings["prompt_ms"] = (time.perf_counter() - turn_start) * 1000

        self.user_turns += 1
        nudge_block = self._nudge_block()

        max_rounds = self.config["agent"]["max_tool_rounds"]
        executed_calls: set[str] = set()
        # How many times each call has failed this turn, so a retry after a
        # fixed precondition is allowed but a persistently failing call is not.
        failed_calls: dict[str, int] = {}
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
        echo_retry_nudged = False
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
                    raw_reply, streamed_live = self._generate_reply(
                        messages, chunk_prefix=chunk_prefix, timings=timings)
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
                        self._prompt_cache = None
                        self._cached_prompt_ids = None
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
            if (not echo_retry_nudged
                    and (learn.looks_like_observation_echo(display)
                         or learn.looks_degenerate(display))):
                echo_retry_nudged = True
                self.output_fn(
                    "  [Echo] Reply impersonated a system observation; "
                    "regenerating...")
                self.history.append({"role": "user", "content": (
                    "[System observation: your last reply was discarded. "
                    "You wrote text in the '[System observation: ...]' "
                    "form, or repeated one line over and over. That form "
                    "is how the system speaks to you — it is never part "
                    "of your reply, and you must never invent one. "
                    "Answer the user directly now, in your own voice, "
                    "once.]"
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
                if (not display.strip() and not blank_retry_nudged
                        and (executed_calls or _is_greeting(user_input)
                             or action_req)):
                    blank_retry_nudged = True
                    self.output_fn(
                        "  [Blank] Reply came back empty; "
                        "prompting the model to respond...")
                    if action_req and not executed_calls:
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
                               "and the user's request is not yet answered. " if executed_calls else ""
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
                if action_req and not executed_calls and not blank_retry_nudged:
                    blank_retry_nudged = True
                    self.output_fn(
                        "  [Action] Model described the action but didn't "
                        "call a tool — prompting to retry...")
                    self.history.append({"role": "user", "content": (
                        "[System observation: you described what you would do "
                        "but did not actually call a tool. The user asked you "
                        "to perform an action. Emit one of these tags exactly, "
                        "on its own: <browse>https://...</browse> to open a "
                        "page, <click>visible text</click> to click, "
                        "<type>words</type> to type, <press>Enter</press> to "
                        "press a key, <read>https://...</read> to read a page. "
                        "Then answer in one short line. Do not just describe "
                        "what you will do — actually do it.]"
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
        # The security policy is not writable from inside the assistant, by any
        # route, before any other gate gets a say. A refusal, not a
        # confirmation prompt: every other high-risk call ends in "ask the
        # user", which is the right answer when the user is the one who wants
        # it and the wrong one here, because the attack this guards against is
        # precisely an instruction that arrived pretending to be them. A file
        # that can be unlocked by a convincing enough message is not locked.
        if security.blocks_tool_call(name, params):
            safety.log_security_event("policy_write_blocked", {
                "tool": name, "params": params,
            })
            return security.refusal_message(f"tool '{name}'")

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
        risk = safety.assess_tool_risk(name, params, self.config)
        allowed, reason = safety.maybe_confirm(name, params, risk, self.config, self.confirm_fn)
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

        log_score = self.config.get("safety", {}).get("log_score", 2)
        if risk["risk_score"] >= log_score:
            annotation = safety.risk_annotation(risk)
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
            return f"Reading {url} {'succeeded' if ok else 'failed'}.\nContent:\n{out}"

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
                out += _browser_peek(self.browser)
            return out

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
