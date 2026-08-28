"""Pure text heuristics for the chat loop: what a command was reaching for,
whether a message is a greeting or an action request, what mood it carries.

Every function here is a pure function of its arguments -- no model, no
session, no I/O -- which is why they can live outside ChatSession and be
tested directly.
"""

import re
import shlex
import sys
from pathlib import Path





# Bare words the model reaches for when it means "launch this desktop app".
# Only the ones actually seen failing, plus their obvious spellings — a wide
# net here would turn a genuine "command not found" into a surprise app launch.
_GUI_APP_ALIASES: dict[str, str] = {
    "chrome": "Google Chrome",
    "chromebrowser": "Google Chrome",
    "chrome-app": "Google Chrome",
    "chrome_app": "Google Chrome",
    "googlechrome": "Google Chrome",
    "google-chrome": "Google Chrome",
    "safari": "Safari",
    "spotify": "Spotify",
    "finder": "Finder",
    "terminal.app": "Terminal",
    "notes": "Notes",
    "calendar": "Calendar",
    "messages": "Messages",
    "mail": "Mail",
    "preview": "Preview",
    "vscode": "Visual Studio Code",
    "code-app": "Visual Studio Code",
}


# The app each invented name is reaching for, keyed by the word that survives
# whatever the model wrapped around it. _GUI_APP_ALIASES below is an exact-match
# table of misspellings, and an exact-match table of misspellings is a game you
# cannot win: it covered chrome/chromebrowser/chrome-app, so the model moved on
# to launch-chrome, chrome-x and start-chrome — 56 failures the table could not
# see, 15 of them on 2026-08-24 alone, in five identical three-step cycles.
#
# There is no need to guess the wrapper. By the time this is consulted the shell
# has already said the binary does not exist, so the only question left is which
# app the word was gesturing at, and a stem answers that for every spelling at
# once — including the ones nobody has invented yet.
_GUI_APP_STEMS: dict[str, str] = {
    "chrome": "Google Chrome",
    "safari": "Safari",
    "spotify": "Spotify",
    "finder": "Finder",
    "vscode": "Visual Studio Code",
    "calendar": "Calendar",
    "messages": "Messages",
    "preview": "Preview",
    "terminal": "Terminal",
}


def _gui_app_from_stem(word: str) -> str | None:
    """The app a non-existent single-word command was gesturing at, or None.

    Matches a stem as a whole part once the word is split on the separators
    models actually use (launch-chrome, chrome_app, start.chrome), and as a bare
    substring only for stems long enough that a coincidence is implausible
    (chromebrowser). Only ever reached for a word the shell has already failed
    to find, so a real binary can never be captured by it.
    """
    lowered = word.lower()
    parts = set(re.split(r"[-_.]+", lowered))
    for stem, app in _GUI_APP_STEMS.items():
        if stem in parts or (len(stem) >= 5 and stem in lowered):
            return app
    return None


def _gui_app_for(cmd: str, output: str) -> str | None:
    """The macOS app a failed bare command was probably trying to launch.

    Returns None unless the command is a single bare word (no arguments, no
    shell syntax) that names a known GUI app AND the failure was specifically
    "command not found". A command that failed for any other reason is a real
    failure and must be reported as one.
    """
    if sys.platform != "darwin":
        return None
    if "command not found" not in output.lower():
        return None
    word = cmd.strip().strip("'\"")
    if not word or len(word.split()) != 1:
        return None
    return _GUI_APP_ALIASES.get(word.lower()) or _gui_app_from_stem(word)


def _looks_like_shell_command(cmd: str) -> bool:
    """Returns true for those cmmds that uses shell syntax, saves ya time. it checks headers and common things ifykykyk."""

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


# just some detection words for a large langugage model, HOW FUNNY LMAO.
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

# Even more LMAOLMAO
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


# just more regex for search alike words
_EXPLICIT_SEARCH_RE = re.compile(
    r"\b(?:search\s+it|search\s+online|search\s+the\s+web|search\s+now|"
    r"google\s+it|look\s+it\s+up|just\s+search|check\s+online|check\s+the\s+web|"
    r"look\s+online|verify\s+online)\b",
    re.IGNORECASE,
)

# damn it, i just had to yk? my bad but these are filter stopwards
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
    """ Just does the search command where the user didnt even freaking bother to say so the ai figures it out. good on ya lmao."""

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

# greetings
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


# do keywords
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


# Verbs that make a turn a request to DO something. Deliberately broad, and
# biased toward answering "yes, they asked": this feeds a confirmation gate, and
# a gate that interrupts ordinary work is one the user turns off. The cost of a
# false "yes" is the status quo; the cost of a false "no" is a prompt on a turn
# that deserved none. So the gate fires only on turns carrying no action verb at
# all — a question asked purely for discussion.
_ACTION_VERBS = frozenset("""
run runs execute launch start stop restart kill open close
install uninstall delete remove create make build deploy write edit change
set update fix send download upload copy move rename clear reset apply
enable disable turn put add save train schedule click type press scroll
browse visit navigate go read check test try use call fetch scrape search
find show list tell give pull push commit sync backup restore print
mkdir touch rm ls cat grep chmod curl wget git npm pip brew
""".split())

_PATHY_RE = re.compile(
    "https?://"
    r"|(?:^|\s)[~.]?/\S"
    r"|\b\w+\.(?:py|js|ts|json|md|txt|sh|log|csv|ya?ml|toml|html|css)\b")


# "do" is an action verb in "do it" and an auxiliary in "what do you reckon".
# Matched separately so the second does not read as a request.
_IMPERATIVE_DO_RE = re.compile(
    r"^\s*(?:please\s+|just\s+|now\s+)*do\b|\bdo\s+(?:it|this|that|so)\b",
    re.IGNORECASE)


def _asks_for_action(text: str) -> bool:
    """Did the user's own turn request that something be done?

    Used only to decide whether a shell/filesystem call needs confirming, never
    to refuse one. A turn with no action verb, no path and no URL is a question
    asked for its answer — and a tool that reaches the machine on such a turn
    was the model's own idea, which is worth one prompt.
    """
    if not text or not text.strip():
        return False
    if text.strip().startswith("/"):
        return True  # a slash command is itself an instruction
    if _is_action_request(text) or _PATHY_RE.search(text):
        return True
    if _IMPERATIVE_DO_RE.search(text):
        return True
    words = set(re.findall(r"[a-z]+", text.lower()))
    return bool(words & _ACTION_VERBS)


def _repair_project_path_command(cmd: str) -> str | None:
    """Rewrite a command whose path argument names a real project file wrongly.

    Returns the corrected command, or None when there is nothing to fix.

    Telling the model was not enough. The observation already names the file,
    the directory the command ran in, and what to use instead — and live
    2026-08-25 the model answered "Let me re-try that using the correct path"
    and reissued the identical /Users/huygpt/agi/... path it had just been
    told was wrong. This is the same shape as the GUI-app launches: a repeated,
    unambiguous mistake that another sentence of prompt does not fix.

    Substitution can only ever point INTO the project — _project_paths_in
    verifies containment — so a repaired command cannot reach anything the
    original was not already allowed to.
    """
    from symbio import constants as _c
    try:
        args = shlex.split(cmd)
    except ValueError:
        return None
    if not args:
        return None
    changed = False
    for i, arg in enumerate(args[1:], start=1):
        if arg.startswith("-") or not arg.strip():
            continue
        # Judge the path the way the command will: relative paths resolve
        # against the sandbox directory it runs in, not against this process's
        # cwd — which is the project, and would make every project-relative
        # path look fine right up until the shell failed to find it.
        as_written = Path(arg) if Path(arg).is_absolute() else _c.SANDBOX_DIR / arg
        if as_written.exists():
            continue
        matches = _project_paths_in(f"x {arg}")
        if len(matches) != 1:
            continue
        args[i] = str(_c.PROJECT_DIR / matches[0])
        changed = True
    return " ".join(shlex.quote(a) for a in args) if changed else None


def _annotate_sandbox_cwd(cmd: str, out: str) -> str:
    """Explain a "no such file" that is really a working-directory mismatch.

    Keyed on the OUTPUT, not the exit status: a pipeline reports the exit code
    of its last stage, so `ls missing.txt | awk '{print $5}'` exits 0 and the
    harness announces "Shell command exited ok" over an ls that failed. Seen
    live 2026-08-24, twice, and both times the model concluded the user's file
    did not exist.
    """
    if "no such file" not in out.lower():
        return out
    from symbio import constants as _c
    missing = _project_paths_in(cmd)
    if not missing:
        return out
    return out + (
        f"\n[These exist in the project and are almost certainly what was "
        f"meant: {', '.join(missing)}. Commands run in {_c.SANDBOX_DIR}, not "
        f"the project root, so a relative path is resolved against that "
        f"directory — the file is there, this command looked in the wrong "
        f"place. Use read_file, which resolves against the project, or pass "
        f"the full path. Do NOT tell the user the file does not exist.]")


def _project_paths_in(cmd: str) -> list[str]:
    """Path-looking arguments of `cmd` that exist under the project root.

    Used only to explain a "no such file" failure: shell commands are run from
    the sandbox directory, so a project-relative path the model wrote is real
    but unreachable, and the shell's error cannot say so.
    """
    from symbio import constants as _c
    try:
        args = shlex.split(cmd)
    except ValueError:
        args = cmd.split()
    found: list[str] = []
    for arg in args[1:]:
        if arg.startswith("-") or not arg.strip():
            continue
        rel = arg.lstrip("./").lstrip("/")
        if not rel:
            continue
        # Try the path as written, then progressively drop leading components.
        # A model that invents an absolute path usually gets the tail right and
        # the prefix wrong: live 2026-08-25 it asked for
        # /Users/huygpt/agi/symbio/app/web.py — the real tree is under
        # .../Downloads/agi — and on being told "no such file" reported that the
        # user's file did not exist. The suffix symbio/app/web.py names it
        # exactly.
        #
        # At least two components must survive, so this never resolves a bare
        # filename that several directories could satisfy.
        parts = [q for q in rel.split("/") if q not in ("", ".")]
        for start in range(len(parts) - 1):
            suffix = "/".join(parts[start:])
            try:
                candidate = (_c.PROJECT_DIR / suffix).resolve()
                candidate.relative_to(_c.PROJECT_DIR.resolve())
            except (ValueError, OSError):
                break
            if candidate.exists():
                if suffix not in found:
                    found.append(suffix)
                break
    return found


def _is_substantive(text: str) -> bool:
    """Does this reply contain an answer, as opposed to merely characters?

    A turn that ends with "." or "+" has told the user nothing, but it passes
    every `display.strip()` check in the loop and so counts as a real reply:
    the blank-retry nudge does not fire and neither does the end-of-turn
    fallback. Seen live 2026-08-24 — a save_memory call succeeded and the
    entire visible answer was a single full stop.

    One alphanumeric character is enough. "42" and "ok" are real answers; a
    line of punctuation is not.
    """
    return any(ch.isalnum() for ch in text)


# navigation
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


# per turn user mood detection, used to nudge the model to adapt its tone. The model itself

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
# repeating
_AFFECT_EXASPERATION_NORM = tuple(
    re.sub(r"(.)\1+", r"\1", p) for p in _AFFECT_EXASPERATION
)

# even more terms
_CMD_START_RE = re.compile(
    r"^\s*(?:tell|search|find|get|show|give|make|run|open|fix|check|look|go|"
    r"stop|help|explain|list|try|call|fetch|write|read|delete|create|start|"
    r"shut|send|bring|take|install|build|deploy|test|restart|update|download|"
    r"copy|clear|reset|quit|exit|please|just|now)\b", re.IGNORECASE,
)




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


# Model-emitted mood tag 
# tag to emit for types of emotiuons. too basic but who tf cares
_MOOD_TAG_RE = re.compile(r"<mood>\s*([a-zA-Z]+)\s*</mood>", re.IGNORECASE)
_VALID_MOODS = {
    "angry", "frustrated", "impatient", "confused", "sad", "anxious",
    "grateful", "happy", "excited", "curious", "neutral",
}
