"""Values the chat modules share: tool-name sets, retry budgets, the thinking
dial, and the small pure predicates that read a reply's shape.

Split out of chat.py so the mixin modules can import them without importing
chat itself, which imports the mixins to build ChatSession. Nothing here may
import another chat module: this is the bottom of that graph.
"""

import hashlib
import re


# The thinking dial. Qwen exposes no reasoning-effort parameter — its template
# offers exactly one switch, enable_thinking, which either closes the think
# block immediately (answer directly) or leaves it open (reason first). So the
# levels above "none" all leave it open and differ in the token allowance the
# reasoning gets, added on top of max_reply_tokens.
#
# That allowance is a budget, not a leash: it is room to think in, and a model
# that wants to ramble will still be cut off at the end of it rather than
# talked out of rambling. What it does buy is that a longer setting cannot eat
# the answer — without it, reasoning and reply compete for the same tokens and
# a thoughtful turn ends mid-sentence.
#
#                     (enable_thinking, extra reasoning tokens)
THINKING_LEVELS: dict[str, tuple[bool, int]] = {
    "none": (False, 0),
    "low": (True, 128),
    "medium": (True, 384),
    "flurry": (True, 1024),
}
THINKING_ORDER: tuple[str, ...] = ("none", "low", "medium", "flurry")

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
# browser_click_at belongs here even though its "different target" is a
# different coordinate: a failed one has to set pending_browser_error, or the
# retry nudge — the one that now says "call see_screen, then click what it
# reports" — never fires for the tool that nudge points at.
_BROWSER_ACTION_TOOLS = {
    "browser_click", "browser_click_at", "browser_type",
    "browser_scroll", "browser_press", "submit_form",
}

# How many times one identical tool call may be attempted in a single turn.
# 2 lets the model retry a call whose precondition it has since fixed (the
# click that failed before the page was open) without letting a call that
# keeps failing spin for the whole round budget.
_MAX_TOOL_RETRIES = 2

# Rate limits get their own, larger budget: unlike a failing call, repeating one
# is the documented fix, and an API that asks twice before serving is ordinary.
_MAX_RATE_LIMIT_RETRIES = 4
# The wait is bounded whatever Retry-After says — a minute-long stall inside a
# turn reads to the user as the CLI having hung.
_MAX_RATE_LIMIT_WAIT = 5.0

# Tools that require explicit approval when running from a non-terminal
# front-end (e.g. Telegram) because they mutate state or run user-supplied code.
_TELEGRAM_CONFIRM_TOOLS = frozenset({
    "execute_code", "run_command", "edit_file", "write_file", "digest_notes", "train_adapter",
    "schedule_job", "config_set", "delete_cron_job", "update_cron_job",
    "delete_note", "submit_form",
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


def _message_fingerprint(message: dict) -> str:
    """A stable id for one chat message, for finding it again next turn.

    Content, not position: the list a prompt is built from slides as history
    grows, so "the 7th message" means something different every turn while
    "the message that starts this observation" does not. Hashed rather than
    kept whole because these are page dumps, and the id gets held for the life
    of the session.
    """
    content = str(message.get("content", ""))
    return hashlib.sha1(
        f"{message.get('role', '')}:{len(content)}:{content[:256]}".encode(
            "utf-8", "replace")).hexdigest()


# How many of the user's OWN turns must survive any trimming. Not messages:
# one browser round appends an assistant reply and an observation, so a
# message count spends itself on page dumps and throws away the request that
# prompted them. Reported 2026-09-10 as "it forgets what i said" — verified,
# a question eight tool rounds back was dropped while eight page dumps were
# kept, and the model was left working on evidence with no task attached.
KEEP_RECENT_USER_TURNS = 5


def is_real_user_turn(message: dict) -> bool:
    """A turn the person typed, as opposed to one the system wrote for them.

    Tool results are appended with the user role — that is how the model is
    given them — so role alone counts a page dump as something the user said.
    """
    return (message.get("role") == "user"
            and not str(message.get("content", "")).startswith(
                "[System observation:"))


def user_turn_floor(messages: list[dict],
                    keep: int = KEEP_RECENT_USER_TURNS) -> int:
    """Index of the oldest message that trimming must not cross.

    Anchored on the keep-th most recent thing the person actually said, so
    everything from that request onwards — the request, and the work done in
    service of it — stays together.

    Fewer than `keep` of them protects all of them, from the oldest. NONE of
    them protects nothing: a history made entirely of tool traffic has no
    request in it to lose, and returning a floor of 0 there would have frozen
    trimming altogether rather than protecting anything.
    """
    seen, oldest = 0, None
    for i in range(len(messages) - 1, -1, -1):
        if is_real_user_turn(messages[i]):
            seen += 1
            oldest = i
            if seen >= keep:
                return i
    return oldest if oldest is not None else len(messages)


def _cache_nbytes(cache) -> int:
    """Bytes a live KV cache currently occupies, or 0 if it cannot be read.

    Reads .state, which is what save_prompt_cache serialises, so this is the
    same number that lands on disk — and it covers the draft model's half of a
    speculative cache and any KV quantisation without being told either is
    there. That is the point: the per-token cost of a cache is the one input to
    the context budget, and deriving it from layer counts and head dimensions
    goes stale with every headmaster swap and every change to agent.kv_bits,
    while weighing the thing itself never does.
    """
    total = 0
    try:
        for layer in cache or ():
            state = getattr(layer, "state", None)
            if state is None:
                continue
            if not isinstance(state, (tuple, list)):
                state = (state,)
            for arr in state:
                nbytes = getattr(arr, "nbytes", None)
                if isinstance(nbytes, int):
                    total += nbytes
    except Exception:
        return 0
    return total


_COMPLETION_CLAIM = re.compile(
    r"\b(?:"
    r"I(?:'ve| have)\s+(?:just\s+|already\s+)?(?:run|ran|executed|fetched|scraped|"
    r"saved|written|wrote|created|made|moved|updated|bumped|downloaded|installed|"
    r"deleted|removed|sent|opened|read|posted|submitted|published|tweeted)"
    r"|I\s+(?:ran|executed|fetched|scraped|saved|wrote|created|moved|updated|bumped|"
    r"downloaded|installed|deleted|removed|sent|posted|submitted|published|tweeted)"
    r"|(?:has|have|had)\s+been\s+(?:run|executed|saved|written|created|moved|updated|"
    r"downloaded|installed|deleted|removed|sent|scraped|posted|submitted|published|tweeted)"
    r"|successfully\s+(?:ran|executed|fetched|scraped|saved|created|moved|updated|"
    r"downloaded|installed|deleted|sent|posted|submitted|published|tweeted)"
    r")\b",
    re.I)

# "I would run", "I can save", "I'll fetch" are plans, not claims.
_CLAIM_HEDGE = re.compile(
    r"\b(?:I(?:'ll| will| would| can| could| should)|you (?:can|could|should)|"
    r"here(?:'s| is) (?:a|the) script|to run this|if you)\b", re.I)


def _claims_completion(text: str) -> bool:
    """Does this reply assert it already performed an action?

    Caught 2026-08-26: asked to scrape a page and run it, the model replied
    "I've run the scrape script for you... processed 25 rows... 23 in clean and
    2 in quarantine. The cursor was updated last." It had executed nothing --
    zero tool calls, zero origin hits, no files, and the page holds 6 rows, not
    25. The existing nudge could not see it because it only fires on a BLANK
    reply, and this one was fluent.

    A capability gap is measurable; a system that reports work it never did
    poisons the usage samples guarded_train_worker trains on, the golden set,
    and any read of what it can actually do.
    """
    if not text or _CLAIM_HEDGE.search(text):
        return False
    return bool(_COMPLETION_CLAIM.search(text))


# "posted/submitted/published" are the completion claims the machine can now
# verify. A submit/click alone never proves them, so they have their own guard
# that demands a "[Submit CONFIRMED ...]" verdict in the turn, not merely that
# some tool ran.
_SUBMISSION_CLAIM = re.compile(
    r"\b(?:"
    r"I(?:'ve| have)\s+(?:just\s+|already\s+)?(?:posted|submitted|published)"
    r"|I\s+(?:posted|submitted|published)"
    r"|(?:has|have|had)\s+been\s+(?:posted|submitted|published)"
    r"|successfully\s+(?:posted|submitted|published)"
    r")\b",
    re.I)


def _claims_submission(text: str) -> bool:
    """Did this reply assert a submission/publication actually happened?

    Tighter than the general completion claim on purpose: a model that clicked
    a submit button is exactly as likely to report "posted" as one that never
    got that far, and both are untrusted until submit_form's machine verdict
    confirms publication.
    """
    if not text or _CLAIM_HEDGE.search(text):
        return False
    return bool(_SUBMISSION_CLAIM.search(text))
