"""Values the chat modules share: tool-name sets, retry budgets, the thinking
dial, and the small pure predicates that read a reply's shape.

Split out of chat.py so the mixin modules can import them without importing
chat itself, which imports the mixins to build ChatSession. Nothing here may
import another chat module: this is the bottom of that graph.
"""

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
_BROWSER_ACTION_TOOLS = {
    "browser_click", "browser_type", "browser_scroll", "browser_press",
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


_COMPLETION_CLAIM = re.compile(
    r"\b(?:"
    r"I(?:'ve| have)\s+(?:just\s+|already\s+)?(?:run|ran|executed|fetched|scraped|"
    r"saved|written|wrote|created|made|moved|updated|bumped|downloaded|installed|"
    r"deleted|removed|sent|opened|read)"
    r"|I\s+(?:ran|executed|fetched|scraped|saved|wrote|created|moved|updated|bumped|"
    r"downloaded|installed|deleted|removed|sent)"
    r"|(?:has|have|had)\s+been\s+(?:run|executed|saved|written|created|moved|updated|"
    r"downloaded|installed|deleted|removed|sent|scraped)"
    r"|successfully\s+(?:ran|executed|fetched|scraped|saved|created|moved|updated|"
    r"downloaded|installed|deleted|sent)"
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
