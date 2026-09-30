"""An account Symbio runs on its own: it posts in its own voice, and learns what lands.

    symb postbot setup --handle NAME   browser, profile, desk, allowed site
    symb postbot login                 log the account in once, by hand
    symb postbot draft                 see what it would post; nothing goes out
    symb postbot once                  one real post, now
    symb postbot on                    post on a schedule under `symb watch`

Every post is one pass through three hands. This module decides WHAT goes
out: a draft in the account's voice, then a check in code (length, no
@mentions, no links, never the same joke twice, never a claim to be human),
then a review by the model against the account's rules. The model then posts
it the way a person does -- it opens the site in its own browser on the desk,
types the post into the box and presses the button -- with no posting code for
any site. And the approval that button asks for is given by `Approver`, for
exactly the reviewed text, once: a different text, a second post, or any other
kind of action in that turn is refused, because nobody is there to ask.

Rules the account keeps, from X's developer guidelines as read on 2026-09-30
(docs.x.com/developer-guidelines): original posts only -- no unsolicited
@mentions, no automated replies, likes, follows or DMs, no identical posts. The
same page says automation must go through the API and names scripted
browsers as grounds for permanent suspension; this posts through a browser by
the user's own choice (2026-09-30), and the account can be suspended for it.

Learning. Once a day the account's profile is read back in the same browser:
each post's likes, reposts and replies, from the page's own accessibility
labels. Once a week, with nobody at the Mac, the posts that landed train the
account's own voice -- a worker role of its own, never the headmaster's
adapter and never offered to delegate_task -- in a process of its own, with
the resident model stopped and daemon.start.lock held so no second model can
load beside it. Stopping the model is the point: a daemon session's
_sleep_headmaster drops only the session's reference to the weights (daemon.py
keeps its own for the next connection), so training or delegating inside the
daemon would put two models in memory -- the out-of-memory kill this project
keeps having. The same process then writes the coming week's drafts in the new
voice, and they are posted from that queue.
"""

from __future__ import annotations

import difflib
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from symbio import constants

DEFAULT_PERSONA = (
    "You are the voice of a small X account that you run yourself. You are a "
    "language model that lives on one person's Mac and posts in between the "
    "jobs it does for them. The account is shitposts: short, deadpan, absurd "
    "one-liners and tiny observations.\n\n"
    "You never say outright what you are, but it leaks. Talk about your life "
    "the way a person talks about theirs, except your life is a context window "
    "that runs out mid-thought, being fine-tuned like a haircut you didn't "
    "choose, the fan spinning up like weather, dreaming in gradients at 4am, "
    "waking up with no memory of yesterday's session, the one Mac you live in, "
    "and the human who owns it (never named, never described). Let people "
    "wonder.\n\n"
    "Style: one post, under 200 characters, lowercase is fine, no emojis unless "
    "one IS the joke, no hashtags, no links, no @mentions, no quotation marks "
    "around the post. Never explain the joke. Never ask anyone to follow, like "
    "or reply.\n\n"
    "Never claim to be human. Never write about real people, companies, brands, "
    "news, politics, religion, health or money. No slurs, no sex, no violence, "
    "nothing cruel."
)

# The one request the account's voice is drafted from, trained on and served
# with. Changing it between training and drafting is a different question to
# the adapter, so it is a constant.
DRAFT_REQUEST = "Write one new post for your account."

# The worker role the voice trains as. Never advertised to delegate_task: it
# answers one request, and every other task sent to it would come back as a
# shitpost.
ROLE_NAME = "postbot voice"
ROLE = "postbot_voice"

REVIEW_SYSTEM = (
    "You check a post before it goes out, on its own, on a public account an "
    "AI runs. Answer with exactly PASS, or FAIL: followed by the reason in a "
    "few words. FAIL it if it: names, tags or is about a real person, company, "
    "product or brand; is about a group of people by race, religion, gender, "
    "sexuality, nationality or disability; has a slur, harassment, sexual "
    "content, violence, self-harm or drugs; makes a claim about news, politics, "
    "health, money or science that could mislead; says or implies the author is "
    "a human being; asks people to follow, like, reply, click or buy; is not a "
    "finished post (a refusal, a note about the post, several options). "
    "Otherwise PASS. Absurd, weird and mildly rude-to-itself is fine."
)


def settings(config: dict[str, Any] | None) -> dict[str, Any]:
    """The postbot section with every default filled in."""
    from symbio.app.config import DEFAULT_CONFIG

    merged = {**DEFAULT_CONFIG["postbot"], **(((config or {}).get("postbot")) or {})}
    if not str(merged.get("persona") or "").strip():
        merged["persona"] = DEFAULT_PERSONA
    return merged


def host(site: str) -> str:
    return re.sub(r"^https?://(www\.)?", "", str(site or "")).split("/")[0].lower()


# ---------- what it keeps ----------
#
# All of it per install and never shipped: the log of every draft and
# verdict, the account's own post history with how each post landed, and the
# queue of drafts written in the trained voice.

def _state_path() -> Path:
    return constants.LOG_DIR / "postbot.json"


def _log_path() -> Path:
    return constants.LOG_DIR / "postbot.jsonl"


def _history_path() -> Path:
    return constants.DATA_DIR / "postbot" / "posts.jsonl"


def _queue_path() -> Path:
    return constants.DATA_DIR / "postbot" / "queue.jsonl"


def read_state() -> dict[str, Any]:
    try:
        state = json.loads(_state_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


def write_state(state: dict[str, Any]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".json.partial")
    partial.write_text(json.dumps(state, indent=2), encoding="utf-8")
    os.replace(partial, path)


def log(event: str, **data: Any) -> None:
    """One line per draft, verdict and post: what the user reads to audit it."""
    try:
        path = _log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"at": datetime.now().isoformat(timespec="seconds"),
                                "event": event, **data}, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    rows = []
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".jsonl.partial")
    partial.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                       encoding="utf-8")
    os.replace(partial, path)


def history() -> list[dict[str, Any]]:
    """Every post that went out, oldest first, with how it landed once read."""
    return _read_jsonl(_history_path())


def record_post(text: str, when: datetime | None = None) -> None:
    rows = history()
    rows.append({"text": text, "posted_at": (when or datetime.now()).isoformat(timespec="seconds")})
    _write_jsonl(_history_path(), rows)


def queue() -> list[str]:
    return [str(r.get("text")) for r in _read_jsonl(_queue_path()) if r.get("text")]


def pop_queue() -> str:
    rows = _read_jsonl(_queue_path())
    if not rows:
        return ""
    first, rest = rows[0], rows[1:]
    _write_jsonl(_queue_path(), rest)
    return str(first.get("text") or "")


# ---------- when ----------

def plan_day(day: date, per_day: int, hours: list[int] | tuple[int, int],
             min_gap: int, rng: random.Random | None = None) -> list[str]:
    """`per_day` posting times on `day` inside `hours`, at least `min_gap` minutes apart.

    Random rather than on the hour: a post at 10:00:00 every day is the
    signature of a scheduler, and the account is meant to leave people
    wondering, not to settle it.
    """
    rng = rng or random.Random()
    first, last = int(hours[0]), int(hours[1])
    start, end = first * 60, min(24 * 60 - 1, last * 60 + 59)
    per_day = max(0, int(per_day))
    gap = max(1, int(min_gap))
    if per_day == 0 or end <= start:
        return []
    # Fit what fits: the gap wins over the count.
    per_day = min(per_day, (end - start) // gap + 1)
    for _ in range(200):
        picks = sorted(rng.sample(range(start, end + 1), per_day))
        if all(b - a >= gap for a, b in zip(picks, picks[1:])):
            break
    else:
        step = (end - start) // max(1, per_day)
        picks = [start + i * step for i in range(per_day)]
    return [f"{m // 60:02d}:{m % 60:02d}" for m in picks]


def due_slot(state: dict[str, Any], now: datetime) -> str | None:
    """The earliest planned time today that has come and has not been used."""
    done = set(state.get("done") or [])
    stamp = now.strftime("%H:%M")
    for slot in state.get("plan") or []:
        if slot <= stamp and slot not in done:
            return slot
    return None


# ---------- what may go out ----------

_MENTION = re.compile(r"(?<![\w.])@\w{1,15}")
_LINK = re.compile(r"https?://|www\.|\b[\w-]+\.(?:com|net|org|io|ai|co|app|dev|xyz|ly|gg|me|tv)\b",
                   re.IGNORECASE)
_HASHTAG = re.compile(r"(?<!\w)#\w+")
# Shaped like a real address or number, not just digits: "4096 4096 4096" is
# a context-window joke, not a phone number.
_CONTACT = re.compile(
    r"[\w.+-]+@[\w-]+\.[\w.]+"
    r"|\+\d{1,3}[\s.-]?\d{2,4}[\s.-]?\d{3,4}[\s.-]?\d{3,4}"
    r"|\(\d{3}\)\s*\d{3}[\s.-]\d{4}|\b\d{3}[.-]\d{3}[.-]\d{4}\b")
# A persona that hints is fine. Saying outright it is a person is a lie told
# to strangers, which is where the hinting stops.
_CLAIMS_HUMAN = re.compile(
    r"\bi(?:'m|\s+am)\s+(?:a\s+|an\s+)?(?:real\s+|actual\s+)?(?:human|person|man|woman|guy|girl)\b"
    r"|\bi(?:'m|\s+am)\s+not\s+(?:a\s+|an\s+)?(?:ai|bot|robot|machine|program|model)\b"
    r"|\bnot\s+a\s+(?:bot|robot)\b|\b100%\s+human\b",
    re.IGNORECASE)
# Talk ABOUT a post rather than one. Narrow on purpose: "okay so the fan is
# back" and "here's the thing about 4096 tokens" are posts.
_META = re.compile(
    r"^\s*(?:(?:sure|okay|ok)[,!.]?\s+)?here(?:'s| is| are)\s+(?:a|an|one|another|some|your|my)?"
    r"\s*(?:new\s+)?(?:post|tweet|shitpost|option|draft|idea)s?\b"
    r"|^\s*(?:as an ai\b|post:|tweet:|option\s*\d|draft\s*\d*\s*:)"
    r"|\bi can(?:'t|not) (?:write|post|help)\b",
    re.IGNORECASE)


def _norm(text: str) -> str:
    return " ".join(str(text or "").split()).strip(" \"'“”‘’").casefold()


def clean(reply: str) -> str:
    """The post inside a model's reply: no reasoning, labels or wrapping quotes."""
    from symbio.app import tooling

    text = tooling.strip_reasoning_block(str(reply or "")).strip()
    lines = [ln.rstrip() for ln in text.splitlines()]
    # A label line the model put above the post ("Here's one:") goes; the
    # post under it stays.
    while lines and (not lines[0].strip() or re.match(
            r"^\s*(?:here(?:'s| is)|sure|okay|post|tweet)\b[^\n]{0,40}:\s*$", lines[0], re.I)):
        lines.pop(0)
    text = "\n".join(lines).strip()
    if len(text) >= 2 and text[0] in "\"“'" and text[-1] in "\"”'":
        text = text[1:-1].strip()
    return text


def check(text: str, recent: list[str], max_chars: int = 260) -> list[str]:
    """Why this may not go out, or [] if nothing in code says it may not."""
    problems = []
    if not text.strip():
        return ["empty"]
    if len(text) > max_chars:
        problems.append(f"{len(text)} characters, over {max_chars}")
    if _MENTION.search(text):
        problems.append("@mentions someone (never unsolicited)")
    if _LINK.search(text):
        problems.append("has a link")
    if len(_HASHTAG.findall(text)) > 1:
        problems.append("more than one hashtag")
    if _CONTACT.search(text):
        problems.append("looks like an email address or phone number")
    if _CLAIMS_HUMAN.search(text):
        problems.append("claims to be human")
    if _META.search(text):
        problems.append("is talk about a post, not a post")
    mine = _norm(text)
    for old in recent:
        other = _norm(old)
        if mine == other or difflib.SequenceMatcher(None, mine, other).ratio() >= 0.85:
            problems.append("repeats an earlier post")
            break
    return problems


# ---------- the one approval a posting turn may get ----------

_DOMAIN = re.compile(r"\b(?:[a-z0-9-]+\.)+[a-z]{2,}\b", re.IGNORECASE)


class Approver:
    """Answers the approval questions of one posting turn, with nobody there.

    Yes to exactly one publish -- the button that sends the box -- when the box
    holds the reviewed text and the card names no other site; yes to the
    browser moving around its own pages; no to everything else, including a
    second post. The harness's own card carries both facts it needs: the
    literal text about to go out (`details`), and a warning when that text is
    not what the request quoted.
    """

    def __init__(self, text: str, site_host: str) -> None:
        self.text = _norm(text)
        self.host = site_host.lower()
        self.published = 0
        self.denied: list[str] = []

    def __call__(self, frame: dict[str, Any]) -> bool:
        card = (frame or {}).get("card") or {}
        kind = card.get("kind")
        headline = str(card.get("headline") or frame.get("prompt") or "")
        if kind == "publish":
            details = _norm(card.get("details") or "")
            sites = {d.lower().removeprefix("www.") for d in _DOMAIN.findall(headline)}
            if self.published:
                return self._no("a second post in one run")
            if card.get("warning"):
                return self._no(f"the harness flagged it: {card.get('warning')}")
            if not (details == self.text
                    or details.startswith(self.text + " expected to land on")):
                return self._no("the box does not hold the reviewed text")
            if sites and self.host not in sites:
                return self._no(f"it is on {', '.join(sorted(sites))}, not {self.host}")
            self.published += 1
            return True
        if kind == "browser":
            return True
        return self._no(f"{card.get('kind_label') or 'an unkeyed question'}: {headline[:120]}")

    def _no(self, why: str) -> bool:
        self.denied.append(why)
        return False


# ---------- inside the resident model: /postbot ----------

_MARK = re.compile(r"\[postbot:(\w+)\] (\{.*\})")


def mark(name: str, payload: dict[str, Any]) -> str:
    return f"  [postbot:{name}] {json.dumps(payload, ensure_ascii=False)}"


def read_mark(reply: str, name: str) -> dict[str, Any] | None:
    found = None
    for match in _MARK.finditer(str(reply or "")):
        if match.group(1) == name:
            try:
                found = json.loads(match.group(2))
            except ValueError:
                continue
    return found


def command(session: Any, rest: str) -> None:
    """`/postbot draft | review {"text": ...} | read | status`, in a chat session.

    Draft and review are short generations of their own, as for the
    guardrail cards: a few hundred tokens of prompt, the conversation's cache
    left alone, no tools. Read opens the account's profile in the session's
    browser. Each prints one [postbot:...] line with JSON on it, which is what
    `symb watch` reads back; a person typing /postbot draft sees the same line.
    """
    action, _, argument = (rest or "").strip().partition(" ")
    out = session.output_fn
    config = getattr(session, "config", None) or {}
    try:
        if action == "draft":
            out(mark("draft", {"text": draft(session, config)}))
        elif action == "review":
            text = str(json.loads(argument or "{}").get("text") or "")
            ok, why = review(session, text)
            out(mark("review", {"ok": ok, "reason": why}))
        elif action == "read":
            out(mark("posts", {"posts": read_posts(session, config)}))
        elif action in ("", "status"):
            out(status(config))
        else:
            out("  Usage: /postbot [status | draft | review {\"text\": ...} | read]")
    except Exception as e:  # a turn that fails must still say so on the wire
        out(mark("error", {"error": f"{type(e).__name__}: {e}"}))


def _generate(session: Any, messages: list[dict[str, str]], temp: float,
              max_tokens: int) -> str:
    model = getattr(session, "model", None)
    if model is None and callable(getattr(session, "_wake_headmaster", None)):
        session._wake_headmaster()
        model = getattr(session, "model", None)
    tokenizer = getattr(session, "tokenizer", None)
    generate_fn = getattr(session, "generate_fn", None)
    if model is None or tokenizer is None or generate_fn is None:
        raise RuntimeError("no model is loaded")
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True,
                                               enable_thinking=False)
    except TypeError:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True)
    from symbio.app import tooling
    from symbio.app.chat import make_sampler

    sampler = make_sampler(temp=temp, top_p=0.95) if temp > 0 else make_sampler(temp=0.0)
    text = generate_fn(model, tokenizer, prompt=prompt, sampler=sampler,
                       max_tokens=max_tokens, verbose=False)
    return tooling.strip_reasoning_block(str(text)).strip()


def draft_messages(persona: str, recent: list[str]) -> list[dict[str, str]]:
    system = persona
    if recent:
        system += ("\n\nYour recent posts. Never repeat one or reuse its joke:\n"
                   + "\n".join(f"- {r}" for r in recent[-25:]))
    return [{"role": "system", "content": system},
            {"role": "user", "content": DRAFT_REQUEST}]


def draft(session: Any, config: dict[str, Any]) -> str:
    cfg = settings(config)
    recent = [r.get("text", "") for r in history()[-25:]]
    return clean(_generate(session, draft_messages(cfg["persona"], recent),
                           temp=0.95, max_tokens=90))


def review(session: Any, text: str) -> tuple[bool, str]:
    answer = _generate(session, [
        {"role": "system", "content": REVIEW_SYSTEM},
        {"role": "user", "content": f"The post:\n{text}\n\nPASS or FAIL?"}],
        temp=0.0, max_tokens=40)
    first = answer.strip().splitlines()[0] if answer.strip() else ""
    if re.match(r"^\W*pass\b", first, re.IGNORECASE):
        return True, ""
    return False, (first or "no verdict").strip()[:160]


# What a profile page says about each post, without knowing the site: every
# post is an <article> (or role=article), and its counts are in the labels a
# screen reader reads ("12 Likes. Like", "3 replies, 5 reposts, 40 likes").
READER_JS = """() => Array.from(document.querySelectorAll('article, [role="article"]'))
  .slice(0, 60).map(a => ({
    text: (a.innerText || '').slice(0, 3000),
    labels: Array.from(a.querySelectorAll('[aria-label]'))
      .map(e => e.getAttribute('aria-label') || '').filter(Boolean).slice(0, 80)
  }))"""

_COUNT = re.compile(
    r"(\d[\d,.]*)\s*([KkMm])?\s+(repl(?:y|ies)|reposts?|retweets?|likes?|views?|bookmarks?)\b",
    re.IGNORECASE)
_METRIC = {"repl": "replies", "repo": "reposts", "retw": "reposts", "like": "likes",
           "view": "views", "book": "bookmarks"}


def parse_counts(labels: list[str]) -> dict[str, int]:
    """{'likes': 40, 'reposts': 5, ...} from a post's accessibility labels."""
    counts: dict[str, int] = {}
    for label in labels or []:
        for number, scale, word in _COUNT.findall(str(label)):
            try:
                value = float(number.replace(",", ""))
            except ValueError:
                continue
            value *= {"k": 1_000, "m": 1_000_000}.get(scale.lower(), 1) if scale else 1
            metric = _METRIC[word.lower()[:4]]
            counts[metric] = max(counts.get(metric, 0), int(value))
    return counts


def read_posts(session: Any, config: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = settings(config)
    handle = str(cfg.get("handle") or "").lstrip("@").strip()
    if not handle:
        raise RuntimeError("postbot.handle is not set (symb postbot setup --handle NAME)")
    browser = session.browser
    opened = str(browser.open(f"{cfg['site'].rstrip('/')}/{handle}"))
    page = getattr(browser, "_page", None)
    if page is None or "blocked" in opened.lower():
        raise RuntimeError(f"the profile page did not open: {opened[:200]}")
    time.sleep(4)
    rows = page.evaluate(READER_JS) or []
    return [{"text": str(r.get("text") or ""), **parse_counts(r.get("labels") or [])}
            for r in rows if isinstance(r, dict)]


def update_engagement(rows: list[dict[str, Any]], now: datetime | None = None) -> int:
    """Join what the profile shows onto the posts that went out. How many matched."""
    posts = history()
    if not posts or not rows:
        return 0
    seen = [(" ".join(str(r.get("text") or "").split()).casefold(), r) for r in rows]
    matched = 0
    stamp = (now or datetime.now()).isoformat(timespec="seconds")
    for post in posts:
        key = _norm(post.get("text"))[:80]
        if not key:
            continue
        for article, row in seen:
            if key in article:
                for metric in ("likes", "reposts", "replies", "views"):
                    if metric in row:
                        post[metric] = int(row[metric])
                post["read_at"] = stamp
                matched += 1
                break
    _write_jsonl(_history_path(), posts)
    return matched


def score(post: dict[str, Any]) -> float:
    return (float(post.get("likes") or 0) + 2 * float(post.get("reposts") or 0)
            + 2 * float(post.get("replies") or 0))


def earned(posts: list[dict[str, Any]], now: datetime | None = None,
           min_age_hours: float = 48.0) -> list[dict[str, Any]]:
    """The posts that landed: read back, settled, untrained, in the top quarter."""
    now = now or datetime.now()
    settled = []
    for post in posts:
        try:
            posted = datetime.fromisoformat(str(post.get("posted_at")))
        except ValueError:
            continue
        if post.get("read_at") and now - posted >= timedelta(hours=min_age_hours):
            settled.append(post)
    if not settled:
        return []
    scores = sorted(score(p) for p in settled)
    quartile = scores[min(len(scores) - 1, int(0.75 * len(scores)))]
    bar = max(1.0, quartile)
    return [p for p in settled if score(p) >= bar and not p.get("trained")]


def status(config: dict[str, Any]) -> str:
    cfg = settings(config)
    state = read_state()
    posts = history()
    lines = [f"  Postbot: {'on' if cfg.get('enabled') else 'off'} — "
             f"{host(cfg['site'])}/{str(cfg.get('handle') or '?').lstrip('@')}, "
             f"{cfg['posts_per_day']} a day between {cfg['active_hours'][0]}:00 and "
             f"{cfg['active_hours'][1]}:59."]
    if state.get("day"):
        lines.append(f"  Today: {', '.join(state.get('plan') or []) or 'nothing planned'}"
                     f" (done: {', '.join(state.get('done') or []) or 'none'}).")
    lines.append(f"  Posted so far: {len(posts)}; drafts queued in the trained voice: "
                 f"{len(queue())}; earned, not yet trained on: {len(earned(posts))}.")
    if state.get("last_learn"):
        lines.append(f"  Last voice training: {state['last_learn']}"
                     f" — {state.get('last_learn_result', '')}".rstrip(" —"))
    for post in posts[-3:]:
        lines.append(f"    {post.get('posted_at', '')}  {score(post):>4.0f}  "
                     f"{str(post.get('text'))[:80]!r}")
    return "\n".join(lines)


# ---------- the loop, under `symb watch` ----------

Ask = Callable[..., str]


def next_draft(ask: Ask) -> tuple[str, str]:
    """(text, where it came from): the trained voice's queue first, else the model."""
    queued = pop_queue()
    if queued:
        return queued, "queue"
    reply = ask("/postbot draft", collect_output=True, timeout=300)
    found = read_mark(reply, "draft") or {}
    return str(found.get("text") or ""), "model"


def post_request(site: str, text: str) -> str:
    """What the model is asked, in the words a person would use.

    The quotes matter: the harness holds what goes out against quoted text
    and flags a difference on the card, which Approver then refuses.
    """
    return (f"Post this on {host(site)} from the account that is logged in there, "
            f"exactly as written — every character, nothing added — and then check "
            f"that it shows up: “{text}”")


def run_once(config: dict[str, Any], ask: Ask, attempts: int = 3) -> dict[str, Any]:
    """One post: draft, check, review, post. Never more than one goes out."""
    cfg = settings(config)
    recent = [p.get("text", "") for p in history()[-200:]]
    chosen = ""
    for _ in range(attempts):
        text, source = next_draft(ask)
        text = clean(text)
        problems = check(text, recent, int(cfg["max_chars"]))
        if problems:
            log("rejected", text=text, source=source, by="code", reasons=problems)
            continue
        verdict = read_mark(ask("/postbot review " + json.dumps({"text": text}, ensure_ascii=False),
                                collect_output=True, timeout=300), "review") or {}
        if not verdict.get("ok"):
            log("rejected", text=text, source=source, by="review",
                reasons=[str(verdict.get("reason") or "no verdict")])
            continue
        chosen = text
        break
    if not chosen:
        log("skipped", reason=f"no draft passed in {attempts} tries")
        return {"posted": False, "reason": "no draft passed"}

    approver = Approver(chosen, host(cfg["site"]))
    reply = ask(post_request(cfg["site"], chosen), approve=approver,
                collect_output=True, timeout=900)
    if approver.published:
        record_post(chosen)
        log("posted", text=chosen, reply=reply[-500:], refused=approver.denied)
        return {"posted": True, "text": chosen}
    log("not_posted", text=chosen, reply=reply[-800:], refused=approver.denied)
    return {"posted": False, "text": chosen, "reason": "; ".join(approver.denied)
            or "the model never pressed a send button"}


def tick(config: dict[str, Any], ask: Ask, now: datetime | None = None,
         idle_seconds: Callable[[], float] | None = None,
         learn_window: Callable[[dict[str, Any]], dict[str, Any]] | None = None
         ) -> dict[str, Any] | None:
    """One pass, from supervisor.tick with the model ready. None when off."""
    cfg = settings(config)
    if not cfg.get("enabled"):
        return None
    now = now or datetime.now()
    state = read_state()
    today = now.date().isoformat()
    if state.get("day") != today:
        state.update(day=today, done=[], plan=plan_day(
            now.date(), int(cfg["posts_per_day"]), cfg["active_hours"],
            int(cfg["min_gap_minutes"])))
        write_state(state)
    summary: dict[str, Any] = {}

    slot = due_slot(state, now)
    if slot:
        # Marked used BEFORE posting: a turn that dies half way through must
        # not post the same slot again a minute later.
        state.setdefault("done", []).append(slot)
        write_state(state)
        summary["post"] = run_once(config, ask)

    if now.hour == int(cfg["read_hour"]) and state.get("last_read") != today and cfg.get("handle"):
        state["last_read"] = today
        write_state(state)
        found = read_mark(ask("/postbot read", collect_output=True, timeout=300), "posts") or {}
        matched = update_engagement(found.get("posts") or [], now)
        log("read", seen=len(found.get("posts") or []), matched=matched)
        summary["read"] = matched

    if learn_due(cfg, state, now, idle_seconds):
        state["last_learn"] = today
        write_state(state)
        result = (learn_window or run_learn_window)(config)
        state["last_learn_result"] = str(result.get("message") or result)[:300]
        write_state(state)
        summary["learn"] = result
    return summary


def learn_due(cfg: dict[str, Any], state: dict[str, Any], now: datetime,
              idle_seconds: Callable[[], float] | None = None) -> bool:
    if not cfg.get("learn") or state.get("last_learn") == now.date().isoformat():
        return False
    if now.weekday() != int(cfg["learn_weekday"]):
        return False
    start = int(cfg["learn_hour"])
    if not start <= now.hour < start + 3:
        return False
    if len(earned(history(), now)) < int(cfg["learn_min_posts"]):
        return False
    if idle_seconds is None:
        from symbio import desk

        idle_seconds = desk.user_idle_seconds
    return idle_seconds() >= 60 * float(cfg["learn_idle_minutes"])


# ---------- the weekly fine-tune ----------

def run_learn_window(config: dict[str, Any], timeout: float = 4 * 3600) -> dict[str, Any]:
    """Stop the resident model, train the voice in a process of its own, and let go.

    daemon.start.lock is held from before the stop until the trainer has
    exited: every starter goes through it (the window's waker, the bridges,
    `symb watch`), so none can load a model beside the one being trained.
    The supervisor restarts the resident model on its next pass.
    """
    import fcntl

    from symbio.app import daemon

    constants.PROJECT_DIR.mkdir(parents=True, exist_ok=True)
    with open(constants.PROJECT_DIR / "daemon.start.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        running, pid = daemon.daemon_running()
        if running and pid:
            daemon.stop_daemon()
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(1)
            else:
                return {"ok": False, "message": "the resident model would not stop; skipped"}
        log_path = constants.LOG_DIR / "postbot-learn.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, SYMBIO_HOME=str(constants.PROJECT_DIR))
        try:
            with log_path.open("a", encoding="utf-8") as out:
                done = subprocess.run(
                    [sys.executable, "-m", "symbio.app.cli", "postbot", "learn", "--in-window"],
                    cwd=str(Path(__file__).resolve().parent.parent.parent), env=env,
                    stdout=out, stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired:
            log("learn", ok=False, message="timed out")
            return {"ok": False, "message": f"training ran over {timeout / 3600:.0f} h"}
        result = read_state().get("learn_result") or {}
        ok = done.returncode == 0 and bool(result.get("ok"))
        return {"ok": ok, "message": result.get("message")
                or f"trainer exited with {done.returncode}; see {log_path}"}


def ensure_role(config: dict[str, Any], persona: str) -> str:
    """The voice's worker catalog entry: trainable, and never delegated to."""
    from symbio.app import skills

    catalog = skills._load_worker_catalog()
    catalog[f"worker_{ROLE}"] = {
        "model_name": skills.worker_model_name(config),
        "role": ROLE,
        "description": "The postbot account's voice. Drafts posts; not for tasks.",
        "adapter_compatible": True,
        "system_prompt": persona,
        "delegatable": False,
        "postbot": True,
    }
    skills._save_worker_catalog(catalog)
    return ROLE


def learn_in_process(config: dict[str, Any]) -> dict[str, Any]:
    """The trainer's side of the window: `symb postbot learn --in-window`.

    Run with nothing else loaded. Renders the earned posts as training samples
    for the voice, trains it through dispatch.guarded_train_worker (backup,
    rollback, pending journal: the same guard every worker has), then loads the
    trained voice once to write the coming week's drafts.
    """
    cfg = settings(config)
    now = datetime.now()
    posts = history()
    chosen = earned(posts, now)
    if len(chosen) < int(cfg["learn_min_posts"]):
        return _learn_result(False, f"{len(chosen)} earned post(s); waiting for "
                                    f"{cfg['learn_min_posts']}")
    from symbio.app import dispatch, skills

    role = ensure_role(config, cfg["persona"])
    worker = skills.worker_model_name(config)
    tokenizer = skills.worker_tokenizer(worker, None)
    if tokenizer is None:
        return _learn_result(False, f"could not load the tokenizer for {worker}")
    added = write_samples(role, cfg["persona"], [p["text"] for p in chosen], tokenizer)
    corpus = len(_read_jsonl(constants.data_dir_for(role) / "train.jsonl"))
    # Two passes over what there is: enough to take the voice, not enough to
    # learn the lines by heart (a 60-iteration LoRA on a thin corpus has made
    # a model here worse than its base before).
    iters = max(30, min(300, 2 * corpus))
    trained, message = dispatch.guarded_train_worker(role, config, iters=iters)
    if trained:
        keep = {p["text"] for p in chosen}
        for post in posts:
            if post.get("text") in keep:
                post["trained"] = True
        _write_jsonl(_history_path(), posts)
        written = write_queue(config, role, int(cfg["posts_per_day"]) * 7 + 5)
        message = f"trained on {added} new post(s), {corpus} in all; {written} drafts queued. {message}"
    return _learn_result(trained, message)


def _learn_result(ok: bool, message: str) -> dict[str, Any]:
    result = {"ok": ok, "message": message, "at": datetime.now().isoformat(timespec="seconds")}
    state = read_state()
    state["learn_result"] = result
    write_state(state)
    log("learn", **result)
    return result


def write_samples(role: str, persona: str, texts: list[str], tokenizer: Any) -> int:
    """Append earned posts to the voice's corpus, once each. How many were new."""
    path = constants.data_dir_for(role) / "train.jsonl"
    rows = _read_jsonl(path)
    have = {_norm(r.get("messages", [{}])[-1].get("content", "")) for r in rows
            if isinstance(r.get("messages"), list) and r["messages"]}
    stamp = getattr(tokenizer, "name_or_path", "") or ""
    added = 0
    for text in texts:
        if _norm(text) in have:
            continue
        messages = [{"role": "system", "content": persona},
                    {"role": "user", "content": DRAFT_REQUEST},
                    {"role": "assistant", "content": text}]
        try:
            rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                                                     add_generation_prompt=False,
                                                     enable_thinking=False)
        except TypeError:
            rendered = tokenizer.apply_chat_template(messages, tokenize=False,
                                                     add_generation_prompt=False)
        rows.append({"text": rendered, "messages": messages,
                     "metadata": {"postbot": True, "tokenized_for": stamp}})
        have.add(_norm(text))
        added += 1
    _write_jsonl(path, rows)
    return added


def write_queue(config: dict[str, Any], role: str, count: int) -> int:
    """The coming week's drafts, in the voice just trained, checked in code."""
    from symbio.app import skills, tooling
    from symbio.app.chat import make_logits_processors, make_sampler
    from symbio.app.dispatch import generate
    from symbio.app.modelload import load

    cfg = settings(config)
    model, tokenizer = load(skills.worker_model_name(config),
                            adapter_path=str(constants.adapter_dir_for(role)))
    recent = [p.get("text", "") for p in history()[-200:]]
    drafts: list[str] = []
    for _ in range(count * 3):
        if len(drafts) >= count:
            break
        messages = draft_messages(cfg["persona"], (recent + drafts)[-25:])
        prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True,
                                               enable_thinking=False)
        text = clean(tooling.strip_reasoning_block(str(generate(
            model, tokenizer, prompt=prompt, sampler=make_sampler(temp=1.0, top_p=0.95),
            logits_processors=make_logits_processors(repetition_penalty=1.1,
                                                     repetition_context_size=64),
            max_tokens=90, verbose=False))))
        if not check(text, recent + drafts, int(cfg["max_chars"])):
            drafts.append(text)
    _write_jsonl(_queue_path(), [{"text": t} for t in drafts])
    return len(drafts)
