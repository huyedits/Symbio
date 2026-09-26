"""The decision model: what kind of turn is this, decided off the GPU.

Before the 14B sees a message, something has to decide whether it should
think first. That was a regex (chat_text.needs_thinking). This is a small
learned model instead, and it runs on the Neural Engine side of the machine:

1. Apple Intelligence's on-device model, when it is enabled — it fills in a
   routing schema (symbio_ane "decide").
2. Otherwise a nearest-neighbour vote: the message is embedded and takes the
   vote of the closest labelled examples below. The embedding is MiniLM as a
   Core ML program on the Neural Engine when it has been built
   (symbio_ane/build_text_encoder.py), else Apple's NLContextualEmbedding —
   which, measured with macmon, runs on the CPU, not the Neural Engine.
3. Otherwise, or when the vote is too close to call, the regex.

The examples are generic seeds written here, not anyone's conversations.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from typing import Any

from symbio import constants

# What each label means for the turn. think: reason before answering.
LABELS = {
    "chat": {"think": False},     # greetings, small talk, quick facts, thanks
    "search": {"think": False},   # needs current or outside information
    "screen": {"think": False},   # look at / read the screen
    "action": {"think": True},    # do something on the machine: open, click, post, files
    "code": {"think": True},      # write, fix, explain code or errors
    "reason": {"think": True},    # plan, compare, analyse, multi-step problems
}

SEEDS: dict[str, list[str]] = {
    "chat": [
        "hey", "hi there!", "good morning", "how's it going?", "thanks, that helps",
        "lol nice", "what's 3+3?", "what's the capital of france?", "tell me a joke",
        "who are you?", "what's your name?", "cool, thanks!", "how are you today?",
        "what does 'ephemeral' mean?", "ok sounds good", "goodnight",
        "what's 9 times 7?", "what's 15% of 80?", "who painted the mona lisa?",
        "what does CPU stand for?", "tell me something interesting",
    ],
    "search": [
        "what's the weather in melbourne today?", "latest news about apple",
        "who won the game last night?", "what's the bitcoin price right now?",
        "search the web for mlx release notes", "when does the next iphone come out?",
        "look up the population of canberra", "what time does bunnings close today?",
        "find reviews of the m4 macbook air", "is the website down right now?",
        "what's the exchange rate usd to aud", "any updates on the openai lawsuit?",
    ],
    "screen": [
        "what's on my screen?", "look at my screen and tell me what's in front",
        "read the error message on screen", "what does that dialog say?",
        "can you see the browser window?", "what text is in the sidebar?",
        "take a look at this page", "what app is open right now?",
        "read me what the popup says", "what's written on the button?",
    ],
    "action": [
        "open safari", "click the post button", "tweet hello world", "open spotify and play music",
        "create a folder called invoices on my desktop", "delete the old logs in downloads",
        "set a reminder for 5pm to call mum", "send the form", "close all my chrome tabs",
        "move these files into the archive folder", "turn the volume down", "schedule a backup every night",
    ],
    "code": [
        "fix this error: KeyError 'model_name'", "write a python script that renames files",
        "why does my train.py crash with out of memory?", "refactor this function to be faster",
        "explain this traceback", "write a bash one-liner to count lines in all .py files",
        "add a unit test for the parser", "what's wrong with this regex?",
        "convert this javascript to typescript", "debug the websocket reconnect loop",
        "implement a binary search in rust", "review my pull request diff",
    ],
    "reason": [
        "plan a 3 day trip to tokyo on a budget", "compare these two job offers for me",
        "help me decide between a mac mini and a macbook", "think carefully: which option is cheapest over 5 years?",
        "analyse the pros and cons of renting vs buying", "if a train leaves at 3pm going 80km/h, when does it arrive 200km away?",
        "design a database schema for a library", "outline an essay on climate policy",
        "prove that the sum of two odd numbers is even", "what's the best strategy for this chess position?",
        "break this project into milestones", "estimate how long it would take to read 30 books",
        "should I move to a new city or stay where I am?", "which laptop should I buy for coding?",
        "is it worth upgrading my phone this year?", "should I learn rust or go first?",
    ],
}

K = 5
MIN_MARGIN = 0.02      # below this the vote is a coin toss: fall back to the regex

_lock = threading.Lock()
_index: dict[str, Any] = {}


def _space() -> str:
    """Which embedding the vectors come from: never mix the two."""
    from symbio.app import ane

    return "minilm-ane" if ane.encoder_dir() is not None else "nl-contextual"


def _embed(texts: list[str], timeout: float = 60.0) -> dict[str, Any]:
    from symbio.app import ane

    if _space() == "minilm-ane":
        return ane.encode(texts)
    return ane.request({"op": "embed", "texts": texts}, timeout=timeout)


def _seed_key() -> str:
    material = json.dumps(SEEDS, sort_keys=True) + _space()
    return hashlib.sha256(material.encode()).hexdigest()[:12]


def _load_index() -> list[tuple[str, list[float]]] | None:
    """Seed vectors, embedded once and cached beside the helper binary."""
    with _lock:
        if _index.get("key") == _seed_key():
            return _index["rows"]
        path = constants.PROJECT_DIR / "cache" / f"decider-{_seed_key()}.json"
        rows = None
        try:
            rows = [(label, vector) for label, vector in json.loads(path.read_text())]
        except (OSError, ValueError):
            texts = [(label, text) for label, items in SEEDS.items() for text in items]
            answer = _embed([t for _, t in texts])
            if not answer.get("ok"):
                return None
            rows = [(label, vector) for (label, _), vector in zip(texts, answer["vectors"])]
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(rows))
            except OSError:
                pass
        _index.update(key=_seed_key(), rows=rows)
        return rows


def classify(message: str) -> dict[str, Any] | None:
    """{"label", "think", "margin", "source"} from the embedding vote, or None."""
    rows = _load_index()
    if not rows:
        return None
    answer = _embed([message], timeout=5)
    if not answer.get("ok") or not answer.get("vectors"):
        return None
    query = answer["vectors"][0]
    scored = sorted(((sum(a * b for a, b in zip(query, vector)), label) for label, vector in rows),
                    reverse=True)[:K]
    votes: dict[str, float] = {}
    for score, label in scored:
        votes[label] = votes.get(label, 0.0) + math.exp(score * 20)   # closer counts more
    ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
    total = sum(votes.values())
    top, top_votes = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    return {"label": top, "think": LABELS[top]["think"],
            "margin": (top_votes - second) / total, "source": _space(),
            "ms": answer.get("ms")}


def warm() -> None:
    """Start the helper and embed the seeds before the first turn needs them."""
    try:
        if _load_index():
            classify("warm up")
    except Exception:
        pass


def decide(message: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """The turn's decision, from the best source that can give one.

    Measured 2026-09-26 on 31 held-out messages (think / no-think), with
    macmon on the power rails:

        vote over MiniLM on the Neural Engine   30/31   <1 ms   ANE 1.8 W
        Apple Intelligence's on-device model    27/31   ~0.7 s  ANE 4.7 W
        vote over Apple's contextual embedding  30/31   ~8 ms   CPU only
        regex                                   19/31

    So the Neural Engine vote goes first, Apple's model when that encoder has
    not been built (still the Neural Engine), the CPU vote after that, and
    the regex last. A vote too close to call falls through to the next.
    """
    from symbio.app import ane
    from symbio.app.chat_text import needs_thinking

    if ane.enabled(config) and ((config or {}).get("ane") or {}).get("decide", True):
        voted = classify(message) if _space() == "minilm-ane" else None
        if voted and voted["margin"] >= MIN_MARGIN:
            return voted
        apple = ane.decide(message)
        if apple.get("ok"):
            return {"label": apple.get("route", "chat"), "think": bool(apple.get("think")),
                    "source": "apple-intelligence", "ms": apple.get("ms")}
        if voted is None:
            voted = classify(message)
        if voted and voted["margin"] >= MIN_MARGIN:
            return voted
    return {"label": None, "think": needs_thinking(message), "source": "regex"}
