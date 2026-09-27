"""Guardrails: what Symbio may do on its own, in the words you'd use for it.

The gate used to be three lists of tool NAMES plus a risk score, and both
halves were wrong in the same direction. Too rigid: a post to x.com asked
twice — once as "Allow tool 'post_to_x'?", once with the text — and every
prompt spoke in scores and flags ("risk score 3/3 ... Flags: publishes_publicly,
irreversible"). Too porous: the name was the only thing checked, so on
2026-09-27 the model typed "Hi" into X's composer with browser_type and
clicked Post with browser_click — two tools on no list — and it went out,
publicly, as the user, with no question asked. The approval that existed for
posting guarded one route to it.

So the gate now asks about KINDS of action, which the user can read and set:
"Post or send: always ask". A click is judged by what it lands on — a button
that sends the text in a box is `publish` on any site, however the model
reached it — and every question is one plain-English card saying what will
happen, with the exact text or command underneath it. There is no posting
command: the model posts the way a person does, in the page, and the page is
where the question is asked.

Standard library only. The desktop window loads this file by path, the same
way it loads constants.py, because importing the `symbio` package costs it
105 MB of agent it has no use for.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

MODES = ("allow", "risky", "ask", "block")

MODE_LABELS = {
    "allow": "Always allow",
    "risky": "Ask if risky",
    "ask": "Always ask",
    "block": "Never",
}

MODE_HINTS = {
    "allow": "Runs without asking. Anything destructive still asks.",
    "risky": "Asks only when the action looks dangerous or unrequested.",
    "ask": "Asks every time, with what it will do in plain English.",
    "block": "Refused. Symbio is told it isn't allowed and why.",
}

# id, label, what it covers, default mode, the tools that are this kind.
# The defaults are the old behaviour: what used to ask by name asks, what was
# left to the risk score is "risky".
KINDS: tuple[tuple[str, str, str, str, tuple[str, ...]], ...] = (
    ("publish", "Post or send",
     "Posting, replying, sending messages and submitting forms on websites, under your name.",
     "ask", ("submit_form",)),
    ("commands", "Run commands and code",
     "Shell commands and Python on this Mac, and commands on machines you've added.",
     "risky", ("run_command", "terminal", "execute_code", "run_remote")),
    ("files", "Change files",
     "Writing and editing files, and saving commands to run later.",
     "risky", ("write_file", "edit_file", "patch", "save_command")),
    ("desktop", "Use your desktop",
     "Clicking, typing and dragging in other apps, opening apps, recording the screen.",
     "risky", ("desktop_click", "desktop_type", "desktop_press", "desktop_drag",
               "desktop_move", "desktop_scroll", "desktop_hotkey", "open_app",
               "obs_record")),
    ("browser", "Browse the web",
     "Opening pages, and clicking, typing and scrolling in Symbio's own browser.",
     "risky", ("browser_open", "browser_click", "browser_click_at", "browser_type",
               "browser_press", "browser_scroll", "browser_close", "fill_form")),
    ("forget", "Forget things",
     "Deleting notes.",
     "ask", ("delete_note",)),
    ("training", "Train itself",
     "Fine-tuning, rebuilding its adapter, folding notes into training data, realigning.",
     "ask", ("train_adapter", "retrain_adapter", "digest_notes", "realign")),
    ("settings", "Change its settings",
     "Editing Symbio's own configuration.",
     "ask", ("config_set",)),
    ("schedule", "Schedule work",
     "Creating, changing and deleting scheduled jobs.",
     "ask", ("schedule_job", "update_cron_job", "delete_cron_job")),
)

KIND_IDS = tuple(k[0] for k in KINDS)
_LABELS = {k[0]: k[1] for k in KINDS}
_DEFAULTS = {k[0]: k[3] for k in KINDS}
_BY_TOOL = {tool: k[0] for k in KINDS for tool in k[4]}

# Refused whatever the switches say. Listed in the window so they are visible,
# never editable from it.
FLOORS = (
    "Changing its own security policy (security.md) from inside a chat.",
    "Deleting its own memory, adapters or training data with a command or a file write.",
    "Loosening these guardrails itself: a change the model asks for is always shown to you first.",
)

# Kinds a person who is somewhere else (the Telegram gateway) cannot judge
# from there: "click at 400,300" means nothing on a phone. "Ask if risky"
# becomes "always ask" for them, which is what the old name list did.
REMOTE_ASKS = ("commands", "files", "desktop")


def label(kind: str | None) -> str:
    return _LABELS.get(kind or "", "Other actions")


def kind_of(name: str) -> str | None:
    """The kind a tool is by name. A browser click can still turn out to be
    `publish` once it is known what it lands on; see ToolsMixin."""
    return _BY_TOOL.get(name)


def modes(config: dict[str, Any] | None) -> dict[str, str]:
    """Every kind's mode: the user's choice where they made one, else the default."""
    chosen = {}
    if isinstance(config, dict):
        section = config.get("guardrails")
        if isinstance(section, dict) and isinstance(section.get("modes"), dict):
            chosen = section["modes"]
    out = {}
    for kind in KIND_IDS:
        value = str(chosen.get(kind, "")).strip().lower()
        out[kind] = value if value in MODES else _DEFAULTS[kind]
    return out


def mode_for(kind: str | None, config: dict[str, Any] | None,
             remote: bool = False) -> str:
    """How this kind of action is gated. An action of no kind is "risky":
    the risk scorer alone decides, as it always did."""
    if kind is None:
        return "risky"
    mode = modes(config)[kind]
    if remote and mode == "risky" and kind in REMOTE_ASKS:
        return "ask"
    return mode


# ── reading and writing the user's choices ───────────────────────────

def read_section(config_file: Path) -> dict[str, Any]:
    """The `guardrails` section of config.json, or {}."""
    try:
        data = json.loads(Path(config_file).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    section = data.get("guardrails") if isinstance(data, dict) else None
    return section if isinstance(section, dict) else {}


def set_mode(config_file: Path, kind: str, mode: str) -> dict[str, Any]:
    """Write one kind's mode to config.json. Atomic, and nothing else changes.

    Only a known kind and a known mode are accepted: this is reachable from
    the window, so the two lists here ARE the boundary of what it can write.
    """
    if kind not in KIND_IDS:
        return {"ok": False, "error": f"Unknown kind of action: {kind!r}."}
    if mode not in MODES:
        return {"ok": False, "error": f"Unknown mode: {mode!r}."}
    path = Path(config_file)
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, ValueError) as e:
        return {"ok": False, "error": f"Could not read {path.name}: {e}"}
    if not isinstance(data, dict):
        return {"ok": False, "error": f"{path.name} is not a JSON object."}
    section = data.setdefault("guardrails", {})
    if not isinstance(section, dict):
        section = data["guardrails"] = {}
    chosen = section.setdefault("modes", {})
    if not isinstance(chosen, dict):
        chosen = section["modes"] = {}
    before = modes(data).get(kind)
    chosen[kind] = mode
    temporary = path.with_suffix(".json.tmp")
    try:
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    except OSError as e:
        return {"ok": False, "error": f"Could not write {path.name}: {e}"}
    return {"ok": True, "kind": kind, "mode": mode, "was": before}


def describe_all(config: dict[str, Any] | None) -> dict[str, Any]:
    """What the window's Guardrails panel shows."""
    current = modes(config)
    return {
        "kinds": [{"id": k, "label": lab, "hint": hint, "mode": current[k],
                   "default": default}
                  for k, lab, hint, default, _tools in KINDS],
        "modes": [{"id": m, "label": MODE_LABELS[m], "hint": MODE_HINTS[m]}
                  for m in MODES],
        "floors": list(FLOORS),
    }


# ── the log of what was asked and answered ───────────────────────────

LOG_NAME = "guardrails.jsonl"


def record(log_dir: Path, entry: dict[str, Any]) -> None:
    """One decision, appended. A log that cannot be written costs nothing."""
    try:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(log_dir) / LOG_NAME, "a", encoding="utf-8") as f:
            f.write(json.dumps({"at": time.time(), **entry}) + "\n")
    except OSError:
        pass


def recent(log_dir: Path, limit: int = 20) -> list[dict[str, Any]]:
    try:
        lines = (Path(log_dir) / LOG_NAME).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out = []
    for line in reversed(lines[-(limit * 4):]):
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
        if len(out) >= limit:
            break
    return out


# ── the question itself ──────────────────────────────────────────────

class Card(str):
    """An approval question that is still a plain string.

    Every front-end already takes a string — the terminal prints it, Telegram
    sends it — so this IS one, and the parts the window lays out separately
    ride along as attributes: what will happen (`headline`), exactly what
    (`details`: the post, the command), why it is being asked (`reason`), and
    which kind "always allow" would switch.
    """

    headline: str
    details: str
    kind: str | None
    reason: str
    said_by: str
    warning: str

    def __new__(cls, headline: str, details: str = "", kind: str | None = None,
                reason: str = "", said_by: str = "harness", warning: str = ""):
        lines = [headline]
        if warning:
            lines.append(f"  ! {warning}")
        if details:
            lines.extend("  " + line for line in details.splitlines())
        if reason:
            lines.append(f"  ({reason})")
        card = super().__new__(cls, "\n".join(lines))
        card.headline = headline
        card.details = details
        card.kind = kind
        card.reason = reason
        card.said_by = said_by
        card.warning = warning
        return card

    def as_dict(self) -> dict[str, Any]:
        return {"headline": self.headline, "details": self.details,
                "kind": self.kind, "kind_label": label(self.kind) if self.kind else "",
                "reason": self.reason, "said_by": self.said_by,
                "warning": self.warning}


# Risk flags, said the way a person would say them.
_FLAG_WORDS = (
    (r"destructive|rm_r|delete|wipe", "it deletes things"),
    (r"blocked_binary", "it uses a command the sandbox normally blocks"),
    (r"shell_syntax", "it uses shell features like pipes or redirects"),
    (r"path_escape|outside_project", "it reaches outside Symbio's folder"),
    (r"sensitive_(file|path)|secret|credential|ssh", "it touches secrets or keys"),
    (r"sensitive_config", "it changes a setting that affects trust"),
    (r"injection", "the text looks like it was written to steer me"),
    (r"network|curl|download|exfil", "it sends or fetches data over the network"),
    (r"remote_host", "it runs on another machine"),
    (r"privilege|sudo", "it asks for admin rights"),
    (r"unrequested:first_use_after_untrusted",
     "it's the first time I'd use this, right after reading outside text"),
    (r"unrequested:no_action_asked", "you didn't ask me to do anything"),
    (r"publishes_publicly|public_act", "it's public, under your name"),
    (r"irreversible", "it can't be undone from here"),
    (r"screen_capture|records_screen", "it records the screen"),
)


def reason_for(flags: list[str] | tuple[str, ...], mode: str,
               kind: str | None = None, remote: bool = False) -> str:
    """Why this is being asked, as a sentence naming the user's own setting,
    so the card says which switch would change the answer."""
    name = f"“{label(kind)}”" if kind else ""
    if mode == "ask":
        if remote:
            return (f"You're away from this Mac, so {name} asks every time."
                    if kind else "You're away from this Mac, so this asks.")
        return f"You set {name} to ask every time." if kind else ""
    said: list[str] = []
    for flag in flags or ():
        for pattern, words in _FLAG_WORDS:
            if re.search(pattern, str(flag), re.IGNORECASE) and words not in said:
                said.append(words)
                break
    if not said:
        because = "it looked risky"
    elif len(said) == 1:
        because = said[0]
    else:
        because = ", ".join(said[:-1]) + " and " + said[-1]
    if mode == "allow" and kind:
        return f"You set {name} to always allow, but {because}, so this one asks."
    if kind:
        return f"You set {name} to ask if risky, and {because}."
    return f"Asking because {because}."


_QUOTED = re.compile(r"[\"“”‘’'«»]([^\"“”«»]{1,280}?)[\"“”’'«»]")


def quoted_texts(user_text: str) -> list[str]:
    """Text the user put in quotes — the words they want used verbatim.

    Pairs are matched loosely on purpose: the live request was
    `make a tweet “testing"`, a curly quote closed by a straight one.
    """
    found = []
    for m in _QUOTED.finditer(user_text or ""):
        text = m.group(1).strip()
        # An apostrophe pair inside a sentence ("don't ... it's") is not a quote.
        if text and not re.match(r"^[a-z]\b", text) and len(text.split()) <= 60:
            found.append(text)
    return found


def _norm(text: str) -> str:
    return " ".join(str(text or "").split()).strip(" .!?\"'“”").casefold()


def mismatch_warning(user_text: str, outgoing: str) -> str:
    """"You asked for “testing”." when the text about to go out is not the text
    the user quoted. Empty when they quoted nothing, or it matches."""
    wanted = quoted_texts(user_text)
    if not wanted or not outgoing:
        return ""
    if any(_norm(w) == _norm(outgoing) for w in wanted):
        return ""
    return f"You asked for “{wanted[0]}”, not this."
