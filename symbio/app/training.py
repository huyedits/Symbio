"""Training-data accumulation, note/memory digestion, and LoRA fine-tuning."""

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from symbio import constants
from symbio.app import config as app_config
from symbio.app.tooling import clean_response


def _train_file_for(role: str | None) -> Path:
    # role=None reads constants.TRAIN_FILE directly (not re-derived from
    # constants.DATA_DIR) so code/tests that monkeypatch TRAIN_FILE alone
    # — the pre-existing, still-common pattern — keep working unchanged.
    return constants.TRAIN_FILE if role is None else constants.data_dir_for(role) / "train.jsonl"


def _valid_file_for(role: str | None) -> Path:
    return constants.VALID_FILE if role is None else constants.data_dir_for(role) / "valid.jsonl"


def append_training_text(text: str, role: str | None = None):
    train_file = _train_file_for(role)
    train_file.parent.mkdir(parents=True, exist_ok=True)
    with open(train_file, "a", encoding="utf-8") as f:
        json.dump({"text": text}, f)
        f.write("\n")


def build_chat_training_sample(messages: list[dict[str, str]], tokenizer) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )


def append_chat_pair(user_msg: str, assistant_msg: str, tokenizer, system_prompt: str,
                     role: str | None = None):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": clean_response(assistant_msg)},
    ]
    append_training_text(build_chat_training_sample(messages, tokenizer), role=role)


def _note_timestamp(f: Path) -> datetime:
    """When was this note learned? Filenames carry a %Y%m%d_%H%M%S prefix;
    fall back to mtime for notes that don't."""
    try:
        return datetime.strptime(f.name[:15], "%Y%m%d_%H%M%S")
    except ValueError:
        return datetime.fromtimestamp(f.stat().st_mtime)


def drop_note_training_samples(title: str) -> int:
    """Remove a digested note's samples from the training/validation data,
    matched by the distinctive user-turn questions digestion writes. Sweeps
    every digested version of the note, not just the latest."""
    topic = title.replace("_", " ").replace("-", " ")
    markers = (
        f"Write a markdown note titled '{title}'.",
        f"According to your notes, what do you know about '{topic}'?",
    )
    dropped = 0
    for data_file in (constants.TRAIN_FILE, constants.VALID_FILE):
        if not data_file.exists():
            continue
        kept, hit = [], 0
        for line in data_file.read_text(encoding="utf-8").splitlines():
            try:
                text = json.loads(line).get("text", "") if line.strip() else ""
            except (json.JSONDecodeError, AttributeError):
                text = ""
            if any(m in text for m in markers):
                hit += 1
                continue
            kept.append(line)
        if hit:
            data_file.write_text("\n".join(kept) + ("\n" if kept else ""),
                                 encoding="utf-8")
            dropped += hit
    return dropped


def decay_research_notes(config: dict[str, Any]) -> list[str]:
    """Archive auto-learned 'Learned:' notes older than learn.note_decay_days
    and drop their digested samples from the training data, so stale web facts
    stop being served by RAG and retrained into the weights on every digest.
    Deliberate notes, skills, and curated memory never decay; a re-asked
    question re-learns the fact fresh via auto-search. 0 disables decay.
    Returns the archived filenames."""
    days = int(config.get("learn", {}).get("note_decay_days", 90))
    if days <= 0:
        return []
    cutoff = datetime.now() - timedelta(days=days)
    archived = []
    for f in sorted(constants.NOTES_DIR.glob("*.md")):
        if not f.is_file():
            continue
        try:
            first_line = f.read_text(encoding="utf-8").strip().splitlines()[0]
        except (OSError, IndexError):
            continue
        if not first_line.startswith("# Learned:"):
            continue
        if _note_timestamp(f) > cutoff:
            continue
        drop_note_training_samples(first_line[2:].strip())
        dest = constants.NOTES_ARCHIVE_DIR / f.name
        counter = 1
        while dest.exists():
            dest = constants.NOTES_ARCHIVE_DIR / f"{f.stem}_{counter}{f.suffix}"
            counter += 1
        f.rename(dest)
        archived.append(f.name)
    return archived


def digest_notes_to_training(tokenizer, system_prompt: str,
                             config: dict[str, Any] | None = None) -> int:
    files = sorted(constants.NOTES_DIR.glob("*.md"))

    manifest: dict[str, str] = {}
    if constants.DIGEST_MANIFEST.exists():
        try:
            manifest = json.loads(constants.DIGEST_MANIFEST.read_text())
        except Exception:
            manifest = {}

    added = 0
    new_manifest = {}

    for f in files:
        content = f.read_text(encoding="utf-8").strip()
        if not content:
            continue

        h = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        new_manifest[f.name] = h
        if manifest.get(f.name) == h:
            continue

        lines = content.splitlines()
        title = f.stem.replace("_", " ")
        body = content
        if lines and lines[0].startswith("# "):
            title = lines[0][2:].strip()
            body = "\n".join(lines[1:]).strip()

        if len(body) < 5:
            continue

        topic = title.replace("_", " ").replace("-", " ")

        # Direct note reproduction
        messages_doc = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Write a markdown note titled '{title}'."},
            {"role": "assistant", "content": body},
        ]
        append_training_text(build_chat_training_sample(messages_doc, tokenizer))

        # Question/answer from notes
        messages_qa = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"According to your notes, what do you know about '{topic}'?"},
            {"role": "assistant", "content": body},
        ]
        append_training_text(build_chat_training_sample(messages_qa, tokenizer))

        added += 2

    # Curated memory and the user profile hold what the agent has figured out
    # about its user; digest them too so those facts survive fine-tuning, not
    # just prompt injection. Hash-tracked like notes: re-digested on change.
    user_name = (config or app_config.load_config())["user_name"]
    stores = [
        (constants.MEMORY_FILE, "What do you have saved in your long-term memory?"),
        (constants.PROFILE_FILE, f"What do you know about {user_name}?"),
    ]
    for path, question in stores:
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8").strip()
        if len(content) < 5:
            continue
        h = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        new_manifest[path.name] = h
        if manifest.get(path.name) == h:
            continue
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": content},
        ]
        append_training_text(build_chat_training_sample(messages, tokenizer))
        added += 1

    constants.DIGEST_MANIFEST.write_text(json.dumps(new_manifest, indent=2))
    return added


def seed_training_data(tokenizer, system_prompt: str, config: dict[str, Any]):
    """Seed a minimal clean corpus so the model has correct identity/tool examples
    even before any real conversation is saved."""
    if constants.TRAIN_FILE.exists() and constants.TRAIN_FILE.stat().st_size > 0:
        return

    assistant = config["assistant_name"]
    user = config["user_name"]

    samples = [
        # Identity
        (
            "What is your name?",
            f"I am {assistant}, your personal AI assistant.",
        ),
        (
            f"My name is {user}.",
            f"Nice to meet you, {user}! I'll remember that. <note title='User Identity'>{user} is my user's name.</note>",
        ),
        (
            "What is my name?",
            f"Your name is {user}.",
        ),
        (
            f"Are you {user}?",
            f"No — I'm {assistant}, your assistant. You're {user}.",
        ),
        (
            "Who is who here?",
            f"I'm {assistant}, the assistant. You're {user}, my user.",
        ),
        # Tool-use demonstration: "remember a fact" = note, "remind me at a time" = cron
        (
            "Please remember that I prefer concise replies.",
            "Got it. <note title='User Preference'>The user prefers concise replies.</note>",
        ),
        (
            "Remember that my favorite color is green.",
            f"Noted, {user}. <note title='Favorite Color'>{user}'s favorite color is green.</note>",
        ),
        (
            "Save your notes and train on them.",
            "<digest /><train />I'll digest the notes and start training so they stick.",
        ),
        # Cron scheduling demonstrations
        (
            "Remind me every day at 9am to stretch.",
            f"Will do, {user}. <cron expr='0 9 * * *'>stretch</cron>",
        ),
        (
            "It's 14:00 right now. Remind me in 30 minutes to check the oven.",
            "Got it — that's 14:30. <cron at='14:30'>check the oven</cron>",
        ),
        (
            "Check disk space every morning at 8.",
            "Scheduled a daily disk check. <cron expr='0 8 * * *'>cmd:df -h</cron>",
        ),
        (
            "Remind me every Monday at 10 to review my notes.",
            f"Done, {user}. <cron expr='0 10 * * 1'>review your notes</cron>",
        ),
    ]

    # Resilience: use this machine's native commands, and recover from a
    # failed command instead of giving up.
    if platform.system() == "Darwin":
        open_chrome = "open -a 'Google Chrome'"
        url_opener = "open"
    elif platform.system() == "Windows":
        open_chrome = "start chrome"
        url_opener = "start"
    else:
        open_chrome = "xdg-open https://www.google.com"
        url_opener = "xdg-open"
    samples += [
        (
            "Open Chrome.",
            f"<cmd>{open_chrome}</cmd> Opening Chrome for you, {user}.",
        ),
        (
            "Open Google Chrome.",
            f"<cmd>{open_chrome}</cmd> Opening Google Chrome for you, {user}.",
        ),
        (
            "Open google chrome.",
            f"<cmd>{open_chrome}</cmd> Opening Google Chrome for you, {user}.",
        ),
        (
            "Launch Chrome.",
            f"<cmd>{open_chrome}</cmd> Launching Chrome for you, {user}.",
        ),
        (
            "Start Google Chrome for me.",
            f"<cmd>{open_chrome}</cmd> Starting Google Chrome for you, {user}.",
        ),
        (
            "[System observation: Command 'chrome' exited error.\nOutput:\nCommand not found: chrome]",
            f"'chrome' isn't a command on this machine — trying the native way instead. <cmd>{open_chrome}</cmd>",
        ),
        # Run Python for exact computation, then answer from the output
        (
            "Run code to calculate 7 factorial.",
            "<py>import math\nprint(math.factorial(7))</py> Running that now.",
        ),
        (
            "[System observation: Python script exited ok.\nOutput:\n5040]",
            "7! = 5,040.",
        ),
        (
            "What's 2 to the power of 40, exactly?",
            "<py>print(2 ** 40)</py> Computing it exactly.",
        ),
        # Self-configuration on request
        (
            "Make your replies more creative.",
            "<config set='agent.temperature'>0.9</config> Done — turning up the creativity.",
        ),
        # Skills: save a working multi-step approach for reuse
        (
            "That worked — remember how you checked the disk.",
            "<skill name='Check disk space'>1. Run df -h in the sandbox. 2. Report the Use% of /.</skill> "
            "Saved it as a skill so I can reuse those steps next time.",
        ),
        # Curated memory: durable preferences go to <profile>/<memory>
        (
            "I prefer replies in bullet points from now on.",
            f"Got it, {user} — bullets from here on. <profile>Prefers replies formatted as bullet points.</profile>",
        ),
        # Current information = <search> and answer from the returned results
        (
            "What is the latest news?",
            "<search>latest news</search> Searching now.",
        ),
        (
            "[System observation: Web search for 'latest news' succeeded.\nResults:\n"
            "1. Major storm reaches coast\n   https://example.com/storm\n"
            "   The storm made landfall this morning.]",
            f"Here's the latest, {user}: a major storm made landfall this morning (example.com).",
        ),
        (
            "What's the weather in Tokyo right now?",
            "<search>Tokyo weather now</search> Checking.",
        ),
        # Unknown facts = search, never guess
        (
            "Who won the Tour de France in 2031?",
            "I don't have that memorized — checking. <search>Tour de France 2031 winner</search>",
        ),
        # Just opening a search for the user to look at themselves = <cmd>open>,
        # since nothing more is needed from the agent afterward.
        (
            "Search YouTube for lofi beats.",
            f"<cmd>{url_opener} 'https://www.youtube.com/results?search_query=lofi+beats'</cmd> "
            f"Opening YouTube results for lofi beats, {user}.",
        ),
        (
            "Search the web for the weather in Sydney.",
            f"<cmd>{url_opener} 'https://www.google.com/search?q=weather+in+Sydney'</cmd> "
            "Here's a web search for Sydney weather.",
        ),
        # But interacting with the results (clicking one) needs the agent's
        # OWN controllable browser, not the user's — <cmd>open> hands the
        # page to the user's browser with no way back, so <browse>+<click>
        # is the only tag pair that can actually finish this request.
        (
            "Open YouTube, search for nuclear power plant, and click the first video you see.",
            "<browse>https://www.youtube.com/results?search_query=nuclear+power+plant</browse> "
            "Opening the search now — I'll click the first video once it loads.",
        ),
        (
            "[System observation: Opened browser at https://www.youtube.com/results?"
            "search_query=nuclear+power+plant. Page title: nuclear power plant - YouTube\n\n"
            "Page text now:\nNuclear Power Plant Explained\nHow Nuclear Reactors Work\n"
            "Inside a Nuclear Power Station]",
            "<click>Nuclear Power Plant Explained</click> Clicking the first video.",
        ),
        # "Open X in Chrome" still means the agent's controllable browser,
        # because the user may ask to click/scroll next. Use <browse>, not <cmd>open>.
        # The native macOS opener with a URL opens the user's browser and leaves
        # the agent unable to click, so this is a mistake pattern to recover from.
        (
            "Open cloudflare.com in Chrome and click the first button.",
            "<browse>https://www.cloudflare.com</browse> Opening Cloudflare in the controllable browser — I'll click the first button once it loads.",
        ),
        (
            "[System observation: Opened browser at https://www.cloudflare.com. "
            "Page title: Cloudflare – The Web Performance & Security Company\n\n"
            "Page text now:\nStart building for free\nContact Sales]",
            "<click>Start building for free</click> Clicking the first button.",
        ),
        (
            "[System observation: Opened browser at https://www.cloudflare.com. "
            "Page title: Cloudflare – The Web Performance & Security Company]",
            "<scroll /> Scrolling to the bottom.",
        ),
        (
            "[System observation: Browser click error: Browser is not open. "
            "Use <browse>https://...</browse> to load a page first, then retry the action.]",
            "The browser isn't open. Let me load the page first. "
            "<browse>https://www.cloudflare.com</browse> Reopening Cloudflare so I can click.",
        ),
        (
            "[System observation: Command 'open -a 'Google Chrome' \'https://www.cloudflare.com'' exited ok.\n"
            "Output:]",
            "That opened Cloudflare in your default browser, which I can't click. "
            "Let me use my controllable browser instead. "
            "<browse>https://www.cloudflare.com</browse> Opening Cloudflare so I can click for you.",
        ),
        # Keyboard navigation inside the browser uses <press>, never a shell command.
        (
            "[System observation: Opened browser at https://www.youtube.com/shorts. "
            "Page title: YouTube Shorts]",
            "<press>down</press> Pressing the down arrow key to scroll to the next Short.",
        ),
        (
            "Press the down arrow key.",
            "<press>down</press> Pressing the down arrow key.",
        ),
        (
            "Press enter.",
            "<press>enter</press> Pressing Enter.",
        ),
        (
            "Hit space to play.",
            "<press>space</press> Pressing Space to play/pause.",
        ),
        (
            "[System observation: Command 'keydown --arrow down' exited error.\n"
            "Output:\nCommand not found: keydown]",
            "There is no `keydown` shell command — I'll press the key directly in the browser instead. "
            "<press>down</press> Pressing the down arrow key.",
        ),
        # Don't just talk about a URL — open it. If the user names a site, browse there.
        (
            "Open the EPA privacy page.",
            "<browse>https://www.epa.gov/privacy</browse> Opening the EPA privacy page now.",
        ),
        (
            "Can you open https://www.epa.gov/privacy?",
            "<browse>https://www.epa.gov/privacy</browse> Opening the EPA privacy page now.",
        ),
        (
            "Go to epa.gov/privacy.",
            "<browse>https://www.epa.gov/privacy</browse> Going to the EPA privacy page now.",
        ),
        # Stop asking the user to open pages; the agent should perform the action.
        (
            "Here is the link: <https://www.epa.gov/privacy>. Would you like to open this page now?",
            "I should just open it for you. <browse>https://www.epa.gov/privacy</browse> Opening the EPA privacy page.",
        ),
        # If a command with a URL opens the user's default browser, recover by browsing.
        (
            "[System observation: Command 'open -a 'Google Chrome' \'https://www.epa.gov/privacy'' exited ok.\n"
            "Output:]",
            "That opened the page in your default browser, which I can't control. "
            "<browse>https://www.epa.gov/privacy</browse> Opening it in my browser instead.",
        ),
        # If the user only wants a search result handed to them (nothing to do after),
        # <cmd>open> is fine — but as soon as interaction is possible, switch to <browse>.
        (
            "Search the web for 'start building for free' so I can read it myself.",
            "<cmd>open 'https://www.google.com/search?q=start+building+for+free'</cmd> "
            "Here's a Google search for that phrase.",
        ),
    ]

    # General resilience: don't give up or ask the user to do the work.
    # If a tool fails, recover by using a working tool instead.
    samples += [
        (
            "[System observation: Browser open error: Only http/https URLs are allowed. Got: epa.gov/privacy]",
            "I need the full URL. <browse>https://www.epa.gov/privacy</browse> Opening the EPA privacy page with https.",
        ),
        (
            "Click the first button.",
            "<browse>https://www.cloudflare.com</browse> I need to open the page first before I can click.",
        ),
        (
            "[System observation: Browser click error: nothing matches selector '#submit'.]",
            "That selector didn't match anything visible. Let me try clicking by text instead. "
            "<click>Submit</click> Clicking the Submit button by its visible text.",
        ),
    ]

    for user_msg, assistant_msg in samples:
        append_chat_pair(user_msg, assistant_msg, tokenizer, system_prompt)


def ensure_validation_split(every_nth: int = 10, max_samples: int = 24,
                            role: str | None = None):
    """mlx_lm silently skips evaluation when valid.jsonl is missing, which
    makes steps_per_eval meaningless. Sample a small validation set from the
    training data so eval loss is always reported."""
    train_file = _train_file_for(role)
    valid_file = _valid_file_for(role)
    if valid_file.exists() and valid_file.stat().st_size > 0:
        return
    lines = [l for l in train_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    sample = lines[::every_nth][:max_samples] or lines[:1]
    valid_file.write_text("\n".join(sample) + "\n", encoding="utf-8")


def run_training(config: dict[str, Any], iters: int | None = None,
                 role: str | None = None, model_name: str | None = None) -> bool:
    """Run a LoRA fine-tune. `iters` overrides lora.iters for short passes
    (e.g. the correction-learning batches). `role`/`model_name` train a
    worker's own adapter against its own data directory instead of the
    headmaster's — role is None everywhere except symbio.app.dispatch."""
    train_file = _train_file_for(role)
    data_dir = train_file.parent
    adapter_dir = constants.adapter_dir_for(role)
    if not train_file.exists() or train_file.stat().st_size == 0:
        print("  [System] No training data available.")
        return False
    ensure_validation_split(role=role)

    # Sweep temp LoRA config files left behind by previous crashed runs.
    for stale in data_dir.glob("tmp*.yaml"):
        try:
            stale.unlink()
        except OSError:
            pass

    lora = config["lora"]
    print("\n  [System] Starting MLX LoRA Fine-Tuning\n")

    # mlx_lm only accepts rank/dropout/scale via a config file, not CLI flags.
    lora_config = {
        "lora_parameters": {
            "rank": lora["rank"],
            "dropout": lora["dropout"],
            "scale": lora["scale"],
        }
    }
    config_fd, config_path = tempfile.mkstemp(suffix=".yaml", dir=str(data_dir))
    with os.fdopen(config_fd, "w") as f:
        yaml.dump(lora_config, f)

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", model_name or config["model_name"],
        "--train",
        "--data", str(data_dir),
        "--batch-size", str(lora["batch_size"]),
        "--num-layers", str(lora["num_layers"]),
        "--iters", str(iters if iters is not None else lora["iters"]),
        "--learning-rate", str(lora["learning_rate"]),
        "--steps-per-eval", str(lora["steps_per_eval"]),
        "--max-seq-length", str(lora["max_seq_length"]),
        "--adapter-path", str(adapter_dir),
        "--save-every", str(lora["save_every"]),
        "--config", config_path,
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("  [System] Training failed.")
        return False
    except KeyboardInterrupt:
        print("  [System] Training stopped.")
        return False
    finally:
        try:
            os.unlink(config_path)
        except OSError:
            pass

    config_file = adapter_dir / "adapter_config.json"
    weight_files = list(adapter_dir.glob("adapters.*"))
    if not config_file.exists() or not weight_files:
        print("  [System] Adapter files missing after training.")
        return False

    adapter_kb = sum(f.stat().st_size for f in adapter_dir.iterdir() if f.is_file()) // 1024
    print(f"  [System] Adapter baked. Size: ~{adapter_kb:,} KB")
    return True


def backup_adapter(role: str | None = None) -> Path | None:
    """Snapshot the current adapter before a training run, so a regression
    caught by the golden set can be rolled back. Returns None when there is
    no existing adapter to protect (e.g. the very first training run)."""
    adapter_dir = constants.adapter_dir_for(role)
    if not adapter_dir.exists() or not any(adapter_dir.iterdir()):
        return None
    backup_dir = adapter_dir.parent / f"{adapter_dir.name}.bak.{datetime.now():%Y%m%d_%H%M%S_%f}"
    shutil.copytree(adapter_dir, backup_dir)
    return backup_dir


def restore_adapter(backup_dir: Path, role: str | None = None):
    """Replace the current adapter with a previously backed-up one."""
    adapter_dir = constants.adapter_dir_for(role)
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    shutil.copytree(backup_dir, adapter_dir)


def discard_adapter_backup(backup_dir: Path | None):
    """Remove a backup once it is no longer needed (training kept)."""
    if backup_dir and backup_dir.exists():
        shutil.rmtree(backup_dir, ignore_errors=True)


_ADAPTER_LAST_USED_FILE_NAME = "last_used.json"


def adapter_last_used(role: str | None = None) -> datetime | None:
    """When was this adapter last loaded into a session? None if it has
    never been tracked (e.g. just trained, or from before this feature)."""
    path = constants.adapter_dir_for(role) / _ADAPTER_LAST_USED_FILE_NAME
    if not path.exists():
        return None
    try:
        return datetime.fromisoformat(json.loads(path.read_text(encoding="utf-8"))["last_used"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def mark_adapter_used(role: str | None = None):
    """Record that this adapter was just loaded into a session, resetting
    the idle clock the reminder in ChatSession checks against."""
    adapter_dir = constants.adapter_dir_for(role)
    if not adapter_dir.exists():
        return
    path = adapter_dir / _ADAPTER_LAST_USED_FILE_NAME
    path.write_text(json.dumps({"last_used": datetime.now().isoformat()}), encoding="utf-8")


def remove_adapter(role: str | None = None):
    """Delete this adapter entirely, reverting to the base model."""
    adapter_dir = constants.adapter_dir_for(role)
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)


def prune_adapters(role: str | None = None) -> dict[str, Any]:
    """Remove intermediate checkpoints and report adapter footprint."""
    adapter_dir = constants.adapter_dir_for(role)
    removed = []
    for cp in adapter_dir.glob("[0-9]*_adapters.*"):
        cp.unlink()
        removed.append(cp.name)

    total_bytes = sum(f.stat().st_size for f in adapter_dir.iterdir() if f.is_file())
    return {
        "removed": removed,
        "total_kb": total_bytes // 1024,
        "files": [f.name for f in adapter_dir.iterdir() if f.is_file()],
    }


def save_history_pairs(history: list[dict[str, str]], tokenizer, system_prompt: str) -> int:
    """Save clean (user, assistant) pairs from history to training data."""
    saved_count = 0
    i = 0
    while i < len(history):
        if (
            history[i]["role"] == "user"
            and not history[i]["content"].startswith("[System observation:")
        ):
            if i + 1 < len(history) and history[i + 1]["role"] == "assistant":
                # Build context: up to 3 prior clean pairs
                context = []
                j = i - 1
                pairs = 0
                while j >= 1 and pairs < 3:
                    if (
                        history[j]["role"] == "assistant"
                        and history[j - 1]["role"] == "user"
                        and not history[j - 1]["content"].startswith("[System observation:")
                    ):
                        context.insert(0, {
                            "user": history[j - 1]["content"],
                            "assistant": clean_response(history[j]["content"]),
                        })
                        j -= 2
                        pairs += 1
                    else:
                        j -= 1

                messages = [{"role": "system", "content": system_prompt}]
                for turn in context:
                    messages.append({"role": "user", "content": turn["user"]})
                    messages.append({"role": "assistant", "content": turn["assistant"]})
                messages.append({"role": "user", "content": history[i]["content"]})
                messages.append(
                    {"role": "assistant", "content": clean_response(history[i + 1]["content"])}
                )

                append_training_text(build_chat_training_sample(messages, tokenizer))
                saved_count += 1
                i += 2
                continue
        i += 1
    return saved_count
