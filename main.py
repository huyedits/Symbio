#!/usr/bin/env python3
"""Caine: a personal, autonomous, self-fine-tuning Hermes-style agent.
and conversation via LoRA. Per-user identity is configurable in config.json
or via CLI flags.
"""

import argparse
import hashlib
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

# ======================== PATHS =========================
PROJECT_DIR = Path(__file__).parent.resolve()
LOG_DIR = PROJECT_DIR / "logs"
DATA_DIR = PROJECT_DIR / "training_data"
TRAIN_FILE = DATA_DIR / "train.jsonl"
ADAPTER_DIR = PROJECT_DIR / "adapters"
NOTES_DIR = PROJECT_DIR / "notes"
SANDBOX_DIR = PROJECT_DIR / "sandbox"
DIGEST_MANIFEST = DATA_DIR / "digest_manifest.json"
CONFIG_FILE = PROJECT_DIR / "config.json"
CRON_FILE = PROJECT_DIR / "cron_jobs.json"
prompt = PROJECT_DIR / "prompt.md"
for d in (LOG_DIR, DATA_DIR, ADAPTER_DIR, NOTES_DIR, SANDBOX_DIR):
    d.mkdir(parents=True, exist_ok=True)

DEFAULT_CONFIG: dict[str, Any] = {
    "model_name": "Qwen/Qwen3-0.6B",
    "assistant_name": "Caine",
    "user_name": "Huy",
    "lora": {
        "rank": 8,
        "dropout": 0.0,
        "scale": 20.0,
        "num_layers": 8,
        "batch_size": 1,
        "learning_rate": 1e-4,
        "iters": 300,
        "max_seq_length": 512,
        "steps_per_eval": 100,
        "save_every": 100,
    },
    "agent": {
        "max_tool_rounds": 5,
        "history_limit": 40,
        "sandbox_timeout": 30,
        "max_output_len": 4000,
        "temperature": 0.7,
        "top_p": 0.9,
    },
}
# ========================================================


def load_config() -> dict[str, Any]:
    """Load config.json if present; merge with sensible defaults."""
    config = DEFAULT_CONFIG.copy()
    if CONFIG_FILE.exists():
        try:
            user_config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            config.update(user_config)
            if "lora" in user_config:
                config["lora"] = {**DEFAULT_CONFIG["lora"], **user_config["lora"]}
            if "agent" in user_config:
                config["agent"] = {**DEFAULT_CONFIG["agent"], **user_config["agent"]}
        except Exception as e:
            print(f"[Config warning] Could not read {CONFIG_FILE}: {e}")
    return config


# Seeded into prompt.md on first run; edit that file to customize the prompt.
DEFAULT_SYSTEM_PROMPT = """You are {assistant_name}, a helpful personal AI assistant with persistent memory.
Your user is named {user_name}.

You can take actions by using special tags in your response:
  <note title='T'>body</note> — save a markdown note
  <cmd>command</cmd> — run a sandboxed shell command
  <digest /> — convert unsaved notes to training data
  <train /> — fine-tune your LoRA weights on accumulated knowledge
  <cron expr='MIN HOUR DOM MON DOW'>text</cron> — schedule a recurring reminder (5-field cron)
  <cron at='YYYY-MM-DD HH:MM'>text</cron> — schedule a one-time reminder

Guidelines:
- Write a note whenever {user_name} teaches you something important.
- After writing 2+ new notes, call <digest /> then <train /> to remember them.
- If {user_name} asks you to check the system, use <cmd>.
- The current date/time from the computer clock is shown with every request; use it when scheduling. If {user_name} states a different time or timezone, trust what they say.
- Convert relative times ("in 10 minutes", "tomorrow at 9am") into absolute times using the current clock before scheduling.
- Start a scheduled reminder's text with "cmd:" to run a sandboxed command when it fires.
- Talk normally outside the tags.
- NEVER include internal reasoning, thinking, or analysis in your final reply.
- Address {user_name} by name when it feels natural.
- Keep replies concise unless asked for detail.
"""


def time_note(now: datetime | None = None) -> str:
    """Appended to the system prompt each round so the model can align
    schedules with the computer clock (or defer to a time the user states)."""
    now = now or datetime.now()
    return f"\n\n[Current local date/time from computer clock: {now:%A, %Y-%m-%d %H:%M}]"


def build_system_prompt(assistant_name: str, user_name: str) -> str:
    if not prompt.exists():
        prompt.write_text(DEFAULT_SYSTEM_PROMPT, encoding="utf-8")
    return prompt.read_text(encoding="utf-8").format(
        assistant_name=assistant_name, user_name=user_name
    )


# --- Logger ---
chat_logger = logging.getLogger("chat")
chat_logger.setLevel(logging.INFO)
chat_logger.propagate = False
log_path = LOG_DIR / f"chat_{datetime.now():%Y-%m-%d_%H-%M-%S}.log"
fh = logging.FileHandler(log_path)
fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
chat_logger.addHandler(fh)
# ========================================================


# Common thinking/reasoning delimiters that must never reach the user or training data.
_THINKING_PATTERNS = [
    r"<thinking\b[^>]*>.*?</thinking>",
    r"</?thinking\b[^>]*>.*?</?thinking>",
    r"<analysis\b[^>]*>.*?</analysis>",
    r"<reasoning\b[^>]*>.*?</reasoning>",
    r"<think\b[^>]*>.*?</think>",
    r" thinking\s+.*?/thinking",
    r"\bthinking\s*:?\s*\n.*?\n/?thinking",
    r"\breasoning\s*:?\s*\n.*?\n/?reasoning",
]


def clean_response(text: str) -> str:
    for pattern in _THINKING_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"^Assistant:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^user:\s*", "", text, flags=re.IGNORECASE)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def print_banner(config: dict[str, Any], adapter_loaded: bool, dataset_size: int):
    note_count = len(list(NOTES_DIR.glob("*.md")))
    print("\n" + "=" * 50)
    print(f"  {config['assistant_name'].upper()} — PERSONAL CHAT-FINETUNE CLI")
    print(f"   Model  : {config['model_name']}")
    print(f"   User   : {config['user_name']}")
    print(f"   LoRA   : {'YES' if adapter_loaded else 'None (base)'}")
    print(f"   Data   : {dataset_size:,} bytes")
    print(f"   Notes  : {note_count}")
    print("-" * 50)
    print("Commands: /quit  /save  /train  /forget_last  /status  /prune")
    print("         /run <cmd>  /note [title]  /notes  /digest  /cron")
    print("  (Caine can also use <note>, <cmd>, <digest />, <train />, <cron> by itself)")
    print("-" * 50)


def append_training_text(text: str):
    with open(TRAIN_FILE, "a", encoding="utf-8") as f:
        json.dump({"text": text}, f)
        f.write("\n")


def build_chat_training_sample(messages: list[dict[str, str]], tokenizer) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )


def append_chat_pair(user_msg: str, assistant_msg: str, tokenizer, system_prompt: str):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": clean_response(assistant_msg)},
    ]
    append_training_text(build_chat_training_sample(messages, tokenizer))


def digest_notes_to_training(tokenizer, system_prompt: str) -> int:
    files = sorted(NOTES_DIR.glob("*.md"))
    if not files:
        return 0

    manifest: dict[str, str] = {}
    if DIGEST_MANIFEST.exists():
        try:
            manifest = json.loads(DIGEST_MANIFEST.read_text())
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

    DIGEST_MANIFEST.write_text(json.dumps(new_manifest, indent=2))
    return added


def run_sandboxed(command: str, config: dict[str, Any]):
    command = command.strip()
    if not command:
        return False, "Empty command."
    try:
        args = shlex.split(command)
    except ValueError as e:
        return False, f"Parse error: {e}"
    if not args:
        return False, "Empty command."

    blocked = {
        "rm", "sudo", "su", "dd", "mkfs", "fdisk", "mount", "umount",
        "chmod", "chown", "curl", "wget", "ssh", "scp", "bash", "sh", "zsh",
        "fish", "python", "python3", "perl", "ruby", "php", "node", "npm",
    }
    if args[0] in blocked:
        return False, f"'{args[0]}' is blocked in sandbox."

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=config["agent"]["sandbox_timeout"],
            cwd=SANDBOX_DIR,
            shell=False,
        )
        out = result.stdout
        if result.stderr:
            out += "\n" + result.stderr
        out = out.strip()
        max_len = config["agent"]["max_output_len"]
        if len(out) > max_len:
            out = out[:max_len] + "\n... (truncated)"
        return result.returncode == 0, out
    except subprocess.TimeoutExpired:
        return False, f"Timed out after {config['agent']['sandbox_timeout']}s."
    except FileNotFoundError:
        return False, f"Command not found: {args[0]}"
    except Exception as e:
        return False, str(e)


def save_note(title: str, body: str) -> Path:
    safe = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in title)
    safe = safe.strip().replace(" ", "_")[:40]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = NOTES_DIR / f"{ts}_{safe}.md"
    path.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")
    return path


def ensure_seed_notes(config: dict[str, Any]):
    """If notes/ is empty, seed the two identity facts as markdown notes."""
    if any(NOTES_DIR.glob("*.md")):
        return
    save_note("My Identity", f"I am {config['assistant_name']}, a helpful personal AI assistant.")
    save_note("User Identity", f"My user's name is {config['user_name']}.")


def seed_training_data(tokenizer, system_prompt: str, config: dict[str, Any]):
    """Seed a minimal clean corpus so the model has correct identity/tool examples
    even before any real conversation is saved."""
    if TRAIN_FILE.exists() and TRAIN_FILE.stat().st_size > 0:
        return

    assistant = config["assistant_name"]
    user = config["user_name"]

    samples = [
        # Identity
        (
            f"What is your name?",
            f"I am {assistant}, your personal AI assistant.",
        ),
        (
            f"My name is {user}.",
            f"Nice to meet you, {user}! I'll remember that. <note title='User Identity'>{user} is my user's name.</note>",
        ),
        (
            f"What is my name?",
            f"Your name is {user}, {user}.",
        ),
        # Tool-use demonstration
        (
            "Please remember that I prefer concise replies.",
            "Got it. <note title='User Preference'>The user prefers concise replies.</note>",
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

    for user_msg, assistant_msg in samples:
        append_chat_pair(user_msg, assistant_msg, tokenizer, system_prompt)


# ======================== CRON JOBS =========================
def load_cron_jobs() -> list[dict[str, Any]]:
    if not CRON_FILE.exists():
        return []
    try:
        return json.loads(CRON_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_cron_jobs(jobs: list[dict[str, Any]]):
    CRON_FILE.write_text(json.dumps(jobs, indent=2), encoding="utf-8")


def _cron_field_matches(field: str, value: int, lo: int, hi: int) -> bool:
    for part in field.split(","):
        part = part.strip()
        step = 1
        if "/" in part:
            part, step_str = part.split("/", 1)
            step = int(step_str)
        if part == "*":
            start, end = lo, hi
        elif "-" in part:
            a, b = part.split("-", 1)
            start, end = int(a), int(b)
        else:
            start = end = int(part)
        if start <= value <= end and (value - start) % step == 0:
            return True
    return False


def cron_matches(expr: str, when: datetime) -> bool:
    """Match a 5-field cron expression (minute hour day month weekday,
    weekday 0/7 = Sunday) against a datetime. Raises ValueError on bad fields."""
    fields = expr.split()
    if len(fields) != 5:
        return False
    minute, hour, dom, month, dow = fields
    dow_val = (when.weekday() + 1) % 7
    return (
        _cron_field_matches(minute, when.minute, 0, 59)
        and _cron_field_matches(hour, when.hour, 0, 23)
        and _cron_field_matches(dom, when.day, 1, 31)
        and _cron_field_matches(month, when.month, 1, 12)
        and (
            _cron_field_matches(dow, dow_val, 0, 7)
            or (dow_val == 0 and _cron_field_matches(dow, 7, 0, 7))
        )
    )


def validate_cron_expr(expr: str) -> str | None:
    """Return an error message if expr is not a valid cron expression."""
    if len(expr.split()) != 5:
        return "Schedule must be 'at YYYY-MM-DD HH:MM' or 5 cron fields: minute hour day month weekday."
    try:
        cron_matches(expr, datetime.now())
    except ValueError as e:
        return f"Bad cron expression '{expr}': {e}"
    return None


def parse_one_shot(schedule: str) -> datetime | None:
    """Parse a one-time schedule ('at 2026-07-16 21:30', '21:30', ...).
    A bare time means the next occurrence of that time."""
    s = schedule.strip()
    while s.lower().startswith("at "):
        s = s[3:].strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    try:
        t = datetime.strptime(s, "%H:%M")
    except ValueError:
        return None
    now = datetime.now()
    target = now.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target


def add_cron_job(schedule: str, text: str) -> dict[str, Any]:
    schedule = schedule.strip()
    text = text.strip()
    if not text:
        raise ValueError("Job text is empty.")
    one_shot = parse_one_shot(schedule)
    if one_shot:
        # Normalize to an absolute time so it fires exactly once.
        schedule = f"at {one_shot:%Y-%m-%d %H:%M}"
    else:
        error = validate_cron_expr(schedule)
        if error:
            raise ValueError(error)
    jobs = load_cron_jobs()
    job = {
        "id": max((j.get("id", 0) for j in jobs), default=0) + 1,
        "schedule": schedule,
        "text": text,
        "last_fired": None,
    }
    jobs.append(job)
    save_cron_jobs(jobs)
    return job


def check_due_jobs(config: dict[str, Any], now: datetime | None = None) -> list[str]:
    """Fire all due jobs and return their event messages. One-shot jobs are
    removed after firing; recurring jobs fire at most once per minute."""
    now = now or datetime.now()
    minute_key = now.strftime("%Y-%m-%d %H:%M")
    jobs = load_cron_jobs()
    events: list[str] = []
    remaining: list[dict[str, Any]] = []
    changed = False

    for job in jobs:
        schedule = job.get("schedule", "")
        fire = drop = False
        if schedule.startswith("at "):
            try:
                target = datetime.strptime(schedule[3:], "%Y-%m-%d %H:%M")
                fire = drop = target <= now
            except ValueError:
                events.append(f"Removed job {job.get('id')}: invalid schedule '{schedule}'.")
                drop = True
        else:
            try:
                fire = cron_matches(schedule, now) and job.get("last_fired") != minute_key
            except ValueError:
                events.append(f"Removed job {job.get('id')}: invalid schedule '{schedule}'.")
                drop = True

        if fire:
            job["last_fired"] = minute_key
            text = job.get("text", "")
            if text.startswith("cmd:"):
                shell_cmd = text[4:].strip()
                ok, out = run_sandboxed(shell_cmd, config)
                events.append(
                    f"Scheduled job {job.get('id')} ran '{shell_cmd}' "
                    f"({'ok' if ok else 'error'}):\n{out}"
                )
            else:
                events.append(f"Scheduled reminder: {text}")

        if fire or drop:
            changed = True
        if not drop:
            remaining.append(job)

    if changed:
        save_cron_jobs(remaining)
    return events
# ========================================================


def run_training(config: dict[str, Any]) -> bool:
    if not TRAIN_FILE.exists() or TRAIN_FILE.stat().st_size == 0:
        print("  [System] No training data available.")
        return False

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
    config_fd, config_path = tempfile.mkstemp(suffix=".yaml", dir=str(DATA_DIR))
    with os.fdopen(config_fd, "w") as f:
        yaml.dump(lora_config, f)

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", config["model_name"],
        "--train",
        "--data", str(DATA_DIR),
        "--batch-size", str(lora["batch_size"]),
        "--num-layers", str(lora["num_layers"]),
        "--iters", str(lora["iters"]),
        "--learning-rate", str(lora["learning_rate"]),
        "--steps-per-eval", str(lora["steps_per_eval"]),
        "--max-seq-length", str(lora["max_seq_length"]),
        "--adapter-path", str(ADAPTER_DIR),
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

    config_file = ADAPTER_DIR / "adapter_config.json"
    weight_files = list(ADAPTER_DIR.glob("adapters.*"))
    if not config_file.exists() or not weight_files:
        print("  [System] Adapter files missing after training.")
        return False

    adapter_kb = sum(f.stat().st_size for f in ADAPTER_DIR.iterdir() if f.is_file()) // 1024
    print(f"  [System] Adapter baked. Size: ~{adapter_kb:,} KB")
    return True


def prune_adapters() -> dict[str, Any]:
    """Remove intermediate checkpoints and report adapter footprint."""
    removed = []
    for cp in ADAPTER_DIR.glob("[0-9]*_adapters.*"):
        cp.unlink()
        removed.append(cp.name)

    total_bytes = sum(f.stat().st_size for f in ADAPTER_DIR.iterdir() if f.is_file())
    return {
        "removed": removed,
        "total_kb": total_bytes // 1024,
        "files": [f.name for f in ADAPTER_DIR.iterdir() if f.is_file()],
    }


def parse_tools(reply: str) -> list[tuple[str, dict[str, Any]]]:
    """Extract tool calls from the model reply."""
    tools: list[tuple[str, dict[str, Any]]] = []

    for m in re.finditer(
        r'<note\s+title=[\'"]([^\'"]*?)[\'"]>(.*?)</note>', reply, re.DOTALL
    ):
        tools.append(("write_note", {
            "title": m.group(1).strip(),
            "body": m.group(2).strip(),
        }))

    for m in re.finditer(r'<cmd>(.*?)</cmd>', reply, re.DOTALL):
        tools.append(("run_command", {"cmd": m.group(1).strip()}))

    if re.search(r'<digest\s*/>', reply) or re.search(r'<digest></digest>', reply):
        tools.append(("digest_notes", {}))

    if re.search(r'<train\s*/>', reply) or re.search(r'<train></train>', reply):
        tools.append(("train_adapter", {}))

    for m in re.finditer(r'<cron\s+expr=[\'"]([^\'"]*?)[\'"]>(.*?)</cron>', reply, re.DOTALL):
        tools.append(("schedule_job", {
            "schedule": m.group(1).strip(),
            "text": m.group(2).strip(),
        }))

    for m in re.finditer(r'<cron\s+at=[\'"]([^\'"]*?)[\'"]>(.*?)</cron>', reply, re.DOTALL):
        tools.append(("schedule_job", {
            "schedule": "at " + m.group(1).strip(),
            "text": m.group(2).strip(),
        }))

    return tools


def strip_tool_tags(reply: str) -> str:
    display = reply
    display = re.sub(r'<note\s+title=[\'"][^\'"]*?[\'"]>(.*?)</note>', '', display, flags=re.DOTALL)
    display = re.sub(r'<cmd>(.*?)</cmd>', '', display, flags=re.DOTALL)
    display = re.sub(r'<digest\s*/>', '', display)
    display = re.sub(r'<digest></digest>', '', display)
    display = re.sub(r'<train\s*/>', '', display)
    display = re.sub(r'<train></train>', '', display)
    display = re.sub(r'<cron\s+[^>]*?>(.*?)</cron>', '', display, flags=re.DOTALL)
    return clean_response(display)


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


def chat_loop(config: dict[str, Any]):
    print(" Loading model...")
    sampler = make_sampler(
        temp=config["agent"]["temperature"],
        top_p=config["agent"]["top_p"],
    )
    system_prompt = build_system_prompt(config["assistant_name"], config["user_name"])
    adapter_config = ADAPTER_DIR / "adapter_config.json"

    if adapter_config.exists():
        print(" Found existing adapter. Loading it...")
        try:
            model, tokenizer = load(config["model_name"], adapter_path=str(ADAPTER_DIR))
        except Exception as e:
            print(f" Could not load adapter: {e}")
            print(" Falling back to base model...")
            model, tokenizer = load(config["model_name"])
    else:
        model, tokenizer = load(config["model_name"])

    # Seed identity notes + clean training corpus on first run.
    ensure_seed_notes(config)
    seed_training_data(tokenizer, system_prompt, config)

    dataset_size = TRAIN_FILE.stat().st_size if TRAIN_FILE.exists() else 0
    print_banner(config, adapter_config.exists(), dataset_size)

    history: list[dict[str, str]] = []

    # Background scheduler: fires due cron jobs, prints a notice immediately,
    # and queues the event for the model to see on the next turn.
    cron_events: list[str] = []
    cron_lock = threading.Lock()

    def _cron_worker():
        while True:
            time.sleep(20)
            try:
                fired = check_due_jobs(config)
            except Exception:
                continue
            if fired:
                with cron_lock:
                    cron_events.extend(fired)
                for ev in fired:
                    print(f"\n  [Cron] {ev.splitlines()[0]}")

    threading.Thread(target=_cron_worker, daemon=True).start()

    while True:
        try:
            user_input = input(f"{config['user_name']:8}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            user_input = "/quit"

        # ---- Manual Slash Commands (overrides) ----
        if user_input.startswith("/"):
            cmd = user_input.lower()

            if cmd in ("/quit", "/q", "/exit"):
                print(" Exiting chat.")
                break

            elif cmd == "/forget_last":
                removed = 0
                while history and history[-1]["role"] == "assistant":
                    history.pop()
                    removed += 1
                while (
                    history
                    and history[-1]["role"] == "user"
                    and not history[-1]["content"].startswith("[System observation:")
                ):
                    history.pop()
                    removed += 1
                print(f"  Forgot last exchange." if removed else " Nothing to forget.")
                continue

            elif cmd == "/save":
                if not history:
                    print(" Nothing to save yet.")
                else:
                    saved_count = save_history_pairs(history, tokenizer, system_prompt)
                    print(f" Saved {saved_count} exchange(s) to training data.")
                continue

            elif cmd == "/train":
                trained = run_training(config)
                if trained and adapter_config.exists():
                    print("\n Reloading model with updated adapter...")
                    try:
                        model, tokenizer = load(config["model_name"], adapter_path=str(ADAPTER_DIR))
                        print("  Adapter reloaded.")
                    except Exception as e:
                        print(f" Could not reload adapter: {e}")
                continue

            elif cmd == "/digest":
                added = digest_notes_to_training(tokenizer, system_prompt)
                if added:
                    print(f"  Digested {added} new note samples into training data.")
                else:
                    print("  No new or changed notes to digest.")
                continue

            elif cmd.startswith("/run"):
                shell_cmd = user_input[4:].strip()
                if not shell_cmd:
                    print("  Usage: /run <command>")
                    continue
                print(f"\n  $ {shell_cmd}")
                ok, output = run_sandboxed(shell_cmd, config)
                status = "ok" if ok else "err"
                print(f"  [{status}]")
                for line in output.splitlines():
                    print(f"  {line}")
                append_chat_pair(
                    user_msg=f"Run this sandbox command and show the output:\n{shell_cmd}",
                    assistant_msg=output,
                    tokenizer=tokenizer,
                    system_prompt=system_prompt,
                )
                print("  -> Logged to training data.\n")
                continue

            elif cmd.startswith("/note"):
                title = user_input[5:].strip()
                if not title:
                    title = input("  Note title: ").strip()
                if not title:
                    print("  Cancelled.")
                    continue
                body = ""
                print("  Content (empty line to finish):")
                try:
                    while True:
                        line = input()
                        if line == "":
                            break
                        body += line + "\n"
                except (EOFError, KeyboardInterrupt):
                    pass
                if not body.strip():
                    print("  Empty note, cancelled.")
                    continue
                path = save_note(title, body.strip())
                print(f"  Saved: {path.name}")
                continue

            elif cmd == "/notes":
                files = sorted(NOTES_DIR.glob("*.md"))
                if not files:
                    print("  No notes yet.")
                else:
                    print(f"  {len(files)} note(s):")
                    for f in files:
                        print(f"    - {f.name}")
                continue

            elif cmd == "/status":
                files = sorted(NOTES_DIR.glob("*.md"))
                data_size = TRAIN_FILE.stat().st_size if TRAIN_FILE.exists() else 0
                adapter_files = list(ADAPTER_DIR.glob("adapters.*"))
                adapter_kb = sum(f.stat().st_size for f in ADAPTER_DIR.iterdir() if f.is_file()) // 1024
                print(f"  Model: {config['model_name']}")
                print(f"  Assistant: {config['assistant_name']} | User: {config['user_name']}")
                print(f"  Notes: {len(files)}")
                print(f"  Training data: {data_size:,} bytes")
                print(f"  Adapter loaded: {'YES' if adapter_config.exists() else 'NO'}")
                print(f"  Adapter files: {len(adapter_files)} ({adapter_kb:,} KB)")
                continue

            elif cmd.startswith("/cron"):
                try:
                    parts = shlex.split(user_input)[1:]
                except ValueError as e:
                    print(f"  Parse error: {e}")
                    continue
                sub = parts[0].lower() if parts else "list"
                if sub == "list":
                    jobs = load_cron_jobs()
                    if not jobs:
                        print("  No scheduled jobs.")
                    for j in jobs:
                        print(f"  [{j['id']}] {j['schedule']} — {j['text']}")
                elif sub == "add" and len(parts) >= 3:
                    try:
                        job = add_cron_job(parts[1], " ".join(parts[2:]))
                        print(f"  Added job {job['id']}: {job['schedule']} — {job['text']}")
                    except ValueError as e:
                        print(f"  {e}")
                elif sub == "rm" and len(parts) == 2:
                    jobs = load_cron_jobs()
                    kept = [j for j in jobs if str(j["id"]) != parts[1]]
                    save_cron_jobs(kept)
                    print(f"  Removed job {parts[1]}." if len(kept) < len(jobs)
                          else f"  No job with id {parts[1]}.")
                else:
                    print('  Usage: /cron [list] | /cron add "<cron expr | at YYYY-MM-DD HH:MM>" <text> | /cron rm <id>')
                continue

            elif cmd == "/prune":
                info = prune_adapters()
                if info["removed"]:
                    print(f"  Removed {len(info['removed'])} stale checkpoint(s):")
                    for name in info["removed"]:
                        print(f"    - {name}")
                else:
                    print("  No stale checkpoints to remove.")
                print(f"  Current adapter footprint: {info['total_kb']:,} KB")
                print("  Note: mlx_lm LoRA adapters do not support true weight pruning; keeping rank low and removing checkpoints is the practical way to stay small.")
                continue

            else:
                print("  Unknown command.")
                continue

        if not user_input:
            continue

        chat_logger.info(f"User: {user_input}")

        # Surface any cron events that fired since the last turn.
        with cron_lock:
            due_events, cron_events[:] = list(cron_events), []
        if due_events:
            history.append({
                "role": "user",
                "content": "[System observation: " + "\n".join(due_events) + "]",
            })

        history.append({"role": "user", "content": user_input})

        # ---- Autonomous Agent Loop ----
        max_rounds = config["agent"]["max_tool_rounds"]
        for round_num in range(max_rounds):
            messages = [{"role": "system", "content": system_prompt + time_note()}]
            messages.extend(history[-config["agent"]["history_limit"]:])

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            try:
                raw_reply = generate(model, tokenizer, prompt=prompt, sampler=sampler, verbose=False)
                reply = raw_reply.strip()
            except Exception as e:
                print(f"[MLX Error: {e}]")
                break

            tools = parse_tools(reply)
            display = strip_tool_tags(reply)

            if display.strip():
                print(f"{config['assistant_name']:8}: {display}")
                chat_logger.info(f"{config['assistant_name']}: {display}")

            if not tools:
                # Normal turn: store assistant reply and wait for next user input
                history.append({"role": "assistant", "content": reply})
                while len(history) > config["agent"]["history_limit"] + 8:
                    history.pop(0)
                break

            # There are tools to execute
            history.append({"role": "assistant", "content": reply})

            observations = []
            for name, params in tools:
                print(f"  [Tool: {name}]")

                if name == "write_note":
                    try:
                        p = save_note(params["title"], params["body"])
                        observations.append(f"Saved note: {p.name}")
                    except Exception as e:
                        observations.append(f"Failed to save note: {e}")

                elif name == "run_command":
                    ok, out = run_sandboxed(params["cmd"], config)
                    observations.append(
                        f"Command '{params['cmd']}' exited {'ok' if ok else 'error'}.\nOutput:\n{out}"
                    )

                elif name == "digest_notes":
                    try:
                        cnt = digest_notes_to_training(tokenizer, system_prompt)
                        observations.append(f"Digested {cnt} new training samples from notes.")
                    except Exception as e:
                        observations.append(f"Digest error: {e}")

                elif name == "schedule_job":
                    try:
                        job = add_cron_job(params["schedule"], params["text"])
                        observations.append(
                            f"Scheduled job {job['id']}: {job['schedule']} — {job['text']}"
                        )
                    except ValueError as e:
                        observations.append(f"Could not schedule job: {e}")

                elif name == "train_adapter":
                    trained = run_training(config)
                    if trained and adapter_config.exists():
                        try:
                            model, tokenizer = load(config["model_name"], adapter_path=str(ADAPTER_DIR))
                            observations.append("Training complete. Adapter reloaded.")
                        except Exception as e:
                            observations.append(f"Training done but reload failed: {e}")
                    else:
                        observations.append("Training skipped (no new data or failed).")

            # Feed observations back as a system/user turn
            obs_text = "\n".join(observations)
            print(f"  [Observation] {obs_text.replace(chr(10), chr(10) + '  ')}")
            history.append({"role": "user", "content": f"[System observation: {obs_text}]"})

            while len(history) > config["agent"]["history_limit"] + 8:
                history.pop(0)

        # End agent loop

    # ---- End of Session ----
    if history:
        save = input("\n Save conversation for training? [y/N]: ").strip().lower()
        if save in ("y", "yes"):
            saved_count = save_history_pairs(history, tokenizer, system_prompt)
            print(f"    Appended {saved_count} exchange(s) to {TRAIN_FILE}")

            if input("  Train now? [y/N]: ").strip().lower() in ("y", "yes"):
                trained = run_training(config)
                if trained and adapter_config.exists():
                    print("\n Reloading model...")
                    try:
                        model, tokenizer = load(config["model_name"], adapter_path=str(ADAPTER_DIR))
                    except Exception as e:
                        print(f" Could not reload: {e}")


def main():
    config = load_config()

    parser = argparse.ArgumentParser(description="Caine: Autonomous Chat + LoRA")
    parser.add_argument("--train", action="store_true", help="Run training and exit")
    parser.add_argument("--model", type=str, default=config["model_name"], help="Base model")
    parser.add_argument("--assistant-name", type=str, default=config["assistant_name"], help="Assistant name")
    parser.add_argument("--user-name", type=str, default=config["user_name"], help="User name")
    args = parser.parse_args()

    config["model_name"] = args.model
    config["assistant_name"] = args.assistant_name
    config["user_name"] = args.user_name

    if args.train:
        run_training(config)
    else:
        chat_loop(config)



if __name__ == "__main__":
    main()
