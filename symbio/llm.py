"""MLX loading/generation helpers and LoRA training for Symbio."""

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml
from mlx_lm import load

from symbio.config import can_run_lora, detect_model_type
from symbio.constants import (
    ADAPTER_DIR,
    DATA_DIR,
    DEFAULT_CONFIG,
    DIGEST_MANIFEST,
    NOTES_DIR,
    TRAIN_FILE,
    VALID_FILE,
)
from symbio.utils import clean_response


def append_training_text(text: str):
    """Append a single text sample to the training JSONL file."""
    with open(TRAIN_FILE, "a", encoding="utf-8") as f:
        json.dump({"text": text}, f)
        f.write("\n")


def build_chat_training_sample(messages: list[dict[str, str]], tokenizer) -> str:
    """Render a chat list into a training string via the tokenizer's chat template."""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return clean_response(text)


def append_chat_pair(user_msg: str, assistant_msg: str, tokenizer, system_prompt: str):
    """Append a (user, assistant) exchange to the training file."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": clean_response(assistant_msg)},
    ]
    append_training_text(build_chat_training_sample(messages, tokenizer))


def seed_training_data(tokenizer, system_prompt: str, config: dict[str, Any]):
    """Seed clean, balanced corpus for identity, greetings, and tool use."""
    if TRAIN_FILE.exists() and TRAIN_FILE.stat().st_size > 0:
        return

    assistant = config["assistant_name"]
    user = config["user_name"]

    def write(samples: list[tuple[str, str]], path: Path):
        with open(path, "w", encoding="utf-8") as f:
            for user_msg, assistant_msg in samples:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": assistant_msg},
                ]
                f.write(json.dumps({"text": build_chat_training_sample(messages, tokenizer)}) + "\n")

    train_samples = [
        # --- Greetings & chitchat ---
        ("Hi", f"Hi {user}! How can I help you today?"),
        ("Hello", f"Hello {user}! What can I do for you?"),
        ("Hey", f"Hey {user}! What's up?"),
        ("How are you?", f"I'm doing well, thanks for asking, {user}."),
        ("What can you do?", f"I can chat, save memories, read and edit files, search files, run sandboxed commands, execute Python code, search the web, check email, and fine-tune myself on what I learn."),
        # --- Identity: assistant ---
        ("What is your name?", f"My name is {assistant}."),
        ("Who are you?", f"I am {assistant}, your personal AI assistant."),
        ("What's your name?", f"I'm {assistant}."),
        ("Tell me your name.", f"My name is {assistant}."),
        # --- Identity: user ---
        ("What is my name?", f"Your name is {user}."),
        ("Who am I?", f"You are {user}."),
        ("Do you know my name?", f"Yes, your name is {user}."),
        (
            f"My name is {user}.",
            f"Nice to meet you, {user}! I'll remember that. <note title=\"User Identity\">The user's name is {user}.</note>",
        ),
        (
            f"Call yourself {assistant}.",
            f"Got it — my name is {assistant}.",
        ),
        (
            f"Call me {user}.",
            f"Got it — I'll call you {user}.",
        ),
        # --- Combined identity ---
        (
            f"My name is {user}. What is your name?",
            f"Your name is {user}, and my name is {assistant}.",
        ),
        (
            f"My name is {user} and call yourself {assistant}.",
            f"Got it — you are {user}, and I am {assistant}.",
        ),
        (
            "What are our names?",
            f"You are {user} and I am {assistant}.",
        ),
        # --- Memory tool (legacy <note> mapped to memory add) ---
        (
            "Please remember that I prefer concise replies.",
            "Got it. <note title=\"User Preference\">The user prefers concise replies.</note>",
        ),
        (
            "Save your notes and train on them.",
            "<digest /><train />I'll digest the notes and start training so they stick.",
        ),
        # --- File tools ---
        (
            "Read config.json for me.",
            '<tool_call>{"name": "read_file", "arguments": {"path": "config.json"}}</tool_call>Here is the contents of config.json.',
        ),
        (
            "Create a file called todo.txt with 'buy milk'.",
            '<tool_call>{"name": "write_file", "arguments": {"path": "todo.txt", "content": "buy milk"}}</tool_call>Created todo.txt.',
        ),
        (
            "Find all Python files in the project.",
            '<tool_call>{"name": "search_files", "arguments": {"query": "\\.py$", "glob": "*.py"}}</tool_call>Here are the Python files I found.',
        ),
        # --- Terminal ---
        (
            "Check the current date.",
            '<tool_call>{"name": "terminal", "arguments": {"cmd": "date"}}</tool_call>The current date is shown above.',
        ),
        # --- Web ---
        (
            "Search the web for Python 3.14 release date.",
            '<tool_call>{"name": "web_search", "arguments": {"query": "Python 3.14 release date"}}</tool_call>Here are the search results.',
        ),
        # --- Browser ---
        (
            "Open example.com in the browser.",
            '<tool_call>{"name": "browser_open", "arguments": {"url": "https://example.com"}}</tool_call>Opening example.com.',
        ),
        (
            "Scroll down to the next short.",
            '<tool_call>{"name": "browser_scroll", "arguments": {"direction": "down"}}</tool_call>Scrolling down.',
        ),
        # --- Email Q&A ---
        (
            "Can you read my emails?",
            f"I can check your email if you configure IMAP/SMTP via environment variables (EMAIL_ADDRESS, EMAIL_PASSWORD, EMAIL_IMAP_HOST, EMAIL_SMTP_HOST). Once configured, I can list threads, read messages, and send replies.",
        ),
        # --- Domain task ---
        (
            "How do you read an email inbox?",
            f"I can read an inbox if you give me access. <note title=\"Email Reading\">To read email, I need the user's mail credentials or an API key, then I fetch messages, summarize them, and ask before acting.</note>",
        ),
    ]

    valid_samples = [
        ("What is your name?", f"My name is {assistant}."),
        ("What is my name?", f"Your name is {user}."),
        ("Hi", f"Hi {user}! How can I help you today?"),
        (
            f"I'm {user}. Who are you?",
            f"You are {user}, and I am {assistant}.",
        ),
        (
            "Remember that I prefer short answers.",
            '<tool_call>{"name": "note", "arguments": {"action": "add", "target": "user", "content": "The user prefers short answers."}}</tool_call>Noted.',
        ),
    ]

    write(train_samples, TRAIN_FILE)
    write(valid_samples, VALID_FILE)


def run_training(config: dict[str, Any], model_type: str = "dense", iters: int | None = None) -> bool:
    """Run an MLX LoRA fine-tuning pass. Returns True on success."""
    ok, reason = can_run_lora(config, model_type)
    if not ok:
        print(f"  [System] Training skipped: {reason}")
        return False

    if not TRAIN_FILE.exists() or TRAIN_FILE.stat().st_size == 0:
        print("  [System] No training data available.")
        return False

    lora = config["lora"]
    train_iters = iters if iters is not None else lora["iters"]
    print("\n  [System] Starting MLX LoRA Fine-Tuning\n")

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
        "--iters", str(train_iters),
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
    """Remove stale adapter checkpoint files."""
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


def digest_notes_to_training(tokenizer, system_prompt: str, planner: Any | None = None) -> int:
    """Convert all markdown notes into training samples and update the digest manifest."""
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

        messages_doc = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Write a markdown note titled '{title}'."},
            {"role": "assistant", "content": body},
        ]
        sample_doc = build_chat_training_sample(messages_doc, tokenizer)
        append_training_text(sample_doc)
        if planner is not None:
            planner.add_sample(sample_doc, source=f"digest:{f.name}")

        messages_qa = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"According to your notes, what do you know about '{topic}'?"},
            {"role": "assistant", "content": body},
        ]
        sample_qa = build_chat_training_sample(messages_qa, tokenizer)
        append_training_text(sample_qa)
        if planner is not None:
            planner.add_sample(sample_qa, source=f"digest_qa:{f.name}")

        added += 2

    DIGEST_MANIFEST.write_text(json.dumps(new_manifest, indent=2))
    return added


def load_model_with_adapter(config: dict[str, Any], adapter_path: str | Path | bool | None = None):
    """Load the configured model, optionally with a LoRA adapter.

    Pass adapter_path=False to force the base model. Returns
    (model, tokenizer, adapter_loaded).
    """
    path = str(adapter_path) if adapter_path else str(ADAPTER_DIR)
    if adapter_path is False or not (Path(path) / "adapter_config.json").exists():
        model, tokenizer = load(config["model_name"])
        return model, tokenizer, False
    model, tokenizer = load(config["model_name"], adapter_path=path)
    return model, tokenizer, True


def save_history_pairs(
    history: list[dict[str, str]],
    tokenizer,
    system_prompt: str,
    planner: Any | None = None,
) -> int:
    """Save recent (user, assistant) pairs from history as training samples."""
    saved_count = 0
    i = 0
    while i < len(history):
        if history[i]["role"] == "user" and not history[i]["content"].startswith("[System observation"):
            if i + 1 < len(history) and history[i + 1]["role"] == "assistant":
                context = []
                j = i - 1
                pairs = 0
                while j >= 1 and pairs < 3:
                    if (
                        history[j]["role"] == "assistant"
                        and history[j - 1]["role"] == "user"
                        and not history[j - 1]["content"].startswith("[System observation")
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
                messages.append({"role": "assistant", "content": clean_response(history[i + 1]["content"])})

                sample = build_chat_training_sample(messages, tokenizer)
                append_training_text(sample)
                if planner is not None:
                    planner.add_sample(sample, source="history_pair")
                saved_count += 1
                i += 2
                continue
        i += 1
    return saved_count
