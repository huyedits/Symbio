"""Training-data accumulation, note/memory digestion, and LoRA fine-tuning."""

import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from typing import Any

import yaml

from symbio import constants
from symbio.app import config as app_config
from symbio.app.tooling import clean_response


def append_training_text(text: str):
    with open(constants.TRAIN_FILE, "a", encoding="utf-8") as f:
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
        # Web/YouTube search = open a search URL in the browser
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
    ]

    for user_msg, assistant_msg in samples:
        append_chat_pair(user_msg, assistant_msg, tokenizer, system_prompt)


def ensure_validation_split(every_nth: int = 10, max_samples: int = 24):
    """mlx_lm silently skips evaluation when valid.jsonl is missing, which
    makes steps_per_eval meaningless. Sample a small validation set from the
    training data so eval loss is always reported."""
    if constants.VALID_FILE.exists() and constants.VALID_FILE.stat().st_size > 0:
        return
    lines = [l for l in constants.TRAIN_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    sample = lines[::every_nth][:max_samples] or lines[:1]
    constants.VALID_FILE.write_text("\n".join(sample) + "\n", encoding="utf-8")


def run_training(config: dict[str, Any], iters: int | None = None) -> bool:
    """Run a LoRA fine-tune. `iters` overrides lora.iters for short passes
    (e.g. the correction-learning batches)."""
    if not constants.TRAIN_FILE.exists() or constants.TRAIN_FILE.stat().st_size == 0:
        print("  [System] No training data available.")
        return False
    ensure_validation_split()

    # Sweep temp LoRA config files left behind by previous crashed runs.
    for stale in constants.DATA_DIR.glob("tmp*.yaml"):
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
    config_fd, config_path = tempfile.mkstemp(suffix=".yaml", dir=str(constants.DATA_DIR))
    with os.fdopen(config_fd, "w") as f:
        yaml.dump(lora_config, f)

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", config["model_name"],
        "--train",
        "--data", str(constants.DATA_DIR),
        "--batch-size", str(lora["batch_size"]),
        "--num-layers", str(lora["num_layers"]),
        "--iters", str(iters if iters is not None else lora["iters"]),
        "--learning-rate", str(lora["learning_rate"]),
        "--steps-per-eval", str(lora["steps_per_eval"]),
        "--max-seq-length", str(lora["max_seq_length"]),
        "--adapter-path", str(constants.ADAPTER_DIR),
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

    config_file = constants.ADAPTER_DIR / "adapter_config.json"
    weight_files = list(constants.ADAPTER_DIR.glob("adapters.*"))
    if not config_file.exists() or not weight_files:
        print("  [System] Adapter files missing after training.")
        return False

    adapter_kb = sum(f.stat().st_size for f in constants.ADAPTER_DIR.iterdir() if f.is_file()) // 1024
    print(f"  [System] Adapter baked. Size: ~{adapter_kb:,} KB")
    return True


def prune_adapters() -> dict[str, Any]:
    """Remove intermediate checkpoints and report adapter footprint."""
    removed = []
    for cp in constants.ADAPTER_DIR.glob("[0-9]*_adapters.*"):
        cp.unlink()
        removed.append(cp.name)

    total_bytes = sum(f.stat().st_size for f in constants.ADAPTER_DIR.iterdir() if f.is_file())
    return {
        "removed": removed,
        "total_kb": total_bytes // 1024,
        "files": [f.name for f in constants.ADAPTER_DIR.iterdir() if f.is_file()],
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
