"""System prompt, banner, response cleaning, and CLI chat loop for Symbio."""

import logging
import sys
from datetime import datetime
from typing import Any

from mlx_lm import load

from symbio.config import (
    _adapter_matches_model,
    detect_model_type,
    list_model_presets,
    load_config,
    maybe_update_names_from_message,
    save_config,
    setup_names,
    switch_model_preset,
)
from symbio.constants import ADAPTER_DIR, DEFAULT_CONFIG, LOG_DIR, NOTES_DIR, SCREENSHOTS_DIR, TRAIN_FILE
from symbio.learn import _looks_like_correction, learn_from_last_correction, maybe_train_on_mistakes
from symbio.llm import (
    append_chat_pair,
    digest_notes_to_training,
    prune_adapters,
    run_training,
    save_history_pairs,
    seed_training_data,
)
from symbio.utils import clean_response, ensure_seed_notes, save_note, strip_generation_artifacts

# --- Logger ---
chat_logger = logging.getLogger("chat")
chat_logger.setLevel(logging.INFO)
chat_logger.propagate = False
log_path = LOG_DIR / f"chat_{datetime.now():%Y-%m-%d_%H-%M-%S}.log"
fh = logging.FileHandler(log_path)
fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
chat_logger.addHandler(fh)


def build_system_prompt(assistant_name: str, user_name: str, _tools: list[dict[str, Any]]) -> str:
    """Build the Hermes-style system prompt used by the agent."""
    return (
        f"You are {assistant_name}, a helpful personal AI assistant. "
        f"The assistant's name is {assistant_name}. The user's name is {user_name}. "
        "Never confuse or swap these two names.\n\n"
        "You can take actions by emitting tags in your response. "
        "Use the format that matches the task:\n"
        "  <note title=\"T\">body</note> — save a fact as a markdown note\n"
        "  <tool_call>{\"name\": \"...\", \"arguments\": {...}}</tool_call> — Hermes-style tool (also used for terminal with cmd=\"...\")\n"
        "  <cmd>command</cmd> — legacy shell command (still supported)\n\n"
        "Common Hermes tools:\n"
        "  note — save a fact as a markdown note\n"
        "  read_file — read a project file\n"
        "  search_files — search file contents\n"
        "  execute_code — run Python code in the sandbox\n"
        "  list_threads — list unread emails\n"
        "  get_thread — read one email thread\n"
        "  web_search / web_extract — web lookup\n"
        "  browser_open / browser_navigate / browser_click / browser_type / browser_scroll / browser_screenshot — control a web browser\n"
        "  browser_get_text / browser_get_html / browser_evaluate — read the current page\n"
        "  desktop_screenshot / desktop_click / desktop_type / desktop_press — control the macOS desktop\n\n"
        "Guidelines:\n"
        "- If 'Retrieved context' is provided in the conversation, rely on it FIRST for factual/memory questions.\n"
        "  Answer from that context when possible; say 'I do not have that in my notes' only when it is absent.\n"
        "- For live actions — read_file, terminal, execute_code, search_files, list_threads/get_thread, email, browser or desktop — always call the matching tool.\n"
        "  Do not answer from memory when the user asks to read a file, run a command, check email, or use the browser/desktop.\n"
        "- When the user corrects your answer, reply with the corrected answer directly. Do not use a note or other tool for the correction itself.\n"
        "- Save facts with <note>. Use Hermes <tool_call> for tools, including terminal (set cmd=\"...\").\n"
        "- To list files in a directory, use terminal with cmd=\"ls\"; do NOT use search_files.\n"
        "- To check unread emails or list emails, use list_threads. Only use get_thread when the user asks to read a specific email or provides an id.\n"
        "- To run Python code, use execute_code and start the code with 'from symbio_tools import *'.\n"
        "  symbio_tools only provides read_file, write_file, patch, search_files, terminal, web_search, web_extract;\n"
        "  for everything else use standard Python (e.g. import math; print(math.factorial(7))).\n"
        "- Take initiative: never tell the user to run a tool or do a step you can do yourself.\n"
        "  If a tool says the browser is not open, call browser_open with the right URL and continue the task.\n"
        "- Talk normally outside tags. Never include internal reasoning or analysis.\n"
        f"- When you speak, 'I/me/my' means {assistant_name}; 'you/your' means {user_name}.\n"
        "- Keep replies concise. After a tool succeeds, answer the user; do NOT repeat the tool.\n"
        "- If the user tells you a new name for themselves or for you, confirm the new name in your reply.\n"
        "  Do not use a note or other tool to update the identity; the system handles that automatically.\n\n"
        "Examples of correct identity answers:\n"
        f"User: What is my name?\nAssistant: Your name is {user_name}.\n"
        f"User: What is your name?\nAssistant: My name is {assistant_name}.\n"
        f"User: Who am I?\nAssistant: You are {user_name}.\n"
        f"User: Who are you?\nAssistant: I am {assistant_name}.\n"
        f"User: My name is {user_name}.\nAssistant: Got it — I'll call you {user_name} from now on.\n"
        f"User: Call yourself {assistant_name}.\nAssistant: Got it — my name is {assistant_name}.\n\n"
        "Examples of correct tool use:\n"
        f"User: Remember that I like coffee.\n"
        f'Assistant: <note title=\"User Preference\">{user_name} likes coffee.</note>Noted.\n'
        f"User: Show me config.json.\n"
        f'Assistant: <tool_call>{{"name": "read_file", "arguments": {{"path": "config.json"}}}}</tool_call>Reading config.json.\n'
        f"User: What is in the project directory?\n"
        f'Assistant: <tool_call>{{"name": "terminal", "arguments": {{"cmd": "ls -la"}}}}</tool_call>Listing the project directory.\n'
        f"User: Open example.com in the browser.\n"
        f'Assistant: <tool_call>{{"name": "browser_open", "arguments": {{"url": "https://example.com"}}}}</tool_call>Opening example.com.\n'
        f"User: Click the 'More information' link.\n"
        f'Assistant: <tool_call>{{"name": "browser_click", "arguments": {{"text": "More information"}}}}</tool_call>Clicking the link.\n'
        f"User: Take a screenshot of the page.\n"
        f'Assistant: <tool_call>{{"name": "browser_screenshot", "arguments": {{}}}}</tool_call>Taking a screenshot.\n'
        f"User: Scroll down to the next short.\n"
        f'Assistant: <tool_call>{{"name": "browser_scroll", "arguments": {{"direction": "down"}}}}</tool_call>Scrolling down.\n'
        f"User: Type 'hello' into the search box and submit.\n"
        f'Assistant: <tool_call>{{"name": "browser_type", "arguments": {{"selector": "input[name=q]", "text": "hello", "press_enter": true}}}}</tool_call>Typing and submitting.\n'
        f"User: What text is on the page?\n"
        f'Assistant: <tool_call>{{"name": "browser_get_text", "arguments": {{}}}}</tool_call>Reading the page text.\n'
        f"User: Take a desktop screenshot.\n"
        f'Assistant: <tool_call>{{"name": "desktop_screenshot", "arguments": {{}}}}</tool_call>Taking a desktop screenshot.'
    )


def print_banner(config: dict[str, Any], adapter_loaded: bool, dataset_size: int):
    """Print the Symbio CLI banner."""
    note_count = len(list(NOTES_DIR.glob("*.md")))
    screenshot_count = len(list(SCREENSHOTS_DIR.glob("*.png")))
    print("\n" + "=" * 50)
    model_type = config.get("_model_type", "unknown")
    ft_mode = config.get("model", {}).get("moe_fine_tuning_mode", "rag_only")
    print(f"  {config['assistant_name'].upper()} — PERSONAL CHAT-FINETUNE CLI")
    print(f"   Model  : {config['model_name']} ({model_type})")
    print(f"   User   : {config['user_name']}")
    print(f"   LoRA   : {'YES' if adapter_loaded else 'None (base)'} ({ft_mode})")
    print(f"   Data   : {dataset_size:,} bytes")
    print(f"   Notes  : {note_count}")
    print(f"   Shots  : {screenshot_count}")
    print("-" * 50)
    print("Commands: /quit  /save  /train  /learn  /forget_last  /status  /prune  /setup  /review")
    print("         /run <cmd>  /note [title]  /notes  /digest")
    print("  (Symbio can also use <note>, <cmd>, <digest />, <train />,")
    print("   Hermes <tool_call>{...}</tool_call>, and browser/desktop tools)")
    print("-" * 50)


def chat_loop(config: dict[str, Any]):
    """Run the interactive Symbio CLI loop."""
    # Lazy import to break a circular dependency: chat -> agent -> chat.
    from symbio.agent import AIAgent

    print(" Loading model...")
    adapter_config = ADAPTER_DIR / "adapter_config.json"
    adapter_loaded = adapter_config.exists()

    if adapter_loaded and not _adapter_matches_model(config):
        print(f" [Warning] Existing adapter was trained for a different model. Loading base model only.")
        adapter_loaded = False
        model, tokenizer = load(config["model_name"])
    elif adapter_loaded:
        print(" Found existing adapter. Loading it...")
        try:
            model, tokenizer = load(config["model_name"], adapter_path=str(ADAPTER_DIR))
            adapter_loaded = True
        except Exception as e:
            print(f" Could not load adapter: {e}")
            print(" Falling back to base model...")
            model, tokenizer = load(config["model_name"])
            adapter_loaded = False
    else:
        model, tokenizer = load(config["model_name"])

    model_type = detect_model_type(model)
    config["_model_type"] = model_type
    print(f" Model type: {model_type}")

    if config.get("first_run") or not config.get("user_name") or not config.get("assistant_name"):
        setup_names(config, first_run=True)

    ensure_seed_notes(config)
    temp_prompt = build_system_prompt(config["assistant_name"], config["user_name"], [])
    seed_training_data(tokenizer, temp_prompt, config)

    agent = AIAgent(config, model, tokenizer, adapter_loaded)

    dataset_size = TRAIN_FILE.stat().st_size if TRAIN_FILE.exists() else 0
    print_banner(config, agent.adapter_loaded, dataset_size)

    while True:
        try:
            user_input = input(f"{config['user_name']:8}: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            user_input = "/quit"

        if user_input.startswith("/"):
            cmd = user_input.lower()

            if cmd in ("/quit", "/q", "/exit"):
                print(" Exiting chat.")
                break

            elif cmd == "/forget_last":
                removed = agent.forget_last()
                print(f"  Forgot last exchange." if removed else " Nothing to forget.")
                continue

            elif cmd == "/save":
                if not agent.history:
                    print(" Nothing to save yet.")
                else:
                    saved_count = save_history_pairs(agent.history, agent.tokenizer, agent.system_prompt, agent.planner)
                    print(f" Saved {saved_count} exchange(s) to training data.")
                continue

            elif cmd == "/train" or cmd.startswith("/train "):
                force = "--force" in user_input.lower().split()
                model_type = config.get("_model_type", "dense")
                ok, reason = agent.planner.should_train(model_type=model_type)
                if not ok and not force:
                    print(f"  [Planner] {reason}")
                    print("  Use '/train --force' to override.")
                    continue
                if force:
                    print("  [Planner] /train --force: skipping planner checks.")
                elif ok:
                    print(f"  [Planner] {reason}")
                trained = run_training(config, model_type=model_type)
                if trained and adapter_config.exists():
                    print("\n Reloading model with updated adapter...")
                    try:
                        agent.reload_adapter()
                        print("  Adapter reloaded.")
                    except Exception as e:
                        print(f" Could not reload adapter: {e}")
                continue

            elif cmd == "/learn":
                if not config.get("learn", {}).get("enabled", True):
                    print("  /learn is disabled in config.")
                    continue
                if not agent.history:
                    print("  Nothing to learn from yet.")
                    continue
                note_path = learn_from_last_correction(agent)
                if not note_path:
                    continue
                trained = maybe_train_on_mistakes(config, tokenizer, agent.system_prompt, agent)
                if trained and adapter_config.exists():
                    print("  Reloading model with learned corrections...")
                    try:
                        agent.reload_adapter()
                        print("  Adapter reloaded.")
                    except Exception as e:
                        print(f"  Could not reload adapter: {e}")
                continue

            elif cmd == "/review":
                status = agent.planner.status()
                print("  Training planner status:")
                print(f"    Enabled: {status['enabled']}")
                print(f"    Recorded turns: {status['turn_count']}")
                print(f"    Top note refs: {status['top_note_refs']}")
                print(f"    Pending samples: {status['pending_samples']}")
                print(f"    Approved samples: {status['approved_samples']}")
                print(f"    Rejected samples: {status['rejected_samples']}")
                model_type = config.get("_model_type", "dense")
                ok, reason = agent.planner.should_train(model_type=model_type)
                print(f"    Can train: {'YES' if ok else 'NO'} — {reason}")
                continue

            elif cmd == "/digest":
                added = digest_notes_to_training(agent.tokenizer, agent.system_prompt, agent.planner)
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
                from symbio.sandbox import _run_sandboxed
                ok, output = _run_sandboxed(shell_cmd, config)
                status = "ok" if ok else "err"
                print(f"  [{status}]")
                for line in output.splitlines():
                    print(f"  {line}")
                append_chat_pair(
                    user_msg=f"Run this sandbox command and show the output:\n{shell_cmd}",
                    assistant_msg=output,
                    tokenizer=tokenizer,
                    system_prompt=agent.system_prompt,
                )
                print("  -> Logged to training data.\n")
                continue

            elif cmd == "/note" or cmd.startswith("/note "):
                title = user_input[len("/note"):].strip()
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
                screenshots = sorted(SCREENSHOTS_DIR.glob("*.png"))
                data_size = TRAIN_FILE.stat().st_size if TRAIN_FILE.exists() else 0
                adapter_files = list(ADAPTER_DIR.glob("adapters.*"))
                adapter_kb = sum(f.stat().st_size for f in ADAPTER_DIR.iterdir() if f.is_file()) // 1024
                print(f"  Model: {config['model_name']}")
                print(f"  Assistant: {config['assistant_name']} | User: {config['user_name']}")
                print(f"  Notes: {len(files)}")
                print(f"  Screenshots: {len(screenshots)}")
                print(f"  Training data: {data_size:,} bytes")
                print(f"  Adapter loaded: {'YES' if agent.adapter_loaded else 'NO'}")
                print(f"  Adapter files: {len(adapter_files)} ({adapter_kb:,} KB)")
                print(f"  Session log: {agent.session_log.name}")
                pstatus = agent.planner.status()
                print(f"  Planner turns: {pstatus['turn_count']} | approved: {pstatus['approved_samples']} | pending: {pstatus['pending_samples']}")
                continue

            elif cmd == "/setup":
                changed = setup_names(config, first_run=False)
                agent.update_identity(config["assistant_name"], config["user_name"])
                print("  Names updated and identity notes saved.")
                if changed:
                    print("  Tip: run /digest then /train so I learn the new identity.")
                continue

            elif cmd == "/model" or cmd.startswith("/model "):
                preset = user_input[len("/model"):].strip()
                if not preset:
                    list_model_presets(config)
                else:
                    switch_model_preset(config, preset)
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
            # Empty Enter: move the cursor back up and clear the line so the
            # prompt stays anchored instead of stacking a new "name:" line.
            # Nothing is logged, persisted, or sent to the model.
            if sys.stdout.isatty():
                sys.stdout.write("\033[F\033[K")
                sys.stdout.flush()
            continue

        # Detect in-conversation name changes (e.g. "My name is Alice" or "Your name is Jarvis").
        if maybe_update_names_from_message(user_input, config):
            agent.update_identity(config["assistant_name"], config["user_name"])
            print(
                f"  [System] Updated identity: I am {config['assistant_name']}, "
                f"you are {config['user_name']}."
            )

        chat_logger.info(f"User: {user_input}")

        # Auto-detect natural corrections ("No, ...", "Actually ...", repeated question).
        is_correction, correction_reason = _looks_like_correction(user_input, agent.history, config)

        agent.run(user_input)

        if is_correction and config.get("learn", {}).get("enabled", True):
            print(f"\n  [System] Correction detected ({correction_reason}). Saving mistake note...")
            note_path = learn_from_last_correction(agent)
            if note_path:
                maybe_train_on_mistakes(config, tokenizer, agent.system_prompt, agent)
                if adapter_config.exists():
                    try:
                        agent.reload_adapter()
                        print("  [System] Adapter reloaded with any new corrections.")
                    except Exception as e:
                        print(f"  [System] Could not reload adapter: {e}")

    if agent.history:
        try:
            save = input("\n Save conversation for training? [y/N]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            save = "n"
        if save in ("y", "yes"):
            saved_count = save_history_pairs(agent.history, agent.tokenizer, agent.system_prompt, agent.planner)
            print(f"    Appended {saved_count} exchange(s) to {TRAIN_FILE}")

            try:
                train_now = input("  Train now? [y/N]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                train_now = "n"
            if train_now in ("y", "yes"):
                trained = run_training(config, model_type=config.get("_model_type", "dense"))
                if trained and adapter_config.exists():
                    print("\n Reloading model...")
                    try:
                        agent.reload_adapter()
                    except Exception as e:
                        print(f" Could not reload: {e}")
