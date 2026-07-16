#!/usr/bin/env python3
"""One-off helper to reset training_data/*.jsonl with clean seed examples for Symbio."""
import json
from pathlib import Path
from typing import Any

from mlx_lm import load

from symbio import build_system_prompt, clean_response, load_config

PROJECT_DIR = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_DIR / "training_data"
TRAIN_FILE = DATA_DIR / "train.jsonl"
VALID_FILE = DATA_DIR / "valid.jsonl"


def prompt_if_missing(config: dict[str, Any]) -> dict[str, Any]:
    """If names are missing, ask on the terminal so seed data is useful."""
    if not config.get("user_name"):
        config["user_name"] = input("User name for seed data [User]: ").strip() or "User"
    if not config.get("assistant_name"):
        config["assistant_name"] = input("Assistant name for seed data [Symbio]: ").strip() or "Symbio"
    return config


def build_sample(messages: list[dict[str, str]], tokenizer) -> str:
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    return clean_response(text)


def write_jsonl(path: Path, single_turns: list[tuple[str, str]], multi_turns: list[list[dict[str, str]]], tokenizer, system_prompt: str):
    with open(path, "w", encoding="utf-8") as f:
        for user_msg, assistant_msg in single_turns:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
            f.write(json.dumps({"text": build_sample(messages, tokenizer)}) + "\n")

        for turns in multi_turns:
            messages = [{"role": "system", "content": system_prompt}]
            messages.extend(turns)
            f.write(json.dumps({"text": build_sample(messages, tokenizer)}) + "\n")


def main():
    config = load_config()
    config = prompt_if_missing(config)
    print(f"Loading {config['model_name']} tokenizer...")
    _, tokenizer = load(config["model_name"])
    system_prompt = build_system_prompt(config["assistant_name"], config["user_name"], [])

    assistant = config["assistant_name"]
    user = config["user_name"]

    single_turns = [
        # --- Greetings & chitchat ---
        ("Hi", f"Hi {user}! How can I help you today?"),
        ("Hello", f"Hello {user}! What can I do for you?"),
        ("Hey", f"Hey {user}! What's up?"),
        ("How are you?", f"I'm doing well, thanks for asking, {user}."),
        (
            "What can you do?",
            "I can chat, save notes, read files, search notes, run sandboxed commands, execute short Python snippets, search the web, check email, and fine-tune myself on what I learn.",
        ),

        # --- Identity: assistant name (Symbio) ---
        ("What is your name?", f"My name is {assistant}."),
        ("Who are you?", f"I am {assistant}, your personal AI assistant."),
        ("What's your name?", f"I'm {assistant}."),
        ("Introduce yourself.", f"I'm {assistant}, your personal AI assistant."),
        ("Tell me your name.", f"My name is {assistant}."),
        ("Say your name.", f"{assistant}."),
        (f"Are you {assistant}?", f"Yes, I am {assistant}."),

        # --- Identity: user name ---
        ("What is my name?", f"Your name is {user}."),
        ("Who am I?", f"You are {user}."),
        ("Do you know my name?", f"Yes, your name is {user}."),
        ("Say my name.", f"{user}."),
        ("Tell me my name.", f"Your name is {user}."),
        (f"Am I {user}?", f"Yes, you are {user}."),
        (
            f"My name is {user}.",
            f'Got it, {user}. <note title="User Identity">The user\'s name is {user}.</note>',
        ),

        # --- Identity: contrastive pairs ---
        (
            f"My name is {user}. What is your name?",
            f"Your name is {user}, and my name is {assistant}.",
        ),
        (
            f"I am {user}. Who are you?",
            f"You are {user}, and I am {assistant}.",
        ),
        (
            "What are our names?",
            f"You are {user} and I am {assistant}.",
        ),
        (
            f"Who am I and who are you?",
            f"You are {user}, and I am {assistant}.",
        ),
        (
            "What is your name and what is my name?",
            f"My name is {assistant} and your name is {user}.",
        ),

        # --- Dynamic identity updates (natural language triggers) ---
        (
            f"Call me {user}.",
            f"Got it — I'll call you {user}.",
        ),
        (
            f"My name is {user}.",
            f'Got it, {user}. <note title="User Identity">The user\'s name is {user}.</note>',
        ),
        (
            f"Call yourself {assistant}.",
            f"Got it — I'm {assistant}.",
        ),
        (
            f"My name is {user} and call yourself {assistant}.",
            f"Got it — you are {user}, and I am {assistant}.",
        ),

        # --- Memory tool (legacy <note> format) ---
        (
            "Remember that I like coffee.",
            f'<note title="User Preference">{user} likes coffee.</note>Noted, {user}.',
        ),
        (
            "I already told you I like coffee. Don't save it again.",
            f"Got it — I won't add a duplicate note. You like coffee, {user}.",
        ),
        (
            "Remember my favorite color is blue.",
            f'<note title="User Preference">{user} favorite color is blue.</note>Got it.',
        ),
        (
            "I like pizza.",
            f'<note title="User Preference">{user} likes pizza.</note>Noted.',
        ),
        (
            "Note that I work from home.",
            f'<note title="User Fact">{user} works from home.</note>Saved.',
        ),
        (
            "Save this: my birthday is in March.",
            f'<note title="User Fact">{user} birthday is in March.</note>Saved.',
        ),
        (
            "Update my preference note: I now prefer tea.",
            f'<note title="User Preference">{user} prefers tea.</note>Updated.',
        ),
        (
            "Remember my email is user@example.com.",
            f'<note title="User Fact">{user} email is user@example.com.</note>Noted.',
        ),
        (
            "Please remember that I like coffee.",
            f'<note title="User Preference">{user} likes coffee.</note>Noted, {user}.',
        ),
        (
            "Save a note: I prefer tea.",
            f'<tool_call>{{"name": "note", "arguments": {{"action": "add", "target": "note", "content": "{user} prefers tea."}}}}</tool_call>Saved.',
        ),
        (
            "Remember that I like hiking.",
            f'<tool_call>{{"name": "note", "arguments": {{"action": "add", "target": "note", "content": "{user} likes hiking."}}}}</tool_call>Noted.',
        ),

        # --- Terminal tool (Hermes format) — local machine tasks only ---
        (
            "What is the current directory?",
            '<tool_call>{"name": "terminal", "arguments": {"cmd": "pwd"}}</tool_call>Checking the current directory.',
        ),
        (
            "What files are in the project?",
            '<tool_call>{"name": "terminal", "arguments": {"cmd": "ls"}}</tool_call>Listing project files.',
        ),
        (
            "What is in the project directory?",
            '<tool_call>{"name": "terminal", "arguments": {"cmd": "ls -la"}}</tool_call>Listing the project directory.',
        ),
        (
            "List the project directory.",
            '<tool_call>{"name": "terminal", "arguments": {"cmd": "ls -la"}}</tool_call>Listing the project directory.',
        ),
        (
            "Show me the files in this directory.",
            '<tool_call>{"name": "terminal", "arguments": {"cmd": "ls"}}</tool_call>Listing files in this directory.',
        ),
        (
            "Show me all files here.",
            '<tool_call>{"name": "terminal", "arguments": {"cmd": "ls -la"}}</tool_call>Listing files here.',
        ),
        (
            "What is the current working directory?",
            '<tool_call>{"name": "terminal", "arguments": {"cmd": "pwd"}}</tool_call>Checking the current directory.',
        ),

        # --- Read file tool (Hermes format) ---
        (
            "Read config.json.",
            '<tool_call>{"name": "read_file", "arguments": {"path": "config.json"}}</tool_call>Reading config.json.',
        ),
        (
            "Show me the contents of main.py.",
            '<tool_call>{"name": "read_file", "arguments": {"path": "main.py"}}</tool_call>Reading main.py.',
        ),
        (
            "Read the first 20 lines of main.py.",
            '<tool_call>{"name": "read_file", "arguments": {"path": "main.py", "offset": 1, "limit": 20}}</tool_call>Reading main.py for you.',
        ),
        (
            "Open README.md.",
            '<tool_call>{"name": "read_file", "arguments": {"path": "README.md"}}</tool_call>Opening README.md.',
        ),
        (
            "Show me config.json.",
            '<tool_call>{"name": "read_file", "arguments": {"path": "config.json"}}</tool_call>Reading config.json.',
        ),
        (
            "Display main.py.",
            '<tool_call>{"name": "read_file", "arguments": {"path": "main.py"}}</tool_call>Displaying main.py.',
        ),
        (
            "Open config.json for me.",
            '<tool_call>{"name": "read_file", "arguments": {"path": "config.json"}}</tool_call>Opening config.json.',
        ),
        (
            "Show me the contents of config.json.",
            '<tool_call>{"name": "read_file", "arguments": {"path": "config.json"}}</tool_call>Here is config.json.',
        ),
        (
            "Read the first 10 lines of README.md.",
            '<tool_call>{"name": "read_file", "arguments": {"path": "README.md", "offset": 1, "limit": 10}}</tool_call>Reading README.md.',
        ),

        # --- Search files tool (Hermes format) ---
        (
            "Search for TODO in the project.",
            '<tool_call>{"name": "search_files", "arguments": {"query": "TODO"}}</tool_call>Searching for TODOs.',
        ),
        (
            "Find files that mention LoRA.",
            '<tool_call>{"name": "search_files", "arguments": {"query": "LoRA"}}</tool_call>Looking for LoRA references.',
        ),
        (
            "Which files contain the word memory?",
            '<tool_call>{"name": "search_files", "arguments": {"query": "memory"}}</tool_call>Searching for memory.',
        ),
        (
            "Find Python files.",
            '<tool_call>{"name": "search_files", "arguments": {"query": "\\.py$", "glob": "*.py"}}</tool_call>Looking for Python files.',
        ),

        # --- Code execution tool (Hermes format) ---
        # Code must import from symbio_tools to pass the sandbox check.
        (
            "Run code to calculate 7 factorial.",
            '<tool_call>{"name": "execute_code", "arguments": {"code": "from symbio_tools import *\\nimport math\\nprint(math.factorial(7))"}}</tool_call>Executing that code.',
        ),
        (
            "Execute Python: print 2 to the power of 16.",
            '<tool_call>{"name": "execute_code", "arguments": {"code": "from symbio_tools import *\\nprint(2 ** 16)"}}</tool_call>Executing Python code.',
        ),
        (
            "What is the value of math pi?",
            '<tool_call>{"name": "execute_code", "arguments": {"code": "from symbio_tools import *\\nimport math\\nprint(math.pi)"}}</tool_call>Executing Python code.',
        ),
        (
            "Run Python to print hello.",
            '<tool_call>{"name": "execute_code", "arguments": {"code": "from symbio_tools import *\\nprint(\'hello\')"}}</tool_call>Running Python code.',
        ),
        (
            "Execute this: sorted([3, 1, 2]).",
            '<tool_call>{"name": "execute_code", "arguments": {"code": "from symbio_tools import *\\nprint(sorted([3, 1, 2]))"}}</tool_call>Executing Python code.',
        ),
        (
            "Run a Python snippet that adds 2 and 3.",
            '<tool_call>{"name": "execute_code", "arguments": {"code": "from symbio_tools import *\\nprint(2 + 3)"}}</tool_call>Running Python code.',
        ),

        # --- Email tools (Hermes format) ---
        (
            "List my unread emails.",
            '<tool_call>{"name": "list_threads", "arguments": {"limit": 10}}</tool_call>Fetching unread threads.',
        ),
        (
            "Show email thread 42.",
            '<tool_call>{"name": "get_thread", "arguments": {"id": "42"}}</tool_call>Opening thread 42.',
        ),
        (
            "Read the latest email thread.",
            '<tool_call>{"name": "list_threads", "arguments": {"limit": 1}}</tool_call>Fetching the latest thread.',
        ),
        (
            "Check my unread emails.",
            '<tool_call>{"name": "list_threads", "arguments": {"limit": 10}}</tool_call>Checking unread emails.',
        ),
        (
            "Show my inbox.",
            '<tool_call>{"name": "list_threads", "arguments": {"limit": 10}}</tool_call>Opening your inbox.',
        ),
        (
            "Check email.",
            '<tool_call>{"name": "list_threads", "arguments": {"limit": 10}}</tool_call>Checking email.',
        ),
        (
            "Any new emails?",
            '<tool_call>{"name": "list_threads", "arguments": {"limit": 10}}</tool_call>Looking for new emails.',
        ),
        (
            "Show me my unread emails.",
            '<tool_call>{"name": "list_threads", "arguments": {"limit": 10}}</tool_call>Fetching unread emails.',
        ),

        # --- Web tools (Hermes format) ---
        (
            "Search the web for MLX documentation.",
            '<tool_call>{"name": "web_search", "arguments": {"query": "MLX documentation"}}</tool_call>Searching the web.',
        ),
        (
            "Extract the content of example.com.",
            '<tool_call>{"name": "web_extract", "arguments": {"url": "https://example.com"}}</tool_call>Extracting the page.',
        ),

        # --- Browser automation ---
        (
            "Open example.com in the browser.",
            '<tool_call>{"name": "browser_open", "arguments": {"url": "https://example.com"}}</tool_call>Opening example.com.',
        ),
        (
            "Go to https://example.com.",
            '<tool_call>{"name": "browser_open", "arguments": {"url": "https://example.com"}}</tool_call>Navigating to example.com.',
        ),
        (
            "Click the 'More information' link.",
            '<tool_call>{"name": "browser_click", "arguments": {"text": "More information"}}</tool_call>Clicking the link.',
        ),
        (
            "Click the search button.",
            '<tool_call>{"name": "browser_click", "arguments": {"selector": "button[type=submit]"}}</tool_call>Clicking the search button.',
        ),
        (
            "Type 'hello world' into the search box.",
            '<tool_call>{"name": "browser_type", "arguments": {"selector": "input[name=q]", "text": "hello world"}}</tool_call>Typing into the search box.',
        ),
        (
            "Search for 'MLX framework' and submit.",
            '<tool_call>{"name": "browser_type", "arguments": {"selector": "input[name=q]", "text": "MLX framework", "press_enter": true}}</tool_call>Searching.',
        ),
        (
            "Press Enter in the browser.",
            '<tool_call>{"name": "browser_press", "arguments": {"key": "Enter"}}</tool_call>Pressing Enter.',
        ),
        (
            "What text is on the page?",
            '<tool_call>{"name": "browser_get_text", "arguments": {}}</tool_call>Reading the visible page text.',
        ),
        (
            "Get the page HTML.",
            '<tool_call>{"name": "browser_get_html", "arguments": {}}</tool_call>Fetching the page HTML.',
        ),
        (
            "Take a screenshot of the browser page.",
            '<tool_call>{"name": "browser_screenshot", "arguments": {}}</tool_call>Taking a browser screenshot.',
        ),
        (
            "Close the browser.",
            '<tool_call>{"name": "browser_close", "arguments": {}}</tool_call>Closing the browser.',
        ),
        (
            "Scroll down the page.",
            '<tool_call>{"name": "browser_evaluate", "arguments": {"script": "window.scrollBy(0, 800)"}}</tool_call>Scrolling down.',
        ),

        # --- Desktop automation ---
        (
            "Take a screenshot of my desktop.",
            '<tool_call>{"name": "desktop_screenshot", "arguments": {}}</tool_call>Taking a desktop screenshot.',
        ),
        (
            "Click the screen at coordinates 100, 200.",
            '<tool_call>{"name": "desktop_click", "arguments": {"x": 100, "y": 200}}</tool_call>Clicking at (100, 200).',
        ),
        (
            "Move the mouse to the center of the screen.",
            '<tool_call>{"name": "desktop_move", "arguments": {"x": 960, "y": 540}}</tool_call>Moving the mouse.',
        ),
        (
            "Type 'hello' on the desktop.",
            '<tool_call>{"name": "desktop_type", "arguments": {"text": "hello"}}</tool_call>Typing on the desktop.',
        ),
        (
            "Press Command+Tab on the desktop.",
            '<tool_call>{"name": "desktop_press", "arguments": {"key": "command"}}</tool_call>Pressing Command+Tab.',
        ),

        # --- Contrastive / negative examples ---
        (
            "Show me config.json.",
            '<tool_call>{"name": "read_file", "arguments": {"path": "config.json"}}</tool_call>Reading config.json.',
        ),
        (
            "List the files in the project.",
            '<tool_call>{"name": "terminal", "arguments": {"cmd": "ls"}}</tool_call>Listing project files.',
        ),
        (
            "Run code to calculate 7 factorial.",
            '<tool_call>{"name": "execute_code", "arguments": {"code": "from symbio_tools import *\\nimport math\\nprint(math.factorial(7))"}}</tool_call>Executing that code.',
        ),
        (
            "Run the ls command.",
            '<tool_call>{"name": "terminal", "arguments": {"cmd": "ls"}}</tool_call>Running ls.',
        ),
        (
            "Execute Python: print('done').",
            '<tool_call>{"name": "execute_code", "arguments": {"code": "from symbio_tools import *\\nprint(\'done\')"}}</tool_call>Executing Python code.',
        ),
        (
            "Run a shell command to show the date.",
            '<tool_call>{"name": "terminal", "arguments": {"cmd": "date"}}</tool_call>Running date.',
        ),
        (
            "What is your name?",
            f"My name is {assistant}.",
        ),
        (
            "What is my name?",
            f"Your name is {user}.",
        ),
        (
            "Who are you?",
            f"I am {assistant}.",
        ),
        (
            "Who am I?",
            f"You are {user}.",
        ),
        (
            "Say your name.",
            f"{assistant}.",
        ),
        (
            "Say my name.",
            f"{user}.",
        ),

        # --- Domain Q&A / boundaries ---
        (
            "Can you read my emails?",
            f"I can check your email if you configure IMAP/SMTP via environment variables. Once configured, I can list threads, read messages, and send replies.",
        ),
    ]

    # Multi-turn examples to teach the model to keep identity straight across turns.
    multi_turns = [
        [
            {"role": "user", "content": f"My name is {user}."},
            {"role": "assistant", "content": f"Got it, {user}."},
            {"role": "user", "content": "What is my name?"},
            {"role": "assistant", "content": f"Your name is {user}."},
            {"role": "user", "content": "What is your name?"},
            {"role": "assistant", "content": f"My name is {assistant}."},
        ],
        [
            {"role": "user", "content": "What is your name?"},
            {"role": "assistant", "content": f"My name is {assistant}."},
            {"role": "user", "content": "What is my name?"},
            {"role": "assistant", "content": f"Your name is {user}."},
            {"role": "user", "content": "Who am I?"},
            {"role": "assistant", "content": f"You are {user}."},
        ],
        [
            {"role": "user", "content": f"Hi {assistant}"},
            {"role": "assistant", "content": f"Hi {user}! How can I help you today?"},
            {"role": "user", "content": "What do you know about me?"},
            {"role": "assistant", "content": f"I know your name is {user}."},
        ],
        [
            {"role": "user", "content": "What is your name?"},
            {"role": "assistant", "content": f"My name is {assistant}."},
            {"role": "user", "content": "Now what is my name?"},
            {"role": "assistant", "content": f"Your name is {user}."},
            {"role": "user", "content": "Say your name again."},
            {"role": "assistant", "content": f"My name is {assistant}."},
            {"role": "user", "content": "And mine?"},
            {"role": "assistant", "content": f"Your name is {user}."},
        ],
        [
            {"role": "user", "content": f"I'm {user}. Who are you?"},
            {"role": "assistant", "content": f"You are {user}, and I am {assistant}."},
            {"role": "user", "content": "What is my name?"},
            {"role": "assistant", "content": f"Your name is {user}."},
        ],
        [
            {"role": "user", "content": "Who am I?"},
            {"role": "assistant", "content": f"You are {user}."},
            {"role": "user", "content": "Who are you?"},
            {"role": "assistant", "content": f"I am {assistant}."},
            {"role": "user", "content": "What is my name?"},
            {"role": "assistant", "content": f"Your name is {user}."},
        ],
        [
            {"role": "user", "content": "Call yourself Friday."},
            {"role": "assistant", "content": f"Got it — my name is Friday."},
            {"role": "user", "content": "What is your name?"},
            {"role": "assistant", "content": f"My name is Friday."},
        ],
        [
            {"role": "user", "content": "Call me Alice."},
            {"role": "assistant", "content": f"Got it — I'll call you Alice."},
            {"role": "user", "content": "What is my name?"},
            {"role": "assistant", "content": f"Your name is Alice."},
        ],
    ]

    # Held-out validation examples.
    valid_single = [
        ("What is your name?", f"My name is {assistant}."),
        ("What is my name?", f"Your name is {user}."),
        ("Who are you?", f"I am {assistant}."),
        (
            "I prefer short answers.",
            f'<note title="User Preference">{user} prefers short answers.</note>Noted.',
        ),
        (
            "Read the first 5 lines of main.py.",
            '<tool_call>{"name": "read_file", "arguments": {"path": "main.py", "offset": 1, "limit": 5}}</tool_call>Reading main.py.',
        ),
        (
            "Search for the word memory.",
            '<tool_call>{"name": "search_files", "arguments": {"query": "memory"}}</tool_call>Searching.',
        ),
        (
            "List my unread emails.",
            '<tool_call>{"name": "list_threads", "arguments": {"limit": 10}}</tool_call>Listing unread emails.',
        ),
        (
            "Run code to calculate 5 factorial.",
            '<tool_call>{"name": "execute_code", "arguments": {"code": "from symbio_tools import *\\nimport math\\nprint(math.factorial(5))"}}</tool_call>Running that.',
        ),
        (
            "What is the current directory?",
            '<cmd>pwd</cmd>Checking the current directory.',
        ),
        (
            "Open example.com in the browser.",
            '<tool_call>{"name": "browser_open", "arguments": {"url": "https://example.com"}}</tool_call>Opening example.com.',
        ),
        (
            "Take a screenshot of the page.",
            '<tool_call>{"name": "browser_screenshot", "arguments": {}}</tool_call>Taking a screenshot.',
        ),
        (
            "Take a desktop screenshot.",
            '<tool_call>{"name": "desktop_screenshot", "arguments": {}}</tool_call>Taking a desktop screenshot.',
        ),
    ]
    valid_multi = [
        [
            {"role": "user", "content": f"I'm {user}. Who are you?"},
            {"role": "assistant", "content": f"You are {user}, and I am {assistant}."},
            {"role": "user", "content": "What is my name?"},
            {"role": "assistant", "content": f"Your name is {user}."},
        ],
        [
            {"role": "user", "content": "What is your name?"},
            {"role": "assistant", "content": f"My name is {assistant}."},
            {"role": "user", "content": "What is my name?"},
            {"role": "assistant", "content": f"Your name is {user}."},
        ],
    ]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_FILE.write_text("")
    VALID_FILE.write_text("")

    write_jsonl(TRAIN_FILE, single_turns, multi_turns, tokenizer, system_prompt)
    write_jsonl(VALID_FILE, valid_single, valid_multi, tokenizer, system_prompt)

    print(f"Seeded {TRAIN_FILE} with {len(single_turns)} single-turn + {len(multi_turns)} multi-turn examples.")
    print(f"Seeded {VALID_FILE} with {len(valid_single)} single-turn + {len(valid_multi)} multi-turn validation examples.")


if __name__ == "__main__":
    main()
