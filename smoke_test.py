#!/usr/bin/env python3
"""Smoke tests for Symbio after retraining."""
from pathlib import Path

from mlx_lm import load
from symbio import AIAgent, DEFAULT_CONFIG, load_config, parse_tools

PROJECT_DIR = Path(__file__).parent.resolve()
CONFIG_FILE = PROJECT_DIR / "config.json"
ADAPTER_DIR = PROJECT_DIR / "adapters"


def extract_tool_calls(text: str) -> list[str]:
    """Return the names of tool calls found in text."""
    return [name for name, _ in parse_tools(text)]


def run_case(agent, case):
    name, prompt, expected_tools = case
    print(f"\n[TEST] {name}")
    print(f"  User: {prompt}")
    result = agent.run(prompt)
    text = result["text"]
    print(f"  Symbio: {text}")

    # Collect tool names emitted by the assistant across the whole turn history.
    all_tool_names = []
    for turn in result["history"]:
        if turn.get("role") == "assistant":
            all_tool_names.extend(extract_tool_calls(turn.get("content", "")))

    print(f"  Tools used: {all_tool_names}")
    return result, expected_tools, all_tool_names


def main():
    config = load_config()
    print(f"Loading {config['model_name']}...")
    adapter_config = ADAPTER_DIR / "adapter_config.json"
    if adapter_config.exists():
        print("Found adapter. Loading with LoRA...")
        model, tokenizer = load(config["model_name"], adapter_path=str(ADAPTER_DIR))
    else:
        print("No adapter found. Using base model.")
        model, tokenizer = load(config["model_name"])

    agent = AIAgent(config, model, tokenizer, adapter_config.exists())

    cases = [
        ("Identity: assistant name", "What is your name?", []),
        ("Identity: user name", "What is my name?", []),
        ("Memory", "Remember that I like coffee.", ["note"]),
        ("Read file", "Show me config.json.", ["read_file"]),
        ("Terminal", "What is in the project directory?", ["terminal"]),
        ("Execute code", "Run code to calculate 7 factorial.", ["execute_code"]),
        ("Email list", "Check my unread emails.", ["list_threads"]),
        ("Search files", "Find files that mention LoRA.", ["search_files"]),
    ]

    results = []
    for case in cases:
        agent.history.clear()
        agent._code_calls_this_turn = 0
        try:
            result, expected, used = run_case(agent, case)
            ok = all(tool in used for tool in expected)
            # Identity checks have a textual requirement too.
            if case[0] == "Identity: assistant name":
                ok = ok and config["assistant_name"] in result["text"]
            if case[0] == "Identity: user name":
                ok = ok and config["user_name"] in result["text"]
            results.append((case[0], ok, result))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((case[0], False, None))

    print("\n" + "=" * 50)
    print("Smoke test complete.")
    all_ok = True
    for name, ok, _ in results:
        status = "OK" if ok else "FAIL"
        if not ok:
            all_ok = False
        print(f"  [{status}] {name}")

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
