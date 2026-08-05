"""Rewrite tool_few_shots in symbio/tools.py to a minimal, all-legacy-tag
form matching the LoRA training data, leading with a greeting example so
greetings no longer trigger a tool call. Newlines inside assistant content
are built with chr(10) to avoid escape-sequence ambiguity.
"""
import re
import shutil

path = "symbio/tools.py"
shutil.copy2(path, path + ".bak.fewshots")
src = open(path, encoding="utf-8").read()

NL = " + chr(10) + "

new_fn = '''def tool_few_shots(config: dict[str, Any]) -> list[dict[str, str]]:
    """Minimal legacy short-tag examples matching the LoRA training data:
    greetings -> prose (no tool), actions -> one legacy tag."""
    uname = config["user_name"]
    return [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": f"Hi {uname}! What can I do for you?"},
        {"role": "user", "content": "open chrome to the apple website"},
        {"role": "assistant", "content": "<cmd>open -a 'Google Chrome' 'https://www.apple.com'</cmd>''' + NL + '''"Opening Apple.com in Chrome."},
        {"role": "user", "content": "what's the weather in sydney"},
        {"role": "assistant", "content": "<search>current weather Sydney</search>''' + NL + '''"Looking up the weather for you."},
        {"role": "user", "content": f"remember that {uname} likes coffee"},
        {"role": "assistant", "content": f'<note title="User Preference">{uname} likes coffee.</note>''' + NL + '''"Noted."},
        {"role": "user", "content": "how much free disk space do I have"},
        {"role": "assistant", "content": "<cmd>df -h</cmd>''' + NL + '''"Checking disk space."},
    ]


'''

pattern = re.compile(r"def tool_few_shots\(config.*?\n(?=def tool_metadata\b)", re.S)
new_src, n = pattern.subn(new_fn, src)
assert n == 1, f"expected 1 replacement, got {n}"
open(path, "w", encoding="utf-8").write(new_src)
print("replaced tool_few_shots; backup at", path + ".bak.fewshots")