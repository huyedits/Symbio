"""Augment train.jsonl with diverse tool-request phrasings.

The adapter overfit because tool examples had too few phrasings (lots of
"Open Chrome." but zero "open chrome to X website"). This script appends many
varied user phrasings mapped to the correct short-tag tool call, so the adapter
learns intent -> tag as a generalizable function rather than memorizing exact
strings. Templated with the real system prompt + Qwen tokenizer to match
inference exactly.
"""
import json
import shutil
from mlx_lm.utils import load_tokenizer
from symbio.app import prompts

MODEL = "mlx-community/Qwen3-8B-3bit"
ANAME, UNAME = "Caine", "Huy"

# (user phrasing, assistant response) pairs. Assistant uses the short-tag format
# that parse_tools recognizes: <cmd>, <browse>, <search>, <note>.
SITES = {
    "apple": ("apple.com", "https://www.apple.com", "Apple.com"),
    "google": ("google.com", "https://www.google.com", "Google"),
    "youtube": ("youtube.com", "https://www.youtube.com", "YouTube"),
    "github": ("github.com", "https://github.com", "GitHub"),
    "cloudflare": ("cloudflare.com", "https://www.cloudflare.com", "Cloudflare"),
    "wikipedia": ("wikipedia.com", "https://www.wikipedia.org", "Wikipedia"),
    "reddit": ("reddit.com", "https://www.reddit.com", "Reddit"),
    "netflix": ("netflix.com", "https://www.netflix.com", "Netflix"),
}

pairs = []

# --- Browser: just opening (shell open -a Chrome) ---
open_phrasings = [
    "open chrome to the {n} website",
    "open chrome to {d}",
    "open chrome to {n}",
    "open {d} in chrome",
    "open {d} in Chrome",
    "go to {d} on chrome",
    "launch chrome and go to {d}",
    "take me to {d} in chrome",
    "can you open chrome to {d}",
    "open google chrome to {d}",
    "navigate to {d} in chrome",
    "fire up chrome and head to {d}",
    "open chrome, go to {n} website",
    "i want to visit {d} in chrome",
    "pull up {d} in chrome for me",
    "open {n} in chrome",
    "bring up {d} in chrome",
    "hop on chrome and open {d}",
]
for key, (domain, url, name) in SITES.items():
    for tmpl in open_phrasings:
        u = tmpl.format(n=key, d=domain)
        a = f"<cmd>open -a 'Google Chrome' '{url}'</cmd>\nOpening {name} in Chrome."
        pairs.append((u, a))

# --- Browser automation (controllable <browse>) ---
auto_pairs = [
    ("open {d} in your browser and click the first button",
     "<browse>{u}</browse>\nOpening {name} in the controllable browser."),
    ("go to {d} and read the page",
     "<browse>{u}</browse>\nOpening {name} in the controllable browser."),
    ("browse to {d} and scroll down",
     "<browse>{u}</browse>\nOpening {name} in the controllable browser."),
    ("open {d} in the controllable browser",
     "<browse>{u}</browse>\nOpening {name} in the controllable browser."),
]
for key, (domain, url, name) in SITES.items():
    for ut, at in auto_pairs:
        pairs.append((ut.format(d=domain), at.format(u=url, name=name)))

# --- Web search (<search>) ---
search_pairs = [
    ("what's the weather in sydney", "<search>current weather Sydney</search>\nLooking up the weather in Sydney."),
    ("search the web for mlx-lm docs", "<search>mlx-lm documentation</search>\nSearching the web for mlx-lm docs."),
    ("google how to cook rice", "<search>how to cook rice</search>\nSearching for how to cook rice."),
    ("look up the capital of france", "<search>capital of France</search>\nLooking that up."),
    ("find out who won the world cup this year", "<search>who won the world cup this year</search>\nSearching the web."),
    ("search youtube for lofi beats", "<search>lofi beats</search>\nSearching YouTube for lofi beats."),
    ("what's the latest news on AI", "<search>latest AI news</search>\nSearching the web for AI news."),
    ("search for python asyncio tutorial", "<search>python asyncio tutorial</search>\nSearching for a Python asyncio tutorial."),
    ("look up the weather in new york city", "<search>current weather New York City</search>\nLooking up the weather for you."),
    ("google the distance to the moon", "<search>distance to the moon</search>\nSearching for the distance to the moon."),
]
pairs.extend(search_pairs)

# --- Terminal (<cmd>) ---
term_pairs = [
    ("how much free disk space do i have", "<cmd>df -h</cmd>\nChecking disk space."),
    ("list the files in this directory", "<cmd>ls -la</cmd>\nListing the directory."),
    ("what's my current directory", "<cmd>pwd</cmd>\nChecking the current directory."),
    ("show me the contents of config.json", "<cmd>cat config.json</cmd>\nShowing config.json."),
    ("how many files are in the notes folder", "<cmd>ls notes | wc -l</cmd>\nCounting files in notes."),
    ("what time is it", "<cmd>date</cmd>\nChecking the current time."),
    ("show me the first 20 lines of training.log", "<cmd>head -20 training.log</cmd>\nShowing the first 20 lines of training.log."),
    ("find all python files here", "<cmd>find . -name '*.py'</cmd>\nFinding Python files."),
]
pairs.extend(term_pairs)

# --- Notes/memory (<note>) ---
note_pairs = [
    ("remember that i like coffee", "<note title=\"User Preference\">Huy likes coffee.</note>\nNoted."),
    ("remember my favorite color is blue", "<note title=\"User Preference\">Huy's favorite color is blue.</note>\nNoted."),
    ("remember that i prefer dark mode", "<note title=\"User Preference\">Huy prefers dark mode.</note>\nNoted."),
    ("note that the deploy happens at 5pm", "<note title=\"Schedule\">Deploy happens at 5pm.</note>\nNoted."),
    ("remember i'm working on the symbio project", "<note title=\"User Project\">Huy is working on the Symbio project.</note>\nNoted."),
]
pairs.extend(note_pairs)

# De-duplicate pairs (in case site/phrase combos collide)
seen = set()
unique = []
for u, a in pairs:
    if u in seen:
        continue
    seen.add(u)
    unique.append((u, a))
pairs = unique

print(f"generated {len(pairs)} augmentation pairs")

tok = load_tokenizer(MODEL)
system_prompt = prompts.build_system_prompt(ANAME, UNAME)

src = "training_data/train.jsonl"
bak = "training_data/train.jsonl.bak.augment"
shutil.copy2(src, bak)
print(f"backup: {bak}")

added = 0
with open(src, "a", encoding="utf-8") as f:
    for user_msg, assistant_msg in pairs:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
        )
        f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        added += 1

print(f"appended {added} samples to {src}")
print("verify: last line starts with system prompt + first pair:")
with open(src, encoding="utf-8") as f:
    lines = f.readlines()
print(f"  total lines now: {len(lines)}")