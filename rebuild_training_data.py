"""Rebuild train.jsonl for thinking-ON training that actually fits max_seq_length.

Root cause of every failed retrain: the real system prompt is ~3899 tokens
(persona 1939 + tools JSON 2257), but max_seq_length was 512. Truncation cut
from the end, so the assistant tool tag -- the actual training target -- was
thrown away in 100% of tool samples. The adapter was learning to predict the
first paragraph of the system prompt, nothing else.

This script fixes that by:
  1. Using a CONCISE training-only system prompt (~150 tokens) with identity
     + tool-tag format. (Inference keeps the full prompt; the adapter learns
     the intent->reason->tag behavior, which generalizes.)
  2. Exploding multi-turn samples into single-turn (user, assistant) pairs so
     every sample is short and the assistant target is never truncated.
  3. Adding a real (short) reasoning block before tool-tag answers, so the
     data matches enable_thinking=True inference (model reasons then answers).
  4. Re-rendering with apply_chat_template(enable_thinking=True) so the think
     delimiters come from the Qwen3 template itself (consistent with inference).
  5. Keeping only samples that fit comfortably under max_seq_length.
"""
import json
import re
import shutil
from mlx_lm.utils import load_tokenizer

OPEN = "".join(chr(c) for c in [0x3c, 0x74, 0x68, 0x69, 0x6e, 0x6b, 0x3e])
CLOSE = "".join(chr(c) for c in [0x3c, 0x2f, 0x74, 0x68, 0x69, 0x6e, 0x6b, 0x3e])

MODEL = "mlx-community/Qwen3-8B-3bit"
ANAME, UNAME = "Caine", "Huy"
MAX_KEEP_TOKENS = 700  # keep samples that fit; max_seq_length will be set to 768

CONCISE_SYSTEM = (
    f"You are {ANAME}, a helpful personal AI assistant. Your user is {UNAME}.\n"
    "You can act by emitting a tool tag in your reply. Examples:\n"
    "<cmd>open -a 'Google Chrome' 'https://apple.com'</cmd> - open a site in Chrome\n"
    "<cmd>df -h</cmd> - run a shell command\n"
    "<browse>https://url</browse> - open a page in your controllable browser\n"
    "<search>current weather Sydney</search> - web search\n"
    "<note title=\"Topic\">fact</note> - save a note\n"
    "<py>print(2+2)</py> - run Python\n"
    "Use ONE tool tag when the user wants an action; reply in prose with no tag "
    "for chat, greetings, or questions. Keep replies concise."
)

# ---------------------------------------------------------------------------
# Parse a templated string back into [{role, content}, ...].
# Template format: <|im_start|>role\n content <|im_end|>\n
_MSG_RE = re.compile(r"<\|im_start\|>(system|user|assistant|tool)\n(.*?)<\|im_end\|>", re.S)


def parse_messages(text):
    msgs = []
    for m in _MSG_RE.finditer(text):
        role = m.group(1)
        content = m.group(2)
        # Strip any think block the template may have inserted, keep pure answer.
        if role == "assistant":
            # remove a leading empty/real think block
            content = re.sub(
                re.escape(OPEN) + r"[^\n]*\n.*?" + re.escape(CLOSE) + r"\n*", "", content, flags=re.S
            )
            content = content.lstrip("\n")
        msgs.append({"role": role, "content": content})
    return msgs


# ---------------------------------------------------------------------------
# Synthesize a short reasoning line from a tool-tag answer.
def reasoning_for(answer):
    a = answer.strip()
    # <cmd>...</cmd>
    m = re.search(r"<cmd>(.*?)</cmd>", a, re.S)
    if m:
        inner = m.group(1).strip()
        if inner.startswith("open ") or inner.startswith("open -a"):
            return "The user wants to open something in an app. I'll use the open command."
        return f"The user wants me to run a command. I'll run: {inner[:80]}."
    m = re.search(r"<browse>(.*?)</browse>", a, re.S)
    if m:
        return "The user wants controllable browser access. I'll open the page with the browse tool."
    m = re.search(r"<search>(.*?)</search>", a, re.S)
    if m:
        return f"The user wants a web search. I'll search for: {m.group(1).strip()[:80]}."
    m = re.search(r"<note\b[^>]*>(.*?)</note>", a, re.S)
    if m:
        return "The user wants me to remember something. I'll save a note."
    if re.search(r"<py\b", a):
        return "The user wants a computation. I'll run a short Python snippet."
    if re.search(r"<memory\b|<profile\b", a):
        return "The user wants me to store a fact. I'll append it to memory."
    if "<read>" in a:
        return "The user wants a page's content. I'll fetch it."
    # Hermes-style JSON tool call
    m = re.search(r'\{"name"\s*:\s*"([^"]+)"', a)
    if m:
        return f"The user wants an action. I'll call the {m.group(1)} tool."
    return None


def has_tool(answer):
    return bool(re.search(
        r"<cmd>|<browse>|<search>|<note\b|<py\b|<memory\b|<profile\b|<read>|"
        r'<click>|<type\b|<scroll\b|<press\b|<browser_close\b|\{"name"\s*:',
        answer))


# ---------------------------------------------------------------------------
def main():
    tok = load_tokenizer(MODEL)
    src = "training_data/train.jsonl"
    bak = "training_data/train.jsonl.bak.rebuild"
    shutil.copy2(src, bak)
    print("backup:", bak)

    kept = 0
    skipped_long = 0
    tool_pairs = 0
    chat_pairs = 0
    out_lines = []

    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            text = json.loads(line)["text"]
            msgs = parse_messages(text)
            # Build single-turn (user, assistant) pairs from adjacent turns.
            for i in range(len(msgs) - 1):
                if msgs[i]["role"] != "user" or msgs[i + 1]["role"] != "assistant":
                    continue
                user_msg = msgs[i]["content"].strip()
                answer = msgs[i + 1]["content"].strip()
                if not user_msg or not answer:
                    continue
                is_tool = has_tool(answer)
                reasoning = reasoning_for(answer) if is_tool else None
                # Build assistant message content WITH the reasoning block so the
                # template's enable_thinking branch re-renders it correctly.
                if reasoning:
                    asst_content = f"{OPEN}\n{reasoning}\n{CLOSE}\n\n{answer}"
                else:
                    asst_content = answer  # template inserts empty think block
                messages = [
                    {"role": "system", "content": CONCISE_SYSTEM},
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": asst_content},
                ]
                rendered = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False,
                    enable_thinking=True,
                )
                n = len(tok.encode(rendered))
                if n > MAX_KEEP_TOKENS:
                    skipped_long += 1
                    continue
                out_lines.append(json.dumps({"text": rendered}, ensure_ascii=False))
                kept += 1
                if is_tool:
                    tool_pairs += 1
                else:
                    chat_pairs += 1

    with open(src, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")

    print(f"kept samples: {kept}  (tool: {tool_pairs}, chat: {chat_pairs})")
    print(f"skipped (> {MAX_KEEP_TOKENS} tok): {skipped_long}")
    print("done")


if __name__ == "__main__":
    main()