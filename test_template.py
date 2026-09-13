from mlx_lm.utils import load_tokenizer

# Build the Qwen3 reasoning delimiters from codepoints so this file contains
# none of those characters literally (which confuses some tooling).
OPEN = "".join(chr(c) for c in [0x3c, 0x74, 0x68, 0x69, 0x6e, 0x6b, 0x3e])
CLOSE = "".join(chr(c) for c in [0x3c, 0x2f, 0x74, 0x68, 0x69, 0x6e, 0x6b, 0x3e])

tok = load_tokenizer("mlx-community/Qwen3-8B-3bit")
msgs = [
    {"role": "user", "content": "open chrome to apple website"},
    {"role": "assistant", "content": "<cmd>open -a 'Google Chrome' 'https://www.apple.com'</cmd>\nOpening Apple.com in Chrome."},
]

for et in (True, False):
    for agp in (False, True):
        try:
            t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=agp, enable_thinking=et)
        except TypeError:
            t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=agp)
        print(f"=== enable_thinking={et} add_generation_prompt={agp} ===")
        print("open-delim present:", OPEN in t, "| close-delim present:", CLOSE in t)
        print("TAIL:", repr(t[-300:]))
        print()