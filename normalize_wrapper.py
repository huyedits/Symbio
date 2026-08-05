import json
import re
import shutil

# Qwen3's chat template always wraps assistant turns with empty reasoning
# delimiters, even when enable_thinking=False (that flag is a no-op for this
# tokenizer). The inference generation prompt ends with the empty block, so
# the model is conditioned to emit the answer right after it. Training data
# must therefore have the same empty block before every assistant answer.
# Earlier I stripped these (clean_think.py) which broke train/inference
# consistency. This script RESTORES the wrapper to every assistant turn that
# lacks it, making the whole dataset consistent with inference.
OPEN = "".join(chr(c) for c in [0x3c, 0x74, 0x68, 0x69, 0x6e, 0x6b, 0x3e])
CLOSE = "".join(chr(c) for c in [0x3c, 0x2f, 0x74, 0x68, 0x69, 0x6e, 0x6b, 0x3e])
WRAPPER = OPEN + "\n\n" + CLOSE + "\n\n"

src = "training_data/train.jsonl"
bak = "training_data/train.jsonl.bak.wrapper"
shutil.copy2(src, bak)
print("backup:", bak)

# Match an assistant turn opening that is NOT already followed by the open delim.
# We insert the wrapper right after "<|im_start|>assistant\n".
asst_open_re = re.compile(r"<\|im_start\|>assistant\n")
# To avoid double-wrapping, only insert if the open delim is not already next.
asst_already_re = re.compile(r"<\|im_start\|>assistant\n" + re.escape(OPEN))

changed = 0
already = 0
out_lines = []
with open(src, encoding="utf-8") as f:
    for line in f:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        o = json.loads(line)
        t = o.get("text", "")
        # Count current state
        n_assist = len(asst_open_re.findall(t))
        n_already = len(asst_already_re.findall(t))
        # Insert wrapper after every assistant opening that lacks it.
        # Do it iteratively to handle multiple assistant turns per sample.
        new_t = asst_already_re.sub("\x00ALREADY\x00", t)  # protect already-wrapped
        new_t = asst_open_re.sub("<|im_start|>assistant\n" + WRAPPER, new_t)
        new_t = new_t.replace("\x00ALREADY\x00", "<|im_start|>assistant\n" + OPEN)
        if new_t != t:
            changed += 1
            o["text"] = new_t
        already += n_already
        out_lines.append(json.dumps(o, ensure_ascii=False))

with open(src, "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines) + "\n")

print("lines changed (wrapper inserted):", changed)
print("assistant turns already wrapped:", already)
print("done")