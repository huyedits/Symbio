import json, re, shutil, sys

# Build Qwen3 think tags from codepoints so nothing in this file depends on
# how an editor/terminal renders the literal tag characters.
OPEN = ''.join(chr(c) for c in [0x3c, 0x74, 0x68, 0x69, 0x6e, 0x6b, 0x3e])      # <think>
CLOSE = ''.join(chr(c) for c in [0x3c, 0x2f, 0x74, 0x68, 0x69, 0x6e, 0x6b, 0x3e])  # </think>

src = 'training_data/train.jsonl'
bak = 'training_data/train.jsonl.bak.thinkfix'

shutil.copy2(src, bak)
print('backup:', bak)

# Strip <think>...</think> blocks (non-greedy, DOTALL) and any stray tags.
block_re = re.compile(re.escape(OPEN) + r'\b[^>]*>.*?' + re.escape(CLOSE), re.DOTALL | re.IGNORECASE)
open_re  = re.compile(re.escape(OPEN) + r'\b[^>]*>', re.IGNORECASE)
close_re = re.compile(re.escape(CLOSE), re.IGNORECASE)
nl_re    = re.compile(r'\n{3,}', re.DOTALL)

def count_think_assist(text):
    n = 0
    for m in re.finditer(r'<\|im_start\|>assistant\s*(.*?)<\|im_end\|>', text, re.DOTALL):
        if CLOSE in m.group(1):
            n += 1
    return n

changed_lines = 0
removed_blocks = 0
before_assist = 0
after_assist = 0
sample_before = None
sample_after = None

out_lines = []
with open(src, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.rstrip('\n')
        if not line.strip():
            continue
        o = json.loads(line)
        t = o.get('text', '')
        before_assist += count_think_assist(t)
        n_blocks = len(block_re.findall(t))
        new_t = block_re.sub('', t)
        new_t = open_re.sub('', new_t)
        new_t = close_re.sub('', new_t)
        new_t = nl_re.sub('\n\n', new_t).strip()
        if n_blocks:
            removed_blocks += n_blocks
        if new_t != t:
            changed_lines += 1
            o['text'] = new_t
            if sample_before is None:
                sample_before = t
                sample_after = new_t
        after_assist += count_think_assist(new_t)
        out_lines.append(json.dumps(o, ensure_ascii=False))

with open(src, 'w', encoding='utf-8') as f:
    f.write('\n'.join(out_lines) + '\n')

print('lines changed:', changed_lines)
print('think blocks removed:', removed_blocks)
print('assistant turns with think-close: before =', before_assist, '| after =', after_assist)
if sample_before is not None:
    print('\n--- SAMPLE before ---')
    print(sample_before[:400])
    print('\n--- SAMPLE after ---')
    print(sample_after[:400])