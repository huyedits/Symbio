"""Does a model get better at repairing its own code by training on its own
passing fixes? Rounds on held-out bugs, each step skipped once its output
exists, so a crash resumes where it stopped.

  base1, base2   no adapter, held-out bugs (two runs: the run-to-run noise)
  gen1           the model repairs the TRAIN bugs; fixes that pass -> data1
  r1             LoRA on data1, held-out bugs
  ctrl           LoRA on as many GOLD fixes (the exact reverse mutation, not its work)
  gen2           the r1 model repairs the train bugs again -> data2 = data1 + new fixes
  r2             LoRA on data2, held-out bugs

Real self-improvement: r1 above both baselines by more than their gap, r2
above r1, both above ctrl.

    MODEL=... GEN=mlx|hf CLONE=<repo copy> OUT=<dir> python bench/self_repair/rsi.py

GEN=hf trains with symbio.cuda_lora (CUDA or ROCm); LOAD_BITS=16 on a big card.
"""
import json, os, random, re, shutil, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL = os.environ["MODEL"]
GEN = os.environ.get("GEN", "mlx")
CLONE = Path(os.environ["CLONE"]).resolve()
OUT = Path(os.environ.get("OUT", HERE / "runs")).resolve()
OUT.mkdir(parents=True, exist_ok=True)
TRAIN, HELD = HERE / "train_bugs.json", HERE / "heldout_bugs.json"
LAYERS = int(os.environ.get("RSI_LAYERS", 4))
# 1e-4 diverged (loss 7-14 by iter 50, 1/40 fixed); 2e-5 with warmup + cosine decay held
LR = float(os.environ.get("RSI_LR", 2e-5))
LOAD_BITS = os.environ.get("LOAD_BITS", "16" if GEN == "hf" else "4")
PY = sys.executable


def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(OUT / "rsi.log", "a") as f:
        f.write(line + "\n")


def bench(tag, bugs, adapter=None):
    rec = OUT / f"{tag}.repair.json"
    if (OUT / f"{tag}.done").exists():
        return json.loads(rec.read_text())
    env = dict(os.environ, SYMBIO_HOME=str(CLONE), GEN=GEN, LOAD_BITS=LOAD_BITS)
    env.pop("ADAPTER", None)
    if adapter:
        env["ADAPTER"] = str(adapter)
    log(f"bench {tag} on {bugs.name} adapter={adapter}")
    with open(OUT / f"{tag}.log", "w") as f:
        subprocess.run([PY, str(HERE / "drive.py"), MODEL, tag, str(bugs), str(OUT)],
                       cwd=CLONE, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    d = json.loads(rec.read_text())
    log(f"RESULT {tag}: {sum(r['fixed'] for r in d)}/{len(d)} fixed")
    (OUT / f"{tag}.done").write_text("ok")
    return d


def self_samples(records):
    """The prompt and reply of the cycle that fixed the bug — its own work, kept only when the tests passed."""
    return [{"key": f"{r['src']}:{r['i']}",
             "messages": [{"role": "user", "content": r["prompts"][-1]},
                          {"role": "assistant", "content": r["replies"][-1]}]}
            for r in records if r.get("fixed") and r.get("prompts") and r.get("replies")]


def gold_samples(records, bugs, n):
    """The same first prompts, answered with the exact reverse mutation it did not write."""
    pool = [r for r in records if r.get("prompts")]
    random.Random(5).shuffle(pool)
    out = []
    for r in pool[:n]:
        b = bugs[r["i"]]
        reply = f"<<<< OLD\n{b['new'].rstrip(chr(10))}\n====\n{b['old'].rstrip(chr(10))}\n>>>> NEW"
        out.append({"key": f"gold:{r['src']}:{r['i']}",
                    "messages": [{"role": "user", "content": r["prompts"][0]},
                                 {"role": "assistant", "content": reply}]})
    return out


def train(name, samples):
    ad = OUT / f"adapter_{name}"
    if any(ad.glob("adapter*.safetensors")):
        return ad
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    texts = [tok.apply_chat_template(s["messages"], tokenize=False) for s in samples]
    lens = [len(tok.encode(t, add_special_tokens=False)) for t in texts]
    keep = [i for i, n in enumerate(lens) if n <= 8192]
    seq = min(8192, max(lens) + 64)
    data = OUT / f"data_{name}"
    data.mkdir(exist_ok=True)
    # mlx_lm reads chat rows; symbio.cuda_lora reads formatted text and masks the prompt itself
    rows = [json.dumps({"messages": samples[i]["messages"]} if GEN == "mlx" else {"text": texts[i]})
            for i in keep]
    (data / "train.jsonl").write_text("\n".join(rows) + "\n")
    (data / "valid.jsonl").write_text("\n".join(rows[:max(1, len(rows) // 5)]) + "\n")
    iters = max(20, 2 * len(keep))
    log(f"train {name}: {len(keep)} samples (dropped {len(samples) - len(keep)} >8192 tok), "
        f"max_seq {seq}, iters {iters}, layers {LAYERS}")
    common = ["--model", MODEL, "--train", "--data", str(data), "--iters", str(iters),
              "--batch-size", "1", "--num-layers", str(LAYERS), "--max-seq-length", str(seq),
              "--learning-rate", str(LR), "--adapter-path", str(ad)]
    if GEN == "mlx":
        warm = max(1, iters // 6)
        cfg = OUT / f"train_{name}.yaml"  # JSON is valid YAML
        cfg.write_text(json.dumps({"lr_schedule": {"name": "cosine_decay", "warmup": warm,
                                                   "warmup_init": LR / 100,
                                                   "arguments": [LR, iters - warm, LR / 10]}}))
        cmd = [PY, "-m", "mlx_lm", "lora", *common, "-c", str(cfg), "--mask-prompt",
               "--grad-checkpoint", "--steps-per-report", "10", "--steps-per-eval", "10",
               "--val-batches", "3", "--save-every", "10"]
    else:
        cmd = [PY, "-m", "symbio.cuda_lora", *common, "--steps-per-eval", "100000",
               "--load-in-bits", LOAD_BITS, "--grad-checkpoint", "true"]
    with open(OUT / f"train_{name}.log", "w") as f:
        subprocess.run(cmd, cwd=CLONE, env=dict(os.environ, SYMBIO_HOME=str(CLONE)),
                       stdout=f, stderr=subprocess.STDOUT, check=True)
    text = (OUT / f"train_{name}.log").read_text().replace("\r", "\n")
    vals = {int(m.group(1)): float(m.group(2)) for m in re.finditer(r"Iter (\d+): Val loss ([\d.naif]+)", text)}
    saved = {it: v for it, v in vals.items() if (ad / f"{it:07d}_adapters.safetensors").exists()}
    if not vals:
        tail = text.splitlines()[-8:]
        log(f"trained {name}: " + " | ".join(l for l in tail if "loss" in l.lower())[-300:])
    else:
        log(f"trained {name}: val " + " ".join(f"{it}:{v:.3f}" for it, v in sorted(vals.items())))
    if saved:
        # the checkpoint with the lowest val loss, not the last: a late spike is not kept
        best = min(saved, key=lambda it: (saved[it] != saved[it], saved[it]))
        if best != iters:
            shutil.copy2(ad / f"{best:07d}_adapters.safetensors", ad / "adapters.safetensors")
        log(f"{name}: using iter {best} (val {saved[best]:.3f}); final iter {iters} val {vals.get(iters)}")
    return ad


if __name__ == "__main__":
    train_bugs = json.loads(TRAIN.read_text())
    bench("rsi_base1", HELD)
    bench("rsi_base2", HELD)
    gen1 = bench("rsi_gen1", TRAIN)
    data1 = self_samples(gen1)
    log(f"data1: {len(data1)} self-made fixes")
    a1 = train("r1", data1)
    bench("rsi_r1", HELD, a1)
    actrl = train("ctrl", gold_samples(gen1, train_bugs, len(data1)))
    bench("rsi_ctrl", HELD, actrl)
    gen2 = bench("rsi_gen2", TRAIN, a1)
    seen = {s["key"] for s in data1}
    data2 = data1 + [s for s in self_samples(gen2) if s["key"] not in seen]
    log(f"data2: {len(data2)} self-made fixes ({len(data2) - len(data1)} new from the round-1 model)")
    a2 = train("r2", data2)
    bench("rsi_r2", HELD, a2)
    log("DONE")
