# Symbio

**A local agent that fine-tunes itself on your corrections — and rolls back the runs that made it worse.**

Runs on your Mac. No cloud inference, no subscription, no telemetry by default.
What it learns about you goes into weights that only your copy has.

[![GitHub](https://img.shields.io/badge/GitHub-Symbio-black?logo=github)](https://github.com/huyedits/Symbio)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](#license)

<!-- TODO: 60-second clip — correct it, it trains, a bad run rolls back, it passes -->

```bash
curl -fsSL https://raw.githubusercontent.com/huyedits/Symbio/main/setup.sh | bash
```

Needs an Apple silicon Mac (M1 or later), 16 GB of memory recommended, and about 8 GB of free disk.
Setup picks a model that fits your RAM. [Other ways to install ↓](#install)

---

## Why

Correct most agents today and they make the same mistake tomorrow. A longer
system prompt slows every turn and teaches the model nothing.

Symbio turns corrections into training data. When enough pile up, it runs a
small LoRA fine-tune on your machine, checks the result against a regression
battery, and **throws the new weights away if anything got worse**.

```text
you correct it ─► mistake note ─► training sample ─► LoRA fine-tune
                                                          │
          keep the adapter ◄── passes? ◄── golden-set check
                                   │
                                   └─ worse? ─► automatic rollback
```

## What it has measured

Same machine (16 GB M4), held-out tasks, graded by executing the answer or
checking the world it changed — not by the model grading itself. Most are one
run; treat them as results, not benchmarks.

| What | Before | After |
|---|---|---|
| Learning an unfamiliar CLI from its own attempts ([curve](#the-curve-51-checkpoints-held-out-battery-single-shot)) | 0/13 | **13/13** by iteration 100 |
| Earning its own training data in a new environment, no human samples ([rounds](#fougain)) | 10/45 | **26/45** |
| …six-step tasks in that environment | 0/19 | **7/19** |
| Triggered per-skill adapters vs. no adapter, 60 held-out cases | 29/60 | **59/60** |
| Same 14B, self-repairing 20 injected bugs, harness fixes only | 3/20 | **7/20** |

And what did **not** work, measured just as carefully:

- One always-on adapter for everything scored **51/69 against the base model's 60/69*ne-tune at all. That is why skills get their own adapters.
- Six-step chains could not be learned from a corpus that contained none. Self-teaching earns vocabulary and short combinations, not arbitrary depth.
- Training past the first perfect score changed nothing: 900 of 1,000 iterations were

Details, failures included, are further down. Every bench number names its run.

## What it does

- **Learns from corrections.** It detects them, saves a note, and trains once there a
- **Guards every fine-tune.** It runs a golden-set check and rolls back automatically on any regression.
- **Skills with their own adapters.** Write a procedure in Markdown, and a small worker model trains on it in about 30 seconds.
- **Decides to train by itself**, using its own `train_adapter` tool. This is logged llback.
- **Uses tools:** a sandboxed shell and Python, a browser, desktop control through the accessibility tree, and web research saved to notes.
- **Asks before acting.** Sending, posting and deleting need your approval, on any site.
- **Stays yours.** Notes, sessions, training data and adapters stay on your Mac: in `~/.symbio` for a pip install, or in the checkout for a clone. Upgrading never touches them.
- **Runs where you are:** terminal, desktop window, Telegram, or other apps through MCP and ACP.

## Honest limits

- Apple silicon only. MLX has no CUDA path yet; one is written but has never run.
- A 16 GB Mac fits an 8B comfortably and a 14B tightly. A 14B plus an open browser ca
- Fine-tunes take minutes and real power: about 17 Wh for 150 steps on the 14B.
- Self-training can overfit. The rollback catches regressions it can measure, not ones it cannot.
- The tool sandbox is best effort. It is not a security boundary.
- This is one person's project, and many features are experimental.

---

## Install

### One line

```bash
curl -fsSL https://raw.githubusercontent.com/huyedits/Symbio/main/setup.sh | bash
```

This clones into `~/Symbio` (set `INSTALL_DIR=...` to change it) and runs `./install.sh`. Run it again to update.

### With pip

```bash
pip install "symbio-cli[mlx]"
symb setup
symb chat
```

### To work on it

```bash
git clone https://github.com/huyedits/Symbio
cd Symbio
./install.sh --dev
```

On first launch the setup wizard asks for names, a model and features. Press
Enter at the model step to take one sized to your Mac: Qwen3-14B at 24 GB and
up, Qwen3-8B from 12 GB, and a 3B below that. Run it again any time with `symb setup`.

<details>
<summary>Installer options</summary>

```bash
./install.sh --prefetch-model   # download the model during install
./install.sh --no-browser       # skip the Chromium download
./install.sh --with-vision      # let the assistant see screenshots (mlx-vlm)
./install.sh --dev              # test dependencies
./install.sh --no-shell         # don't drop into the environment afterwards
./install.sh --venv PATH        # custom virtualenv location
```

</details>
