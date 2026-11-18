# Symbio

**A local agent that fine-tunes itself on your corrections — and rolls back the runs that made it worse.**

Runs on your Mac. No cloud inference, no subscription, no telemetry by default.
What it learns about you goes into weights that only your copy has.

[![GitHub](https://img.shields.io/badge/GitHub-Symbio-black?logo=github)](https://github.com/huyedits/Symbio)
[![Live Demo](https://img.shields.io/badge/%F0%9F%A4%97-Live%20Demo-yellow)](https://huggingface.co/spaces/HuyEdits/symbio-demo)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](#license)

```bash
curl -fsSL https://raw.githubusercontent.com/huyedits/Symbio/main/setup.sh | bash
```

Needs an Apple silicon Mac (M1 or later), 16 GB of memory recommended, and about
8 GB of free disk. Setup picks a model that fits your RAM.
[Other ways to install](#install).

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

The same loop runs over the environment: a failed tool call followed by a
recovery that worked becomes a training example too.

## What it has measured

Same machine (16 GB M4 unless noted), held-out tasks, graded by executing the
answer or checking the world it changed, not by the model grading itself.
Most are one run; treat them as results, not benchmarks.

| What | Before | After |
|---|---|---|
| Learning an unfamiliar CLI from its own attempts ([curve](#the-curve-51-checkpoints-held-out-battery-single-shot)) | 0/13 | **13/13** by iteration 100 |
| Earning its own training data in a new environment, no human samples ([rounds](#four-rounds-earn-train-earn-again)) | 10/45 | **26/45** |
| ...six-step tasks in that environment | 0/19 | **7/19** |
| Triggered per-skill adapters vs. no adapter, 60 held-out cases | 29/60 | **59/60** |
| Same 14B, self-repairing 20 injected bugs, harness fixes only | 3/20 | **7/20** |
| Tool calling, BFCL n=120 | - | **94-96%** |

And what did **not** work, measured just as carefully:

- One always-on adapter for everything scored **51/69 against the base model's
  60/69**. It made things worse, and did not justify a separate fine-tune at
  all. That is why skills get their own adapters.
- Six-step chains could not be learned from a corpus that contained none.
  Self-teaching earns vocabulary and short combinations, not arbitrary depth.
- Training past the first perfect score changed nothing: 900 of 1,000
  iterations were wasted compute.
- A harder 19-task battery peaked at 16/19 and **wobbled** (15, then 8, then 16
  across three checkpoints). A single checkpoint is not a measurement.

Details, failures included, are further down. The harnesses live in `bench/`;
a number whose run output is not in the tree is still one run, not a benchmark.

## What it does

- **Learns from corrections.** Detects them, saves a note, and trains once
  there are enough.
- **Guards every fine-tune.** Runs a golden-set check and rolls back
  automatically on any regression.
- **Skills with their own adapters.** Write a procedure in Markdown, and a
  small worker model trains on it.
- **Decides to train by itself**, using its own `train_adapter` tool. This is
  logged, and the golden set still rolls it back.
- **Uses tools:** a sandboxed shell and Python, a browser, desktop control
  through the accessibility tree, files, notes, web research, scheduled jobs
  and saved scripts.
- **Asks before acting.** Sending, posting and deleting need your approval, on
  any site.
- **Stays yours.** Notes, sessions, training data and adapters stay on your
  Mac: in `~/.symbio` for a pip install, or in the checkout for a clone.
  Upgrading never touches them.
- **Runs where you are:** terminal, panelled TUI, desktop window, Telegram, or
  other apps through MCP and ACP.

## Honest limits

- Apple silicon only. There is a CUDA backend in the tree, but it has never
  been executed on an NVIDIA or AMD device.
- A 16 GB Mac fits an 8B comfortably and a 14B tightly. A 14B plus an open
  browser can run out of memory.
- Fine-tunes take minutes and real power.
- Self-training can overfit. The rollback catches regressions it can measure,
  not ones it cannot.
- The tool sandbox is best effort. It is not a security boundary.
- Benchmarks are one-person, mostly one-run measurements. Read the direction,
  not the decimals.
- This is one person's project, and many features are experimental.

---

## Install

### One line

```bash
curl -fsSL https://raw.githubusercontent.com/huyedits/Symbio/main/setup.sh | bash
```

This clones into `~/Symbio` (set `INSTALL_DIR=...` to change it) and runs
`./install.sh`. Run it again to update.

### With pip

```bash
pip install "symbio-cli[mlx]"
symb setup
symb chat
```

The `[mlx]` extra is the inference and training engine. It is macOS/Apple
silicon only and deliberately optional, because `mlx` publishes no sdist and a
hard dependency would make `pip install` fail during resolution elsewhere.

### To work on it

```bash
git clone https://github.com/huyedits/Symbio
cd Symbio
./install.sh --dev
```

On first launch the setup wizard asks for names, a model and features. Press
Enter at the model step to take one sized to your Mac: Qwen3-14B at 24 GB and
up, Qwen3-8B from 12 GB, and a 3B below that. Run it again any time with
`symb setup`.

<details>
<summary>Installer options</summary>

```bash
./install.sh --prefetch-model   # download the model during install
./install.sh --no-browser       # skip the Chromium download
./install.sh --with-vision      # let the assistant see screenshots (mlx-vlm)
./install.sh --with-native      # include experimental native extras
./install.sh --dev              # test dependencies
./install.sh --no-shell         # don't drop into the environment afterwards
./install.sh --venv PATH        # custom virtualenv location
```

</details>

---

## What is actually in the repo

The package is about 75k lines across `symbio/`, `symbio_desktop/`,
`symbio_tui/`, `symbio_pet/` and `symbio_ane/`. The parts that matter:

| Path | What it does |
|---|---|
| `symbio/app/chat.py` | The interactive agent loop, prompt cache and model lifecycle |
| `symbio/app/learn.py` | Correction detection, mistake notes and learning batches |
| `symbio/app/training.py` | LoRA runs, corpus checks, preflight, early stop and backups |
| `symbio/app/golden.py` | Golden-set scoring, regression detection and remedy samples |
| `symbio/app/skills.py`, `skill_eval.py` | Skill notes, worker adapters and three-way evaluation |
| `symbio/app/tooling.py`, `tool_docs.py` | Tool catalog, prompt index and per-tool Markdown files |
| `symbio/app/dispatch.py` | Mixture-of-Agents workers and their guarded training |
| `symbio/app/memory.py`, `symbio/store.py` | Notes, durable memory and SQLite/FTS session search |
| `symbio/app/sandbox.py`, `computer.py` | Shell/Python execution, Playwright browser, desktop input |
| `symbio/app/daemon.py`, `cli.py` | Resident model daemon, `symb` command line, `symb watch` |
| `symbio_desktop/` | Local web window, native WKWebView, ACP, MCP bridge, OpenAI-compatible API |
| `symbio_tui/`, `symbio_pet/` | Panelled terminal UI and the desktop cat |
| `symbio_ane/` | Optional Swift helper for OCR and on-device vision |
| `tools/*.md` | One Markdown file per tool; the prompt carries the index |
| `bench/` | Research harnesses and recorded results |
| `tests/`, `tests_mcp/` | About 2,700 pytest tests |

One catalog is advertised to every loop. The chat loop, daemon and Telegram use
its dispatcher directly; `AIAgent` (the standalone loop) bridges any tool it
does not own to the same code. A parity test keeps a tool from being advertised
and unrunnable.

Runtime state lives in one workspace: `config.json`, `prompt.md`, `notes/`,
`sessions/`, `training_data/`, `adapters/`, `logs/` and the browser profile.
A clone uses the clone as its workspace; a pip install uses `~/.symbio` unless
`SYMBIO_HOME` says otherwise.

---

## How it learns

```text
1. You use Symbio
2. It makes a mistake
3. You correct it
4. The correction is detected
5. The mistake becomes training data
6. Enough examples accumulate
7. A LoRA fine-tune runs
8. The golden set grades it
9. Kept, or rolled back
```

Corrections are classified as **knowledge** (a wrong fact) or **reflex** (the
wrong shape: a nonexistent tool, a guessed argument). Knowledge wants ordinary
weight over the whole corpus; reflex wants the exact form repeated harder.
A batch picks the dials, and one batch is still one LoRA pass.

The threshold is not a constant. It scales with the corpus, because every
sample already in `train.jsonl` competes with the new ones: five more samples
are most of an epoch against 50 and a rounding error against 882. It rises by
log, not by multiple, and severe mistakes pull it back down. It is clamped to
2..20; `learn.mistake_threshold` sets the base and
`learn.scale_threshold_with_corpus` turns the scaling off.

The `/learn` command still triggers learning from the previous correction
manually.

### Learning from tool mistakes

A failed call followed by a recovery that worked is training data too. Symbio
treats environment failures (`No results found.`, `Timed out`, `Browser is not
open.`, an empty accessibility tree, a click that changed nothing) as failures,
not answers. The turn keeps going, and the recovery it finds is captured.

Only a confirmed successful recovery is kept.

### Skills

```bash
symb skill new "Fix wifi"
symb skill list
symb skill eval "Fix wifi"
```

A skill starts as `notes/skills/<name>.md`, with a hidden health log beside it.
After enough examples, a dedicated worker adapter trains under
`adapters/workers/<role>/`. Skill evaluation compares three conditions:

| Condition | Procedure in prompt? | Measures |
|---|---|---|
| `base` | No | What the base model already knows |
| `prompted` | Yes | The "just prompt it" baseline |
| `adapter` | No | What the LoRA adapter learned |

The adapter is not given the procedure. A high score demonstrates recall of the
trained procedure; it does not prove general intelligence.

---

## Training safety

A loop that trains itself needs a way to notice it has got worse, because the
loss will not tell it. Every check below is a refusal the run can lose to, not
a warning it prints.

**Before the run.** Broken tool-call samples are dropped, unparseable rows are
removed and counted, prompt masking is checked against the model's own chat
template, and `lora.keys` is checked against the module tree. A memory
preflight refuses a run the machine cannot hold; accepting one is not a failed
run, it is a Jetsam kill. The preflight also reads swap and power state and
tries cheap reliefs (shrink the window to the corpus, then gradient
checkpointing) before refusing with the cheapest recipe named.

**The baseline.** The golden battery scores identity, tool-tag formatting and
the contracts that must not break against the current model before anything is
trained. Without a before, "it still passes" is not evidence.

**During the run.** The early-stop monitor keeps the best validation loss. At
patience it kills the trainer and restores the best checkpoint. An implausible
loss (nan, or a number no real run produces) aborts instead of being recorded
as an improvement.

**After the run.** The battery runs again. A case that passed before and fails
now is a regression, and the backup adapter is restored: the fine-tune is
discarded, not shipped. Before rolling back it tries once more with extra
iterations and remedy samples for the specific failing cases. A wildcard check
then asks about subjects absent from the corpus, where a narrow fine-tune shows
up as confident nonsense.

**What the loss cannot say.** The weight-delta report compares the new adapter
against the pre-train backup, per tensor and overall. A run whose LoRA targets
matched nothing produces a respectable curve and an adapter identical to the
one before it; movement is the only thing that distinguishes the two, and
"no previous adapter" and "the shape changed" are reported as unknown rather
than as zero.

**Around all of it.** Adapters are sealed and verified at load time against a
Merkle root, so weights that changed since sealing are refused. In-flight runs
are journalled with a pid and a boot id, so an OOM kill leaves a resumable
record instead of a half-written adapter. Worker adapters get the same
treatment against their own golden sets, one role at a time.

---

## The interfaces

- **Terminal.** `symb` or `symbio` starts chat. `symb tui` uses the panelled
  interface; plain scrollback remains the fallback.
- **Desktop window.** `symbio-desktop` opens a chat window in the browser. No
  model has to be running first: it starts `symb daemon` and waits. `--window`
  opens a native WKWebView instead.
- **Daemon and watch.** `symb daemon start` loads the model once.
  `symb watch` keeps it loaded, restarts it if it dies, and runs due cron jobs
  and granted tasks.
- **Telegram.** `symb gateway start`, with allowed chat IDs configured.
  Dangerous actions use inline approval.
- **ACP.** `symb acp` is an Agent Client Protocol agent for Zed, the VS Code
  and JetBrains plugins, Toad and other hosts.
- **MCP.** `symb connect claude-desktop` and `symb connect hermes` register the
  MCP bridge, giving those apps `ask_symbio` and `symbio_status`.
- **OpenAI-compatible API.** `symb openai` serves the local agent on
  `127.0.0.1`.
- **Desktop control.** `see_screen` reads the frontmost window through the
  macOS accessibility tree; `desktop_click`, `desktop_type`, `desktop_press`,
  `desktop_scroll`, `desktop_drag`, `desktop_move`, `desktop_wait` and
  `open_app` act on it. Vision is the fallback for windows that draw their own
  interface.
- **Its own screen.** `symb desk on` gives Symbio a virtual display; every
  desktop tool works there instead of on yours. `symb desk peek` shows a
  picture, `symb desk off` returns the tools to your screen.
- **The pet.** `symb pet` puts a cat on your desktop and draws the fine-tune:
  the cat is the base model, the collar charm is the adapter, the tail is the
  loss, and the golden gate keeps or swats the charm. Requires the `[pet]`
  extra.

---

## Safety and permissions

What Symbio may do without asking is set per **kind of action**, in plain
language: *Post or send*, *Run commands and code*, *Change files*, *Use your
desktop*, *Browse the web*, *Forget things*, *Train itself*, *Change its
settings*, *Schedule work*. Each is **Allow**, **If risky**, **Ask** or
**Never**.

- A click is judged by what it lands on. A button that sends the text in a box
  is *Post or send* on any site, whichever tool did it.
- The approval card shows the model's sentence for the action and the exact
  post or command from the harness. If the words about to go out are not the
  ones you quoted, the card says so.
- A denied action is not retried through another tool.
- The sandbox reduces accidental damage. Shell and Python run with your user's
  privileges; it is not a security boundary.
- Telegram tokens and other secrets belong in the environment or config, not
  source control.

---

## Tests and benchmarks

```bash
venv/bin/python -m pytest -q          # about 2,700 tests
venv/bin/python tests/smoke_test.py   # the canonical integration check
```

The whole suite lives in `tests/`, with MCP tests in `tests_mcp/`. End-to-end
tests load the configured MLX model and may run short LoRA updates.

The benchmark harnesses live in `bench/`: the CLI learning curve, the reef
environment, self-repair and self-play, skill evaluation, BFCL tool calling,
activation recording and neuron growing, ARC, paged MoE and ARC runners. Some
run outputs are local-only; the README's numbers name the run they came from.
The grading harnesses validate themselves before grading a model: reference
answers must score near 100% and degenerate answers near 0%.

### The curve: 51 checkpoints, held-out battery, single-shot

| iteration | 0 | 20 | 40 | 60 | 80 | **100** | 140 | 160-1000 |
| --------- | -: | -: | -: | -: | -: | ------: | --: | -------: |
| score | 0/13 | 3/13 | 9/13 | 11/13 | 12/13 | **13/13** | 12/13 | 13/13 x43 |

**Perfect score at iteration 100 of 1000.** The other 900 iterations, 90% of
the compute, changed nothing. Training cost 1.5 s per iteration at 8.4 GB peak.

A later 19-task battery peaked at 16/19 and wobbled. What was left at the peak
was not syntax: the model could emit the right token and then not stop
(`prod` became `prodd`, `--identity` became `--identity-name`). That is a
sampling/emitter failure, not something more training data reaches.

### Four rounds: earn, train, earn again

| | trained on | solved | depth 6 |
| --- | --- | ---: | ---: |
| 1. base model | - | 10/45 | **0/19** |
| 2. first adapter | its own 10 samples | 16/45 | **0/19** |
| 3. more runway | same 10 samples, 12 rounds | 19/45 | **3/19** |
| 4. chains in the corpus | 19 samples including 3 six-step | 26/45 | 7/19 |
| 5. ten tasks held out | 21 samples, none held out | 21/45 | 5/19 |

Round 4 scores higher and is the weaker number: five scored tasks had their own
answers in its training data. Round 5 is the honest one. On tasks never in any
corpus, the base model solves 2/10 and the self-taught one solves 4/10; the
first six-step solve on a reserved task lands there. Ten reserved tasks and four
reserved chains is a small sample; every cell is one run.

### The lesson that cost the most

The first benchmark mimicked the real AWS CLI and the model scored 13/13 before
training. It was reciting pretraining. A benchmark a model already knows
measures recall and reports it as learning.

The second lesson was the harness, not the model: an eval that attached only
`q_proj`/`v_proj` loaded 32 of 112 tensors and `load_weights(strict=False)`
dropped the rest in silence. A silent partial load is indistinguishable from a
model that learned nothing. The harness now reads the run's own
`adapter_config.json` and refuses to score a checkpoint whose tensors have
nowhere to land.

---

## Configuration

Configuration lives in the workspace as `config.json`.

```bash
symb config
symb config get agent.temperature
symb config set agent.temperature 0.7
```

Some defaults that matter:

| Key | Default | Purpose |
|---|---:|---|
| `agent.max_tool_rounds` | `15` | Tool rounds per turn |
| `agent.tool_catalog` | `index` | Prompt carries the tool index; `full` inlines every schema |
| `lora.rank` | `8` | LoRA rank |
| `lora.num_layers` | `8` | Adapter layers, counted from the output end |
| `lora.iters` | `300` | Floor for a full retrain; scaled to the corpus |
| `lora.max_seq_length` | `3072` | Training window |
| `learn.auto` | `true` | Detect corrections automatically |
| `learn.auto_train` | `true` | Train at the threshold |
| `learn.mistake_threshold` | `5` | Base corrections before auto-training |
| `learn.golden_rollback_on_regression` | `true` | Revert a worse adapter |
| `dispatch.enabled` | `false` | Worker delegation |
| `telemetry.enabled` | `false` | Anonymous telemetry; opt-in only |

---

## Security

Symbio runs locally, but **local does not mean automatically safe**.

- Review untrusted code before executing it.
- Do not give Symbio access to files you would not give a local program.
- Pay attention to permission prompts.
- A denied action is not retried through another tool.
- Keep `SYMBIO_TELEGRAM_TOKEN` and other secrets out of source control.

The environment variable `SYMBIO_TELEGRAM_TOKEN` takes precedence over the
token in `config.json`.

---

## Contributing

Contributions are welcome: non-Apple hardware support, model/backend
integrations, training performance, memory optimization, evaluation
methodology, browser automation, worker types, tests and security hardening.
See [`CONTRIBUTING.md`](CONTRIBUTING.md).

If you find a bug, include your Mac model and generation, unified memory, model
preset, relevant configuration, logs and reproduction steps.

## License

Apache 2.0.

If Symbio is useful, a star helps other people find it.
