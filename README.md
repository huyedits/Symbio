# Symbio
Your machine, your way.

[![Live Demo on Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20Demo-Hugging%20Face%20Spaces-blue)](https://huggingface.co/spaces/HuyEdits/symbio-demo)

Symbio develops as you tell it what to do in repeat.

**Try the interactive demo** — the agent's real tag parser, self-correction miner, research memory, and RAG retriever running in your browser: https://huggingface.co/spaces/HuyEdits/symbio-demo

Symbio works with a feature called **MOA**, MOA is a mixture of agents. In staid of one big model to fine tune in staid the big model calls for smaller models to act on the task via tool call, the small model executes and if it fails then will return back to the big model to consult then the big model tries until it works then a .md file is made for both sides. IF they make the same mistake over five times or whatever you set it to then the smaller model and big model gets fine tuned, for the smaller model it would be how to execute it and how it did it well and the big model will fine tune its response to the smaller model to efficiently use tokens.
## What it does

- Chat through a local CLI or a Telegram bot.
- Save facts and notes as markdown files in `notes/`.
- Read, write, search, and patch files inside the project directory.
- Run sandboxed shell commands and short Python snippets.
- Check email via IMAP/SMTP (when configured).
- Digest notes into training data and fine-tune a LoRA adapter on the fly.
- Persist every conversation turn to JSONL and an SQLite store.

## Quick start

```bash
# Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install `symbio` / `symb` as system-wide commands.
# Editable install links the current repo so code edits take effect immediately:
pip install -e .

# Start chatting
symbio
# Or the short alias:
#   symb
```

If you prefer an isolated, non-editable install (e.g. with `pipx`):

```bash
pipx install .
# or, from any directory containing this repo:
pipx install /path/to/agi
```

Make sure `~/.local/bin` (or your pip/pipx bin directory) is on your `PATH`.

On first run, Symbio asks for your name and its name. These are saved to `config.json`.
## Why custom?
AI agents tend to forget and also not personalised to the work you want the agent to do, as well as the agent being in the cloud which brings on the costs and privacy risk. This repo helps you have access to a highly aggressive persoanlised model that does not leave your machine unless you ask it to.
## Configuration

Edit `config.json` to change the model, LoRA settings, or agent behavior. You can also use the CLI:

```bash
symb config                    # show full config (bot token redacted)
symb config get agent.temperature
symb config set agent.temperature 0.7
symb config set telegram.allowed_chat_ids '[123456789]'
```

| Key | Default | Note |
|---|---|---|
| `model_name` | `mlx-community/Qwen2.5-3B-Instruct-4bit` | Base MLX model |
| `assistant_name` | `Symbio` | What the assistant calls itself |
| `user_name` | *(asked at first run)* | Your name |
| `agent.max_turns` | `5` | Max tool rounds per user turn |
| `agent.temperature` | `0.1` | Sampling temperature (low for deterministic tool use) |
| `lora.rank` | `8` | LoRA rank (adapter width) |
| `lora.dropout` | `0.1` | LoRA dropout to reduce overfitting |
| `lora.scale` | `5.0` | LoRA adapter scale |
| `lora.num_layers` | `8` | Number of layers to attach adapters to |
| `lora.iters` | `50` | LoRA iterations per `/train` run |
| `lora.max_seq_length` | `2048` | Maximum sequence length during training |
| `lora.learning_rate` | `1e-4` | LoRA learning rate |
| `lora.save_every` | `50` | Checkpoint frequency during training |
| `learn.boost_factor` | `3` | Copies of a correction sample written to training data |
| `learn.batch_train_iters` | `25` | Iterations for threshold-triggered correction training |
| `learn.mistake_threshold` | `5` | Mistake notes collected before auto-training runs |
| `telegram.bot_token` | *(prompted)* | Telegram bot token from @BotFather |
| `telegram.allowed_chat_ids` | `[]` | Chat IDs allowed to use the bot (required) |

## CLI

After installing (`pip install -e .`) the `symbio` and `symb` commands are available. During development you can use the `symb` wrapper script in the repo root.

```bash
symb                    # Start interactive chat
symb chat               # Same as above
symb config             # Open interactive config editor
symb config show        # Print config.json (token redacted)
symb config get <key>   # Print one value, e.g. agent.temperature
symb config set <key> <value>
symb train              # Run LoRA training
symb gateway status     # Check Telegram gateway readiness
symb gateway start      # Start the Telegram bot
symb gateway stop       # Stop a running gateway
```

Legacy `python main.py` flags still work:

```bash
python main.py --telegram
python main.py --train
```

## Telegram bot

Run Symbio as a Telegram bot so you can chat from your phone:

```bash
symb gateway start
# Legacy equivalent: python main.py --telegram
```

Check gateway readiness first:

```bash
symb gateway status
```

On first run you will be prompted for a bot token from [@BotFather](https://t.me/botfather). The token is saved to `config.json`. For better security, set the environment variable `SYMBIO_TELEGRAM_TOKEN` instead; it overrides the config file.

You must add your Telegram chat ID to `telegram.allowed_chat_ids`:

```bash
symb config set telegram.allowed_chat_ids '[123456789]'
```

Send any message to the bot, then copy the chat ID from the refusal message if you haven't set it yet.

Dangerous actions from Telegram — blocked shell commands, new browser domains, Python code, config changes, cron jobs, digest, and training — ask for approval via an inline keyboard before running.

## Slash commands

| Command | Description |
|---|---|
| `/quit` | Exit the chat |
| `/save` | Save the current conversation to training data |
| `/train` | Run LoRA fine-tuning and reload the adapter |
| `/learn` | Manually learn from your last correction (auto-learn is on by default) |
| `/digest` | Convert notes into training samples |
| `/note [title]` | Create a markdown note |
| `/notes` | List saved notes |
| `/status` | Show model, adapter, notes, and session info |
| `/setup` | Change assistant/user names |
| `/model` | List model presets |
| `/model <preset>` | Switch to a named model preset (restart to load) |
| `/run <cmd>` | Run a sandboxed shell command |
| `/forget_last` | Remove the last exchange from history |
| `/prune` | Remove stale adapter checkpoints |

## Learning from corrections

Symbio detects natural corrections automatically and turns them into training data without you typing `/learn`. Instead of training on every single correction, it saves each mistake as a markdown note in `notes/mistakes/` and only fine-tunes once enough notes have accumulated.

Typical flow:
```
You:      What is my name?
Symbio:   Your name is Bob.
You:      No, I'm Alice.
Symbio:   Your name is Alice.
          [System] Correction detected (correction phrase).
          Saved mistake note: 20260715_123456_What_is_my_name.md
          1/5 mistake note(s) collected. Training will run after 4 more correction(s).
```

Symbio will:
1. Detect correction phrases ("No, ...", "Actually ...", "That's wrong", etc.) or an exact repeat of your last question.
2. Extract the original question, the wrong answer, the user's correction, and the corrected answer.
3. Save them as a markdown note in `notes/mistakes/`.
4. When `learn.mistake_threshold` (default 5) notes have accumulated, digest them into `training_data/train.jsonl` and run a short LoRA update (`learn.batch_train_iters`, default 25).
5. Archive the used mistake notes to `notes/mistakes/archive/` and reload the adapter.

The `/learn` command is still available to force a mistake note from the last correction, but it is no longer required.

### Learning from its own tool mistakes

The same mistake-note pipeline also captures a second, fully automatic pattern that needs no user involvement at all: a tool call that fails, immediately followed by one that works. This is exactly the "wrong command, try the right one" pattern already hand-seeded into every install's base training data —

```
You:      Open Chrome.
Symbio:   <cmd>chrome</cmd>
          [Tool: run_command]
          [Observation] Command 'chrome' exited error.
                        Output:
                        Command not found: chrome
Symbio:   'chrome' isn't a command here — trying the native way. <cmd>open -a 'Google Chrome'</cmd>
          [Learn] Tool mistake captured: 20260721_213045_System_observation_Command_chrome.md
```

— except now it's learned from real usage, not just the seed examples. It feeds into the exact same `notes/mistakes/` → threshold → digest → guarded-training pipeline as conversational corrections above, so both count toward the same `learn.mistake_threshold`. Nothing is saved if the model keeps failing without ever finding a working alternative within the turn — only a confirmed fix gets captured.

Tune the behaviour in `config.json`:

| Key | Default | Note |
|---|---|---|
| `learn.enabled` | `true` | Enable correction learning |
| `learn.auto` | `true` | Detect corrections automatically |
| `learn.auto_train` | `true` | Run the fine-tune automatically when the threshold is reached |
| `learn.mistake_threshold` | `5` | Number of mistake notes before a batch fine-tune runs |
| `learn.batch_train_iters` | `25` | LoRA iterations for the threshold-triggered batch update |
| `learn.boost_factor` | `3` | Copies of each correction sample written per mistake note |
| `learn.correction_phrases` | `[...]` | Phrases that trigger correction detection |

## Fine-tuning details

Symbio uses **LoRA** (Low-Rank Adaptation) via Apple's **MLX-LM** framework. The base model weights stay frozen; only small adapter matrices are trained on curated conversation, notes, and corrections. Training is invoked through the official `mlx_lm lora` CLI:

```bash
symb train            # full pass using lora.iters
```

The resulting adapter is saved to `adapters/` and loaded automatically on the next start.

| Setting | Default | What it controls |
|---|---|---|
| `lora.rank` | `8` | Width of the low-rank matrices |
| `lora.num_layers` | `8` | How many transformer layers get adapters |
| `lora.scale` | `5.0` | Adapter output scaling |
| `lora.dropout` | `0.1` | Dropout for regularization |
| `lora.learning_rate` | `1e-4` | Training step size |
| `lora.iters` | `50` | Iterations for `/train` |
| `lora.max_seq_length` | `2048` | Training context length |
| `lora.save_every` | `50` | Checkpoint frequency |

### Golden set: catching a fine-tune that silently breaks things

Every LoRA update (`/train`, the `train_adapter` tool, the auto-training that follows enough corrections, or the end-of-session prompt) is checked against a small, fixed **golden set** — prompts that exercise behavior baked into every install's seed training data: stating its own name, not confusing itself with the user, emitting the right tool tag for code/notes/reminders/search, and not degenerating into repeated phrases. Each check is single-turn and side-effect-free (no tool is actually executed), so it's safe to run automatically.

Symbio grades the golden set before training (the baseline) and again after reloading the new adapter. If a case that passed before now fails, it's a **regression**, and the previous adapter is restored automatically:

```
  [Golden] Regression: 2 case(s) newly failing (run_code_for_math, web_search_unknown).
  [Golden] Rolled back to the previous adapter.
```

Run it manually anytime with `/golden` to see the current pass/fail breakdown without training.

| Key | Default | Note |
|---|---|---|
| `learn.golden_set_enabled` | `true` | Grade every LoRA update against the golden set |
| `learn.golden_rollback_on_regression` | `true` | Automatically restore the previous adapter on a regression |
| `learn.golden_regression_threshold` | `0` | Newly-failing cases allowed before it counts as a regression |
| `learn.golden_max_tokens` | `150` | Max tokens generated per golden-set case |

### Idle-adapter reminders

If a trained adapter exists on disk but the current session isn't using it (most commonly after switching `model_name` to something the adapter isn't compatible with), Symbio tracks how long it's sat unused. Past `learn.adapter_idle_days`, it asks once whether to remove it:

```
  A saved LoRA adapter hasn't been used in 45 day(s) (not loaded with the
  current model). Remove it to free up space? [y/N]:
```

Answering yes deletes it; declining or saying "keep" both just leave it alone and reset the grace period, so the reminder won't repeat until it's been idle that long again. Nothing is ever removed without an explicit yes. Check the current status anytime with `/status`.

| Key | Default | Note |
|---|---|---|
| `learn.adapter_idle_reminder_enabled` | `true` | Ask about removing an adapter that's gone unused |
| `learn.adapter_idle_days` | `30` | Days unused before the reminder fires |
Exmaple:
<img width="1300" height="89" alt="Screenshot 2026-07-23 at 11 23 21 am" src="https://github.com/user-attachments/assets/c4e02593-f527-44dc-9bcb-181f329360ad" />
<img width="272" height="475" alt="Screenshot 2026-07-23 at 11 22 52 am" src="https://github.com/user-attachments/assets/e8e7475a-aac8-455b-b978-3996f1d4d3fd" />


## Mixture of agents: delegating to smaller worker models

One model doing everything — from picking a browser click to answering a factual question — means every micro-decision pays the cost of the headmaster's full system prompt and persona. Symbio can instead hand a bounded sub-task off to a smaller, faster **worker** model, and each worker can be fine-tuned independently on its own narrow task, with its own adapter, separate from the headmaster's.

This is off by default (`dispatch.enabled: false`) — it loads and runs additional models on your machine, a bigger resource commitment than anything else here, so it's opt-in.

### How it works

The headmaster requests delegation the same way it requests any other tool:

```
<delegate role='summarize'>the full text to condense</delegate>
```

or the Hermes form: `<tool_call>{"name": "delegate_task", "arguments": {"role": "summarize", "task": "..."}}</tool_call>`.

`symbio/app/worker_models.json` is the catalog of available workers — model, role, description, rough memory footprint. Ships with two roles:

| Role | What it does |
|---|---|
| `summarize` | Condenses page/document text handed off by the headmaster |
| `browser` | Picks the next click/type/scroll action from the current page text, in a bounded loop, using the same `BrowserSession` the headmaster's own browser tools drive |

Workers load lazily on first use and are evicted LRU-style once `dispatch.max_resident_workers` is exceeded, or after sitting idle past `dispatch.worker_idle_unload_minutes` — sequential by default (one resident worker) to fit alongside the headmaster on a typical machine, but this is a real, working setting: raise `max_resident_workers` if you have the RAM to keep several loaded at once.

### Fine-tuning a worker

Every delegated task's (input, output) pair is recorded as a training sample under that worker's own data directory (`training_data/workers/<role>/`) — real usage builds the corpus. Training a worker reuses the exact golden-set-guarded-rollback machinery the headmaster's own `/train` uses: a small, role-scoped golden set (e.g. "does the browser worker still reply with a known action verb") is checked before and after training, and a regression rolls the worker's adapter back automatically, the same way `_guarded_train` protects the headmaster's. Worker adapters live under `adapters/workers/<role>/`, fully separate from the headmaster's own `adapters/`.

| Key | Default | Note |
|---|---|---|
| `dispatch.enabled` | `false` | Turn on delegation |
| `dispatch.max_resident_workers` | `1` | How many worker models can be loaded at once |
| `dispatch.worker_idle_unload_minutes` | `10` | Unload a worker after this long unused |
| `dispatch.max_worker_rounds` | `4` | Round cap for a multi-step worker task (e.g. browser) |
| `dispatch.worker_golden_set_enabled` | `true` | Golden-check a worker's adapter around training |
| `dispatch.worker_golden_rollback_on_regression` | `true` | Auto-rollback a worker's adapter on regression |

Enabling dispatch also needs `"delegate"` in `tools.enabled_groups` — it's included by default going forward, but an existing `config.json` written before this feature won't have picked it up automatically; add it with `/config set tools.enabled_groups '[...]'` if delegation seems to silently do nothing.

## Dynamic names

### Supported user-name phrasings

- *"My name is Alice."*
- *"Call me Bob."*
- *"You can call me Charlie."*
- *"From now on call me Dana."*
- *"Change my name to Eve."*
- *"I go by Frank."*

### Supported assistant-name phrasings

- *"Call yourself Jarvis."*
- *"I will call you Friday."*
- *"I'm going to call you HAL."*
- *"Change your name to Jeeves."*
- *"Set your name as Alfred."*

> Note: *"Your name is X"* is intentionally **not** treated as an assistant rename because small models often confuse it with the user's name.

## Tool formats

Symbio understands two ways to call tools:

- **Legacy XML tags**:
  - `<note title="User Preference">The user likes coffee.</note>` — save a note
  - `<cmd>ls</cmd>` — run a sandboxed command (legacy, still supported)
  - `<digest />` / `<train />` — digest notes or train

- **Hermes JSON-in-XML** (preferred):
  ```xml
  <tool_call>{"name": "read_file", "arguments": {"path": "config.json"}}</tool_call>
  <tool_call>{"name": "terminal", "arguments": {"cmd": "ls -la"}}</tool_call>
  <tool_call>{"name": "note", "arguments": {"action": "add", "target": "note", "content": "The user likes coffee."}}</tool_call>
  ```

## Security notes

- `terminal` and `execute_code` are best-effort sandboxes. They run with the privileges of the user who started the program and are scoped to the project directory.
- `execute_code` requires the script to import from `symbio_tools` (or the backward-compatible `caine_tools` alias) and blocks known dangerous imports.
- Do not paste untrusted code into the agent without reviewing it first.

## Architecture

The project is organized as a `symbio/` Python package with a thin `main.py` wrapper:

```
.
├── main.py              # Delegates to the modern CLI in symbio/app/cli.py
├── symbio/
│   ├── constants.py     # Paths, DEFAULT_CONFIG
│   ├── app/
│   │   ├── cli.py         # symbio / symb command-line interface
│   │   ├── chat.py        # ChatSession, agent loop, slash commands
│   │   ├── config.py      # Defaults, loading, redaction, token prompt
│   │   ├── training.py    # Training data and LoRA fine-tuning via mlx_lm
│   │   ├── learn.py       # Correction detection and batch learning
│   │   ├── golden.py      # Golden set: regression checks around every LoRA update
│   │   ├── dispatch.py    # MoA: WorkerPool, delegated tasks, worker fine-tuning
│   │   ├── worker_models.json  # Catalog of available worker models/roles
│   │   ├── memory.py      # Notes, memory, profile management
│   │   ├── sandbox.py     # Sandboxed commands and Python execution
│   │   ├── computer.py    # Browser automation helpers
│   │   ├── cron.py        # Scheduled jobs and reminders
│   │   ├── telegram.py    # Telegram bot gateway
│   │   └── tooling.py     # Tag parsing and tool stripping
│   └── utils.py         # Shared helpers
├── rag.py               # Lightweight keyword-based RAG
├── README.md
├── docs/
│   └── adapter-marketplace.md  # Design doc, not yet implemented
├── config.json          # User configuration
├── models.json          # Model presets
├── notes/               # Markdown notes / memory
├── training_data/       # train.jsonl and valid.jsonl (workers/<role>/ for MoA workers)
├── adapters/            # LoRA adapter weights (workers/<role>/ for MoA workers)
├── logs/                # Session logs
├── sessions/            # Session stores
├── screenshots/         # Browser screenshots
└── sandbox/             # Scratch space for code execution
```

## Roadmap / high-priority contributions

We are actively looking for help on:

1. **CUDA port** — MLX is Apple Silicon only. A PyTorch or Transformers backend would let Symbio run on NVIDIA/AMD hardware.
2. **llama.cpp backend** — Support GGUF models through llama.cpp for broader model coverage and lower memory use.
3. **LoRA optimization** — Faster adapter swaps, gradient checkpointing, and memory-efficient training.
4. **Refactoring** — Cleaner separation between inference, tools, training, and storage; better test coverage.
5. **Sparse / quantized adapters** — Experiment with QLoRA, 8-bit/4-bit base models, and sparse LoRA updates.

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, testing, and how to open issues/PRs.


 > future projection <
- [ ] **Add MCP (Model Context Protocol)**
- [x] **Add More Tools** — live browser (`<browse>`/`<click>`/`<type>`/`<scroll>`), `<skill>`, permission-gated sandbox
- [x] **Self correction when hallucinating**
- [x] **Be able to learn new skills on the fly**
- [x] **Remember new info found from web research** — auto-saved as `Learned:` notes, trained in on digest
- [x] **Add Telegram bot** — full tool loop with inline-keyboard approval for dangerous actions
- [x] **Mixture of agents** — headmaster delegates bounded sub-tasks to smaller, independently fine-tunable worker models (`dispatch.enabled`, off by default)
- [ ] **Adapter marketplace** — design doc: [docs/adapter-marketplace.md](docs/adapter-marketplace.md); not yet implemented
- [ ] **Add Other Messaging Platforms**
- [ ] **Prune Old Weights (Future Milestone)**


## Star History

<a href="https://www.star-history.com/?repos=huyedits%2FSymbio&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=huyedits/Symbio&type=date&theme=dark&legend=top-left&sealed_token=BARd7crHixQjaz11nJTAZ7mVM0hzMRPkR0XsWnt0JCDpfGb7UODGYP_v1vWqVZ7oBnYNeBSjSPD41Jz3zptiRq5d4it22dMAG2hzDZp-hqN1WUU71TnCUQzen-QuIt_rS3gQGtX2rxkJBNKMo5q86C2O0Q4om5BuX_2rj91AZGictnTvSaGS7Yb0fayE" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=huyedits/Symbio&type=date&theme=light&legend=top-left&sealed_token=BARd7crHixQjaz11nJTAZ7mVM0hzMRPkR0XsWnt0JCDpfGb7UODGYP_v1vWqVZ7oBnYNeBSjSPD41Jz3zptiRq5d4it22dMAG2hzDZp-hqN1WUU71TnCUQzen-QuIt_rS3gQGtX2rxkJBNKMo5q86C2O0Q4om5BuX_2rj91AZGictnTvSaGS7Yb0fayE" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=huyedits/Symbio&type=date&legend=top-left&sealed_token=BARd7crHixQjaz11nJTAZ7mVM0hzMRPkR0XsWnt0JCDpfGb7UODGYP_v1vWqVZ7oBnYNeBSjSPD41Jz3zptiRq5d4it22dMAG2hzDZp-hqN1WUU71TnCUQzen-QuIt_rS3gQGtX2rxkJBNKMo5q86C2O0Q4om5BuX_2rj91AZGictnTvSaGS7Yb0fayE" />
 </picture>
</a>
