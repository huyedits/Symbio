# Symbio
Your machine, your way.

[![Live Demo on Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20Demo-Hugging%20Face%20Spaces-blue)](https://huggingface.co/spaces/HuyEdits/symbio-demo)

Symbio develops as you tell it what to do in repeat.

**Try the interactive demo** — the agent's real tag parser, self-correction miner, research memory, and RAG retriever running in your browser: https://huggingface.co/spaces/HuyEdits/symbio-demo

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
│   │   ├── memory.py      # Notes, memory, profile management
│   │   ├── sandbox.py     # Sandboxed commands and Python execution
│   │   ├── computer.py    # Browser automation helpers
│   │   ├── cron.py        # Scheduled jobs and reminders
│   │   ├── telegram.py    # Telegram bot gateway
│   │   └── tooling.py     # Tag parsing and tool stripping
│   └── utils.py         # Shared helpers
├── rag.py               # Lightweight keyword-based RAG
├── README.md
├── config.json          # User configuration
├── models.json          # Model presets
├── notes/               # Markdown notes / memory
├── training_data/       # train.jsonl and valid.jsonl
├── adapters/            # LoRA adapter weights
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
